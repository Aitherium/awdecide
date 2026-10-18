#!/usr/bin/env python3
"""train_decider -- fit the nerve's heads, score them against what serves today, gate.

Input: rows from ``build_decider_dataset`` (their ``split`` is time-ordered per task).
Per task, candidates are {logistic, mlp} x {hash, embed+hash (when --embed-url)};
the one with the best VALIDATION Brier is kept. Then, on the TEST tail only:

  new      the kept candidate
  current  what answers that decision today without an LLM: the door's neural rung
           (``--current``, the live ``neural-rung.npz``) where it has a head and
           answers, else the prior rung (training class frequencies)

The current neural rung was trained on 80 % of the judge forks, so a time tail of
the same corpus contains rows it has SEEN. For door tasks the comparison therefore
also runs on ``clean`` -- test rows whose fork the rung held out -- and the gate uses
``clean`` when it has at least MIN_TEST_ROWS rows.

Promotion (``heads.promotion``): a task's head replaces current only with paired
evidence on >= 50 held-out rows -- the bootstrap 95 % CI of its per-row Brier gain
excludes 0 AND accuracy is not significantly worse (exact McNemar) nor >1 pt lower. Promoted heads go to
``<out>/decider.npz``; every head goes to ``<out>/decider.candidate.npz``; the model card
is ``<out>/decider.card.json``. Nothing is written inside a git work tree.

    python -m awdecide.decider.train_decider --rows rows.jsonl --out-dir <dir> \\
        [--embed-url https://host:port/v1/embeddings --embed-model <m> --ca-bundle <pem>] \\
        [--current <ckpt>/neural-rung.npz]
    python -m awdecide.decider.train_decider --self-test

Exit 0 every evidenced task promoted; 1 a task's candidate would REGRESS what serves
(it is not written to the serving file); 2 could not run (no numpy, no rows, embedder
asked for and down, output committable).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

MIN_LABEL_ROWS = 5         # a label seen fewer times in train is dropped from the task
MIN_TRAIN_ROWS = 30
CURRENT_SEED = 7           # the door's fit_neural_rung split seed
CURRENT_HOLDOUT = 0.2


def _held_by_current(fork: str) -> bool:
    """Did the door's neural rung hold this fork OUT of its training? Same formula as
    decide.fit_neural_rung (seed 7, 20 %). Its forced bench forks are not replayed,
    so a few clean rows may be counted as seen -- the conservative direction."""
    return (zlib.crc32((str(CURRENT_SEED) + "|" + fork).encode("utf-8")) % 1000) < int(
        round(CURRENT_HOLDOUT * 1000))


def _committable(path: Path) -> bool:
    from awdecide.decider.build_decider_dataset import committable

    return committable(path)


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            if line.strip():
                r = json.loads(line)
                for k in ("context", "kind", "task", "label", "split"):
                    if k not in r:
                        raise ValueError(f"{path.name}:{ln}: row has no {k!r}")
                rows.append(r)
    return rows


def prepare(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """Drop failed-route rows (imitating a failure teaches the failure) and labels too
    rare to learn; return (rows, label order, notes)."""
    notes: Dict[str, Any] = {}
    if rows and rows[0]["kind"] == "route":
        kept = [r for r in rows if r.get("outcome") != "failure"]
        notes["dropped_failed_outcome"] = len(rows) - len(kept)
        rows = kept
    c = Counter(r["label"] for r in rows if r["split"] == "train")
    labels = sorted(lab for lab, n in c.items() if n >= MIN_LABEL_ROWS)
    rare = {lab: n for lab, n in Counter(r["label"] for r in rows).items() if lab not in labels}
    if rare:
        notes["dropped_rare_labels"] = rare
    return [r for r in rows if r["label"] in labels], labels, notes


class Current:
    """What serves the decision today: the door's neural rung where it answers, else
    the prior. Loaded through awdecide.door so the SAME code that serves scores it."""

    def __init__(self, npz: Optional[Path], svc_dir: Optional[str]) -> None:
        self.rung = self.feats = None
        self.error = ""
        if npz is None:
            return
        tmp = Path(tempfile.mkdtemp(prefix="decider-cur-"))
        try:
            from awdecide import door as door_mod

            _cd, decide = door_mod.load_service(svc_dir, str(tmp))
            self.rung = decide.NeuralRung.load(Path(npz))
            self.feats = decide.HashedFeatures()
            if self.rung is None:
                self.error = f"{npz}: no usable heads"
        except Exception as exc:  # noqa: BLE001 -- reported; the prior stands in
            self.error = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def proba(self, rows: List[Dict[str, Any]], labels: List[str], prior: Any) -> Tuple[Any, int]:
        """[n, k] in ``labels`` order, and how many rows the neural rung answered."""
        prob = prior.copy()
        answered = 0
        if self.rung is None:
            return prob, 0
        for i, r in enumerate(rows):
            d = r.get("door")
            if not d:
                continue
            try:
                got = self.rung.predict(r["kind"], d["domain"], d["state"], d["question"],
                                        list(labels), self.feats)
            except Exception:  # noqa: BLE001 -- a head that cannot score = prior
                got = None
            if got is None:
                continue
            probs = got["probabilities"]
            v = np.array([float(probs.get(lab, 0.0)) for lab in labels], dtype=np.float64)
            if v.sum() > 0:
                prob[i] = v / v.sum()
                answered += 1
        return prob, answered


def train_task(task: str, rows: List[Dict[str, Any]], embedder: Any, current: Current,
               recipes: List[str], seed: int = 7) -> Dict[str, Any]:
    from awdecide.decider import heads as hd
    from awdecide.decider.features import featurize

    rows, labels, notes = prepare(rows)
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    out: Dict[str, Any] = {"task": task, "kind": rows[0]["kind"] if rows else None,
                           "labels": labels, "rows": {s: len(v) for s, v in split.items()},
                           "notes": notes}
    if len(split["train"]) < MIN_TRAIN_ROWS or len(labels) < 2 or not split["test"]:
        out["skipped"] = (f"{len(split['train'])} train rows, {len(labels)} labels, "
                          f"{len(split['test'])} test rows -- nothing to fit or score")
        return out
    idx = {lab: i for i, lab in enumerate(labels)}
    ys = {s: np.array([idx[r["label"]] for r in v], dtype=np.int64) for s, v in split.items()}
    best: Optional[Tuple[float, Any, str]] = None
    cands: Dict[str, Any] = {}
    for recipe in recipes:
        xs = {s: featurize(recipe, [r["context"] for r in v], embedder) if v else None
              for s, v in split.items()}
        for arch in ("logistic", "mlp"):
            h = hd.fit_head(task, out["kind"], labels, recipe, arch, xs["train"], ys["train"],
                           xs["val"], ys["val"], seed=seed)
            val = hd.scores(h.proba(xs["val"]), ys["val"]) if split["val"] else {"brier": None}
            cands[f"{arch}/{recipe}"] = val
            key = val["brier"] if val["brier"] is not None else float("inf")
            if best is None or key < best[0]:
                best = (key, h, recipe)
            h.meta["_test_P"] = h.proba(xs["test"])
    assert best is not None
    head = best[1]
    p_new = head.meta.pop("_test_P")
    prior = hd.prior_proba(ys["train"], len(labels), len(split["test"]))
    p_cur, answered = current.proba(split["test"], labels, prior)
    yt = ys["test"]
    out["candidates_val"] = cands
    out["chosen"] = f"{head.arch}/{head.recipe}"
    out["new"] = hd.scores(p_new, yt)
    out["current"] = hd.scores(p_cur, yt)
    out["current_neural_answered"] = answered
    out["prior"] = hd.scores(prior, yt)
    gate_new, gate_cur, gate_y, on = p_new, p_cur, yt, "test"
    if any(r.get("door") for r in split["test"]) and current.rung is not None:
        m = np.array([_held_by_current(r["door"]["fork"]) for r in split["test"]], dtype=bool)
        out["clean_new"] = hd.scores(p_new[m], yt[m])
        out["clean_current"] = hd.scores(p_cur[m], yt[m])
        if int(m.sum()) >= hd.MIN_TEST_ROWS:
            gate_new, gate_cur, gate_y, on = p_new[m], p_cur[m], yt[m], "clean"
    verdict = hd.promotion(gate_new, gate_cur, gate_y)
    verdict["on"] = on
    verdict["thin"] = len(gate_y) < hd.MIN_TEST_ROWS
    out["promotion"] = verdict
    # the per-row predictions are not kept; the head is
    for k in [k for k in head.meta if k.startswith("_")]:
        head.meta.pop(k)
    head.meta.update({"test": out["new"], "rows_train": len(split["train"]),
                      "promoted": verdict["promote"]})
    out["_head"] = head
    return out


def _fmt(s: Optional[Dict[str, Any]]) -> str:
    if not s or not s.get("n"):
        return f"{'-':>6} {'-':>6} {'-':>6}"
    return f"{s['accuracy']:>6.3f} {s['brier']:>6.3f} {s['ece']:>6.3f}"


def table(results: List[Dict[str, Any]]) -> str:
    lines = [f"{'task':<22}{'kind':<8}{'n':>5}  {'new acc/brier/ece':<21}  "
             f"{'current acc/brier/ece':<21} {'nrl':>4}  verdict"]
    for r in results:
        if "skipped" in r:
            lines.append(f"{r['task']:<22}{str(r['kind']):<8}{'-':>5}  skipped: {r['skipped']}")
            continue
        p = r["promotion"]
        word = "PROMOTE" if p["promote"] else ("REFUSE(thin)" if p["thin"] else "REFUSE")
        lines.append(f"{r['task']:<22}{r['kind']:<8}{r['new']['n']:>5}  {_fmt(r['new'])}  "
                     f"{_fmt(r['current'])} {r['current_neural_answered']:>4}  "
                     f"{word} [{r['chosen']}] {p['why']}")
        if "clean_new" in r:
            lines.append(f"{'  (clean: rung held out)':<30}{r['clean_new']['n']:>5}  "
                         f"{_fmt(r['clean_new'])}  {_fmt(r['clean_current'])}")
    by_kind: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        if "skipped" not in r:
            by_kind.setdefault(r["kind"], []).append(r)
    lines.append("per kind (row-weighted over its tasks):")
    for kind, rs in sorted(by_kind.items()):
        n = sum(r["new"]["n"] for r in rs)

        def w(side: str, key: str, rs: List[Dict[str, Any]] = rs, n: int = n) -> float:
            return sum(r[side][key] * r[side]["n"] for r in rs) / n

        lines.append(f"  {kind:<8}{n:>6}  new acc {w('new', 'accuracy'):.3f} brier "
                     f"{w('new', 'brier'):.3f} | current acc {w('current', 'accuracy'):.3f} "
                     f"brier {w('current', 'brier'):.3f}")
    return "\n".join(lines)


def run(rows: List[Dict[str, Any]], out_dir: Path, embedder: Any, current: Current,
        meta: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]]]:
    from awdecide.decider import heads as hd

    recipes = ["hash"] + (["embed+hash"] if embedder is not None else [])
    tasks = sorted({r["task"] for r in rows})
    results = [train_task(t, [r for r in rows if r["task"] == t], embedder, current, recipes)
               for t in tasks]
    heads = [r.pop("_head") for r in results if "_head" in r]
    promoted = [h for h in heads if h.meta.get("promoted")]
    regress = [r["task"] for r in results if "promotion" in r
               and not r["promotion"]["promote"] and not r["promotion"]["thin"]]
    card = dict(meta)
    card.update({"trained_at": time.strftime("%Y-%m-%dT%hd:%M:%S%z"), "recipes": recipes,
                 "promotion_rule": {"brier": "paired bootstrap 95% CI of the gain, lower bound > 0",
                                    "accuracy": f"not significantly worse (exact McNemar, alpha {hd.ALPHA})",
                                    "max_accuracy_drop": hd.MAX_ACC_DROP,
                                    "min_test_rows": hd.MIN_TEST_ROWS,
                                    "bootstrap": [hd.BOOTSTRAP_N, hd.BOOTSTRAP_SEED]},
                 "tasks": results, "promoted": [h.task for h in promoted],
                 "regressed": regress, "current_error": current.error or None})
    out_dir.mkdir(parents=True, exist_ok=True)
    if heads:
        hd.save_heads(out_dir / "decider.candidate.npz", heads, {k: v for k, v in card.items()
                                                               if k != "tasks"})
    serving = out_dir / "decider.npz"
    if promoted:
        hd.save_heads(serving, promoted, {k: v for k, v in card.items() if k != "tasks"})
    (out_dir / "decider.card.json").write_text(json.dumps(card, indent=2, default=str),
                                               encoding="utf-8")
    print(table(results))
    print(f"promoted: {[h.task for h in promoted] or 'none'}; regress (refused): {regress or 'none'}")
    if current.error:
        print(f"current rung: {current.error}")
    return (1 if regress else 0), results


# ---------------------------------------------------------------- self-test --


def _synthetic(n: int, learnable: Optional[bool], task: str, seed: int) -> List[Dict[str, Any]]:
    """``learnable`` True: the signal word means the same thing throughout. None: the
    labels are pure noise. False: it FLIPS for the newest 20 % (the test tail) -- a head that learned train is then
    confidently wrong, deterministically worse than the prior, which is the case the
    gate exists for (a noise task can beat the prior by chance on 80 rows)."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        y = rng.random() < 0.5
        flip = learnable is False and i >= int(n * 0.8)
        word = "alpha" if (y != flip) else "omega"
        if learnable is None:          # pure noise: the word says nothing about y
            word = rng.choice(["alpha", "omega"])
        ctx = f"kind=yesno || case {i} signal {word} noise {rng.randint(0, 9)}"
        rows.append({"context": ctx, "kind": "yesno", "task": task, "label": "yes" if y else "no",
                     "label_space": ["no", "yes"], "outcome": None, "source": "synthetic",
                     "ts": float(i)})
    from awdecide.decider.build_decider_dataset import time_split

    time_split(rows)
    return rows


def self_test() -> int:
    """Prove the gate can fail: a head worse than the prior must be REFUSED (exit 1, not
    written to the serving file) while a learnable task beside it is promoted."""
    from awdecide.decider import heads as hd

    fails = []
    y = np.array([i % 2 for i in range(200)])
    sure = np.eye(2)[y] * 0.9 + 0.05            # right on every row, p=0.95
    flat = np.full((200, 2), 0.5)
    if hd.promotion(flat, sure, y)["promote"]:
        fails.append("a head worse than current was promoted")
    if not hd.promotion(sure, flat, y)["promote"]:
        fails.append("a clearly better head was refused")
    if hd.promotion(sure[:40], flat[:40], y[:40])["promote"]:
        fails.append("40 rows were treated as evidence")
    wrong = sure.copy()
    wrong[:30] = wrong[:30, ::-1]                # loses 30 rows outright, sharper elsewhere
    if hd.promotion(wrong, np.eye(2)[y] * 0.6 + 0.2, y)["promote"]:
        fails.append("significantly lower accuracy was promoted")
    tmp = Path(tempfile.mkdtemp(prefix="decider-selftest-"))
    try:
        rows = (_synthetic(400, True, "yesno.signal", 1) + _synthetic(400, False, "yesno.shifted", 4)
                + _synthetic(400, None, "yesno.noise", 2))
        code, results = run(rows, tmp, None, Current(None, None), {"self_test": True})
        by = {r["task"]: r for r in results}
        if code != 1:
            fails.append(f"exit {code} with a regressing head in the batch (want 1)")
        if not by["yesno.signal"]["promotion"]["promote"]:
            fails.append(f"learnable task refused: {by['yesno.signal']['promotion']}")
        if by["yesno.shifted"]["promotion"]["promote"]:
            fails.append("a head worse than the prior was promoted")
        # the 2026-10-16 hole: pure-noise labels, 80 test rows, a point estimate
        # that beat the prior -- the paired CI must refuse it
        if by["yesno.noise"]["promotion"]["promote"]:
            fails.append(f"noise head promoted: {by['yesno.noise']['promotion']['why']}")
        served = hd.load_heads(tmp / "decider.npz")
        if set(served) != {"yesno.signal"}:
            fails.append(f"serving file holds {sorted(served)} (want only yesno.signal)")
        if set(hd.load_heads(tmp / "decider.candidate.npz")) != {"yesno.signal", "yesno.shifted", "yesno.noise"}:
            fails.append("candidate file does not keep the refused head for inspection")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for f in fails:
        print("SELF-TEST FAIL:", f)
    print("self-test:", "ok" if not fails else f"{len(fails)} failure(s)")
    return 0 if not fails else 1


def _git_sha() -> Optional[str]:
    try:
        r = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows")
    ap.add_argument("--out-dir", default=os.environ.get("AITHER_DECIDER_DIR"))
    ap.add_argument("--embed-url", default=os.environ.get("AITHER_DECIDER_EMBED_URL"))
    ap.add_argument("--embed-model", default=os.environ.get("AITHER_DECIDER_EMBED_MODEL"))
    ap.add_argument("--ca-bundle", default=os.environ.get("AITHER_CA_BUNDLE"))
    ap.add_argument("--embed-cache", default=None, help="sqlite cache (default <out-dir>/embed-cache.sqlite)")
    ap.add_argument("--current", default=None, help="the serving neural-rung.npz to compare against")
    ap.add_argument("--svc-dir", default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if np is None:
        print("numpy is required (pip install awdecide[door])")
        return 2
    if a.self_test:
        return self_test()
    if not a.rows or not a.out_dir:
        print("--rows and --out-dir are required")
        return 2
    out_dir = Path(a.out_dir)
    if _committable(out_dir / "decider.npz"):
        print(f"refusing: {out_dir} is committable in a git work tree -- weights never go in a repo")
        return 2
    try:
        rows = load_rows(Path(a.rows))
    except (OSError, ValueError) as exc:
        print(f"cannot read rows: {exc}")
        return 2
    if not rows:
        print("no rows")
        return 2
    embedder = None
    if a.embed_url:
        if not a.embed_model:
            print("--embed-url needs --embed-model")
            return 2
        from awdecide.decider.features import Embedder

        embedder = Embedder(a.embed_url, a.embed_model,
                            Path(a.embed_cache) if a.embed_cache else out_dir / "embed-cache.sqlite",
                            ca_bundle=a.ca_bundle)
        try:
            embedder.embed(["probe"])
        except Exception as exc:  # noqa: BLE001 -- asked for and down = cannot judge
            print(f"embedder {a.embed_url} down: {type(exc).__name__}: {exc}")
            return 2
    current = Current(Path(a.current) if a.current else None, a.svc_dir)
    meta = {"rows_file": str(a.rows), "rows": len(rows), "code_sha": _git_sha(),
            "embed_model": a.embed_model if embedder else None,
            "current": str(a.current) if a.current else None}
    code, _ = run(rows, out_dir, embedder, current, meta)
    if embedder is not None:
        print(f"embedder: {embedder.calls} calls, {embedder.hits} cache hits")
    return code


if __name__ == "__main__":
    sys.exit(main())
