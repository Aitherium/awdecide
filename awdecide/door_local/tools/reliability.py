#!/usr/bin/env python3
"""reliability -- is the door's `probability` a probability, on REAL forks?

The coin bench (tools/calibration_bench.py) proves calibration on a coin whose
p we chose. This tool proves it -- or refutes it -- on the forks the door has
actually answered: every recorded probability is paired with the outcome that
followed it, and the pairs are binned into a reliability table.

    python tools/reliability.py                       # AITHER_WM_CKPT_DIR (or D:/awdecide/wm-ckpt)
    python tools/reliability.py --ckpt-dir /path --json out.json --png out.png
    python tools/reliability.py --generate            # bench-generated pairs into a TEMP ckpt dir
    python tools/reliability.py --self-test

Two honest sources of (probability, outcome) pairs, reported SEPARATELY:

  journaled   decisions.jsonl -- what the door ANSWERED and with what number
              (decide.py `_record`), joined on decision_id to the outcome row
              that followed (`outcome`). This is the door's own claim graded
              by what happened next. Rows written before 2026-09-20 carry
              only `confidence` (probability was not journaled); those are
              tabulated as a third, clearly-labelled set and never drive the
              verdict, because confidence is evidence strength, not P(right).
  replayed    every domain-decide.*.transitions.jsonl -- the per-fork outcome
              journal -- replayed IN ORDER through a fresh copy of the real
              engine: before each outcome is observed, what would the engine
              have said P(reward>0 | state, action) was? (prequential
              calibration: the number the door would have stated at that
              moment, from the outcomes before it). No prediction = no pair.

Table per set: 10 bins of predicted probability, count, mean predicted,
observed frequency of reward>0, |gap|; ECE (count-weighted mean |gap|), MCE
(max |gap|), Brier score; broken out by source (engine/neighbor/llm/prior),
by kind (yesno/choice/score) and by domain.

Exit 0 when every set with enough pairs has ECE <= --max-ece (default 0.10),
1 when one does not (the number is NOT a probability there), 2 when fewer
than --min-pairs (default 30) real pairs exist in any set -- and it SAYS how
many exist, because that count is itself the finding.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CKPT = os.environ.get("AITHER_WM_CKPT_DIR") or "D:/awdecide/wm-ckpt"
SOURCES = ("engine", "neighbor", "llm", "prior")


# ----------------------------------------------------------------- the metrics
def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def reliability(pairs: List[Dict[str, Any]], bins: int = 10) -> Dict[str, Any]:
    """Reliability table + ECE / MCE / Brier for a list of {p, y} rows."""
    n = len(pairs)
    table = []
    ece = 0.0
    mce = 0.0
    brier = 0.0
    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(bins)]
    for r in pairs:
        p = clip01(float(r["p"]))
        y = 1.0 if r["y"] else 0.0
        brier += (p - y) ** 2
        idx = min(bins - 1, int(p * bins))
        buckets[idx].append(r)
    for i, b in enumerate(buckets):
        lo, hi = i / bins, (i + 1) / bins
        if not b:
            table.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0, "mean_p": None, "obs": None, "gap": None})
            continue
        mean_p = sum(clip01(float(r["p"])) for r in b) / len(b)
        obs = sum(1 for r in b if r["y"]) / len(b)
        gap = abs(mean_p - obs)
        ece += gap * len(b) / n
        mce = max(mce, gap)
        table.append(
            {
                "bin": f"[{lo:.1f},{hi:.1f})",
                "n": len(b),
                "mean_p": round(mean_p, 4),
                "obs": round(obs, 4),
                "gap": round(gap, 4),
            }
        )
    return {
        "n": n,
        "ece": round(ece, 4) if n else None,
        "mce": round(mce, 4) if n else None,
        "brier": round(brier / n, 4) if n else None,
        "base_rate": round(sum(1 for r in pairs if r["y"]) / n, 4) if n else None,
        "mean_p": round(sum(clip01(float(r["p"])) for r in pairs) / n, 4) if n else None,
        "bins": table,
    }


def breakout(pairs: List[Dict[str, Any]], key: str, bins: int) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in pairs:
        groups[str(r.get(key) or "unknown")].append(r)
    out = {}
    for k in sorted(groups, key=lambda g: -len(groups[g])):
        rep = reliability(groups[k], bins)
        out[k] = {"n": rep["n"], "ece": rep["ece"], "mce": rep["mce"], "brier": rep["brier"],
                  "base_rate": rep["base_rate"], "mean_p": rep["mean_p"]}
    return out


# -------------------------------------------------------- source 1: journaled
def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # torn tail line from a crash mid-append
    except OSError:
        return rows
    return rows


def load_journaled(ckpt: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """(proper_pairs, legacy_confidence_pairs, info) from decisions.jsonl.

    A pair needs a `decision` row and an `outcome` row with the same
    decision_id. Outcomes posted with domain+state+answer (teach rows) have
    decision_id null and pair with nothing -- that is by design: the door made
    no claim about them."""
    path = ckpt / "decisions.jsonl"
    info: Dict[str, Any] = {
        "path": str(path),
        "present": path.is_file(),
        "decision_rows": 0,
        "outcome_rows": 0,
        "outcomes_with_decision_id": 0,
        "outcomes_matched": 0,
        "decisions_without_outcome": 0,
        "pairs_probability": 0,
        "pairs_confidence_legacy": 0,
        "skipped_no_number": 0,
    }
    if not path.is_file():
        return [], [], info
    decisions: Dict[str, Dict[str, Any]] = {}
    outcomes: Dict[str, Dict[str, Any]] = {}
    for row in _read_jsonl(path):
        kind = row.get("kind")
        did = row.get("decision_id")
        if kind == "decision" and did:
            info["decision_rows"] += 1
            decisions[str(did)] = row
        elif kind == "outcome":
            info["outcome_rows"] += 1
            if did:
                info["outcomes_with_decision_id"] += 1
                outcomes.setdefault(str(did), row)  # the FIRST outcome grades the claim
    proper: List[Dict[str, Any]] = []
    legacy: List[Dict[str, Any]] = []
    for did, dec in decisions.items():
        out = outcomes.get(did)
        if out is None:
            info["decisions_without_outcome"] += 1
            continue
        info["outcomes_matched"] += 1
        try:
            y = float(out.get("reward")) > 0.0
        except (TypeError, ValueError):
            continue
        base = {
            "y": y,
            "source": dec.get("source") or "unknown",
            "kind": dec.get("question_kind") or "unknown",
            "domain": dec.get("domain") or "unknown",
            "decision_id": did,
        }
        if "probability" in dec:
            p = dec.get("probability")
            if p is None:  # source=none: the door said it did not know
                info["skipped_no_number"] += 1
                continue
            proper.append({**base, "p": float(p), "field": "probability"})
            info["pairs_probability"] += 1
        else:
            conf = dec.get("confidence")
            if conf is None or (dec.get("source") == "none"):
                info["skipped_no_number"] += 1
                continue
            legacy.append({**base, "p": float(conf), "field": "confidence", "kind": "unknown(legacy)"})
            info["pairs_confidence_legacy"] += 1
    return proper, legacy, info


# -------------------------------------------------------- source 2: replayed
def replay_transitions(
    ckpt: Path, scratch: Path, max_rows: int = 5000
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Prequential replay of every decide.* outcome journal through the real
    engine, rooted in `scratch` so nothing is written back to `ckpt`."""
    info: Dict[str, Any] = {"journals": 0, "rows": 0, "predicted": 0, "unpredicted": 0,
                            "per_domain": {}, "engine": None}
    try:
        journals = sorted(ckpt.glob("domain-decide.*.transitions.jsonl"))
    except OSError:
        journals = []
    info["journals"] = len(journals)
    if not journals:
        return [], info
    os.environ["AITHER_WM_CKPT_DIR"] = str(scratch)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import importlib

    import code_domains  # type: ignore

    code_domains = importlib.reload(code_domains)  # pick up the scratch _CKPT_DIR
    if not code_domains._MLP_OK:
        info["engine"] = "world_model package not importable -- replay skipped"
        return [], info
    info["engine"] = "code_domains.DomainEngines (fresh, scratch ckpt dir)"
    pairs: List[Dict[str, Any]] = []
    for path in journals:
        domain = path.name[len("domain-"): -len(".transitions.jsonl")]
        rows = _read_jsonl(path)
        if len(rows) > max_rows:
            rows = rows[-max_rows:]
        engines = code_domains.DomainEngines()
        got = 0
        for r in rows:
            obs, action, next_obs = r.get("obs"), r.get("action"), r.get("next_obs")
            if not (isinstance(obs, str) and isinstance(next_obs, str)) or action is None:
                continue
            try:
                reward = float(r.get("reward", 0.0))
            except (TypeError, ValueError):
                continue
            info["rows"] += 1
            try:
                eng = engines._engine(domain)
                pred = eng.predict(code_domains._desc_hash(obs), action)
            except Exception:  # noqa: BLE001 -- an engine that cannot predict = no pair
                pred = None
            if pred is not None:
                try:
                    value = float(pred[1])
                except (TypeError, ValueError, IndexError):
                    value = None
                if value is not None and math.isfinite(value):
                    pairs.append(
                        {
                            "p": clip01((value + 1.0) / 2.0),
                            "y": reward > 0.0,
                            "source": "engine",
                            "kind": "yesno" if str(action) in ("yes", "no") else "choice",
                            "domain": domain,
                        }
                    )
                    got += 1
                else:
                    info["unpredicted"] += 1
            else:
                info["unpredicted"] += 1
            try:
                engines.observe(domain, obs, str(action), next_obs, reward, bool(r.get("done", False)))
            except Exception as exc:  # noqa: BLE001
                info["per_domain"][domain] = {"error": f"{type(exc).__name__}: {exc}"}
                break
        info["predicted"] += got
        info["per_domain"][domain] = {"rows": len(rows), "pairs": got}
    return pairs, info


# ------------------------------------------------- bench-generated pairs (opt)
def generate(scratch: Path, decisions: int = 400, p_right: float = 0.70) -> Dict[str, Any]:
    """Run decide_bench + judge_bench (--skip-llm shape) IN-PROCESS against a
    temp ckpt dir with a Decider that journals, so decisions.jsonl there holds
    fresh (probability, outcome) pairs from the same code the service runs."""
    os.environ["AITHER_WM_CKPT_DIR"] = str(scratch)
    for p in (str(ROOT), str(HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import importlib

    import code_domains  # type: ignore

    code_domains = importlib.reload(code_domains)
    import decide  # type: ignore

    decide = importlib.reload(decide)
    import judge as judge_mod  # type: ignore
    import decide_bench  # type: ignore
    import judge_bench  # type: ignore

    if not code_domains._MLP_OK:
        raise RuntimeError("world_model package not importable")
    t0 = time.perf_counter()
    shapes = decide_bench.request_shapes()
    rng = random.Random(3)
    seq = [rng.choice(shapes) for _ in range(decisions)]
    brain = decide_bench.SimBrain(p_right, 0.0)  # latency is irrelevant to calibration
    brain.truth = dict(shapes)
    d = decide.Decider(code_domains.DomainEngines(), llm=brain, record_dir=scratch, embed_enabled=False)
    learn = decide_bench.run_learning(d, brain, seq)

    dj = decide.Decider(
        code_domains.DomainEngines(), llm_enabled=False, record_dir=scratch, embed_enabled=False
    )
    j = judge_mod.Judge(dj)
    dom = "decide.bench.judge"
    judge_bench.run_door(_JudgeAdapter(j), dom)  # cold: source=none, no claim, no pair
    judge_bench.teach_all(_JudgeAdapter(j), dom)
    corrected = 0
    for _pass in range(2):  # warm passes: the door claims a probability, the label grades it
        for _name, output, crits in judge_bench.CASES:
            res = j.judge(output, [c for c, _ in crits], domain=dom)
            for (_c, truth), v in zip(crits, res.get("verdicts") or []):
                if v.get("decision_id") and v.get("pass") is not None:
                    j.correct(v["decision_id"], v["pass"] is truth)
                    corrected += 1
    return {
        "ckpt_dir": str(scratch),
        "decide_bench": {"decisions": decisions, "accuracy": round(learn["accuracy"], 3),
                         "llm_calls": learn["llm_calls"]},
        "judge_bench": {"cases": len(judge_bench.CASES), "corrected": corrected},
        "wall_s": round(time.perf_counter() - t0, 1),
    }


class _JudgeAdapter:
    def __init__(self, j: Any) -> None:
        self.j = j

    def judge(self, output: str, criteria: List[str], domain: str) -> dict:
        return self.j.judge(output, criteria, domain=domain)

    def teach(self, output: str, criterion: str, should_pass: bool, domain: str) -> dict:
        return self.j.teach(output, criterion, should_pass, domain=domain)


# ------------------------------------------------------------------ rendering
def _table(rep: Dict[str, Any]) -> List[str]:
    lines = [f"  {'bin':12}{'n':>7}{'mean p':>9}{'observed':>10}{'|gap|':>8}"]
    for b in rep["bins"]:
        if b["n"] == 0:
            lines.append(f"  {b['bin']:12}{0:>7}{'-':>9}{'-':>10}{'-':>8}")
        else:
            lines.append(
                f"  {b['bin']:12}{b['n']:>7}{b['mean_p']:>9.3f}{b['obs']:>10.3f}{b['gap']:>8.3f}"
            )
    lines.append(
        f"  n={rep['n']}  ECE={rep['ece']}  MCE={rep['mce']}  Brier={rep['brier']}"
        f"  base rate={rep['base_rate']}  mean p={rep['mean_p']}"
    )
    return lines


def _breakout_lines(title: str, bo: Dict[str, Dict[str, Any]]) -> List[str]:
    lines = [f"  by {title}:  {'group':34}{'n':>6}{'ECE':>8}{'MCE':>8}{'Brier':>8}{'base':>7}"]
    for k, v in bo.items():
        lines.append(
            f"  {'':12}{k[:34]:34}{v['n']:>6}{v['ece']:>8.3f}{v['mce']:>8.3f}{v['brier']:>8.3f}{v['base_rate']:>7.2f}"
        )
    return lines


def render(report: Dict[str, Any]) -> str:
    out: List[str] = []
    out.append(f"reliability -- ckpt dir {report['ckpt_dir']}  ({report['label']})")
    j = report["journaled_info"]
    out.append(
        f"  decisions.jsonl: {'present' if j['present'] else 'ABSENT'}; {j['decision_rows']} decision rows, "
        f"{j['outcome_rows']} outcome rows ({j['outcomes_with_decision_id']} carry a decision_id); "
        f"{j['outcomes_matched']} matched -> {j['pairs_probability']} probability pairs, "
        f"{j['pairs_confidence_legacy']} legacy confidence-only pairs, {j['skipped_no_number']} with no number"
    )
    r = report["replayed_info"]
    out.append(
        f"  transitions journals: {r['journals']} decide.* forks, {r['rows']} outcome rows; "
        f"engine predicted before {r['predicted']} of them ({r['unpredicted']} cold, no claim)"
    )
    for key, title in (
        ("journaled", "JOURNALED: the door's stated `probability` vs the outcome posted for that decision_id"),
        ("replayed", "REPLAYED (prequential): engine's P(reward>0|state,action) before each journaled outcome"),
        ("legacy_confidence", "LEGACY: `confidence` (probability not journaled before 2026-09-20) vs outcome -- NOT a probability claim"),
    ):
        sec = report["sets"].get(key)
        out.append("")
        out.append(title)
        if not sec or sec["overall"]["n"] == 0:
            out.append("  (no pairs)")
            continue
        out.extend(_table(sec["overall"]))
        for bk, bt in (("by_source", "source"), ("by_kind", "kind"), ("by_domain", "domain")):
            if sec.get(bk):
                out.extend(_breakout_lines(bt, sec[bk]))
    out.append("")
    out.append(f"VERDICT: {report['verdict']}")
    return "\n".join(out)


def _png(report: Dict[str, Any], path: str) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 -- optional, never required
        return f"png skipped: matplotlib not importable ({type(exc).__name__})"
    sets = [(k, v) for k, v in report["sets"].items() if v and v["overall"]["n"]]
    if not sets:
        return "png skipped: no pairs"
    fig, axes = plt.subplots(1, len(sets), figsize=(5 * len(sets), 4.5), squeeze=False)
    for ax, (name, sec) in zip(axes[0], sets):
        xs = [b["mean_p"] for b in sec["overall"]["bins"] if b["n"]]
        ys = [b["obs"] for b in sec["overall"]["bins"] if b["n"]]
        ns = [b["n"] for b in sec["overall"]["bins"] if b["n"]]
        ax.plot([0, 1], [0, 1], "--", color="#999")
        ax.scatter(xs, ys, s=[20 + 3 * math.sqrt(n) for n in ns])
        ax.plot(xs, ys, "-")
        ax.set_title(f"{name}  n={sec['overall']['n']}  ECE={sec['overall']['ece']}")
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency reward>0")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return f"wrote {path}"


# ------------------------------------------------------------------ assemble
def build_report(
    ckpt: Path,
    scratch: Path,
    *,
    bins: int,
    min_pairs: int,
    max_ece: float,
    replay: bool,
    label: str,
    max_rows: int = 5000,
) -> Tuple[Dict[str, Any], int]:
    proper, legacy, jinfo = load_journaled(ckpt)
    if replay:
        replayed, rinfo = replay_transitions(ckpt, scratch, max_rows=max_rows)
    else:
        replayed, rinfo = [], {"journals": 0, "rows": 0, "predicted": 0, "unpredicted": 0,
                               "per_domain": {}, "engine": "replay disabled"}

    def section(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "overall": reliability(pairs, bins),
            "by_source": breakout(pairs, "source", bins),
            "by_kind": breakout(pairs, "kind", bins),
            "by_domain": breakout(pairs, "domain", bins),
        }

    sets = {
        "journaled": section(proper),
        "replayed": section(replayed),
        "legacy_confidence": section(legacy),
    }
    judged: List[str] = []
    failing: List[str] = []
    for key in ("journaled", "replayed"):
        rep = sets[key]["overall"]
        if rep["n"] >= min_pairs:
            doms = sets[key]["by_domain"]
            top = max(doms.items(), key=lambda kv: kv[1]["n"]) if doms else None
            judged.append(
                f"{key}: n={rep['n']} ECE={rep['ece']} over {len(doms)} domain(s)"
                + (f", {top[1]['n']} of them from {top[0]}" if top else "")
            )
            if rep["ece"] is not None and rep["ece"] > max_ece:
                failing.append(f"{key}: ECE {rep['ece']} > {max_ece} over {rep['n']} pairs")
    if not judged:
        rc = 2
        verdict = (
            f"COULD NOT JUDGE: {len(proper)} journaled probability pairs and {len(replayed)} "
            f"replayed pairs exist; need >= {min_pairs} in a set. "
            f"({len(legacy)} legacy confidence-only pairs are tabulated but are not probability claims.)"
        )
    elif failing:
        rc = 1
        verdict = "NOT CALIBRATED: " + "; ".join(failing)
    else:
        rc = 0
        verdict = f"CALIBRATED within ECE {max_ece}: " + "; ".join(judged)
    report = {
        "generated_at": int(time.time()),
        "label": label,
        "ckpt_dir": str(ckpt),
        "bins": bins,
        "min_pairs": min_pairs,
        "max_ece": max_ece,
        "journaled_info": jinfo,
        "replayed_info": rinfo,
        "pairs": {"journaled": len(proper), "replayed": len(replayed), "legacy_confidence": len(legacy)},
        "sets": sets,
        "verdict": verdict,
        "exit": rc,
    }
    return report, rc


# ------------------------------------------------------------------ self-test
def _synthetic(kind: str, n: int, seed: int = 5) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        if kind == "calibrated":
            p = rng.random()
            y = rng.random() < p
        elif kind == "collapsed":  # always 0.99 on a fair coin
            p = 0.99
            y = rng.random() < 0.5
        else:
            raise ValueError(kind)
        rows.append({"p": p, "y": y, "source": SOURCES[i % 4], "kind": "choice", "domain": "decide.synth"})
    return rows


def self_test() -> int:
    fails: List[str] = []
    good = reliability(_synthetic("calibrated", 5000))
    if not (good["ece"] is not None and good["ece"] < 0.03):
        fails.append(f"perfectly calibrated set read ECE {good['ece']} (expected < 0.03)")
    bad = reliability(_synthetic("collapsed", 5000))
    if not (bad["ece"] is not None and bad["ece"] > 0.40):
        fails.append(f"collapsed set read ECE {bad['ece']} (expected > 0.40)")
    if not (bad["mce"] is not None and bad["mce"] > 0.40 and bad["brier"] > 0.40):
        fails.append(f"collapsed set MCE/Brier {bad['mce']}/{bad['brier']} (expected > 0.40)")
    if reliability([])["ece"] is not None:
        fails.append("empty set produced an ECE")
    # the journal pairing + the exit contract, on a synthetic ckpt dir
    with tempfile.TemporaryDirectory(prefix="reliab-selftest-") as td:
        ck = Path(td) / "ck"
        ck.mkdir()
        rows = []
        rng = random.Random(9)
        for i in range(60):
            p = rng.random()
            did = f"d{i:04d}"
            rows.append({"kind": "decision", "decision_id": did, "domain": "decide.t", "state": "s",
                         "answer": "a", "confidence": 0.5, "probability": p,
                         "probability_source": "outcomes", "question_kind": "choice",
                         "source": "engine", "ts": 0})
            rows.append({"kind": "outcome", "decision_id": did, "domain": "decide.t", "answer": "a",
                         "reward": 1.0 if rng.random() < p else -1.0, "ts": 0})
        rows.append({"kind": "decision", "decision_id": "legacy1", "domain": "decide.t", "state": "s",
                     "answer": "a", "confidence": 0.7, "source": "engine", "ts": 0})
        rows.append({"kind": "outcome", "decision_id": "legacy1", "domain": "decide.t", "answer": "a",
                     "reward": 1.0, "ts": 0})
        rows.append({"kind": "outcome", "decision_id": None, "domain": "decide.t", "answer": "a",
                     "reward": 1.0, "ts": 0})
        (ck / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        proper, legacy, info = load_journaled(ck)
        if len(proper) != 60 or len(legacy) != 1 or info["outcomes_with_decision_id"] != 61:
            fails.append(f"pairing: proper={len(proper)} legacy={len(legacy)} info={info}")
        rep, rc = build_report(ck, Path(td) / "scratch", bins=10, min_pairs=30, max_ece=0.25,
                               replay=False, label="self-test")
        if rc != 0:
            fails.append(f"60 calibrated pairs should exit 0, got {rc}: {rep['verdict']}")
        rep, rc = build_report(ck, Path(td) / "scratch", bins=10, min_pairs=100, max_ece=0.25,
                               replay=False, label="self-test")
        if rc != 2 or "60 journaled" not in rep["verdict"]:
            fails.append(f"too few pairs should exit 2 and say how many: rc={rc} {rep['verdict']}")
        # a collapsed journal must exit 1
        rows = []
        for i in range(60):
            did = f"c{i:04d}"
            rows.append({"kind": "decision", "decision_id": did, "domain": "decide.c", "state": "s",
                         "answer": "a", "confidence": 0.99, "probability": 0.99,
                         "question_kind": "yesno", "source": "engine", "ts": 0})
            rows.append({"kind": "outcome", "decision_id": did, "domain": "decide.c", "answer": "a",
                         "reward": 1.0 if i % 2 else -1.0, "ts": 0})
        (ck / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        rep, rc = build_report(ck, Path(td) / "scratch", bins=10, min_pairs=30, max_ece=0.10,
                               replay=False, label="self-test")
        if rc != 1:
            fails.append(f"collapsed journal should exit 1, got {rc}: {rep['verdict']}")
        render(rep)  # must not raise
    if fails:
        print("SELF-TEST FAILED:\n  " + "\n  ".join(fails))
        return 1
    print(
        f"SELF-TEST PASSED: calibrated ECE {good['ece']} < 0.03; collapsed ECE {bad['ece']} > 0.40; "
        "journal pairing, exit 0/1/2 contract hold"
    )
    return 0


# ----------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT, help="journal dir (AITHER_WM_CKPT_DIR)")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--min-pairs", type=int, default=30)
    ap.add_argument("--max-ece", type=float, default=0.10)
    ap.add_argument("--no-replay", action="store_true", help="skip the prequential replay of transitions journals")
    ap.add_argument("--max-rows", type=int, default=5000, help="replay at most this many rows per journal (the tail)")
    ap.add_argument("--generate", action="store_true",
                    help="run decide_bench + judge_bench in-process into a TEMP ckpt dir and grade THAT")
    ap.add_argument("--decisions", type=int, default=400, help="--generate: decide_bench decisions")
    ap.add_argument("--label", default=None, help="how to label this run (default: live journals / bench-generated)")
    ap.add_argument("--json", help="write the full report here ('-' = print it to stdout instead of the tables)")
    ap.add_argument("--png", help="reliability diagram (needs matplotlib; skipped if absent)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    scratch = Path(tempfile.mkdtemp(prefix="reliability-"))
    gen: Optional[Dict[str, Any]] = None
    if a.generate:
        try:
            gen = generate(scratch / "gen", decisions=a.decisions)
        except Exception as exc:  # noqa: BLE001
            print(f"COULD NOT JUDGE: bench generation failed: {type(exc).__name__}: {exc}")
            return 2
        ckpt = Path(gen["ckpt_dir"])
        label = a.label or "bench-generated (decide_bench + judge_bench in-process, temp ckpt dir)"
    else:
        ckpt = Path(a.ckpt_dir)
        label = a.label or "live journals"
        if not ckpt.is_dir():
            print(f"COULD NOT JUDGE: ckpt dir {ckpt} is not a directory")
            return 2

    report, rc = build_report(
        ckpt, scratch / "replay", bins=a.bins, min_pairs=a.min_pairs, max_ece=a.max_ece,
        replay=not a.no_replay, label=label, max_rows=a.max_rows,
    )
    if gen:
        report["generated"] = gen
    if a.png:
        report["png"] = _png(report, a.png)
    if a.json == "-":
        print(json.dumps(report, indent=2))
        return rc
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not a.quiet:
        print(render(report))
        if gen:
            print(f"  bench generation: {json.dumps(gen)}")
        if a.png:
            print(f"  {report['png']}")
        if a.json:
            print(f"  wrote {a.json}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
