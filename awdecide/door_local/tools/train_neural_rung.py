#!/usr/bin/env python3
"""train_neural_rung -- fit the door's NEURAL rung and say how good it is.

Gap G1 (2026-09-20): a never-seen state used to cost a ~2 s local LLM call that
returned a confidence it made up. This trains one linear (logistic) head per
KIND (yesno / choice / score) over a vector for "<domain> || <state> ||
<question> || <option>" and fits ONE temperature per kind on a fork-held-out
split, so the door can answer a cold state in tens of milliseconds with a
probability that survives a reliability plot. decide.py serves it as source
"neural" between the neighbour rung and the LLM.

Training rows:
  (a) every decide.* journal under AITHER_WM_CKPT_DIR (Decider.dataset reads them;
      the calibration coin, decide.*.coin, is the instrument, never the data)
  (b) labelled judge cases: tools/eval_cases/cases.jsonl when present, else the
      snapshot the last run left in <ckpt>/neural-rung.cases.jsonl, else
      tools/judge_bench.py CASES
Held out: 20 % of FORKS (crc32 of the fork key, deterministic) plus, by default,
every fork the judge bench itself asks about -- so the bench's 'door cold
(neural)' leg is a real cold test, not a replay of the training set.

Representations (both persist in the same .npz; the door prefers the head with
the better held-out accuracy, hashed on a tie, and falls back to hashed whenever
the embedder is away):
  hashed  option-salted crossed features (state fields, field x field, question
          word x field, question/output word bags), 4096 dims. No fleet needed.
  fleet   your embedder (1024-dim) as [e_ctx*e_opt, e_ctx, e_opt] concatenated
          with the hashed block = 7168 dims.

  python tools/train_neural_rung.py --report                      # train + numbers
  python tools/train_neural_rung.py --embed-url http://127.0.0.1:8229 --report
  python tools/train_neural_rung.py --self-test                   # proves it can fail

Exit 0 trained and sane; 1 a head is worse than chance on held-out forks (the
model is NOT written in that case); 2 could not train (no rows, numpy missing,
a corrupt case row).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

DEFAULT_CKPT_HOST = "D:/awdecide/wm-ckpt"


def _ckpt_dir(arg: Optional[str]) -> Path:
    if arg:
        return Path(arg)
    env = os.environ.get("AITHER_WM_CKPT_DIR")
    if env:
        return Path(env)
    try:
        from hostpaths import host_path  # type: ignore

        return Path(host_path(DEFAULT_CKPT_HOST, "ckpt dir"))
    except Exception:  # noqa: BLE001 -- inside the container the env var is set
        return Path("/models/world-model")


def _bench_forks(dm: Any) -> Set[str]:
    """Fork keys the judge bench asks about (forced into held-out)."""
    try:
        import judge as judge_mod  # type: ignore
        import judge_bench  # type: ignore
    except Exception:  # noqa: BLE001 -- no bench here: nothing to force
        return set()
    forks: Set[str] = set()
    j = judge_mod.Judge(None, "decide.judge")
    for _name, output, crits in judge_bench.CASES:
        for c, _truth in crits:
            it = j._items(output, [c], "decide.judge")[0]
            forks.add(dm.neural_fork_key("yesno", it["domain"], it["state"]))
    return forks


def _ood_probe(dm: Any, rung: Any, feats: Any) -> Dict[str, Any]:
    """Score the candidate on `judge_bench`'s cases -- a DIFFERENT distribution.

    None of the bench's forks appear in `cases.jsonl`, which is exactly what
    makes it worth running: every in-corpus gate passed a model this probe
    measured at 69% when answered (11/16, lower bound 0.4440) against an LLM rung
    that gets ~0.96 on the same items.
    """
    try:
        import judge as judge_mod  # type: ignore
        import judge_bench  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "why": f"{type(exc).__name__}: {exc}"}

    j = judge_mod.Judge(None, "decide.judge")
    answered = right = total = 0
    for _name, output, crits in judge_bench.CASES:
        for c, truth in crits:
            it = j._items(output, [c], "decide.judge")[0]
            total += 1
            got = rung.predict("yesno", it["domain"], it["state"], it["question"],
                               ["yes", "no"], feats)
            if got is None or got["probability"] < dm.NEURAL_MIN:
                continue          # declined: the LLM rung answers, which is fine
            answered += 1
            right += int((got["answer"] == "yes") is bool(truth))
    acc = round(right / answered, 4) if answered else None
    return {
        "available": True,
        "source": "judge_bench.CASES (no fork of which is in the training corpus)",
        "items": total,
        "answered": answered,
        "coverage": round(answered / total, 4) if total else None,
        "accuracy": acc,
        "accuracy_lower95": round(dm.wilson_lower(right, answered), 4) if answered else None,
    }


def _bench_rows(dm: Any) -> List[Dict[str, Any]]:
    """judge_bench CASES as training rows -- the fallback when no cases file exists."""
    import judge_bench  # type: ignore

    tmp = Path(tempfile.mkdtemp(prefix="nr-bench-")) / "cases.jsonl"
    with open(tmp, "w", encoding="utf-8") as fh:
        for name, output, crits in judge_bench.CASES:
            fh.write(
                json.dumps(
                    {
                        "id": name,
                        "output": output,
                        "criteria": [{"text": c, "label": bool(t)} for c, t in crits],
                    }
                )
                + "\n"
            )
    return dm.neural_rows_from_cases(tmp)


def _ms_per_prediction(dm: Any, rung: Any, feats: Any, reps: int) -> Optional[float]:
    """Median wall ms for one judge-shaped prediction (state + a 4000-char
    question + 2 options) through `feats`. None when that path has no head."""
    if not rung.has(feats.name, "yesno"):
        return None
    state = "crit:the-test-suite-passed|len:<5000|tb:0|err:1|pass:10+|fail:1|exit:none|hit:2/4"
    question = "Criterion: the test suite passed\n\n" + ("E   AssertionError: nope\n" * 160)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        rung.predict("yesno", "decide.judge.bench", state, question, ["yes", "no"], feats)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return round(statistics.median(ts), 2)


def _answerable_probes(dm: Any, rung: Any, rows: List[Dict[str, Any]], want: int) -> List[Any]:
    """(state, question) pairs the yesno head answers at the serving threshold.
    Latency does not depend on whether the answer is RIGHT, so training forks are
    legitimate here -- what they buy is a timing of the answering path instead of
    the declining one. Empty list = the head answers nothing; the caller then
    times the fixed fork and the histogram says 'none'."""
    if not rung.has("hashed", "yesno"):
        return []
    feats, out, seen = dm.HashedFeatures(), [], set()
    for r in rows:
        if r.get("kind") != "yesno" or len(out) >= want:
            continue
        key = (r["domain"], r["state"], r["question"])
        if key in seen:
            continue
        seen.add(key)
        try:
            p = rung.predict(
                "yesno", r["domain"], r["state"], r["question"], ["yes", "no"], feats
            )
        except Exception:  # noqa: BLE001 -- a probe that will not score is not a probe
            continue
        if p["probability"] >= dm.NEURAL_MIN:
            out.append((r["state"], r["question"]))
    return out


def _ms_per_decision(
    dm: Any,
    model_path: str,
    reps: int,
    *,
    domain: str = "decide.judge.e2e",
    state_fmt: str = (
        "crit:the-test-suite-passed|len:<5000|tb:0|err:1|pass:10+|fail:{i}|exit:none|hit:2/4"
    ),
    question: Optional[str] = None,
    probes: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Median wall ms for a whole Decider.decide() served by this model file --
    normalize + engine miss + neighbor skip + the head + the recording, which is
    what a CALLER waits for. _ms_per_prediction times the head alone; this is the
    number the rung exists to move (the llm rung is ~2 s on the same fork).

    Every rep uses a DIFFERENT state so every rep is genuinely cold: a repeated
    fork would be answered by the engine and this would time the wrong rung. The
    source histogram is returned WITH the timing precisely so a reader can see
    which rung was actually timed -- a fast 'none' is not a fast answer.

    `probes` is a list of (state, question) the head is known to answer, so the
    headline number times an ANSWER rather than a decline; without it the fixed
    judge-shaped fork is used and may well come back 'none' (it does on the real
    corpus today), which the histogram then says out loud."""
    import tempfile

    try:
        import code_domains  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"error": f"code_domains unimportable: {str(exc)[:80]}"}
    if not getattr(code_domains, "_MLP_OK", False):
        return {"error": "world_model package not importable; no in-proc door to time"}
    rec = Path(tempfile.mkdtemp(prefix="nr-e2e-"))
    d = dm.Decider(
        code_domains.DomainEngines(),
        llm=None,
        llm_enabled=False,      # a fall-through must cost 0 here, not a DNS timeout
        embed_enabled=False,    # no neighbor rung: this times the NEURAL path
        record_dir=rec,
        neural_model=Path(model_path),
        neural_enabled=True,
    )
    if question is None:
        question = "Criterion: the test suite passed\n\n" + ("E   AssertionError: nope\n" * 160)
    items = (
        [(s, q) for s, q in probes[:reps]]
        if probes
        else [(state_fmt.format(i=i), question) for i in range(reps)]
    )
    ts: List[float] = []
    sources: Dict[str, int] = {}
    for i, (state, q) in enumerate(items):
        # A fresh domain per rep: the fork must be one the ENGINE has never seen,
        # or the engine rung answers it and this times the wrong thing.
        item = {
            "domain": f"{domain}.{i}" if probes else domain,
            "kind": "yesno",
            "state": state,
            "question": q,
            "options": ["yes", "no"],
        }
        t0 = time.perf_counter()
        got = d.decide(item)
        ts.append((time.perf_counter() - t0) * 1000.0)
        src = got.get("source") or "none"
        sources[src] = sources.get(src, 0) + 1
    order = sorted(ts)
    return {
        "ms_median": round(statistics.median(ts), 2),
        "ms_p95": round(order[max(0, int(len(order) * 0.95) - 1)], 2),
        "ms_max": round(max(ts), 2),
        "n": len(ts),
        "sources": sources,
        "probe": "held-out-ish forks the head answers" if probes else "fixed judge-shaped fork",
        "llm": "disabled (a fall-through costs 0 here; live it is the ~2 s brain call)",
    }


def _embed_fn(url: str):
    import decide as dm  # type: ignore

    def embed(texts: List[str]):
        return dm.embed_texts(texts, url=url, timeout=20.0)

    return embed


def train(a: argparse.Namespace) -> int:
    # The ckpt dir must be in the env BEFORE code_domains is imported: it reads
    # AITHER_WM_CKPT_DIR at import time, and a late set reads zero journal rows
    # while reporting success (the first run of this trainer did exactly that).
    ckpt = _ckpt_dir(a.ckpt)
    os.environ["AITHER_WM_CKPT_DIR"] = str(ckpt)
    import code_domains  # type: ignore
    import decide as dm  # type: ignore

    if dm.np is None:
        print("COULD NOT TRAIN: numpy is not importable here")
        return 2
    if not code_domains._MLP_OK:
        print("COULD NOT TRAIN: world_model package not importable (journals cannot be read)")
        return 2
    d = dm.Decider(code_domains.DomainEngines(), llm=None, embed_enabled=False,
                   neural_enabled=False)

    # rows -------------------------------------------------------------------
    try:
        rows = dm.neural_rows_from_journals(d)
    except Exception as exc:  # noqa: BLE001
        print(f"COULD NOT TRAIN: journals unreadable: {type(exc).__name__}: {exc}")
        return 2
    journal_rows = len(rows)
    cases_used = None
    snapshot = ckpt / "neural-rung.cases.jsonl"
    candidates = [Path(a.cases)] if a.cases else [ROOT / "tools" / "eval_cases" / "cases.jsonl",
                                                  snapshot]
    for cand in candidates:
        if cand.is_file():
            try:
                rows += dm.neural_rows_from_cases(cand)
            except ValueError as exc:
                print(f"COULD NOT TRAIN: {exc}")
                return 2
            cases_used = cand
            break
    if cases_used is None:
        rows += _bench_rows(dm)
        cases_used = Path("tools/judge_bench.py CASES (no cases file found)")
    elif a.snapshot_cases and cases_used.resolve() != snapshot.resolve():
        try:
            ckpt.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cases_used, snapshot)
        except OSError as exc:
            print(f"warning: could not snapshot cases into {snapshot}: {exc}")
    extra_rows = 0
    for spec in (a.extra_rows or []):
        p = Path(spec)
        if not p.is_file():
            print(f"COULD NOT TRAIN: --extra-rows {p} does not exist")
            return 2
        before = len(rows)
        with open(p, encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError as exc:
                    # A trainer that skips bad labels in silence trains on
                    # whatever is left and reports success.
                    print(f"COULD NOT TRAIN: {p}:{ln}: {exc}")
                    return 2
                missing = {"kind", "domain", "state", "option", "label", "fork"} - set(r)
                if missing:
                    print(f"COULD NOT TRAIN: {p}:{ln}: row missing {sorted(missing)}")
                    return 2
                rows.append(r)
        extra_rows += len(rows) - before
    if not rows:
        print("COULD NOT TRAIN: zero training rows")
        return 2

    # representations ----------------------------------------------------------
    feats: List[Any] = []
    embed_note = "hashed only (--embedder hashed)"
    if a.embedder in ("hashed", "auto", "both"):
        feats.append(dm.HashedFeatures())
    if a.embedder in ("fleet", "auto", "both"):
        url = (a.embed_url or dm.EMBED_URL).rstrip("/")
        fn = _embed_fn(url)
        try:
            t0 = time.perf_counter()
            fn(["probe"])
            embed_note = f"fleet embedder at {url} ({(time.perf_counter() - t0) * 1000:.0f} ms probe)"
            feats.append(dm.FleetFeatures(fn))
        except Exception as exc:  # noqa: BLE001
            embed_note = f"fleet embedder at {url} unreachable: {str(exc)[:100]}"
            if a.embedder == "fleet":
                print(f"COULD NOT TRAIN: {embed_note}")
                return 2
    if not feats:
        print("COULD NOT TRAIN: no representation available")
        return 2

    forced = _bench_forks(dm) if a.holdout_bench else set()
    t0 = time.perf_counter()
    rung, report = dm.fit_neural_rung(
        rows, feats, holdout_frac=a.holdout, forced_holdout=forced, l2=a.l2, steps=a.steps,
        lr=a.lr,
    )
    report["fit_s"] = round(time.perf_counter() - t0, 1)
    report["embedder_note"] = embed_note
    report["cases_path"] = str(cases_used)
    report["journal_rows"] = journal_rows
    report["extra_rows"] = extra_rows
    # What the FEATURES allow, regardless of how many rows arrive. Printed with
    # the verdict so "grow the labelled corpus" is never advised for a corpus
    # whose ceiling is already under the floor.
    try:
        from feature_ceiling import ceiling as _ceiling  # noqa: PLC0415
        per_fam: Dict[str, Any] = {}
        fams = {dm.NeuralRung.family_of(r["domain"]) for r in rows}
        for fam in sorted(fams):
            sub = [r for r in rows if dm.NeuralRung.family_of(r["domain"]) == fam]
            got = _ceiling(sub)
            if got.get("ceiling") is not None and got["cases"] >= 50:
                per_fam[fam] = got
        report["feature_ceiling"] = per_fam
    except Exception as exc:  # noqa: BLE001 -- an advisory miss is not a bad model
        report["feature_ceiling"] = {"error": f"{type(exc).__name__}: {str(exc)[:80]}"}
    # lift the primary head's per-family decomposition to the top level so the
    # printer does not have to know which head is primary
    for _k, _h in (report.get("heads") or {}).items():
        if _k.endswith(":yesno") and _h.get("family_heldout"):
            report["family_heldout"] = _h["family_heldout"]
            break
    report["extra_sources"] = [str(Path(s).name) for s in (a.extra_rows or [])]
    report["ckpt_dir"] = str(ckpt)
    report["ms_per_prediction"] = {}
    for f in feats:
        try:
            report["ms_per_prediction"][f.name] = _ms_per_prediction(dm, rung, f, a.reps)
        except Exception as exc:  # noqa: BLE001
            report["ms_per_prediction"][f.name] = f"failed: {str(exc)[:80]}"

    # sanity gates: worse than chance, or calibration made worse -> not written
    bad: List[str] = []
    for key, h in report["heads"].items():
        if "skipped" in h:
            continue
        if h["heldout_accuracy"] is not None and h["heldout_forks_scored"] >= 20:
            if h["heldout_accuracy"] < 0.5:
                bad.append(f"{key}: held-out accuracy {h['heldout_accuracy']} < 0.5")
    # The temperature minimises held-out NLL, the proper scoring rule; 10-bin ECE
    # on <100 forks is noisy enough to move the other way, so a worse ECE is
    # REPORTED (warnings), not a reason to refuse the model.
    warns: List[str] = []
    for key, h in report["heads"].items():
        if "skipped" in h or h["ece_after"] is None or h["ece_before"] is None:
            continue
        if h["ece_after"] > h["ece_before"] + 1e-9:
            warns.append(f"{key}: ECE after temperature {h['ece_after']} > before {h['ece_before']}")
    report["gates_failed"] = bad
    report["warnings"] = warns
    # Promotion: a head serves only if it EARNS the serving path (enough held-out
    # forks answered at the serving threshold, accurately enough). Everything else
    # is written beside it as a candidate, so the door keeps what it had.
    # The out-of-distribution probe, BEFORE promotion is decided. Every
    # in-corpus gate passed a model this refuted in one run (2026-09-20).
    try:
        report["ood"] = _ood_probe(dm, rung, dm.HashedFeatures())
    except Exception as exc:  # noqa: BLE001 -- a probe miss must not be a pass
        report["ood"] = {"available": False, "why": f"{type(exc).__name__}: {exc}"}
    report["promotion"] = dm.neural_promotion(report, ood=report.get("ood"))
    if a.promote_anyway:
        report["promotion"]["promote"] = True
        report["promotion"]["forced_by"] = "--promote-anyway"
    out = Path(a.out) if a.out else ckpt / dm.NEURAL_FILE
    if bad or not rung.heads:
        report["saved"] = None
    else:
        rung.meta.update({k: v for k, v in report.items() if k != "heads"})
        promoting = bool(report["promotion"]["promote"])
        target = out if promoting else dm.candidate_path(out)
        # Only the heads that individually earned it reach the SERVING path; the
        # candidate file keeps everything so a refused head stays inspectable.
        only = (
            {k for k, ok in (report["promotion"].get("passed") or {}).items() if ok}
            if promoting else None
        )
        report["saved"] = str(rung.save(target, only=only))
        if promoting and report["promotion"].get("refused"):
            report["refused_heads"] = report["promotion"]["refused"]
        report["promoted"] = bool(report["promotion"]["promote"])
        # End-to-end, through the file just written -- a head that is fast in a
        # micro-benchmark and slow behind the ladder has not moved the gap. Time
        # forks the head ANSWERS (a decline is fast for the wrong reason); the
        # source histogram in the output is what proves which happened.
        try:
            report["ms_per_decision"] = _ms_per_decision(
                dm, report["saved"], a.reps,
                probes=_answerable_probes(dm, rung, rows, a.reps),
            )
        except Exception as exc:  # noqa: BLE001 -- a timing miss is not a bad model
            report["ms_per_decision"] = {"error": f"{type(exc).__name__}: {str(exc)[:80]}"}

    if a.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_report(report)
    for w in warns:
        print(f"warning: {w}")
    if bad:
        print("\nVERDICT: model NOT written --", "; ".join(bad))
        return 1
    if not rung.heads:
        print("\nVERDICT: nothing fit (every kind skipped) -- model not written")
        return 2
    if report["promotion"]["promote"]:
        print(f"\nVERDICT: neural rung PROMOTED to {report['saved']} "
              f"(head {report['promotion']['head']})")
    else:
        print(f"\nVERDICT: HELD -- written as a candidate to {report['saved']}, NOT serving.")
        for r in report["promotion"]["reasons"]:
            print(f"  {r}")
        _ood = report.get("ood") or {}
        if _ood.get("available"):
            print(f"  out-of-distribution probe ({_ood.get('source')}):")
            print(f"    answered {_ood.get('answered')}/{_ood.get('items')} "
                  f"(coverage {_ood.get('coverage')}) at accuracy {_ood.get('accuracy')}, "
                  f"95% lower bound {_ood.get('accuracy_lower95')}")
        elif _ood:
            print(f"  out-of-distribution probe UNAVAILABLE: {_ood.get('why')}")
        _ceil = report.get("feature_ceiling") or {}
        _blocked = {k: v for k, v in _ceil.items()
                    if isinstance(v, dict) and (v.get("ceiling") or 1.0) < dm.NEURAL_PROMOTE_MIN_ACC}
        if _blocked:
            print("  MORE ROWS CANNOT FIX THIS. The features cap these families below "
                  f"the {dm.NEURAL_PROMOTE_MIN_ACC} floor:")
            for fam, v in sorted(_blocked.items()):
                print(f"    {fam}: ceiling {v['ceiling']} over {v['states']} distinct "
                      f"states from {v['cases']} cases (majority {v['majority']})")
                worst = (v.get("worst_states") or [{}])[0]
                if worst.get("state"):
                    print(f"      worst bucket: n={worst['cases']} purity={worst['purity']} "
                          f"{worst['state'][:76]}")
            print("  Richer features, or a floor that fits the task -- not more rows. "
                  "`tools/feature_ceiling.py` is the measurement.")
        else:
            print("  The door keeps the rung it had; nothing regressed. Grow the labelled "
                      "corpus and run this again, or --promote-anyway to override deliberately.")
    return 0


def _print_report(r: Dict[str, Any]) -> None:
    print(f"rows: {r['rows']}  per source: {r['rows_per_source']}  "
          f"(journal rows {r.get('journal_rows')}, cases: {r.get('cases_path')})")
    if r.get("extra_rows"):
        print(f"extra rows: {r['extra_rows']} from {', '.join(r.get('extra_sources') or [])}")
    fam = r.get("family_heldout") or {}
    if fam:
        print("\nheld-out BY DOMAIN FAMILY -- the number the floor is about is "
              "`decide.judge`;")
        print("a big extra corpus can lift the headline while that one falls.")
        print(f"  {'family':22}{'rows':>7}{'forks':>7}{'acc':>8}")
        for name in sorted(fam, key=lambda k: -fam[k]["rows"]):
            f = fam[name]
            acc = "n/a" if f["accuracy"] is None else f"{f['accuracy']:.3f}"
            print(f"  {name:22}{f['rows']:>7}{f['forks']:>7}{acc:>8}")
    print(f"held-out: {int(r['holdout_frac'] * 100)}% of forks + {r['forced_holdout_forks']} "
          f"bench forks forced; embedders: {r['embedders']}; {r.get('embedder_note')}")
    print(f"fit: {r.get('fit_s')} s\n")
    hdr = (f"{'head':16}{'train':>7}{'held':>6}{'forks':>7}{'acc':>7}{'ECE pre':>9}"
           f"{'ECE post':>10}{'T':>7}{'dim':>6}  note")
    print(hdr)
    for key, h in r["heads"].items():
        if "skipped" in h:
            print(f"{key:16}{h['rows_train']:>7}{h['rows_heldout']:>6}  skipped: {h['skipped']}")
            continue
        note = "TOO SMALL to mean anything" if h["too_small"] else ""
        acc = "n/a" if h["heldout_accuracy"] is None else f"{h['heldout_accuracy']:.3f}"
        pre = "n/a" if h["ece_before"] is None else f"{h['ece_before']:.3f}"
        post = "n/a" if h["ece_after"] is None else f"{h['ece_after']:.3f}"
        print(f"{key:16}{h['rows_train']:>7}{h['rows_heldout']:>6}{h['heldout_forks_scored']:>7}"
              f"{acc:>7}{pre:>9}{post:>10}{h['temperature']:>7.3f}{h['dim']:>6}  {note}")
    print(f"\nms per prediction (median, judge-shaped item, 2 options): {r['ms_per_prediction']}")
    e2e = r.get("ms_per_decision")
    if e2e:
        if e2e.get("error"):
            print(f"ms per DECISION (end to end): not measured -- {e2e['error']}")
        else:
            print(f"ms per DECISION (end to end, {e2e['n']} cold forks through the whole "
                  f"ladder, {e2e.get('probe')}): median {e2e['ms_median']} ms, p95 "
                  f"{e2e.get('ms_p95')} ms, max {e2e['ms_max']} ms, answered by "
                  f"{e2e['sources']}")
    for key, h in r["heads"].items():
        g = (h or {}).get("at_serving_threshold")
        if g:
            print(f"  {key} at the serving threshold p>={g['threshold']}: would answer "
                  f"{g['forks_answered']}/{g['forks_scored']} held-out forks "
                  f"(coverage {g['coverage']}) at accuracy {g['accuracy']}")


def self_test() -> int:
    """Proves the trainer can FAIL: a corrupt case row raises, a separable
    synthetic set trains to >= 0.9 held-out, temperature moves probabilities but
    never the argmax, and a saved model reloads to the same answer."""
    import decide as dm  # type: ignore

    if dm.np is None:
        print("SELF-TEST DEAD: numpy missing")
        return 2
    fails: List[str] = []
    tmp = Path(tempfile.mkdtemp(prefix="nr-selftest-"))
    os.environ["AITHER_WM_CKPT_DIR"] = str(tmp)

    # 1. a corrupt row must raise, naming the line
    bad = tmp / "bad.jsonl"
    bad.write_text(
        '{"output": "ok", "criteria": [{"text": "fine", "label": true}]}\n'
        '{"output": "ok", "criteria": [{"text": "fine", "label": "yes"}]}\n',
        encoding="utf-8",
    )
    try:
        dm.neural_rows_from_cases(bad)
        fails.append("corrupt row (label 'yes') was accepted")
    except ValueError as exc:
        if ":2:" not in str(exc):
            fails.append(f"corrupt row raised without the line number: {exc}")

    # 2. separable synthetic forks: state carries the answer
    rows = []
    # 320, not 160: a 20% fork holdout of 160 leaves ~23 held-out forks, and a
    # PERFECT head on 23 forks has a Wilson 95% lower bound of 0.8569 -- below
    # the 0.9 floor, correctly. The self-test's claim is 'a separable head IS
    # promoted', so it needs enough evidence to make that claim provable.
    for i in range(320):
        good = i % 2 == 0
        state = f"build:{'green' if good else 'red'},id:{i},tail:{'ok' if good else 'boom'}"
        for opt, lab in (("yes", good), ("no", not good)):
            rows.append(dm._row("yesno", "decide.selftest", state, "did it pass?", opt,
                                1.0 if lab else 0.0, "synthetic"))
    rung, rep = dm.fit_neural_rung(rows, [dm.HashedFeatures()], steps=300)
    h = rep["heads"]["hashed:yesno"]
    if h["heldout_accuracy"] is None or h["heldout_accuracy"] < 0.9:
        fails.append(f"synthetic held-out accuracy {h['heldout_accuracy']} < 0.9")
    if not h["calibrated"]:
        fails.append("temperature was not fitted on a non-empty held-out split")

    # 3. temperature changes probabilities, never the argmax
    feats = dm.HashedFeatures()
    args = ("yesno", "decide.selftest", "build:red,id:9999,tail:boom", "did it pass?",
            ["yes", "no"], feats)
    p1 = rung.predict(*args, temperature=0.5)
    p2 = rung.predict(*args, temperature=4.0)
    if p1["answer"] != p2["answer"]:
        fails.append("temperature changed the argmax")
    if abs(p1["probability"] - p2["probability"]) < 1e-3:
        fails.append("temperature did not change the probability")
    if p1["answer"] != "no":
        fails.append(f"a red build was graded {p1['answer']!r}")

    # 3b. the promotion gate: a separable head earns the serving path; the same
    # report with a poor accuracy-at-threshold is HELD
    promo = dm.neural_promotion(rep)
    if not promo["promote"]:
        fails.append(f"a perfectly separable head was not promoted: {promo['reasons']}")
    weak = {"heads": {"hashed:yesno": dict(rep["heads"]["hashed:yesno"])}}
    weak["heads"]["hashed:yesno"]["at_serving_threshold"] = {
        "threshold": 0.75, "forks_scored": 75, "forks_answered": 20,
        "coverage": 0.27, "accuracy": 0.75,
    }
    if dm.neural_promotion(weak)["promote"]:
        fails.append("a head that would answer 20 forks at 0.75 accuracy was promoted")
    thin = {"heads": {"hashed:yesno": dict(rep["heads"]["hashed:yesno"])}}
    thin["heads"]["hashed:yesno"]["at_serving_threshold"] = {
        "threshold": 0.75, "forks_scored": 75, "forks_answered": 3,
        "coverage": 0.04, "accuracy": 1.0,
    }
    if dm.neural_promotion(thin)["promote"]:
        fails.append("a head judged on only 3 answered forks was promoted")

    # 4. save -> load -> same answer; corrupt file -> None, not a crash
    path = rung.save(tmp / "neural-rung.npz")
    back = dm.NeuralRung.load(path)
    if back is None or back.predict(*args)["probabilities"] != rung.predict(*args)["probabilities"]:
        fails.append("saved model did not reload to the same probabilities")
    (tmp / "corrupt.npz").write_bytes(b"not an npz")
    if dm.NeuralRung.load(tmp / "corrupt.npz") is not None:
        fails.append("a corrupt model file loaded as a model")
    if dm.NeuralRung.load(tmp / "missing.npz") is not None:
        fails.append("a missing model file loaded as a model")

    # 5. the end-to-end timer must time the NEURAL rung, and must SAY SO when it
    # did not. A timer that reports 0.4 ms while the door actually answered
    # 'none' would make an inactive rung look like a fast one -- that is the
    # exact lie this step exists to catch, so it is asserted in both directions.
    probe = dict(
        domain="decide.selftest",
        state_fmt="build:red,id:{i},tail:boom",
        question="did it pass?",
    )
    e2e = _ms_per_decision(dm, str(path), 5, **probe)
    if e2e.get("error"):
        fails.append(f"end-to-end timer could not run: {e2e['error']}")
    else:
        if not e2e["sources"].get("neural"):
            fails.append(f"timed a door that never used the neural rung: {e2e['sources']}")
        if not (e2e["ms_median"] > 0):
            fails.append(f"end-to-end median {e2e['ms_median']} is not a positive time")
        gone = _ms_per_decision(dm, str(tmp / "missing.npz"), 3, **probe)
        if not gone.get("error") and gone["sources"].get("neural"):
            fails.append("a MISSING model file still reported neural answers")

    for f in fails:
        print(f"SELF-TEST FAIL: {f}")
    print("SELF-TEST", "FAILED" if fails else
          f"OK (synthetic held-out acc {h['heldout_accuracy']}, ECE {h['ece_before']} -> "
          f"{h['ece_after']}, T={h['temperature']})")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", help="AITHER_WM_CKPT_DIR (journals in, model out)")
    ap.add_argument("--cases", help="labelled cases jsonl (default: tools/eval_cases/cases.jsonl)")
    ap.add_argument("--out", help="model path (default: <ckpt>/neural-rung.npz)")
    ap.add_argument("--embedder", choices=["auto", "hashed", "fleet", "both"], default="auto")
    ap.add_argument("--embed-url", help="fleet embedder base URL (host: http://127.0.0.1:8229)")
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--no-holdout-bench", dest="holdout_bench", action="store_false",
                    help="do NOT force the judge bench's forks into held-out")
    ap.add_argument("--extra-rows", action="append", metavar="PATH",
                    help="extra pre-built neural rows as jsonl (repeatable); "
                         "tools/tool_event_rows.py writes this shape")
    ap.add_argument("--family-report", action="store_true",
                    help="held-out accuracy broken down by domain family -- "
                         "without it, a big extra corpus can outvote the judge "
                         "forks and the headline number hides it")
    ap.add_argument("--no-snapshot-cases", dest="snapshot_cases", action="store_false",
                    help="do not copy the cases file into <ckpt>/neural-rung.cases.jsonl")
    # 800 / 1e-4 measured best on the harvested judge corpus 2026-09-20 (held-out
    # fork accuracy 0.707 vs 0.640 at 400/1e-3, ECE-after 0.062 vs 0.059).
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--reps", type=int, default=20, help="timing repetitions")
    ap.add_argument(
        "--promote-anyway",
        action="store_true",
        help="write to the SERVING path even if the head did not earn it "
        "(deliberate override; the reasons are still printed)",
    )
    ap.add_argument("--report", action="store_true", help="(default output; kept for the brief)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    return train(a)


if __name__ == "__main__":
    sys.exit(main())
