#!/usr/bin/env python3
"""judge_heldout_bench -- the HONEST version of judge_bench's "door warm 100%".

`judge_bench` teaches the door on the same 24 forks it then grades. That number
is a cache hit and was published as if it were generalisation; this file is the
correction. It splits the 134 labelled cases in `tools/eval_cases/cases.jsonl`,
teaches the door on the TRAIN half only, and grades the TEST half it has never
been told the answer to.

Two splits, because they ask different questions and only the pair is honest:

  item    a random split of the CASES. Test items are unseen, but a test item's
          criteria may share a feature signature with a taught one -- this is
          the realistic "second week of running evals" case, and it is the one
          an LLM-judge setup is actually being compared against.
  source  leave-one-SOURCE-out (pytest / ruff / curl / git / podman / ...): the
          test family was never taught at all. This is the hard case and the
          number that says whether the door generalises to a NEW tool's output
          or only to more of what it has seen.

Baselines printed alongside, because an accuracy with no baseline is a number
with no claim attached:

  majority   always answer the most common label in TRAIN. A judge that cannot
             beat this has learned nothing.
  warm       teach on TEST and then grade TEST -- i.e. exactly what judge_bench
             reports. Printed so the gap between it and `item` is visible and
             nobody republishes the cache hit by accident.

An `unknown` verdict is never scored correct. Coverage and
accuracy-when-answered are both printed: a leg that abstains on half the items
and is right on the rest is not 29% accurate at judging.

  python tools/judge_heldout_bench.py                  # both splits
  python tools/judge_heldout_bench.py --split item --seed 7 --json
  python tools/judge_heldout_bench.py --self-test

Exit 0 when every leg ran and held-out accuracy beat the majority baseline,
1 when it did not, 2 when it could not run (no cases, no importable door).
Never 0 on silence.

WHAT THIS MEASURED, 2026-09-20, so nobody has to re-derive it:

  item split     38.3% overall, 47.2% coverage, 81.2% accuracy WHEN ANSWERED,
                 against a 50.5% majority baseline. 46.7% of the test criteria
                 had been taught verbatim -- coverage and that number track
                 each other.
  strict source  0.8% coverage on 247 criteria never taught in any form.
  warm           99.1%, which is what judge_bench reports.

The door is a CACHE WITH A FAIL-CLOSED MISS, not a judge that generalises. On a
fork it has been taught it is ~99% right for zero model calls; on a fork it has
not, it abstains and the call falls to the LLM rung. "Decisions you already made
should never be paid for twice" is exactly the claim the numbers support, and
"it judges new things for free" is not one it has ever supported. The saving
equals your repeat rate, which for an eval suite re-run on every commit is high
-- but it has to be stated as a repeat rate, not as accuracy.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
SVC = HERE.parent
CASES_PATH = HERE / "eval_cases" / "cases.jsonl"

# (id, source, output, [(criterion, label), ...])
Case = Tuple[str, str, str, List[Tuple[str, bool]]]


def load_cases(path: Path) -> List[Case]:
    if not path.is_file():
        return []
    out: List[Case] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        crits = [
            (c["text"], bool(c["label"]))
            for c in (row.get("criteria") or [])
            if isinstance(c, dict) and "text" in c and "label" in c
        ]
        if not crits:
            continue
        src = str(row.get("source") or "?").split(":", 1)[0]
        out.append((str(row.get("id") or "?"), src, str(row.get("output") or ""), crits))
    return out


def _make_judge(ckpt: Path):
    """An in-process judge against a FRESH ckpt dir, so nothing this bench does
    touches the live journals and no run inherits another run's teaching.

    No brain and no neural rung: an llm rung would make this a measurement of
    the brain, not of what the door learned from the outcomes it was given, and
    the two legs would not be comparable run to run.
    """
    import os

    sys.path.insert(0, str(SVC))
    os.environ["AITHER_WM_CKPT_DIR"] = str(ckpt)
    import code_domains  # noqa: E402  -- path set above
    import decide  # noqa: E402
    import judge as judge_mod  # noqa: E402

    if not code_domains._MLP_OK:
        raise RuntimeError("world_model package not importable")
    d = decide.Decider(
        code_domains.DomainEngines(),
        llm=None,
        embed_enabled=False,
        neural_enabled=False,
    )
    return judge_mod.Judge(d)


def _grade(judge, cases: List[Case], domain: str) -> Dict[str, Any]:
    t0, right, total, srcs = time.perf_counter(), 0, 0, Counter()
    for _cid, _src, output, crits in cases:
        res = judge.judge(output, [c for c, _ in crits], domain=domain)
        for (_c, truth), v in zip(crits, res.get("verdicts") or []):
            total += 1
            right += int(v.get("pass") is truth)
            srcs[v.get("source", "?")] += 1
    answered = total - srcs.get("none", 0)
    return {
        "items": total,
        "accuracy_pct": round(100.0 * right / max(1, total), 1),
        "answered": answered,
        "coverage_pct": round(100.0 * answered / max(1, total), 1),
        "accuracy_when_answered_pct": (
            round(100.0 * right / answered, 1) if answered else None
        ),
        "model_calls": srcs.get("llm", 0),
        "sources": dict(srcs),
        "wall_s": round(time.perf_counter() - t0, 1),
    }


def _teach(judge, cases: List[Case], domain: str) -> int:
    n = 0
    for _cid, _src, output, crits in cases:
        for c, truth in crits:
            judge.teach(output, c, truth, domain=domain)
            n += 1
    return n


def _majority(train: List[Case], test: List[Case]) -> Dict[str, Any]:
    """Always answer TRAIN's most common label. The floor any real judge clears."""
    labels = Counter(lb for _i, _s, _o, cr in train for _c, lb in cr)
    guess = labels.most_common(1)[0][0] if labels else True
    total = sum(len(cr) for _i, _s, _o, cr in test)
    right = sum(int(lb is guess) for _i, _s, _o, cr in test for _c, lb in cr)
    return {
        "guess": guess,
        "items": total,
        "accuracy_pct": round(100.0 * right / max(1, total), 1),
    }


def _leg_worker() -> int:
    """`--_leg <json-file>`: teach, grade, print the report as JSON. One leg, one
    process, one fresh store."""
    payload = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    train = [tuple(c[:3]) + ([tuple(x) for x in c[3]],) for c in payload["train"]]
    test = [tuple(c[:3]) + ([tuple(x) for x in c[3]],) for c in payload["test"]]
    with tempfile.TemporaryDirectory(prefix="judge-heldout-leg-") as d:
        judge = _make_judge(Path(d))
        taught = _teach(judge, train, payload["domain"])  # type: ignore[arg-type]
        got = _grade(judge, test, payload["domain"])  # type: ignore[arg-type]
    got["taught"] = taught
    print("___LEG___" + json.dumps(got))
    return 0


def _run_leg(train: List[Case], test: List[Case], domain: str, label: str) -> Dict[str, Any]:
    """Run the leg in a SUBPROCESS.

    `AITHER_WM_CKPT_DIR` is read at module import, so a second `_make_judge()` in
    the same process keeps writing to the FIRST leg's temp dir. Measured
    2026-09-20: fold 1 scored 0.0% coverage (honest) and folds 2..9 scored ~100%,
    because fold 1's train set had already taught every other source -- i.e.
    every later fold's TEST family -- into the store they all shared. An
    in-process loop cannot measure held-out anything here.
    """
    with tempfile.TemporaryDirectory(prefix=f"judge-heldout-{label}-") as d:
        arg = Path(d) / "leg.json"
        arg.write_text(
            json.dumps({"train": train, "test": test, "domain": domain}),
            encoding="utf-8",
        )
        p = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--_leg", str(arg)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(SVC), timeout=1800,
        )
    line = next(
        (ln for ln in (p.stdout or "").splitlines() if ln.startswith("___LEG___")), None
    )
    if line is None:
        raise RuntimeError(
            f"leg {label} produced no report (exit {p.returncode}): "
            f"{((p.stderr or p.stdout or '').strip().splitlines() or [''])[-1][:300]}"
        )
    got = json.loads(line[len("___LEG___"):])
    got["train_cases"] = len(train)
    got["test_cases"] = len(test)
    return got


def run_item_split(cases: List[Case], seed: int, frac: float, domain: str) -> Dict[str, Any]:
    rows = list(cases)
    random.Random(seed).shuffle(rows)
    cut = int(len(rows) * frac)
    train, test = rows[:cut], rows[cut:]
    seen = {c for _i, _s, _o, cr in train for c, _lb in cr}
    tot = sum(len(cr) for _i, _s, _o, cr in test)
    hit = sum(1 for _i, _s, _o, cr in test for c, _lb in cr if c in seen)
    out = {
        "split": "item",
        "seed": seed,
        "heldout": _run_leg(train, test, domain, "item"),
        "majority": _majority(train, test),
        # the cache hit judge_bench reports, on the SAME test set
        "warm_cache_hit": _run_leg(test, test, domain, "warm"),
        # the number that EXPLAINS the coverage: without it a reader guesses
        "verbatim_taught_pct": round(100.0 * hit / max(1, tot), 1),
        "verbatim_taught": hit,
        "test_criteria": tot,
    }
    return out


def _strip_seen_criteria(train: List[Case], test: List[Case]) -> Tuple[List[Case], int, int]:
    """Drop from TEST every criterion whose TEXT was taught in TRAIN.

    Without this, leave-one-source-out is not held out at all: the door keys a
    fork on the criterion, and 24-58% of each source's criteria are shared
    verbatim with the other sources ("the command succeeded" appears 46 times
    across 9 tools). The first run of this bench scored 98.9% on the supposedly
    HARD split and 38.3% on the easy one -- backwards, which is the tell. That
    98.9% was mostly the door recalling forks it had been taught under another
    tool's name. Measured 2026-09-20.
    """
    seen = {c for _i, _s, _o, cr in train for c, _lb in cr}
    kept: List[Case] = []
    dropped = 0
    for cid, src, out, cr in test:
        fresh = [(c, lb) for c, lb in cr if c not in seen]
        dropped += len(cr) - len(fresh)
        if fresh:
            kept.append((cid, src, out, fresh))
    return kept, dropped, sum(len(c[3]) for c in kept)


def run_source_split(cases: List[Case], domain: str, min_test: int = 5) -> Dict[str, Any]:
    by_src: Dict[str, List[Case]] = {}
    for c in cases:
        by_src.setdefault(c[1], []).append(c)
    folds: List[Dict[str, Any]] = []
    for src, held in sorted(by_src.items()):
        if len(held) < min_test:
            continue
        train = [c for c in cases if c[1] != src]
        leg = _run_leg(train, held, domain, f"src-{src}")
        leg["held_source"] = src
        leg["majority"] = _majority(train, held)
        # and again with every verbatim-taught criterion removed from the test
        strict_test, dropped, remaining = _strip_seen_criteria(train, held)
        leg["criteria_also_in_train"] = dropped
        leg["strict_items"] = remaining
        if remaining >= 5:
            s = _run_leg(train, strict_test, domain, f"strict-{src}")
            leg["strict"] = {
                "items": s["items"],
                "accuracy_pct": s["accuracy_pct"],
                "coverage_pct": s["coverage_pct"],
                "accuracy_when_answered_pct": s["accuracy_when_answered_pct"],
                "majority_pct": _majority(train, strict_test)["accuracy_pct"],
            }
        else:
            leg["strict"] = None
        folds.append(leg)
    if not folds:
        return {"split": "source", "folds": [], "overall": None}
    tot = sum(f["items"] for f in folds)
    right = sum(f["accuracy_pct"] / 100.0 * f["items"] for f in folds)
    ans = sum(f["answered"] for f in folds)
    maj = sum(f["majority"]["accuracy_pct"] / 100.0 * f["items"] for f in folds)
    strict_folds = [f["strict"] for f in folds if f.get("strict")]
    strict_tot = sum(s["items"] for s in strict_folds)
    strict_overall = None
    if strict_tot:
        strict_overall = {
            "items": strict_tot,
            "accuracy_pct": round(
                100.0 * sum(s["accuracy_pct"] / 100.0 * s["items"] for s in strict_folds)
                / strict_tot, 1),
            "coverage_pct": round(
                100.0 * sum(s["coverage_pct"] / 100.0 * s["items"] for s in strict_folds)
                / strict_tot, 1),
            "majority_pct": round(
                100.0 * sum(s["majority_pct"] / 100.0 * s["items"] for s in strict_folds)
                / strict_tot, 1),
        }
    return {
        "split": "source",
        "folds": folds,
        "overall": {
            "items": tot,
            "accuracy_pct": round(100.0 * right / max(1, tot), 1),
            "coverage_pct": round(100.0 * ans / max(1, tot), 1),
            "majority_pct": round(100.0 * maj / max(1, tot), 1),
            "model_calls": sum(f["model_calls"] for f in folds),
            "criteria_also_in_train": sum(f["criteria_also_in_train"] for f in folds),
            "strict": strict_overall,
        },
    }


def _fmt(rep: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    it = rep.get("item")
    if it:
        h, m, w = it["heldout"], it["majority"], it["warm_cache_hit"]
        out.append(
            f"item split (seed {it['seed']}) -- {h['train_cases']} taught cases, "
            f"{h['test_cases']} UNSEEN test cases"
        )
        out.append(f"  {'leg':30}{'acc':>8}{'cover':>8}{'acc|ans':>10}{'calls':>7}")
        out.append(
            f"  {'held-out (the real number)':30}{h['accuracy_pct']:>7.1f}%"
            f"{h['coverage_pct']:>7.1f}%"
            f"{(h['accuracy_when_answered_pct'] or 0):>9.1f}%{h['model_calls']:>7}"
        )
        out.append(f"  {'majority baseline':30}{m['accuracy_pct']:>7.1f}%{'-':>8}{'-':>10}{0:>7}")
        out.append(
            f"  {'warm = judge_bench cache hit':30}{w['accuracy_pct']:>7.1f}%"
            f"{w['coverage_pct']:>7.1f}%"
            f"{(w['accuracy_when_answered_pct'] or 0):>9.1f}%{w['model_calls']:>7}"
        )
        out.append(
            f"  gap warm - held-out = "
            f"{round(w['accuracy_pct'] - h['accuracy_pct'], 1)} points "
            "(that gap IS the cache hit)"
        )
        out.append(
            f"  coverage {h['coverage_pct']}% vs {it['verbatim_taught_pct']}% of test"
            f" criteria taught VERBATIM ({it['verbatim_taught']}/{it['test_criteria']})"
        )
        out.append(
            "  -- those two tracking each other is the finding: the door answers the"
        )
        out.append(
            "     forks it was TAUGHT and abstains on the rest. It is a cache with a"
        )
        out.append(
            "     fail-closed miss, not a judge that generalises. Repeats are free at"
        )
        out.append(
            f"     {h['accuracy_when_answered_pct']}% accuracy; novel forks cost a model call."
        )
    sp = rep.get("source")
    if sp and sp.get("overall"):
        o = sp["overall"]
        out.append("")
        out.append("leave-one-SOURCE-out -- the test family was never taught at all")
        out.append(
            f"  {'held source':16}{'test':>6}{'acc':>7}{'cov':>6}{'maj':>6}"
            f"{'|':>3}{'taught':>7}{'strict':>7}{'acc':>7}{'cov':>6}{'maj':>6}"
        )
        for f in sp["folds"]:
            s = f.get("strict")
            tail = (
                f"{'|':>3}{f['criteria_also_in_train']:>7}{s['items']:>7}"
                f"{s['accuracy_pct']:>6.1f}%{s['coverage_pct']:>5.1f}%{s['majority_pct']:>5.1f}%"
                if s else f"{'|':>3}{f['criteria_also_in_train']:>7}{'few':>7}"
            )
            out.append(
                f"  {f['held_source']:16}{f['items']:>6}{f['accuracy_pct']:>6.1f}%"
                f"{f['coverage_pct']:>5.1f}%{f['majority']['accuracy_pct']:>5.1f}%{tail}"
            )
        st = o.get("strict") or {}
        tail = (
            f"{'|':>3}{o.get('criteria_also_in_train', 0):>7}{st['items']:>7}"
            f"{st['accuracy_pct']:>6.1f}%{st['coverage_pct']:>5.1f}%{st['majority_pct']:>5.1f}%"
            if st else ""
        )
        out.append(
            f"  {'OVERALL':16}{o['items']:>6}{o['accuracy_pct']:>6.1f}%"
            f"{o['coverage_pct']:>5.1f}%{o['majority_pct']:>5.1f}%{tail}"
        )
        out.append(
            "  `taught` = criteria whose TEXT was already taught under ANOTHER tool's"
        )
        out.append(
            "  name. The STRICT columns drop those and are the honest number; the"
        )
        out.append(
            "  left-hand ones include forks the door had already been told."
        )
    return out


def self_test() -> int:
    fails: List[str] = []

    def check(name: str, ok: bool) -> None:
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
        if not ok:
            fails.append(name)

    # loader
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.jsonl"
        p.write_text(
            json.dumps({"id": "a", "source": "pytest:x", "output": "o",
                        "criteria": [{"text": "t", "label": True}]}) + "\n"
            + json.dumps({"id": "b", "source": "ruff", "output": "o2",
                          "criteria": []}) + "\n",
            encoding="utf-8",
        )
        rows = load_cases(p)
        check("loader drops a case with no criteria", len(rows) == 1)
        check("loader strips the source's suffix", rows[0][1] == "pytest")
        check("a missing file loads as empty, never as a pass", load_cases(Path(d) / "nope") == [])

    # the majority baseline is a real baseline, not a constant
    tr = [("1", "s", "o", [("c", True), ("c2", True), ("c3", False)])]
    te = [("2", "s", "o", [("c", True), ("c2", False)])]
    m = _majority(tr, te)
    check("majority picks TRAIN's most common label", m["guess"] is True)
    check("majority scores on TEST only", m["items"] == 2 and m["accuracy_pct"] == 50.0)

    # an all-unknown grade must not read as accurate
    class _Dumb:
        def judge(self, output, criteria, domain):
            return {"verdicts": [{"pass": None, "source": "none"} for _ in criteria]}

    g = _grade(_Dumb(), te, "d")
    check("an abstaining judge scores 0%, not 100%", g["accuracy_pct"] == 0.0)
    check("an abstaining judge reports 0% coverage", g["coverage_pct"] == 0.0)
    check("accuracy_when_answered is None, not 0, on no answers",
          g["accuracy_when_answered_pct"] is None)

    # the isolation arm: two legs in a row, the second taught NOTHING. If the
    # store leaked, the second would answer from the first leg's teaching.
    tiny = [
        ("x1", "s", "the build failed with exit code 2", [("the build failed", True)]),
        ("x2", "s", "the build failed with exit code 3", [("the build failed", True)]),
    ]
    try:
        warm = _run_leg(tiny, tiny, "decide.judge.selftest", "iso-a")
        cold = _run_leg([], tiny, "decide.judge.selftest", "iso-b")
        check("a taught leg answers", warm["coverage_pct"] > 0.0)
        check(
            "an untaught leg run AFTER it answers nothing (no shared store)",
            cold["coverage_pct"] == 0.0,
        )
    except Exception as e:  # noqa: BLE001 -- a leg that cannot run is a FAIL, not a skip
        check(f"isolation arm ran ({type(e).__name__}: {e})", False)

    print(f"\n{'PASS' if not fails else 'FAIL'}: {len(fails)} failing check(s)")
    return 1 if fails else 0


def main(argv: Optional[List[str]] = None) -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--_leg":
        return _leg_worker()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", default=str(CASES_PATH))
    ap.add_argument("--split", choices=("item", "source", "both"), default="both")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train-frac", type=float, default=0.5)
    # the engine registry only knows `code`, `sandbox` and `decide.<fork>` --
    # a bare "judge" raises ValueError on the first teach()
    ap.add_argument("--domain", default="decide.judge.heldout")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        return self_test()

    cases = load_cases(Path(a.cases))
    if len(cases) < 20:
        print(f"COULD NOT JUDGE: {len(cases)} labelled case(s) at {a.cases}; need >= 20")
        return 2
    try:
        with tempfile.TemporaryDirectory() as d:
            _make_judge(Path(d))
    except Exception as e:  # noqa: BLE001 -- any import failure is could-not-judge
        print(f"COULD NOT JUDGE: door not importable from {SVC}: {type(e).__name__}: {e}")
        return 2

    rep: Dict[str, Any] = {
        "cases": len(cases),
        "criteria": sum(len(c[3]) for c in cases),
        "sources": dict(Counter(c[1] for c in cases)),
        "generated_at": int(time.time()),
    }
    if a.split in ("item", "both"):
        rep["item"] = run_item_split(cases, a.seed, a.train_frac, a.domain)
    if a.split in ("source", "both"):
        rep["source"] = run_source_split(cases, a.domain)

    if a.json:
        print(json.dumps(rep, indent=2))
    else:
        for line in _fmt(rep):
            print(line)

    # the verdict: held-out must beat the majority baseline, or the door has
    # learned nothing that transfers.
    rc = 0
    verdicts: List[str] = []
    it = rep.get("item")
    if it:
        beat = it["heldout"]["accuracy_pct"] > it["majority"]["accuracy_pct"]
        verdicts.append(
            f"item split: held-out {it['heldout']['accuracy_pct']}% vs majority "
            f"{it['majority']['accuracy_pct']}% -- {'BEATS' if beat else 'DOES NOT BEAT'}"
        )
        rc = max(rc, 0 if beat else 1)
    sp = rep.get("source")
    if sp and sp.get("overall"):
        o = sp["overall"]
        # judged on the STRICT number: the leaky one is the door recalling forks
        # it was taught under another tool's name, and scoring that as a pass is
        # the same cache hit this file exists to correct.
        st = o.get("strict")
        if st:
            beat = st["accuracy_pct"] > st["majority_pct"]
            verdicts.append(
                f"source split (strict, {st['items']} never-taught criteria): "
                f"{st['accuracy_pct']}% vs majority {st['majority_pct']}% -- "
                f"{'BEATS' if beat else 'DOES NOT BEAT'}"
            )
            rc = max(rc, 0 if beat else 1)
        else:
            verdicts.append(
                "source split: COULD NOT JUDGE -- no fold had >= 5 never-taught criteria"
            )
            rc = max(rc, 2)
    if not verdicts:
        print("\nVERDICT: no leg ran")
        return 2
    print("\nVERDICT: " + "; ".join(verdicts))
    return rc


if __name__ == "__main__":
    sys.exit(main())
