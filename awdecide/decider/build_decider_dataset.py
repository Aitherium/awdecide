#!/usr/bin/env python3
"""build_decider_dataset -- unify every journaled decision into typed decider rows.

One row per decision:
    {context, kind, task, label, label_space, outcome, source, ts, split, n}
    kind   yesno | choice | route | reflex
    task   the head it trains: yesno.door, choice.door, yesno.route_outcome,
           choice.agent, choice.intent, route.brain, reflex.nerve
    route  labels are ``react`` | ``think:<brain>`` (brains: see awdecide.decider.BRAINS)

Sources (each optional; a missing one is reported, not fatal):
  door      the decide door's journals (decide.* forks) and labelled judge cases --
            the exact rows the door's neural rung trains on (read from a COPY of the
            checkpoint dir, so the live dir is never written)
  library   <library>/Training/...:
              routing_decisions/decisions.jsonl   choice.agent   (oracle chain head)
              routing_decisions/outcomes.jsonl    outcome join by task_hash, if present
              intent_map/routing_records.jsonl    yesno.route_outcome
              intent_classification/sessions.jsonl choice.intent (label_type intent_type)
              tool_decisions/decisions-*.jsonl    route.brain
  darklink  world-journal *.jsonl under --darklink-dir: ``nerve_label`` steps -> reflex.nerve

Dedupe is by (task, context): the majority label wins (latest on a tie), so one
context can never sit on both sides of the split. The split is TIME-ordered per task:
the oldest 85 % of the training share is ``train``, the next 15 % ``val`` (model
selection + temperature), the newest ``--test-frac`` is ``test``.

    python -m awdecide.decider.build_decider_dataset --library <dir> --ckpt <dir> \\
        --out <dir>/rows.jsonl

Exit 0 rows written; 2 nothing to write, or --out committable in a git work tree (training
data and weights never go in a repo).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from awdecide.decider import BRAINS, ROUTE_SPACE
from awdecide.decider.features import context_from_feats

NERVE_ROUTES = ("none", "reflex", "think")


def _jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Rows of a jsonl file; a torn line (a writer mid-append) is skipped and counted."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                _BAD[str(path)] += 1
                continue
            if isinstance(r, dict):
                yield r


_BAD: Counter = Counter()


def _row(context: str, kind: str, task: str, label: str, space: Iterable[str],
         source: str, ts: float, outcome: Any = None, **extra: Any) -> Dict[str, Any]:
    r = {"context": context, "kind": kind, "task": task, "label": label,
         "label_space": sorted(set(space)), "outcome": outcome, "source": source,
         "ts": float(ts)}
    r.update(extra)
    return r


# ------------------------------------------------------------------- door --


def door_rows(ckpt: Path, cases: Optional[Path], svc_dir: Optional[str]) -> List[Dict[str, Any]]:
    """The door's own training rows, re-shaped one-per-decision. Reads a temp COPY of
    the journals so building a dataset can never write into a live checkpoint dir."""
    from awdecide import door as door_mod

    tmp = Path(tempfile.mkdtemp(prefix="decider-ckpt-"))
    mtimes: Dict[str, float] = {}
    for p in sorted(Path(ckpt).glob("domain-decide.*.transitions.jsonl")):
        shutil.copy2(p, tmp / p.name)
        mtimes[p.name[len("domain-"):-len(".transitions.jsonl")]] = p.stat().st_mtime
    try:
        code_domains, decide = door_mod.load_service(svc_dir, str(tmp))
        decider = decide.Decider(code_domains.DomainEngines(), llm=None, embed_enabled=False)
        raw = decide.neural_rows_from_journals(decider)
        cpath = cases if cases is not None else Path(ckpt) / "neural-rung.cases.jsonl"
        case_rows = decide.neural_rows_from_cases(cpath) if cpath.is_file() else []
        base = cpath.stat().st_mtime if cpath.is_file() else 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # cases carry no timestamp: file order is the clock (the harvest appends)
    for i, r in enumerate(case_rows):
        r["_ts"] = base - (len(case_rows) - i) * 1e-3
    for r in raw:
        r["_ts"] = mtimes.get(r["domain"], 0.0)
    by_fork: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    space: Dict[str, set] = defaultdict(set)
    for r in raw + case_rows:
        by_fork[r["fork"]].append(r)
        space[r["domain"]].add(r["option"])
    out: List[Dict[str, Any]] = []
    for fork, rs in by_fork.items():
        r0 = rs[0]
        agg: Dict[str, List[float]] = defaultdict(list)
        for r in rs:
            agg[r["option"]].extend([r["label"]] * max(1, int(r.get("n", 1))))
        mean = {o: sum(v) / len(v) for o, v in agg.items()}
        kind = r0["kind"]
        if kind == "yesno":
            if "yes" in mean:
                py = mean["yes"]
            elif "no" in mean:
                py = 1.0 - mean["no"]
            else:
                continue
            if py == 0.5:
                continue
            label, labels = ("yes" if py > 0.5 else "no"), ["yes", "no"]
        elif kind == "choice":
            best = max(mean, key=lambda o: mean[o])
            if mean[best] <= 0.5:      # nothing was RIGHT here; no label to learn
                continue
            label, labels = best, sorted(space[r0["domain"]])
        else:
            continue                   # score rows have no categorical label
        ctx = decide.neural_context(r0["domain"], r0["state"], r0["question"])
        out.append(_row(ctx, kind, f"{kind}.door", label, labels,
                        "door." + r0["source"], max(r["_ts"] for r in rs),
                        outcome=round(max(mean.values()), 4),
                        door={"domain": r0["domain"], "state": r0["state"],
                              "question": r0["question"], "fork": fork}))
    return out


# ---------------------------------------------------------------- library --


def brain_of(model: str) -> Optional[str]:
    """The brain a model string names, or None (a fast path that is not a brain)."""
    m = str(model or "").lower()
    if "bonsai" in m:
        return "bonsai2"
    if "deepseek" in m:
        return "deepseek-pool"
    if "gemma" in m:
        return "gemma4-vision"
    if "orchestrator" in m:
        return "orchestrator"
    return None


def route_label(rec: Dict[str, Any]) -> Optional[str]:
    """``react`` | ``think:<brain>`` for one tool-decision record, None = no label.

    react  = answered by the fast lane with no reasoning and no agentic loop.
    think  = reasoning or an agentic loop ran, or the turn went to a bigger brain."""
    model = str(rec.get("model_used") or "")
    if model in ("error", "") or "pipeline" in model:
        return None
    brain = brain_of(model)
    deep = bool(rec.get("reasoning_used") or rec.get("agentic_used"))
    if brain in (None, "orchestrator") and not deep:
        if brain is None and "fastpath" not in model:
            return None                # an unknown lane: do not guess
        return "react"
    return f"think:{brain or 'orchestrator'}"


def library_rows(lib: Path) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    tr = Path(lib) / "Training"
    rows: List[Dict[str, Any]] = []
    missing: Dict[str, str] = {}

    p = tr / "routing_decisions" / "decisions.jsonl"
    outcomes: Dict[str, Any] = {}
    po = tr / "routing_decisions" / "outcomes.jsonl"
    if po.is_file():
        for r in _jsonl(po):
            if r.get("task_hash"):
                outcomes[str(r["task_hash"])] = r.get("success", r.get("outcome"))
    else:
        missing["routing_decisions/outcomes.jsonl"] = "absent (RouterOracle never paired one)"
    if p.is_file():
        recs = [r for r in _jsonl(p) if r.get("oracle_chain") and r.get("task_summary")]
        space = {str(r["oracle_chain"][0]).lower() for r in recs}
        for r in recs:
            ctx = f"kind=choice task=agent effort={r.get('effort')} || {r['task_summary']}"
            rows.append(_row(ctx, "choice", "choice.agent", str(r["oracle_chain"][0]).lower(),
                             space, "routing_decisions", r.get("ts") or 0,
                             outcome=outcomes.get(str(r.get("task_hash")))))
    else:
        missing["routing_decisions/decisions.jsonl"] = "absent"

    p = tr / "intent_map" / "routing_records.jsonl"
    if p.is_file():
        for r in _jsonl(p):
            if r.get("outcome") not in ("success", "failure"):
                continue
            tools = ",".join(sorted(str(t) for t in (r.get("tools_used") or [])))[:200]
            ctx = (f"kind=yesno task=route_outcome intent={r.get('intent_type')} "
                   f"effort={r.get('effort')} agent={r.get('agent') or '-'} "
                   f"via={r.get('route_source') or '-'} domain={r.get('domain') or '-'} "
                   f"|| tools {tools or 'none'}")
            rows.append(_row(ctx, "yesno", "yesno.route_outcome",
                             "yes" if r["outcome"] == "success" else "no", ["yes", "no"],
                             "intent_map", r.get("ts") or 0, outcome=r["outcome"]))
    else:
        missing["intent_map/routing_records.jsonl"] = "absent"

    p = tr / "intent_classification" / "sessions.jsonl"
    if p.is_file():
        # tool_category rows are MULTI-label (one query, many rows): not a choice
        recs = [r for r in _jsonl(p) if r.get("label_type") == "intent_type" and r.get("query")]
        space = {str(r["label"]) for r in recs}
        for r in recs:
            rows.append(_row(f"kind=choice task=intent || {r['query']}", "choice",
                             "choice.intent", str(r["label"]), space, "intent_classification",
                             r.get("ts") or 0, outcome="confirmed" if r.get("confirmed") else None))
    else:
        missing["intent_classification/sessions.jsonl"] = "absent"

    files = sorted(glob.glob(str(tr / "tool_decisions" / "decisions-*.jsonl")))
    if not files:
        missing["tool_decisions/decisions-*.jsonl"] = "absent"
    for f in files:
        for r in _jsonl(Path(f)):
            lab = route_label(r)
            if lab is None or not r.get("message"):
                continue
            ctx = (f"kind=route intent={r.get('intent_type')} effort={r.get('intent_effort')} "
                   f"cats={','.join(sorted(map(str, r.get('detected_categories') or [])))[:120]} "
                   f"|| {r['message']}")
            rows.append(_row(ctx, "route", "route.brain", lab, ROUTE_SPACE, "tool_decisions",
                             r.get("ts") or 0, outcome="success" if r.get("success") else "failure"))
    return rows, missing


# --------------------------------------------------------------- darklink --


def darklink_rows(root: Optional[Path]) -> List[Dict[str, Any]]:
    """``nerve_label`` link.step rows -> reflex.nerve. The journal shape is the one
    darklink.dd2_agents.Journal writes (payload.meta.link_kind / payload.meta.nerve)."""
    if root is None or not Path(root).exists():
        return []
    rows: List[Dict[str, Any]] = []
    for f in sorted(Path(root).rglob("*.jsonl")):
        for r in _jsonl(f):
            pay = r.get("payload") if isinstance(r.get("payload"), dict) else r
            meta = pay.get("meta") if isinstance(pay, dict) else None
            if not isinstance(meta, dict) or meta.get("link_kind") != "nerve_label":
                continue
            nerve = meta.get("nerve") or {}
            lab, feats = nerve.get("label"), nerve.get("feats")
            if lab not in NERVE_ROUTES or not isinstance(feats, dict):
                continue
            rows.append(_row(context_from_feats(str(nerve.get("stimulus") or "fight"), feats),
                             "reflex", "reflex.nerve", lab, NERVE_ROUTES, "darklink.nerve",
                             r.get("ts") or r.get("t") or f.stat().st_mtime,
                             outcome=nerve.get("brain")))
    return rows


# ------------------------------------------------------------ dedupe/split --


def dedupe(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per (task, context): majority label (latest on a tie), latest ts, n."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["context"])].append(r)
    out = []
    for rs in groups.values():
        rs.sort(key=lambda r: r["ts"])
        c = Counter(r["label"] for r in rs)
        top = max(c.values())
        winner = [r for r in rs if c[r["label"]] == top][-1]
        row = dict(winner)
        row["n"] = len(rs)
        row["label_agreement"] = round(top / len(rs), 4)
        row["ts"] = rs[-1]["ts"]
        out.append(row)
    return out


def time_split(rows: List[Dict[str, Any]], test_frac: float = 0.2, val_frac: float = 0.15) -> None:
    """In place, per task, oldest -> newest: train | val | test."""
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    for rs in by.values():
        rs.sort(key=lambda r: (r["ts"], r["context"]))
        n = len(rs)
        n_test = int(round(n * test_frac))
        n_trainval = n - n_test
        n_val = int(round(n_trainval * val_frac))
        for i, r in enumerate(rs):
            r["split"] = ("train" if i < n_trainval - n_val
                          else ("val" if i < n_trainval else "test"))


def committable(path: Path) -> bool:
    """True when ``path`` sits in a git work tree and is NOT ignored there -- i.e. a
    ``git add`` would pick it up. Data and weights must never be committable. If git
    itself cannot answer, inside a work tree counts as committable (fail closed)."""
    path = path.resolve()
    root = next((p for p in [path.parent, *path.parent.parents] if (p / ".git").exists()), None)
    if root is None:
        return False
    try:
        r = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", str(path)],
                           capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return True
    return r.returncode != 0


def report(raw: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"raw_rows": len(raw), "deduped_rows": len(rows),
                           "raw_by_source": dict(Counter(r["source"] for r in raw)),
                           "by_task": {}}
    for task in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["task"] == task]
        out["by_task"][task] = {
            "kind": rs[0]["kind"], "rows": len(rs),
            "splits": dict(Counter(r["split"] for r in rs)),
            "sources": dict(Counter(r["source"] for r in rs)),
            "labels": dict(Counter(r["label"] for r in rs).most_common(12)),
        }
    out["by_kind"] = dict(Counter(r["kind"] for r in rows))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--library", default=os.environ.get("AITHER_LIBRARY"),
                    help="Library root (contains Training/); env AITHER_LIBRARY")
    ap.add_argument("--ckpt", default=os.environ.get("AITHER_WM_CKPT_DIR"),
                    help="door checkpoint dir (decide journals); env AITHER_WM_CKPT_DIR")
    ap.add_argument("--cases", default=None, help="judge cases jsonl (default <ckpt>/neural-rung.cases.jsonl)")
    ap.add_argument("--svc-dir", default=None, help="door service tree (default: awdecide.door's)")
    ap.add_argument("--darklink-dir", default=os.environ.get("AITHER_DARKLINK_WORLD_DIR"))
    ap.add_argument("--out", required=True, help="rows.jsonl (outside any git work tree)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.15)
    a = ap.parse_args(argv)

    out = Path(a.out)
    if committable(out):
        print(f"refusing: {out} is inside a git work tree and not ignored -- data never goes in a repo")
        return 2
    raw: List[Dict[str, Any]] = []
    notes: Dict[str, str] = {}
    if a.ckpt:
        try:
            raw += door_rows(Path(a.ckpt), Path(a.cases) if a.cases else None, a.svc_dir)
        except Exception as exc:  # noqa: BLE001 -- one source down is a note, not a crash
            notes["door"] = f"{type(exc).__name__}: {exc}"[:300]
    else:
        notes["door"] = "no --ckpt"
    if a.library:
        lr, miss = library_rows(Path(a.library))
        raw += lr
        notes.update(miss)
    else:
        notes["library"] = "no --library"
    dl = darklink_rows(Path(a.darklink_dir) if a.darklink_dir else None)
    raw += dl
    if not dl:
        notes["darklink"] = "0 nerve_label rows" + ("" if a.darklink_dir else " (no --darklink-dir)")
    rows = dedupe(raw)
    time_split(rows, a.test_frac, a.val_frac)
    if not rows:
        print(json.dumps({"rows": 0, "notes": notes}, indent=2))
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["task"], r["ts"]))
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = report(raw, rows)
    rep["notes"] = notes
    rep["torn_lines_skipped"] = dict(_BAD)
    rep["brains"] = list(BRAINS)
    out.with_suffix(".report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
