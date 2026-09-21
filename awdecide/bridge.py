"""bridge -- the door's outcomes into awdecide's Brier ledger.

The world-model decision door journals every claim and every outcome
(`decisions.jsonl`, decide.py `_record` / `outcome`) and every per-fork
transition (`domain-decide.<fork>.transitions.jsonl`, code_domains.observe).
This module reads those and lands them in the awdecide Ledger, so a calibration
check judges the door's real numbers instead of an empty store.

Three sources, each labelled in the row's `backend` so nothing is conflated:

  journaled   decision rows carrying `probability` (written since 2026-09-20)
              joined to the FIRST outcome row with the same decision_id.
              backend = door:<source>, id = the door's decision_id.
              Rows without `probability` (pre-2026-09-20, confidence only) are
              COUNTED and skipped: confidence is evidence strength, not P(right).
  replayed    each per-fork transitions journal replayed IN ORDER through a fresh
              engine in a scratch dir (prequential): the P(reward>0 | state,
              action) the engine would have stated before each outcome it then
              observed. backend = door:engine:replay, id = replay:<sha> of
              (domain, row index, obs, action). Needs the service tree; skipped
              with a count when it is not importable.
  bench       --generate: the door's own benches (tools/decide_bench.py
              run_learning, tools/judge_bench.py warm passes, no LLM) run
              IN-PROCESS against a TEMP ckpt dir, then ingested as `journaled`
              with the key prefix `bench/`. Real code, simulated world.

Idempotent by decision id: re-running ingests nothing twice. Every row's key is
`<label>/<domain>`; `awdecide reliability` breaks the table out by backend.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .door import awdecide_kind, load_service, resolve_pairs_from_door
from .ledger import Ledger

logger = logging.getLogger("awdecide.bridge")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
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
                    continue  # torn tail line from a crash mid-append: counted by the caller
    except OSError as exc:
        # A journal we cannot read is not an empty journal. Say so; the caller's
        # counts then show 0 pairs next to a named reason instead of a clean zero.
        logger.warning("awdecide.bridge: cannot read %s (%s) -- 0 rows from it", path, exc)
    return rows


def _blank() -> Dict[str, Any]:
    return {"decision_rows": 0, "outcome_rows": 0, "matched": 0, "legacy_confidence_only": 0,
            "source_none": 0, "ingested": 0, "already_present": 0, "malformed": 0}


# ------------------------------------------------------------ journaled
def ingest_journal(ckpt_dir: Path, ledger: Ledger, *, label: str = "live") -> Dict[str, Any]:
    """decisions.jsonl -> ledger. Returns counts."""
    info = _blank()
    path = Path(ckpt_dir) / "decisions.jsonl"
    info["path"] = str(path)
    info["present"] = path.is_file()
    if not path.is_file():
        return info
    rows = read_jsonl(path)
    info["decision_rows"] = sum(1 for r in rows if r.get("kind") == "decision")
    info["outcome_rows"] = sum(1 for r in rows if r.get("kind") == "outcome")
    for pair in resolve_pairs_from_door(rows):
        info["matched"] += 1
        source = str(pair.get("source") or "none")
        if source == "none":
            info["source_none"] += 1
            continue
        if "probability" not in pair:
            info["legacy_confidence_only"] += 1
            continue
        p = pair.get("probability")
        if p is None:
            info["source_none"] += 1
            continue
        try:
            landed = ledger.ingest(
                str(pair["decision_id"]),
                key=f"{label}/{pair.get('domain') or 'unknown'}",
                kind=awdecide_kind(pair.get("question_kind") or "choice"),
                value=None if pair.get("answer") is None else str(pair.get("answer")),
                probability=float(p),
                backend=f"door:{source}",
                state_sha=hashlib.sha256(str(pair.get("state") or "").encode(
                    "utf-8", "replace")).hexdigest()[:16],
                ts=pair.get("ts"),
                correct=bool(pair["correct"]),
            )
        except (TypeError, ValueError):
            info["malformed"] += 1
            continue
        info["ingested" if landed else "already_present"] += 1
    return info


# ------------------------------------------------------------- replayed
def ingest_replay(ckpt_dir: Path, ledger: Ledger, *, label: str = "live",
                  svc_dir: Optional[str] = None, max_rows: int = 5000) -> Dict[str, Any]:
    """Prequential replay of every decide.* transitions journal -> ledger."""
    info: Dict[str, Any] = {"journals": 0, "rows": 0, "predicted": 0, "cold": 0, "ingested": 0,
                            "already_present": 0, "engine": None, "per_domain": {}}
    ckpt_dir = Path(ckpt_dir)
    try:
        journals = sorted(ckpt_dir.glob("domain-decide.*.transitions.jsonl"))
    except OSError:
        journals = []
    info["journals"] = len(journals)
    if not journals:
        return info
    scratch = tempfile.mkdtemp(prefix="awdecide-replay-")
    try:
        code_domains, _decide = load_service(svc_dir, scratch)
    except Exception as e:  # noqa: BLE001 -- no engine here: say so, ingest nothing
        info["engine"] = f"unavailable ({type(e).__name__}: {e})"
        return info
    info["engine"] = f"code_domains.DomainEngines (scratch {scratch})"
    for path in journals:
        domain = path.name[len("domain-"): -len(".transitions.jsonl")]
        rows = read_jsonl(path)
        offset = 0
        if len(rows) > max_rows:
            offset = len(rows) - max_rows
            rows = rows[-max_rows:]
        engines = code_domains.DomainEngines()
        got = 0
        for i, r in enumerate(rows):
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
            value = None
            if pred is not None:
                try:
                    value = float(pred[1])
                except (TypeError, ValueError, IndexError):
                    value = None
            if value is not None and math.isfinite(value):
                p = max(0.0, min(1.0, (value + 1.0) / 2.0))
                did = "replay:" + hashlib.sha256(
                    f"{domain}\n{offset + i}\n{obs}\n{action}".encode("utf-8", "replace")
                ).hexdigest()[:16]
                landed = ledger.ingest(
                    did, key=f"{label}/{domain}",
                    kind="bool" if str(action) in ("yes", "no") else "choice",
                    value=str(action), probability=p, backend="door:engine:replay",
                    state_sha=hashlib.sha256(obs.encode("utf-8", "replace")).hexdigest()[:16],
                    correct=reward > 0.0)
                info["predicted"] += 1
                info["ingested" if landed else "already_present"] += 1
                got += 1
            else:
                info["cold"] += 1
            try:
                engines.observe(domain, obs, str(action), next_obs, reward,
                                bool(r.get("done", False)))
            except Exception as exc:  # noqa: BLE001
                info["per_domain"][domain] = {"error": f"{type(exc).__name__}: {exc}"}
                break
        info["per_domain"].setdefault(domain,
                                      {"rows": len(rows), "pairs": got})
    return info


# ---------------------------------------------------------------- bench
def generate_bench(scratch: Path, *, svc_dir: Optional[str] = None, decisions: int = 400,
                   p_right: float = 0.70, seed: int = 3) -> Dict[str, Any]:
    """Run the door's benches in-process against `scratch` (a TEMP ckpt dir) so
    its decisions.jsonl holds fresh (probability, outcome) pairs from the code
    the service runs. No LLM, no embedder, no network."""
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    code_domains, decide = load_service(svc_dir, str(scratch))
    svc = Path(svc_dir or os.getenv("AITHER_WM_SVC_DIR", "D:/arc-agi-3/arc-world-model-svc"))
    import importlib
    import sys
    tools = str(svc / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    decide_bench = importlib.import_module("decide_bench")
    t0 = time.perf_counter()
    shapes = decide_bench.request_shapes()
    rng = random.Random(seed)
    seq = [rng.choice(shapes) for _ in range(decisions)]
    brain = decide_bench.SimBrain(p_right, 0.0)  # latency is irrelevant to calibration
    brain.truth = dict(shapes)
    d = decide.Decider(code_domains.DomainEngines(), llm=brain, record_dir=scratch,
                       embed_enabled=False)
    learn = decide_bench.run_learning(d, brain, seq)
    out: Dict[str, Any] = {
        "ckpt_dir": str(scratch),
        "decide_bench": {"decisions": decisions, "accuracy": round(learn["accuracy"], 3),
                         "llm_calls": learn["llm_calls"], "engine_pct": learn["engine_pct"]},
    }
    # judge bench, warm passes only (cold = source none = no claim = no pair)
    try:
        judge_mod = importlib.import_module("judge")
        judge_bench = importlib.import_module("judge_bench")
        dj = decide.Decider(code_domains.DomainEngines(), llm_enabled=False, record_dir=scratch,
                            embed_enabled=False)
        j = judge_mod.Judge(dj)
        dom = "decide.bench.judge"
        for _name, output, crits in judge_bench.CASES:
            for crit, truth in crits:
                j.teach(output, crit, truth, domain=dom)
        corrected = 0
        for _pass in range(2):
            for _name, output, crits in judge_bench.CASES:
                res = j.judge(output, [c for c, _ in crits], domain=dom)
                for (_c, truth), v in zip(crits, res.get("verdicts") or []):
                    if v.get("decision_id") and v.get("pass") is not None:
                        j.correct(v["decision_id"], v["pass"] is truth)
                        corrected += 1
        out["judge_bench"] = {"cases": len(judge_bench.CASES), "corrected": corrected}
    except Exception as e:  # noqa: BLE001 -- the judge leg is a bonus; the router leg is the pairs
        out["judge_bench"] = {"skipped": f"{type(e).__name__}: {e}"}
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    return out


# ------------------------------------------------------------------ all
def ingest_door(ckpt_dir: Path, ledger: Ledger, *, label: str = "live", replay: bool = True,
                svc_dir: Optional[str] = None) -> Dict[str, Any]:
    """decisions.jsonl (+ replayed transitions) of one ckpt dir -> ledger."""
    rep: Dict[str, Any] = {"ckpt_dir": str(ckpt_dir), "label": label,
                           "journaled": ingest_journal(ckpt_dir, ledger, label=label)}
    if replay:
        rep["replayed"] = ingest_replay(ckpt_dir, ledger, label=label, svc_dir=svc_dir)
    else:
        rep["replayed"] = {"skipped": "--no-replay"}
    rep["ledger"] = {"path": str(ledger.path), "resolved": len(ledger.resolved()),
                     "pending": ledger.pending()}
    return rep
