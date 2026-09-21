"""The Brier ledger — every decision is a claim that can be resolved.

A probability nobody resolves is a decoration. The ledger records each decided
answer (question, value, probability, backend), lets the caller RESOLVE it
against what actually happened, and reports reliability the way a gate can
read it: overall Brier, the climatology (base-rate) Brier it must beat, and a
per-confidence-bucket table (mean confidence vs observed frequency).

SQLite file, stdlib only, one table. The math is the standard one -- Brier
against the base-rate Brier, plus per-bucket reliability -- so any calibration
gate can read this file and go red on it.

Resolution semantics: `outcome` is whether the RETURNED VALUE was right
(1 / 0). For a bool question, resolving with the truth of the statement is
the same thing after mapping through the value: a "no" answered at 0.8 that
turns out false is correct. `resolve(id, correct=...)` takes the bool and does
nothing clever.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .contract import Decision

DEFAULT_DB = Path(os.getenv("AWDECIDE_DB", str(Path.home() / ".aither" / "awdecide.db")))

BUCKETS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    key TEXT NOT NULL,
    kind TEXT NOT NULL,
    value TEXT,
    probability REAL NOT NULL,
    backend TEXT NOT NULL,
    state_sha TEXT NOT NULL,
    outcome INTEGER,
    brier REAL,
    resolved_at REAL
);
CREATE INDEX IF NOT EXISTS idx_decisions_resolved ON decisions(resolved_at);
"""


def _sha(state: str) -> str:
    import hashlib
    return hashlib.sha256(state.encode("utf-8", "replace")).hexdigest()[:16]


class Ledger:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else DEFAULT_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(self.path)
        self._con.executescript(_SCHEMA)

    def close(self) -> None:
        self._con.close()

    # ------------------------------------------------------------- write
    def record(self, key: str, state: str, d: Decision) -> str:
        """Record a DECIDED answer. Undecided answers are not claims; they are not recorded."""
        if not d.decided:
            return ""
        d.id = d.id or uuid.uuid4().hex[:12]
        self._con.execute(
            "INSERT INTO decisions (id, ts, key, kind, value, probability, backend, state_sha) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (d.id, time.time(), key, d.kind, d.value, float(d.probability), d.backend, _sha(state)))
        self._con.commit()
        return d.id

    def resolve(self, decision_id: str, correct: bool) -> Optional[float]:
        """Resolve a recorded decision; returns its Brier score, None if unknown id."""
        row = self._con.execute("SELECT probability FROM decisions WHERE id=?",
                                (decision_id,)).fetchone()
        if row is None:
            return None
        o = 1 if correct else 0
        brier = (float(row[0]) - o) ** 2
        self._con.execute("UPDATE decisions SET outcome=?, brier=?, resolved_at=? WHERE id=?",
                          (o, brier, time.time(), decision_id))
        self._con.commit()
        return brier

    def ingest(self, decision_id: str, *, key: str, kind: str, value: Optional[str],
               probability: float, backend: str, state_sha: str = "", ts: Optional[float] = None,
               correct: Optional[bool] = None, resolved_at: Optional[float] = None) -> bool:
        """Land a decision made ELSEWHERE (the world-model door's journals, via
        awdecide/bridge.py) under its own id, already resolved when `correct` is
        given. Idempotent: an id already present is left untouched and returns
        False. The probability must be in [0, 1]; anything else raises, because a
        row the gate would count as malformed must not enter by the back door."""
        p = float(probability)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probability {p!r} outside [0, 1]")
        if self.has(decision_id):
            return False
        o = None if correct is None else (1 if correct else 0)
        brier = None if o is None else (p - o) ** 2
        r_at = None if o is None else float(resolved_at if resolved_at is not None else time.time())
        self._con.execute(
            "INSERT OR IGNORE INTO decisions (id, ts, key, kind, value, probability, backend, "
            "state_sha, outcome, brier, resolved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (decision_id, float(ts if ts is not None else time.time()), key, kind, value, p,
             backend, state_sha, o, brier, r_at))
        self._con.commit()
        return True

    def has(self, decision_id: str) -> bool:
        return self._con.execute("SELECT 1 FROM decisions WHERE id=?",
                                 (decision_id,)).fetchone() is not None

    # -------------------------------------------------------------- read
    def resolved(self) -> List[Tuple[float, int]]:
        return [(float(p), int(o)) for p, o in self._con.execute(
            "SELECT probability, outcome FROM decisions WHERE outcome IS NOT NULL")]

    def pending(self) -> int:
        return int(self._con.execute(
            "SELECT COUNT(*) FROM decisions WHERE outcome IS NULL").fetchone()[0])

    def reliability(self) -> Dict[str, Any]:
        rows = self.resolved()
        out: Dict[str, Any] = {"db": str(self.path), "resolved": len(rows),
                               "pending": self.pending()}
        if not rows:
            out["verdict"] = "no resolved decisions -- nothing to judge"
            return out
        base = sum(o for _, o in rows) / len(rows)
        brier = sum((p - o) ** 2 for p, o in rows) / len(rows)
        clim = sum((base - o) ** 2 for _, o in rows) / len(rows)
        buckets = []
        for lo, hi in BUCKETS:
            b = [(p, o) for p, o in rows if lo <= p < hi]
            if not b:
                buckets.append({"range": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": 0})
                continue
            mc = sum(p for p, _ in b) / len(b)
            fr = sum(o for _, o in b) / len(b)
            buckets.append({"range": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": len(b),
                            "mean_conf": round(mc, 3), "observed": round(fr, 3),
                            "gap": round(abs(mc - fr), 3)})
        by_backend: Dict[str, Dict[str, Any]] = {}
        for backend, p, o in self._con.execute(
                "SELECT backend, probability, outcome FROM decisions WHERE outcome IS NOT NULL"):
            s = by_backend.setdefault(backend, {"n": 0, "sq": 0.0, "hits": 0})
            s["n"] += 1
            s["sq"] += (float(p) - int(o)) ** 2
            s["hits"] += int(o)
        out.update({"brier": round(brier, 4), "climatology": round(clim, 4),
                    "beats_base_rate": brier < clim, "buckets": buckets,
                    "by_backend": {b: {"n": s["n"], "brier": round(s["sq"] / s["n"], 4),
                                       "base_rate": round(s["hits"] / s["n"], 3)}
                                   for b, s in sorted(by_backend.items())}})
        out["verdict"] = ("calibrated information" if brier < clim
                          else "NO information beyond the base rate")
        return out

    def export_platform_jsonl(self, path: Path) -> int:
        """Write resolved rows in the platform `predictions.jsonl` shape
        (status correct/incorrect + confidence) so any reader of that store sees them."""
        n = 0
        with Path(path).open("w", encoding="utf-8") as f:
            for d_id, p, o, key in self._con.execute(
                    "SELECT id, probability, outcome, key FROM decisions "
                    "WHERE outcome IS NOT NULL"):
                f.write(json.dumps({"id": d_id, "claim": key, "confidence": float(p),
                                    "status": "correct" if o else "incorrect"}) + "\n")
                n += 1
        return n
