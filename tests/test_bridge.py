"""bridge tests -- the door's journals land in the ledger once, labelled, graded.

The journal arm is hermetic (a hand-written decisions.jsonl). The replay and
bench arms need the door's service tree + world_model package and are skipped
with that reason otherwise.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

from awdecide import Ledger, bridge
from awdecide.door import DEFAULT_SVC_DIR

_SVC = Path(os.getenv("AITHER_WM_SVC_DIR", DEFAULT_SVC_DIR))


def _engine_importable() -> bool:
    if not ((_SVC / "decide.py").is_file() and (_SVC / "code_domains.py").is_file()):
        return False
    os.environ.setdefault("AITHER_WM_CKPT_DIR", tempfile.mkdtemp(prefix="awdecide-bridge-"))
    if str(_SVC) not in sys.path:
        sys.path.insert(0, str(_SVC))
    try:
        return bool(getattr(importlib.import_module("code_domains"), "_MLP_OK", False))
    except Exception:  # noqa: BLE001
        return False


needs_engine = pytest.mark.skipif(not _engine_importable(),
                                  reason=f"door service tree not importable from {_SVC}")


def _journal(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_journal_ingest_is_labelled_graded_and_idempotent(tmp_path):
    ck = tmp_path / "ckpt"
    ck.mkdir()
    _journal(ck / "decisions.jsonl", [
        {"kind": "decision", "decision_id": "a1", "domain": "decide.x", "state": "s1",
         "answer": "yes", "probability": 0.9, "question_kind": "yesno", "source": "engine",
         "ts": 1.0},
        {"kind": "outcome", "decision_id": "a1", "reward": 1.0, "ts": 2.0},
        {"kind": "decision", "decision_id": "a2", "domain": "decide.x", "state": "s2",
         "answer": "b", "probability": 0.3, "question_kind": "choice", "source": "neighbor",
         "ts": 3.0},
        {"kind": "outcome", "decision_id": "a2", "reward": -1.0, "ts": 4.0},
        {"kind": "decision", "decision_id": "a3", "domain": "decide.x", "state": "s3",
         "answer": "b", "confidence": 0.7, "source": "engine", "ts": 5.0},  # legacy
        {"kind": "outcome", "decision_id": "a3", "reward": 1.0, "ts": 6.0},
        {"kind": "decision", "decision_id": "a4", "domain": "decide.x", "state": "s4",
         "answer": "b", "probability": 2.5, "source": "engine", "ts": 7.0},  # malformed p
        {"kind": "outcome", "decision_id": "a4", "reward": 1.0, "ts": 8.0},
    ])
    led = Ledger(tmp_path / "l.db")
    r = bridge.ingest_door(ck, led, label="unit", replay=False)
    j = r["journaled"]
    assert (j["matched"], j["ingested"], j["legacy_confidence_only"],
            j["malformed"]) == (4, 2, 1, 1)
    assert r["replayed"] == {"skipped": "--no-replay"}
    r2 = bridge.ingest_door(ck, led, label="unit", replay=False)
    assert r2["journaled"]["ingested"] == 0 and r2["journaled"]["already_present"] == 2
    rep = led.reliability()
    assert rep["resolved"] == 2
    assert rep["by_backend"]["door:engine"] == {"n": 1, "brier": pytest.approx(0.01),
                                                 "base_rate": 1.0}
    assert rep["by_backend"]["door:neighbor"]["brier"] == pytest.approx(0.09)
    rows = led._con.execute("SELECT id, key, kind FROM decisions ORDER BY id").fetchall()
    assert rows == [("a1", "unit/decide.x", "bool"), ("a2", "unit/decide.x", "choice")]
    led.close()


def test_missing_journal_is_reported_not_invented(tmp_path):
    led = Ledger(tmp_path / "l.db")
    r = bridge.ingest_journal(tmp_path / "nowhere", led)
    assert r["present"] is False and r["ingested"] == 0
    led.close()


def test_ledger_ingest_refuses_bad_probability(tmp_path):
    led = Ledger(tmp_path / "l.db")
    with pytest.raises(ValueError):
        led.ingest("z", key="k", kind="bool", value="yes", probability=1.5, backend="door:engine")
    assert led.ingest("z", key="k", kind="bool", value="yes", probability=0.5,
                      backend="door:engine", correct=True)
    assert not led.ingest("z", key="k", kind="bool", value="no", probability=0.1,
                          backend="door:engine", correct=False)  # first write wins
    assert led.resolved() == [(0.5, 1)]
    led.close()


@needs_engine
def test_replay_is_prequential_and_idempotent(tmp_path):
    """A 3/4 fork replayed in order: the first outcome per (state, action) is cold
    (no claim), later ones carry the engine's P(reward>0) before it observed them."""
    ck = tmp_path / "ckpt"
    ck.mkdir()
    rows = [{"obs": "s", "action": "a", "next_obs": "s", "reward": r, "done": False}
            for r in (1.0, 1.0, -1.0, 1.0, 1.0, 1.0)]
    _journal(ck / "domain-decide.unit.replay.transitions.jsonl", rows)
    led = Ledger(tmp_path / "l.db")
    r = bridge.ingest_replay(ck, led, label="unit", svc_dir=str(_SVC))
    assert r["journals"] == 1 and r["rows"] == 6
    assert r["cold"] >= 1 and r["predicted"] == 6 - r["cold"] and r["ingested"] == r["predicted"]
    r2 = bridge.ingest_replay(ck, led, label="unit", svc_dir=str(_SVC))
    assert r2["ingested"] == 0 and r2["already_present"] == r["predicted"]
    rep = led.reliability()
    assert rep["resolved"] == r["predicted"]
    assert set(rep["by_backend"]) == {"door:engine:replay"}
    # the probabilities are the engine's, in [0, 1], and not all the same after the -1
    ps = sorted(p for p, _ in led.resolved())
    assert 0.0 <= ps[0] <= ps[-1] <= 1.0 and ps[0] < ps[-1]
    led.close()


@needs_engine
def test_generate_bench_yields_graded_pairs_from_the_real_door(tmp_path):
    scratch = tmp_path / "bench"
    rep = bridge.generate_bench(scratch, svc_dir=str(_SVC), decisions=60)
    assert rep["decide_bench"]["decisions"] == 60
    assert (scratch / "decisions.jsonl").is_file()
    led = Ledger(tmp_path / "l.db")
    r = bridge.ingest_door(scratch, led, label="bench", replay=False)
    assert r["journaled"]["ingested"] >= 60 and r["journaled"]["legacy_confidence_only"] == 0
    rel = led.reliability()
    assert rel["resolved"] >= 60 and "door:engine" in rel["by_backend"]
    keys = led._con.execute("SELECT DISTINCT key FROM decisions").fetchall()
    assert all(k.startswith("bench/") for (k,) in keys)
    led.close()
