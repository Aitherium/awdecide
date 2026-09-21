"""Contract parity: awdecide -> DoorBackend and the door itself agree.

The same six questions asked through the awdecide Ladder (DoorBackend rung)
and directly through the door's in-process `decide.Decider` must give the same
value and the same calibrated probability -- and where the door says
source=none, awdecide must abstain (decided=False) instead of surfacing the
door's placeholder option.

Runs in-process against a TEMP ckpt dir: the door's service tree
(AITHER_WM_SVC_DIR, default D:/arc-agi-3/arc-world-model-svc) and its
world_model package must be importable; otherwise the module is skipped with
that reason (a skip is visible, a silent pass is not).
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

import pytest

from awdecide import DoorBackend, Ladder, Ledger, Question
from awdecide.door import DEFAULT_SVC_DIR, decision_from_door, door_request

_SVC = Path(os.getenv("AITHER_WM_SVC_DIR", DEFAULT_SVC_DIR))
_HAVE_TREE = (_SVC / "decide.py").is_file() and (_SVC / "code_domains.py").is_file()


def _engine_importable() -> bool:
    if not _HAVE_TREE:
        return False
    os.environ.setdefault("AITHER_WM_CKPT_DIR", tempfile.mkdtemp(prefix="awdecide-parity-"))
    if str(_SVC) not in sys.path:
        sys.path.insert(0, str(_SVC))
    try:
        return bool(getattr(importlib.import_module("code_domains"), "_MLP_OK", False))
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _engine_importable(),
    reason=f"door service tree / world_model package not importable from {_SVC}",
)

FORK = "parity.router"
DOMAIN = f"decide.{FORK}"
LANES = ["fast-local", "orchestrator", "reasoner"]


@pytest.fixture(scope="module")
def door_and_decider(tmp_path_factory):
    from awdecide.door import make_decider

    ckpt = tmp_path_factory.mktemp("ckpt")
    decider = make_decider(str(_SVC), str(ckpt), llm_enabled=False, embed_enabled=False)
    # teach the fork: outcomes the engine can answer from
    for _ in range(4):
        decider.outcome({"domain": DOMAIN, "state": "kind:code,len:short",
                         "answer": "orchestrator", "reward": 1.0})
    decider.outcome({"domain": DOMAIN, "state": "kind:code,len:short",
                     "answer": "fast-local", "reward": -1.0})
    for _ in range(3):
        decider.outcome({"domain": DOMAIN, "state": "kind:chat,len:short",
                         "answer": "fast-local", "reward": 1.0})
    decider.outcome({"domain": DOMAIN, "state": "kind:chat,len:short",
                     "answer": "fast-local", "reward": -1.0})
    # a yes/no fork and a scored fork
    for r in (1.0, 1.0, -1.0):
        decider.outcome({"domain": "decide.parity.build", "state": "pytest:42 passed",
                         "answer": "yes", "reward": r})
    for r in (1.0, 1.0):
        decider.outcome({"domain": "decide.parity.sev", "state": "traceback+timeout",
                         "answer": "high", "reward": r})
    decider.outcome({"domain": "decide.parity.sev", "state": "traceback+timeout",
                     "answer": "low", "reward": -1.0})
    return decider, ckpt


QUESTIONS = [
    # (fork, state, Question)
    (FORK, "kind:code,len:short", Question.choice(LANES)),                       # engine
    (FORK, "kind:chat,len:short", Question.choice(LANES)),                       # engine, 3/4
    (FORK, "kind:never-seen,len:long", Question.choice(LANES)),                  # prior
    ("parity.build", "pytest:42 passed", Question.bool()),                       # engine yes
    ("parity.sev", "traceback+timeout", Question.score(["low", "mid", "high"])),  # engine
    ("parity.cold", "nothing known here", Question.bool()),                      # none -> abstain
]


def test_parity_six_questions(door_and_decider):
    decider, _ = door_and_decider
    seen_sources = set()
    for fork, state, q in QUESTIONS:
        direct = decider.decide(door_request(state, q, f"decide.{fork}"))
        via = Ladder([DoorBackend(fork=fork, decider=decider)]).decide_one(state, q)
        seen_sources.add(direct["source"])
        if direct["source"] == "none":
            assert not via.decided and via.value is None and via.probability == 0.0
            assert any("door:none" in r for r in via.reasons)
            continue
        if direct.get("probability") is None:
            # measured 2026-09-20: the door's `prior` rung answers with probability=None
            # (its alternatives stay the engine rows, which have no values), so the
            # contract ABSTAINS -- an answer without a number is not a claim.
            assert not via.decided and via.value is None
            assert any("no probability" in r for r in via.reasons), via.reasons
            continue
        assert via.decided, (fork, state, direct, via.reasons)
        assert via.value == str(direct["answer"])
        assert via.probability == pytest.approx(float(direct["probability"]))
        assert via.backend == f"door:{direct['source']}"
        # the door's per-option calibrated numbers survive unrenormalized
        for opt, p in (direct.get("probabilities") or {}).items():
            if opt in via.probabilities and opt != via.value:
                assert via.probabilities[opt] == pytest.approx(p)
    # the six questions must have exercised evidence, the prior, and an abstention
    assert {"engine", "prior", "none"} <= seen_sources, seen_sources


def test_min_confidence_is_applied_on_the_probability(door_and_decider):
    decider, _ = door_and_decider
    q = Question.choice(LANES, min_confidence=0.99)
    d = Ladder([DoorBackend(fork=FORK, decider=decider)]).decide_one("kind:chat,len:short", q)
    assert not d.decided and d.probabilities["fast-local"] > 0.0
    assert any("< min_confidence" in r for r in d.reasons)


def test_resolve_teaches_the_door_and_the_ledger(door_and_decider, tmp_path):
    decider, ckpt = door_and_decider
    door = DoorBackend(fork=FORK, decider=decider)
    led = Ledger(tmp_path / "l.db")
    q = Question.choice(LANES)
    before = decider.decide(door_request("kind:code,len:short", q, DOMAIN))["learned_from"]
    d = Ladder([door]).decide_one("kind:code,len:short", q)
    did = led.record("route", "kind:code,len:short", d)
    assert did == d.id and len(did) == 16  # the door's own decision_id names the claim
    r = door.resolve(did, correct=True, ledger=led)
    assert r["door"]["ok"] is True and r["brier"] == pytest.approx((d.probability - 1) ** 2)
    after = decider.decide(door_request("kind:code,len:short", q, DOMAIN))["learned_from"]
    assert after == before + 1
    assert led.reliability()["resolved"] == 1
    led.close()
    # and the door journaled both halves under that id
    text = (ckpt / "decisions.jsonl").read_text(encoding="utf-8")
    assert text.count(did) >= 2


def test_decision_from_door_is_pure():
    q = Question.bool()
    d = decision_from_door(q, {"decision_id": "x", "answer": "no", "source": "engine",
                               "probability": 0.8, "probabilities": {"no": 0.8}})
    assert d.decided and d.value == "no" and d.probability == 0.8
    assert d.probabilities["yes"] == pytest.approx(0.2)
    d = decision_from_door(q, {"decision_id": "x", "answer": "yes", "source": "none",
                               "probability": None})
    assert not d.decided and d.backend == "door:none"
    d = decision_from_door(q, {"decision_id": "x", "answer": "maybe", "source": "llm",
                               "probability": 0.7})
    assert not d.decided and "not one of the options" in d.reasons[0]
