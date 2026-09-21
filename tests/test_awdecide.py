"""awdecide tests -- the contract, the ladder, the ledger, the CLI grammar."""
from __future__ import annotations

import pytest

from awdecide import (
    CallableBackend,
    Ladder,
    Ledger,
    Question,
    RulesBackend,
    parse_question_spec,
)
from awdecide._selftest import run_self_test


def test_self_test_passes():
    assert run_self_test() == 0


def test_question_validation():
    with pytest.raises(ValueError):
        Question.choice(["only-one"])
    with pytest.raises(ValueError):
        Question.choice(["a", "a"])
    with pytest.raises(ValueError):
        Question("maybe", ("a", "b"))
    assert Question.bool().options == ("yes", "no")


def test_rules_never_guess():
    q = Question.choice(["billing", "sales"])
    d = Ladder([RulesBackend([(r"refund", "billing")])]).decide_one("hello", q)
    assert not d.decided and d.value is None and d.probability == 0.0


def test_mass_on_unknown_option_is_dropped():
    q = Question.choice(["a", "b"])
    d = Ladder([CallableBackend(lambda s, qq: {"a": 0.5, "zzz": 5.0}, "odd")]).decide_one("s", q)
    assert d.decided and d.value == "a" and d.probability == 1.0


def test_min_confidence_keeps_evidence():
    q = Question.choice(["a", "b"], min_confidence=0.9)
    d = Ladder([CallableBackend(lambda s, qq: {"a": 0.6, "b": 0.4}, "w")]).decide_one("s", q)
    assert not d.decided and abs(d.probabilities["a"] - 0.6) < 1e-9 and d.backend == "w"


def test_ledger_ignores_undecided(tmp_path):
    led = Ledger(tmp_path / "l.db")
    q = Question.bool()
    assert led.record("k", "s", Ladder([]).decide_one("s", q)) == ""
    d = Ladder([CallableBackend(lambda s, qq: {"yes": 0.8, "no": 0.2}, "c")]).decide_one("s", q)
    did = led.record("k", "s", d)
    assert did and led.pending() == 1
    assert led.resolve(did, correct=True) == pytest.approx(0.04)
    assert led.resolve("nope", correct=True) is None
    led.close()


def test_grammar():
    key, q = parse_question_spec("level:score=low,mid,high@0.5")
    assert key == "level" and q.kind == "score" and q.min_confidence == 0.5
    with pytest.raises(ValueError):
        parse_question_spec("level:score")
