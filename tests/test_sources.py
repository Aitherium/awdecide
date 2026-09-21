"""awdecide.sources tests -- ReplState never leaks a value; PredictBackend abstains
on a thin margin and answers on a wide one; the lookup is the shipped default."""
from __future__ import annotations

import pytest
from awdecide import Ladder, Question
from awdecide.sources import (
    AwpredictValueEnv,
    OutcomeLookup,
    PredictBackend,
    ReplState,
    default_predict_backend,
    describe_variables,
    size_bucket,
)

SECRET = "sk-live-THIS-MUST-NOT-APPEAR"


class StubSession:
    """Mimics awrepl.ReplSession: variables() -> {name: 'type: repr'}, inspect() -> len."""

    session_id = "stub-1"

    def __init__(self) -> None:
        self.calls = 0
        self._vars = {
            "token": f"str: '{SECRET}'",
            "rows": "list: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]",
            "cfg": "dict: {}",
            "n": "int: 42",
            "os": "module: <module 'os' (frozen)>",
            "_hidden": "str: 'x'",
        }
        self._len = {"token": len(SECRET), "rows": 12, "cfg": 0}

    def variables(self):
        self.calls += 1
        return dict(self._vars)

    def inspect(self, name):
        info = {"name": name, "type": self._vars[name].split(":")[0]}
        if name in self._len:
            info["len"] = self._len[name]
        return info


def test_size_bucket_edges():
    got = [size_bucket(x) for x in (None, 0, 1, 2, 9, 10, 99, 100, 999, 1000, 9999, 10000)]
    assert got == ["?", "0", "1", "2-9", "2-9", "10-99", "10-99", "100-999", "100-999",
                   "1k-9k", "1k-9k", "10k+"]


def test_repl_state_never_carries_values():
    s = StubSession()
    st = ReplState(s)
    d = str(st)
    assert SECRET not in d and "42" not in d and "[1, 2" not in d
    assert "token:str:10-99" in d and "rows:list:10-99" in d and "cfg:dict:0" in d
    assert "n:int:short" in d  # no __len__: repr length bucket, never the repr
    assert "os:module" in d and "_hidden" not in d and d.startswith("repl[stub-1] vars=")
    # deterministic and order-free
    assert describe_variables(dict(reversed(list(s.variables().items())))) == \
        describe_variables(s.variables())


def test_repl_state_without_inspect_and_with_cap():
    class Bare:
        def variables(self):
            return {f"v{i}": "int: 1" for i in range(70)}
    d = str(ReplState(Bare(), max_vars=64))
    assert "+6 more" in d and "v0:int:short" in d
    empty = ReplState(type("E", (), {"variables": lambda self: {}})())
    assert str(empty).endswith("<empty namespace>")


def test_repl_state_refresh_rereads():
    s = StubSession()
    st = ReplState(s)
    s._vars["new"] = "set: set()"
    assert "new" not in str(st)
    st.refresh()
    assert "new:set:0" in str(st) and s.calls == 2


class StubEnv:
    def __init__(self, table):
        self.table = table

    def value(self, state, option):
        return self.table.get((state, option))


def test_predict_backend_abstains_on_unknown_option_and_thin_margin():
    q = Question.choice(["run", "skip"])
    b = PredictBackend(StubEnv({("s", "run"): 1.0}), margin=0.5)
    assert b.weights("s", q) == {}  # "skip" never valued -> abstain
    thin = PredictBackend(StubEnv({("s", "run"): 0.6, ("s", "skip"): 0.4}), margin=0.5)
    assert thin.weights("s", q) == {}
    d = Ladder([thin]).decide_one("s", q)
    assert not d.decided and any("abstained" in r for r in d.reasons)


def test_predict_backend_answers_on_wide_margin_with_margin_ordered_probability():
    q = Question.choice(["run", "skip", "ask"])
    wide = PredictBackend(StubEnv({("s", "run"): 1.0, ("s", "skip"): -1.0, ("s", "ask"): -1.0}),
                          margin=0.5)
    d = Ladder([wide]).decide_one("s", q)
    assert d.decided and d.value == "run" and d.backend == "predict"
    wider = PredictBackend(StubEnv({("s", "run"): 3.0, ("s", "skip"): -3.0, ("s", "ask"): -3.0}))
    d2 = Ladder([wider]).decide_one("s", q)
    assert d2.probability > d.probability > 1 / 3
    assert wide.last_values == {"run": 1.0, "skip": -1.0, "ask": -1.0}


def test_predict_backend_rejects_bad_env_and_params():
    with pytest.raises(TypeError):
        PredictBackend(object())
    with pytest.raises(ValueError):
        PredictBackend(StubEnv({}), margin=-1)
    with pytest.raises(ValueError):
        PredictBackend(StubEnv({}), temperature=0)
    nan_env = StubEnv({("s", "yes"): float("nan"), ("s", "no"): 0.0})
    assert PredictBackend(nan_env).weights("s", Question.bool()) == {}


def test_lookup_default_learns_from_outcomes():
    b = default_predict_backend(margin=0.5)
    q = Question.bool()
    assert b.name == "predict-lookup" and b.weights("git status", q) == {}
    b.observe("git status", "yes", 1.0)
    b.observe("git status", "no", -1.0)
    d = Ladder([b]).decide_one("git status", q)
    assert d.decided and d.value == "yes" and d.probabilities["yes"] > 0.8
    b.observe("git status", "yes", -1.0)  # both at -1.0 now: zero margin -> abstain
    assert not Ladder([b]).decide_one("git status", q).decided
    b.observe("git status", "no", 1.0)  # last outcome flips the answer
    assert Ladder([b]).decide_one("git status", q).value == "no"
    assert isinstance(b.env, OutcomeLookup) and len(b.env) == 2
    assert b.env.seen("git status", "yes") == 2


def test_lookup_key_coarsens_state():
    lk = OutcomeLookup(key=lambda s: s.split()[0])
    lk.observe("git status", "yes", 1.0)
    assert lk.value("git log", "yes") == 1.0 and lk.value("podman ps", "yes") is None


def test_awpredict_env_duck_types_both_engine_shapes():
    class MLPLike:  # predict(state_hash, action) -> (next, reward, done) | None
        ok = True

        def predict(self, state_hash, action):
            return (0, 0.75, False) if action == "run" else None

    class LeWMLike:  # value(obs) -> float | None
        ok = True

        def value(self, obs):
            return 0.25 if obs.endswith("option:run") else -0.25

    class Degraded:
        ok = False

        def predict(self, *a):
            raise AssertionError("must not be called when ok is False")

    assert AwpredictValueEnv(MLPLike()).value("s", "run") == 0.75
    assert AwpredictValueEnv(MLPLike()).value("s", "skip") is None
    assert AwpredictValueEnv(LeWMLike()).value("s", "run") == 0.25
    assert AwpredictValueEnv(LeWMLike()).value("s", "skip") == -0.25
    assert AwpredictValueEnv(Degraded()).value("s", "run") is None
    q = Question.choice(["run", "skip"])
    narrow = PredictBackend(AwpredictValueEnv(LeWMLike()), margin=0.4)
    broad = PredictBackend(AwpredictValueEnv(LeWMLike()), margin=0.6)
    assert Ladder([narrow]).decide_one("s", q).value == "run"
    assert not Ladder([broad]).decide_one("s", q).decided
