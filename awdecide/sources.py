"""Sources -- where a decision's STATE comes from, and a rung that predicts.

Two things live here, both stdlib-only, both duck-typed against the packages
they pair with (`awrepl`, `awpredict`) so awdecide keeps zero dependencies:

  ReplState(session)      a STABLE state descriptor built from a live awrepl
                          session's variables: names, types and BUCKETED sizes.
                          Never raw values -- the descriptor is a key you may
                          hash, log, and hand to a ledger; a customer's dataframe
                          contents are not.
  PredictBackend(env)     a Ladder rung that answers a Question by asking a
                          value oracle what happens next for EACH option and
                          ABSTAINING unless the margin between the top two
                          options clears a threshold. The oracle is anything
                          with `value(state, option) -> float | None`:
                            OutcomeLookup       the self-updating last-outcome
                                                dictionary (ships as DEFAULT)
                            AwpredictValueEnv   wraps an awpredict engine's
                                                reward/value head
  default_predict_backend()  PredictBackend over an OutcomeLookup.

Why the lookup is the default and not the world model, measured 2026-09-20 by
`tool_outcome_predict_bench` on 323,644 real Bash tool outcomes (95.0% pass),
temporal 80/20 split, scored on the UNSEEN bucket only -- the novel command
shapes, because a self-updating dictionary already owns the seen rows and an
aggregate cannot move:

    UNSEEN accuracy      always-run  majority  lookup-exact  lookup-family
    37,668 rows              0.9571    0.9571        0.9571         0.9291
    1,785 rows (12k window)  0.9434    0.9434        0.9434         0.9098
      + awpredict token-hash 0.9434 (ties the base rate)
      + awpredict whole-hash 0.9412 (loses to it)

Nothing beats the base rate on a shape it has never seen -- there is nothing in
the history to learn from -- so the bench exits 1 and that IS the finding. The
number that shapes this module is skip-precision, not accuracy: 0.9434 on UNSEEN
means about one skipped call in eighteen would really have failed, which is why
PredictBackend ABSTAINS unless every option has a value and the margin clears.
A learned rung becomes the default the day it beats the dictionary, not before.
"""
from __future__ import annotations

import math
import re
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
)

from .backends import Backend
from .contract import Question

Weights = Dict[str, float]

# ---------------------------------------------------------------------------
# ReplState
# ---------------------------------------------------------------------------
_TYPE_REPR = re.compile(r"^([A-Za-z_][\w.]*):\s?(.*)$", re.S)
# awrepl's worker returns {name: "type: repr"}; repr is cut at 100 chars with "...".
_EMPTY_REPRS = {"[]", "{}", "()", "''", '""', "set()", "None", "b''", 'b""', "0", "0.0", "False"}
_OPAQUE_TYPES = {"module", "function", "builtin_function_or_method", "type", "method", "class"}


def size_bucket(n: Optional[int]) -> str:
    """Bucket a length so the descriptor is stable under small edits:
    0 · 1 · 2-9 · 10-99 · 100-999 · 1k-9k · 10k+ · ? (unknown)."""
    if n is None:
        return "?"
    if n <= 1:
        return str(n)
    if n < 10:
        return "2-9"
    if n < 100:
        return "10-99"
    if n < 1000:
        return "100-999"
    if n < 10_000:
        return "1k-9k"
    return "10k+"


def _bucket_from_repr(rep: str) -> str:
    """When inspect() is unavailable, the repr's LENGTH is the only size hint
    the worker gives -- bucketed, never quoted."""
    rep = (rep or "").strip()
    if rep in _EMPTY_REPRS:
        return "0"
    if rep.endswith("...") or len(rep) >= 100:
        return "long"
    return "short"


def describe_variables(variables: Mapping[str, str],
                       inspect: Optional[Callable[[str], Mapping[str, Any]]] = None, *,
                       include_private: bool = False, max_vars: int = 64) -> str:
    """Deterministic descriptor of an awrepl namespace.

    `variables` is what `ReplSession.variables()` returns ({name: "type: repr"});
    `inspect`, when given, is `ReplSession.inspect` and supplies a real `len`.
    Output: `name:type:size` entries, sorted by name, comma-joined, capped at
    `max_vars` (the overflow is counted, not silently dropped). Contains NO
    repr text and NO value -- names, types and size buckets only.
    """
    entries: List[str] = []
    names = sorted(n for n in variables if include_private or not n.startswith("_"))
    for name in names[:max_vars]:
        raw = str(variables.get(name, ""))
        m = _TYPE_REPR.match(raw)
        typ, rep = (m.group(1), m.group(2)) if m else (raw.split(":", 1)[0] or "?", "")
        if typ in _OPAQUE_TYPES:
            entries.append(f"{name}:{typ}")
            continue
        bucket: Optional[str] = None
        if inspect is not None:
            try:
                info = inspect(name) or {}
                if "len" in info and isinstance(info["len"], int):
                    bucket = size_bucket(info["len"])
            except Exception:
                bucket = None  # the worker died mid-call: fall back, never raise
        if bucket is None:
            bucket = _bucket_from_repr(rep)
        entries.append(f"{name}:{typ}:{bucket}")
    if len(names) > max_vars:
        entries.append(f"+{len(names) - max_vars} more")
    return ",".join(entries) if entries else "<empty namespace>"


class ReplState:
    """A decision state read from a live awrepl session (or anything with
    `.variables()` and, optionally, `.inspect(name)`).

        state = ReplState(session)              # reads once, on construction
        answers = ladder.decide(str(state), questions)
        state.refresh()                          # after the next execute()
    """

    def __init__(self, session: Any, *, include_private: bool = False, max_vars: int = 64,
                 use_inspect: bool = True, prefix: str = "repl") -> None:
        self.session = session
        self.include_private = include_private
        self.max_vars = max_vars
        self.use_inspect = use_inspect
        self.prefix = prefix
        self.descriptor = ""
        self.refresh()

    def refresh(self) -> str:
        variables = self.session.variables() or {}
        inspect = getattr(self.session, "inspect", None) if self.use_inspect else None
        body = describe_variables(variables, inspect if callable(inspect) else None,
                                  include_private=self.include_private, max_vars=self.max_vars)
        sid = getattr(self.session, "session_id", "")
        head = f"{self.prefix}" + (f"[{sid}]" if sid else "")
        self.descriptor = f"{head} vars={body}"
        return self.descriptor

    def __str__(self) -> str:
        return self.descriptor


# ---------------------------------------------------------------------------
# Value oracles
# ---------------------------------------------------------------------------
class OutcomeLookup:
    """The self-updating last-outcome dictionary -- the floor every learned
    predictor must beat (WMF doctrine), and the shipped default.

    `value(state, option)` is the last recorded reward for (key(state), option),
    or None when never seen -- the PredictBackend then abstains. `observe()`
    records an outcome; `key` lets a caller coarsen the state (e.g. the command
    family) so the dictionary is not blind on every novel exact state.
    """

    def __init__(self, key: Optional[Callable[[str], str]] = None) -> None:
        self.key = key or (lambda s: s)
        self._last: Dict[Tuple[str, str], float] = {}
        self._count: Dict[Tuple[str, str], int] = {}

    def value(self, state: str, option: str) -> Optional[float]:
        return self._last.get((self.key(state), option))

    def observe(self, state: str, option: str, reward: float) -> None:
        k = (self.key(state), option)
        self._last[k] = float(reward)
        self._count[k] = self._count.get(k, 0) + 1

    def seen(self, state: str, option: str) -> int:
        return self._count.get((self.key(state), option), 0)

    def __len__(self) -> int:
        return len(self._last)


class AwpredictValueEnv:
    """Adapts an awpredict engine to `value(state, option)`.

    Duck-typed on purpose (awdecide imports nothing from awpredict):
      * an engine with `value(obs) -> float|None` (LeWorldModel's value head)
        is asked about the state with the option folded into the observation
        via `fold(state, option)`;
      * an engine with `predict(state_hash, action) -> (next, reward, done)|None`
        (MLPWorldModel) is asked with action = option and the reward is the value.
    `hash_state` maps the descriptor to the engine's state key (default: the
    stable 64-bit sha256 prefix the bench uses). A degraded engine (returns
    None) is a None value, so the rung abstains; nothing is fabricated.
    """

    def __init__(self, engine: Any, *, hash_state: Optional[Callable[[str], Any]] = None,
                 fold: Optional[Callable[[str, str], Any]] = None) -> None:
        self.engine = engine
        self.hash_state = hash_state or _stable_hash
        self.fold = fold or (lambda s, o: f"{s} option:{o}")

    def value(self, state: str, option: str) -> Optional[float]:
        eng = self.engine
        try:
            if getattr(eng, "ok", True) is False:
                return None
            has_predict = callable(getattr(eng, "predict", None))
            has_value = callable(getattr(eng, "value", None))
            if has_predict and not has_value:
                out = eng.predict(self.hash_state(state), option)
                if out is None:
                    return None
                return float(out[1])
            if has_value:
                v = eng.value(self.fold(state, option))
                return None if v is None else float(v)
        except Exception:
            return None
        return None


def _stable_hash(s: str) -> int:
    import hashlib
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


# ---------------------------------------------------------------------------
# PredictBackend
# ---------------------------------------------------------------------------
class PredictBackend(Backend):
    """A rung that answers from predicted per-option value, abstaining on a
    thin margin.

    For a Question with options o_1..o_n it asks `env.value(state, o_i)`. If any
    option has no value (None) the rung abstains -- an oracle that has never
    seen an option cannot rank it. Otherwise it sorts the values; when
    top - second < `margin` it abstains. When it answers, the weights are a
    softmax over value / `temperature`, so `Decision.probability` grows with the
    margin and `min_confidence` still applies downstream.

    `margin` is in the oracle's value units (rewards of +1/-1 make 0.5 mean
    "the leader must be at least a coin-flip's worth ahead").
    """

    def __init__(self, env: Any, *, margin: float = 0.5, temperature: float = 1.0,
                 name: str = "predict") -> None:
        if not callable(getattr(env, "value", None)):
            raise TypeError("PredictBackend env must expose value(state, option) -> float | None")
        if margin < 0:
            raise ValueError("margin must be >= 0")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.env = env
        self.margin = margin
        self.temperature = temperature
        self.name = name
        self.last_values: Dict[str, Optional[float]] = {}

    def values(self, state: str, q: Question) -> Dict[str, Optional[float]]:
        vals: Dict[str, Optional[float]] = {}
        for o in q.options:
            v = self.env.value(state, o)
            vals[o] = None if v is None or not math.isfinite(float(v)) else float(v)
        self.last_values = vals
        return vals

    def weights(self, state: str, q: Question) -> Weights:
        vals = self.values(state, q)
        if any(v is None for v in vals.values()):
            return {}
        ordered = sorted(vals.values(), reverse=True)  # type: ignore[arg-type]
        if len(ordered) >= 2 and (ordered[0] - ordered[1]) < self.margin:
            return {}
        top = ordered[0]
        # every value is a float here -- the None check is directly above.
        return {o: math.exp((v - top) / self.temperature)  # type: ignore[operator]
                for o, v in vals.items()}

    def observe(self, state: str, option: str, reward: float) -> None:
        """Feed an outcome back to an oracle that learns (OutcomeLookup does;
        an awpredict engine is trained by its own observe/train loop)."""
        fn = getattr(self.env, "observe", None)
        if callable(fn):
            fn(state, option, reward)


def default_predict_backend(*, margin: float = 0.5,
                            key: Optional[Callable[[str], str]] = None) -> PredictBackend:
    """The shipped default: the self-updating lookup behind the margin gate.
    Swap the env for an AwpredictValueEnv the day the bench says it beats this."""
    return PredictBackend(OutcomeLookup(key), margin=margin, name="predict-lookup")


__all__ = [
    "ReplState", "describe_variables", "size_bucket",
    "OutcomeLookup", "AwpredictValueEnv", "PredictBackend", "default_predict_backend",
]
