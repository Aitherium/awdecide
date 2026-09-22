"""World-model contracts: the two protocols every engine and environment obeys.

These are runtime-checkable Protocols, written to match what the shipped
engines ACTUALLY expose today (world_model.core.lewm.LeWorldModel — the
canonical copy of the live ARC service's model — and
world_model.core.mlp.MLPWorldModel, extracted from
lib/cognitive/LearnedWorldModel.py). They are the seam the rest of the program
plugs into: adapters map an environment into (observation, action) space, an
engine learns the dynamics, and every consumer (arc solver, code-world
ranking, adk sandbox bootstrap, AitherEvolution scheduling) talks to this
surface only.

Rules of the contract:
  * Degrade loudly, never silently: an engine that cannot operate (torch
    missing, checkpoint unreadable) exposes ``ok == False`` and returns
    None/[] from methods — it must never raise into a caller's turn loop, and
    it must never fabricate a prediction.
  * ``surprise`` is the universal signal: prediction error in latent space,
    normalized so consumers can threshold it. It is the fitness signal for
    AitherEvolution, the VoE gate for the adk learn-safely loop, and the
    anomaly feed for the belief graph.
  * Checkpoint promotion decisions NEVER come from an engine's own buffer —
    only from a held-out gate (wm_latent_gate --recordings law).
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable


@runtime_checkable
class WorldModel(Protocol):
    """A learned latent dynamics model: encode → predict → plan, with surprise.

    ``obs`` is whatever the paired EnvironmentAdapter's ``observe`` returns —
    an ARC grid (list of lists / ndarray), an embedding vector, etc. Engines
    document which observation family they accept; the adapter guarantees it.
    """

    ok: bool

    def observe(self, obs: Any, action: Any, next_obs: Any, *args: Any,
                **kwargs: Any) -> Any:
        """Buffer one transition for training. Returns engine-specific status."""

    def encode(self, obs: Any, cond: Any = None) -> Any:
        """Observation → latent z, or None when degraded."""

    def predict(self, z: Any, action: Any, **kwargs: Any) -> Any:
        """Latent + action → predicted next latent, or None when degraded."""

    def surprise(self, obs: Any, action: Any, next_obs: Any, *args: Any,
                 **kwargs: Any) -> Optional[float]:
        """Prediction error for the observed transition; None when degraded."""

    def train_step(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """One (or a few) optimization steps over the buffer; loss dict or None."""

    def plan(self, *args: Any, **kwargs: Any) -> Any:
        """Search action space in latent imagination toward a goal."""

    def save(self, path: str, *args: Any, **kwargs: Any) -> bool:
        """Checkpoint to disk; True on success (atomic where the engine supports it)."""

    def load(self, path: str, *args: Any, **kwargs: Any) -> Any:
        """Load a checkpoint; engine-specific status. Must not raise on missing file."""


@runtime_checkable
class EnvironmentAdapter(Protocol):
    """Maps one environment family into a WorldModel's observation/action space.

    An adapter is the ONLY thing that knows a domain's shape. Enrolling an
    agent in a new environment means writing (or selecting) an adapter —
    nothing in an engine changes.
    """

    #: short domain tag carried on transitions (e.g. "arc", "code", "sandbox")
    domain: str

    def observe(self, env_state: Any) -> Any:
        """Convert raw environment state into the engine's observation format."""

    def actions(self) -> Sequence[Any]:
        """The discrete action vocabulary for this environment (or a sample of it)."""

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Execute an action for real: (next_env_state, reward, done, info).

        Exploration safety is the adapter's duty: a sandbox adapter executes
        inside the sandbox; a code adapter performs read-only probes unless
        explicitly configured otherwise.
        """


def _protocol_attrs(proto: type) -> set:
    """The member names a Protocol requires — across CPython versions.

    ``__protocol_attrs__`` is a CPython 3.12+ internal. On 3.10/3.11 it does
    not exist, and the original ``getattr(proto, "__protocol_attrs__", set())``
    quietly yielded an EMPTY set there — so ``conforms()`` returned "nothing
    missing" for literally any object, including one with no members at all.
    A conformance check that always passes is worse than none: it is an
    assertion that reads as satisfied.

    CI runs 3.10, so that is where it was live. It surfaced only as the parity
    self-test's `contract:False` leg, which is the self-test doing its job —
    it caught a helper that could no longer fail.

    Raises rather than returning an empty set when the members cannot be
    determined at all: a caller cannot tell "conforms" from "could not look",
    and defaulting to the reassuring one is how the bug happened.
    """
    attrs = getattr(proto, "__protocol_attrs__", None)
    if attrs:
        return set(attrs)
    try:  # 3.8-3.11
        from typing import _get_protocol_attrs  # type: ignore[attr-defined]

        attrs = set(_get_protocol_attrs(proto))
    except (ImportError, AttributeError, TypeError):
        # The internal is gone or changed shape. Not swallowed: this is one
        # named strategy of three, and the LAST one refuses loudly rather than
        # returning an empty set. Recording it keeps the reason visible if the
        # derived fallback later produces a surprising member list.
        attrs = None
    if attrs:
        return attrs
    # Last resort: derive from the class body, minus everything object and
    # Protocol contribute. Keeps working if the internal is renamed again.
    ignore = set(dir(object)) | {
        "__abstractmethods__", "__annotations__", "__dict__", "__doc__",
        "__init__", "__module__", "__parameters__", "__protocol_attrs__",
        "__slots__", "__subclasshook__", "__weakref__", "_is_protocol",
        "_is_runtime_protocol",
    }
    derived = {n for n in dir(proto) if n not in ignore}
    derived |= set(getattr(proto, "__annotations__", {}))
    if not derived:
        raise TypeError(
            f"cannot determine the required members of {proto!r} on "
            f"Python {sys.version_info.major}.{sys.version_info.minor}; "
            f"refusing to report conformance rather than pass by default"
        )
    return derived


def conforms(obj: Any, proto: type) -> List[str]:
    """Return the members of ``proto`` that ``obj`` is missing (empty == conforms).

    Protocol ``isinstance`` checks only see attribute presence; this helper
    names what is absent so a failing conformance check says WHY.
    """
    missing = [name for name in _protocol_attrs(proto) if not hasattr(obj, name)]
    return sorted(missing)
