"""The typed-decision contract — Aither World Decide.

Three primitives, chosen to match what software actually asks of a model
millions of times a day, and to be answerable by ANY backend that can put a
probability on a bounded set:

  choice  pick one of N named options           -> value + per-option probabilities
  score   pick one of N ORDERED levels           -> value + per-level probabilities
  bool    yes / no                               -> probability of yes

Every answer carries `probability` (of the returned value), `confidence` (the
same number, kept as its own field so a caller thresholding on it reads the
intent), `backend` (which rung answered) and `decided`.

`decided=False` is a first-class answer. It means no rung produced a
probability at or above the question's `min_confidence`; `value` is None and
`reasons` say why. A consumer that acts on an undecided answer has chosen to,
in code, where a reviewer can see it. The contract never fabricates.

Nothing here does I/O. The ladder lives in `backends.py`, the outcomes ledger
in `ledger.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

KINDS = ("choice", "score", "bool")


@dataclass(frozen=True)
class Question:
    """One typed question about one state."""

    kind: str
    options: Sequence[str] = ()  # choice: unordered names; score: ORDERED levels, low -> high
    min_confidence: float = 0.0  # below this the answer is decided=False
    prompt: str = ""  # optional natural-language framing for model backends

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.kind == "bool":
            object.__setattr__(self, "options", ("yes", "no"))
        elif len(self.options) < 2:
            raise ValueError(f"{self.kind} needs at least 2 options, got {list(self.options)}")
        if len(set(self.options)) != len(self.options):
            raise ValueError("options must be unique")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")

    @classmethod
    def choice(cls, options: Sequence[str], **kw: Any) -> "Question":
        return cls("choice", tuple(options), **kw)

    @classmethod
    def score(cls, levels: Sequence[str], **kw: Any) -> "Question":
        return cls("score", tuple(levels), **kw)

    @classmethod
    def bool(cls, **kw: Any) -> "Question":  # noqa: A003 - the contract's own name
        return cls("bool", ("yes", "no"), **kw)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "options": list(self.options),
                "min_confidence": self.min_confidence, "prompt": self.prompt}


@dataclass
class Decision:
    """The answer to one Question about one state."""

    kind: str
    value: Optional[str]
    probability: float  # of `value`; 0.0 when undecided
    probabilities: Dict[str, float]  # over every option, sums to 1 when decided
    decided: bool
    backend: str
    reasons: List[str] = field(default_factory=list)
    id: str = ""  # set by the ledger when recorded

    @property
    def confidence(self) -> float:
        return self.probability

    @property
    def p_yes(self) -> Optional[float]:
        return self.probabilities.get("yes") if self.kind == "bool" else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "value": self.value,
            "probability": round(self.probability, 4), "confidence": round(self.probability, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "decided": self.decided, "backend": self.backend, "reasons": list(self.reasons),
        }


def undecided(q: Question, backend: str, reasons: List[str]) -> Decision:
    return Decision(kind=q.kind, value=None, probability=0.0,
                    probabilities={o: 0.0 for o in q.options}, decided=False,
                    backend=backend, reasons=reasons)


def normalize(q: Question, raw: Dict[str, float]) -> Dict[str, float]:
    """Clamp negatives, keep only the question's options, sum to 1.

    A backend that returns mass on an option the question did not name is
    reporting a different question; that mass is dropped and the caller can see
    it in the remainder. Returns {} when nothing usable remains.
    """
    kept = {o: max(0.0, float(raw.get(o, 0.0))) for o in q.options}
    total = sum(kept.values())
    if total <= 0.0:
        return {}
    return {o: v / total for o, v in kept.items()}


def from_probabilities(q: Question, raw: Dict[str, float], backend: str) -> Decision:
    """Build a Decision from a backend's raw probabilities, applying min_confidence."""
    probs = normalize(q, raw)
    if not probs:
        return undecided(q, backend, [f"{backend}: no probability mass on any option"])
    value = max(probs, key=probs.get)
    p = probs[value]
    if p < q.min_confidence:
        d = undecided(q, backend, [f"{backend}: top probability {p:.3f} < min_confidence "
                                   f"{q.min_confidence:.3f}"])
        d.probabilities = probs  # keep the evidence; the verdict is still "not decided"
        return d
    return Decision(kind=q.kind, value=value, probability=p, probabilities=probs,
                    decided=True, backend=backend)
