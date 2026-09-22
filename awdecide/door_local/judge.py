"""judge -- grade unstructured output against criteria, as a bounded decision.

The use case people currently burn a frontier model on: "did the agent actually
do X, Y and Z?" over a transcript, a log, a diff. Reported cost of that with an
LLM judge: ~34 s and ~$0.08 per task. It is the same shape as every other fork
the decision door answers -- yes/no with a reason -- with one difference that
matters more than the model: the raw text NEVER repeats, so a judge keyed on the
text can never learn. Key it on FEATURES and the second identical-shaped
judgement is served from evidence in microseconds.

    Judge(decider).judge(output, criteria, domain="decide.judge.eval")
      -> {verdicts: [{criterion, pass, probability, source, decision_id}], ...}

The state handed to the door is `crit:<slug>|<features>` where features are a
small, stable description of the output: exit status seen, whether a traceback
or an error line is present, pass/fail counts, length bucket, whether the
criterion's own keywords appear. A judgement the door has seen the shape of is
answered by the engine; a new shape falls to the local brain, is recorded, and
the correction (`/judge/outcome`, i.e. a human or a stronger model disagreeing)
teaches it.

Nothing here is model-specific: `decider` is the same Decider the service runs,
so the engine -> neighbour -> llm -> prior ladder, the calibrated probabilities
and the outcome loop all apply unchanged.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

_TRACEBACK = re.compile(r"traceback \(most recent call last\)|^\s+at .+\(.+:\d+\)", re.I | re.M)
_ERROR = re.compile(r"\b(error|exception|failed|failure|fatal|denied|refused)\b", re.I)
_PASSFAIL = re.compile(r"(\d+)\s+(passed|failed|errors?|warnings?)", re.I)
_EXIT = re.compile(r"exit(?:ed with)?\s*(?:code)?\s*[=: ]\s*(-?\d+)", re.I)
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
_SLUG = re.compile(r"[^a-z0-9]+")


def slug(text: str, limit: int = 48) -> str:
    return _SLUG.sub("-", (text or "").strip().lower()).strip("-")[:limit] or "criterion"


def _count_bucket(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    return "2-9" if n < 10 else "10+"


def _bucket(n: int) -> str:
    for edge in (200, 1000, 5000, 20000):
        if n < edge:
            return f"<{edge}"
    return ">=20000"


def features(output: str, criterion: str = "") -> str:
    """A STABLE descriptor of this output. Two runs of the same kind of task
    produce the same string; the raw text never would, and a state that never
    repeats is a state nothing can be learned about."""
    text = output or ""
    counts: Dict[str, int] = {}
    for num, kind in _PASSFAIL.findall(text):
        k = kind.lower().rstrip("s")
        counts[k] = counts.get(k, 0) + int(num)
    exits = {int(m) for m in _EXIT.findall(text)}
    crit_words = {w.lower() for w in _WORD.findall(criterion or "")}
    body_words = {w.lower() for w in _WORD.findall(text)}
    overlap = len(crit_words & body_words)
    parts = [
        f"len:{_bucket(len(text))}",
        f"tb:{int(bool(_TRACEBACK.search(text)))}",
        f"err:{int(bool(_ERROR.search(text)))}",
        # BUCKETED, not exact: "42 passed" and "39 passed" are the same situation,
        # and keying on the number meant a green suite never matched the green
        # suite before it -- the judge would relearn every run and answer nothing
        # from evidence (caught by test_judge_features_are_stable...).
        f"pass:{_count_bucket(counts.get('passed', 0))}",
        f"fail:{_count_bucket(counts.get('failed', 0) + counts.get('error', 0))}",
        f"exit:{'none' if not exits else ('0' if exits == {0} else 'nonzero')}",
        f"hit:{min(overlap, 9)}/{min(len(crit_words), 9)}",
    ]
    return "|".join(parts)


class Judge:
    """Structured grading on top of the decision door."""

    def __init__(self, decider: Any, default_domain: str = "decide.judge") -> None:
        self._d = decider
        self._domain = default_domain

    def _items(self, output: str, criteria: Sequence[str], domain: str) -> List[Dict[str, Any]]:
        return [
            {
                "domain": f"{domain}.{slug(c)}",
                "state": f"crit:{slug(c)}|{features(output, c)}",
                "kind": "yesno",
                "question": (
                    f"Criterion: {c}\n\n"
                    "Judging this output, is the criterion satisfied? Answer yes or no.\n\n"
                    f"--- output (first 4000 chars) ---\n{(output or '')[:4000]}"
                ),
            }
            for c in criteria
        ]

    def judge(
        self,
        output: str,
        criteria: Sequence[str],
        domain: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> Dict[str, Any]:
        domain = domain or self._domain
        items = self._items(output, criteria, domain)
        for it in items:
            it["min_confidence"] = float(min_confidence)
        answers = (self._d.decide_batch(items) or {}).get("answers") or []
        verdicts: List[Dict[str, Any]] = []
        for crit, ans in zip(criteria, answers):
            ans = ans or {}
            # A judge that cannot judge must NOT read as a pass. With no evidence
            # and no model the door answers with the first option ("yes"), so a
            # failing build would have been graded PASSED in silence -- the worst
            # failure this thing can have. source=none => pass is None, unknown.
            unknown = ans.get("source") in (None, "none")
            verdicts.append(
                {
                    "criterion": crit,
                    "pass": None if unknown else ans.get("answer") == "yes",
                    "unknown": unknown,
                    "probability": ans.get("p_yes"),
                    "confidence": ans.get("confidence", 0.0),
                    "source": ans.get("source", "none"),
                    "learned_from": ans.get("learned_from", 0),
                    "decision_id": ans.get("decision_id"),
                }
            )
        served = [v["source"] for v in verdicts]
        return {
            "verdicts": verdicts,
            "passed": sum(1 for v in verdicts if v["pass"] is True),
            "failed": sum(1 for v in verdicts if v["pass"] is False),
            "unknown": sum(1 for v in verdicts if v["pass"] is None),
            "of": len(verdicts),
            "from_evidence": sum(1 for s in served if s in ("engine", "neighbor")),
            "from_model": sum(1 for s in served if s == "llm"),
            "state": features(output, criteria[0] if criteria else ""),
        }

    def correct(self, decision_id: str, verdict_was_right: bool) -> Dict[str, Any]:
        """Teach: a human (or a stronger judge) agreed or disagreed. This is the
        loop an LLM judge does not have -- it is wrong the same way forever."""
        return self._d.outcome(
            {"decision_id": str(decision_id), "reward": 1.0 if verdict_was_right else -1.0}
        )

    def teach(
        self, output: str, criterion: str, should_pass: bool, domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """Teach without a prior judgement: this output, this criterion, this truth."""
        domain = domain or self._domain
        return self._d.outcome(
            {
                "domain": f"{domain}.{slug(criterion)}",
                "state": f"crit:{slug(criterion)}|{features(output, criterion)}",
                "answer": "yes" if should_pass else "no",
                "reward": 1.0,
            }
        )
