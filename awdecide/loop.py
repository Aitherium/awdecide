"""The loop -- ask, act, resolve; the next identical decision costs nothing.

A ladder answers a question. A loop REMEMBERS the answer's outcome. `Loop` wraps
any `Ladder` with the one rung a stateless decider cannot have: **evidence** --
the resolved outcomes of this exact (key, state) in the ledger.

    loop = Loop(Ladder([ChatBackend(url, model)]), Ledger())
    d = loop.decide("test-runner", "lang:py,changed:tests", Question.choice([...]))
    ...act on d.value, look at what happened...
    loop.resolve(d.id, correct=it_worked)
    loop.decide(...same key, same state...)   # backend="evidence", no model call

What the evidence rung does, in order:

  1. An option resolved RIGHT more often than wrong here answers, at its
     Laplace-smoothed hit rate -- so a 60/40 coin reads ~0.60, never 0.99.
  2. For a bool, evidence against one side is evidence for the other.
  3. Otherwise the ladder is asked -- but every option already resolved WRONG
     here is withheld from it, so a mistake that was reported is not repeated.

A backend whose probability is not earned (`calibrated = False`: a chat model's
bare label, a heuristic) has it REPLACED by that backend's measured hit rate at
this key -- 0.5 until outcomes exist. Logprob and door rungs keep their own.

`state` must be a STABLE descriptor: the same situation must produce the same
string, or nothing can be learned about it. Stdlib only.
"""
from __future__ import annotations

import json
import random
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .backends import Backend, Ladder, Weights
from .contract import Decision, Question, undecided
from .ledger import Ledger, _sha

EVIDENCE = "evidence"
PRIOR_WEIGHT = 2.0   # pseudo-observations the fork-wide record is worth


def laplace(hits: float, n: float) -> float:
    return (hits + 1.0) / (n + 2.0)


class ChatBackend(Backend):
    """Any OpenAI-wire chat endpoint, asked for exactly one label. For servers
    that do not return logprobs (most hosted ones). Its weight is a bare pick,
    so it is `calibrated = False`: the loop supplies the probability."""

    name = "chat"
    calibrated = False

    def __init__(self, base_url: str, model: str, *, token: str = "",
                 timeout: float = 60.0) -> None:
        if not base_url or not model:
            raise ValueError("ChatBackend needs an explicit base_url and model")
        self.base_url = base_url.rstrip("/")
        self.model, self.token, self.timeout = model, token, timeout

    def weights(self, state: str, q: Question) -> Weights:
        prompt = (f"State:\n{state}\n\n{q.prompt or 'Which option applies?'}\n"
                  f"Answer with exactly one of these labels and nothing else: "
                  + ", ".join(q.options))
        body = json.dumps({"model": self.model, "temperature": 0, "max_tokens": 32,
                           "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
        url = self.base_url + ("" if self.base_url.endswith("/v1") else "/v1")
        req = urllib.request.Request(url + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                text = json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]
        except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"{type(e).__name__}: {e}") from e
        picked = match_label(str(text or ""), list(q.options))
        return {picked: 1.0} if picked else {}


def match_label(text: str, options: List[str]) -> Optional[str]:
    """The one option the text names. Prose naming two options names none."""
    t = text.strip().strip("\"'`.").lower()
    for o in options:
        if t == o.lower():
            return o
    hits = [o for o in options if o.lower() in t]
    return hits[0] if len(hits) == 1 else None


class Loop:
    def __init__(self, ladder: Ladder, ledger: Ledger) -> None:
        self.ladder, self.ledger = ladder, ledger
        self._uncalibrated = {b.name for b in ladder.backends
                              if getattr(b, "calibrated", True) is False}

    # ---------------------------------------------------------------- ask
    def decide(self, key: str, state: str, q: Question) -> Decision:
        if not key or not key.strip():
            raise ValueError("key is required -- it names WHICH decision this is")
        if not isinstance(state, str) or not state.strip():
            raise ValueError("state must be a non-empty STABLE descriptor of the situation")
        seen = self._evidence(key, state, q)
        d = self._from_evidence(q, seen, self._prior(key))
        if d is None:
            wrong = {o for o, (n, hits) in seen.items() if laplace(hits, n) <= 0.5}
            open_opts = [o for o in q.options if o not in wrong]
            if len(open_opts) == 1:        # everything else was tried here and failed
                d = self._shape(q, open_opts[0], 0.5, EVIDENCE,
                                ["evidence: the only option not resolved wrong here"])
            else:
                ask = q if len(open_opts) == len(q.options) or len(open_opts) < 2 else \
                    Question(q.kind, tuple(open_opts), 0.0, q.prompt)
                d = self.ladder.decide_one(state, ask)
                if d.decided and d.backend in self._uncalibrated:
                    n, hits = self._backend_rate(key, d.backend)
                    d = self._shape(q, d.value, laplace(hits, n), d.backend,
                                    d.reasons + [f"{d.backend}: probability is its measured hit "
                                                 f"rate at this key (n={int(n)})"])
                elif d.decided:
                    d.probabilities = {o: d.probabilities.get(o, 0.0) for o in q.options}
        if d.decided and d.probability < q.min_confidence:
            held = undecided(q, d.backend, d.reasons + [
                f"{d.backend}: {d.probability:.3f} < min_confidence {q.min_confidence:.3f}"])
            held.probabilities = d.probabilities
            return held
        self.ledger.record(key, state, d)
        return d

    def resolve(self, decision_id: str, correct: bool) -> Optional[float]:
        return self.ledger.resolve(decision_id, bool(correct))

    def teach(self, key: str, state: str, value: str, correct: bool, kind: str = "choice") -> str:
        """Record an outcome observed WITHOUT asking first (a human's pick, a log)."""
        d = Decision(kind=kind, value=value, probability=0.5, probabilities={value: 0.5},
                     decided=True, backend="taught")
        did = self.ledger.record(key, state, d)
        self.ledger.resolve(did, bool(correct))
        return did

    # ------------------------------------------------------------ evidence
    def _evidence(self, key: str, state: str, q: Question) -> Dict[str, Tuple[float, float]]:
        rows = self.ledger._con.execute(
            "SELECT value, COUNT(*), SUM(outcome) FROM decisions WHERE key=? AND state_sha=? "
            "AND outcome IS NOT NULL GROUP BY value", (key, _sha(state)))
        return {v: (float(n), float(h or 0)) for v, n, h in rows if v in q.options}

    def _backend_rate(self, key: str, backend: str) -> Tuple[float, float]:
        n, h = self.ledger._con.execute(
            "SELECT COUNT(*), SUM(outcome) FROM decisions WHERE key=? AND backend=? "
            "AND outcome IS NOT NULL", (key, backend)).fetchone()
        return float(n or 0), float(h or 0)

    def _prior(self, key: str) -> float:
        """How often an evidence answer has been right AT THIS KEY. One right
        outcome in a fork where evidence is right 99% of the time is worth more
        than one in a coin-flip fork; a flat prior reads both as 0.667 and is
        under-confident exactly where the loop is strongest."""
        n, hits = self._backend_rate(key, EVIDENCE)
        return laplace(hits, n)

    def _from_evidence(self, q: Question, seen: Dict[str, Tuple[float, float]],
                       prior: float = 0.5) -> Optional[Decision]:
        if not seen:
            return None
        best, (n, hits) = max(seen.items(), key=lambda kv: laplace(kv[1][1], kv[1][0]))
        if laplace(hits, n) <= 0.5:          # the VERDICT uses the flat prior ...
            p = laplace(hits, n)
        else:                                # ... the PROBABILITY uses the fork's own record
            p = (hits + PRIOR_WEIGHT * prior) / (n + PRIOR_WEIGHT)
        total = int(sum(v[0] for v in seen.values()))
        if p > 0.5:
            return self._shape(q, best, p, EVIDENCE,
                               [f"evidence: {total} resolved outcome(s) here"])
        if q.kind == "bool":
            other = "no" if best == "yes" else "yes"
            return self._shape(q, other, 1.0 - p, EVIDENCE,
                               [f"evidence: {best!r} resolved wrong here ({total} outcome(s))"])
        return None

    @staticmethod
    def _shape(q: Question, value: str, p: float, backend: str, reasons: List[str]) -> Decision:
        rest = [o for o in q.options if o != value]
        probs = {o: (1.0 - p) / len(rest) for o in rest} if rest else {}
        probs[value] = p
        return Decision(kind=q.kind, value=value, probability=p, probabilities=probs,
                        decided=True, backend=backend, reasons=reasons)

    def stats(self) -> Dict[str, Any]:
        out = self.ledger.reliability()
        out["answered_by"] = dict(self.ledger._con.execute(
            "SELECT backend, COUNT(*) FROM decisions GROUP BY backend"))
        out["keys"] = [{"key": k, "states": s, "resolved": int(r or 0)} for k, s, r in
                       self.ledger._con.execute(
                           "SELECT key, COUNT(DISTINCT state_sha), SUM(outcome IS NOT NULL) "
                           "FROM decisions GROUP BY key ORDER BY 3 DESC LIMIT 50")]
        return out


# --------------------------------------------------------------- the bench
def bench(states: int = 40, repeats: int = 10, brain_acc: float = 0.70,
          brain_ms: float = 300.0, seed: int = 7, db: str = ":memory:") -> Dict[str, Any]:
    """The SAME imperfect brain used two ways on `states` situations seen
    `repeats` times each: called every time, or behind the loop. Nothing is
    tuned -- change any argument and rerun."""
    from pathlib import Path

    from .backends import CallableBackend
    rng = random.Random(seed)
    opts = ["a", "b", "c", "d"]
    truth = {f"s{i}": rng.choice(opts) for i in range(states)}
    calls = {"n": 0}

    def brain(state: str, q: Question) -> Weights:
        calls["n"] += 1
        open_opts = list(q.options)
        if truth[state] in open_opts and rng.random() < brain_acc:
            return {truth[state]: 1.0}
        wrong = [o for o in open_opts if o != truth[state]]
        return {rng.choice(wrong or open_opts): 1.0}

    order = [s for s in truth for _ in range(repeats)]
    rng.shuffle(order)
    q = Question.choice(opts)
    static_right = sum(next(iter(brain(s, q))) == truth[s] for s in order)
    static_calls, calls["n"] = calls["n"], 0

    rung = CallableBackend(brain, name="brain")
    rung.calibrated = False  # type: ignore[attr-defined]
    loop = Loop(Ladder([rung]), Ledger(Path(db)))
    right, by = 0, {}
    for s in order:
        d = loop.decide("bench", s, q)
        ok = d.value == truth[s]
        right += ok
        by[d.backend] = by.get(d.backend, 0) + 1
        loop.resolve(d.id, ok)
    total = len(order)
    rel = loop.ledger.reliability()
    return {
        "decisions": total, "brain_accuracy": brain_acc,
        "static": {"accuracy": round(static_right / total, 4), "model_calls": static_calls,
                   "model_seconds": round(static_calls * brain_ms / 1000.0, 1)},
        "loop": {"accuracy": round(right / total, 4), "model_calls": calls["n"],
                 "model_seconds": round(calls["n"] * brain_ms / 1000.0, 1), "answered_by": by,
                 "brier": rel.get("brier"), "climatology": rel.get("climatology")},
    }
