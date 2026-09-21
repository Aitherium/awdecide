"""The backend ladder — rungs that can put a probability on a bounded set.

A ladder is an ordered list of backends. `Ladder.decide` asks each in turn and
returns the FIRST answer that clears the question's `min_confidence`; if none
does, the answer is `decided=False` with every rung's reason. Cheap and
deterministic rungs go first (rules), learned local models next (anything
callable: a nanoGPT, an sklearn model, a platform classify surface), and a
language model's logprobs last — the same shape as "tiny model above threshold,
LLM below", except every rung is yours and every answer lands in the ledger.

Three rungs ship here, all stdlib:

  RulesBackend     a table of (predicate, option) pairs. A matched rule answers
                   with probability 1.0; no match ABSTAINS (returns {}), it does
                   not guess. Rules never emit a probability they did not earn.
  CallableBackend  wraps any `fn(state, question) -> {option: weight}`. This is
                   the door for a local model. Exceptions abstain with the reason.
  LogprobBackend   an OpenAI-wire /v1/chat/completions server with `logprobs`:
                   the model is asked to answer with exactly one option label,
                   the top logprobs of the first generated token are read, and
                   the mass over the option labels is the distribution. No
                   parsing of prose; a label that is not an option is dropped.

A backend returns raw weights; the contract normalizes and applies
`min_confidence`. A backend that raises is a rung that abstained, never a crash.
"""
from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .contract import Decision, Question, from_probabilities, undecided

Weights = Dict[str, float]
Predicate = Callable[[str], bool]


class Backend:
    name = "backend"

    def weights(self, state: str, q: Question) -> Weights:  # pragma: no cover - interface
        raise NotImplementedError


class RulesBackend(Backend):
    name = "rules"

    def __init__(self, rules: Sequence[Tuple[Any, str]] = ()) -> None:
        """rules: (predicate, option). A predicate is a callable(state)->bool or a
        regex string (searched case-insensitively). First match wins."""
        self._rules: List[Tuple[Predicate, str]] = []
        for pred, option in rules:
            self.add(pred, option)

    def add(self, pred: Any, option: str) -> "RulesBackend":
        if isinstance(pred, str):
            rx = re.compile(pred, re.IGNORECASE)
            self._rules.append((lambda s, _rx=rx: bool(_rx.search(s)), option))
        elif callable(pred):
            self._rules.append((pred, option))
        else:
            raise TypeError("rule predicate must be a regex string or a callable")
        return self

    def weights(self, state: str, q: Question) -> Weights:
        for pred, option in self._rules:
            if option in q.options and pred(state):
                return {option: 1.0}
        return {}


class CallableBackend(Backend):
    def __init__(self, fn: Callable[[str, Question], Weights], name: str = "callable") -> None:
        self._fn = fn
        self.name = name

    def weights(self, state: str, q: Question) -> Weights:
        out = self._fn(state, q)
        if not isinstance(out, dict):
            raise TypeError(f"{self.name}: expected a dict of weights, got {type(out).__name__}")
        return out


class LogprobBackend(Backend):
    name = "logprob"

    def __init__(self, base_url: str, model: str, *, token: str = "",
                 timeout: float = 60.0, top_logprobs: int = 20) -> None:
        if not base_url:
            raise ValueError("LogprobBackend needs an explicit base_url -- never a guessed one")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.token = token
        self.timeout = timeout
        self.top_logprobs = max(2, min(20, top_logprobs))

    @staticmethod
    def _prompt(state: str, q: Question) -> str:
        labels = ", ".join(q.options)
        framing = q.prompt or {
            "choice": "Which option applies?",
            "score": "Which level applies? Levels are ordered low to high.",
            "bool": "Is the statement true?",
        }[q.kind]
        return (f"State:\n{state}\n\n{framing}\nAnswer with exactly one of these labels and "
                f"nothing else: {labels}")

    def weights(self, state: str, q: Question) -> Weights:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": self._prompt(state, q)}],
            "max_tokens": 4, "temperature": 0,
            "logprobs": True, "top_logprobs": self.top_logprobs,
        }).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            raise RuntimeError(f"{type(e).__name__}: {e}") from e
        return self.weights_from_response(data, q)

    @staticmethod
    def weights_from_response(data: Dict[str, Any], q: Question) -> Weights:
        """Mass over option labels from the first token's top_logprobs.

        Labels are matched on their first token, case-insensitively, after
        stripping whitespace; an option whose first token collides with another
        option's is a schema problem the caller sees as split mass, not a guess.
        """
        try:
            content = data["choices"][0]["logprobs"]["content"]
            top = content[0]["top_logprobs"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"no logprobs in response ({type(e).__name__})") from e
        out: Weights = {}
        for entry in top:
            tok = str(entry.get("token", "")).strip().lower()
            if not tok:
                continue
            p = math.exp(float(entry.get("logprob", -math.inf)))
            for opt in q.options:
                if opt.lower().startswith(tok) or tok.startswith(opt.lower()):
                    out[opt] = out.get(opt, 0.0) + p
        return out


class Ladder:
    def __init__(self, backends: Sequence[Backend]) -> None:
        self.backends = list(backends)

    def decide_one(self, state: str, q: Question) -> Decision:
        reasons: List[str] = []
        best: Optional[Decision] = None  # strongest sub-threshold evidence, kept, not acted on
        for b in self.backends:
            own = getattr(b, "decide", None)
            if callable(own):
                # A rung that owns a CALIBRATED probability (the world-model door,
                # awdecide/door.py) hands back a Decision; renormalizing its
                # per-option numbers into weights would destroy the calibration.
                try:
                    d = own(state, q)
                except Exception as e:
                    reasons.append(f"{b.name}: abstained ({e})")
                    continue
                if d is None or (not d.decided and not any(d.probabilities.values())):
                    reasons.extend(d.reasons if d is not None else [f"{b.name}: abstained"])
                    continue
                if d.decided:
                    d.reasons = reasons + d.reasons + [f"{d.backend}: answered"]
                    return d
                reasons.extend(d.reasons)
                if best is None or max(d.probabilities.values()) > max(
                        best.probabilities.values(), default=0.0):
                    best = d
                continue
            try:
                raw = b.weights(state, q)
            except Exception as e:  # a rung that failed is a rung that abstained
                reasons.append(f"{b.name}: abstained ({e})")
                continue
            if not raw:
                reasons.append(f"{b.name}: abstained (no rule / no mass)")
                continue
            d = from_probabilities(q, raw, b.name)
            if d.decided:
                d.reasons = reasons + [f"{b.name}: answered"]
                return d
            reasons.extend(d.reasons)
            top = max(d.probabilities.values(), default=0.0)
            if best is None or top > max(best.probabilities.values(), default=0.0):
                best = d
        if not self.backends:
            reasons.append("ladder: no backends configured")
        out = undecided(q, "none", reasons)
        if best is not None:
            out.probabilities = best.probabilities
            out.backend = best.backend
        return out

    def decide(self, state: str, questions: Dict[str, Question]) -> Dict[str, Decision]:
        return {key: self.decide_one(state, q) for key, q in questions.items()}


def parse_question_spec(spec: str) -> Tuple[str, Question]:
    """CLI grammar: `key:choice=a,b,c` · `key:score=low,mid,high` · `key:bool`
    with an optional `@0.7` min_confidence suffix, e.g. `category:choice=a,b@0.6`."""
    m = re.fullmatch(r"([A-Za-z_][\w-]*):(choice|score|bool)(?:=([^@]+))?(?:@([0-9.]+))?", spec)
    if not m:
        raise ValueError(f"bad question spec {spec!r}; "
                         "want key:choice=a,b | key:score=l1,l2 | key:bool")
    key, kind, opts, conf = m.groups()
    kw: Dict[str, Any] = {"min_confidence": float(conf)} if conf else {}
    if kind == "bool":
        return key, Question.bool(**kw)
    if not opts:
        raise ValueError(f"{key}: {kind} needs options")
    options = [o.strip() for o in opts.split(",") if o.strip()]
    if kind == "choice":
        return key, Question.choice(options, **kw)
    return key, Question.score(options, **kw)


def default_ladder(*, rules: Optional[RulesBackend] = None, logprob_url: str = "",
                   logprob_model: str = "", token: str = "") -> Ladder:
    rungs: List[Backend] = []
    if rules is not None:
        rungs.append(rules)
    if logprob_url:
        rungs.append(LogprobBackend(logprob_url, logprob_model or "default", token=token))
    return Ladder(rungs)
