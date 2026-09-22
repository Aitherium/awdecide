"""multi -- one messy context, many typed decisions, ONE call (gap G2, 2026-09-20).

The shape people actually want from a decision layer ("fraud, churn, escalation,
priority in one call over the same user history / device data / chat"):

    POST /decide/multi
      {"context": "<the messy blob>", "fork": "risk",
       "questions": [{"name": "fraud",      "kind": "yesno"},
                     {"name": "churn",      "kind": "score"},
                     {"name": "escalation", "kind": "choice", "options": ["none","tier1","tier2"]},
                     {"name": "priority",   "kind": "choice", "options": ["low","med","high"]}]}
   -> {"answers": {"fraud": {...full /decide answer...}, "churn": {...}, ...},
       "state_key": "...", "from_evidence": n, "from_model": n, "llm_calls": 0|1, ...}

How it maps onto the door (decide.py), and why it can LEARN:

* ONE state descriptor per context (`context_key`), shared by every question. The
  raw blob never repeats, so a key that carried it could never be learned about.
  The key is `<shape features>|h:<12 hex>`, and here is exactly what is and is
  not in it:

    IN   len bucket, line-count bucket, traceback/error-word presence, pass/fail
         buckets, exit status, key=value pair-count bucket (judge.features-style
         bucketing, same helpers), and a sha256 prefix of the NORMALISED text.
         Normalisation: lower-case, whitespace collapsed, ISO/clock timestamps,
         UUIDs, hex runs >= 8 and digit runs >= 4 replaced by placeholders.
    OUT  the fork, the question names/kinds/options and their order, casing,
         whitespace, timestamps, ids, long numbers (normalised away).

  So two contexts that differ only by a timestamp or a request id collapse to
  one key (learnable); two that differ by one word do not -- that gap is the
  neighbour rung's (the key, not the blob, is what gets embedded; the shape
  features lead the string on purpose so near shapes embed near). Trade-off
  stated, not hidden: a hash key is exact-match learning first, similarity second.

* Every question becomes its own domain `decide.<fork>.<name>` with that key as
  state, and they fan out through `Decider.decide_batch` -- so a question the door
  has evidence on answers from the engine in microseconds, and only the COLD ones
  reach the LLM rung.

* The LLM rung is prompted ONCE per context: `SharedLLM` is handed to
  `decide_batch(llm=...)`; the first cold question triggers one prompt with the
  context ONCE and ALL the questions, the answer is parsed back per question and
  every other cold question is served from that cache. `llm_calls` in the response
  is the measured count (0 when every question was warm).

* Outcomes teach per question through the unchanged `/decide/outcome`: each answer
  carries its own `decision_id`; explicit teaching uses `domain=decide.<fork>.<name>`
  and `state=<state_key>` (both are in the response).

Honesty notes the response carries rather than hides:
  `unlearnable`  question names whose kind is `score` with no `options`: the door
                 answers a continuous 0..1 from the LLM and the engine has no
                 option to attach evidence to. Give options (a scale) to learn.
  `llm_calls`    0 or 1 for the whole call, never N.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import decide as decide_mod
import judge as judge_mod

logger = logging.getLogger("wm.multi")

MAX_QUESTIONS = 16
MAX_CONTEXT_CHARS = 32_000
PROMPT_CONTEXT_CHARS = 6_000

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_FORK = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$", re.I)
_TS_ISO = re.compile(r"\d{4}-\d{2}-\d{2}[t ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(z|[+-]\d{2}:?\d{2})?", re.I)
_TS_CLOCK = re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?(\.\d+)?\b")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_HEX = re.compile(r"\b[0-9a-f]{8,}\b", re.I)
_NUM = re.compile(r"\d{4,}")
_WS = re.compile(r"\s+")
_KV = re.compile(r"\b[a-z_][a-z0-9_]{1,40}\s*[:=]\s*\S", re.I)


# ------------------------------------------------------------------- the key
def normalise(text: str) -> str:
    """Lower-case, collapse whitespace, and blank out the parts of a blob that
    change on every run without changing the situation."""
    t = (text or "").lower()
    t = _TS_ISO.sub("<ts>", t)
    t = _DATE.sub("<date>", t)
    t = _TS_CLOCK.sub("<time>", t)
    t = _UUID.sub("<uuid>", t)
    t = _HEX.sub("<hex>", t)
    t = _NUM.sub("<n>", t)
    return _WS.sub(" ", t).strip()


def context_key(context: str) -> str:
    """ONE stable descriptor for a context blob (see the module docstring for
    exactly what is and is not in it)."""
    text = context or ""
    counts: Dict[str, int] = {}
    for num, kind in judge_mod._PASSFAIL.findall(text):
        k = kind.lower().rstrip("s")
        counts[k] = counts.get(k, 0) + int(num)
    exits = {int(m) for m in judge_mod._EXIT.findall(text)}
    lines = text.count("\n") + (1 if text.strip() else 0)
    norm = normalise(text)
    parts = [
        f"len:{judge_mod._bucket(len(text))}",
        f"lines:{judge_mod._count_bucket(lines)}",
        f"tb:{int(bool(judge_mod._TRACEBACK.search(text)))}",
        f"err:{int(bool(judge_mod._ERROR.search(text)))}",
        f"pass:{judge_mod._count_bucket(counts.get('passed', 0))}",
        f"fail:{judge_mod._count_bucket(counts.get('failed', 0) + counts.get('error', 0))}",
        f"exit:{'none' if not exits else ('0' if exits == {0} else 'nonzero')}",
        f"kv:{judge_mod._count_bucket(len(_KV.findall(text)))}",
        f"h:{hashlib.sha256(norm.encode('utf-8', 'replace')).hexdigest()[:12]}",
    ]
    return "|".join(parts)


# ------------------------------------------------------------- the shared LLM
_SYSTEM_MULTI = (
    "You are a decision function. You are given ONE context and several typed "
    "questions about it. Answer with ONE JSON object and nothing else, keyed by "
    'question name: {"<name>": {"answer": <one of that question\'s allowed answers, '
    'verbatim>, "confidence": <0..1>}, ...}. Answer EVERY question. No prose, no '
    "markdown, no thinking text."
)


def _allowed(q: Dict[str, Any]) -> str:
    if q["kind"] == "score" and q["options"] is None:
        return "a real number between 0 and 1"
    return ", ".join(json.dumps(o) for o in q["options"])


def build_multi_prompt(context: str, questions: List[Dict[str, Any]]) -> str:
    """The context ONCE, then every question. `questions` are normalised items
    (name, kind, options, question)."""
    lines = ["Context:", "<<<", (context or "")[:PROMPT_CONTEXT_CHARS], ">>>", "", "Questions:"]
    for i, q in enumerate(questions, 1):
        extra = f" -- {q['question']}" if q.get("question") else ""
        lines.append(f"{i}. name={q['name']} kind={q['kind']} allowed: {_allowed(q)}{extra}")
    lines.append("")
    lines.append(
        f"Answer all {len(questions)} questions as ONE JSON object keyed by name."
    )
    return "\n".join(lines)


def _json_objects(text: str):
    """Every JSON object parseable from a `{` in the text, outermost first."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text or ""):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text, i)
        except ValueError:
            continue
        if isinstance(obj, dict):
            yield obj


def parse_multi_answer(text: Any, names: List[str]) -> Dict[str, Dict[str, Any]]:
    """{name: {"answer", "confidence"}} for every name the model answered.
    Accepts the requested nested shape, an `{"answers": {...}}` wrapper, and
    bare values (`{"fraud": "yes"}` -> confidence 0.5). Raises ValueError when
    no object names any question -- the caller treats that as no LLM answer."""
    if isinstance(text, dict):
        candidates = [text]
    else:
        candidates = list(_json_objects(str(text)))
    for obj in candidates:
        if "answers" in obj and isinstance(obj["answers"], dict):
            obj = obj["answers"]
        hit = {k: v for k, v in obj.items() if k in names}
        if not hit:
            continue
        out: Dict[str, Dict[str, Any]] = {}
        for name, val in hit.items():
            if isinstance(val, dict):
                ans = val.get("answer")
                conf = val.get("confidence", 0.5)
            else:
                ans, conf = val, 0.5
            try:
                conf = max(0.0, min(1.0, float(conf)))
            except (TypeError, ValueError):
                conf = 0.5
            out[name] = {"answer": ans, "confidence": conf}
        return out
    raise ValueError(f"no per-question JSON object in LLM answer: {str(text)[:120]!r}")


class SharedLLM:
    """The LLM rung for ONE context. `Decider._ask_llm` calls it once per COLD
    question, from pool threads; the first call sends ONE prompt (context once,
    all questions) and caches every question's answer, later calls are served
    from the cache. `calls` is the number of brain round trips: 0 or 1, and a
    failed round trip is cached too (one failure, not N retries)."""

    def __init__(
        self,
        base: Callable[..., Any],
        context: str,
        questions: List[Dict[str, Any]],
        domain_of: Dict[str, str],
    ) -> None:
        self._base = base
        self._prompt = build_multi_prompt(context, questions)
        self._names = [q["name"] for q in questions]
        self._name_of_domain = {d: n for n, d in domain_of.items()}
        self._max_tokens = 48 + 40 * len(questions)
        self._lock = threading.Lock()
        self._answers: Optional[Dict[str, Dict[str, Any]]] = None
        self._error: Optional[BaseException] = None
        self.calls = 0

    def __call__(self, prompt: str) -> Dict[str, Any]:
        domain = decide_mod.prompt_domain(prompt)
        name = self._name_of_domain.get(domain or "")
        if name is None:
            raise LookupError(f"shared LLM asked for an unknown fork {domain!r}")
        with self._lock:
            if self._answers is None and self._error is None:
                self.calls += 1
                try:
                    raw = self._base(
                        self._prompt, system=_SYSTEM_MULTI, max_tokens=self._max_tokens
                    )
                    self._answers = parse_multi_answer(raw, self._names)
                except Exception as exc:  # noqa: BLE001 -- cached: one failure per context
                    self._error = exc
        if self._error is not None:
            raise self._error
        got = (self._answers or {}).get(name)
        if got is None:
            raise LookupError(f"the brain did not answer question {name!r}")
        return got


# ------------------------------------------------------------------ the door
class Multi:
    """One context, many typed decisions, over the same Decider the service runs.

    `llm(prompt, *, system, max_tokens) -> str | dict` is the brain for the
    shared prompt; None = the local brain via decide.llm_chat_text."""

    def __init__(self, decider: Any, llm: Optional[Callable[..., Any]] = None) -> None:
        self._d = decider
        self._llm = llm if llm is not None else decide_mod.llm_chat_text

    @staticmethod
    def normalize(body: Dict[str, Any]) -> Dict[str, Any]:
        context = body.get("context")
        if not isinstance(context, str) or not context.strip():
            raise decide_mod.DecideError("context must be a non-empty string")
        if len(context) > MAX_CONTEXT_CHARS:
            raise decide_mod.DecideError(f"context is over {MAX_CONTEXT_CHARS} chars")
        fork = str(body.get("fork") or "").strip()
        if fork.startswith("decide."):
            fork = fork[len("decide.") :]
        if not fork or not _FORK.match(fork):
            raise decide_mod.DecideError(
                "fork must be a short name ([a-z0-9_.-], e.g. 'risk'); the domain "
                "for each question becomes decide.<fork>.<name>"
            )
        raw_qs = body.get("questions")
        if not isinstance(raw_qs, list) or not raw_qs:
            raise decide_mod.DecideError("questions must be a non-empty list")
        if len(raw_qs) > MAX_QUESTIONS:
            raise decide_mod.DecideError(f"at most {MAX_QUESTIONS} questions per call")
        learn = bool(body.get("learn", True))
        questions: List[Dict[str, Any]] = []
        seen = set()
        for raw in raw_qs:
            if not isinstance(raw, dict):
                raise decide_mod.DecideError("each question must be an object")
            name = str(raw.get("name") or "").strip()
            if not _NAME.match(name):
                raise decide_mod.DecideError(
                    f"question name {name!r} must match [a-z][a-z0-9_]{{0,31}}"
                )
            if name in seen:
                raise decide_mod.DecideError(f"duplicate question name {name!r}")
            seen.add(name)
            item = decide_mod.Decider.normalize(
                {
                    "domain": f"decide.{fork}.{name}",
                    "state": "pending",  # replaced by the context key below
                    "kind": raw.get("kind", "choice"),
                    "options": raw.get("options"),
                    "question": raw.get("question", ""),
                    "min_confidence": raw.get("min_confidence", 0.0),
                    "learn": learn,
                }
            )
            item["name"] = name
            questions.append(item)
        return {"context": context, "fork": fork, "questions": questions, "learn": learn}

    def decide_multi(self, body: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        norm = self.normalize(body)
        key = context_key(norm["context"])
        questions = norm["questions"]
        domain_of = {q["name"]: q["domain"] for q in questions}
        items = []
        for q in questions:
            it = {k: v for k, v in q.items() if k != "name"}
            it["state"] = key
            items.append(it)
        shared = SharedLLM(self._llm, norm["context"], questions, domain_of)
        res = self._d.decide_batch(items, llm=shared)
        answers = {q["name"]: a for q, a in zip(questions, res["answers"])}
        sources = [str(a.get("source") or "none") for a in answers.values()]
        by_source: Dict[str, int] = {}
        for s in sources:
            by_source[s] = by_source.get(s, 0) + 1
        # from_evidence is "not a model call, not a guess", so a rung added to
        # decide.py LATER is counted rather than silently dropped. Measured
        # 2026-09-20: the neural rung (G1) landed mid-session and an allowlist of
        # ("engine", "neighbor") reported from_evidence=0 for an answer that came
        # from 611 learned outcomes -- 4 questions, 3 accounted for. `by_source`
        # is the raw tally so no rung can hide inside a bucket either.
        return {
            "answers": answers,
            "state_key": key,
            "fork": norm["fork"],
            "questions": len(questions),
            "by_source": by_source,
            "from_evidence": sum(1 for s in sources if s not in ("llm", "prior", "none")),
            "from_model": sum(1 for s in sources if s == "llm"),
            "from_prior": sum(1 for s in sources if s == "prior"),
            "unknown": sum(1 for s in sources if s == "none"),
            "llm_calls": shared.calls,
            "unlearnable": [
                q["name"] for q in questions if q["kind"] == "score" and q["options"] is None
            ],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }


__all__ = [
    "MAX_QUESTIONS",
    "MAX_CONTEXT_CHARS",
    "Multi",
    "SharedLLM",
    "build_multi_prompt",
    "context_key",
    "normalise",
    "parse_multi_answer",
]
