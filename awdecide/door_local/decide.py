"""decide -- the decision primitive: choice / score / yesno, batched, LEARNING.

Owner, 2026-09-19: "I want to see my system outperforming this Jev bullshit today."

What Jev sells (stripped of its marketing): a typed classifier-as-a-service for
bounded forks -- pick one of N, place on a scale, P(yes) -- batched, sub-second,
placed at every routing fork, with "read fresh state" as the verify step. It is a
STATIC model: the decision it made yesterday does not make today's better.

What this is: the same three primitives on the same one door, backed first by the
world model's per-domain engine (tabular -> hybrid -> neural, journaled, replayed
on restart), then by the fleet's own local brain through MicroScheduler, never by
a cloud call. Every decision has an id; POST the outcome back and the engine
observes (state, answer) -> reward. The next time that state (or, once the engine
goes neural, a state LIKE it) comes back, the answer comes from the table in
microseconds with a confidence that reflects how often it was right -- and the
LLM is not called at all. That is the sentence Jev cannot say.

Contract (in-fleet: http://127.0.0.1:8197/decide; public: /v1/decide
on the wm gateway):

  POST /decide
    {"domain": "decide.router", "state": "<stable descriptor>", "kind": "choice",
     "options": ["local", "cloud"], "question": "...", "min_confidence": 0.6}
  -> {"decision_id", "answer", "confidence", "source": "engine|llm|prior|none",
      "learned_from": n, "latency_ms", "alternatives": [{"answer", "value", "n"}]}

  POST /decide/batch {"items": [...]}      -> {"answers": [...], "latency_ms"}
  POST /decide/multi {"context", "fork", "questions": [{name, kind, options?}]}
                                            -> one context, many typed decisions,
                                               ONE brain prompt for the cold ones
                                               (multi.py)
  POST /decide/outcome {"decision_id", "reward": -1..1}   (or domain+state+answer)
  GET  /decide/stats                        per-domain: decisions, engine-served %,
                                            llm-served %, outcomes, mean reward

Kinds:
  choice  options[] -> one option.
  score   options[] are the scale points (["1","2","3","4","5"]) or omitted for a
          0..1 real; the answer is the point (string) or the number.
  yesno   options default ["yes","no"]; confidence is P(answer).

Source ladder, per item:
  engine  the domain engine has a prediction for (state, option) -> argmax reward,
          confidence from evidence count and margin. Free. Microseconds.
  llm     no prediction (cold state) -> the local brain answers a strict-JSON
          prompt. One local call, no cloud.
  prior   the LLM is unavailable -> domain-wide per-option mean reward (what has
          worked in this domain regardless of state). Confidence is low and SAYS so.
  none    nothing to go on -> first option, confidence 0.0. Never a fabricated
          number: a caller that reads confidence 0 knows it is guessing.

Designed to run with no torch on the host (tests) and inside the world-model
container (prod): the only hard dependency is the DomainEngines it is handed.
"""

from __future__ import annotations

import hashlib
import json
import math
import logging
import os
import re
import ssl
import threading
import time
import zlib
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("wm.decide")

KINDS = ("choice", "score", "yesno")
_YESNO = ["yes", "no"]
# Confidence from evidence: n / (n + K). K=2 -> one observation is 0.33, three
# are 0.6, eight are 0.8. Margin between the best and the runner-up scales it so
# a coin-flip table (two options both at reward 0.5) never reads as certain.
_EVIDENCE_K = 2.0

LLM_URL = os.environ.get("AITHER_DECIDE_LLM_URL", "http://127.0.0.1:8080/v1").rstrip(
    "/"
)
LLM_MODEL = os.environ.get("AITHER_DECIDE_LLM_MODEL", "default")
LLM_TIMEOUT = float(os.environ.get("AITHER_DECIDE_LLM_TIMEOUT", "20"))
LLM_KEY = os.environ.get("AITHER_DECIDE_LLM_KEY") or os.environ.get("OPENAI_API_KEY") or "local"
_BATCH_WORKERS = int(os.environ.get("AITHER_DECIDE_BATCH_WORKERS", "16"))

# The neighbor rung: awembed's distilled micro-embedder (1024-dim, OpenAI-shaped
# /v1/embeddings, network aliases your embedding server).
# Measured 2026-09-19 on decision descriptors: sim(code-long, code-medium)=0.874,
# sim(code-long, image)=0.746 -- a wider near/far gap than nomic (0.891/0.832),
# which is what a threshold needs. Empty URL disables the rung.
EMBED_URL = os.environ.get("AITHER_DECIDE_EMBED_URL", "http://127.0.0.1:8229").rstrip("/")
EMBED_MODEL = os.environ.get("AITHER_DECIDE_EMBED_MODEL", "local-embed")
EMBED_TIMEOUT = float(os.environ.get("AITHER_DECIDE_EMBED_TIMEOUT", "8"))
NEIGHBOR_MIN_SIM = float(os.environ.get("AITHER_DECIDE_NEIGHBOR_MIN_SIM", "0.85"))
# The floor belongs to the EMBEDDER. 0.85 was measured for awembed's distilled
# micro-embedder (near 0.874 / far 0.746). On nomic the same 0.85 has an 11.59%
# false-neighbor rate on this door's own labelled states and costs accuracy
# (96.0% vs 97.5% with the rung off); 0.96 is the smallest floor with ZERO false
# neighbors there and lifts the bench to 98.5% on 25% fewer model calls.
# Re-measure with tools/neighbor_threshold.py before adding a row.
EMBEDDER_FLOORS: Dict[str, float] = {
    "local-embed": 0.85,
    "nomic-embed-text": 0.96,
}
# When the primary embedder is unreachable, the fleet's one model door also
# serves /v1/embeddings. Empty disables the fallback.
EMBED_FALLBACK_URL = os.environ.get(
    "AITHER_DECIDE_EMBED_FALLBACK_URL", LLM_URL[: -len("/v1")] if LLM_URL.endswith("/v1") else ""
).rstrip("/")
EMBED_FALLBACK_MODEL = os.environ.get("AITHER_DECIDE_EMBED_FALLBACK_MODEL", "nomic-embed-text")


def embedder_floor(model: str) -> float:
    """The neighbor floor for `model`. An explicit env override wins; an embedder
    nobody measured gets the STRICTEST known floor, never the loosest."""
    if os.environ.get("AITHER_DECIDE_NEIGHBOR_MIN_SIM"):
        return NEIGHBOR_MIN_SIM
    return EMBEDDER_FLOORS.get(model, max(EMBEDDER_FLOORS.values()))
NEIGHBOR_K = int(os.environ.get("AITHER_DECIDE_NEIGHBOR_K", "5"))

# The neural rung (2026-09-20, gap G1): a cold answer that is neither a table
# lookup nor a 2 s LLM call. One linear (logistic) head per KIND over a vector for
# "<domain> || <state> || <question> || <option>", trained by
# tools/train_neural_rung.py on every journaled outcome plus the labelled judge
# cases, temperature-calibrated on a FORK-held-out split. It answers only when its
# calibrated top probability clears NEURAL_MIN; otherwise the ladder falls to the
# LLM exactly as before. Missing model file = rung inactive, never an error.
NEURAL_MIN = float(os.environ.get("AITHER_DECIDE_NEURAL_MIN", "0.75"))
# What a head must SCORE on held-out forks, at NEURAL_MIN, before it is allowed
# to serve. Measured 2026-09-20 and this is why the gate exists: on 75 held-out
# judge forks the first trained head answered 27% of them at 75.0% accuracy,
# while the LLM rung it would have preempted scored 96.0% on the same forks. A
# rung that is faster and better calibrated but less accurate must not take that
# work silently -- so the trainer writes an unpromoted head beside the serving
# path and says so, and the door keeps calling the brain.
NEURAL_PROMOTE_MIN_ACC = float(os.environ.get("AITHER_DECIDE_NEURAL_PROMOTE_ACC", "0.90"))
# "can this be judged at all", NOT a quality gate -- the Wilson bound in
# neural_promotion is what refuses thin evidence, continuously and without a
# constant anyone has to re-derive when the corpus moves.
NEURAL_PROMOTE_MIN_FORKS = int(os.environ.get("AITHER_DECIDE_NEURAL_PROMOTE_FORKS", "20"))
NEURAL_MODEL_ENV = "AITHER_DECIDE_NEURAL_MODEL"
NEURAL_FILE = "neural-rung.npz"
NEURAL_HASHED_DIM = 4096
NEURAL_QUESTION_CHARS = 768
NEURAL_RELOAD_S = 60.0
try:  # numpy only -- no torch for this rung. Absent (selfhost image without
    import numpy as np  # WITH_TORCH) the rung is simply inactive.
except ImportError:  # pragma: no cover - environment dependent
    np = None  # type: ignore[assignment]


class DecideError(ValueError):
    """A malformed request. Mapped to 422 by the route."""


# --------------------------------------------------------------------------- LLM
# Where the internal CA lives, container by container (measured 2026-09-19): the
# world-model container mounts its chain at /certs/ca-chain.pem and sets no
# SSL_CERT_FILE; the solver sets SSL_CERT_FILE=/etc/tls/ca-bundle.pem. An env var
# wins; otherwise the first bundle that exists. No bundle = default trust, and the
# fleet's self-signed front door then fails LOUDLY (CERTIFICATE_VERIFY_FAILED).
_CA_CANDIDATES = (
    "/certs/ca-chain.pem",
    "/etc/tls/ca-bundle.pem",
    "/etc/aither/tls/ca-chain.pem",
    "/etc/ssl/certs/door-ca.pem",
)


def _ca_bundle() -> Optional[str]:
    for cand in (
        os.environ.get("AITHER_DECIDE_CA"),
        os.environ.get("SSL_CERT_FILE"),
        os.environ.get("REQUESTS_CA_BUNDLE"),
        *_CA_CANDIDATES,
    ):
        if cand and Path(cand).is_file():
            return cand
    return None


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    bundle = _ca_bundle()
    if bundle:
        ctx.load_verify_locations(bundle)
    return ctx


_SYSTEM_ONE = (
    "You are a decision function. Answer with ONE JSON object and nothing "
    'else: {"answer": <one of the allowed answers, verbatim>, '
    '"confidence": <0..1>, "why": <at most 12 words>}. No prose, no '
    "markdown, no thinking text."
)


def llm_chat_text(
    prompt: str,
    *,
    system: str = _SYSTEM_ONE,
    max_tokens: int = 120,
    url: str = LLM_URL,
    model: str = LLM_MODEL,
    timeout: float = LLM_TIMEOUT,
) -> str:
    """One local chat call; returns the assistant text with any <think> block
    stripped. Raises on any transport failure. The JSON shaping is the caller's
    (`llm_chat_json` for one answer, multi.py for one-context-many-questions)."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        url + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_KEY}"},
    )
    ctx = _ssl_context() if url.startswith("https") else None
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        doc = json.loads(r.read().decode("utf-8", "replace"))
    text = (doc.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    # Reasoning models wrap the answer; the caller takes the LAST {...} in the text.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S)


def llm_chat_json(
    prompt: str, *, url: str = LLM_URL, model: str = LLM_MODEL, timeout: float = LLM_TIMEOUT
) -> Dict[str, Any]:
    """One local chat call that must answer with a JSON object. Raises on any
    transport or parse failure -- the caller decides what ignorance means."""
    text = llm_chat_text(prompt, url=url, model=model, timeout=timeout)
    m = None
    for m in re.finditer(r"\{[^{}]*\}", text, flags=re.S):
        pass
    if m:
        return json.loads(m.group(0))
    # A model that ignores the JSON instruction and answers with the bare option
    # (measured live 2026-09-19: the 8B brain replied `reasoner`) is still an
    # answer; the caller checks it against the allowed set and drops the rest.
    bare = text.strip().strip("`'\"").strip()
    if bare and len(bare) <= 64 and "\n" not in bare:
        return {"answer": bare, "confidence": 0.5, "why": "bare answer, no JSON"}
    raise ValueError(f"no JSON object in LLM answer: {text[:120]!r}")


def embed_texts(
    texts: List[str],
    *,
    url: str = EMBED_URL,
    model: str = EMBED_MODEL,
    timeout: float = EMBED_TIMEOUT,
) -> List[List[float]]:
    """One /v1/embeddings call. Raises on any failure -- the neighbor rung is
    skipped, never faked, when the embedder is away."""
    if not url:
        raise RuntimeError("embedder disabled")
    # The same internal-CA context the LLM rung uses. Without it an https
    # embedder (MicroScheduler, the fleet's one model door) fails verification
    # and the rung goes dark with a message that reads like an outage.
    ctx = _ssl_context() if url.startswith("https") else None

    def _post(inp: Any) -> Dict[str, Any]:
        body: Dict[str, Any] = {"input": inp}
        if model:
            body["model"] = model
        req = urllib.request.Request(
            url + "/v1/embeddings",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    try:
        rows = sorted(_post(texts)["data"], key=lambda d: d.get("index", 0))
    except urllib.error.HTTPError as exc:
        # MicroScheduler takes `input` as ONE STRING and answers 422 to a list.
        # One request per text, in order. Any other status is a real failure.
        if exc.code != 422:
            raise
        rows = [_post(tx)["data"][0] for tx in texts]
    out = []
    for d in rows:
        v = d["embedding"]
        n = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / n for x in v])
    return out


def _cos(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def build_prompt(req: Dict[str, Any]) -> str:
    """The user prompt for one decision. `prompt_domain` is its inverse for the
    domain line; multi.py relies on that pair to route one shared LLM answer
    back to the question that asked (test_multi pins the round trip)."""
    opts = req["options"]
    if req["kind"] == "score" and opts is None:
        allowed = "a real number between 0 and 1"
    else:
        allowed = ", ".join(json.dumps(o) for o in opts)
    return (
        f"Fork: {req['domain']}\nState: {req['state']}\n"
        + (f"Question: {req['question']}\n" if req["question"] else "")
        + f"Kind: {req['kind']}\nAllowed answers: {allowed}\n"
        + "Pick the best answer for this state."
    )


_PROMPT_DOMAIN = re.compile(r"\AFork: (\S+)\n")


def prompt_domain(prompt: str) -> Optional[str]:
    """The domain a `build_prompt` prompt was built for, or None."""
    m = _PROMPT_DOMAIN.match(prompt or "")
    return m.group(1) if m else None


# ---------------------------------------------------------------- neural rung
# Two representations, ONE downstream shape (a vector the head dots with):
#   hashed  option-SALTED crossed features, signed hashing into 4096 dims:
#           state fields, field x field conjunctions, question words, question
#           word x field crosses, and a bag of the question's remaining words.
#           Deterministic across processes (crc32, never str.hash); no fleet needed.
#   fleet   the micro-embedder [e_ctx * e_opt, e_ctx, e_opt] (3 x 1024) CONCATENATED
#           with the hashed block -- the embedding adds semantics, the crosses keep
#           the conjunctions.
# Two things are non-negotiable and were measured (2026-09-20, 433 labelled judge
# forks, held out by state):
#   * salting by option: a plain bag of tokens is ADDITIVE in the option, so
#     z(yes) - z(no) would be the same constant for every state and a linear head
#     could never say "no" for a failing build and "yes" for a green one;
#   * crosses, not raw char n-grams: the label is a CONJUNCTION (criterion word x
#     output feature). n-grams of the raw output scored 0.63 held-out; the crosses
#     alone 0.76; crosses plus the output word-bag 0.77; adding n-grams back on
#     top of the crosses LOWERED it to 0.64.
_WORD_RE = re.compile(r"[a-z0-9_]+")
_FIELD_SPLIT = re.compile(r"[|,;]")
_STOP = frozenset(
    "the criterion and or is are was were a an of to in for on at be this that it by "
    "with as".split()
)


def neural_model_path() -> Path:
    override = os.environ.get(NEURAL_MODEL_ENV)
    if override:
        return Path(override)
    return Path(os.environ.get("AITHER_WM_CKPT_DIR", "/models/world-model")) / NEURAL_FILE


def neural_context(domain: str, state: str, question: str) -> str:
    """The text both trainer and rung see. The question is CAPPED so a judge
    prompt carrying 4000 chars of output cannot drown the 80-char state."""
    return f"{domain} || {state} || {(question or '')[:NEURAL_QUESTION_CHARS]}"


def neural_fork_key(kind: str, domain: str, state: str, question: str = "") -> str:
    """A fork is one DECISION: (kind, domain, state, question).

    It used to be (kind, domain, state), on the argument that "the question is
    context, not identity: two judge rows with the same feature-state and
    different raw outputs are the SAME fork and must land on the same side of
    the held-out split, or the split leaks." That argument is sound, and it is
    why this grain could not ship until the competence manifest did.

    What it cost: 3,494 raw rows collapsed to 390 forks and 780 deduped rows, so
    the learner saw 780 contexts and could not see more however large the corpus
    grew -- while `neural_context` was ALREADY handing the question to the
    featurizer. Dedupe was averaging away a distinction the model is equipped to
    learn. At this grain it sees 3,478.

    What it changes about the CLAIM: held-out forks now share states with
    training, which is the production repeat case (same feature bucket, new raw
    output) and NOT evidence for "answers a state nobody has seen". The report
    therefore breaks the served population into `state_seen` and `state_novel`
    -- measured 0.9962 vs 0.5606 before this landed, and one number over both
    is how a regression ships wearing a 0.96 headline.

    The question is TRUNCATED to the same cap `neural_context` uses, or two
    decisions the features cannot tell apart would become different forks and
    the split would stop meaning anything. An empty question leaves the key
    byte-identical to the old one, so journal and tool-event rows are untouched.
    """
    if not question:
        return f"{kind}|{domain}|{state}"
    q = zlib.crc32(question[:NEURAL_QUESTION_CHARS].encode("utf-8", "replace"))
    return f"{kind}|{domain}|{state}|q{q:08x}"


class HashedFeatures:
    name = "hashed"

    def __init__(self, dim: int = NEURAL_HASHED_DIM, max_words: int = 80) -> None:
        self.dim = int(dim)
        self.max_words = int(max_words)

    def _tokens(self, ctx: str) -> List[bytes]:
        domain, _, rest = ctx.partition(" || ")
        state, _, question = rest.partition(" || ")
        fields = [f.strip().lower() for f in _FIELD_SPLIT.split(state) if f.strip()]
        if len(fields) < 3:  # a free-text state: its words are the fields
            fields = list(dict.fromkeys(w for w in _WORD_RE.findall(state.lower()) if w not in _STOP))[:40]
        qline, _, qrest = question.partition("\n")
        qwords = [w for w in _WORD_RE.findall(qline.lower()) if len(w) > 2 and w not in _STOP]
        toks: List[str] = ["d:" + domain.lower()]
        toks.extend("q:" + w for w in qwords)
        for f in fields:
            toks.append("f:" + f)
            toks.extend("x:" + w + "*" + f for w in qwords)
        for a in range(len(fields)):
            for b in range(a + 1, len(fields)):
                toks.append("p:" + fields[a] + "*" + fields[b])
                toks.extend("xp:" + w + "*" + fields[a] + "*" + fields[b] for w in qwords)
        rest_words = dict.fromkeys(
            w
            for w in _WORD_RE.findall(qrest.lower())
            if len(w) > 2 and w not in _STOP and not w.isdigit()
        )
        toks.extend("o:" + w for w in list(rest_words)[: self.max_words])
        return [t.encode("utf-8", "replace") for t in toks]

    def featurize(self, pairs: List[Any]) -> Any:
        import zlib

        X = np.zeros((len(pairs), self.dim), dtype=np.float32)
        cache: Dict[str, List[bytes]] = {}
        for i, (ctx, opt) in enumerate(pairs):
            toks = cache.get(ctx)
            if toks is None:
                toks = cache[ctx] = self._tokens(ctx)
            salt = zlib.crc32(("opt:" + str(opt).strip().lower()).encode("utf-8", "replace"))
            hs = np.fromiter((zlib.crc32(t, salt) for t in toks), dtype=np.uint32, count=len(toks))
            hs = np.append(hs, np.uint32(salt))  # the option's own identity
            idx = (hs >> np.uint32(1)) % np.uint32(self.dim)
            sign = np.where(hs & np.uint32(1), 1.0, -1.0)
            row = np.bincount(idx, weights=sign, minlength=self.dim)
            row = np.sign(row) * np.log1p(np.abs(row))
            norm = float(np.linalg.norm(row)) or 1.0
            X[i] = row / norm
        return X


class FleetFeatures:
    name = "fleet"

    def __init__(self, embed: Callable[[List[str]], List[List[float]]], batch: int = 32) -> None:
        self._embed = embed
        self._batch = batch
        self._opt_cache: Dict[str, Any] = {}
        self._hashed = HashedFeatures()
        self.dim: Optional[int] = None

    def _vecs(self, texts: List[str]) -> List[Any]:
        out: List[Any] = []
        for i in range(0, len(texts), self._batch):
            for v in self._embed(texts[i : i + self._batch]):
                a = np.asarray(v, dtype=np.float32)
                out.append(a / (float(np.linalg.norm(a)) or 1.0))
        if len(out) != len(texts):
            raise RuntimeError(f"embedder returned {len(out)} vectors for {len(texts)} texts")
        return out

    def featurize(self, pairs: List[Any]) -> Any:
        ctxs = list(dict.fromkeys(c for c, _ in pairs))
        opts = [o for o in dict.fromkeys(str(o) for _, o in pairs) if o not in self._opt_cache]
        vecs = self._vecs(ctxs + [f"option: {o}" for o in opts])
        cvec = dict(zip(ctxs, vecs[: len(ctxs)]))
        for o, v in zip(opts, vecs[len(ctxs) :]):
            self._opt_cache[o] = v
        H = self._hashed.featurize(pairs)
        rows = []
        for (c, o), h in zip(pairs, H):
            ec, eo = cvec[c], self._opt_cache[str(o)]
            rows.append(np.concatenate([ec * eo, ec, eo, h]))
        X = np.stack(rows).astype(np.float32)
        self.dim = int(X.shape[1])
        return X


def _softmax(z: Any) -> Any:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class NeuralRung:
    """Heads keyed "<embedder>:<kind>" -> {w, b, T, rows}; `meta` says what they
    were trained on. `predict` returns None when no head fits the request."""

    def __init__(self, heads: Dict[str, Dict[str, Any]], meta: Dict[str, Any]) -> None:
        self.heads = heads
        self.meta = meta

    @staticmethod
    def family_of(domain: str) -> str:
        """`decide.judge.foo` -> `decide.judge`. The grain a head is fitted at."""
        parts = str(domain or "").split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "")

    @staticmethod
    def key(embedder: str, kind: str, family: str = "") -> str:
        """`hashed:yesno` pooled, `hashed:yesno:decide.judge` per-family.

        The empty family spells exactly what it always did, so an npz written
        before per-family heads existed still loads and still serves.
        """
        return f"{embedder}:{kind}:{family}" if family else f"{embedder}:{kind}"

    def has(self, embedder: str, kind: str, family: str = "") -> bool:
        return self.key(embedder, kind, family) in self.heads

    def resolve(self, embedder: str, kind: str, domain: str) -> Optional[str]:
        """The MOST SPECIFIC head that exists for this decision.

        Per-family first, pooled second, nothing third. Without this a new
        domain's rows dilute every head: measured, 85,009 tool rows took judge
        held-out accuracy 0.707 -> 0.533 in one pooled head.
        """
        fam = self.family_of(domain)
        k = self.key(embedder, kind, fam)
        if fam and k in self.heads:
            return k
        k = self.key(embedder, kind)
        return k if k in self.heads else None

    def accuracy(self, embedder: str, kind: str, domain: str = "") -> float:
        """Measured held-out accuracy of that head; -1 when it was never scored.
        The door picks the BETTER head, not the fancier one -- on this corpus
        the fleet embedding did not beat the hashed crosses and preferring it
        blindly would have shipped the worse number.

        Resolves family-first like predict() does: scoring the POOLED head while
        serving a per-family one compares the wrong two numbers, and once a
        pooled head can be refused it compares -1 to -1."""
        hk = self.resolve(embedder, kind, domain) if domain else self.key(embedder, kind)
        h = (self.heads.get(hk) if hk else None) or {}
        a = h.get("accuracy")
        return float(a) if isinstance(a, (int, float)) else -1.0

    @classmethod
    def load(cls, path: Path) -> Optional["NeuralRung"]:
        """None when the file is missing. A CORRUPT file is logged and also None:
        the rung goes inactive, the door keeps answering from the other rungs."""
        if np is None or not Path(path).is_file():
            return None
        try:
            with np.load(str(path), allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                heads: Dict[str, Dict[str, Any]] = {}
                for k in meta.get("heads", {}):
                    heads[k] = {
                        "w": np.asarray(z[k + ":w"], dtype=np.float32),
                        "b": float(z[k + ":b"]),
                        "T": float(z[k + ":T"]),
                        "rows": int(meta["heads"][k].get("rows_train", 0)),
                        "accuracy": meta["heads"][k].get("heldout_accuracy"),
                        # ABSENT = unguarded, which is what every npz written
                        # before the manifest existed looks like. Old models keep
                        # serving exactly as they did.
                        "domains": (
                            set(np.asarray(z[k + ":domains"], dtype=np.uint64).tolist())
                            if (k + ":domains") in z.files else None
                        ),
                    }
        except Exception as exc:  # noqa: BLE001 -- corrupt/foreign file: rung off, say so
            logger.warning("decide: neural rung file %s unusable (%s); rung inactive", path, exc)
            return None
        if not heads:
            return None
        return cls(heads, meta)

    def save(self, path: Path, only: Optional[Set[str]] = None) -> Path:
        """`only` writes just those heads -- the SERVING path takes the heads that
        individually earned it, never a failing head riding a sibling's pass.
        `meta["heads"]` is filtered to match so `load` stays consistent with the
        arrays. None writes everything, which is what the candidate path wants:
        a refused head must stay inspectable."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        heads = {k: v for k, v in self.heads.items() if only is None or k in only}
        meta = dict(self.meta)
        if only is not None:
            meta["heads"] = {k: v for k, v in (meta.get("heads") or {}).items() if k in only}
            meta["heads_refused"] = sorted(set(self.heads) - set(heads))
        arrays: Dict[str, Any] = {"meta": np.array(json.dumps(meta))}
        for k, h in heads.items():
            arrays[k + ":w"] = np.asarray(h["w"], dtype=np.float32)
            arrays[k + ":b"] = np.array(float(h["b"]))
            arrays[k + ":T"] = np.array(float(h["T"]))
            if h.get("domains") is not None:
                arrays[k + ":domains"] = np.asarray(h["domains"], dtype=np.uint64)
        tmp = path.with_suffix(".npz.tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp, path)
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path

    def logits(self, embedder: str, kind: str, X: Any) -> Any:
        h = self.heads[self.key(embedder, kind)]
        return X.astype(np.float64) @ h["w"].astype(np.float64) + h["b"]

    def predict(
        self,
        kind: str,
        domain: str,
        state: str,
        question: str,
        options: List[str],
        feats: Any,
        temperature: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        hk = self.resolve(feats.name, kind, domain)
        if hk is None or not options:
            return None
        h = self.heads[hk]
        # Outside its competence the rung DECLINES. `_neural_view` reads None as
        # "try the next representation, then fall to the LLM", so this needs no
        # ladder change -- it is the same contract the engine and neighbor rungs
        # already honour when they have no evidence. A NOVEL STATE inside a
        # trained domain is still answered: that is the rung's whole point, and
        # it measures 0.8889 there.
        known = h.get("domains")
        if known is not None and domain_fingerprint(domain) not in known:
            return None
        ctx = neural_context(domain, state, question)
        X = feats.featurize([(ctx, o) for o in options])
        if X.shape[1] != h["w"].shape[0]:
            raise RuntimeError(
                f"neural rung: {feats.name} vector is {X.shape[1]}-dim, head wants "
                f"{h['w'].shape[0]} (embedder changed under the model?)"
            )
        T = float(h["T"] if temperature is None else temperature)
        z = X.astype(np.float64) @ h["w"].astype(np.float64) + h["b"]
        p = _softmax(z / max(1e-6, T))
        i = int(np.argmax(p))
        return {
            "answer": options[i],
            "probability": round(float(p[i]), 4),
            "probabilities": {str(o): round(float(pp), 4) for o, pp in zip(options, p)},
            "embedder": feats.name,
            "learned_from": int(h["rows"]),
            "temperature": round(T, 4),
        }


# -- training (lives here, not in tools/, because the world-model container mounts
# THIS file and not the tools tree; tools/train_neural_rung.py is the CLI over it
# and POST /decide/neural/train is the in-container hook the nightly routine hits).
def _row(kind, domain, state, question, option, label, source, weight=1.0, n=1):
    return {
        "kind": kind,
        "domain": domain,
        "state": state,
        "question": question or "",
        "option": str(option),
        "label": float(max(0.0, min(1.0, label))),
        "weight": float(weight),
        "source": source,
        "n": int(n),
        "fork": neural_fork_key(kind, domain, state, question),
    }


def _kind_of(answers: List[str]) -> str:
    low = {a.strip().lower() for a in answers}
    if low and low <= {"yes", "no"}:
        return "yesno"
    try:
        for a in answers:
            float(a)
        return "score"
    except (TypeError, ValueError):
        return "choice"


def neural_rows_from_journals(
    decider: Any, exclude: str = r"\.coin(\.|$)", limit: int = 50000
) -> List[Dict[str, Any]]:
    """Every decide.* journal (Decider.dataset already reads them). The
    calibration coin is NOT training data -- it is the instrument."""
    ds = decider.dataset(None, limit=limit)
    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for r in ds.get("rows") or []:
        if re.search(exclude, r["domain"]):
            continue
        by_domain.setdefault(r["domain"], []).append(r)
    rows: List[Dict[str, Any]] = []
    for domain, recs in by_domain.items():
        kind = _kind_of([r["answer"] for r in recs])
        for r in recs:
            rows.append(
                _row(
                    kind,
                    domain,
                    r["state"],
                    "",
                    r["answer"],
                    (float(r["reward"]) + 1.0) / 2.0,
                    "journal",
                    n=int(r.get("n", 1)),
                )
            )
    return rows


def neural_rows_from_cases(path: Path, domain: str = "decide.judge") -> List[Dict[str, Any]]:
    """Labelled judge cases (tools/eval_cases/cases.jsonl, or the snapshot the
    trainer leaves in the checkpoint dir): {output, criteria: [{text, label}]}.
    Rows are built through judge.Judge._items so train text == inference text.
    A corrupt row RAISES (ValueError naming the line) -- a trainer that skips
    bad labels in silence trains on whatever is left and reports success."""
    import judge as judge_mod  # noqa: PLC0415 -- same dir, mounted with this file

    rows: List[Dict[str, Any]] = []
    path = Path(path)
    if not path.is_file():
        return rows
    j = judge_mod.Judge(None, domain)
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
                output = str(case["output"])
                crits = case["criteria"]
                if not isinstance(crits, list):
                    raise TypeError("criteria must be a list")
                for c in crits:
                    text, label = str(c["text"]), c["label"]
                    if not isinstance(label, bool):
                        raise TypeError(f"label must be true/false, got {label!r}")
                    it = j._items(output, [text], domain)[0]
                    for opt, lab in (("yes", label), ("no", not label)):
                        rows.append(
                            _row("yesno", it["domain"], it["state"], it["question"], opt,
                                 1.0 if lab else 0.0, "cases")
                        )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path.name}:{ln}: corrupt case row: {exc}") from exc
    return rows


def dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per (fork, option): label = the n-weighted MEAN of what was
    observed, weight = the total n. Two cases whose features collapse to the same
    state are the same question asked twice -- keeping both would weight that
    state double in training AND make the fork eval score a contradiction as if
    both answers were right. Measured 2026-09-20 on the harvested cases: 433 rows
    -> 353 forks, and only 4 of those 353 carry a contradictory label."""
    agg: Dict[Any, Dict[str, Any]] = {}
    for r in rows:
        k = (r["fork"], r["option"])
        cur = agg.get(k)
        if cur is None:
            agg[k] = dict(r)
            agg[k]["weight"] = float(r["weight"]) * max(1, int(r["n"]))
            agg[k]["_lsum"] = float(r["label"]) * agg[k]["weight"]
        else:
            w = float(r["weight"]) * max(1, int(r["n"]))
            cur["weight"] += w
            cur["_lsum"] += float(r["label"]) * w
            cur["n"] += int(r["n"])
    out = []
    for row in agg.values():
        w = row.pop("_lsum") / max(1e-9, row["weight"])
        row["label"] = float(w)
        out.append(row)
    return out


# A family needs this many of its own rows before it earns a head. Below it the
# pooled head is the better estimate; above it, pooling is what hurt.
FAMILY_HEAD_MIN_ROWS = 200


def _family_eval(ev, zev, T: float) -> Dict[str, Dict[str, Any]]:
    """`_fork_eval` again, once per domain family (`decide.judge`, `decide.tool`).

    A head trained on two populations reports one row-weighted accuracy. The
    family the promotion floor is about can regress while that average improves,
    so the average alone is not a verdict.
    """
    if not len(ev):
        return {}
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(ev):
        parts = str(r["domain"]).split(".")
        fam = ".".join(parts[:2]) if len(parts) >= 2 else (parts[0] or "?")
        groups.setdefault(fam, []).append(i)
    out: Dict[str, Dict[str, Any]] = {}
    for fam, idx in groups.items():
        sub = [ev[i] for i in idx]
        sub_z = zev[idx] if hasattr(zev, "__getitem__") else zev
        got = _fork_eval(sub, sub_z, T)
        out[fam] = {
            "rows": len(sub),
            "forks": got["forks"],
            "accuracy": got["accuracy"],
            "ece": got["ece"],
        }
    return out


def _fit_logistic(X, y, w8, l2: float, steps: int, lr: float):
    """Weighted L2-regularised logistic regression by Adam (deterministic, no
    torch). Returns (w, b, final weighted BCE)."""
    n, d = X.shape
    X = X.astype(np.float64)
    y = np.asarray(y, dtype=np.float64)
    w8 = np.asarray(w8, dtype=np.float64)
    w8 = w8 / max(1e-12, w8.sum())
    theta = np.zeros(d + 1)
    m = np.zeros(d + 1)
    v = np.zeros(d + 1)
    b1, b2, eps = 0.9, 0.999, 1e-8
    loss = float("nan")
    for t in range(1, steps + 1):
        z = X @ theta[:d] + theta[d]
        p = 1.0 / (1.0 + np.exp(-z))
        pc = np.clip(p, 1e-7, 1 - 1e-7)
        loss = float(-(w8 * (y * np.log(pc) + (1 - y) * np.log(1 - pc))).sum()) + 0.5 * l2 * float(
            theta[:d] @ theta[:d]
        )
        g = (p - y) * w8
        grad = np.concatenate([X.T @ g + l2 * theta[:d], [g.sum()]])
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad * grad
        theta -= lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
    return theta[:d], float(theta[d]), loss


def _ece(conf, hit, bins: int = 10) -> float:
    conf = np.asarray(conf, dtype=np.float64)
    hit = np.asarray(hit, dtype=np.float64)
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - hit[m].mean())
    return float(ece)


def domain_fingerprint(domain: str) -> int:
    """A stable 64-bit fingerprint of a domain's LAST component, for the
    competence manifest.

    The last component, not the whole domain: `judge.Judge._items` builds
    `f"{domain}.{slug(criterion)}"`, and the BASE varies per caller while the
    slug is the criterion itself. `judge_bench` uses a timestamped base
    (`decide.judge.bench1789952033.neural.the-command-succeeded`), so keying on
    the full string declined all 24 of its forks -- for a criterion the head had
    trained on 375 times. Measured 2026-09-20: on the live corpus 265 domains
    map 1:1 onto 265 last-components, so nothing is coarsened by this.

    blake2b, not hash(): Python's hash is salted per process, so a manifest
    written by the trainer would match nothing at serving time.
    """
    leaf = str(domain).rsplit(".", 1)[-1]
    return int.from_bytes(
        hashlib.blake2b(leaf.encode("utf-8", "replace"), digest_size=8).digest(),
        "big",
    )


def wilson_lower(right: int, n: int, z: float = 1.96) -> float:
    """Wilson score interval, lower bound. 0.0 on no evidence.

    A point estimate is not a floor: 0.90 observed on 20 forks and 0.9962 on 260
    are the same number to `acc >= 0.9` and completely different evidence. This
    makes the count floor statistical instead of hand-tuned, and it scales with
    the corpus without anyone re-deriving a constant.
    """
    if n <= 0:
        return 0.0
    p = right / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / d
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return max(0.0, centre - margin)


def _gated_eval(rows: List[Dict[str, Any]], z, T: float, threshold: float) -> Dict[str, Any]:
    """What the head would actually SERVE at `threshold`: how many held-out forks
    it would answer, and how often it would be right on those. Accuracy over all
    forks flatters a rung that abstains; this is the number the promotion gate
    and the operator both need."""
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["fork"], []).append(i)
    scored = answered = right = 0
    # What ALWAYS PICKING THE MOST COMMON OPTION would score on the forks the
    # head chose to answer. Without this an absolute accuracy floor is no floor
    # at all on a lopsided task: measured 2026-09-20, a head scored 0.9633 where
    # always-say-pass scores 0.9579, and it was promoted.
    majority_right = 0
    by_option: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        by_option.setdefault(str(r["option"]), []).append(i)
    option_rate = {
        o: float(np.mean([rows[i]["label"] > 0.5 for i in idx])) if idx else 0.0
        for o, idx in by_option.items()
    }
    # A fork whose options all sit at exactly 0.5 has no positive and is skipped
    # -- 2 of 390 on the current corpus. Counted now instead of vanishing, because
    # it shrinks forks_scored under the MIN_FORKS check without saying so.
    undecidable = 0
    for idx in groups.values():
        labels = np.array([rows[i]["label"] for i in idx])
        pos = labels > 0.5
        if len(idx) < 2:
            continue
        if not pos.any():
            undecidable += 1
            continue
        scored += 1
        p = _softmax(np.asarray([z[i] for i in idx]) / max(1e-6, T))
        top = int(np.argmax(p))
        if float(p[top]) >= threshold:
            answered += 1
            right += int(pos[top])
            # the option that is most often correct ACROSS the corpus, applied
            # blindly to this fork
            blind = max(idx, key=lambda i: option_rate.get(str(rows[i]["option"]), 0.0))
            majority_right += int(rows[blind]["label"] > 0.5)
    acc = (round(right / answered, 4) if answered else None)
    base = (round(majority_right / answered, 4) if answered else None)
    return {
        "threshold": threshold,
        "forks_scored": scored,
        "forks_answered": answered,
        # forks that could not be scored at all (every option at exactly 0.5)
        "forks_undecidable": undecidable,
        "coverage": (round(answered / scored, 4) if scored else None),
        "accuracy": acc,
        # the floor is applied to THIS, not to `accuracy`: 0.90 on 20 forks and
        # 0.9962 on 260 are the same point estimate and different evidence
        "accuracy_lower95": round(wilson_lower(right, answered), 4) if answered else None,
        "majority_baseline": base,
        "lift_over_majority": (round(acc - base, 4)
                               if acc is not None and base is not None else None),
    }


def _fork_eval(rows: List[Dict[str, Any]], z, T: float) -> Dict[str, Any]:
    """Fork-level: softmax over the fork's labelled options at temperature T; the
    reported probability is the top one, 'right' means the top option's label is
    positive. This is the number a reliability plot of `probability` measures."""
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["fork"], []).append(i)
    confs, hits, nll = [], [], []
    for idx in groups.values():
        labels = np.array([rows[i]["label"] for i in idx])
        pos = labels > 0.5
        if not pos.any() or len(idx) < 2:
            continue
        p = _softmax(np.asarray([z[i] for i in idx]) / max(1e-6, T))
        top = int(np.argmax(p))
        confs.append(float(p[top]))
        hits.append(1.0 if pos[top] else 0.0)
        nll.append(-float(np.log(max(1e-9, float(p[pos].sum())))))
    return {
        "forks": len(confs),
        "accuracy": (round(float(np.mean(hits)), 4) if hits else None),
        "ece": (round(_ece(confs, hits), 4) if confs else None),
        "nll": (round(float(np.mean(nll)), 4) if nll else None),
    }


def fit_neural_rung(
    rows: List[Dict[str, Any]],
    feature_sets: List[Any],
    *,
    holdout_frac: float = 0.2,
    forced_holdout: Optional[set] = None,
    l2: float = 1e-3,
    steps: int = 400,
    lr: float = 0.05,
    seed: int = 7,
) -> Any:
    """Train one head per (embedder, kind). Held-out split is BY FORK (crc32 of
    the fork key) so no state leaks between train and eval; `forced_holdout`
    forks always land in eval (the judge bench's own forks). Returns
    (NeuralRung, report)."""
    import zlib

    if np is None:
        raise RuntimeError("numpy is required to train the neural rung")
    if not rows:
        raise ValueError("no training rows")
    raw_rows = len(rows)
    rows = dedupe_rows(rows)
    forced = set(forced_holdout or ())
    cut = int(round(holdout_frac * 1000))

    def held(fork: str) -> bool:
        if fork in forced:
            return True
        return (zlib.crc32((str(seed) + "|" + fork).encode("utf-8")) % 1000) < cut

    report: Dict[str, Any] = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rows": len(rows),
        "rows_before_dedupe": raw_rows,
        "rows_per_source": {},
        "holdout_frac": holdout_frac,
        "forced_holdout_forks": len(forced),
        "embedders": [f.name for f in feature_sets],
        "heads": {},
    }
    for r in rows:
        report["rows_per_source"][r["source"]] = report["rows_per_source"].get(r["source"], 0) + 1
    heads: Dict[str, Dict[str, Any]] = {}
    kinds = sorted({r["kind"] for r in rows})
    # (kind, family) groups that carry enough of their own rows to be fitted
    # separately, plus the pooled ("") group that stays the fallback.
    fam_counts: Dict[Tuple[str, str], int] = {}
    for r in rows:
        fam_counts[(r["kind"], NeuralRung.family_of(r["domain"]))] = (
            fam_counts.get((r["kind"], NeuralRung.family_of(r["domain"])), 0) + 1
        )
    groups: List[Tuple[str, str]] = [(k, "") for k in kinds]
    for (kind, fam), n in sorted(fam_counts.items()):
        if fam and n >= FAMILY_HEAD_MIN_ROWS:
            groups.append((kind, fam))

    for feats in feature_sets:
        for kind, family in groups:
            krows = [r for r in rows
                     if r["kind"] == kind
                     and (not family or NeuralRung.family_of(r["domain"]) == family)]
            tr = [r for r in krows if not held(r["fork"])]
            ev = [r for r in krows if held(r["fork"])]
            key = NeuralRung.key(feats.name, kind, family)
            if len(tr) < 2 or len({r["label"] > 0.5 for r in tr}) < 2:
                report["heads"][key] = {
                    "skipped": f"{len(tr)} training rows / one class -- nothing to fit",
                    "rows_train": len(tr),
                    "rows_heldout": len(ev),
                }
                continue
            pairs = [
                (neural_context(r["domain"], r["state"], r["question"]), r["option"])
                for r in tr + ev
            ]
            X = feats.featurize(pairs)
            Xtr, Xev = X[: len(tr)], X[len(tr) :]
            ytr = np.array([r["label"] for r in tr])
            w, b, loss = _fit_logistic(Xtr, ytr, [r["weight"] for r in tr], l2, steps, lr)
            ztr = Xtr.astype(np.float64) @ w + b
            zev = Xev.astype(np.float64) @ w + b if len(ev) else np.zeros(0)
            # the held-out rows inside a domain this head was trained on --
            # the only ones predict() will answer once the manifest is in force
            _known_doms = {domain_fingerprint(r["domain"]) for r in tr}
            _keep = [i for i, r in enumerate(ev)
                     if domain_fingerprint(r["domain"]) in _known_doms]
            ev_served = [ev[i] for i in _keep]
            zev_served = zev[_keep] if len(ev) else zev
            _tr_states = {r["state"] for r in tr}
            _seen_idx = [i for i, r in enumerate(ev_served) if r["state"] in _tr_states]
            _novel_idx = [i for i, r in enumerate(ev_served) if r["state"] not in _tr_states]
            before = _fork_eval(ev, zev, 1.0)
            T, calibrated = 1.0, False
            if before["forks"]:
                grid = np.logspace(-1.3, 1.3, 53)
                nlls = [_fork_eval(ev, zev, float(t))["nll"] for t in grid]
                T = float(grid[int(np.argmin(nlls))])
                calibrated = True
            after = _fork_eval(ev, zev, T)
            pev = 1.0 / (1.0 + np.exp(-zev)) if len(ev) else np.zeros(0)
            yev = np.array([r["label"] for r in ev])
            # The competence manifest: every DOMAIN this head was trained on.
            # Measured 2026-09-20 on held-out forks -- inside a trained domain
            # the head reaches 0.8889 on states it has never seen; on a domain
            # it was never trained on it falls to 0.7368 against an LLM rung
            # that gets ~0.96 there. Declining the second is worth more than
            # answering it.
            trained_domains = np.array(
                sorted({domain_fingerprint(r["domain"]) for r in tr}), dtype=np.uint64
            )
            heads[key] = {"w": w.astype(np.float32), "b": b, "T": T, "rows": len(tr),
                          "domains": trained_domains}
            report["heads"][key] = {
                "rows_train": len(tr),
                "rows_heldout": len(ev),
                "forks_train": len({r["fork"] for r in tr}),
                "forks_heldout": len({r["fork"] for r in ev}),
                "train_loss": round(loss, 4),
                "train_accuracy_rows": round(float(np.mean((ztr > 0) == (ytr > 0.5))), 4),
                "heldout_forks_scored": before["forks"],
                "heldout_accuracy": before["accuracy"],
                "ece_before": before["ece"],
                "ece_after": after["ece"],
                "nll_before": before["nll"],
                "nll_after": after["nll"],
                "option_ece_raw": (round(_ece(pev, yev), 4) if len(ev) else None),
                "temperature": round(T, 4),
                "calibrated": calibrated,
                "dim": int(X.shape[1]),
                "too_small": before["forks"] < 20,
                # ONLY the held-out rows this head will actually serve: the
                # manifest makes predict() decline an untrained domain, so
                # scoring the declined ones here would gate on a population the
                # head never answers. Measured: 0.8889 on the domains it knows
                # vs 0.7857 over everything.
                "at_serving_threshold": _gated_eval(ev_served, zev_served, T, NEURAL_MIN),
                "at_serving_threshold_all": _gated_eval(ev, zev, T, NEURAL_MIN),
                "declined_share": (
                    round(1.0 - len(ev_served) / len(ev), 4) if len(ev) else None
                ),
                # With the question in the fork key, a held-out fork can share
                # its STATE with training. That is the production repeat case and
                # it is not evidence about novel states, so both are reported --
                # 0.9962 vs 0.5606 when this was first measured, and one number
                # over the two is how a regression ships with a 0.96 headline.
                "served_state_seen": _gated_eval(
                    [ev_served[i] for i in _seen_idx],
                    zev_served[_seen_idx] if len(ev_served) else zev_served,
                    T, NEURAL_MIN),
                "served_state_novel": _gated_eval(
                    [ev_served[i] for i in _novel_idx],
                    zev_served[_novel_idx] if len(ev_served) else zev_served,
                    T, NEURAL_MIN),
                "family_heldout": _family_eval(ev, zev, T),
            }
    meta = dict(report)
    meta["heads"] = {k: v for k, v in report["heads"].items() if "skipped" not in v}
    return NeuralRung(heads, meta), report


# How much a head must beat "always pick the most common option" by, on the forks
# it would actually answer. 2 points: small enough that a genuinely useful rung
# on a lopsided task can still clear it, large enough that noise cannot.
NEURAL_PROMOTE_MIN_LIFT = 0.02


def neural_promotion(report: Dict[str, Any], ood: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """May this model serve? A head earns the serving path only by answering at
    least NEURAL_PROMOTE_MIN_FORKS held-out forks at NEURAL_MIN with at least
    NEURAL_PROMOTE_MIN_ACC accuracy on the ones it answered. Anything else is
    written beside the serving path as a candidate and NAMED -- the door keeps
    the rung it had (or none), which is the safe direction."""
    reasons: List[str] = []
    best: Optional[str] = None
    # Per head, because promotion writes ONE file: without this a passing head
    # carries every failing sibling onto the serving path (measured 2026-09-20,
    # a decide.tool head would have carried a decide.judge head that had just
    # been refused).
    passed: Dict[str, bool] = {}
    for key, h in (report.get("heads") or {}).items():
        if "skipped" in h:
            continue
        passed[key] = False
        g = h.get("at_serving_threshold") or {}
        n, acc = g.get("forks_answered") or 0, g.get("accuracy")
        if n < NEURAL_PROMOTE_MIN_FORKS:
            reasons.append(
                f"{key}: would answer only {n} of {g.get('forks_scored')} held-out forks at "
                f"p>={NEURAL_MIN} (need {NEURAL_PROMOTE_MIN_FORKS} to judge it)"
            )
        elif (g.get("lift_over_majority") is not None
              and g["lift_over_majority"] < NEURAL_PROMOTE_MIN_LIFT):
            # An absolute floor under the base rate is not a floor. Measured
            # 2026-09-20: 0.9633 accuracy, 0.9579 majority baseline, +0.0054 of
            # actual skill, promoted for clearing 0.9.
            reasons.append(
                f"{key}: {acc} accuracy but always picking the most common option "
                f"scores {g.get('majority_baseline')} on the same forks -- lift "
                f"{g['lift_over_majority']:+.4f}, need {NEURAL_PROMOTE_MIN_LIFT:+.4f}"
            )
        elif acc is None or (g.get("accuracy_lower95") or 0.0) < NEURAL_PROMOTE_MIN_ACC:
            lo = g.get("accuracy_lower95")
            reasons.append(
                f"{key}: {acc} accuracy on the {n} forks it would answer, but the "
                f"95% lower bound is {lo} and the floor is {NEURAL_PROMOTE_MIN_ACC} "
                f"-- a point estimate on {n} forks is not evidence of "
                f"{NEURAL_PROMOTE_MIN_ACC}"
            )
        else:
            passed[key] = True
            best = key if best is None else best
    # An OUT-OF-DISTRIBUTION refusal overrides every in-corpus pass. Measured
    # 2026-09-20: a candidate cleared the lift check, the Wilson bound, the
    # competence manifest AND served-population scoring at 0.9965 -- and
    # judge_bench, whose forks appear nowhere in the corpus, measured the same
    # model at 0.6875 when answered (lower bound 0.4440). All four in-corpus
    # gates are computed on one corpus; they cannot see this.
    ood_block: Optional[str] = None
    if ood and ood.get("available") and ood.get("answered"):
        lo = ood.get("accuracy_lower95") or 0.0
        if lo < NEURAL_PROMOTE_MIN_ACC:
            ood_block = (
                f"out-of-distribution probe: {ood.get('accuracy')} accuracy on the "
                f"{ood.get('answered')} of {ood.get('items')} items it answered "
                f"({ood.get('source')}), 95% lower bound {lo} against a floor of "
                f"{NEURAL_PROMOTE_MIN_ACC} -- every in-corpus gate passed this model"
            )
            reasons.append(ood_block)
    return {
        "promote": best is not None and ood_block is None,
        "head": best if ood_block is None else None,
        "ood_blocked": ood_block,
        # what may be written to the SERVING path, and what may not
        "passed": passed,
        "refused": sorted(k for k, ok in passed.items() if not ok),
        "floor_accuracy": NEURAL_PROMOTE_MIN_ACC,
        "floor_applies_to": "accuracy_lower95 (Wilson 95% lower bound)",
        "floor_lift": NEURAL_PROMOTE_MIN_LIFT,
        "floor_forks": NEURAL_PROMOTE_MIN_FORKS,
        "serving_threshold": NEURAL_MIN,
        "reasons": reasons,
    }


def candidate_path(serving: Path) -> Path:
    return Path(serving).with_name(Path(serving).stem + ".candidate.npz")


def train_neural_rung_from_disk(
    decider: Any,
    *,
    cases_path: Optional[Path] = None,
    embed: Optional[Callable[[List[str]], List[List[float]]]] = None,
    out_path: Optional[Path] = None,
    forced_holdout: Optional[set] = None,
    **fit_kw: Any,
) -> Dict[str, Any]:
    """What the nightly hook runs inside the container: journals + the cases
    snapshot in the checkpoint dir -> fit -> neural-rung.npz. `embed` None or
    unreachable = hashed head only (recorded in the report, never faked)."""
    ckpt = Path(os.environ.get("AITHER_WM_CKPT_DIR", "/models/world-model"))
    cases = Path(cases_path) if cases_path else ckpt / "neural-rung.cases.jsonl"
    rows = neural_rows_from_journals(decider) + neural_rows_from_cases(cases)
    feats: List[Any] = [HashedFeatures()]
    embed_note = "not configured"
    if embed is not None:
        try:
            embed(["probe"])
            feats.append(FleetFeatures(embed))
            embed_note = "fleet embedder reachable"
        except Exception as exc:  # noqa: BLE001 -- say so, train hashed only
            embed_note = f"fleet embedder unreachable: {str(exc)[:120]}"
    rung, report = fit_neural_rung(rows, feats, forced_holdout=forced_holdout, **fit_kw)
    report["embedder_note"] = embed_note
    report["cases_path"] = str(cases)
    report["promotion"] = neural_promotion(report)
    serving = Path(out_path) if out_path else neural_model_path()
    if not rung.heads:
        report["saved"] = None
    elif report["promotion"]["promote"]:
        rung.meta.update({k: v for k, v in report.items() if k != "heads"})
        # same rule as the CLI: only the heads that individually earned it
        report["saved"] = str(rung.save(
            serving,
            only={k for k, ok in (report["promotion"].get("passed") or {}).items() if ok},
        ))
        report["promoted"] = True
    else:
        rung.meta.update({k: v for k, v in report.items() if k != "heads"})
        report["saved"] = str(rung.save(candidate_path(serving)))
        report["promoted"] = False
    return report


# ------------------------------------------------------------------- the decider
class Decider:
    def __init__(
        self,
        domains: Any,
        *,
        llm: Optional[Callable[[str], Dict[str, Any]]] = None,
        record_dir: Optional[Path] = None,
        llm_enabled: bool = True,
        embed: Optional[Callable[[List[str]], List[List[float]]]] = None,
        embed_enabled: bool = True,
        neural_model: Optional[Path] = None,
        neural_enabled: bool = True,
    ) -> None:
        """`domains` is a code_domains.DomainEngines (or anything with `_engine`,
        `observe`). `llm` is injectable for tests; None = the local brain.
        `neural_model` is the trained head file (default: $AITHER_DECIDE_NEURAL_MODEL
        or <ckpt>/neural-rung.npz); missing = the neural rung is inactive."""
        self._domains = domains
        self._neural_path = Path(neural_model) if neural_model else None
        self._neural_enabled = bool(neural_enabled) and np is not None
        self._neural: Optional[NeuralRung] = None
        self._neural_mtime: Optional[float] = None
        self._neural_checked = 0.0
        self._hashed = HashedFeatures() if np is not None else None
        self._fleet_feats: Optional[FleetFeatures] = None
        self._llm = llm if llm is not None else llm_chat_json
        self._llm_enabled = llm_enabled
        self._record_dir = record_dir
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._stats: Dict[str, Dict[str, Any]] = {}
        self._pool = ThreadPoolExecutor(max_workers=_BATCH_WORKERS, thread_name_prefix="decide")
        # neighbor rung: per-domain {state: unit vector} for states the engine has
        # evidence on. Filled as states are learned (outcome) and lazily on lookup.
        self._embed = embed if embed is not None else self._routed_embed
        self._embed_enabled = embed_enabled and bool(
            EMBED_URL or EMBED_FALLBACK_URL or embed is not None)
        self._vectors: Dict[str, Dict[str, List[float]]] = {}
        self._embed_down_until = 0.0
        # ONE embedder per process, fixed at the first successful call. Vectors
        # from two embedders live in different spaces; comparing them is noise
        # that looks like similarity.
        self._embed_route: Optional[Tuple[str, str]] = None
        self._min_sim = NEIGHBOR_MIN_SIM if embed is not None else embedder_floor(EMBED_MODEL)

    # -- request shaping --------------------------------------------------------
    @staticmethod
    def normalize(item: Dict[str, Any]) -> Dict[str, Any]:
        domain = str(item.get("domain") or "").strip()
        state = item.get("state")
        kind = str(item.get("kind") or "choice").strip().lower()
        if not domain.startswith("decide."):
            raise DecideError(
                "domain must start with 'decide.' (one name per fork; the "
                "descriptor you send as `state` must be STABLE for that fork)"
            )
        if not isinstance(state, str) or not state.strip():
            raise DecideError("state must be a non-empty descriptor string")
        if kind not in KINDS:
            raise DecideError(f"kind must be one of {KINDS}")
        options = item.get("options")
        if kind == "yesno":
            options = list(_YESNO) if not options else [str(o).strip().lower() for o in options]
            if sorted(options) != sorted(_YESNO):
                raise DecideError("yesno options are exactly ['yes', 'no']")
        elif kind == "score" and not options:
            options = None  # continuous 0..1
        else:
            if not isinstance(options, list) or len(options) < 2:
                raise DecideError("options must list at least two answers")
            options = [str(o) for o in options]
            if len(set(options)) != len(options):
                raise DecideError("options must be distinct")
        return {
            "domain": domain,
            "state": state.strip(),
            "kind": kind,
            "options": options,
            "question": str(item.get("question") or "").strip(),
            "min_confidence": float(item.get("min_confidence", 0.0) or 0.0),
            "learn": bool(item.get("learn", True)),
        }

    # -- the ladder ---------------------------------------------------------------
    def _engine_view(self, domain: str, state: str, options: List[str]) -> List[Dict[str, Any]]:
        """What the domain engine knows about each option in this state:
        [{answer, value, n}] with value=None when it has no prediction."""
        import hashlib

        eng = self._domains._engine(domain)  # raises ValueError for an unknown domain
        h = int(hashlib.sha256(state.encode("utf-8", "replace")).hexdigest()[:16], 16)
        rows = []
        for opt in options:
            pred = eng.predict(h, opt)
            n = 0
            recs = getattr(eng, "_transitions", {}).get((h, eng._action_key(opt)))
            if recs:
                n = sum(int(r.get("count", 1)) for r in recs)
            rows.append(
                {"answer": opt, "value": (None if pred is None else float(pred[1])), "n": n}
            )
        return rows

    # -- neighbor rung ------------------------------------------------------------
    def _routed_embed(self, texts: List[str]) -> List[List[float]]:
        """Embed through the process's ONE embedder, choosing it on first use:
        the configured primary, else the fleet model door. Never switches once
        chosen -- a second embedder's vectors are not comparable to the first's."""
        if self._embed_route is not None:
            url, model = self._embed_route
            return embed_texts(texts, url=url, model=model)
        last: Optional[Exception] = None
        for url, model in ((EMBED_URL, EMBED_MODEL), (EMBED_FALLBACK_URL, EMBED_FALLBACK_MODEL)):
            if not url:
                continue
            try:
                out = embed_texts(texts, url=url, model=model)
            except Exception as exc:  # noqa: BLE001 -- try the next route
                last = exc
                continue
            self._embed_route = (url, model)
            self._min_sim = embedder_floor(model)
            logger.info("decide: neighbor rung ON via %s (%s), floor %.2f",
                        model, url, self._min_sim)
            return out
        raise last if last is not None else RuntimeError("embedder disabled")

    def _vector(self, domain: str, state: str) -> Optional[List[float]]:
        vecs = self._vectors.setdefault(domain, {})
        v = vecs.get(state)
        if v is not None:
            # "Dead means unaddressed" (volotat/mini-AGI): a HIT moves the state to
            # the back, so the front of the dict is the least-recently-ASKED and
            # that is what eviction removes. Insertion order threw away the
            # hottest states first -- learned early, asked about constantly.
            with self._lock:
                if state in vecs:
                    vecs[state] = vecs.pop(state)
            return v
        if time.time() < self._embed_down_until:
            return None
        try:
            v = self._embed([state])[0]
        except Exception as exc:  # embedder away: skip the rung for 30 s, say so once
            logger.warning(
                "decide: embedder unavailable (%s); neighbor rung off for 30 s", str(exc)[:120]
            )
            self._embed_down_until = time.time() + 30.0
            return None
        with self._lock:
            if len(vecs) > 20_000:
                for k in list(vecs)[:5_000]:
                    vecs.pop(k, None)
            vecs[state] = v
        return v

    def _remember(self, domain: str, state: str) -> None:
        """A state that just received an outcome becomes a neighbor candidate."""
        if self._embed_enabled:
            self._vector(domain, state)

    def _neighbor_view(
        self, domain: str, state: str, options: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Evidence from the K nearest LEARNED states above the similarity floor:
        each neighbor's per-option table, weighted by similarity. Returns
        {rows, sim, neighbors} or None when nothing is near enough."""
        if not self._embed_enabled:
            return None
        known = self._known_states(domain)
        if not known:
            return None
        v = self._vector(domain, state)
        if v is None:
            return None
        vecs = self._vectors.get(domain, {})
        scored = []
        for s in known:
            if s == state:
                continue
            sv = vecs.get(s)
            if sv is None:
                sv = self._vector(domain, s)
                if sv is None:
                    continue
            sim = _cos(v, sv)
            if sim >= self._min_sim:
                scored.append((sim, s))
        if not scored:
            return None
        scored.sort(reverse=True)
        scored = scored[:NEIGHBOR_K]
        agg: Dict[str, List[float]] = {o: [] for o in options}
        weight: Dict[str, float] = {o: 0.0 for o in options}
        count: Dict[str, int] = {o: 0 for o in options}
        for sim, s in scored:
            for row in self._engine_view(domain, s, options):
                if row["value"] is None:
                    continue
                agg[row["answer"]].append(sim * row["value"])
                weight[row["answer"]] += sim
                count[row["answer"]] += row["n"]
        rows = [
            {"answer": o, "value": (sum(agg[o]) / weight[o] if weight[o] else None), "n": count[o]}
            for o in options
        ]
        return {"rows": rows, "sim": round(scored[0][0], 3), "neighbors": [s for _, s in scored]}

    # -- neural rung --------------------------------------------------------------
    def _neural_rung(self) -> Optional[NeuralRung]:
        """The trained heads, reloaded when the file's mtime changes (the nightly
        retrain lands without a restart). Missing file = None, never an error."""
        if not self._neural_enabled:
            return None
        now = time.time()
        if now - self._neural_checked < NEURAL_RELOAD_S:
            return self._neural
        self._neural_checked = now
        path = self._neural_path or neural_model_path()
        try:
            mtime = path.stat().st_mtime
        except OSError:
            self._neural, self._neural_mtime = None, None
            return None
        if mtime != self._neural_mtime:
            self._neural = NeuralRung.load(path)
            self._neural_mtime = mtime
            if self._neural:
                logger.info(
                    "decide: neural rung loaded %s (heads=%s)", path, sorted(self._neural.heads)
                )
        return self._neural

    def _neural_view(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calibrated head answer for this request, fleet embedding first (when
        the model has a fleet head and the embedder is up), hashed otherwise."""
        rung = self._neural_rung()
        if rung is None or req["options"] is None:
            return None
        kind = req["kind"]
        feats: List[Any] = []
        # resolve(), NOT has(): has() asks for the POOLED `embedder:kind` key, so
        # once save(only=...) can refuse the pooled head every surviving
        # per-family head would become unreachable even though predict() would
        # have found it. Measured risk introduced by the per-head save above --
        # both halves land together or the fix is a regression.
        dom = str(req.get("domain") or "")
        fleet_ok = (
            self._embed_enabled
            and rung.resolve("fleet", kind, dom) is not None
            and time.time() >= self._embed_down_until
        )
        # Better head first; hashed on a tie (it needs no network and is ~40x
        # faster). Whichever is second is the fallback if the first path fails.
        fleet_first = fleet_ok and rung.accuracy("fleet", kind, dom) > rung.accuracy(
            "hashed", kind, dom
        )
        if fleet_first:
            if self._fleet_feats is None:
                self._fleet_feats = FleetFeatures(self._embed)
            feats.append(self._fleet_feats)
        if self._hashed is not None and rung.resolve("hashed", kind, dom) is not None:
            feats.append(self._hashed)
        if fleet_ok and not fleet_first:
            if self._fleet_feats is None:
                self._fleet_feats = FleetFeatures(self._embed)
            feats.append(self._fleet_feats)
        for f in feats:
            try:
                return rung.predict(
                    kind, req["domain"], req["state"], req["question"], req["options"], f
                )
            except Exception as exc:  # noqa: BLE001 -- one path failing is not a crash
                if f.name == "fleet":
                    logger.warning(
                        "decide: fleet embedding for the neural rung failed (%s); hashed "
                        "path / llm for 30 s",
                        str(exc)[:120],
                    )
                    self._embed_down_until = time.time() + 30.0
                else:
                    logger.warning("decide: neural rung (%s) failed: %s", f.name, str(exc)[:160])
        return None

    def _known_states(self, domain: str) -> List[str]:
        eng = self._domains._engine(domain)
        cache = getattr(eng, "_state_desc_cache", {})
        return [s for s in cache.values() if isinstance(s, str)]

    def _prior(self, domain: str, options: List[str]) -> List[Dict[str, Any]]:
        """Domain-wide mean reward per option, ignoring state."""
        eng = self._domains._engine(domain)
        agg: Dict[str, List[float]] = {o: [] for o in options}
        for (_, akey), recs in getattr(eng, "_transitions", {}).items():
            for o in options:
                if akey == eng._action_key(o):
                    for r in recs:
                        agg[o].extend([float(r["reward"])] * int(r.get("count", 1)))
        return [
            {"answer": o, "value": (sum(v) / len(v) if v else None), "n": len(v)}
            for o, v in agg.items()
        ]

    @staticmethod
    def _pick(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        known = [r for r in rows if r["value"] is not None]
        if not known:
            return None
        known.sort(key=lambda r: (r["value"], r["n"]), reverse=True)
        best = known[0]
        n = best["n"]
        evidence = n / (n + _EVIDENCE_K)
        if len(known) > 1:
            span = max(abs(r["value"]) for r in known) or 1.0
            margin = min(1.0, max(0.0, (best["value"] - known[1]["value"]) / (2 * span)) + 0.5)
        else:
            margin = 0.75  # only one option ever tried here: partial evidence
        return {"answer": best["answer"], "confidence": round(evidence * margin, 3), "n": n}

    def _ask_llm(
        self, req: Dict[str, Any], llm: Optional[Callable[[str], Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """`llm` overrides the instance's LLM rung for this one request (multi.py
        hands every question of one context the SAME shared callable, so the
        brain is prompted once per context, not once per question)."""
        if not self._llm_enabled:
            return None
        opts = req["options"]
        prompt = build_prompt(req)
        try:
            doc = (llm or self._llm)(prompt)
        except Exception as exc:  # transport, 5xx, parse -- all mean "no LLM answer"
            logger.warning("decide: llm unavailable for %s: %s", req["domain"], str(exc)[:160])
            return None
        if not isinstance(doc, dict):
            # A caller-supplied llm may return None (it could not answer) or a bare
            # string. Neither is a crash: the ladder falls through to prior. This
            # took the judge bench down with AttributeError on NoneType.
            logger.warning(
                "decide: llm returned %s, not a dict -- falling through", type(doc).__name__
            )
            return None
        ans = doc.get("answer")
        conf = doc.get("confidence", 0.5)
        try:
            conf = max(0.0, min(1.0, float(conf)))
        except (TypeError, ValueError):
            conf = 0.5
        if req["kind"] == "score" and opts is None:
            try:
                return {"answer": max(0.0, min(1.0, float(ans))), "confidence": conf}
            except (TypeError, ValueError):
                return None
        ans = str(ans).strip()
        if ans not in opts:
            low = {o.lower(): o for o in opts}
            if ans.lower() in low:
                ans = low[ans.lower()]
            else:
                logger.warning("decide: llm answered outside the options (%r)", ans[:60])
                return None
        return {"answer": ans, "confidence": conf}

    # -- public -----------------------------------------------------------------
    def decide(
        self, item: Dict[str, Any], *, llm: Optional[Callable[[str], Any]] = None
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        req = self.normalize(item)
        options = req["options"]
        out: Dict[str, Any] = {
            "domain": req["domain"],
            "kind": req["kind"],
            "source": "none",
            "answer": None,
            "confidence": 0.0,
            "learned_from": 0,
            "alternatives": [],
        }
        # 1. engine
        if options is not None:
            rows = self._engine_view(req["domain"], req["state"], options)
            out["alternatives"] = rows
            pick = self._pick(rows)
            if pick and pick["confidence"] >= req["min_confidence"]:
                out.update(
                    answer=pick["answer"],
                    confidence=pick["confidence"],
                    source="engine",
                    learned_from=pick["n"],
                )
        # 1b. neighbor: a state LIKE one we learned (micro-embedder), before any model
        if out["answer"] is None and options is not None:
            nb = self._neighbor_view(req["domain"], req["state"], options)
            if nb:
                pick = self._pick(nb["rows"])
                if pick:
                    # scale by how near the nearest neighbor is: at the floor ~0.5x,
                    # at identity 1.0x -- a near-duplicate is almost the same evidence
                    span = max(1e-6, 1.0 - self._min_sim)
                    scale = 0.5 + 0.5 * min(1.0, (nb["sim"] - self._min_sim) / span)
                    conf = round(pick["confidence"] * scale, 3)
                    if conf >= req["min_confidence"]:
                        out.update(
                            answer=pick["answer"],
                            confidence=conf,
                            source="neighbor",
                            learned_from=pick["n"],
                            neighbors=nb["neighbors"],
                            similarity=nb["sim"],
                        )
                        out["alternatives"] = nb["rows"]
        # 1c. neural: a calibrated cross-fork head for a state nobody has seen --
        # tens of milliseconds, a probability that was measured on held-out forks.
        # It answers only above NEURAL_MIN (and the caller's min_confidence);
        # below that the LLM is asked exactly as before.
        neural: Optional[Dict[str, Any]] = None
        if out["answer"] is None and options is not None:
            neural = self._neural_view(req)
            if neural and neural["probability"] >= max(NEURAL_MIN, req["min_confidence"]):
                out.update(
                    answer=neural["answer"],
                    confidence=neural["probability"],
                    source="neural",
                    learned_from=neural["learned_from"],
                    neural_embedder=neural["embedder"],
                )
            else:
                neural = None
        # 2. llm
        if out["answer"] is None:
            got = self._ask_llm(req, llm)
            if got is not None:
                out.update(answer=got["answer"], confidence=got["confidence"], source="llm")
        # 3. prior
        if out["answer"] is None and options is not None:
            pick = self._pick(self._prior(req["domain"], options))
            if pick:
                out.update(
                    answer=pick["answer"],
                    confidence=round(pick["confidence"] * 0.5, 3),
                    source="prior",
                    learned_from=pick["n"],
                )
        # 4. none
        if out["answer"] is None:
            out.update(answer=(options[0] if options else 0.5), confidence=0.0, source="none")
        # Calibrated probabilities, separate from confidence. `probability` is
        # P(this answer is right) as the OUTCOMES measured it: an option's mean
        # reward v in [-1, 1] over what we were told maps to (v + 1) / 2, so a
        # 60/40 coin reads 0.60/0.40 once it has been flipped enough -- it does
        # not collapse to the winner. `confidence` stays the evidence strength.
        # An LLM-sourced answer carries the model's STATED number, flagged as such.
        probs: Dict[str, float] = {}
        for row in out.get("alternatives") or []:
            if row.get("value") is not None:
                probs[str(row["answer"])] = round(
                    max(0.0, min(1.0, (float(row["value"]) + 1.0) / 2.0)), 3
                )
        if out["source"] in ("engine", "neighbor", "prior") and probs:
            out["probabilities"] = probs
            out["probability"] = probs.get(str(out["answer"]))
            out["probability_source"] = "outcomes"
        elif out["source"] == "neural" and neural is not None:
            probs = dict(neural["probabilities"])
            out["probabilities"] = probs
            out["probability"] = neural["probability"]
            out["probability_source"] = "neural (temperature-calibrated on held-out forks)"
        elif out["source"] == "llm":
            out["probability"] = out["confidence"]
            out["probability_source"] = "llm-stated (uncalibrated)"
        else:
            out["probability"] = None
            out["probability_source"] = "none"
        if req["kind"] == "yesno":
            p_yes = probs.get("yes")
            if p_yes is None and "no" in probs:
                p_yes = round(1.0 - probs["no"], 3)
            if p_yes is None and out["source"] == "llm":
                p_yes = (
                    out["confidence"]
                    if out["answer"] == "yes"
                    else round(1.0 - out["confidence"], 3)
                )
            out["p_yes"] = p_yes
        out["decision_id"] = self._record(req, out)
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        self._count(req["domain"], out["source"])
        return out

    def decide_batch(
        self, items: List[Dict[str, Any]], *, llm: Optional[Callable[[str], Any]] = None
    ) -> Dict[str, Any]:
        """Engine hits answer inline; LLM misses fan out across the worker pool so a
        batch of 13 cold questions costs one round trip, not thirteen. `llm`
        overrides the LLM rung for every item of this batch (see _ask_llm)."""
        t0 = time.perf_counter()
        if not isinstance(items, list) or not items:
            raise DecideError("items must be a non-empty list")
        if len(items) > 64:
            raise DecideError("at most 64 items per batch")
        answers = list(self._pool.map(lambda it: self.decide(it, llm=llm), items))
        return {
            "answers": answers,
            "count": len(answers),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def outcome(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Teach the engine: (state, answer) -> reward. By decision_id, or explicit."""
        try:
            reward = float(body.get("reward"))
        except (TypeError, ValueError):
            raise DecideError("reward must be a number (-1..1 recommended)")
        did = body.get("decision_id")
        with self._lock:
            rec = self._pending.pop(str(did), None) if did else None
        if rec is None:
            domain, state, answer = body.get("domain"), body.get("state"), body.get("answer")
            if not (domain and isinstance(state, str) and answer is not None):
                raise DecideError("unknown decision_id; pass domain+state+answer explicitly")
            rec = {"domain": str(domain), "state": state, "answer": str(answer)}
        # What did the engine expect for this (state, answer) BEFORE the outcome?
        # |reward - expectation| / 2 is this decision's surprise; AitherEvolution
        # schedules exploration from the domain's EMA of it (via /domain/status).
        predicted = None
        try:
            for row in self._engine_view(rec["domain"], rec["state"], [str(rec["answer"])]):
                predicted = row["value"]
        except Exception:  # noqa: BLE001 -- a surprise is a hint, never a failure
            predicted = None
        res = self._domains.observe(
            rec["domain"], rec["state"], str(rec["answer"]), rec["state"], reward, False
        )
        surprise = None
        if hasattr(self._domains, "note_surprise"):
            err = 1.0 if predicted is None else abs(reward - float(predicted)) / 2.0
            surprise = self._domains.note_surprise(rec["domain"], err)
        self._remember(rec["domain"], rec["state"])
        st = self._stats.setdefault(rec["domain"], self._blank())
        with self._lock:
            st["outcomes"] += 1
            st["reward_sum"] += reward
        self._journal(
            {
                "kind": "outcome",
                "decision_id": did,
                "domain": rec["domain"],
                "answer": rec["answer"],
                "reward": reward,
                "ts": time.time(),
            }
        )
        return {
            "ok": True,
            "domain": rec["domain"],
            "answer": rec["answer"],
            "reward": reward,
            "observed": res.get("observed"),
            "mode": res.get("mode"),
            "predicted": predicted,
            "surprise_ema": surprise,
        }

    def dataset(self, domain: Optional[str] = None, limit: int = 5000) -> Dict[str, Any]:
        """Training rows for the neural rung: one row per (state, answer) the
        engine has evidence on -- {domain, state, answer, reward, n}. IntentNanoGPT
        / nanobrain / a policy head train on this; the tabular store is the label
        source, so the rows are exactly what the door would answer from evidence."""
        rows: List[Dict[str, Any]] = []
        if domain:
            domains = [domain]
        else:
            # Loaded engines PLUS every fork with a journal on disk: domains replay
            # lazily, so a trainer asking a freshly restarted door would otherwise
            # get zero rows while the evidence sits in the checkpoint volume.
            domains = list(getattr(self._domains, "_engines", {}).keys())
            for name in self._journaled_domains():
                if name not in domains:
                    domains.append(name)
        for d in domains:
            try:
                eng = self._domains._engine(d)
            except Exception:  # noqa: BLE001 -- unknown domain: no rows, not a crash
                continue
            cache = getattr(eng, "_state_desc_cache", {})
            for (h, akey), recs in getattr(eng, "_transitions", {}).items():
                state = cache.get(h)
                if not isinstance(state, str):
                    continue
                n = sum(int(r.get("count", 1)) for r in recs)
                reward = sum(float(r["reward"]) * int(r.get("count", 1)) for r in recs) / max(1, n)
                rows.append(
                    {
                        "domain": d,
                        "state": state,
                        "answer": str(akey),
                        "reward": round(reward, 4),
                        "n": n,
                    }
                )
                if len(rows) >= limit:
                    break
        return {"rows": rows, "count": len(rows), "domains": domains}

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            out = {}
            for d, s in self._stats.items():
                n = s["decisions"] or 1
                out[d] = {
                    "decisions": s["decisions"],
                    "engine_pct": round(100.0 * s["engine"] / n, 1),
                    "neighbor_pct": round(100.0 * s.get("neighbor", 0) / n, 1),
                    "neural_pct": round(100.0 * s.get("neural", 0) / n, 1),
                    "llm_pct": round(100.0 * s["llm"] / n, 1),
                    "prior_pct": round(100.0 * s["prior"] / n, 1),
                    "none_pct": round(100.0 * s["none"] / n, 1),
                    "outcomes": s["outcomes"],
                    "mean_reward": (
                        round(s["reward_sum"] / s["outcomes"], 3) if s["outcomes"] else None
                    ),
                }
            return {
                "domains": out,
                "pending": len(self._pending),
                # What this PROCESS has served is not what the door has LEARNED:
                # domains replay from their journal lazily, on the first decision
                # at that fork. After a restart `domains` is {} while the evidence
                # sits on disk, which reads as "the door forgot" and it did not.
                "learned_on_disk": self._journaled_domains(),
            }

    @staticmethod
    def _journaled_domains() -> Dict[str, int]:
        """{domain: journaled rows} for every decide.* fork with a journal."""
        import os as _os

        ckpt = Path(_os.environ.get("AITHER_WM_CKPT_DIR", "/models/world-model"))
        found: Dict[str, int] = {}
        try:
            paths = sorted(ckpt.glob("domain-decide.*.transitions.jsonl"))
        except OSError:
            return found
        for path in paths[:200]:
            name = path.name[len("domain-") : -len(".transitions.jsonl")]
            try:
                with open(path, "rb") as fh:
                    found[name] = sum(1 for _ in fh)
            except OSError:
                found[name] = -1  # present but unreadable: never silently absent
        return found

    # -- bookkeeping ------------------------------------------------------------
    @staticmethod
    def _blank() -> Dict[str, Any]:
        return {
            "decisions": 0,
            "engine": 0,
            "neighbor": 0,
            "neural": 0,
            "llm": 0,
            "prior": 0,
            "none": 0,
            "outcomes": 0,
            "reward_sum": 0.0,
        }

    def _count(self, domain: str, source: str) -> None:
        with self._lock:
            st = self._stats.setdefault(domain, self._blank())
            st["decisions"] += 1
            st[source] = st.get(source, 0) + 1

    def _record(self, req: Dict[str, Any], out: Dict[str, Any]) -> str:
        did = uuid.uuid4().hex[:16]
        if req["learn"]:
            with self._lock:
                if len(self._pending) > 50_000:  # bounded: forgotten decisions expire
                    for k in list(self._pending)[:10_000]:
                        self._pending.pop(k, None)
                self._pending[did] = {
                    "domain": req["domain"],
                    "state": req["state"],
                    "answer": out["answer"],
                    "ts": time.time(),
                }
        self._journal(
            {
                "kind": "decision",
                "decision_id": did,
                "domain": req["domain"],
                "state": req["state"][:400],
                "answer": out["answer"],
                "confidence": out["confidence"],
                # The calibrated number, journaled next to the confidence so
                # tools/reliability.py can pair it with the outcome that follows.
                # Before 2026-09-20 only `confidence` was written; a reliability
                # table over those rows measures confidence, not probability.
                "probability": out.get("probability"),
                "probability_source": out.get("probability_source"),
                "question_kind": req.get("kind"),
                "source": out["source"],
                "ts": time.time(),
            }
        )
        return did

    def _journal(self, row: Dict[str, Any]) -> None:
        if not self._record_dir:
            return
        try:
            self._record_dir.mkdir(parents=True, exist_ok=True)
            with open(self._record_dir / "decisions.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError as exc:
            logger.warning("decide: journal append failed: %s", exc)
