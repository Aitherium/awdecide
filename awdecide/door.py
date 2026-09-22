"""DoorBackend -- the world-model decision door as a rung of the awdecide ladder.

Two decision planes existed on 2026-09-20: awdecide (this package: the typed
CONTRACT, fail-closed, with a Brier ledger) and the world model's "decision
door" (the awpredict server's `/decide`: a LEARNING ladder engine -> neighbor ->
neural -> llm -> prior -> none, with a probability calibrated from posted
outcomes). The aw-family rule forbids a parallel plane, so this module makes the
door a BACKEND of the contract: awdecide asks, the door answers with the number
it has earned, the outcome awdecide resolves is posted back so the door keeps
learning, and the same decision_id names the claim in both places.

Mapping (awdecide -> door):

    Question.choice(options)   kind=choice, options
    Question.score(levels)     kind=score,  options=levels   (the door's scale points)
    Question.bool()            kind=yesno,  options=[yes, no]

The door's `probability` (P(this answer is right) as the OUTCOMES measured it;
`probability_source` says whether it came from outcomes or is an LLM's stated,
uncalibrated number) becomes Decision.probability. `backend` is
`door:<source>` -- door:engine, door:neighbor, door:neural, door:llm,
door:prior. Source `none` is the door saying it has nothing to go on; that is
an ABSTENTION here (decided=False), never the door's placeholder first option.

Transport, in order:
  * `decider=` an in-process `decide.Decider` handed in (tests; the service itself)
  * the service tree importable on this host (`AITHER_WM_SVC_DIR`, default
    D:/arc-agi-3/arc-world-model-svc) -> a private in-process Decider rooted at
    `ckpt_dir` (AITHER_WM_CKPT_DIR); the LLM and embedder rungs are left ON so the
    ladder is the service's ladder, and they abstain cleanly when unreachable
  * `url=` HTTP: a gateway base ending in `/v1` (Authorization: Bearer <token>,
    public namespace decide.pub.<you>.<fork>) or the service base
    (`/decide`, X-WM-Token on `/decide/outcome`). https verifies against the
    default trust plus AITHER_DECIDE_CA / SSL_CERT_FILE / REQUESTS_CA_BUNDLE --
    never verify=False.

Decision.probabilities carries the door's PER-OPTION calibrated P(right) for
every option it has evidence on (0.0 = no evidence, not "impossible"). It is
not a normalized distribution; `probability` is the graded number.
"""
from __future__ import annotations

import importlib
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contract import Decision, Question, undecided

DEFAULT_SVC_DIR = os.getenv("AITHER_WM_SVC_DIR", "D:/arc-agi-3/arc-world-model-svc")
#: The engine the package ships with, used when no service tree is on the machine.
VENDORED_DIR = Path(__file__).resolve().parent / "door_local"
DEFAULT_URL = os.getenv("AITHER_DECIDE_URL", "http://127.0.0.1:8299/v1")
_KIND_TO_DOOR = {"choice": "choice", "score": "score", "bool": "yesno"}
_DOOR_TO_KIND = {"choice": "choice", "score": "score", "yesno": "bool"}
SOURCES = ("engine", "neighbor", "neural", "llm", "prior")


def door_kind(kind: str) -> str:
    """awdecide kind -> door kind (bool -> yesno)."""
    return _KIND_TO_DOOR[kind]


def awdecide_kind(kind: str) -> str:
    """door kind -> awdecide kind (yesno -> bool); unknown kinds pass through."""
    return _DOOR_TO_KIND.get(str(kind), str(kind))


def fork_domain(fork: str) -> str:
    fork = (fork or "").strip()
    if not fork:
        raise ValueError("DoorBackend needs a fork name (one per decision site)")
    return fork if fork.startswith("decide.") else f"decide.{fork}"


def door_request(state: str, q: Question, domain: str, *, learn: bool = True) -> Dict[str, Any]:
    """The /decide body for one awdecide question. Pure; used by the parity test.

    min_confidence is NOT forwarded: the door thresholds its EVIDENCE STRENGTH
    (`confidence`) and falls through to weaker rungs below it, which would hide
    the engine's evidence from the contract. awdecide thresholds the calibrated
    `probability` itself (decision_from_door) and keeps the evidence.
    """
    return {
        "domain": domain,
        "state": state,
        "kind": door_kind(q.kind),
        "options": list(q.options),
        "question": q.prompt or "",
        "min_confidence": 0.0,
        "learn": bool(learn),
    }


def decision_from_door(q: Question, resp: Dict[str, Any], *, tag: str = "door") -> Decision:
    """Turn a door answer into a Decision. Pure; the parity test pins it.

    source none / no probability -> undecided (the door said it does not know).
    A probability below q.min_confidence -> undecided, evidence kept.
    """
    source = str(resp.get("source") or "none")
    backend = f"{tag}:{source}"
    did = str(resp.get("decision_id") or "")
    answer = resp.get("answer")
    p = resp.get("probability")
    raw = resp.get("probabilities") or {}
    probs: Dict[str, float] = {}
    for o in q.options:
        try:
            probs[o] = max(0.0, min(1.0, float(raw[o]))) if o in raw else 0.0
        except (TypeError, ValueError):
            probs[o] = 0.0
    if q.kind == "bool":
        # the door may know only one side; the other is its complement
        if "yes" in raw and "no" not in raw:
            probs["no"] = round(1.0 - probs["yes"], 4)
        elif "no" in raw and "yes" not in raw:
            probs["yes"] = round(1.0 - probs["no"], 4)
    if source == "none" or answer is None:
        d = undecided(q, backend, [f"{backend}: the door has nothing to go on (source=none)"])
        d.id = did
        return d
    value = str(answer)
    if value not in q.options:
        low = {o.lower(): o for o in q.options}
        if value.lower() in low:
            value = low[value.lower()]
        else:
            d = undecided(q, backend, [f"{backend}: answered {value!r}, not one of the options"])
            d.id = did
            return d
    try:
        p = float(p) if p is not None else None
    except (TypeError, ValueError):
        p = None
    if p is None:
        d = undecided(q, backend, [f"{backend}: answered {value!r} with no probability"])
        d.probabilities = probs
        d.id = did
        return d
    p = max(0.0, min(1.0, p))
    if q.kind == "bool" and value in probs:
        probs[value] = p
        probs["no" if value == "yes" else "yes"] = round(1.0 - p, 4)
    elif value in probs and probs[value] == 0.0:
        probs[value] = p
    reasons = [f"{backend}: probability_source={resp.get('probability_source') or '?'}"
               f" learned_from={resp.get('learned_from', 0)}"]
    if p < q.min_confidence:
        d = undecided(q, backend, reasons + [
            f"{backend}: probability {p:.3f} < min_confidence {q.min_confidence:.3f}"])
        d.probabilities = probs
        d.id = did
        return d
    return Decision(kind=q.kind, value=value, probability=p, probabilities=probs, decided=True,
                    backend=backend, reasons=reasons, id=did)


# ------------------------------------------------------------ in-process
def load_service(svc_dir: Optional[str] = None, ckpt_dir: Optional[str] = None):
    """Import the door's service tree and return (code_domains, decide) or raise.

    code_domains reads AITHER_WM_CKPT_DIR at import; asking for a different
    ckpt_dir than the one already loaded reloads it (module-global, so one
    ckpt dir per process at a time -- tests use one per phase, not two at once).
    """
    svc = Path(svc_dir or DEFAULT_SVC_DIR)
    vendored = False
    if not (svc / "decide.py").is_file() or not (svc / "code_domains.py").is_file():
        # No service tree on this machine: the package carries its own copy of the
        # engine (awdecide/door_local, written only by scripts/vendor_door.py), so
        # `pip install awdecide` is enough to run the door in-process. An explicit
        # svc_dir that does not exist is still an error -- the caller asked for it.
        if svc_dir is not None or not (VENDORED_DIR / "decide.py").is_file():
            raise ImportError(f"door service tree not at {svc} (set AITHER_WM_SVC_DIR)")
        svc, vendored = VENDORED_DIR, True
    if ckpt_dir:
        os.environ["AITHER_WM_CKPT_DIR"] = str(ckpt_dir)
    elif not os.environ.get("AITHER_WM_CKPT_DIR"):
        if not vendored:
            raise ImportError("AITHER_WM_CKPT_DIR is unset and no ckpt_dir given -- refusing to "
                              "journal into the service's default /models/world-model")
        # the vendored door journals under the user's home, never a service path
        home = Path.home() / ".awdecide" / "door"
        home.mkdir(parents=True, exist_ok=True)
        os.environ["AITHER_WM_CKPT_DIR"] = str(home)
    if str(svc) not in sys.path:
        sys.path.insert(0, str(svc))
    code_domains = importlib.import_module("code_domains")
    want = Path(os.environ["AITHER_WM_CKPT_DIR"])
    if Path(str(getattr(code_domains, "_CKPT_DIR", ""))) != want:
        code_domains = importlib.reload(code_domains)
    decide = importlib.import_module("decide")
    if not getattr(code_domains, "_MLP_OK", False):
        raise ImportError("world_model package not importable here (code_domains._MLP_OK False)")
    return code_domains, decide


def make_decider(svc_dir: Optional[str] = None, ckpt_dir: Optional[str] = None, **kw: Any):
    """A private in-process Decider journaling into ckpt_dir. kw -> decide.Decider."""
    code_domains, decide = load_service(svc_dir, ckpt_dir)
    record_dir = Path(os.environ["AITHER_WM_CKPT_DIR"])
    if Path(decide.__file__).resolve().parent == VENDORED_DIR:
        # The vendored door has no fleet behind it: a model rung or an embedder is
        # ON only when the user names one. Otherwise a cold fork reads `none` at
        # once instead of after a connection attempt to a server that is not there.
        kw.setdefault("llm_enabled", bool(os.environ.get("AITHER_DECIDE_LLM_URL")))
        kw.setdefault("embed_enabled", bool(os.environ.get("AITHER_DECIDE_EMBED_URL")))
    return decide.Decider(code_domains.DomainEngines(), record_dir=record_dir, **kw)


# ------------------------------------------------------------------ HTTP
def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    for cand in (os.environ.get("AITHER_DECIDE_CA"), os.environ.get("SSL_CERT_FILE"),
                 os.environ.get("REQUESTS_CA_BUNDLE")):
        if cand and Path(cand).is_file():
            ctx.load_verify_locations(cand)
            break
    return ctx


class DoorHTTP:
    """POST /decide and /decide/outcome over the wire, gateway or service base."""

    def __init__(self, url: str, token: str = "", timeout: float = 30.0) -> None:
        if not url:
            raise ValueError("DoorHTTP needs an explicit url")
        self.url = url.rstrip("/")
        self.gateway = self.url.endswith("/v1")
        self.token = token
        self.timeout = timeout

    def _post(self, path: str, body: Dict[str, Any], *, write: bool = False) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"  # gateway principal
            if write and not self.gateway:
                headers["X-WM-Token"] = self.token  # service write guard
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode("utf-8"),
                                     headers=headers, method="POST")
        ctx = _ssl_context() if self.url.startswith("https") else None
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=ctx) as r:  # noqa: S310
                return json.loads(r.read().decode("utf-8", "replace") or "{}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:160]
            raise RuntimeError(f"HTTP {e.code} from {path}: {detail}") from e
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            raise RuntimeError(f"{type(e).__name__}: {e}") from e

    def decide(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._post("/decide", body)

    def outcome(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._post("/decide/outcome", body, write=True)


# --------------------------------------------------------------- the rung
class DoorBackend:
    """A Ladder rung that answers through the world-model decision door.

    Implements `decide(state, q) -> Decision` (the rung owns its calibrated
    probability, so the Ladder takes the Decision as-is instead of
    renormalizing weights) and `weights()` for callers that only know that
    interface (per-option P(right), which the contract will normalize -- use
    the Ladder path to keep the door's number).
    """

    def __init__(self, url: Optional[str] = None, token: str = "", fork: str = "awdecide", *,
                 decider: Any = None, ckpt_dir: Optional[str] = None,
                 svc_dir: Optional[str] = None, timeout: float = 30.0,
                 learn: bool = True, prefer_inprocess: bool = True) -> None:
        self.domain = fork_domain(fork)
        self.name = "door"
        self.learn = learn
        self.transport = "none"
        self.transport_error = ""
        self._decider = decider
        self._http: Optional[DoorHTTP] = None
        if decider is not None:
            self.transport = "inprocess"
            return
        if url:
            self._http = DoorHTTP(url, token, timeout)
            self.transport = "gateway" if self._http.gateway else "http"
            return
        if prefer_inprocess:
            try:
                self._decider = make_decider(svc_dir, ckpt_dir)
                self.transport = "inprocess"
                return
            except Exception as e:  # noqa: BLE001 -- fall to the wire, say why
                self.transport_error = f"{type(e).__name__}: {e}"
        tok = (token or os.getenv("AITHER_DECIDE_TOKEN")
               or os.getenv("AITHER_WM_INTERNAL_TOKEN") or "")
        self._http = DoorHTTP(DEFAULT_URL, tok, timeout)
        self.transport = "gateway" if self._http.gateway else "http"

    # -- raw door calls ------------------------------------------------------
    def ask(self, body: Dict[str, Any]) -> Dict[str, Any]:
        if self._decider is not None:
            return self._decider.decide(body)
        assert self._http is not None
        return self._http.decide(body)

    def post_outcome(self, body: Dict[str, Any]) -> Dict[str, Any]:
        if self._decider is not None:
            return self._decider.outcome(body)
        assert self._http is not None
        return self._http.outcome(body)

    # -- the rung ------------------------------------------------------------
    def decide(self, state: str, q: Question) -> Decision:
        body = door_request(state, q, self.domain, learn=self.learn)
        try:
            resp = self.ask(body)
        except Exception as e:  # noqa: BLE001 -- a rung that failed is a rung that abstained
            return undecided(q, "door", [f"door: abstained ({self.transport}: {e})"])
        d = decision_from_door(q, resp)
        if self.transport_error and d.reasons:
            d.reasons.insert(0, f"door: in-process unavailable ({self.transport_error}); "
                                f"used {self.transport}")
        return d

    def weights(self, state: str, q: Question) -> Dict[str, float]:
        d = self.decide(state, q)
        return {} if not d.decided else {o: p for o, p in d.probabilities.items() if p > 0.0}

    # -- the loop back -------------------------------------------------------
    def resolve(self, decision_id: str, correct: bool, *, ledger: Any = None,
                reward: Optional[float] = None) -> Dict[str, Any]:
        """Post the outcome to the door (reward +1 right / -1 wrong, or `reward`)
        so it keeps learning; with `ledger`, resolve the same id there too.
        Returns {"door": <door reply or error>, "brier": <ledger brier or None>}."""
        r = float(reward) if reward is not None else (1.0 if correct else -1.0)
        out: Dict[str, Any] = {"decision_id": decision_id, "reward": r, "door": None,
                               "brier": None}
        try:
            out["door"] = self.post_outcome({"decision_id": decision_id, "reward": r})
        except Exception as e:  # noqa: BLE001 -- the ledger still resolves; the door is told next time
            out["door"] = {"error": f"{type(e).__name__}: {e}"}
        if ledger is not None:
            out["brier"] = ledger.resolve(decision_id, correct=correct)
        return out


def resolve_pairs_from_door(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pair `decision` and `outcome` journal rows (decisions.jsonl shape) by
    decision_id. Returns one dict per matched pair with the decision row's
    fields plus `reward` and `correct` (reward > 0). The FIRST outcome grades."""
    decisions: Dict[str, Dict[str, Any]] = {}
    outcomes: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        did = r.get("decision_id")
        if not did:
            continue
        if r.get("kind") == "decision":
            decisions[str(did)] = r
        elif r.get("kind") == "outcome":
            outcomes.setdefault(str(did), r)
    out = []
    for did, dec in decisions.items():
        o = outcomes.get(did)
        if o is None:
            continue
        try:
            reward = float(o.get("reward"))
        except (TypeError, ValueError):
            continue
        out.append({**dec, "reward": reward, "correct": reward > 0.0})
    return out
