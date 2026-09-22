"""code_domains — non-grid (embedding/descriptor) domain engines for the WM service.

The resident LeWM model speaks ARC grids. The world-model program (slice 2,
2026-08-01) adds DOMAIN engines so non-grid environments — code-world first —
can observe transitions and query surprise through the SAME service, instead
of inventing a parallel one. Engines are world_model.core.mlp.MLPWorldModel
instances (tabular -> hybrid -> neural over hashed descriptor embeddings),
imported from the canonical packages/world-model tree, which is bind-mounted
read-only into this container exactly like lewm.py — no copy, no fork.

Contract (mirrored by the consumers in landmark_map.py / codegraph core.py):
  observe:  {domain, obs, action, next_obs, reward?, done?}   obs = descriptor str
  surprise: {domain, items: [{id, obs, action, next_obs}, ...]}
            -> {surprises: {id: float|null}}  null = engine has no prediction yet
            (cold start / unseen state); the CALLER decides what ignorance means.

Durability (D-ref, closed 2026-08-01): every observe is journaled to
`_CKPT_DIR/domain-<name>.transitions.jsonl` (the same host bind mount the
neural autosaves use, so `docker volume` lifecycle commands cannot touch it)
and replayed into a fresh engine on first use after a restart — the tabular
store now SURVIVES restarts. The journal is tail-capped at
AITHER_WM_DOMAIN_JOURNAL_MAX lines (default 50k) at replay time. A journal
write failure is logged as ERROR once per domain (the in-memory observe
still succeeded; what is lost is durability, and silence would hide that).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("wm.domains")

# The package is mounted at /app/wm/world_model in the container (compose),
# and lives in the monorepo on the host (tests / parity checks).
# Host candidates: env override first, then the canonical deploy root (C:), then the
# old working tree (D:). 2026-09-19: only D: was named and it no longer holds the
# package on this box, so every host-side test of this module failed at import
# while the container (which mounts it) was fine -- a host-only blind spot.
_HOST_PKGS = [Path(p) for p in (os.environ.get("AITHER_WORLD_MODEL_PKG"),
                                r"C:\source\packages\world-model",
                                r"D:\source\packages\world-model",
                                "/mnt/c/source/packages/world-model") if p]
for cand in (Path(__file__).resolve().parent, *_HOST_PKGS):
    if (cand / "world_model").exists() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))

try:
    from world_model.core.mlp import MLPWorldModel
    _MLP_OK = True
except Exception as exc:  # loud degrade: /domain/* will 503, never fabricate
    MLPWorldModel = None  # type: ignore[assignment]
    _MLP_OK = False
    logger.error("code_domains: world_model package unavailable (%s) — "
                 "/domain endpoints will refuse loudly", exc)

_CKPT_DIR = Path(os.environ.get("AITHER_WM_CKPT_DIR", "/models/world-model"))

# The allowlist is a GUARD, not a ceiling. Every other knob in this file is
# already env-driven and every mechanism below (engine, journal, autosave,
# tabular->hybrid->neural escalation) is per-domain and generic, so nothing but
# this hardcoded pair stopped the platform teaching this model about its OTHER
# systems and graphs. It stays an ALLOWLIST rather than becoming open, because
# the real cost of a new domain is not registration — it is a STABLE DESCRIPTOR.
# Two callers writing different descriptors for the same state make the engine
# confidently wrong, and `surprise` is exactly the signal you cannot afford to
# have quietly wrong. So: opt in explicitly, one name at a time.
#
# Extend without editing code, by either:
#   * AITHER_WM_DOMAINS="infra,graph_code,graph_deploy"   (needs a recreate), or
#   * a `domains.allow` file in the ckpt dir, one name per line — that directory
#     is a host bind mount, so a RESTART picks it up with no compose change.
_DEFAULT_DOMAINS = {"code", "sandbox"}


def _extra_domains() -> set:
    names: set = set()
    for raw in os.environ.get("AITHER_WM_DOMAINS", "").split(","):
        if raw.strip():
            names.add(raw.strip())
    try:
        allow_file = _CKPT_DIR / "domains.allow"
        if allow_file.is_file():
            for line in allow_file.read_text(encoding="utf-8").splitlines():
                line = line.split("#", 1)[0].strip()
                if line:
                    names.add(line)
    except OSError as exc:
        # Loud: a domain the operator believed was registered would otherwise be
        # rejected as "unknown", which reads as a caller bug, not a missing file.
        logger.error("code_domains: could not read %s/domains.allow (%s) — "
                     "only env-declared domains are registered", _CKPT_DIR, exc)
    return names


_ALLOWED_DOMAINS = _DEFAULT_DOMAINS | _extra_domains()
_AUTOSAVE_EVERY = int(os.environ.get("AITHER_WM_DOMAIN_AUTOSAVE_EVERY", "200"))
_JOURNAL_MAX = int(os.environ.get("AITHER_WM_DOMAIN_JOURNAL_MAX", "50000"))


def _desc_hash(desc: str) -> int:
    """Stable state id for a descriptor (matches the MLP engine's hashing idiom)."""
    return int(hashlib.sha256(desc.encode("utf-8", "replace")).hexdigest()[:16], 16)


class DomainEngines:
    """Thread-safe registry of per-domain MLP engines behind the service routes."""

    def __init__(self) -> None:
        self._engines: Dict[str, Any] = {}
        self._observed: Dict[str, int] = {}
        # Rolling surprise EMA per domain (alpha 0.05), updated on every
        # non-None surprise a consumer queries. This is the scheduling signal
        # AitherEvolution reads from /domain/status: high EMA = the model does
        # not understand this domain yet = explore it next. None until the
        # first graded answer — ignorance is not a number here either.
        self._surprise_ema: Dict[str, Optional[float]] = {}
        self._journal_err_logged: Dict[str, bool] = {}
        self._lock = threading.Lock()

    @property
    def ok(self) -> bool:
        return _MLP_OK

    def _journal_path(self, domain: str) -> Path:
        return _CKPT_DIR / f"domain-{domain}.transitions.jsonl"

    def _replay_journal(self, domain: str, eng) -> int:
        """Rebuild the tabular store from the on-disk journal (D-ref).
        Tail-caps the journal to _JOURNAL_MAX lines (rewriting the file) so
        it cannot grow unbounded. Returns transitions replayed."""
        path = self._journal_path(domain)
        if not path.exists():
            return 0
        rows = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # torn tail line from a crash mid-append
        except OSError as exc:
            logger.error("code_domains: cannot read journal %s (%s) — "
                         "tabular store starts COLD", path, exc)
            return 0
        if len(rows) > _JOURNAL_MAX:
            rows = rows[-_JOURNAL_MAX:]
            try:
                tmp = path.with_suffix(".jsonl.tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                os.replace(tmp, path)
                logger.info("code_domains: journal %s tail-capped to %d rows",
                            path, _JOURNAL_MAX)
            except OSError as exc:
                logger.warning("code_domains: journal cap rewrite failed: %s", exc)
        n = 0
        for r in rows:
            obs, next_obs = r.get("obs"), r.get("next_obs")
            if not (isinstance(obs, str) and isinstance(next_obs, str)):
                continue
            eng.observe(_desc_hash(obs), r.get("action"), _desc_hash(next_obs),
                        float(r.get("reward", 0.0)), bool(r.get("done", False)),
                        state_desc=obs, next_state_desc=next_obs)
            n += 1
        return n

    def _engine(self, domain: str):
        if not _MLP_OK:
            raise RuntimeError("world_model package unavailable in this container")
        # `decide.<fork>` names are open by construction (2026-09-19): the decision
        # door namespaces one domain per fork, and the descriptor-stability burden
        # sits with the fork that owns it. Everything else stays on the allowlist.
        if domain not in _ALLOWED_DOMAINS and not (
                domain.startswith("decide.") and len(domain) > 7 and domain.replace(
                    ".", "").replace("_", "").replace("-", "").isalnum()):
            raise ValueError(f"unknown domain {domain!r}; allowed: {sorted(_ALLOWED_DOMAINS)} "
                             f"or any decide.<fork>")
        with self._lock:
            eng = self._engines.get(domain)
            if eng is None:
                eng = MLPWorldModel()
                ckpt = _CKPT_DIR / f"domain-{domain}.pt"
                if ckpt.exists() and eng.load(ckpt):
                    logger.info("code_domains: loaded neural weights for %s from %s",
                                domain, ckpt)
                replayed = self._replay_journal(domain, eng)
                if replayed:
                    logger.info("code_domains: replayed %d journaled transitions "
                                "for %s — tabular store is WARM after restart",
                                replayed, domain)
                self._engines[domain] = eng
                self._observed[domain] = replayed
            return eng

    def observe(self, domain: str, obs: str, action: str, next_obs: str,
                reward: float = 0.0, done: bool = False) -> Dict[str, Any]:
        eng = self._engine(domain)
        with self._lock:
            eng.observe(_desc_hash(obs), action, _desc_hash(next_obs), reward, done,
                        state_desc=obs, next_state_desc=next_obs)
            self._observed[domain] += 1
            count = self._observed[domain]
            journaled = True
            try:
                _CKPT_DIR.mkdir(parents=True, exist_ok=True)
                with open(self._journal_path(domain), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"obs": obs, "action": action,
                                        "next_obs": next_obs, "reward": reward,
                                        "done": done}) + "\n")
            except OSError as exc:
                journaled = False
                if not self._journal_err_logged.get(domain):
                    self._journal_err_logged[domain] = True
                    logger.error("code_domains: journal append FAILED for %s (%s) "
                                 "— observes still work in-memory but will NOT "
                                 "survive a restart", domain, exc)
            if _AUTOSAVE_EVERY > 0 and count % _AUTOSAVE_EVERY == 0:
                try:
                    _CKPT_DIR.mkdir(parents=True, exist_ok=True)
                    eng.save(_CKPT_DIR / f"domain-{domain}.pt")
                except Exception as exc:
                    logger.warning("code_domains: autosave failed for %s: %s", domain, exc)
        return {"buffered": True, "domain": domain, "observed": count,
                "mode": eng.mode, "journaled": journaled}

    def surprise_batch(self, domain: str,
                       items: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        eng = self._engine(domain)
        out: Dict[str, Optional[float]] = {}
        with self._lock:
            for item in items:
                sid = str(item.get("id"))
                obs, action, next_obs = item.get("obs"), item.get("action"), item.get("next_obs")
                if not (isinstance(obs, str) and isinstance(next_obs, str)):
                    out[sid] = None
                    continue
                s = eng.surprise(_desc_hash(obs), action, _desc_hash(next_obs))
                out[sid] = s
                if s is not None:
                    prev = self._surprise_ema.get(domain)
                    self._surprise_ema[domain] = (
                        s if prev is None else 0.95 * prev + 0.05 * s
                    )
        return out

    def note_surprise(self, domain: str, surprise: float) -> Optional[float]:
        """Fold one graded prediction error into the domain's surprise EMA (the
        signal AitherEvolution schedules by). The decision door calls this on
        every outcome with |reward - predicted| / 2, so a fork the door still
        gets wrong reads as high-surprise = explore it next."""
        s = max(0.0, min(1.0, float(surprise)))
        with self._lock:
            prev = self._surprise_ema.get(domain)
            self._surprise_ema[domain] = s if prev is None else 0.95 * prev + 0.05 * s
            return self._surprise_ema[domain]

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "ok": _MLP_OK,
                # D-ref closed: observes are journaled and replayed on restart.
                # Reported per-domain as journaled=False on any append failure.
                "persistent_tabular": True,
                "domains": {
                    d: {"observed": self._observed.get(d, 0), "mode": e.mode,
                        "surprise_ema": self._surprise_ema.get(d),
                        "journaled": not self._journal_err_logged.get(d, False)}
                    for d, e in self._engines.items()
                },
            }
