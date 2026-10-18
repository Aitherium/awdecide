"""DeciderScorer -- a trained decider head behind darklink's nerve ``Scorer`` seam.

darklink's nerve (``darklink.nerve``, branch feat/darklink-nerve, PR #13189 -- not
merged here, and NOT imported: this adapter is duck-typed against its Protocol)
asks one question per stimulus:

    class Scorer(Protocol):
        def probs(self, feats: dict[str, float]) -> dict[str, float]: ...

over the routes ``none | reflex | think``. This adapter answers it from the
``reflex.nerve`` head in a decider npz, built from the ``nerve_label`` rows the nerve
journals (``build_decider_dataset`` -> ``train_decider``). The feature dict is turned
into text by ``features.context_from_feats`` -- the same function that built the
training rows -- and scored with the head's recipe.

Fail-safe, in this order: no file / no head / a head that needs an embedder this
process does not have / any exception -> the ``fallback`` scorer (pass the nerve's own
``LinearScorer``); no fallback -> all mass on ``think`` (when the nerve cannot judge,
the brain decides). The model file is re-read when its mtime changes.

Online learning: the nerve's ``label()`` steps its scorer only when it IS a
``LinearScorer``. Wrapped in this adapter it does not learn online; the nightly
re-train is the learning path. Keep the nerve's ``LinearScorer`` as ``fallback`` and
it keeps serving (and its weights stay what the journal last saved).

    from darklink.nerve import Nerve, LinearScorer
    from awdecide.decider.scorer import DeciderScorer
    nerve = Nerve(path, scorer=DeciderScorer("<dir>/decider.npz", fallback=LinearScorer()))
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

NERVE_ROUTES = ("none", "reflex", "think")


class DeciderScorer:
    """``probs(feats) -> {route: p}`` from a decider head; see the module docstring."""

    def __init__(self, model_path: "str | os.PathLike[str] | None", task: str = "reflex.nerve",
                 fallback: Optional[Any] = None, routes: Sequence[str] = NERVE_ROUTES,
                 stimulus: str = "fight", embedder: Optional[Any] = None) -> None:
        self.path = Path(model_path) if model_path else None
        self.task, self.fallback, self.routes = task, fallback, tuple(routes)
        self.stimulus, self.embedder = stimulus, embedder
        self.error = ""
        self.served = 0
        self.fell_back = 0
        self._head: Any = None
        self._mtime: Optional[float] = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self.path is None:
            return None
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            self._head, self._mtime = None, None
            return None
        if mtime != self._mtime:
            from awdecide.decider.heads import load_heads

            self._mtime = mtime
            try:
                self._head = load_heads(self.path).get(self.task)
                self.error = "" if self._head is not None else f"no {self.task} head in {self.path.name}"
            except Exception as exc:  # noqa: BLE001 -- a corrupt file must not stop the game
                self._head, self.error = None, f"{type(exc).__name__}: {exc}"[:200]
        return self._head

    def _safe(self, feats: Mapping[str, float]) -> Dict[str, float]:
        self.fell_back += 1
        if self.fallback is not None:
            try:
                return dict(self.fallback.probs(dict(feats)))
            except Exception as exc:  # noqa: BLE001
                self.error = f"fallback: {type(exc).__name__}: {exc}"[:200]
        return {r: (1.0 if r == "think" else 0.0) for r in self.routes}

    def probs(self, feats: Dict[str, float]) -> Dict[str, float]:
        with self._lock:
            head = self._load()
        if head is None or (head.recipe != "hash" and self.embedder is None):
            return self._safe(feats)
        try:
            from awdecide.decider.features import context_from_feats, featurize

            xm = featurize(head.recipe, [context_from_feats(self.stimulus, feats)], self.embedder)
            p = head.proba(xm)[0]
            got = {lab: float(v) for lab, v in zip(head.labels, p)}
            out = {r: got.get(r, 0.0) for r in self.routes}
            s = sum(out.values())
            if s <= 0:
                return self._safe(feats)
            self.served += 1
            return {r: v / s for r, v in out.items()}
        except Exception as exc:  # noqa: BLE001 -- shape drift, missing numpy: fall back
            self.error = f"{type(exc).__name__}: {exc}"[:200]
            return self._safe(feats)
