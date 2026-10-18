"""Feature recipes for the decider: a hashed bag (no network) and a cached embedding.

``hash``        signed crc32 hashing of word unigrams, bigrams and ``k=v`` fields into
                ``HASH_DIM`` dims, log-scaled, L2-normalised. Deterministic across
                processes and machines (crc32, never ``hash()``).
``embed+hash``  an OpenAI-shaped ``/v1/embeddings`` vector (L2-normalised)
                concatenated with the hashed block.

Embeddings are cached by sha256(model + text) in a sqlite file, so a re-train costs
nothing for text it has already seen. The cache lives wherever the caller says --
never in the source tree.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import ssl
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover -- the trainer reports it and exits 2
    np = None  # type: ignore[assignment]

HASH_DIM = 2048
EMBED_MAX_CHARS = 2000
_WORD = re.compile(r"[a-z0-9_:+.-]+")
_STOP = frozenset("the a an of to in for on at be is are was were and or it this that with as by".split())


def _tokens(context: str) -> List[str]:
    """Tokens of one context. Segments split by ``||`` keep their index so a word in
    the header (kind / intent / effort) and the same word in the free text differ."""
    toks: List[str] = []
    for si, seg in enumerate(context.lower().split("||")[:6]):
        words = [w for w in _WORD.findall(seg) if w not in _STOP][:120]
        for w in words:
            toks.append(f"{si}:{w}")
            if "=" in w:  # a k=v field also counts as its key
                toks.append(f"{si}:k:{w.split('=', 1)[0]}")
        toks.extend(f"{si}:{a}_{b}" for a, b in zip(words, words[1:]))
    return toks


def hashed(contexts: Sequence[str], dim: int = HASH_DIM) -> Any:
    """[n, dim] float32 -- the ``hash`` recipe."""
    if np is None:
        raise RuntimeError("numpy is required for decider features")
    xs = np.zeros((len(contexts), dim), dtype=np.float32)
    for i, ctx in enumerate(contexts):
        toks = _tokens(ctx)
        if not toks:
            continue
        hs = np.fromiter((zlib.crc32(t.encode("utf-8", "replace")) for t in toks),
                         dtype=np.uint32, count=len(toks))
        idx = (hs >> np.uint32(1)) % np.uint32(dim)
        sign = np.where(hs & np.uint32(1), 1.0, -1.0)
        row = np.bincount(idx, weights=sign, minlength=dim)
        row = np.sign(row) * np.log1p(np.abs(row))
        n = float(np.linalg.norm(row)) or 1.0
        xs[i] = row / n
    return xs


def _bucket(v: float) -> str:
    """A feature value as a coarse token: 0, low, mid, high, 1 (and the sign)."""
    if v == 0:
        return "0"
    a = abs(v)
    b = "1" if a >= 0.999 else ("high" if a >= 0.66 else ("mid" if a >= 0.33 else "low"))
    return b if v > 0 else "neg" + b


def context_from_feats(stimulus: str, feats: Mapping[str, float]) -> str:
    """The text a numeric feature dict is scored as -- the SAME function builds the
    training row (from a journaled ``nerve_label``) and the serving input, so the two
    can never drift apart."""
    parts = [f"{k}={_bucket(float(v))}" for k, v in sorted(feats.items())
             if isinstance(v, (int, float)) and k != "bias"]
    return f"kind=reflex stimulus={stimulus} || " + " ".join(parts)


class Embedder:
    """``/v1/embeddings`` client with a sha-keyed sqlite cache.

    ``url`` is the full endpoint (``https://host:port/v1/embeddings``). TLS is verified
    against ``ca_bundle`` when given, else the system store -- never disabled.
    """

    def __init__(self, url: str, model: str, cache_path: Optional[Path] = None,
                 ca_bundle: Optional[str] = None, timeout: float = 60.0, batch: int = 32,
                 post: Optional[Callable[[List[str]], List[List[float]]]] = None) -> None:
        self.url, self.model = url, model
        self.timeout, self.batch = float(timeout), int(batch)
        self._ctx = ssl.create_default_context(cafile=ca_bundle) if ca_bundle else None
        self._post = post or self._http
        self.dim: Optional[int] = None
        self.calls = 0
        self.hits = 0
        self._db: Optional[sqlite3.Connection] = None
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(cache_path))
            self._db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")

    def _key(self, text: str) -> str:
        return hashlib.sha256((self.model + "\x00" + text).encode("utf-8", "replace")).hexdigest()

    def _http(self, texts: List[str]) -> List[List[float]]:
        body = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(self.url, data=body,
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx) as r:
            doc = json.loads(r.read().decode("utf-8"))
        data = sorted(doc.get("data") or [], key=lambda d: int(d.get("index", 0)))
        return [d["embedding"] for d in data]

    def embed(self, texts: Sequence[str]) -> Any:
        """[n, dim] float32, L2-normalised. Raises if the endpoint answers short."""
        if np is None:
            raise RuntimeError("numpy is required for decider features")
        clipped = [t[:EMBED_MAX_CHARS] for t in texts]
        keys = [self._key(t) for t in clipped]
        got: Dict[str, Any] = {}
        if self._db is not None:
            uniq = list(dict.fromkeys(keys))
            for i in range(0, len(uniq), 500):
                chunk = uniq[i:i + 500]
                q = "SELECT k, v FROM emb WHERE k IN (%s)" % ",".join("?" * len(chunk))
                for k, v in self._db.execute(q, chunk):
                    got[k] = np.frombuffer(v, dtype=np.float32)
        self.hits += sum(1 for k in keys if k in got)
        todo = list(dict.fromkeys(t for t, k in zip(clipped, keys) if k not in got))
        for i in range(0, len(todo), self.batch):
            part = todo[i:i + self.batch]
            vecs = self._post(part)
            self.calls += 1
            if len(vecs) != len(part):
                raise RuntimeError(f"embedder returned {len(vecs)} vectors for {len(part)} texts")
            rows = []
            for t, v in zip(part, vecs):
                a = np.asarray(v, dtype=np.float32)
                a = a / (float(np.linalg.norm(a)) or 1.0)
                got[self._key(t)] = a
                rows.append((self._key(t), a.tobytes()))
            if self._db is not None:
                self._db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
                self._db.commit()
        if not keys:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        out = np.stack([got[k] for k in keys]).astype(np.float32)
        self.dim = int(out.shape[1])
        return out


def featurize(recipe: str, contexts: Sequence[str], embedder: Optional[Embedder] = None) -> Any:
    """One recipe -> [n, d] float32. ``embed+hash`` without an embedder raises."""
    hb = hashed(contexts)
    if recipe == "hash":
        return hb
    if recipe == "embed+hash":
        if embedder is None:
            raise RuntimeError("recipe embed+hash needs an embedder")
        return np.concatenate([embedder.embed(contexts), hb], axis=1).astype(np.float32)
    raise ValueError(f"unknown feature recipe {recipe!r}")
