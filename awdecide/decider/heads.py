"""Heads, metrics and the promotion rule for the decider. numpy only.

Two architectures per task, both softmax over the task's label space:
  logistic  xm @ W + b
  mlp       relu(xm @ W1 + b1) @ W2 + b2   (one hidden layer)
Both train with Adam on weighted cross-entropy plus L2, early-stop on a validation
tail, and get ONE temperature fitted on that tail. The test tail is never seen until
``evaluate``.

Promotion (``promotion``) is the only gate that decides whether a head may replace
what serves today. It is paired and statistical: the Brier gain's bootstrap CI must
exclude 0, and accuracy must not be significantly worse (exact McNemar) -- a point
estimate on a few dozen rows let a noise head through.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

#: a head may cost at most this much accuracy (absolute) and still be promoted --
#: and only if its Brier is strictly lower
MAX_ACC_DROP = 0.01
#: fewer held-out rows than this and the comparison is not evidence: refuse
MIN_TEST_ROWS = 50
#: paired bootstrap of the per-row Brier difference (seeded: same rows, same verdict)
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 13
#: McNemar significance for "accuracy is worse"
ALPHA = 0.05
ECE_BINS = 10


# ------------------------------------------------------------------ metrics --


def _onehot(y: Any, k: int) -> Any:
    y1h = np.zeros((len(y), k), dtype=np.float64)
    y1h[np.arange(len(y)), y] = 1.0
    return y1h


def brier(prob: Any, y: Any) -> float:
    """Multi-class Brier: mean over rows of sum over classes (p - onehot)^2.
    0 is perfect, 2 is confidently wrong. pre two-class row scores twice the binary Brier."""
    prob = np.asarray(prob, dtype=np.float64)
    return float(np.mean(np.sum((prob - _onehot(y, prob.shape[1])) ** 2, axis=1))) if len(y) else float("nan")


def ece(prob: Any, y: Any, bins: int = ECE_BINS) -> float:
    """Top-label expected calibration error over equal-width confidence bins."""
    prob = np.asarray(prob, dtype=np.float64)
    if not len(y):
        return float("nan")
    conf = prob.max(axis=1)
    hit = (prob.argmax(axis=1) == np.asarray(y)).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            tot += float(m.mean()) * abs(float(hit[m].mean()) - float(conf[m].mean()))
    return tot


def scores(prob: Any, y: Any) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    if not len(y):
        return {"n": 0, "accuracy": None, "brier": None, "ece": None}
    return {"n": int(len(y)),
            "accuracy": round(float(np.mean(np.asarray(prob).argmax(axis=1) == y)), 4),
            "brier": round(brier(prob, y), 4), "ece": round(ece(prob, y), 4)}


def softmax(z: Any) -> Any:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


# -------------------------------------------------------------------- heads --


@dataclass
class Head:
    """One trained head: arrays + what it needs to be served."""

    task: str
    kind: str
    labels: List[str]
    recipe: str                     # "hash" | "embed+hash"
    arch: str                       # "logistic" | "mlp"
    params: Dict[str, Any]
    T: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def logits(self, xm: Any) -> Any:
        xm = np.asarray(xm, dtype=np.float64)
        p = self.params
        if self.arch == "logistic":
            return xm @ p["W"] + p["b"]
        hid = np.maximum(0.0, xm @ p["W1"] + p["b1"])
        return hid @ p["W2"] + p["b2"]

    def proba(self, xm: Any, temp: Optional[float] = None) -> Any:
        return softmax(self.logits(xm) / max(1e-6, float(self.T if temp is None else temp)))


def _init(arch: str, d: int, k: int, hidden: int, rng: Any) -> Dict[str, Any]:
    if arch == "logistic":
        return {"W": np.zeros((d, k)), "b": np.zeros(k)}
    return {"W1": rng.normal(0, np.sqrt(2.0 / d), (d, hidden)), "b1": np.zeros(hidden),
            "W2": rng.normal(0, np.sqrt(1.0 / hidden), (hidden, k)), "b2": np.zeros(k)}


def _grads(arch: str, p: Dict[str, Any], xm: Any, y1h: Any, w: Any, l2: float) -> Any:
    """(loss, grads) for weighted softmax cross-entropy + L2 on weight matrices."""
    sw = float(w.sum()) or 1.0
    if arch == "logistic":
        pp = softmax(xm @ p["W"] + p["b"])
        gl = (pp - y1h) * w[:, None] / sw
        loss = -float(np.sum(w * np.log(np.maximum(1e-12, (pp * y1h).sum(axis=1))))) / sw
        loss += 0.5 * l2 * float(np.sum(p["W"] ** 2))
        return loss, {"W": xm.T @ gl + l2 * p["W"], "b": gl.sum(axis=0)}
    pre = xm @ p["W1"] + p["b1"]
    hid = np.maximum(0.0, pre)
    pp = softmax(hid @ p["W2"] + p["b2"])
    g2 = (pp - y1h) * w[:, None] / sw
    loss = -float(np.sum(w * np.log(np.maximum(1e-12, (pp * y1h).sum(axis=1))))) / sw
    loss += 0.5 * l2 * (float(np.sum(p["W1"] ** 2)) + float(np.sum(p["W2"] ** 2)))
    d_hid = (g2 @ p["W2"].T) * (pre > 0)
    return loss, {"W2": hid.T @ g2 + l2 * p["W2"], "b2": g2.sum(axis=0),
                  "W1": xm.T @ d_hid + l2 * p["W1"], "b1": d_hid.sum(axis=0)}


def _val_nll(head: Head, xm: Any, y: Any, temp: float = 1.0) -> float:
    prob = head.proba(xm, temp)
    return -float(np.mean(np.log(np.maximum(1e-12, prob[np.arange(len(y)), y]))))


def fit_head(task: str, kind: str, labels: Sequence[str], recipe: str, arch: str,
             x_tr: Any, ytr: Any, x_va: Any, yva: Any, wtr: Optional[Any] = None, *,
             l2: float = 1e-4, lr: float = 0.01, epochs: int = 60, batch: int = 256,
             hidden: int = 128, patience: int = 6, seed: int = 7) -> Head:
    """Adam, minibatch, early stop on validation NLL; then a temperature on validation."""
    rng = np.random.default_rng(seed)
    x_tr = np.asarray(x_tr, dtype=np.float64)
    k = len(labels)
    y1h = _onehot(np.asarray(ytr), k)
    w = np.ones(len(ytr)) if wtr is None else np.asarray(wtr, dtype=np.float64)
    p = _init(arch, x_tr.shape[1], k, hidden, rng)
    m = {n: np.zeros_like(v) for n, v in p.items()}
    v2 = {n: np.zeros_like(v) for n, v in p.items()}
    head = Head(task, kind, list(labels), recipe, arch, p)
    best, best_p, bad, t = float("inf"), {n: a.copy() for n, a in p.items()}, 0, 0
    has_val = x_va is not None and len(yva) > 0
    for ep in range(epochs):
        order = rng.permutation(len(ytr))
        for i in range(0, len(order), batch):
            ix = order[i:i + batch]
            _, g = _grads(arch, p, x_tr[ix], y1h[ix], w[ix], l2)
            t += 1
            for n in p:
                m[n] = 0.9 * m[n] + 0.1 * g[n]
                v2[n] = 0.999 * v2[n] + 0.001 * g[n] ** 2
                p[n] -= lr * (m[n] / (1 - 0.9 ** t)) / (np.sqrt(v2[n] / (1 - 0.999 ** t)) + 1e-8)
        if not has_val:
            continue
        nll = _val_nll(head, x_va, np.asarray(yva))
        if nll < best - 1e-5:
            best, best_p, bad = nll, {n: a.copy() for n, a in p.items()}, 0
            head.meta["best_epoch"] = ep + 1
        else:
            bad += 1
            if bad >= patience:
                break
    if has_val:
        head.params = best_p
        grid = np.logspace(-1.0, 1.0, 41)
        nlls = [_val_nll(head, x_va, np.asarray(yva), float(temp)) for temp in grid]
        head.T = float(grid[int(np.argmin(nlls))])
        head.meta["val_nll"] = round(min(nlls), 4)
    return head


def prior_proba(ytr: Any, k: int, n: int, alpha: float = 1.0) -> Any:
    """The prior rung: training class frequencies (add-alpha), the same for every row."""
    c = np.bincount(np.asarray(ytr, dtype=np.int64), minlength=k).astype(np.float64) + alpha
    return np.tile(c / c.sum(), (n, 1))


# ---------------------------------------------------------------- promotion --


def mcnemar_exact(new_hit: Any, cur_hit: Any) -> Dict[str, Any]:
    """Exact two-sided McNemar on paired correctness: b = rows only current got right,
    c = rows only new got right; p = P(Binomial(b + c, 1/2) as extreme as observed)."""
    new_hit, cur_hit = np.asarray(new_hit, bool), np.asarray(cur_hit, bool)
    b = int(np.sum(cur_hit & ~new_hit))
    c = int(np.sum(new_hit & ~cur_hit))
    m = b + c
    if m == 0:
        return {"b": b, "c": c, "p": 1.0}
    k = min(b, c)
    tail = sum(math.comb(m, i) for i in range(k + 1)) / 2.0 ** m
    return {"b": b, "c": c, "p": min(1.0, 2.0 * tail)}


def brier_rows(prob: Any, y: Any) -> Any:
    prob = np.asarray(prob, dtype=np.float64)
    return np.sum((prob - _onehot(y, prob.shape[1])) ** 2, axis=1)


def bootstrap_ci(d: Any, n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
                 alpha: float = 0.05) -> Any:
    """Percentile CI of the mean of ``d`` over paired row resamples (seeded)."""
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def promotion(p_new: Any, p_cur: Any, y: Any, max_acc_drop: float = MAX_ACC_DROP,
              min_rows: int = MIN_TEST_ROWS, alpha: float = ALPHA) -> Dict[str, Any]:
    """May the new head replace what serves, on these SAME held-out rows? Paired evidence:

    1. at least ``min_rows`` rows, else REFUSE "too few rows";
    2. the paired bootstrap 95 % CI of (Brier_current - Brier_new) per row has its
       LOWER bound > 0 -- the improvement is not noise;
    3. accuracy is not significantly worse: refuse when new loses more paired rows
       than it wins and exact McNemar p < ``alpha``;
    4. and, as before, accuracy not lower by more than ``max_acc_drop``.
    Measured why (2026-10-16): a head on pure-noise labels beat the prior on 80 rows,
    0.625 vs 0.537 accuracy and a lower Brier, under the point-estimate rule."""
    y = np.asarray(y, dtype=np.int64)
    p_new, p_cur = np.asarray(p_new, dtype=np.float64), np.asarray(p_cur, dtype=np.float64)
    n = int(len(y))
    sn, sc = scores(p_new, y), scores(p_cur, y)
    out: Dict[str, Any] = {"n": n, "new": sn, "current": sc}
    if n < min_rows or len(p_new) != n or len(p_cur) != n:
        out.update(promote=False, why=f"too few rows: {n} held-out (need {min_rows})")
        return out
    d = brier_rows(p_cur, y) - brier_rows(p_new, y)
    lo, hi = bootstrap_ci(d)
    mc = mcnemar_exact(p_new.argmax(axis=1) == y, p_cur.argmax(axis=1) == y)
    out.update(brier_gain=round(float(d.mean()), 4), brier_gain_ci95=[round(lo, 4), round(hi, 4)],
               mcnemar=mc)
    na, ca = float(sn["accuracy"]), float(sc["accuracy"])
    if not lo > 0:
        out.update(promote=False, why=f"Brier gain {d.mean():+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] "
                                      "lower bound <= 0 -- not evidence of a gain")
    elif mc["b"] > mc["c"] and mc["p"] < alpha:
        out.update(promote=False, why=f"accuracy significantly worse: loses {mc['b']} rows, wins "
                                      f"{mc['c']}, McNemar p={mc['p']:.4f}")
    elif na < ca - max_acc_drop:
        out.update(promote=False, why=f"accuracy {na:.4f} is {ca - na:.4f} below current {ca:.4f}")
    else:
        out.update(promote=True, why=f"Brier gain {d.mean():+.4f} CI [{lo:+.4f}, {hi:+.4f}], acc "
                                     f"{ca:.4f} -> {na:.4f} (McNemar p={mc['p']:.3f})")
    return out


# ------------------------------------------------------------------ storage --


def save_heads(path: Path, heads: Sequence[Head], meta: Dict[str, Any]) -> Path:
    """One npz (arrays + a json ``meta``) written atomically, plus ``<stem>.json``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(meta)
    doc["heads"] = {}
    arrays: Dict[str, Any] = {}
    for h in heads:
        doc["heads"][h.task] = {"kind": h.kind, "labels": h.labels, "recipe": h.recipe,
                                "arch": h.arch, "T": h.T, **h.meta}
        for n, a in h.params.items():
            arrays[f"{h.task}:{n}"] = np.asarray(a, dtype=np.float32)
    arrays["meta"] = np.array(json.dumps(doc))
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)
    path.with_suffix(".json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def load_heads(path: Path) -> Dict[str, Head]:
    """task -> Head. pre missing file is {}; a corrupt one raises (the caller decides)."""
    path = Path(path)
    if not path.is_file():
        return {}
    out: Dict[str, Head] = {}
    with np.load(str(path), allow_pickle=False) as z:
        doc = json.loads(str(z["meta"]))
        for task, h in (doc.get("heads") or {}).items():
            names = ("W", "b") if h["arch"] == "logistic" else ("W1", "b1", "W2", "b2")
            params = {n: np.asarray(z[f"{task}:{n}"], dtype=np.float64) for n in names}
            extra = {k: v for k, v in h.items() if k not in ("kind", "labels", "recipe", "arch", "T")}
            out[task] = Head(task, h["kind"], list(h["labels"]), h["recipe"], h["arch"], params,
                             float(h["T"]), extra)
    return out
