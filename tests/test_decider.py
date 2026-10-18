"""awdecide.decider: dataset shaping, heads, the promotion gate, the nerve adapter."""

from __future__ import annotations

import json
import random

import pytest

np = pytest.importorskip("numpy")

from awdecide.decider import ROUTE_SPACE  # noqa: E402
from awdecide.decider import build_decider_dataset as bd  # noqa: E402
from awdecide.decider import heads as hd  # noqa: E402
from awdecide.decider import train_decider as td  # noqa: E402
from awdecide.decider.features import Embedder, context_from_feats, hashed  # noqa: E402
from awdecide.decider.scorer import DeciderScorer  # noqa: E402


def test_hashed_is_deterministic_and_normalised():
    a = hashed(["kind=route || fix the build", "kind=route || hello"])
    b = hashed(["kind=route || fix the build", "kind=route || hello"])
    assert a.shape == (2, 2048) and np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert not np.array_equal(a[0], a[1])


@pytest.mark.parametrize("rec,want", [
    ({"model_used": "x-orchestrator", "reasoning_used": False, "agentic_used": False}, "react"),
    ({"model_used": "x-orchestrator", "reasoning_used": True}, "think:orchestrator"),
    ({"model_used": "bonsai2-27b-5090"}, "think:bonsai2"),
    ({"model_used": "deepseek_api"}, "think:deepseek-pool"),
    ({"model_used": "gemma4-12b"}, "think:gemma4-vision"),
    ({"model_used": "search_fastpath"}, "react"),
    ({"model_used": "error"}, None),
    ({"model_used": "content_pipeline"}, None),
    ({"model_used": "mystery-lane"}, None),
])
def test_route_label(rec, want):
    got = bd.route_label(rec)
    assert got == want
    assert got is None or got in ROUTE_SPACE


def test_dedupe_majority_and_time_split_puts_newest_in_test():
    rows = [bd._row("ctx", "yesno", "t", lab, ["yes", "no"], "s", ts)
            for ts, lab in [(1, "yes"), (2, "no"), (3, "yes")]]
    rows += [bd._row(f"c{i}", "yesno", "t", "yes", ["yes", "no"], "s", 10 + i) for i in range(20)]
    d = bd.dedupe(rows)
    one = [r for r in d if r["context"] == "ctx"][0]
    assert one["label"] == "yes" and one["n"] == 3 and one["ts"] == 3
    bd.time_split(d, test_frac=0.2, val_frac=0.15)
    test_ts = [r["ts"] for r in d if r["split"] == "test"]
    train_ts = [r["ts"] for r in d if r["split"] == "train"]
    assert test_ts and min(test_ts) > max(train_ts)


def test_darklink_nerve_label_rows(tmp_path):
    feats = {"bias": 1.0, "in_combat": 1.0, "new_boss": 1.0}
    step = {"tick": 3, "kind": "link.step", "actor": "dd2", "ts": 5.0,
            "payload": {"meta": {"link_kind": "nerve_label",
                                 "nerve": {"stimulus": "fight", "label": "think", "feats": feats}}}}
    other = {"payload": {"meta": {"link_kind": "nerve", "nerve": {"route": "reflex"}}}}
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "journal.jsonl").write_text(
        json.dumps(step) + "\n" + json.dumps(other) + "\n{torn", encoding="utf-8")
    rows = bd.darklink_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["task"] == "reflex.nerve" and rows[0]["label"] == "think"
    assert rows[0]["context"] == context_from_feats("fight", feats)


def _sure(y, conf=0.95):
    return np.eye(2)[y] * (2 * conf - 1) + (1 - conf)


def test_promotion_needs_paired_evidence():
    y = np.array([i % 2 for i in range(200)])
    flat = np.full((200, 2), 0.5)
    assert hd.promotion(_sure(y), flat, y)["promote"] is True
    assert hd.promotion(flat, _sure(y), y)["promote"] is False
    r = hd.promotion(_sure(y)[:49], flat[:49], y[:49])
    assert r["promote"] is False and r["why"].startswith("too few rows")
    # equal on every row: gain 0, CI [0, 0] -> not evidence
    assert hd.promotion(flat, flat.copy(), y)["promote"] is False


def test_promotion_refuses_a_lucky_noise_gain():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 80)
    cur = np.full((80, 2), 0.5)
    new = cur.copy()
    new[:, 0] = np.where(rng.random(80) < 0.5, 0.55, 0.45)   # coin-flip leanings
    new[:, 1] = 1 - new[:, 0]
    r = hd.promotion(new, cur, y)
    assert r["promote"] is False and r["brier_gain_ci95"][0] <= 0


def test_promotion_refuses_significantly_worse_accuracy():
    y = np.array([i % 2 for i in range(200)])
    new = _sure(y, 0.99)
    new[:30] = new[:30, ::-1]
    r = hd.promotion(new, _sure(y, 0.8), y)
    assert r["promote"] is False and r["mcnemar"]["b"] == 30 and r["mcnemar"]["p"] < 0.05


def test_mcnemar_exact():
    hit = np.array([True] * 10)
    assert hd.mcnemar_exact(hit, hit)["p"] == 1.0
    m = hd.mcnemar_exact(np.array([False] * 6), np.array([True] * 6))
    assert m["b"] == 6 and m["c"] == 0 and m["p"] == pytest.approx(2 / 64)


def test_metrics():
    pm = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert hd.brier(pm, np.array([0, 1])) == 0.0
    assert hd.brier(pm, np.array([1, 0])) == 2.0
    assert hd.ece(np.array([[0.5, 0.5]] * 4), np.array([0, 0, 1, 1])) == pytest.approx(0.0)


def test_self_test_proves_the_gate_refuses_a_worse_head():
    assert td.self_test() == 0


def _reflex_head(tmp_path):
    rng = random.Random(0)
    rows = []
    for i in range(300):
        boss = rng.random() < 0.5
        lab = "think" if boss else "reflex"
        rows.append({"context": context_from_feats("fight", {"bias": 1.0, "new_boss": float(boss),
                                                             "in_combat": 1.0, "foes": rng.random()}),
                     "kind": "reflex", "task": "reflex.nerve", "label": lab, "ts": float(i)})
    bd.time_split(rows)
    code, results = td.run(rows, tmp_path, None, td.Current(None, None), {})
    assert results[0]["promotion"]["promote"], results[0]
    return tmp_path / "decider.npz"


def test_scorer_serves_the_head_and_maps_routes(tmp_path):
    path = _reflex_head(tmp_path)
    s = DeciderScorer(path)
    p = s.probs({"bias": 1.0, "new_boss": 1.0, "in_combat": 1.0, "foes": 0.4})
    assert set(p) == {"none", "reflex", "think"} and p["none"] == 0.0
    assert abs(sum(p.values()) - 1.0) < 1e-9 and p["think"] > 0.9
    q = s.probs({"bias": 1.0, "new_boss": 0.0, "in_combat": 1.0, "foes": 0.4})
    assert q["reflex"] > 0.9 and s.served == 2 and s.fell_back == 0


class _Fallback:
    def probs(self, feats):
        return {"none": 0.2, "reflex": 0.7, "think": 0.1}


def test_scorer_falls_back_safely(tmp_path):
    assert DeciderScorer(tmp_path / "absent.npz", fallback=_Fallback()).probs({})["reflex"] == 0.7
    assert DeciderScorer(None).probs({}) == {"none": 0.0, "reflex": 0.0, "think": 1.0}
    bad = tmp_path / "decider.npz"
    bad.write_bytes(b"not an npz")
    s = DeciderScorer(bad, fallback=_Fallback())
    assert s.probs({})["reflex"] == 0.7 and s.error
    # a head that needs an embedder, in a process without one
    path = _reflex_head(tmp_path / "m")
    heads = hd.load_heads(path)
    heads["reflex.nerve"].recipe = "embed+hash"
    hd.save_heads(tmp_path / "e.npz", list(heads.values()), {})
    assert DeciderScorer(tmp_path / "e.npz").probs({})["think"] == 1.0


def test_embedder_caches_by_sha(tmp_path):
    calls = []

    def post(texts):
        calls.append(list(texts))
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    e = Embedder("https://unused.invalid/v1/embeddings", "m", tmp_path / "c.sqlite", post=post)
    a = e.embed(["ab", "abc", "ab"])
    assert a.shape == (3, 3) and len(calls) == 1 and calls[0] == ["ab", "abc"]
    e2 = Embedder("https://unused.invalid/v1/embeddings", "m", tmp_path / "c.sqlite", post=post)
    b = e2.embed(["abc", "ab"])
    assert len(calls) == 1 and e2.hits == 2 and np.allclose(b[0], a[1])


def test_heads_roundtrip(tmp_path):
    xm = hashed([f"x {i % 2}" for i in range(40)])
    y = np.array([i % 2 for i in range(40)])
    h = hd.fit_head("t", "yesno", ["no", "yes"], "hash", "mlp", xm, y, xm, y, epochs=5)
    hd.save_heads(tmp_path / "m.npz", [h], {"k": 1})
    g = hd.load_heads(tmp_path / "m.npz")["t"]
    assert np.allclose(g.proba(xm), h.proba(xm), atol=1e-5) and g.arch == "mlp"
