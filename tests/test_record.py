"""awdecide.record -- the DecisionRecord join, its export schema and honest stats."""
from __future__ import annotations

import json
import sqlite3
import time

import pytest

# isort: split
# (awdecide is first-party under the package's ruff config and third-party under the
# python-quality gate's --isolated run; a split block reads sorted to both)
from awdecide import record as rec
from awdecide.contract import Decision
from awdecide.ledger import Ledger


def _decide(led: Ledger, key: str, p: float, *, backend: str = "bash-hook",
            correct=None, ts=None) -> str:
    d = Decision(kind="choice", value="run", probability=p,
                 probabilities={"run": p, "skip": 1.0 - p}, decided=True, backend=backend)
    did = led.record(key, "state", d)
    if ts is not None:
        led._con.execute("UPDATE decisions SET ts=? WHERE id=?", (ts, did))
        led._con.commit()
    if correct is not None:
        led.resolve(did, correct)
    return did


@pytest.fixture()
def led(tmp_path):
    lg = Ledger(tmp_path / "decide.db")
    yield lg
    lg.close()


# ------------------------------------------------------------------ the join
def test_attach_joins_ctx_to_its_decision_and_outcome(led):
    did = _decide(led, "tool.bash", 0.9, correct=True)
    assert rec.attach(led, did, session_id="s1", tool_use_id="toolu_1", rung="evidence",
                      model="bonsai2", think=False, latency_ms=3.5, tokens=12,
                      state_text="git status", options=["run", "skip"])
    rows = list(rec.iter_records(led))
    assert len(rows) == 1
    r = rows[0]
    assert r["id"] == did and r["outcome"] == 1 and r["p"] == pytest.approx(0.9)
    assert r["ctx"]["session_id"] == "s1" and r["ctx"]["tool_use_id"] == "toolu_1"
    assert r["ctx"]["think"] is False and r["ctx"]["options"] == ["run", "skip"]
    assert r["rung"] == "evidence" and r["ctx"]["latency_ms"] == 3.5


def test_attach_twice_merges_and_never_blanks_a_field(led):
    did = _decide(led, "decide.router", 0.6, backend="local:chat")
    rec.attach(led, did, session_id="s1", rung="llm")
    rec.attach(led, did, mcts_visits={"fast": 12, "deep": 30}, mcts_value=0.41)
    (r,) = rec.iter_records(led)
    assert r["ctx"]["session_id"] == "s1" and r["ctx"]["rung"] == "llm"
    assert r["ctx"]["mcts_visits"] == {"fast": 12, "deep": 30}
    assert r["ctx"]["mcts_value"] == pytest.approx(0.41)


def test_unknown_metadata_lands_in_features_not_on_the_floor(led):
    did = _decide(led, "decide.router", 0.6)
    rec.attach(led, did, caller="adk.choose", features={"a": 1}, user_flag="x")
    (r,) = rec.iter_records(led)
    assert r["ctx"]["features"] == {"a": 1, "user_flag": "x"}
    assert r["ctx"]["caller"] == "adk.choose"


def test_attach_uncommitted_rides_the_callers_transaction(tmp_path):
    led = Ledger(tmp_path / "d.db")
    did = _decide(led, "tool.bash", 0.9)
    rec.attach(led._con, did, commit=False, session_id="s")
    led._con.rollback()
    assert list(rec.iter_records(tmp_path / "d.db"))[0]["ctx"] is None
    rec.attach(led._con, did, commit=False, session_id="s")
    led.resolve(did, True)  # its commit carries the ctx row
    led.close()
    (r,) = rec.iter_records(tmp_path / "d.db")
    assert r["ctx"]["session_id"] == "s" and r["outcome"] == 1


def test_state_text_is_scrubbed_and_cut(led):
    did = _decide(led, "tool.bash", 0.9)
    secret = "Bearer " + "abcdefghij" * 3
    rec.attach(led, did, state_text="curl -H 'Authorization: %s' https://u:pw@h/x?token=zz9 "
               % secret + "x" * 5000)
    (r,) = rec.iter_records(led)
    text = r["ctx"]["state_text"]
    assert "abcdefghij" not in text and "pw@" not in text and "zz9" not in text
    assert len(text) <= rec.STATE_TEXT_MAX
    assert rec.scrub("curl -u alice:s3cret https://h") == "curl -u <secret> https://h"
    assert rec.scrub("password=hunter2 api_key=k1") == "password=<secret> api_key=<secret>"


def test_a_ledger_without_the_ctx_table_still_reads(tmp_path):
    led = Ledger(tmp_path / "old.db")
    _decide(led, "tool.bash", 0.9, correct=True)
    led.close()
    (r,) = rec.iter_records(tmp_path / "old.db")
    assert r["ctx"] is None and r["rung"] == "evidence"  # bash-hook -> evidence
    con = sqlite3.connect(tmp_path / "old.db")
    assert con.execute("SELECT name FROM sqlite_master WHERE name='decision_ctx'"
                       ).fetchone() is None  # reading never created it


# ------------------------------------------------------------------ export
def test_export_schema_and_since_filter(led, tmp_path):
    old = _decide(led, "tool.bash", 0.8, correct=False, ts=time.time() - 3600)
    new = _decide(led, "tool.bash", 0.95, correct=True)
    rec.attach(led, new, session_id="s", tool_use_id="t")
    pend = _decide(led, "tool.bash", 0.7)
    out = tmp_path / "x.jsonl"
    counts = rec.export_records(led, out, since=time.time() - 60)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["id"] for r in rows] == [new, pend] and old not in {r["id"] for r in rows}
    assert counts == {"rows": 2, "joined": 1, "resolved": 1, "joined_resolved": 1}
    want = {"schema", "id", "ts", "key", "kind", "value", "p", "backend", "rung",
            "state_sha", "ctx", "outcome", "brier", "resolved_at"}
    assert all(set(r) == want for r in rows)
    assert rows[0]["schema"] == rec.SCHEMA_ID and rows[0]["brier"] == pytest.approx(0.0025)
    assert set(rows[0]["ctx"]) == set(rec.CTX_FIELDS)
    assert rows[1]["outcome"] is None


def test_export_filters_and_iso_since(led, tmp_path):
    a = _decide(led, "tool.bash", 0.9, correct=True)
    _decide(led, "tool.bash", 0.9)
    _decide(led, "bench/decide.bench.router", 0.9, backend="door:engine", correct=True)
    rec.attach(led, a, session_id="s")
    out = tmp_path / "y.jsonl"
    c = rec.export_records(led, out, since="2000-01-01T00:00:00Z", resolved_only=True,
                           joined_only=True)
    assert c["rows"] == 1 and json.loads(out.read_text())["id"] == a


# ------------------------------------------------------------------ stats
def test_bss_excludes_bench_and_demo_rows(led):
    # 8 tool.bash rows: base rate 0.75, every claim at 0.75 -> BSS exactly 0
    for o in [1, 1, 1, 1, 1, 1, 0, 0]:
        _decide(led, "tool.bash", 0.75, correct=bool(o))
    # perfect bench rows and a demo coin would flatter the headline if counted
    for _ in range(50):
        _decide(led, "bench/decide.bench.router", 1.0, backend="door:engine", correct=True)
    _decide(led, "live/decide.demo.20260920075641.coin", 0.5, backend="door:engine:replay",
            correct=True)
    s = rec.stats(led)
    o = s["overall"]
    assert o["n"] == 8 and o["base_rate"] == 0.75
    assert o["brier"] == pytest.approx(0.1875) and o["climatology"] == pytest.approx(0.1875)
    assert o["bss"] == pytest.approx(0.0)
    assert s["excluded"]["n"] == 51
    assert list(s["by_family"]) == ["tool.bash"]


def test_bss_is_positive_only_when_the_probability_carries_information(led):
    for _ in range(10):
        _decide(led, "decide.router", 0.9, backend="local:evidence", correct=True)
        _decide(led, "decide.router", 0.1, backend="local:evidence", correct=False)
    s = rec.stats(led)
    r = s["by_family"]["decide.router"]
    assert r["bss"] == pytest.approx(1 - 0.01 / 0.25)
    assert r["ece"] == pytest.approx(0.1)
    assert list(s["by_rung"]) == ["evidence"]


def test_per_rung_uses_ctx_rung_over_backend(led):
    a = _decide(led, "decide.router", 0.8, backend="local:chat", correct=True)
    _decide(led, "decide.router", 0.8, backend="local:chat", correct=False)
    rec.attach(led, a, rung="think")
    s = rec.stats(led)
    assert s["by_rung"]["think"]["n"] == 1 and s["by_rung"]["chat"]["n"] == 1
    assert s["joined_rows"] == 1
    assert "decide.router @ think" in s["by_family_rung"]


def test_stale_pending_rows_count_as_expired(led):
    now = time.time()
    for _ in range(3):
        _decide(led, "tool.bash", 0.9, ts=now - 8 * 86400)   # lost claims
    _decide(led, "tool.bash", 0.9, ts=now - 3600)            # still open
    _decide(led, "tool.bash", 0.9, correct=True)
    s = rec.stats(led, now=now)
    o = s["overall"]
    assert (o["n"], o["pending"], o["expired"]) == (1, 1, 3)
    assert rec.stats(led, now=now, expire_days=30)["overall"]["expired"] == 0


def test_family_and_exclusion_rules():
    assert rec.family("tool.bash") == "tool.bash"
    assert rec.family("live/decide.router.x.y") == "live/decide.router"
    assert rec.family("decide.20260920.coin") == "decide.*"
    assert rec.is_excluded("bench/decide.bench.router")
    assert rec.is_excluded("live/decide.demo.20260920075641.coin")
    assert not rec.is_excluded("tool.bash") and not rec.is_excluded("decide.benchmarks")
    assert rec.rung_of("door:engine") == "engine" and rec.rung_of("bash-hook") == "evidence"


def test_cli_stats_and_export(led, tmp_path, capsys):
    did = _decide(led, "tool.bash", 0.9, correct=True)
    rec.attach(led, did, session_id="s")
    db = str(led.path)
    assert rec.main(["--db", db, "stats", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["overall"]["n"] == 1
    out = tmp_path / "o.jsonl"
    assert rec.main(["--db", db, "export", "--out", str(out), "--since", "0"]) == 0
    assert json.loads(out.read_text())["ctx"]["session_id"] == "s"
    assert rec.main(["--db", str(tmp_path / "missing.db"), "stats"]) == 2
