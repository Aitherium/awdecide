"""The loop, the chat rung and the MCP/HTTP doors. Each test fails if the loop is cut."""
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from awdecide import CallableBackend, Ladder, Ledger, Loop, Question
from awdecide import mcp as mcp_mod
from awdecide.loop import ChatBackend, bench, match_label

AB = Question.choice(["fast", "deep"])


def _loop(fn=None, calibrated=False):
    rungs = []
    if fn is not None:
        rung = CallableBackend(fn, name="brain")
        rung.calibrated = calibrated
        rungs.append(rung)
    return Loop(Ladder(rungs), Ledger(Path(":memory:")))


def test_nothing_knows_is_undecided_and_is_not_recorded_as_a_claim():
    loop = _loop()
    d = loop.decide("router", "kind:code", AB)
    assert d.decided is False and d.value is None and d.id == ""
    assert loop.ledger.pending() == 0


def test_a_resolved_decision_is_answered_from_evidence_with_no_model_call():
    calls = []
    loop = _loop(lambda s, q: calls.append(s) or {"deep": 1.0})
    first = loop.decide("router", "kind:code", AB)
    assert first.backend == "brain" and len(calls) == 1
    loop.resolve(first.id, correct=True)
    again = loop.decide("router", "kind:code", AB)
    assert again.backend == "evidence" and again.value == "deep"
    assert len(calls) == 1, "the repeat must not reach the model"


def test_an_answer_resolved_wrong_is_withheld_from_the_ladder():
    offered = []

    def brain(_s, q):
        offered.append(list(q.options))
        return {q.options[0]: 1.0}
    loop = _loop(brain)
    q = Question.choice(["a", "b", "c"])
    d = loop.decide("k", "s", q)
    loop.resolve(d.id, correct=False)
    d2 = loop.decide("k", "s", q)
    assert offered[1] == ["b", "c"] and d2.value == "b"
    assert set(d2.probabilities) == {"a", "b", "c"}, "the answer is over the ORIGINAL options"


def test_when_every_other_option_failed_here_the_last_one_answers_without_a_call():
    calls = []
    loop = _loop(lambda s, q: calls.append(1) or {q.options[0]: 1.0})
    loop.teach("k", "s", "fast", correct=False)
    d = loop.decide("k", "s", AB)
    assert d.value == "deep" and d.backend == "evidence" and calls == []


def test_bool_evidence_against_one_side_answers_the_other():
    loop = _loop()
    loop.teach("gate", "s", "yes", correct=False, kind="bool")
    d = loop.decide("gate", "s", Question.bool())
    assert d.value == "no" and d.backend == "evidence"


def test_a_probability_is_fitted_to_outcomes_not_pinned_at_certainty():
    loop = _loop()
    for i in range(100):                                      # a 60/40 coin
        loop.teach("coin", "s", "heads", correct=i % 5 < 3)
    d = loop.decide("coin", "s", Question.choice(["heads", "tails"]))
    assert d.value == "heads" and 0.55 < d.probability < 0.65


def test_an_uncalibrated_rung_gets_its_measured_hit_rate_a_calibrated_one_keeps_its_own():
    loop = _loop(lambda s, q: {q.options[0]: 1.0})
    assert loop.decide("f", "s0", AB).probability == 0.5          # unmeasured
    for i in range(8):
        loop.resolve(loop.decide("f", f"t{i}", AB).id, correct=True)
    assert loop.decide("f", "fresh", AB).probability == 0.9       # (8+1)/(8+2)
    own = _loop(lambda s, q: {"fast": 0.8, "deep": 0.2}, calibrated=True)
    assert own.decide("f", "s", AB).probability == pytest.approx(0.8)


def test_min_confidence_holds_the_answer_and_keeps_the_evidence():
    loop = _loop(lambda s, q: {q.options[0]: 1.0})
    d = loop.decide("f", "s", Question.choice(["a", "b"], min_confidence=0.8))
    assert d.decided is False and d.value is None and d.probabilities["a"] == 0.5
    assert loop.ledger.pending() == 0


@pytest.mark.parametrize("key,state", [("", "s"), ("k", ""), ("k", "  ")])
def test_a_decision_without_a_key_or_a_state_is_refused(key, state):
    with pytest.raises(ValueError):
        _loop().decide(key, state, AB)


def test_prose_naming_two_options_names_none():
    assert match_label("Deep.", ["fast", "deep"]) == "deep"
    assert match_label("either fast or deep", ["fast", "deep"]) is None


def test_the_bench_beats_the_same_brain_called_every_time_and_the_base_rate():
    r = bench(seed=7)
    assert r["static"]["model_calls"] == 400 and r["loop"]["model_calls"] < 100
    assert r["loop"]["accuracy"] > r["static"]["accuracy"] + 0.15
    assert r["loop"]["brier"] < r["loop"]["climatology"], "probabilities must carry information"


class _FakeLLM(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *_a):
        return

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeLLM.seen.append((self.path, self.headers.get("Authorization"), body))
        raw = json.dumps({"choices": [{"message": {"content": " deep\n"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture()
def fake_llm():
    _FakeLLM.seen = []
    srv = HTTPServer(("127.0.0.1", 0), _FakeLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.mark.parametrize("suffix", ["", "/v1"])
def test_the_chat_rung_asks_an_openai_wire_server_for_one_label(fake_llm, suffix):
    w = ChatBackend(fake_llm + suffix, "m", token="t").weights("state", AB)
    assert w == {"deep": 1.0}
    path, auth, body = _FakeLLM.seen[0]
    assert path == "/v1/chat/completions" and auth == "Bearer t" and body["model"] == "m"


def test_a_dead_brain_is_a_rung_that_abstained_not_a_crash():
    loop = Loop(Ladder([ChatBackend("http://127.0.0.1:9", "m", timeout=2)]),
                Ledger(Path(":memory:")))
    d = loop.decide("k", "s", AB)
    assert d.decided is False and any("abstained" in r for r in d.reasons)


def _rpc(loop, mid, method, params=None):
    return mcp_mod.handle(loop, {"jsonrpc": "2.0", "id": mid, "method": method,
                                 "params": params or {}})


def test_mcp_lists_the_tools_and_closes_the_loop_end_to_end():
    loop = _loop(lambda s, q: {"deep": 1.0})
    assert _rpc(loop, 1, "initialize")["result"]["serverInfo"]["name"] == "awdecide"
    assert mcp_mod.handle(loop, {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    names = [t["name"] for t in _rpc(loop, 2, "tools/list")["result"]["tools"]]
    assert names == ["decide", "decide_outcome", "decide_teach", "decide_stats"]
    args = {"fork": "router", "state": "kind:code", "options": ["fast", "deep"]}
    first = json.loads(_rpc(loop, 3, "tools/call", {"name": "decide", "arguments": args})
                       ["result"]["content"][0]["text"])
    assert first["source"] == "brain" and first["answer"] == "deep"
    out = _rpc(loop, 4, "tools/call", {"name": "decide_outcome", "arguments": {
        "decision_id": first["decision_id"], "correct": True}})
    assert out["result"]["isError"] is False
    again = json.loads(_rpc(loop, 5, "tools/call", {"name": "decide", "arguments": args})
                       ["result"]["content"][0]["text"])
    assert again["source"] == "evidence"
    bad = _rpc(loop, 6, "tools/call", {"name": "decide", "arguments": {"fork": "x"}})
    assert bad["result"]["isError"] is True
    assert "error" in _rpc(loop, 7, "nope/nope")


def test_mcp_yesno_and_teach_cover_the_no_brain_path():
    loop = _loop()
    args = {"fork": "retry", "state": "err:timeout", "kind": "yesno"}
    d = mcp_mod.call(loop, "decide", args)
    assert d["decided"] is False and d["decision_id"] == ""
    with pytest.raises(ValueError):
        mcp_mod.call(loop, "decide_outcome", {"decision_id": "", "correct": True})
    mcp_mod.call(loop, "decide_teach", {**args, "value": "yes", "correct": True})
    assert mcp_mod.call(loop, "decide", args)["answer"] == "yes"


def test_http_serves_the_same_loop_and_refuses_a_malformed_request():
    box, ready = [], threading.Event()

    def run():          # the ledger's connection belongs to the thread that serves, as in serve()
        box.append(HTTPServer(("127.0.0.1", 0), mcp_mod.handler(_loop())))
        ready.set()
        box[0].serve_forever()
    threading.Thread(target=run, daemon=True).start()
    assert ready.wait(5)
    srv = box[0]
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())
    try:
        post("/decide/teach", {"fork": "k", "state": "s", "value": "deep", "correct": True})
        d = post("/decide", {"fork": "k", "state": "s", "options": ["fast", "deep"]})
        assert d["answer"] == "deep" and d["source"] == "evidence"
        with pytest.raises(urllib.error.HTTPError) as e:
            post("/decide", {"fork": "k"})
        assert e.value.code == 422
    finally:
        srv.shutdown()
