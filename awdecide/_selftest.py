"""`awdecide --self-test` — the contract must be able to FAIL.

Arms (each prints one line; the run exits 0 only when all pass):
  1 rules rung answers a choice at p=1.0 and the answer is decided
  2 an empty ladder answers decided=False with a reason -- fail-closed, never a guess
  3 min_confidence turns a weak callable answer into decided=False and keeps the evidence
  4 the logprob rung turns a real OpenAI-wire logprobs payload into a distribution
    over the option labels only (served by an in-process HTTP server -- no network)
  5 the ledger records, resolves, and reports: a calibrated set beats the base rate,
    an overconfident set does not; an undecided answer is never recorded
  6 the CLI grammar parses every primitive and rejects a malformed spec
  7 the world-model door as a rung: choice/score/bool map onto choice/score/yesno,
    the door's CALIBRATED probability survives the ladder unrenormalized,
    source=none ABSTAINS (never the door's placeholder option), resolve() posts the
    outcome back; and the bridge lands a door journal in the ledger idempotently,
    skipping confidence-only legacy rows (fake in-process HTTP door -- no network)
"""
from __future__ import annotations

import json
import random
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import List

from .backends import CallableBackend, Ladder, LogprobBackend, RulesBackend, parse_question_spec
from .contract import Question
from .ledger import Ledger


def _arm(results: List[str], name: str, ok: bool, detail: str = "") -> None:
    results.append(f"  arm {name}: {'OK' if ok else 'FAIL'}{(' -- ' + detail) if detail else ''}")


class _FakeLLM(BaseHTTPRequestHandler):
    """Answers every chat completion with a logprobs payload over 'billing'/'sales'."""

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        n = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(n)
        body = {"choices": [{"message": {"content": "billing"}, "logprobs": {"content": [{
            "token": "billing", "logprob": -0.2231,
            "top_logprobs": [
                {"token": "billing", "logprob": -0.2231},   # 0.80
                {"token": "sales", "logprob": -1.8971},     # 0.15
                {"token": "The", "logprob": -2.9957},       # 0.05 -> not an option, dropped
            ]}]}}]}
        data = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a: object) -> None:  # silence
        pass


def run_self_test() -> int:
    res: List[str] = []
    rnd = random.Random(11)

    # 1 rules
    q = Question.choice(["billing", "technical", "sales"])
    d = Ladder([RulesBackend([(r"refund|invoice", "billing")])]).decide_one(
        "Customer asks about a refund on invoice 4411", q)
    _arm(res, "1 rules", d.decided and d.value == "billing" and d.probability == 1.0
         and d.backend == "rules", d.to_dict()["value"] or "undecided")

    # 2 empty ladder -> fail-closed
    d = Ladder([]).decide_one("anything", q)
    _arm(res, "2 fail-closed", (not d.decided) and d.value is None and d.probability == 0.0
         and any("no backends" in r for r in d.reasons), "; ".join(d.reasons))

    # 3 min_confidence
    weak = CallableBackend(lambda s, qq: {"billing": 0.4, "technical": 0.35, "sales": 0.25}, "weak")
    q7 = Question.choice(["billing", "technical", "sales"], min_confidence=0.7)
    d = Ladder([weak]).decide_one("x", q7)
    _arm(res, "3 min_confidence", (not d.decided) and abs(d.probabilities["billing"] - 0.4) < 1e-9
         and any("< min_confidence" in r for r in d.reasons))

    # 4 logprob rung over a fake OpenAI-wire server
    srv = HTTPServer(("127.0.0.1", 0), _FakeLLM)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        lb = LogprobBackend(f"http://127.0.0.1:{srv.server_port}", "fake")
        d = Ladder([lb]).decide_one("Customer asks about a refund", q)
        ok = (d.decided and d.value == "billing" and abs(d.probability - 0.8 / 0.95) < 1e-3
              and abs(sum(d.probabilities.values()) - 1.0) < 1e-9
              and d.probabilities["technical"] == 0.0)
        _arm(res, "4 logprob", ok, json.dumps(d.to_dict()["probabilities"]))
    finally:
        srv.shutdown()
        srv.server_close()

    # 5 ledger
    with tempfile.TemporaryDirectory() as td:
        led = Ledger(Path(td) / "awdecide.db")
        led.record("k", "s", Ladder([]).decide_one("s", q))  # undecided -> not recorded
        skipped_ok = led.pending() == 0
        # calibrated: outcome drawn at the stated probability
        for _ in range(300):
            p = rnd.choice([0.55, 0.7, 0.85, 0.95])
            dd = CallableBackend(lambda s, qq, _p=p: {"billing": _p, "sales": 1 - _p}, "cal")
            ans = Ladder([dd]).decide_one("s", Question.choice(["billing", "sales"]))
            did = led.record("k", "s", ans)
            led.resolve(did, correct=rnd.random() < ans.probability)
        rep = led.reliability()
        cal_ok = rep["beats_base_rate"] and rep["resolved"] == 300
        # overconfident, fresh ledger
        led2 = Ledger(Path(td) / "over.db")
        for _ in range(200):
            dd = CallableBackend(lambda s, qq: {"billing": 0.95, "sales": 0.05}, "over")
            ans = Ladder([dd]).decide_one("s", Question.choice(["billing", "sales"]))
            led2.resolve(led2.record("k", "s", ans), correct=rnd.random() < 0.5)
        rep2 = led2.reliability()
        over_ok = not rep2["beats_base_rate"]
        exported = led.export_platform_jsonl(Path(td) / "predictions.jsonl")
        led.close()
        led2.close()
        _arm(res, "5 ledger", skipped_ok and cal_ok and over_ok and exported == 300,
             f"cal brier={rep.get('brier')} clim={rep.get('climatology')} "
             f"over brier={rep2.get('brier')} clim={rep2.get('climatology')}")

    # 6 CLI grammar
    try:
        k1, q1 = parse_question_spec("category:choice=billing,technical@0.6")
        k2, q2 = parse_question_spec("urgency:score=low,mid,high")
        k3, q3 = parse_question_spec("urgent:bool")
        try:
            parse_question_spec("nope:maybe=1")
            bad_rejected = False
        except ValueError:
            bad_rejected = True
        _arm(res, "6 grammar", q1.kind == "choice" and q1.min_confidence == 0.6
             and q2.kind == "score" and q2.options == ("low", "mid", "high")
             and q3.kind == "bool" and q3.options == ("yes", "no") and bad_rejected)
    except Exception as e:  # pragma: no cover
        _arm(res, "6 grammar", False, repr(e))

    # 7 door rung + bridge, against a fake door
    try:
        _arm(res, "7 door", *_door_arm())
    except Exception as e:  # pragma: no cover
        _arm(res, "7 door", False, repr(e))

    print("awdecide --self-test")
    print("\n".join(res))
    ok = all(": OK" in r for r in res)
    print("RESULT:", "OK" if ok else "FAIL")
    return 0 if ok else 2


class _FakeDoor(BaseHTTPRequestHandler):
    """A door that answers from a table keyed on the request kind, and records
    outcomes. Shapes copied from decide.py Decider.decide / outcome."""

    seen: List[dict] = []
    outcomes: List[dict] = []

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path.endswith("/decide/outcome"):
            _FakeDoor.outcomes.append({**body, "auth": self.headers.get("X-WM-Token")})
            reply = {"ok": True, "reward": body.get("reward")}
        else:
            _FakeDoor.seen.append(body)
            kind = body.get("kind")
            if kind == "yesno":
                reply = {"decision_id": "d-yes", "answer": "yes", "confidence": 0.6,
                         "source": "engine", "probabilities": {"yes": 0.7},
                         "probability": 0.7, "probability_source": "outcomes", "p_yes": 0.7,
                         "learned_from": 3}
            elif kind == "score":
                reply = {"decision_id": "d-score", "answer": "high", "confidence": 0.5,
                         "source": "neighbor", "probabilities": {"low": 0.2, "high": 0.65},
                         "probability": 0.65, "probability_source": "outcomes"}
            elif body.get("state") == "cold":
                reply = {"decision_id": "d-none", "answer": body["options"][0],
                         "confidence": 0.0, "source": "none", "probability": None,
                         "probability_source": "none"}
            else:
                # per-option calibrated P(right): does NOT sum to 1 on purpose
                reply = {"decision_id": "d-choice", "answer": "orchestrator", "confidence": 0.8,
                         "source": "engine",
                         "probabilities": {"orchestrator": 0.9, "fast-local": 0.3},
                         "probability": 0.9, "probability_source": "outcomes",
                         "learned_from": 8}
        data = json.dumps(reply).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a: object) -> None:
        pass


def _door_arm() -> "tuple[bool, str]":
    from . import bridge
    from .door import DoorBackend

    srv = HTTPServer(("127.0.0.1", 0), _FakeDoor)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    notes: List[str] = []
    try:
        door = DoorBackend(f"http://127.0.0.1:{srv.server_port}", token="t0k", fork="router")
        lad = Ladder([door])
        c = lad.decide_one("kind:code", Question.choice(["fast-local", "orchestrator"]))
        s = lad.decide_one("x", Question.score(["low", "mid", "high"]))
        b = lad.decide_one("x", Question.bool())
        none = lad.decide_one("cold", Question.choice(["a", "b"]))
        strict = lad.decide_one("kind:code", Question.choice(["fast-local", "orchestrator"],
                                                             min_confidence=0.95))
        kinds = [r.get("kind") for r in _FakeDoor.seen[:3]]
        ok_map = kinds == ["choice", "score", "yesno"] and \
            _FakeDoor.seen[0]["domain"] == "decide.router"
        ok_choice = c.decided and c.value == "orchestrator" and c.probability == 0.9 \
            and c.backend == "door:engine" and c.id == "d-choice" \
            and c.probabilities["fast-local"] == 0.3  # unrenormalized
        ok_score = s.decided and s.value == "high" and s.probability == 0.65 \
            and s.backend == "door:neighbor"
        ok_bool = b.decided and b.value == "yes" and b.probability == 0.7 \
            and abs(b.probabilities["no"] - 0.3) < 1e-9
        ok_none = (not none.decided) and none.value is None and none.backend == "none" \
            and any("door:none" in r for r in none.reasons)
        ok_strict = (not strict.decided) and strict.probabilities["orchestrator"] == 0.9
        notes.append(f"map={ok_map} choice={ok_choice} score={ok_score} bool={ok_bool} "
                     f"none={ok_none} strict={ok_strict}")
        with tempfile.TemporaryDirectory() as td:
            led = Ledger(Path(td) / "door.db")
            did = led.record("route", "kind:code", c)
            r = door.resolve(did, correct=True, ledger=led)
            ok_resolve = did == "d-choice" and r["door"] == {"ok": True, "reward": 1.0} \
                and abs(r["brier"] - 0.01) < 1e-9 and _FakeDoor.outcomes[-1]["auth"] == "t0k"
            # bridge: a door journal with 2 proper pairs, 1 legacy, 1 none, 1 unresolved
            ck = Path(td) / "ckpt"
            ck.mkdir()
            rows = [
                {"kind": "decision", "decision_id": "j1", "domain": "decide.live.r", "state": "s",
                 "answer": "a", "confidence": 0.6, "probability": 0.8,
                 "probability_source": "outcomes", "question_kind": "choice",
                 "source": "engine", "ts": 1.0},
                {"kind": "outcome", "decision_id": "j1", "reward": 1.0, "ts": 2.0},
                {"kind": "decision", "decision_id": "j2", "domain": "decide.live.r", "state": "s",
                 "answer": "yes", "probability": 0.4, "question_kind": "yesno",
                 "source": "prior", "ts": 3.0},
                {"kind": "outcome", "decision_id": "j2", "reward": -1.0, "ts": 4.0},
                {"kind": "outcome", "decision_id": "j2", "reward": 1.0, "ts": 5.0},  # 2nd ignored
                {"kind": "decision", "decision_id": "j3", "domain": "decide.live.r", "state": "s",
                 "answer": "a", "confidence": 0.9, "source": "engine", "ts": 6.0},  # legacy
                {"kind": "outcome", "decision_id": "j3", "reward": 1.0, "ts": 7.0},
                {"kind": "decision", "decision_id": "j4", "domain": "decide.live.r", "state": "s",
                 "answer": "a", "probability": None, "source": "none", "ts": 8.0},
                {"kind": "outcome", "decision_id": "j4", "reward": 1.0, "ts": 9.0},
                {"kind": "decision", "decision_id": "j5", "domain": "decide.live.r", "state": "s",
                 "answer": "a", "probability": 0.5, "source": "llm", "ts": 10.0},  # unresolved
                {"kind": "outcome", "decision_id": None, "reward": 1.0, "ts": 11.0},  # teach row
            ]
            with (ck / "decisions.jsonl").open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
                f.write("{torn\n")
            r1 = bridge.ingest_journal(ck, led, label="t")
            r2 = bridge.ingest_journal(ck, led, label="t")
            rep = led.reliability()
            ok_bridge = (r1["ingested"], r1["legacy_confidence_only"], r1["source_none"],
                         r1["matched"]) == (2, 1, 1, 4) \
                and r2["ingested"] == 0 and r2["already_present"] == 2 \
                and rep["resolved"] == 3 and rep["by_backend"]["door:prior"]["n"] == 1 \
                and abs(rep["by_backend"]["door:prior"]["brier"] - 0.16) < 1e-9
            led.close()
        notes.append(f"resolve={ok_resolve} bridge={ok_bridge}")
        return (ok_map and ok_choice and ok_score and ok_bool and ok_none and ok_strict
                and ok_resolve and ok_bridge), " ".join(notes)
    finally:
        srv.shutdown()
        srv.server_close()
