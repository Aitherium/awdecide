"""The loop as an MCP server (stdio) and as a tiny HTTP service. Stdlib only.

    awdecide mcp      # Claude Code, Codex, Cursor -- any MCP host
    awdecide serve    # POST /decide, /decide/outcome, GET /decide/stats

Four tools: `decide`, `decide_outcome`, `decide_teach`, `decide_stats`. The host's own model
makes the first call of a decision; the ledger answers every repeat.

The brain is optional. With `AWDECIDE_LLM_URL` + `AWDECIDE_LLM_MODEL` set (any
OpenAI-wire endpoint; `AWDECIDE_LLM_KEY` or `OPENAI_API_KEY` for a bearer) an
unseen decision is asked of that model. Without it an unseen decision comes
back `decided=false` and the caller decides -- then teaches the outcome, and
the next sighting is answered from evidence either way.
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from .backends import Backend, Ladder
from .contract import KINDS, Question
from .ledger import Ledger
from .loop import ChatBackend, Loop

PROTOCOL = "2024-11-05"
DEFAULT_PORT = 8297


def env_ladder() -> Ladder:
    rungs: List[Backend] = []
    url, model = os.getenv("AWDECIDE_LLM_URL", ""), os.getenv("AWDECIDE_LLM_MODEL", "")
    if url and model:
        rungs.append(ChatBackend(url, model, token=os.getenv("AWDECIDE_LLM_KEY")
                                 or os.getenv("OPENAI_API_KEY") or ""))
    return Ladder(rungs)


def build_loop(db: Optional[Path] = None) -> Loop:
    return Loop(env_ladder(), Ledger(db))


def _question(a: Dict[str, Any]) -> Question:
    kind = {"yesno": "bool"}.get(str(a.get("kind") or "choice"), str(a.get("kind") or "choice"))
    kw = {"min_confidence": float(a.get("min_confidence") or 0.0),
          "prompt": str(a.get("question") or "")}
    if kind == "bool":
        return Question.bool(**kw)
    return Question(kind, tuple(str(o) for o in (a.get("options") or ())), **kw)


def call(loop: Loop, name: str, a: Dict[str, Any]) -> Dict[str, Any]:
    """One tool call. Raises ValueError on a malformed request."""
    if name == "decide":
        key = str(a.get("fork") or a.get("key") or a.get("domain") or "")
        d = loop.decide(key, a.get("state", ""), _question(a))
        out = d.to_dict()
        out.update({"decision_id": d.id, "answer": d.value, "source": d.backend})
        return out
    if name == "decide_outcome":
        did = str(a.get("decision_id") or "")
        correct = a.get("correct")
        if correct is None:
            reward = float(a.get("reward") or 0.0)
            if reward == 0.0:
                raise ValueError("give correct=true|false (or a non-zero reward)")
            correct = reward > 0
        brier = loop.resolve(did, bool(correct))
        if brier is None:
            raise ValueError(f"unknown decision_id {did!r} (an undecided answer has none; "
                             "teach it with decide_teach)")
        return {"ok": True, "brier": round(brier, 4)}
    if name == "decide_teach":
        key = str(a.get("fork") or a.get("key") or "")
        if not key or not a.get("state") or a.get("value") is None or a.get("correct") is None:
            raise ValueError("decide_teach needs fork, state, value and correct")
        return {"ok": True, "decision_id": loop.teach(key, a["state"], str(a["value"]),
                                                      bool(a["correct"]))}
    if name == "decide_stats":
        return loop.stats()
    raise ValueError(f"unknown tool {name!r}")


_STR = {"type": "string"}
_DECIDE = (
    "Ask a BOUNDED decision (pick one option / yes-no / a level on a scale). If this exact "
    "situation has resolved outcomes it is answered from them (source=evidence, no model "
    "call). `state` MUST be a stable descriptor: the same situation must give the same "
    "string. decided=false means nothing earned an answer -- decide yourself. After acting, "
    "ALWAYS report what happened with decide_outcome (or decide_teach).")
_OUTCOME = ("Report whether a decision turned out RIGHT. This is what makes the next identical "
            "decision free, and what retires a wrong answer for good.")
_TEACH = "Record an outcome for a choice you made yourself (decide returned decided=false)."
_STATS = ("What has been learned: Brier vs the base rate, the calibration buckets, which rung "
          "answered how often, per-fork counts.")
TOOLS = [
    {"name": "decide", "description": _DECIDE,
     "inputSchema": {"type": "object", "required": ["fork", "state"], "properties": {
         "fork": {**_STR, "description": "which decision this is, e.g. 'test-runner'"},
         "state": {**_STR, "description": "stable descriptor, e.g. 'lang:py,changed:tests'"},
         "kind": {**_STR, "enum": list(KINDS) + ["yesno"], "default": "choice"},
         "options": {"type": "array", "items": _STR},
         "question": _STR, "min_confidence": {"type": "number", "default": 0}}}},
    {"name": "decide_outcome", "description": _OUTCOME,
     "inputSchema": {"type": "object", "required": ["decision_id", "correct"], "properties": {
         "decision_id": _STR, "correct": {"type": "boolean"}}}},
    {"name": "decide_teach", "description": _TEACH,
     "inputSchema": {"type": "object", "required": ["fork", "state", "value", "correct"],
                     "properties": {"fork": _STR, "state": _STR, "value": _STR,
                                    "correct": {"type": "boolean"}}}},
    {"name": "decide_stats", "description": _STATS,
     "inputSchema": {"type": "object", "properties": {}}},
]


def handle(loop: Loop, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One JSON-RPC message in, one out; None for a notification."""
    mid, method = msg.get("id"), msg.get("method", "")
    if mid is None:
        return None
    if method == "initialize":
        result: Dict[str, Any] = {
            "protocolVersion": (msg.get("params") or {}).get("protocolVersion") or PROTOCOL,
            "capabilities": {"tools": {}}, "serverInfo": {"name": "awdecide", "version": "1"}}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        p = msg.get("params") or {}
        try:
            text, err = json.dumps(call(loop, str(p.get("name")), p.get("arguments") or {})), False
        except (ValueError, TypeError) as e:
            text, err = str(e), True
        result = {"content": [{"type": "text", "text": text}], "isError": err}
    else:
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": -32601, "message": f"no method {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def stdio(db: Optional[Path] = None) -> int:
    loop = build_loop(db)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            reply = handle(loop, json.loads(line))
        except ValueError:
            reply = {"jsonrpc": "2.0", "id": None,
                     "error": {"code": -32700, "message": "parse error"}}
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()
    return 0


def handler(loop: Loop) -> type:
    routes = {"/decide": "decide", "/decide/outcome": "decide_outcome",
              "/decide/teach": "decide_teach"}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:
            return

        def _send(self, code: int, body: Dict[str, Any]) -> None:
            raw = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, {"status": "ok", "service": "awdecide"})
            elif self.path == "/decide/stats":
                self._send(200, loop.stats())
            else:
                self._send(404, {"detail": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            tool = routes.get(self.path)
            if tool is None:
                self._send(404, {"detail": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                self._send(200, call(loop, tool, json.loads(self.rfile.read(n) or b"{}")))
            except (ValueError, TypeError) as e:
                self._send(422, {"detail": str(e)})
    return H


def serve(port: int = DEFAULT_PORT, host: str = "127.0.0.1", db: Optional[Path] = None) -> int:
    loop = build_loop(db)               # one thread: the ledger's connection is not shared
    srv = HTTPServer((host, port), handler(loop))
    print(f"awdecide: http://{host}:{port}  brain="
          f"{'on' if loop.ladder.backends else 'none'}", file=sys.stderr)
    srv.serve_forever()
    return 0
