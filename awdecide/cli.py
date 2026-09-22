"""awdecide CLI.

    awdecide ask  --state "..." category:choice=billing,technical,sales urgent:bool
                  level:score=low,mid,high [--rule 'refund|charge=billing' ...]
                  [--logprob-url URL --model M] [--record]
                  [--door-fork router [--door-url URL --door-token T]]
    awdecide resolve <id> --correct | --wrong [--door-fork router]
    awdecide reliability
    awdecide ingest-door <ckpt_dir> [--label live] [--no-replay] [--svc-dir D]
    awdecide ingest-door --generate [--decisions 400]   # bench-generated, TEMP ckpt dir
    awdecide mcp                    # stdio MCP server: Claude Code, Codex, Cursor
    awdecide serve [--port 8297]    # the same loop over HTTP
    awdecide bench [--seed N]       # one imperfect brain: called every time vs the loop
    awdecide --self-test

Exit codes: 0 = every question decided · 3 = at least one undecided (fail-closed,
the JSON says why) · 2 = the command could not run. `--self-test` exits 0 only
when every arm passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from . import __version__
from .backends import Ladder, LogprobBackend, RulesBackend, parse_question_spec
from .contract import Question
from .ledger import Ledger


def _read_state(args: argparse.Namespace) -> str:
    if args.state_file:
        return Path(args.state_file).read_text(encoding="utf-8")
    if args.state is not None:
        return args.state
    return sys.stdin.read()


def _build_ladder(args: argparse.Namespace) -> Ladder:
    rungs: List[Any] = []
    if args.rule:
        rb = RulesBackend()
        for spec in args.rule:
            if "=" not in spec:
                raise ValueError(f"bad --rule {spec!r}; want REGEX=option")
            rx, opt = spec.rsplit("=", 1)
            rb.add(rx, opt.strip())
        rungs.append(rb)
    if args.logprob_url:
        rungs.append(LogprobBackend(args.logprob_url, args.model or "default", token=args.token))
    if getattr(args, "door_fork", ""):
        from .door import DoorBackend
        rungs.append(DoorBackend(args.door_url or None, args.door_token or "", args.door_fork,
                                 ckpt_dir=args.door_ckpt or None, svc_dir=args.door_svc or None))
    return Ladder(rungs)


def cmd_ask(args: argparse.Namespace) -> int:
    try:
        questions: Dict[str, Question] = dict(parse_question_spec(s) for s in args.question)
        ladder = _build_ladder(args)
    except ValueError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 2
    state = _read_state(args)
    answers = ladder.decide(state, questions)
    if args.record:
        led = Ledger(Path(args.db) if args.db else None)
        for key, d in answers.items():
            led.record(key, state, d)
        led.close()
    out = {k: d.to_dict() for k, d in answers.items()}
    print(json.dumps(out, indent=2 if not args.compact else None))
    return 0 if all(d.decided for d in answers.values()) else 3


def cmd_resolve(args: argparse.Namespace) -> int:
    led = Ledger(Path(args.db) if args.db else None)
    brier = led.resolve(args.id, correct=bool(args.correct))
    led.close()
    if brier is None:
        print(json.dumps({"error": f"unknown decision id {args.id}"}), file=sys.stderr)
        return 2
    out: Dict[str, Any] = {"id": args.id, "correct": bool(args.correct), "brier": round(brier, 4)}
    if getattr(args, "door_fork", ""):
        # the door made this claim; tell it what happened so it keeps learning
        from .door import DoorBackend
        door = DoorBackend(args.door_url or None, args.door_token or "", args.door_fork,
                           ckpt_dir=args.door_ckpt or None, svc_dir=args.door_svc or None)
        out["door"] = door.resolve(args.id, bool(args.correct))["door"]
    print(json.dumps(out))
    return 0


def cmd_ingest_door(args: argparse.Namespace) -> int:
    """The door's journals -> the ledger, then the gate's own verdict on the result."""
    import tempfile

    from . import bridge

    led = Ledger(Path(args.db) if args.db else None)
    before = len(led.resolved())
    report: Dict[str, Any] = {"ledger": str(led.path), "resolved_before": before}
    try:
        if args.generate:
            scratch = Path(tempfile.mkdtemp(prefix="awdecide-bench-"))
            report["generated"] = bridge.generate_bench(
                scratch, svc_dir=args.svc_dir or None, decisions=args.decisions)
            report["ingest"] = bridge.ingest_door(scratch, led, label="bench", replay=False,
                                                  svc_dir=args.svc_dir or None)
        else:
            if not args.ckpt_dir:
                print(json.dumps({"error": "ingest-door needs <ckpt_dir> or --generate"}),
                      file=sys.stderr)
                return 2
            report["ingest"] = bridge.ingest_door(Path(args.ckpt_dir), led, label=args.label,
                                                  replay=not args.no_replay,
                                                  svc_dir=args.svc_dir or None)
    except Exception as e:  # noqa: BLE001 -- could not run is exit 2, never a silent 0
        report["error"] = f"{type(e).__name__}: {e}"
        print(json.dumps(report, indent=2), file=sys.stderr)
        led.close()
        return 2
    report["resolved_after"] = len(led.resolved())
    report["reliability"] = led.reliability()
    led.close()
    print(json.dumps(report, indent=2))
    return 0 if report["resolved_after"] > before or before > 0 else 3


def cmd_reliability(args: argparse.Namespace) -> int:
    led = Ledger(Path(args.db) if args.db else None)
    rep = led.reliability()
    led.close()
    print(json.dumps(rep, indent=2))
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp import stdio
    return stdio(Path(args.db) if args.db else None)


def cmd_serve(args: argparse.Namespace) -> int:
    from .mcp import serve
    return serve(args.port, args.host, Path(args.db) if args.db else None)


def cmd_bench(args: argparse.Namespace) -> int:
    from .loop import bench
    print(json.dumps(bench(args.states, args.repeats, args.brain_acc, seed=args.seed), indent=2))
    return 0


def _door_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--door-fork", default="",
                    help="add the world-model decision door as a rung, at this fork "
                         "(domain decide.<fork>)")
    sp.add_argument("--door-url", default="",
                    help="door over HTTP (gateway base ending /v1, or the service base); "
                         "default: in-process when the service tree is importable, else "
                         "$AITHER_DECIDE_URL")
    sp.add_argument("--door-token", default="", help="Bearer / X-WM-Token for the door")
    sp.add_argument("--door-ckpt", default="", help="in-process: AITHER_WM_CKPT_DIR")
    sp.add_argument("--door-svc", default="", help="in-process: AITHER_WM_SVC_DIR")


def cmd_door_bench(args: argparse.Namespace) -> int:
    from .local import run_bench

    rest = [a for a in (args.rest or []) if a != "--"]
    return run_bench(args.name, rest, ckpt_dir=args.ckpt_dir, json_out=args.json)


def main(argv: List[str] | None = None) -> int:
    # GENERATED doctor intercept (gen_aw_doctor.py) -- do not edit
    _dv = locals().get("argv")
    if (_dv if _dv is not None else __import__("sys").argv[1:])[:1] == ["doctor"]:
        from ._doctor import report
        return report()
    # GENERATED repo-state intercept (gen_aw_doctor.py) -- do not edit
    try:
        from awgit import state as _aw_state
    except Exception:
        _aw_state = None
    if _aw_state is not None:
        _sv = locals().get("argv")
        if _aw_state.cli_banner(_sv if _sv is not None else __import__("sys").argv[1:]):
            return 0
    p = argparse.ArgumentParser(prog="awdecide", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"awdecide {__version__}")
    p.add_argument("--self-test", action="store_true", help="prove the contract can fail")
    p.add_argument("--db", default="",
                   help="ledger path (default $AWDECIDE_DB or ~/.aither/awdecide.db)")
    sub = p.add_subparsers(dest="command")

    a = sub.add_parser("ask", help="ask typed questions of one state")
    a.add_argument("question", nargs="+",
                   help="key:choice=a,b | key:score=l1,l2 | key:bool [@min_conf]")
    a.add_argument("--state", default=None)
    a.add_argument("--state-file", default="")
    a.add_argument("--rule", action="append", default=[], help="REGEX=option (rules rung)")
    a.add_argument("--logprob-url", default="", help="OpenAI-wire base URL with logprobs")
    a.add_argument("--model", default="")
    a.add_argument("--token", default="")
    a.add_argument("--record", action="store_true", help="record decided answers in the ledger")
    a.add_argument("--compact", action="store_true")
    _door_args(a)
    a.set_defaults(fn=cmd_ask)

    r = sub.add_parser("resolve", help="resolve a recorded decision against its outcome")
    r.add_argument("id")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--correct", action="store_true")
    g.add_argument("--wrong", action="store_false", dest="correct")
    _door_args(r)
    r.set_defaults(fn=cmd_resolve)

    i = sub.add_parser("ingest-door",
                       help="the world-model door's journals -> this ledger (idempotent)")
    i.add_argument("ckpt_dir", nargs="?", default="",
                   help="the door's AITHER_WM_CKPT_DIR (decisions.jsonl + domain-decide.*)")
    i.add_argument("--label", default="live", help="key prefix for the rows (default live)")
    i.add_argument("--no-replay", action="store_true",
                   help="skip the prequential replay of the per-fork transitions journals")
    i.add_argument("--generate", action="store_true",
                   help="run the door's benches in-process into a TEMP ckpt dir and ingest that "
                        "(label bench)")
    i.add_argument("--decisions", type=int, default=400, help="--generate: router decisions")
    i.add_argument("--svc-dir", default="", help="door service tree (AITHER_WM_SVC_DIR)")
    i.set_defaults(fn=cmd_ingest_door)

    s = sub.add_parser("reliability", help="Brier vs base rate and the bucket table")
    s.set_defaults(fn=cmd_reliability)

    m = sub.add_parser("mcp", help="stdio MCP server (Claude Code, Codex, Cursor)")
    m.set_defaults(fn=cmd_mcp)
    v = sub.add_parser("serve", help="the loop over HTTP: POST /decide, /decide/outcome")
    v.add_argument("--port", type=int, default=8297)
    v.add_argument("--host", default="127.0.0.1")
    v.set_defaults(fn=cmd_serve)
    b = sub.add_parser("bench", help="one imperfect brain: called every time vs the loop")
    b.add_argument("--states", type=int, default=40)
    b.add_argument("--repeats", type=int, default=10)
    b.add_argument("--brain-acc", type=float, default=0.70)
    b.add_argument("--seed", type=int, default=7)
    b.set_defaults(fn=cmd_bench)
    db = sub.add_parser("door-bench",
                        help="run one of the door's own benches from the installed package "
                             "(judge-holdout, compact, cache-cost, calibration, judge)")
    db.add_argument("name")
    db.add_argument("--ckpt-dir", default=None,
                    help="scratch journal dir (default ~/.awdecide/bench-ckpt)")
    db.add_argument("--json", action="store_true")
    db.add_argument("rest", nargs=argparse.REMAINDER, help="passed through to the bench")
    db.set_defaults(fn=cmd_door_bench)

    args = p.parse_args(argv)
    if args.self_test:
        from ._selftest import run_self_test
        return run_self_test()
    if not args.command:
        p.print_help()
        return 2
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
