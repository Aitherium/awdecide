#!/usr/bin/env python
"""compact_bench -- the honest measurement of compact.py on REAL tool output.

Corpus: tool outputs generated on this box (tools/compact_corpus/, regenerate
with --regen): pytest on a passing and on a failing test file, ruff over a dirty
directory, a podman build log, `git log --stat`, `curl -v` against a refusing
port and against a 401. Each carries a HAND-LABELLED set of must-keep lines
(the error line, the summary line, the failing test name, the exit code, the
HTTP status, the commit subject) written as regexes in LABELS below so a
regenerated corpus keeps its labels; a label that matches no line is a dead
label and the bench exits 2 rather than passing on it.

Legs (all in-process, one FRESH journal dir per run so the bench never teaches
the fleet's door by accident -- pass --ckpt-dir to make it):

  rules-only    no door: always-keep rules + the run squeeze
  door-cold     the door with no evidence on these forks (engine empty; the LLM
                rung on with --llm, otherwise off -> `source=none` -> keep)
  door-taught   the same door after ONE teaching pass: every cold decision was
                graded against the labels (compact.grade) and posted back with
                compact_outcome, wrong verdicts with their counterfactual

Metrics per item and leg: lines before/after, tokens before/after (a 4-chars-
per-token ESTIMATE, not a tokenizer count), % lines dropped, recall of the
must-keep lines. Recall below 100% on ANY gating leg = the run fails (exit 1).

Diagnostic, reported and never gating: `door-only` (always-keep rules OFF, cold
and taught) -- what the door alone would have dropped, i.e. why the rules exist.

Also timed: compact() on a synthetic 1,000-line pytest output, rules-only and
door-warm, ms per call after a warm-up.

Exit: 0 recall 100% on every gating leg, 1 a leg lost a must-keep line,
2 could not run (corpus missing, door not importable, dead label).
--json prints the table as JSON; --self-test proves the bench can fail.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
SVC = HERE.parent
sys.path.insert(0, str(SVC))

import compact  # noqa: E402

CORPUS = HERE / "compact_corpus"
TOKEN_CHARS = 4  # an estimate; said in every table this prints

# name -> (tool_name, [must-keep regexes]). Every regex must match >= 1 line.
LABELS: Dict[str, Any] = {
    "pytest_pass.txt": ("pytest", [r"^=+ \d+ passed in .* =+$"]),
    "pytest_fail.txt": (
        "pytest",
        [
            r"^FAILED \S+::test_boom",
            r"^FAILED \S+::test_key_error",
            r"^E\s+assert 1 == 2",
            r"^E\s+KeyError: 'missing'",
            r"^=+ \d+ failed, \d+ passed in .* =+$",
        ],
    ),
    "ruff.txt": ("ruff", [r"^Found \d+ errors?\.?$"]),
    "git_log_stat.txt": (
        "git",
        [
            r"^commit [0-9a-f]{40}$",  # every commit header
            r"^    (docs|fix|feat|chore|refactor|test|perf|build|ci)\(",  # subjects
            r"^ \d+ files? changed",  # every commit's stat summary
        ],
    ),
    "podman_build.txt": (
        "podman",
        [r"^COMMIT ", r"^Successfully tagged", r"^exit code: \d+$"],
    ),
    "curl_401.txt": (
        "curl",
        [r"^< HTTP/1\.1 401", r'"detail":"missing bearer token"', r"^exit code: \d+$"],
    ),
    "curl_refused.txt": (
        "curl",
        [r"^curl: \(7\)", r"Connection refused", r"^exit code: \d+$"],
    ),
}

GATING_LEGS = ("rules-only", "door-cold", "door-taught")


class BenchError(RuntimeError):
    """Could not run (exit 2)."""


# ----------------------------------------------------------------- corpus
def regen(corpus: Path, repo: Path) -> None:
    """Regenerate the corpus by running the real commands on this box. Slow
    (the passing pytest set takes ~3 min); items whose command is unavailable
    are left as they were and named."""
    corpus.mkdir(parents=True, exist_ok=True)
    aos = repo / "AitherOS"
    scratch = Path(tempfile.mkdtemp(prefix="compact-bench-"))
    fail_dir = scratch / "failsuite"
    fail_dir.mkdir()
    (fail_dir / "test_scratch_fail.py").write_text(
        (corpus / "test_scratch_fail.py.txt").read_text(encoding="utf-8"), encoding="utf-8"
    )

    def run(cmd: Sequence[str], cwd: Path, out: Path, shell: bool = False, tail_exit: bool = False):
        try:
            p = subprocess.run(
                cmd, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=900, shell=shell,
            )
            text = p.stdout + p.stderr
            if tail_exit:
                text += f"exit code: {p.returncode}\n"
            out.write_text(text, encoding="utf-8")
            print(f"  regenerated {out.name}: {text.count(chr(10))} lines (rc {p.returncode})")
        except Exception as exc:  # noqa: BLE001 -- a missing tool leaves the old file
            print(f"  could not regenerate {out.name}: {exc}")

    tests = sorted((aos / "dev" / "tests").glob("test_check_*.py"))[:12]
    run([sys.executable, "-m", "pytest", *[str(t) for t in tests], "-v", "-p",
         "no:cacheprovider", "--tb=short"], aos, corpus / "pytest_pass.txt")
    run([sys.executable, "-m", "pytest", "test_scratch_fail.py", "-v", "-p", "no:cacheprovider",
         "--tb=short"], fail_dir, corpus / "pytest_fail.txt")
    run(["ruff", "check", "--isolated", "--select", "E,F,W,I", "--no-fix", "--output-format",
         "concise", "lib/quality"], aos, corpus / "ruff.txt")
    run(["git", "log", "--stat", "-n", "40"], repo, corpus / "git_log_stat.txt")
    run(["curl", "-v", "-m", "3", "http://127.0.0.1:9/nothing"], repo,
        corpus / "curl_refused.txt", tail_exit=True)
    run(["curl", "-v", "-m", "5", "http://127.0.0.1:8299/v1/decide/batch", "-H",
         "Content-Type: application/json", "-d", '{"items":[]}'], repo,
        corpus / "curl_401.txt", tail_exit=True)
    print("  podman_build.txt is not regenerated here (needs the WSL distro); see the file's "
          "first line for the build it came from")
    shutil.rmtree(scratch, ignore_errors=True)


def load_corpus(corpus: Path) -> List[Dict[str, Any]]:
    items = []
    for name, (tool, patterns) in LABELS.items():
        path = corpus / name
        if not path.is_file():
            raise BenchError(f"corpus file missing: {path} (run with --regen)")
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        must: List[int] = []
        for pat in patterns:
            rx = re.compile(pat)
            hits = [i for i, ln in enumerate(lines) if rx.search(ln)]
            if not hits:
                raise BenchError(f"dead label in {name}: /{pat}/ matches no line")
            must.extend(hits)
        items.append({"name": name, "tool": tool, "text": text,
                      "must_keep": sorted(set(must)), "lines": len(compact.shapes(text))})
    return items


# ------------------------------------------------------------------- legs
def measure(item: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    text = item["text"]
    rc = compact.recall(result, item["must_keep"], text)
    before, after = compact.estimate_tokens(text), compact.estimate_tokens(result["kept"])
    total = result["total_lines"] or 1
    return {
        "lines_before": result["total_lines"],
        "lines_after": result["kept_lines"],
        "tokens_before_est": before,
        "tokens_after_est": after,
        "drop_pct": round(100.0 * result["dropped"] / total, 1),
        "recall": rc["recall"],
        "missing": rc["missing"],
        "sources": result.get("source_counts", {}),
        "distinct_shapes": result.get("distinct_shapes", 0),
        "latency_ms": result.get("latency_ms"),
    }


def run_legs(items: List[Dict[str, Any]], decider: Any, *, diagnostics: bool = True
             ) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """{leg: {item: metrics}}. `decider` is the cold door; it is taught in place
    between door-cold and door-taught."""
    table: Dict[str, Dict[str, Dict[str, Any]]] = {leg: {} for leg in GATING_LEGS}
    if diagnostics:
        table["door-only-cold (diagnostic)"] = {}
        table["door-only-taught (diagnostic)"] = {}
    cold_results: Dict[str, Dict[str, Any]] = {}
    for it in items:
        table["rules-only"][it["name"]] = measure(it, compact.compact(it["text"], it["tool"]))
        r = compact.compact(it["text"], it["tool"], decider=decider)
        cold_results[it["name"]] = r
        table["door-cold"][it["name"]] = measure(it, r)
        if diagnostics:
            table["door-only-cold (diagnostic)"][it["name"]] = measure(
                it, compact.compact(it["text"], it["tool"], decider=decider,
                                    always_keep_rules=False))
    # ONE teaching pass, from the cold decisions
    taught = 0
    for it in items:
        r = cold_results[it["name"]]
        grades = compact.grade(r, it["must_keep"])
        res = compact.compact_outcome(r["decisions"], grades, decider, tool_name=it["tool"])
        taught += sum(1 for x in res if "error" not in x)
        # what the door-only diagnostic would need: every shape's truth, so the
        # taught diagnostic measures a door that was told about must-keep shapes
        if diagnostics:
            rows = compact.shapes(it["text"])
            mk = set(it["must_keep"])
            for shape in {row["shape"] for row in rows}:
                keep = any(row["i"] in mk for row in rows if row["shape"] == shape)
                try:
                    compact.teach_shape(decider, it["tool"], shape, keep)
                except Exception:  # noqa: BLE001 -- diagnostic only
                    pass
    for it in items:
        table["door-taught"][it["name"]] = measure(
            it, compact.compact(it["text"], it["tool"], decider=decider))
        if diagnostics:
            table["door-only-taught (diagnostic)"][it["name"]] = measure(
                it, compact.compact(it["text"], it["tool"], decider=decider,
                                    always_keep_rules=False))
    table["_taught_decisions"] = taught  # type: ignore[assignment]
    return table


def time_1000(decider: Any, reps: int = 5) -> Dict[str, float]:
    text = compact._pytest_like(1000)
    out: Dict[str, float] = {}
    compact.compact(text, "pytest")
    t0 = time.perf_counter()
    for _ in range(reps):
        compact.compact(text, "pytest")
    out["rules_only_ms"] = round((time.perf_counter() - t0) * 1000 / reps, 1)
    # warm the door: teach every shape once so the engine answers
    r = compact.compact(text, "pytest", decider=decider)
    compact.compact_outcome(r["decisions"], {k: True for k in
                            [d["decision_id"] for d in r["decisions"]]}, decider,
                            tool_name="pytest")
    compact.compact(text, "pytest", decider=decider)
    t0 = time.perf_counter()
    for _ in range(reps):
        r = compact.compact(text, "pytest", decider=decider)
    out["door_warm_ms"] = round((time.perf_counter() - t0) * 1000 / reps, 1)
    out["door_warm_sources"] = r["source_counts"]  # type: ignore[assignment]
    out["lines"] = r["total_lines"]  # type: ignore[assignment]
    return out


# ----------------------------------------------------------------- report
def render(table: Dict[str, Any], timing: Optional[Dict[str, Any]], llm: bool) -> str:
    lines = []
    lines.append("compact_bench -- tokens are a 4-chars-per-token ESTIMATE, not a tokenizer count")
    lines.append(f"door LLM rung: {'ON' if llm else 'OFF (cold = source none = keep)'}")
    for leg, rows in table.items():
        if leg.startswith("_"):
            continue
        lines.append("")
        lines.append(f"[{leg}]")
        lines.append(f"  {'item':<20}{'lines':>12}{'tokens~':>14}"
                     f"{'drop%':>7}{'recall':>8}  sources")
        for name, m in rows.items():
            lines.append(
                f"  {name:<20}{m['lines_before']:>5}->{m['lines_after']:<5}"
                f"{m['tokens_before_est']:>6}->{m['tokens_after_est']:<6}"
                f"{m['drop_pct']:>6.1f}%{m['recall']*100:>7.1f}%  {m['sources']}"
            )
            for miss in m["missing"][:3]:
                lines.append(f"      MISSING line {miss['i']}: {miss['line'][:100]}")
        tb = sum(m["tokens_before_est"] for m in rows.values())
        ta = sum(m["tokens_after_est"] for m in rows.values())
        lb = sum(m["lines_before"] for m in rows.values())
        la = sum(m["lines_after"] for m in rows.values())
        worst = min(m["recall"] for m in rows.values()) if rows else 1.0
        lines.append(f"  TOTAL lines {lb}->{la} ({100.0 * (lb - la) / max(1, lb):.1f}% dropped), "
                     f"tokens~ {tb}->{ta} ({100.0 * (tb - ta) / max(1, tb):.1f}% saved), "
                     f"worst recall {worst * 100:.1f}%")
    if timing:
        lines.append("")
        lines.append(f"[timing, {timing['lines']}-line synthetic pytest output, in-process]")
        lines.append(f"  rules-only {timing['rules_only_ms']} ms/call; door warm "
                     f"{timing['door_warm_ms']} ms/call (sources {timing['door_warm_sources']})")
    lines.append("")
    lines.append(f"teaching pass posted {table.get('_taught_decisions', 0)} outcomes")
    return "\n".join(lines)


def verdict(table: Dict[str, Any]) -> int:
    for leg in GATING_LEGS:
        for m in table.get(leg, {}).values():
            if m["recall"] < 1.0:
                return 1
    return 0


# -------------------------------------------------------------- self-test
def _self_test() -> int:
    """The bench must FAIL when a door drops a labelled line. Runs on the
    synthetic fixture with scripted doors; no service import needed."""
    fails: List[str] = []

    def check(name: str, ok: bool) -> None:
        print(("ok   " if ok else "FAIL ") + name)
        if not ok:
            fails.append(name)

    text = compact._pytest_like(300)
    rows = compact.shapes(text)
    must = [i for i, r in enumerate(rows) if r["line"].startswith("FAILED ")
            or "1 failed" in r["line"]]
    item = {"name": "synthetic", "tool": "pytest", "text": text, "must_keep": must,
            "lines": len(rows)}
    m = measure(item, compact.compact(text, "pytest"))
    check("rules-only on the fixture: recall 100%", m["recall"] == 1.0)
    check("tokens are estimated at 4 chars/token",
          m["tokens_before_est"] == (len(text) + 3) // TOKEN_CHARS)

    # a door that drops everything WITH rules off must lose the labels -> the
    # bench must notice (this is the diagnostic leg's whole point)
    drop_all = compact._ScriptedDoor("no")
    m2 = measure(item, compact.compact(text, "pytest", decider=drop_all,
                                       always_keep_rules=False))
    check("door-only drop-all loses must-keep lines (recall < 1)", m2["recall"] < 1.0)
    check("verdict() fails a table whose gating leg lost a line",
          verdict({"rules-only": {"synthetic": m}, "door-cold": {"synthetic": m2},
                   "door-taught": {"synthetic": m}}) == 1)
    check("verdict() passes a clean table",
          verdict({leg: {"synthetic": m} for leg in GATING_LEGS}) == 0)

    # a dead label is exit 2, not a pass
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        for name in LABELS:
            (p / name).write_text("nothing here\n" * 30, encoding="utf-8")
        try:
            load_corpus(p)
            check("dead label raises BenchError", False)
        except BenchError as exc:
            check("dead label raises BenchError", "dead label" in str(exc))
        (p / "pytest_pass.txt").unlink()
        try:
            load_corpus(p)
            check("missing corpus file raises BenchError", False)
        except BenchError as exc:
            check("missing corpus file raises BenchError", "missing" in str(exc))

    # the real corpus, when present, has live labels
    try:
        items = load_corpus(CORPUS)
        check("shipped corpus loads with every label matching",
              len(items) == len(LABELS) and all(it["must_keep"] for it in items))
    except BenchError as exc:
        check(f"shipped corpus loads ({exc})", False)

    # render never crashes on an empty timing
    txt = render({"rules-only": {"synthetic": m}, "door-cold": {"synthetic": m2},
                  "door-taught": {"synthetic": m}, "_taught_decisions": 0}, None, False)
    check("render prints the ESTIMATE disclaimer", "ESTIMATE" in txt)
    print(f"\n{'PASS' if not fails else 'FAIL'}: {len(fails)} failing check(s)")
    return 1 if fails else 0


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--repo", default=os.environ.get("AITHER_REPO", r"C:\source"))
    ap.add_argument("--regen", action="store_true", help="re-run the commands that made the corpus")
    ap.add_argument("--llm", action="store_true",
                    help="cold leg asks the local brain (MicroScheduler) for unseen shapes")
    ap.add_argument("--llm-url", default=os.environ.get("AITHER_DECIDE_LLM_URL",
                                                        "https://127.0.0.1:8150/v1"))
    ap.add_argument("--llm-timeout", type=float, default=60.0)
    ap.add_argument("--ckpt-dir", help="journal dir for the door (default: a fresh temp dir)")
    ap.add_argument("--no-diagnostics", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return _self_test()
    corpus = Path(a.corpus)
    if a.regen:
        regen(corpus, Path(a.repo))
    try:
        items = load_corpus(corpus)
    except BenchError as exc:
        print(f"[compact_bench] cannot run: {exc}", file=sys.stderr)
        return 2
    tmp = None
    if a.ckpt_dir:
        os.environ["AITHER_WM_CKPT_DIR"] = a.ckpt_dir
    else:
        tmp = tempfile.mkdtemp(prefix="compact-bench-ckpt-")
        os.environ["AITHER_WM_CKPT_DIR"] = tmp
    os.environ["AITHER_DECIDE_LLM_URL"] = a.llm_url
    os.environ["AITHER_DECIDE_LLM_TIMEOUT"] = str(a.llm_timeout)
    os.environ.setdefault("AITHER_DECIDE_EMBED_URL", "")  # the neighbor rung is off here
    try:
        decider = compact.in_process_decider(llm_enabled=bool(a.llm), embed_enabled=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[compact_bench] cannot run: in-process door unavailable ({exc})", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    table = run_legs(items, decider, diagnostics=not a.no_diagnostics)
    timing = None if a.no_timing else time_1000(decider)
    wall = round(time.perf_counter() - t0, 1)
    code = verdict(table)
    if a.json:
        print(json.dumps({"table": table, "timing": timing, "llm": bool(a.llm),
                          "wall_s": wall, "exit": code,
                          "ckpt_dir": os.environ["AITHER_WM_CKPT_DIR"],
                          "token_estimate": f"{TOKEN_CHARS} chars/token"},
                         indent=1, default=str))
    else:
        print(render(table, timing, bool(a.llm)))
        print(f"wall {wall}s; journal {os.environ['AITHER_WM_CKPT_DIR']}")
        print("RESULT: " + ("PASS -- recall 100% on every gating leg" if code == 0
                            else "FAIL -- a gating leg dropped a must-keep line"))
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
