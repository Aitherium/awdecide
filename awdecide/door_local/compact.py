"""compact -- per-line keep/drop over long tool output, on the decision door.

The most-upvoted use of Jev this week (r/ClaudeCode, 2026-09-19): a hook that
asks a classifier PER LINE of a long tool output whether the model needs to see
it, so a 1,000-line pytest run reaches the model as "42 passed" plus the lines
that matter. It never reads prose; it classifies SHAPES. That is a keep/drop
yes/no fork on a repeating shape -- exactly the decision door's home turf --
with one thing a static classifier cannot do: when a dropped line turns out to
matter, the correction teaches the door and that shape is kept from then on.

    compact(text, "pytest") -> {kept, dropped, kept_lines, total_lines,
                                decisions, source_counts, ...}

How a line is handled, in order:

1. ALWAYS-KEEP, deterministic, never sent to the door: the first 2 lines, the
   last 5 lines, any traceback frame, any summary line (N passed/failed, exit
   code, `error:`, FAILED/ERROR, `N files changed`) and any line carrying an
   error word. Dropping one of those on a model's say-so is the one failure this
   thing is not allowed to have, so no model gets asked.
2. Everything else is turned into a STABLE shape descriptor (the idea in
   judge.features): kind of line (dots / pytest-pass / path:line / timestamped
   log / diffstat / curl chatter / blank / other), bucketed length, how many
   times the previous line had the same kind, and position (first/middle/last
   third). One yesno item per DISTINCT shape -- not per line -- goes to the
   door as ONE decide_batch at fork `decide.compact.<tool>`. A shape the door
   has learned is answered by the engine in microseconds; a new one falls to
   the local brain; `source=none` (nothing to go on) KEEPS the line.
3. Lines the door kept (or could not judge) are then squeezed by a rule that
   cannot touch an always-keep line: a run of >= `collapse_runs` consecutive
   lines of the same kind keeps its first two and last one. Runs of dropped
   lines collapse into one marker: `... (N similar lines dropped)`.

The question sent to the door names the SHAPE only -- never the line's text --
so nothing from a tool's output (an env dump, a token in a URL) leaves the
process through the LLM rung.

Modes: in-process (`in_process_decider()`: imports decide + code_domains from
this tree with AITHER_WM_CKPT_DIR pointing at the door's journal) or remote
(`RemoteDecider(url)`: posts /decide/batch and /decide/outcome to the in-fleet
door, or to the public gateway's /v1 prefix with a bearer).

Teaching: `compact_outcome(decisions, verdict_was_right, decider)` posts the
reward for each decision id AND, when the verdict was wrong, the counterfactual
answer as a positive example -- the engine picks the best KNOWN answer, so a
lone negative on "yes" would still pick "yes" (test_decide teaches the
alternative explicitly for the same reason). `grade(result, must_keep)` turns a
hand-labelled set of line indexes into those verdicts; tools/compact_bench.py
is the honest measurement.

Pure stdlib. `python compact.py --self-test` proves the invariants can fail.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

FORK_PREFIX = "decide.compact."
ALWAYS_KEEP_HEAD = 2
ALWAYS_KEEP_TAIL = 5
DEFAULT_BUDGET_LINES = 40
DEFAULT_COLLAPSE_RUNS = 4
MAX_BATCH = 64  # decide.Decider.decide_batch refuses more; shapes are chunked

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")
_TRACEBACK = re.compile(
    r"^Traceback \(most recent call last\)|^\s+File \".+\", line \d+|"
    r"^\s+at .+\(.+:\d+\)|^\s+at .+:\d+:\d+|^\s*[\^~]{2,}\s*$|^Error was raised",
    re.I,
)
_SUMMARY = re.compile(
    r"\b\d+\s+(passed|failed|errors?|skipped|warnings?|deselected|xfailed|xpassed|"
    r"subtests?)\b|exit(?:ed)?(?:\s+with)?(?:\s+code)?\s*[=: ]\s*-?\d+|"
    r"\berror\s*:|^\s*E\s{2,}|^(FAILED|ERROR|PASSED|XFAIL)\b\s+\S+::|"
    r"^=+\s.*\s=+$|^Found \d+ errors?|All checks passed|^\s*\d+ files? changed|"
    r"^npm ERR!|^error\[|^\s*(fatal|panic)\b|^curl: \(\d+\)|"
    r"^(FAIL|ERROR|FAILED)\b|Ran \d+ tests?|^OK\b|^FAILED \("
    # A pytest progress line whose verdict is the single letter F or E:
    # `tests/x.py::test_y F  [ 48%]`. Its SHAPE is byte-identical to the 310
    # PASSED lines around it (a token run collapses `PASSED` and `F` alike), so
    # without this the one line naming the failing test is swallowed into
    # "... (305 similar lines dropped)" -- the model is told something failed and
    # cannot be told what. Found by the adversarial pass 2026-09-20; the bench
    # could not see it because no corpus item carried this shape. Classing it as
    # a summary also puts it in ALWAYS_KEEP_KINDS, so a confidently-wrong door
    # cannot drop it either.
    r"|::\S+\s+[FE]\b|^\S+\.py\s+[.sx]*[FE][.sxFE]*\s*(?:\[\s*\d+%\])?\s*$",
    re.I | re.M,
)
_ERROR_WORD = re.compile(
    r"\b(error|errors|exception|failed|failure|fatal|denied|refused|reset|timed out|"
    r"timeout|cannot|can't|not found|no such|missing|invalid|unable|aborted|abort|"
    r"killed|oom|panic|segfault|segmentation|critical|traceback|assert|assertion|"
    r"unauthori[sz]ed|forbidden|conflict|rejected|corrupt)\b",
    re.I,
)
_DOTS = re.compile(r"^[.FsExX]{3,}\s*(\[\s*\d+%\])?\s*$|^[\s.]{3,}$|^[#=>\-\s\[\]]{4,}\d{1,3}%.*$")
_PYTEST_PASS = re.compile(r"\bPASSED\b|\bSKIPPED\b|\bXPASS\b|::\S+\s+\.\s*$|\bok\b\s*$")
_PATHLINE = re.compile(r"^[\w./\\~-]+\.\w{1,5}:\d+(:\d+)?\b")
_LOG = re.compile(
    r"^\[?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}|^\[?\d{2}:\d{2}:\d{2}[.,]?\d*\]?\s|"
    r"^\[?(INFO|DEBUG|WARN|WARNING|TRACE)\]?[:\s]"
)
_DIFFSTAT = re.compile(r"^\s\S.*\|\s+\d+\s*[+\-]*\s*$|^\s\S.*\|\s+Bin\b")
_CURL = re.compile(r"^[*<>}{] |^\* |^[<>] $")
_GIT_HDR = re.compile(r"^(commit [0-9a-f]{7,40}|Author:|Date:|Merge:)\s?")
_BUILD = re.compile(
    r"^(STEP|Step) \d+(/\d+)?[: ]|^--> |^\[\d+/\d+\] |^Successfully\b|^COMMIT\b|"
    r"^Getting image source"
)
_PODMAN_LAYER = re.compile(r"^(Copying blob|Copying config|Writing manifest|Storing signatures)")


def _len_bucket(n: int) -> str:
    for edge in (40, 80, 160):
        if n < edge:
            return f"<{edge}"
    return ">=160"


def _rep_bucket(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n < 5:
        return "2-4"
    return "5-19" if n < 20 else "20+"


def line_kind(line: str) -> str:
    """The coarse class of ONE line (ANSI stripped). Order matters: the
    always-keep classes come first so a PASSED line that also says `error:`
    is a summary, not a pass."""
    s = _ANSI.sub("", line).rstrip()
    if not s.strip():
        return "blank"
    if _TRACEBACK.search(s):
        return "tb"
    if _SUMMARY.search(s):
        return "summary"
    if _ERROR_WORD.search(s):
        return "err"
    if _DOTS.match(s):
        return "dots"
    if _PYTEST_PASS.search(s):
        return "pytest-pass"
    if _PATHLINE.match(s):
        return "pathline"
    if _LOG.match(s):
        return "log"
    if _DIFFSTAT.match(s):
        return "diffstat"
    if _GIT_HDR.match(s):
        return "git-hdr"
    if _PODMAN_LAYER.match(s):
        return "layer"
    if _BUILD.match(s):
        return "build-step"
    if _CURL.match(s):
        return "curl"
    return "other"


ALWAYS_KEEP_KINDS = frozenset({"tb", "summary", "err"})


def shapes(text: str) -> List[Dict[str, Any]]:
    """One row per line: {i, line, kind, shape, always_keep, reason}. `shape` is
    the STABLE descriptor the door is keyed on; two outputs of the same kind of
    run produce the same set of shapes even though no line repeats."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    n = len(lines)
    rows: List[Dict[str, Any]] = []
    prev_kind, rep = None, 0
    for i, line in enumerate(lines):
        kind = line_kind(line)
        rep = rep + 1 if kind == prev_kind else 0
        prev_kind = kind
        pos = "first" if i * 3 < n else ("last" if i * 3 >= 2 * n else "middle")
        reason = None
        if i < ALWAYS_KEEP_HEAD:
            reason = "head"
        elif i >= n - ALWAYS_KEEP_TAIL:
            reason = "tail"
        elif kind in ALWAYS_KEEP_KINDS:
            reason = kind
        shape = f"kind:{kind}|len:{_len_bucket(len(line))}|rep:{_rep_bucket(rep)}|pos:{pos}"
        rows.append(
            {
                "i": i,
                "line": line,
                "kind": kind,
                "shape": shape,
                "always_keep": reason is not None,
                "reason": reason,
            }
        )
    return rows


def _domain(tool_name: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", (tool_name or "tool").strip().lower()).strip("_")
    return FORK_PREFIX + (slug or "tool")


def _question(tool_name: str, shape: str) -> str:
    # Shape words only. The line's text never leaves the process this way.
    return (
        f"A line of `{tool_name}` output has this shape: {shape}. Should the model "
        "that ran the tool see this line, or is it filler it can do without (dots, "
        "a passing test, a routine log line, a diffstat row)? yes = keep, no = drop."
    )


class RemoteDecider:
    """A Decider-shaped client over HTTP: the in-fleet door
    (http://127.0.0.1:8197) or the public gateway (…/v1). Only the
    two methods compact() needs."""

    def __init__(
        self, url: str, *, token: Optional[str] = None, timeout: float = 90.0,
        ca_bundle: Optional[str] = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.token = (
            token
            or os.environ.get("AITHER_DECIDE_TOKEN")
            or os.environ.get("AITHER_WM_INTERNAL_TOKEN")
            or ""
        )
        self.timeout = timeout
        self.ca_bundle = ca_bundle or os.environ.get("SSL_CERT_FILE") or os.environ.get(
            "REQUESTS_CA_BUNDLE"
        )

    def _ctx(self):
        if not self.url.startswith("https"):
            return None
        import ssl

        ctx = ssl.create_default_context()
        if self.ca_bundle and os.path.isfile(self.ca_bundle):
            ctx.load_verify_locations(self.ca_bundle)
        return ctx

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
            headers["X-WM-Token"] = self.token
        req = urllib.request.Request(
            self.url + path, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx()) as r:
                return json.loads(r.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"door HTTP {e.code} on {path}: {e.read().decode('utf-8', 'replace')[:200]}"
            ) from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise RuntimeError(f"door unreachable at {self.url}{path}: {e}") from None

    def decide_batch(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._post("/decide/batch", {"items": items})

    def outcome(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._post("/decide/outcome", body)


def in_process_decider(**kw: Any):
    """The door in this process: decide.Decider over code_domains.DomainEngines,
    reading/writing the journal under AITHER_WM_CKPT_DIR. Raises ImportError when
    the world_model package is not importable here (the caller falls back to a
    RemoteDecider or to rules-only; it never fakes a door)."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import code_domains  # noqa: WPS433 -- the service tree is the package
    import decide as decide_mod

    if not code_domains._MLP_OK:
        raise ImportError("world_model package unavailable: the door cannot run in-process")
    return decide_mod.Decider(code_domains.DomainEngines(), **kw)


def _ask_door(
    decider: Any, tool_name: str, unique_shapes: List[str], min_confidence: float
) -> Dict[str, Dict[str, Any]]:
    """One decide_batch per <=64 distinct shapes. Any transport failure means
    'unknown' for every shape in that chunk -- never a drop."""
    domain = _domain(tool_name)
    verdicts: Dict[str, Dict[str, Any]] = {}
    for start in range(0, len(unique_shapes), MAX_BATCH):
        chunk = unique_shapes[start : start + MAX_BATCH]
        items = [
            {
                "domain": domain,
                "state": s,
                "kind": "yesno",
                "question": _question(tool_name, s),
                "min_confidence": float(min_confidence),
            }
            for s in chunk
        ]
        try:
            answers = (decider.decide_batch(items) or {}).get("answers") or []
        except Exception as exc:  # noqa: BLE001 -- ignorance keeps, and says why
            for s in chunk:
                verdicts[s] = {"verdict": "unknown", "source": "error", "error": str(exc)[:160]}
            continue
        for s, ans in zip(chunk, answers):
            ans = ans or {}
            src = ans.get("source") or "none"
            if src == "none" or ans.get("answer") not in ("yes", "no"):
                verdict = "unknown"
            else:
                verdict = "keep" if ans.get("answer") == "yes" else "drop"
            verdicts[s] = {
                "verdict": verdict,
                "source": src,
                "answer": ans.get("answer"),
                "confidence": ans.get("confidence", 0.0),
                "p_yes": ans.get("p_yes"),
                "decision_id": ans.get("decision_id"),
                "learned_from": ans.get("learned_from", 0),
            }
        for s in chunk[len(answers) :]:  # a short answer list is ignorance too
            verdicts[s] = {"verdict": "unknown", "source": "short-batch"}
    return verdicts


# A door verdict of "unknown" with source "none" is the door ANSWERING: it saw
# this shape, had no evidence, and declined. That is the verdict `compact()`
# promises will keep the line.
#
# "error", "short-batch" and "no-door" are NOT answers -- the door was
# unreachable, the batch too small, or there was no door at all. There is no
# verdict to honour, and absent-door behaviour is defined as rules-only. Making
# those non-squeezable was measured 2026-09-20 to turn a door OUTAGE into ZERO
# compaction (1490 -> 1490 lines), which is how a compactor gets switched off.
_COULD_NOT_JUDGE = ("none",)


def _squeezable(r: Dict[str, Any]) -> bool:
    """A row the squeeze may collapse at all."""
    return not (r["always_keep"] or r["verdict"] != "keep")


def _unjudged(r: Dict[str, Any]) -> bool:
    """The door SAW this shape and declined (source "none").

    Not the same as an absent door ("error", "short-batch", "no-door"): there is
    no verdict to honour then, and absent-door behaviour is defined as
    rules-only. Treating those as unjudged was measured 2026-09-20 to turn a door
    outage into ZERO compaction.
    """
    return r.get("door_verdict") == "unknown" and r.get("door_source") in _COULD_NOT_JUDGE


def _same_run(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """May `b` continue the run `a` started?

    Same kind always. And when either end is a shape the door could not judge,
    the SHAPE must match too. The squeeze leaves a visible "(N similar lines
    dropped)" marker, so collapsing a block of one repeated unjudged shape loses
    nothing; what must never happen is a LONE unjudged shape being swallowed by a
    run of a different shape that merely shares its kind. That is precisely the
    single-letter-F blocker: one line, same kind as the 310 PASSED lines around
    it, different shape, gone into their marker.
    """
    if a["kind"] != b["kind"]:
        return False
    if _unjudged(a) or _unjudged(b):
        return a["shape"] == b["shape"]
    return True


def _squeeze(rows: List[Dict[str, Any]], collapse_runs: int) -> int:
    """Rule 3: runs of >= collapse_runs consecutive KEPT, non-always-keep lines
    of one kind keep first two + last one. Returns how many it dropped.

    A row the door could not judge is not squeezable and BREAKS the run, so it
    survives and does not silently join its neighbours' fate."""
    dropped = 0
    i, n = 0, len(rows)
    while i < n:
        r = rows[i]
        if not _squeezable(r):
            i += 1
            continue
        j = i
        while j < n and _squeezable(rows[j]) and _same_run(r, rows[j]):
            j += 1
        if j - i >= collapse_runs:
            for k in range(i + 2, j - 1):
                rows[k]["verdict"] = "drop"
                rows[k]["dropped_by"] = "collapse"
                dropped += 1
        i = j
    return dropped


def compact(
    text: str,
    tool_name: str = "tool",
    budget_lines: int = DEFAULT_BUDGET_LINES,
    *,
    decider: Any = None,
    min_confidence: float = 0.0,
    collapse_runs: int = DEFAULT_COLLAPSE_RUNS,
    min_lines: int = 0,
    always_keep_rules: bool = True,
) -> Dict[str, Any]:
    """Compact one tool output. `decider` is a decide.Decider, a RemoteDecider, or
    None (rules only). Returns kept text, counts, per-shape decisions (with the
    decision ids to teach) and where each verdict came from.
    `always_keep_rules=False` sends EVERY line's shape to the door -- a bench
    diagnostic that measures what the door alone would drop; never the default."""
    t0 = time.perf_counter()
    rows = shapes(text)
    if not always_keep_rules:
        for r in rows:
            r["always_keep"], r["reason"] = False, None
    total = len(rows)
    if total <= min_lines:
        return {
            "kept": text,
            "dropped": 0,
            "kept_lines": total,
            "total_lines": total,
            "decisions": [],
            "source_counts": {},
            "tool": tool_name,
            "domain": _domain(tool_name),
            "skipped": f"{total} lines <= min_lines {min_lines}",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
    for r in rows:
        r["verdict"] = "keep"
        r["dropped_by"] = None
        # default for always-keep rows, which are never sent to the door
        r["door_verdict"] = None
        r["door_source"] = None
    # 2. the door, one item per distinct shape among the door-eligible lines
    by_shape: Dict[str, List[int]] = {}
    for r in rows:
        if not r["always_keep"]:
            by_shape.setdefault(r["shape"], []).append(r["i"])
    verdicts: Dict[str, Dict[str, Any]] = {}
    if decider is not None and by_shape:
        verdicts = _ask_door(decider, tool_name, list(by_shape), min_confidence)
    door_dropped = 0
    for shape, idxs in by_shape.items():
        v = verdicts.get(shape) or {"verdict": "unknown", "source": "no-door"}
        for i in idxs:
            # record what the door said even when it said nothing -- the squeeze
            # needs to tell "approved keep" from "could not judge"
            rows[i]["door_verdict"] = v["verdict"]
            rows[i]["door_source"] = v.get("source")
        if v["verdict"] == "drop":
            for i in idxs:
                rows[i]["verdict"] = "drop"
                rows[i]["dropped_by"] = "door"
            door_dropped += len(idxs)
    # 3. the squeeze rule, tightened once if still over budget
    collapsed = _squeeze(rows, collapse_runs)
    kept_now = sum(1 for r in rows if r["verdict"] == "keep")
    if budget_lines and kept_now > budget_lines and collapse_runs > 2:
        collapsed += _squeeze(rows, 2)
    # render: dropped runs of one kind -> one marker
    out: List[str] = []
    i = 0
    while i < total:
        r = rows[i]
        if r["verdict"] == "keep":
            out.append(r["line"])
            i += 1
            continue
        j = i
        while j < total and rows[j]["verdict"] == "drop" and rows[j]["kind"] == r["kind"]:
            j += 1
        if r["kind"] != "blank":  # a dropped blank line needs no marker
            out.append(
                f"... ({j - i} similar lines dropped)" if j - i > 1 else "... (1 line dropped)"
            )
        i = j
    kept_lines = sum(1 for r in rows if r["verdict"] == "keep")
    source_counts: Dict[str, int] = {}
    decisions: List[Dict[str, Any]] = []
    for shape, idxs in by_shape.items():
        v = verdicts.get(shape) or {"verdict": "unknown", "source": "no-door"}
        source_counts[v["source"]] = source_counts.get(v["source"], 0) + 1
        decisions.append(
            {
                "shape": shape,
                "decision_id": v.get("decision_id"),
                "answer": v.get("answer"),
                "verdict": v["verdict"],
                "source": v["source"],
                "confidence": v.get("confidence", 0.0),
                "learned_from": v.get("learned_from", 0),
                "lines": idxs,
                "count": len(idxs),
                **({"error": v["error"]} if v.get("error") else {}),
            }
        )
    return {
        "kept": "\n".join(out),
        "dropped": total - kept_lines,
        "kept_lines": kept_lines,
        "total_lines": total,
        "door_dropped": door_dropped,
        "collapsed": collapsed,
        "always_kept": sum(1 for r in rows if r["always_keep"]),
        "distinct_shapes": len(by_shape),
        "decisions": decisions,
        "source_counts": source_counts,
        "tool": tool_name,
        "domain": _domain(tool_name),
        "over_budget": bool(budget_lines) and kept_lines > budget_lines,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


def compact_outcome(
    decisions: Sequence[Any],
    verdict_was_right: Any,
    decider: Any,
    *,
    tool_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Teach. `decisions` are decision ids (str) or the `decisions` rows compact()
    returned; `verdict_was_right` is one bool for all, or a list/dict per id.
    A wrong verdict posts the negative reward on what was answered AND +1 on the
    other answer, so the engine has a better KNOWN option to pick next time.
    Needs the door's write token in remote mode (AITHER_WM_INTERNAL_TOKEN)."""
    out: List[Dict[str, Any]] = []
    rows = [d if isinstance(d, dict) else {"decision_id": str(d)} for d in decisions]
    for idx, row in enumerate(rows):
        did = row.get("decision_id")
        if isinstance(verdict_was_right, dict):
            right = verdict_was_right.get(did, verdict_was_right.get(idx))
        elif isinstance(verdict_was_right, (list, tuple)):
            right = verdict_was_right[idx]
        else:
            right = verdict_was_right
        if right is None:
            continue
        right = bool(right)
        res: Dict[str, Any] = {"decision_id": did, "verdict_was_right": right}
        answer = row.get("answer")
        shape = row.get("shape")
        domain = _domain(tool_name) if tool_name else row.get("domain")
        try:
            if did:
                res["outcome"] = decider.outcome(
                    {"decision_id": str(did), "reward": 1.0 if right else -1.0}
                )
            elif shape and answer in ("yes", "no") and domain:
                res["outcome"] = decider.outcome(
                    {"domain": domain, "state": shape, "answer": answer,
                     "reward": 1.0 if right else -1.0}
                )
            if not right and shape and answer in ("yes", "no") and domain:
                other = "no" if answer == "yes" else "yes"
                res["counterfactual"] = decider.outcome(
                    {"domain": domain, "state": shape, "answer": other, "reward": 1.0}
                )
        except Exception as exc:  # noqa: BLE001 -- teaching that fails says so
            res["error"] = str(exc)[:200]
        out.append(res)
    return out


def teach_shape(
    decider: Any, tool_name: str, shape: str, keep: bool
) -> Dict[str, Any]:
    """Teach without a prior decision: lines of this shape from this tool should
    be kept (True) or dropped (False). The engine's only evidence-based way to
    learn a shape it has never been asked about."""
    return decider.outcome(
        {"domain": _domain(tool_name), "state": shape, "answer": "yes" if keep else "no",
         "reward": 1.0}
    )


def grade(result: Dict[str, Any], must_keep: Iterable[int]) -> Dict[str, bool]:
    """{decision_id_or_shape: verdict_was_right} for a hand-labelled output: a
    shape's verdict is right when it kept a shape holding a must-keep line, or
    dropped a shape holding none. An `unknown` (kept) verdict on filler is graded
    wrong, so the door learns to drop what a rule left standing."""
    mk = set(int(i) for i in must_keep)
    grades: Dict[str, bool] = {}
    for d in result.get("decisions") or []:
        holds_must_keep = any(i in mk for i in d["lines"])
        if d["verdict"] == "drop":
            right = not holds_must_keep
        else:  # keep or unknown
            right = holds_must_keep
        grades[d.get("decision_id") or d["shape"]] = right
    return grades


def recall(result: Dict[str, Any], must_keep: Iterable[int], text: str) -> Dict[str, Any]:
    """Which must-keep lines survived. Compares by exact line text, so a line
    that was kept is found even after markers were inserted."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept_text = result.get("kept", "")
    kept_set = set(kept_text.split("\n"))
    missing = []
    for i in must_keep:
        i = int(i)
        if 0 <= i < len(lines) and lines[i] not in kept_set:
            missing.append({"i": i, "line": lines[i][:160]})
    wanted = len(list(must_keep))
    return {
        "must_keep": wanted,
        "found": wanted - len(missing),
        "missing": missing,
        "recall": (1.0 if wanted == 0 else round((wanted - len(missing)) / wanted, 4)),
    }


def estimate_tokens(text: str) -> int:
    """~4 chars per token. An ESTIMATE, not a tokenizer count; the bench says so."""
    return (len(text) + 3) // 4


# ------------------------------------------------------------------ self-test
class _ScriptedDoor:
    """Answers a fixed yes/no for every shape and records outcomes."""

    def __init__(self, answer: Optional[str] = "no", source: str = "engine") -> None:
        self.answer, self.source, self.outcomes, self.batches = answer, source, [], 0

    def decide_batch(self, items):
        self.batches += 1
        if len(items) > MAX_BATCH:
            raise ValueError("at most 64 items per batch")
        return {
            "answers": [
                {
                    "answer": self.answer or "yes",
                    "source": self.source if self.answer else "none",
                    "confidence": 0.9 if self.answer else 0.0,
                    "decision_id": f"d{i}",
                }
                for i, _ in enumerate(items)
            ]
        }

    def outcome(self, body):
        self.outcomes.append(body)
        return {"ok": True, **body}


def _pytest_like(n_pass: int = 400, fail: bool = True) -> str:
    lines = ["============================= test session starts =============================",
             "platform win32 -- Python 3.12.6, pytest-8.3.2", "collected 401 items", ""]
    for i in range(n_pass):
        lines.append(f"dev/tests/test_mod_{i // 20}.py::test_case_{i} PASSED"
                     f"            [{(i * 100) // 401:3d}%]")
    if fail:
        lines += [
            "dev/tests/test_last.py::test_boom FAILED                             [100%]",
            "", "=================================== FAILURES ===================================",
            "__________________________________ test_boom ___________________________________",
            "", "    def test_boom():", ">       assert compute() == 3",
            "E       assert 2 == 3", "", "dev/tests/test_last.py:12: AssertionError",
            "=========================== short test summary info ============================",
            "FAILED dev/tests/test_last.py::test_boom - assert 2 == 3",
            f"========================= 1 failed, {n_pass} passed in 3.21s "
            "=========================",
        ]
    else:
        lines.append(f"============================== {n_pass} passed in 2.10s "
                     "==============================")
    return "\n".join(lines) + "\n"


def _self_test() -> int:
    fails: List[str] = []

    def check(name: str, ok: bool) -> None:
        print(("ok   " if ok else "FAIL ") + name)
        if not ok:
            fails.append(name)

    text = _pytest_like()
    rows = shapes(text)
    must = [i for i, r in enumerate(rows) if "FAILED" in r["line"] or "1 failed" in r["line"]
            or r["line"].startswith("E ") or "AssertionError" in r["line"]]
    check("must-keep lines exist in the fixture", len(must) >= 4)
    check("all must-keep lines are always_keep by rule",
          all(rows[i]["always_keep"] for i in must))
    check("PASSED lines are door-eligible, not always-keep",
          any(r["kind"] == "pytest-pass" and not r["always_keep"] for r in rows))

    # rules only: no door, squeeze collapses the PASSED run, recall 100%
    r0 = compact(text, "pytest")
    rc = recall(r0, must, text)
    check("rules-only keeps every must-keep line", rc["recall"] == 1.0)
    check("rules-only still compacts (squeeze)", r0["dropped"] > 300)
    check("rules-only marks every door shape unknown/no-door",
          all(d["source"] == "no-door" for d in r0["decisions"]))
    check("marker text is the documented one", "similar lines dropped)" in r0["kept"])
    check("kept text ends with the summary line",
          "1 failed, 400 passed" in r0["kept"].splitlines()[-1])

    # a door that says DROP everything: must-keep lines still survive
    door = _ScriptedDoor("no")
    r1 = compact(text, "pytest", decider=door)
    check("door=drop-all: ONE batch for the whole output", door.batches == 1)
    check("door=drop-all: recall still 100%", recall(r1, must, text)["recall"] == 1.0)
    check("door=drop-all: drops more than rules alone", r1["dropped"] > r0["dropped"])
    check("door=drop-all: decisions carry ids and line indexes",
          all(d["decision_id"] and d["lines"] for d in r1["decisions"]))

    # a door with nothing to go on (source none) keeps
    none_door = _ScriptedDoor(None)
    r2 = compact(text, "pytest", decider=none_door)
    check("source=none keeps (door_dropped == 0)", r2["door_dropped"] == 0)
    check("source=none is reported as unknown",
          all(d["verdict"] == "unknown" for d in r2["decisions"]))

    # a door that raises keeps, and says why
    class Boom:
        def decide_batch(self, items):
            raise ConnectionError("door down")

    r3 = compact(text, "pytest", decider=Boom())
    # NOT equality. Equality encoded the bug: a could-not-judge row used to be
    # indistinguishable from an approved keep, so the squeeze compressed it and
    # the two paths matched. Now a down door keeps strictly MORE (83 vs 27 on
    # this corpus item), which is the safe direction. The invariant that was
    # always meant is the superset one.
    _rules_kept = set(r0["kept"].splitlines())
    _err_kept = set(r3["kept"].splitlines())
    _lost = {ln for ln in _rules_kept - _err_kept if "similar lines dropped" not in ln}
    check("door error loses NO line the rules kept", r3["door_dropped"] == 0 and not _lost)
    check("door error still compacts -- it does not pass the whole output through",
          r3["kept_lines"] < r3["total_lines"])
    check("door error is reported on the decisions", all(d.get("error") for d in r3["decisions"]))

    # teaching: a wrong 'keep' posts the negative AND the counterfactual
    keep_door = _ScriptedDoor("yes")
    r4 = compact(text, "pytest", decider=keep_door)
    g = grade(r4, must)
    check("grade marks kept filler shapes wrong", any(v is False for v in g.values()))
    res = compact_outcome(r4["decisions"], g, keep_door, tool_name="pytest")
    wrong = [x for x in res if not x["verdict_was_right"]]
    check("wrong verdicts post outcome + counterfactual",
          wrong and all("counterfactual" in x and "outcome" in x for x in wrong))
    check("counterfactual teaches the OTHER answer",
          all(x["counterfactual"]["answer"] == "no" for x in wrong))

    # >64 distinct shapes are chunked, never refused (the descriptor space is
    # kind x len x rep x pos, so a real output rarely gets there; the door's
    # 64-item cap must still never turn into a refusal)
    chunk_door = _ScriptedDoor("no")
    fake_shapes = [f"kind:other|len:<40|rep:{i}|pos:middle" for i in range(130)]
    v = _ask_door(chunk_door, "weird", fake_shapes, 0.0)
    check("more than 64 shapes -> several batches, none refused",
          chunk_door.batches == 3 and len(v) == 130
          and all(x["verdict"] == "drop" for x in v.values()))

    # secret safety: the question never carries the line text
    q = _question("pytest", "kind:log|len:<80|rep:0|pos:middle")
    check("question is shape-only", "kind:log" in q and "sk-" not in q)

    # shapes are stable across two runs of the same kind of output
    a = {r["shape"] for r in shapes(_pytest_like(400))}
    b = {r["shape"] for r in shapes(_pytest_like(397))}
    check("shape set is stable across two runs of the same kind", a == b)

    # short outputs pass through untouched with min_lines
    r6 = compact("one\ntwo\n", "x", min_lines=60)
    check("min_lines short-circuits", r6.get("skipped") and r6["kept"] == "one\ntwo\n")

    # ANSI never breaks classification
    check("ANSI-coloured PASSED still classifies as pytest-pass",
          line_kind("\x1b[32mdev/t.py::test_a PASSED\x1b[0m") == "pytest-pass")
    check("curl failure line is always-keep",
          line_kind("curl: (7) Failed to connect to 127.0.0.1 port 9 after 1 ms")
          in ALWAYS_KEEP_KINDS)
    check("git diffstat row is door-eligible",
          line_kind(" lib/core/x.py | 12 +++---") == "diffstat")
    check("exit code line is a summary", line_kind("exit code: 1") == "summary")


    # The adversarial pass, 2026-09-20: a single-letter F/E verdict shares its
    # skeleton with the PASSED lines around it, so the only line naming the failing
    # test was dropped with them. It is classed as a summary now, which also puts it
    # in ALWAYS_KEEP_KINDS so a confidently-wrong door cannot drop it either.
    f_line = "tests/test_zeta_router.py::test_capability_denies_foreign_tenant F  [ 48%]"
    check("single-letter F verdict is a summary", line_kind(f_line) == "summary")
    check("dotted F progress line is a summary", line_kind("tests/test_b.py ....F...   [ 48%]") == "summary")
    check("a PASSED line is not a summary", line_kind("tests/test_a.py::test_ok PASSED  [ 47%]") != "summary")
    check("an all-dots line is not a summary", line_kind("tests/test_a.py ........   [ 47%]") != "summary")


    # The critic's finding, 2026-09-20: `compact()` promises an unknown verdict
    # KEEPS the line, and the squeeze was dropping it anyway because the verdict
    # loop only ever wrote "drop" -- a could-not-judge row sat at "keep",
    # indistinguishable from an approved one. Both directions are checked, so the
    # fix cannot be discharged by simply never squeezing anything.
    def _row(kind, verdict="keep", dv=None, ds=None, ak=False, shape="S"):
        return {"i": 0, "kind": kind, "shape": shape, "always_keep": ak,
                "verdict": verdict, "dropped_by": None,
                "door_verdict": dv, "door_source": ds}

    run_ok = [_row("plain", dv="keep", ds="engine") for _ in range(8)]
    check("a run the door APPROVED still collapses", _squeeze(run_ok, 3) == 5)

    # a cold door answers `none` for everything; one REPEATED shape is filler and
    # must still collapse, or the first run of the compactor saves nothing
    run_unk = [_row("plain", dv="unknown", ds="none") for _ in range(8)]
    check("a repeated shape the door could not judge still collapses",
          _squeeze(run_unk, 3) == 5)

    run_err = [_row("plain", dv="unknown", ds="error") for _ in range(8)]
    check("an UNREACHABLE door degrades to rules-only, it does not stop the squeeze",
          _squeeze(run_err, 3) == 5)

    run_nodoor = [_row("plain", dv="unknown", ds="no-door") for _ in range(8)]
    check("rules-only (no-door) still collapses -- squeeze is its only compressor",
          _squeeze(run_nodoor, 3) == 5)

    # THE ONE THAT MATTERS: a lone unjudged SHAPE inside a run of another shape
    # that shares its kind -- the single-letter-F blocker, in miniature.
    lone = ([_row("plain", dv="unknown", ds="none", shape="PASSED") for _ in range(5)]
            + [_row("plain", dv="unknown", ds="none", shape="FAILVERDICT")]
            + [_row("plain", dv="unknown", ds="none", shape="PASSED") for _ in range(5)])
    _squeeze(lone, 3)
    check("a LONE unjudged shape breaks the run and survives inside one",
          lone[5]["verdict"] == "keep")
    check("and its same-shape neighbours are still collapsed around it",
          sum(1 for r in lone if r["verdict"] == "drop") > 0)

    # and the same line when the door APPROVED its neighbours
    lone2 = ([_row("plain", dv="keep", ds="engine", shape="PASSED") for _ in range(5)]
             + [_row("plain", dv="unknown", ds="none", shape="FAILVERDICT")]
             + [_row("plain", dv="keep", ds="engine", shape="PASSED") for _ in range(5)])
    _squeeze(lone2, 3)
    check("a lone unjudged shape survives a run of APPROVED neighbours too",
          lone2[5]["verdict"] == "keep")

    print(f"\n{'PASS' if not fails else 'FAIL'}: {len(fails)} failing check(s)")
    return 1 if fails else 0


def _main(argv: List[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("file", nargs="?", help="tool output file, or - for stdin")
    ap.add_argument("--tool", default="tool")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET_LINES)
    ap.add_argument("--url", help="remote door (…:8197 or gateway …/v1); default in-process")
    ap.add_argument("--rules-only", action="store_true", help="no door at all")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return _self_test()
    if not a.file:
        ap.print_usage()
        return 2
    text = sys.stdin.read() if a.file == "-" else Path(a.file).read_text(
        encoding="utf-8", errors="replace"
    )
    decider = None
    mode = "rules-only"
    if not a.rules_only:
        if a.url:
            decider, mode = RemoteDecider(a.url), f"remote {a.url}"
        else:
            try:
                decider, mode = in_process_decider(), "in-process"
            except Exception as exc:  # noqa: BLE001
                print(f"[compact] in-process door unavailable ({exc}); rules only", file=sys.stderr)
    res = compact(text, a.tool, a.budget, decider=decider)
    res["mode"] = mode
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print(res["kept"])
        print(
            f"\n[compact] kept {res['kept_lines']} of {res['total_lines']} lines "
            f"({mode}; sources {res['source_counts']}; {res['latency_ms']} ms)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
