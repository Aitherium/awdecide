#!/usr/bin/env python
"""cache_cost_bench -- what compaction costs in PROMPT-CACHE dollars, by WHERE it happens.

The critique (2026-09-21, of history-editing compaction hooks) is right about one
thing and it is arithmetic, not opinion: prompt caching is PREFIX caching. Delete
message 2 from "1,2,3,4,5,6" and 3..6 are re-written at the cache-WRITE price;
leave 2 alone and 3..6 are re-read at the cache-READ price. So a compactor that
edits history pays for every token AFTER the edit, every time it edits, and it
also drops the reasoning blocks that the API only replays against an unchanged
history. A compactor that shrinks a tool result BEFORE it is appended edits
nothing above it: the prefix stays cached, the reasoning stays, and the saving is
the tokens that never entered the history at all.

This bench simulates one agent session with REAL tool-result sizes (the
compact_corpus: pytest, ruff, git log, podman build, curl -- cycled) and prices
each turn's request the way a prefix cache does:

    cached  = longest common prefix (by segment identity) with the previous request
    read    = tokens in that prefix, at the cache-read price
    write   = every token after it, at the cache-write price

Legs:
  none          every tool result appended raw; nothing ever edited
  history-edit  raw results appended; every K turns the oldest un-edited tool
                results are cut IN PLACE to `--edit-keep` of their size (the hook
                the critique describes). The cut point breaks the prefix.
  append-rules  each result compacted at append time by the always-keep rules only
                (no door): measured per item with compact.compact(decider=None)
  append-door   each result compacted at append time by the TAUGHT door (one
                teaching pass on the corpus, same as compact_bench's door-taught leg,
                on a fresh journal so the fleet's door is never touched)

Assistant text between tool results is a constant `--assistant-tokens` per turn;
reasoning blocks are NOT priced (the API does not bill replayed thinking) but the
history-edit leg reports how many turns of them it would have dropped, because
that is the capability cost the dollars do not show.

Prices: anthropic list, USD per 1M tokens, from --prices (the platform's
model_token_prices.yaml, `input_usd_per_1m`) with the published cache multipliers
cache_write = 1.25 x input, cache_read = 0.10 x input (platform.claude.com pricing,
read 2026-09-21; override with --cache-write-mult / --cache-read-mult). Token
counts are the 4-chars-per-token ESTIMATE compact.py uses -- said in every table.

Exit 0 when append-time compaction (either leg) is cheaper than history-edit over
the session AND the history-edit leg's rewrite tokens exceed its saved tokens at
some point in the session (i.e. the critique's failure mode was reproduced);
1 when the model of the argument does not hold on these numbers; 2 could not run.
--json for the table; --self-test proves the prefix-cache accounting can fail.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
SVC = HERE.parent
sys.path.insert(0, str(SVC))
sys.path.insert(0, str(HERE))

import compact  # noqa: E402

DEFAULT_PRICES = Path(
    os.environ.get("AITHER_REPO", r"C:\source")
) / "AitherOS" / "config" / "model_token_prices.yaml"


# ------------------------------------------------------------- the cache model
@dataclass
class Segment:
    sid: int
    tokens: int
    kind: str  # "system" | "assistant" | "tool" | "thinking"


@dataclass
class Ledger:
    read_tokens: int = 0
    write_tokens: int = 0
    turns: int = 0
    rewrite_events: int = 0
    rewritten_tokens: int = 0  # tokens re-written ONLY because history was edited
    thinking_dropped: int = 0
    per_turn: List[Dict[str, int]] = field(default_factory=list)


def price_request(prev: List[Segment], cur: List[Segment]) -> Tuple[int, int, int]:
    """(read_tokens, write_tokens, prefix_len) for `cur` given the previous request."""
    n = 0
    for a, b in zip(prev, cur):
        if a.sid != b.sid or a.tokens != b.tokens:
            break
        n += 1
    read = sum(s.tokens for s in cur[:n])
    write = sum(s.tokens for s in cur[n:])
    return read, write, n


def simulate(
    results: List[int],
    *,
    assistant_tokens: int,
    thinking_tokens: int,
    system_tokens: int,
    edit_every: int = 0,
    edit_keep: float = 0.4,
    edit_age: int = 3,
) -> Ledger:
    """One session. `results[i]` is the token size of turn i's tool result AS APPENDED
    (already compacted for the append-time legs). `edit_every` > 0 turns on the
    history-edit leg: every K turns, every tool result older than `edit_age` turns
    that has not been cut yet is cut to `edit_keep` of its size, in place."""
    led = Ledger()
    hist: List[Segment] = [Segment(0, system_tokens, "system")]
    prev: List[Segment] = []
    sid = 1
    cut: set = set()
    tool_turn: Dict[int, int] = {}  # sid -> turn appended
    for t, rt in enumerate(results):
        # the assistant's turn: thinking + text + the tool call, then the result
        hist.append(Segment(sid, thinking_tokens, "thinking")); sid += 1
        hist.append(Segment(sid, assistant_tokens, "assistant")); sid += 1
        hist.append(Segment(sid, rt, "tool")); tool_turn[sid] = t; sid += 1
        if edit_every and t > 0 and t % edit_every == 0:
            # the hook the critique describes: shrink old tool results in place
            first_edit: Optional[int] = None
            for i, s in enumerate(hist):
                if s.kind == "tool" and s.sid not in cut and t - tool_turn[s.sid] >= edit_age:
                    s.tokens = max(1, int(s.tokens * edit_keep))
                    cut.add(s.sid)
                    if first_edit is None:
                        first_edit = i
            if first_edit is not None:
                led.rewrite_events += 1
                # everything after the edit point is re-written; the thinking blocks
                # after it can no longer be replayed against an unchanged history
                led.rewritten_tokens += sum(s.tokens for s in hist[first_edit:])
                led.thinking_dropped += sum(1 for s in hist[first_edit:] if s.kind == "thinking")
        read, write, _ = price_request(prev, hist)
        led.read_tokens += read
        led.write_tokens += write
        led.turns += 1
        led.per_turn.append({"turn": t, "read": read, "write": write,
                             "context": sum(s.tokens for s in hist)})
        prev = [Segment(s.sid, s.tokens, s.kind) for s in hist]
    return led


# ------------------------------------------------------------------ prices
def load_input_price(path: Path, model: str) -> Optional[float]:
    """input_usd_per_1m for `model`, from the platform price file. None = unpriced."""
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        row = (data.get("models") or {}).get(model) or {}
        v = row.get("input_usd_per_1m")
        return float(v) if v is not None else None
    except Exception:  # noqa: BLE001 -- unpriced is reported, never zeroed
        return None


def dollars(led: Ledger, input_per_1m: float, wmult: float, rmult: float) -> float:
    return (led.read_tokens * input_per_1m * rmult + led.write_tokens * input_per_1m * wmult) / 1e6


# ------------------------------------------------------------------ corpus
def corpus_sizes(decider: Any) -> Dict[str, List[int]]:
    """Per-item token sizes: raw, rules-only compacted, door-taught compacted."""
    from compact_bench import load_corpus, run_legs  # type: ignore

    items = load_corpus(HERE / "compact_corpus")
    table = run_legs(items, decider, diagnostics=False)
    raw, rules, door = [], [], []
    for it in items:
        raw.append(compact.estimate_tokens(it["text"]))
        rules.append(table["rules-only"][it["name"]]["tokens_after_est"])
        door.append(table["door-taught"][it["name"]]["tokens_after_est"])
    return {"raw": raw, "rules": rules, "door": door,
            "names": [it["name"] for it in items]}  # type: ignore[dict-item]


def cycle(sizes: List[int], turns: int) -> List[int]:
    return [sizes[i % len(sizes)] for i in range(turns)]


# ------------------------------------------------------------------ main
def run(a: argparse.Namespace) -> Tuple[Dict[str, Any], int]:
    tmp = tempfile.mkdtemp(prefix="cache-cost-ckpt-")
    os.environ["AITHER_WM_CKPT_DIR"] = tmp
    decider = compact.in_process_decider(llm_enabled=False, embed_enabled=False)
    sizes = corpus_sizes(decider)
    common = dict(assistant_tokens=a.assistant_tokens, thinking_tokens=a.thinking_tokens,
                  system_tokens=a.system_tokens)
    legs: Dict[str, Ledger] = {
        "none": simulate(cycle(sizes["raw"], a.turns), **common),
        "history-edit": simulate(cycle(sizes["raw"], a.turns), edit_every=a.edit_every,
                                 edit_keep=a.edit_keep, **common),
        "append-rules": simulate(cycle(sizes["rules"], a.turns), **common),
        "append-door": simulate(cycle(sizes["door"], a.turns), **common),
    }
    price = load_input_price(Path(a.prices), a.model)
    rows: Dict[str, Any] = {}
    for name, led in legs.items():
        rows[name] = {
            "read_tokens": led.read_tokens,
            "write_tokens": led.write_tokens,
            "final_context_tokens": led.per_turn[-1]["context"],
            "rewrite_events": led.rewrite_events,
            "rewritten_tokens_due_to_edits": led.rewritten_tokens,
            "thinking_blocks_invalidated": led.thinking_dropped,
            "usd": (round(dollars(led, price, a.cache_write_mult, a.cache_read_mult), 4)
                    if price is not None else None),
        }
    none, he = rows["none"], rows["history-edit"]
    saved_by_edits = none["write_tokens"] + none["read_tokens"] - (he["write_tokens"] + he["read_tokens"])
    report = {
        "turns": a.turns,
        "corpus_items": sizes["names"],
        "tokens_per_result_est": {"raw": sizes["raw"], "rules": sizes["rules"], "door": sizes["door"]},
        "token_estimate": f"{compact.TOKEN_CHARS if hasattr(compact, 'TOKEN_CHARS') else 4} chars/token",
        "model": a.model,
        "input_usd_per_1m": price,
        "cache_write_mult": a.cache_write_mult,
        "cache_read_mult": a.cache_read_mult,
        "edit_every": a.edit_every,
        "edit_keep": a.edit_keep,
        "legs": rows,
        "history_edit_net_tokens_saved_vs_none": saved_by_edits,
        "ckpt_dir": tmp,
    }
    ok_cheaper = all(
        rows[k]["write_tokens"] * a.cache_write_mult + rows[k]["read_tokens"] * a.cache_read_mult
        < he["write_tokens"] * a.cache_write_mult + he["read_tokens"] * a.cache_read_mult
        for k in ("append-rules", "append-door")
    )
    reproduced = he["rewritten_tokens_due_to_edits"] > 0 and he["write_tokens"] > none["write_tokens"]
    report["verdict"] = {
        "append_time_cheaper_than_history_edit": ok_cheaper,
        "history_edit_wrote_more_than_no_compaction": reproduced,
    }
    return report, (0 if (ok_cheaper and reproduced) else 1)


def render(r: Dict[str, Any]) -> str:
    out = [f"session of {r['turns']} tool turns, corpus {', '.join(r['corpus_items'])}; "
           f"tokens are a {r['token_estimate']} estimate",
           f"model {r['model']}: input ${r['input_usd_per_1m']}/1M, cache write x{r['cache_write_mult']}, "
           f"cache read x{r['cache_read_mult']}"
           if r["input_usd_per_1m"] is not None else f"model {r['model']}: UNPRICED (no $ column)",
           f"history-edit: every {r['edit_every']} turns, old results cut to {r['edit_keep']:.0%}",
           "",
           f"{'leg':14}{'cache read':>12}{'cache write':>13}{'final ctx':>11}{'rewrites':>10}"
           f"{'rewritten':>11}{'thinking x':>11}{'USD':>10}"]
    for name, row in r["legs"].items():
        usd = f"${row['usd']:.4f}" if row["usd"] is not None else "-"
        out.append(f"{name:14}{row['read_tokens']:>12,}{row['write_tokens']:>13,}"
                   f"{row['final_context_tokens']:>11,}{row['rewrite_events']:>10}"
                   f"{row['rewritten_tokens_due_to_edits']:>11,}{row['thinking_blocks_invalidated']:>11}"
                   f"{usd:>10}")
    v = r["verdict"]
    out.append("")
    out.append(f"history-edit net tokens vs none: {r['history_edit_net_tokens_saved_vs_none']:+,} "
               f"(negative = it cost MORE than doing nothing)")
    out.append("VERDICT: " + ("append-time compaction is cheaper than history editing, and history "
                              "editing wrote more cache than no compaction at all"
                              if v["append_time_cheaper_than_history_edit"]
                              and v["history_edit_wrote_more_than_no_compaction"]
                              else f"the argument does not hold on these numbers: {v}"))
    return "\n".join(out)


def _self_test() -> int:
    # 1. an unchanged history is all read, no write
    a = [Segment(0, 100, "system"), Segment(1, 50, "tool")]
    r, w, n = price_request(a, [Segment(0, 100, "system"), Segment(1, 50, "tool")])
    if (r, w, n) != (150, 0, 2):
        print(f"SELF-TEST FAILED: unchanged history {r, w, n}"); return 1
    # 2. appending writes only the suffix
    r, w, n = price_request(a, a + [Segment(2, 30, "tool")])
    if (r, w) != (150, 30):
        print(f"SELF-TEST FAILED: append {r, w}"); return 1
    # 3. editing segment 1 in place breaks the prefix there: everything after is written
    r, w, n = price_request(a + [Segment(2, 30, "tool")],
                            [Segment(0, 100, "system"), Segment(1, 20, "tool"), Segment(2, 30, "tool")])
    if (r, w, n) != (100, 50, 1):
        print(f"SELF-TEST FAILED: edit {r, w, n}"); return 1
    # 4. the simulator: history-edit must record a rewrite; append-time must record none
    he = simulate([1000] * 12, assistant_tokens=10, thinking_tokens=10, system_tokens=10,
                  edit_every=4, edit_keep=0.4)
    ap = simulate([400] * 12, assistant_tokens=10, thinking_tokens=10, system_tokens=10)
    if he.rewrite_events == 0 or he.thinking_dropped == 0 or ap.rewrite_events != 0:
        print(f"SELF-TEST FAILED: simulate {he.rewrite_events} {he.thinking_dropped} {ap.rewrite_events}")
        return 1
    if not (ap.write_tokens < he.write_tokens):
        print("SELF-TEST FAILED: append-time did not write less than history-edit"); return 1
    # 5. a mutant that never breaks the prefix would make edits free -- assert it is not
    if he.rewritten_tokens <= 0:
        print("SELF-TEST FAILED: edits were free"); return 1
    print("SELF-TEST PASSED: prefix accounting reads, appends and breaks where it should")
    return 0


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--assistant-tokens", type=int, default=150)
    ap.add_argument("--thinking-tokens", type=int, default=400,
                    help="per-turn reasoning block size; NOT priced, counted when invalidated")
    ap.add_argument("--system-tokens", type=int, default=12000,
                    help="system prompt + tool schemas (Claude Code is ~12k+)")
    ap.add_argument("--edit-every", type=int, default=5)
    ap.add_argument("--edit-keep", type=float, default=0.4)
    ap.add_argument("--prices", default=str(DEFAULT_PRICES))
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--cache-write-mult", type=float, default=1.25)
    ap.add_argument("--cache-read-mult", type=float, default=0.10)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return _self_test()
    try:
        report, rc = run(a)
    except Exception as exc:  # noqa: BLE001
        print(f"[cache_cost_bench] cannot run: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2) if a.json else render(report))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
