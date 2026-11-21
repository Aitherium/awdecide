"""DecisionRecord -- one join key from a decision to its context, rung and outcome.

The ledger (`ledger.py`) keeps a decision as a CLAIM: key, value, probability,
backend, `state_sha`, and later its outcome. That is enough to score a Brier and
nothing else. Measured 2026-11-13 on the live ledger (672,650 rows, 669,442
resolved): not one row can be joined back to the session, turn, tool call or
request that produced it, nor says which rung answered, which model, whether it
thought, how long it took or what the state text was. So none of those outcomes
can become a training row, and nobody can tell whether escalating paid off.

This module adds the missing half WITHOUT a new store and WITHOUT touching
`ledger.py`: a side table `decision_ctx` in the SAME sqlite file, keyed by the
decision id, written by whoever made the decision (`attach`), and a read side
that joins the two:

    attach(db, decision_id, **ctx)        upsert the context of one decision
    export_records(db, out, since=None)   DecisionRecord JSONL (one per decision)
    stats(db)                             Brier / climatology / BSS / ECE per key
                                          family and per rung -- honest: bench and
                                          demo keys excluded, stale pending expired

    python -m awdecide.record stats   [--db PATH] [--json] [--expire-days 7]
    python -m awdecide.record export  --since <epoch|ISO> [--out PATH|-]

Why the stats are per key family and per rung: 99.9% of the live rows are the
trivial `tool.bash` key, where a 97.4% base rate makes any recorder look good.
The headline Brier (0.0255) against that key's climatology (0.0258) is a skill
score of ~0.01, not "calibrated information". A Brier skill score
BSS = 1 - Brier / climatology is the number that says whether the probability
carries anything beyond the base rate; it is reported for every group.

Stdlib only and no relative imports on purpose: the Bash post-hook loads this
file BY PATH from the repo tree when the served awdecide snapshot predates it.
`decisions` is only ever READ here; the one table this module writes is its own.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

SCHEMA_ID = "awdecide.decision_record/1"
DEFAULT_DB = Path(os.getenv("AWDECIDE_DB", str(Path.home() / ".aither" / "awdecide.db")))
#: pending rows older than this are not "still open" claims, they are lost ones
EXPIRE_DAYS = 7.0
#: state text kept per decision; the Bash hook's commands are cut at the same size
STATE_TEXT_MAX = 2048
#: the ledger's reliability buckets (ledger.BUCKETS), duplicated to stay stdlib-only
BUCKETS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]

CTX_FIELDS = (
    "ts", "caller", "session_id", "turn_id", "tool_use_id", "request_id", "parent_id",
    "rung", "model", "think", "latency_ms", "tokens", "state_text", "options",
    "features", "embed_ref", "mcts_visits", "mcts_value",
)
#: API name -> column, for the three that are stored as JSON
_JSON_COLS = {"options": "options_json", "features": "features_json",
              "mcts_visits": "mcts_visits_json"}
_COLS = ("decision_id",) + tuple(_JSON_COLS.get(f, f) for f in CTX_FIELDS)

_CTX_SQL = (
    "CREATE TABLE IF NOT EXISTS decision_ctx ("
    " decision_id TEXT PRIMARY KEY, ts REAL, caller TEXT, session_id TEXT, turn_id TEXT,"
    " tool_use_id TEXT, request_id TEXT, parent_id TEXT, rung TEXT, model TEXT,"
    " think INTEGER, latency_ms REAL, tokens INTEGER, state_text TEXT, options_json TEXT,"
    " features_json TEXT, embed_ref TEXT, mcts_visits_json TEXT, mcts_value REAL)",
    "CREATE INDEX IF NOT EXISTS idx_decision_ctx_session ON decision_ctx(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_decision_ctx_parent ON decision_ctx(parent_id)",
)

# ------------------------------------------------------------------ scrubbing
# A second net over state text, AFTER whatever redaction the caller already did
# (the Bash hook runs its own credential-shape `redact` first). It covers the
# shapes that are not a vendor prefix: auth headers, key=value secrets, URL
# credentials and long opaque blobs.
_SCRUB = [
    (re.compile(r"(?i)\b(authorization|x-[a-z-]*token|x-api-key)\s*:\s*[^\s'\"]+(\s+[^\s'\"]+)?"),
     r"\1: <secret>"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer <secret>"),
    (re.compile(r"(?i)\b([A-Z0-9_]*(?:password|passwd|secret|token|api[_-]?key|apikey)"
                r"[A-Z0-9_]*)\s*([=:])\s*(\"[^\"]*\"|'[^']*'|[^\s&;|]+)"), r"\1\2<secret>"),
    (re.compile(r"(?i)(--(?:password|token|api-key|secret)[= ])\S+"), r"\1<secret>"),
    (re.compile(r"://[^/\s:@]+:[^/\s@]+@"), "://<secret>@"),
    (re.compile(r"(\s(?:-u|--user)[ =])[^\s:]+:\S+"), r"\1<secret>"),
    (re.compile(r"\b[A-Za-z0-9+/_-]{40,}={0,2}"), "<blob>"),
]


def scrub(text: Any, limit: int = STATE_TEXT_MAX) -> str:
    """Secret-shaped spans replaced, then cut to `limit` characters."""
    out = str(text or "")
    for rx, rep in _SCRUB:
        out = rx.sub(rep, out)
    return out[:limit]


# ------------------------------------------------------------------ connection
DbLike = Union[str, Path, sqlite3.Connection, Any]


def _connect(db: DbLike, readonly: bool = False) -> Tuple[sqlite3.Connection, bool]:
    """(connection, owned). A Ledger (anything with `_con`) or a connection is
    borrowed; a path is opened, read-only when asked so a stats run never takes a
    write lock on the file the Bash hook writes on every command."""
    if isinstance(db, sqlite3.Connection):
        return db, False
    con = getattr(db, "_con", None)
    if isinstance(con, sqlite3.Connection):
        return con, False
    path = Path(db) if db else DEFAULT_DB
    if readonly:
        con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path, timeout=3)
    return con, True


def ensure(con: sqlite3.Connection) -> None:
    """Create `decision_ctx` if missing. Individual statements, not executescript:
    executescript COMMITs first, which would split the caller's transaction."""
    for sql in _CTX_SQL:
        con.execute(sql)


def _has_ctx(con: sqlite3.Connection) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND "
                       "name='decision_ctx'").fetchone() is not None


# ------------------------------------------------------------------ write
def _dumps(v: Any) -> Optional[str]:
    if v is None:
        return None
    return json.dumps(v, sort_keys=True, default=str, separators=(",", ":"))


def attach(db: DbLike, decision_id: str, *, commit: bool = True, **ctx: Any) -> bool:
    """Upsert the context of one decision. Returns False for an empty id.

    Known fields (CTX_FIELDS) land in their columns; any other keyword is folded
    into `features`, so caller metadata is never dropped on the floor. A second
    attach for the same id MERGES: a None never overwrites a value already there,
    so an MCTS pass or an outcome hook can add its fields later.

    `commit=False` leaves the write in the caller's open transaction -- the Bash
    hook lands it inside the same commit as the outcome, paying no extra fsync.
    """
    did = str(decision_id or "")
    if not did:
        return False
    extra = {k: ctx.pop(k) for k in list(ctx) if k not in CTX_FIELDS}
    if extra:
        feats = dict(ctx.get("features") or {})
        for k, v in extra.items():
            feats.setdefault(k, v)
        ctx["features"] = feats
    row: Dict[str, Any] = {"decision_id": did}
    for f in CTX_FIELDS:
        v = ctx.get(f)
        if f == "ts" and v is None:
            v = time.time()
        elif f == "think" and v is not None:
            v = 1 if v else 0
        elif f == "state_text" and v is not None:
            v = scrub(v)
        elif f in _JSON_COLS:
            v = _dumps(v)
        elif f in ("latency_ms", "mcts_value") and v is not None:
            v = float(v)
        elif f == "tokens" and v is not None:
            v = int(v)
        elif v is not None and not isinstance(v, (int, float)):
            v = str(v)
        row[_JSON_COLS.get(f, f)] = v
    con, owned = _connect(db)
    try:
        ensure(con)
        cols = ",".join(_COLS)
        marks = ",".join("?" for _ in _COLS)
        merge = ",".join(f"{c}=COALESCE(excluded.{c}, decision_ctx.{c})"
                         for c in _COLS if c != "decision_id")
        con.execute(f"INSERT INTO decision_ctx ({cols}) VALUES ({marks}) "
                    f"ON CONFLICT(decision_id) DO UPDATE SET {merge}",
                    tuple(row[c] for c in _COLS))
        if commit or owned:
            con.commit()
    finally:
        if owned:
            con.close()
    return True


# ------------------------------------------------------------------ keys
_NUMERIC = re.compile(r"^\d+$")


def is_excluded(key: str) -> bool:
    """Bench fixtures and demo forks are not decisions anyone acted on."""
    k = str(key or "")
    prefix, _, rest = k.rpartition("/")
    if prefix and any(p in ("bench", "demo") for p in prefix.split("/")):
        return True
    return any(seg in ("bench", "demo") for seg in rest.split("."))


def family(key: str) -> str:
    """`tool.bash` -> `tool.bash`; `live/decide.router.x.y` -> `live/decide.router`.
    Numeric segments (timestamps, ids) never split a family."""
    k = str(key or "")
    prefix, _, rest = k.rpartition("/")
    segs = ["*" if _NUMERIC.match(s) else s for s in rest.split(".")][:2]
    return (prefix + "/" if prefix else "") + ".".join(segs)


def rung_of(backend: str) -> str:
    """The rung a row WITHOUT a ctx row was answered on, read off its backend:
    `door:engine` -> `engine`, `local:evidence` -> `evidence`, and the Bash hook's
    `bash-hook` -> `evidence` (it stamps what the evidence rung would have said)."""
    b = str(backend or "none")
    if b == "bash-hook":
        return "evidence"
    for pre in ("door:", "local:"):
        if b.startswith(pre):
            return b[len(pre):]
    return b


# ------------------------------------------------------------------ read
def _parse_since(since: Any) -> Optional[float]:
    if since is None or since == "":
        return None
    if isinstance(since, (int, float)):
        return float(since)
    s = str(since).strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return float(s)
    from datetime import datetime, timezone

    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _loads(v: Optional[str]) -> Any:
    if v is None:
        return None
    try:
        return json.loads(v)
    except ValueError:
        return v


def iter_records(db: DbLike, since: Any = None, *, include_excluded: bool = False
                 ) -> Iterable[Dict[str, Any]]:
    """DecisionRecord dicts in time order. `ctx` is None when nothing was attached."""
    con, owned = _connect(db, readonly=True)
    try:
        t0 = _parse_since(since)
        has_ctx = _has_ctx(con)
        ctx_cols = [c for c in _COLS if c != "decision_id"]
        sel = ("d.id, d.ts, d.key, d.kind, d.value, d.probability, d.backend, d.state_sha, "
               "d.outcome, d.brier, d.resolved_at")
        if has_ctx:
            sel += ", c.decision_id, " + ", ".join("c." + c for c in ctx_cols)
            src = "decisions d LEFT JOIN decision_ctx c ON c.decision_id = d.id"
        else:
            src = "decisions d"
        where, args = "", ()
        if t0 is not None:
            where, args = " WHERE d.ts >= ?", (t0,)
        for r in con.execute(f"SELECT {sel} FROM {src}{where} ORDER BY d.ts", args):
            (did, ts, key, kind, value, p, backend, sha, o, brier, r_at) = r[:11]
            if not include_excluded and is_excluded(key):
                continue
            ctx = None
            rung = rung_of(backend)
            if has_ctx and r[11] is not None:
                vals = dict(zip(ctx_cols, r[12:]))
                ctx = {}
                for f in CTX_FIELDS:
                    if f == "ts":
                        continue
                    col = _JSON_COLS.get(f, f)
                    v = vals.get(col)
                    ctx[f] = _loads(v) if f in _JSON_COLS else v
                ctx["ts"] = vals.get("ts")
                if ctx.get("think") is not None:
                    ctx["think"] = bool(ctx["think"])
                rung = ctx.get("rung") or rung
            yield {"schema": SCHEMA_ID, "id": did, "ts": ts, "key": key, "kind": kind,
                   "value": value, "p": p, "backend": backend, "rung": rung,
                   "state_sha": sha, "ctx": ctx,
                   "outcome": None if o is None else int(o),
                   "brier": brier, "resolved_at": r_at}
    finally:
        if owned:
            con.close()


def export_records(db: DbLike, out_path: Union[str, Path], since: Any = None, *,
                   resolved_only: bool = False, joined_only: bool = False,
                   include_excluded: bool = False) -> Dict[str, int]:
    """Write DecisionRecord JSONL (`-` = stdout). Returns the counts written:
    rows, joined (a ctx row exists), resolved, joined_resolved."""
    counts = {"rows": 0, "joined": 0, "resolved": 0, "joined_resolved": 0}
    out = sys.stdout if str(out_path) == "-" else Path(out_path).open("w", encoding="utf-8")
    try:
        for rec in iter_records(db, since, include_excluded=include_excluded):
            joined = rec["ctx"] is not None
            resolved = rec["outcome"] is not None
            if (resolved_only and not resolved) or (joined_only and not joined):
                continue
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")
            counts["rows"] += 1
            counts["joined"] += joined
            counts["resolved"] += resolved
            counts["joined_resolved"] += joined and resolved
    finally:
        if out is not sys.stdout:
            out.close()
    return counts


def _bucket_case() -> str:
    arms = " ".join(f"WHEN d.probability < {hi} THEN {i}" for i, (_, hi) in enumerate(BUCKETS))
    return f"CASE {arms} ELSE {len(BUCKETS) - 1} END"


class _Acc:
    __slots__ = ("n", "hits", "sq", "pend", "exp", "b")

    def __init__(self) -> None:
        self.n = 0
        self.hits = 0
        self.sq = 0.0
        self.pend = 0
        self.exp = 0
        self.b = [[0, 0.0, 0] for _ in BUCKETS]  # n, sum p, hits

    def add(self, b: int, n: int, hits: int, sp: float, sq: float, pend: int,
            exp: int) -> None:
        self.n += n
        self.hits += hits
        self.sq += sq
        self.pend += pend
        self.exp += exp
        cell = self.b[b]
        cell[0] += n
        cell[1] += sp
        cell[2] += hits

    def report(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"n": self.n, "pending": self.pend, "expired": self.exp}
        if not self.n:
            return out
        base = self.hits / self.n
        brier = self.sq / self.n
        clim = base * (1.0 - base)
        ece = 0.0
        buckets = []
        for (lo, hi), (bn, sp, bh) in zip(BUCKETS, self.b):
            rng = f"{lo:.1f}-{min(hi, 1.0):.1f}"
            if not bn:
                buckets.append({"range": rng, "n": 0})
                continue
            mc, fr = sp / bn, bh / bn
            ece += bn / self.n * abs(mc - fr)
            buckets.append({"range": rng, "n": bn, "mean_conf": round(mc, 4),
                            "observed": round(fr, 4), "gap": round(fr - mc, 4)})
        out.update({"base_rate": round(base, 4), "brier": round(brier, 5),
                    "climatology": round(clim, 5),
                    "bss": None if clim <= 0 else round(1.0 - brier / clim, 4),
                    "ece": round(ece, 4), "buckets": buckets})
        return out


def stats(db: DbLike = None, *, now: Optional[float] = None,
          expire_days: float = EXPIRE_DAYS) -> Dict[str, Any]:
    """Brier, climatology, Brier skill score and ECE per key family and per rung.

    Excluded from every skill number: bench/* and demo keys (counted under
    `excluded`). A pending row older than `expire_days` is reported as `expired`,
    not as an open claim. `gap` in a bucket is observed - mean confidence, so a
    positive gap is UNDER-confidence."""
    con, owned = _connect(db, readonly=True)
    try:
        now = time.time() if now is None else float(now)
        cutoff = now - expire_days * 86400.0
        has_ctx = _has_ctx(con)
        rung_sql = "COALESCE(c.rung, '')" if has_ctx else "''"
        src = ("decisions d LEFT JOIN decision_ctx c ON c.decision_id = d.id" if has_ctx
               else "decisions d")
        nctx_sql = "SUM(c.decision_id IS NOT NULL)" if has_ctx else "0"
        q = (f"SELECT d.key, d.backend, {rung_sql}, {_bucket_case()} AS b,"
             " SUM(d.outcome IS NOT NULL), COALESCE(SUM(d.outcome), 0),"
             " COALESCE(SUM(CASE WHEN d.outcome IS NOT NULL THEN d.probability END), 0),"
             " COALESCE(SUM(CASE WHEN d.outcome IS NOT NULL THEN"
             "   (d.probability - d.outcome) * (d.probability - d.outcome) END), 0),"
             " SUM(d.outcome IS NULL AND d.ts >= ?), SUM(d.outcome IS NULL AND d.ts < ?),"
             f" {nctx_sql}")
        q += f" FROM {src} GROUP BY 1, 2, 3, 4"
        overall, excluded = _Acc(), _Acc()
        fams: Dict[str, _Acc] = {}
        rungs: Dict[str, _Acc] = {}
        fam_rung: Dict[str, _Acc] = {}
        joined = 0
        for key, backend, rung, b, n, hits, sp, sq, pend, exp, nctx in con.execute(
                q, (cutoff, cutoff)):
            args = (int(b), int(n or 0), int(hits or 0), float(sp or 0.0), float(sq or 0.0),
                    int(pend or 0), int(exp or 0))
            if is_excluded(key):
                excluded.add(*args)
                continue
            joined += int(nctx or 0)
            r = rung or rung_of(backend)
            f = family(key)
            overall.add(*args)
            fams.setdefault(f, _Acc()).add(*args)
            rungs.setdefault(r, _Acc()).add(*args)
            fam_rung.setdefault(f + " @ " + r, _Acc()).add(*args)
        label = str(db) if isinstance(db, (str, Path)) else (
            "<connection>" if db is not None else str(DEFAULT_DB))
        out = {"db": label,
               "schema": SCHEMA_ID, "ctx_table": has_ctx, "joined_rows": joined,
               "expire_days": expire_days, "overall": overall.report(),
               "by_family": {k: v.report() for k, v in sorted(fams.items(),
                                                              key=lambda kv: -kv[1].n)},
               "by_rung": {k: v.report() for k, v in sorted(rungs.items(),
                                                            key=lambda kv: -kv[1].n)},
               "by_family_rung": {k: v.report() for k, v in sorted(
                   fam_rung.items(), key=lambda kv: -kv[1].n)},
               "excluded": {k: v for k, v in excluded.report().items()
                            if k in ("n", "pending", "expired", "brier")}}
        return out
    finally:
        if owned:
            con.close()


# ------------------------------------------------------------------ CLI
def _fmt(name: str, r: Dict[str, Any]) -> str:
    if not r.get("n"):
        return f"  {name:<44} n=0  pending={r.get('pending', 0)} expired={r.get('expired', 0)}"
    bss = "n/a" if r.get("bss") is None else f"{r['bss']:+.4f}"
    return (f"  {name:<44} n={r['n']:<7} base={r['base_rate']:.4f} brier={r['brier']:.5f} "
            f"clim={r['climatology']:.5f} BSS={bss} ECE={r['ece']:.4f} "
            f"pending={r['pending']} expired={r['expired']}")


def _print_stats(s: Dict[str, Any]) -> None:
    print(f"awdecide DecisionRecord stats  db={s['db']}  ctx_table={s['ctx_table']}  "
          f"joined_rows={s['joined_rows']}  (pending older than {s['expire_days']:g}d = expired)")
    print(_fmt("OVERALL (bench/demo excluded)", s["overall"]))
    ex = s["excluded"]
    print(f"  excluded bench/demo rows: n={ex.get('n', 0)} pending={ex.get('pending', 0)}")
    for title, k in (("by key family", "by_family"), ("by rung", "by_rung"),
                     ("by family @ rung", "by_family_rung")):
        print(title + ":")
        for name, r in s[k].items():
            print(_fmt(name, r))
    print("reliability (gap = observed - mean_conf; + is under-confident):")
    for name, r in s["by_family"].items():
        for b in r.get("buckets") or []:
            if b.get("n"):
                print(f"  {name:<20} {b['range']}  n={b['n']:<7} conf={b['mean_conf']:.3f} "
                      f"obs={b['observed']:.3f} gap={b['gap']:+.3f}")


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m awdecide.record",
                                 description="DecisionRecord: export and honest skill stats")
    ap.add_argument("--db", default=None, help="ledger path (default: AWDECIDE_DB or "
                    "~/.aither/awdecide.db)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stats", help="Brier / climatology / BSS / ECE per family and rung")
    st.add_argument("--json", action="store_true")
    st.add_argument("--expire-days", type=float, default=EXPIRE_DAYS)
    ex = sub.add_parser("export", help="DecisionRecord JSONL")
    ex.add_argument("--since", default=None, help="epoch seconds or ISO-8601")
    ex.add_argument("--out", default="-", help="output path, - for stdout")
    ex.add_argument("--resolved-only", action="store_true")
    ex.add_argument("--joined-only", action="store_true")
    ex.add_argument("--include-excluded", action="store_true",
                    help="keep bench/demo rows")
    a = ap.parse_args(argv)
    db = a.db or DEFAULT_DB
    if not Path(db).exists():
        print(f"no ledger at {db}", file=sys.stderr)
        return 2
    if a.cmd == "stats":
        s = stats(db, expire_days=a.expire_days)
        if a.json:
            print(json.dumps(s, indent=2))
        else:
            _print_stats(s)
        return 0
    counts = export_records(db, a.out, a.since, resolved_only=a.resolved_only,
                            joined_only=a.joined_only, include_excluded=a.include_excluded)
    print(json.dumps({"exported": counts, "out": a.out, "since": a.since}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
