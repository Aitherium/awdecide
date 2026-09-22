#!/usr/bin/env python3
"""vendor_door -- carry the decision door's engine into awdecide, scrubbed, and prove it.

`awdecide.door_local` is the door WITHOUT a server: the same `decide.py` /
`judge.py` / `compact.py` that run behind the HTTP door, the domain engines they
answer from (`code_domains.py` + the `world_model` tabular engine), and the bench
scripts plus the GENERATED half of the labelled dataset -- so every number in
"Decisions You Already Made Should Never Be Paid For Twice" reproduces from
`pip install awdecide`, with no fleet and no private tree.

THIS SCRIPT IS THE ONLY THING ALLOWED TO WRITE `awdecide/door_local/`. A vendored
copy is a fork waiting to happen, so the same script is the parity check: it
re-derives the copy from the sources into a temp dir and diffs. What it carries
is not byte-identical to the source, on purpose -- a small, DECLARED set of
rewrites (SCRUB below) removes the shape of one particular deployment: service
host names, a model alias, absolute paths on the author's machine, and internal
ticket / checker ids in comments. Nothing behavioural: every rewrite is a string
default that the environment already overrides, and every vendored module is
byte-compiled afterwards. The transcribed rows of the dataset (real logs of one
fleet) are NOT carried; the generated rows (outputs of real commands run on a
scratch tree) are.

    python scripts/vendor_door.py --sync       # (re)derive door_local from the sources
    python scripts/vendor_door.py              # check: door_local == derive(sources)?
    python scripts/vendor_door.py --offline    # check: door_local == VENDORED.json?
    python scripts/vendor_door.py --self-test  # prove the drift detector fires

Exit 0 in step, 1 diverged / missing, 2 could not judge (no sources and not --offline).
Sources: $AITHER_WM_SVC_DIR (the door tree) and $AITHER_WORLD_MODEL_PKG (the dir
that CONTAINS `world_model/`), with the author's checkouts as fallbacks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "awdecide"
DEST = PKG / "door_local"
MANIFEST = DEST / "VENDORED.json"

SVC_CANDIDATES = [os.environ.get("AITHER_WM_SVC_DIR"), r"D:\arc-agi-3\arc-world-model-svc"]
WM_CANDIDATES = [
    os.environ.get("AITHER_WORLD_MODEL_PKG"),
    str(HERE.parents[3] / "packages" / "world-model"),
    r"C:\AitherOS-Fresh\packages\world-model",
]
PRICES_CANDIDATES = [
    os.environ.get("AITHER_MODEL_PRICES"),
    str(HERE.parents[2] / "config" / "model_token_prices.yaml"),
]

#: Files carried from the door tree, relative to it. Order is the manifest order.
DOOR_FILES = [
    "decide.py", "judge.py", "compact.py", "multi.py", "code_domains.py",
    "tools/hostpaths.py", "tools/judge_bench.py", "tools/judge_holdout_bench.py",
    "tools/judge_heldout_bench.py", "tools/compact_bench.py", "tools/cache_cost_bench.py",
    "tools/calibration_bench.py", "tools/reliability.py", "tools/train_neural_rung.py",
    "tools/eval_cases/README.md",
]
CORPUS_DIR = "tools/compact_corpus"          # every file, as is
CASES = "tools/eval_cases/cases.jsonl"       # generated rows only
WM_SUFFIXES = (".py",)                       # world_model: code only, no tests

#: The declared rewrites. (pattern, replacement, why). Applied to every carried
#: text file in this order; the manifest records the version so a check can tell
#: "the scrub changed" from "the source changed".
SCRUB_VERSION = 2
SCRUB: List[Tuple[str, str, str]] = [
    (r"https://aitheros-microscheduler:8150/v1", "http://127.0.0.1:8080/v1",
     "the default brain is whatever OpenAI-style server the user runs, not one fleet's"),
    (r"https://aitheros-world-model:8197", "http://127.0.0.1:8197", "in-fleet door host"),
    (r"http://aither-code-embed:8229", "http://127.0.0.1:8229", "in-fleet embedder host"),
    (r"aither-code-embed-0\.6b", "local-embed", "one deployment's embedder alias"),
    (r"aither-code-embed / aither-vllm-code-embed", "your embedding server", "aliases"),
    (r"aither-code-embed", "your embedder", "alias in prose"),
    (r"\baither-orchestrator\b", "default", "one deployment's model alias"),
    (r"localhost/aitheros-genesis:latest", "localhost/example-service:latest", "fixture text"),
    # no \b here: in a JSON row the name follows a literal backslash-n, and a word
    # boundary between `n` and `a` does not exist -- the escape hides the name
    (r"aitheros-[a-z0-9-]+", "the-service", "container names in prose and captured output"),
    (r"garg\.aitherium\.com", "client.example.com", "a customer's hostname in a captured log"),
    (r"garg[a-z0-9-]*", "client", "a customer's name in captured output"),
    (r"/app/AitherOS/Library/Data/tls/ca-chain\.pem", "/etc/ssl/certs/door-ca.pem", "CA path"),
    (r"[A-Za-z]:[\\/]AitherOS-Fresh", lambda m: m.group(0)[:3] + "source", "author's checkout"),
    (r"/mnt/c/AitherOS-Fresh", "/mnt/c/source", "author's checkout (WSL)"),
    (r"arc-agi-3", "awdecide", "author's data root"),
    (r"\bD-\d{3,5}\b", "D-ref", "internal ledger id"),
    (r"\b(NX|SEC|PQ|HYG|QS|EMC|ST|AWG|NAV|CSR|ACG|MCP|BW|ONB|AC|WT)\d{3}[A-Z]?\b",
     lambda m: m.group(1) + "-ref", "internal checker rule id"),
]

INIT_PY = '''"""awdecide.door_local -- the decision door in-process, no server.

Vendored by scripts/vendor_door.py from the door's service tree (see VENDORED.json
for the exact sources, hashes and the declared rewrites). Do not edit files here;
edit the source and re-run `--sync`.

The vendored modules import each other by their flat names (`import decide`,
`import judge`, `import code_domains`), exactly as they do in the service, so this
package puts its own directory first on `sys.path` when imported. Those names are
therefore taken while it is loaded -- if your project has a top-level `decide`
or `judge` module, import this before it or use `awdecide.local` only.

    from awdecide.door_local import decider, judge_for
    d = decider()                     # journals under $AITHER_WM_CKPT_DIR or ~/.awdecide/door
    j = judge_for(d)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
os.environ.setdefault("AITHER_WM_CKPT_DIR", str(Path.home() / ".awdecide" / "door"))
Path(os.environ["AITHER_WM_CKPT_DIR"]).mkdir(parents=True, exist_ok=True)

TOOLS = HERE / "tools"
PRICES = TOOLS / "model_token_prices.yaml"


def decider(llm=None, **kw):
    """A Decider over the vendored engines. `llm` is an optional callable
    prompt -> {"answer", "confidence"}; None means no model rung (cold forks read
    `source=none` and are never guessed at)."""
    import code_domains  # type: ignore
    import decide  # type: ignore

    if not code_domains._MLP_OK:
        raise RuntimeError("the vendored world_model engine failed to import")
    kw.setdefault("embed_enabled", False)
    return decide.Decider(code_domains.DomainEngines(), llm=llm, **kw)


def judge_for(d):
    import judge  # type: ignore

    return judge.Judge(d)
'''


# ------------------------------------------------------------------ helpers
def _first_dir(cands: List[Optional[str]], probe: str) -> Optional[Path]:
    for c in cands:
        if c and (Path(c) / probe).exists():
            return Path(c)
    return None


def _first_file(cands: List[Optional[str]]) -> Optional[Path]:
    for c in cands:
        if c and Path(c).is_file():
            return Path(c)
    return None


def scrub(text: str) -> str:
    for pat, rep, _why in SCRUB:
        text = re.sub(pat, rep, text)
    return text


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def synthetic_git_log(n: int = 40) -> str:
    """A deterministic `git log --stat -n 40` of a repository that does not exist:
    conventional-commit subjects, two-line bodies, per-file stat rows and the
    summary line, so every compact_bench label for this item has the same number
    of hits per commit as on a real log. Content is generic on purpose."""
    kinds = ["feat", "fix", "docs", "chore", "refactor", "test", "perf", "build", "ci"]
    scopes = ["api", "cli", "ledger", "router", "docs", "bench", "store", "auth"]
    files = ["src/api/routes.py", "src/cli/main.py", "src/ledger/store.py", "README.md",
             "tests/test_routes.py", "tests/test_store.py", "docs/guide.md", "pyproject.toml"]
    out: List[str] = []
    for i in range(n):
        h = hashlib.sha1(f"synthetic-commit-{i}".encode()).hexdigest()
        kind, scope = kinds[i % len(kinds)], scopes[(i * 3) % len(scopes)]
        out += [f"commit {h}", "Author: Example Dev <dev@example.com>",
                f"Date:   Mon Sep {1 + i % 28} 12:{i % 60:02d}:00 2026 +0000", "",
                f"    {kind}({scope}): change number {i} in the {scope} module", "",
                f"    Explains why change {i} was made and what it replaces.",
                "    Verified by the suite that covers this module.", ""]
        k = 1 + (i % 3)
        total_ins, total_del = 0, 0
        for j in range(k):
            f = files[(i + j) % len(files)]
            ins, dele = 3 + (i * 7 + j) % 40, (i + j) % 9
            total_ins += ins
            total_del += dele
            out.append(f" {f:<28} | {ins + dele:>3} " + "+" * min(ins, 20) + "-" * min(dele, 8))
        out.append(f" {k} file{'s' if k != 1 else ''} changed, {total_ins} insertions(+), "
                   f"{total_del} deletions(-)")
        out.append("")
    return "\n".join(out) + "\n"


def _git_head(path: Path) -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=path,
                              capture_output=True, text=True, encoding="utf-8",
                              timeout=20).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def derive(svc: Path, wm: Path, prices: Optional[Path], out: Path) -> Dict[str, str]:
    """Build the vendored tree at `out`. Returns {relpath: sha256} of what was written."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    written: Dict[str, str] = {}

    def put(rel: str, data: bytes) -> None:
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        written[rel] = _sha(data)

    for rel in DOOR_FILES:
        src = svc / rel
        if not src.is_file():
            raise FileNotFoundError(f"door source missing: {src}")
        put(rel, scrub(src.read_text(encoding="utf-8")).encode("utf-8"))
    for src in sorted((svc / CORPUS_DIR).iterdir()):
        if not src.is_file() or src.name.startswith("."):
            continue
        if src.name == "git_log_stat.txt":
            # ours is `git log --stat -n 40` of the platform repo: forty commit
            # subjects naming what we run. Same SHAPE, neutral content -- the
            # bench's labels (commit header, conventional subject, stat summary)
            # match it exactly as they match the real one.
            put(f"{CORPUS_DIR}/{src.name}", synthetic_git_log().encode("utf-8"))
            continue
        text = src.read_text(encoding="utf-8", errors="replace")
        put(f"{CORPUS_DIR}/{src.name}", scrub(text).encode("utf-8"))
    kept, dropped = [], 0
    for line in (svc / CASES).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") == "generated":
            kept.append(json.dumps(row, ensure_ascii=False))
        else:
            dropped += 1
    put(CASES, (scrub("\n".join(kept)) + "\n").encode("utf-8"))
    for src in sorted((wm / "world_model").rglob("*")):
        if not src.is_file() or src.suffix not in WM_SUFFIXES:
            continue
        if "tests" in src.relative_to(wm).parts or "__pycache__" in src.parts:
            continue
        put(src.relative_to(wm).as_posix(), scrub(src.read_text(encoding="utf-8")).encode("utf-8"))
    if prices is not None:
        put("tools/model_token_prices.yaml",
            scrub(prices.read_text(encoding="utf-8")).encode("utf-8"))
    put("__init__.py", INIT_PY.encode("utf-8"))
    put("tools/__init__.py", b"")
    written["_cases_dropped_transcribed"] = str(dropped)  # recorded, not a file
    return written


def compile_all(root: Path) -> List[str]:
    bad = []
    for p in sorted(root.rglob("*.py")):
        try:
            py_compile.compile(str(p), doraise=True)
        except py_compile.PyCompileError as exc:
            bad.append(f"{p.relative_to(root)}: {exc.msg.splitlines()[0]}")
    return bad


def snapshot(root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.name != "VENDORED.json":
            out[p.relative_to(root).as_posix()] = _sha(p.read_bytes())
    return out


def diff(a: Dict[str, str], b: Dict[str, str]) -> List[str]:
    out = []
    for k in sorted(set(a) | set(b)):
        if k.startswith("_"):
            continue
        if k not in a:
            out.append(f"only in door_local: {k}")
        elif k not in b:
            out.append(f"only in derived: {k}")
        elif a[k] != b[k]:
            out.append(f"differs: {k}")
    return out


# ------------------------------------------------------------------ modes
def sync(svc: Path, wm: Path, prices: Optional[Path]) -> int:
    with tempfile.TemporaryDirectory(prefix="door-vendor-") as td:
        tmp = Path(td) / "door_local"
        written = derive(svc, wm, prices, tmp)
        bad = compile_all(tmp)
        if bad:
            print("REFUSED: scrubbed copy does not compile:\n  " + "\n  ".join(bad))
            return 1
        if DEST.exists():
            shutil.rmtree(DEST)
        shutil.copytree(tmp, DEST, ignore=shutil.ignore_patterns("__pycache__"))
    manifest = {
        "scrub_version": SCRUB_VERSION,
        # the patterns themselves name the shapes being removed, so the manifest
        # records their hash and the reasons, never the patterns
        "scrub_sha256": _sha(json.dumps([p for p, _r, _w in SCRUB]).encode("utf-8")),
        "scrub_reasons": [w for _p, _r, w in SCRUB],
        # commits, not paths: a path names the author's machine, a commit names the source
        "door_source": {"repo": "arc-world-model-svc", "commit": _git_head(svc)},
        "world_model_source": {"repo": "AitherOS packages/world-model", "commit": _git_head(wm)},
        "prices_source": "AitherOS config/model_token_prices.yaml" if prices else None,
        "cases_dropped_transcribed": int(written.pop("_cases_dropped_transcribed", "0")),
        "files": written,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"vendored {len(written)} files into {DEST} "
          f"(door {manifest['door_source']['commit']}, "
          f"world_model {manifest['world_model_source']['commit']}, "
          f"{manifest['cases_dropped_transcribed']} transcribed rows not carried)")
    return 0


def check(svc: Optional[Path], wm: Optional[Path], prices: Optional[Path], offline: bool) -> int:
    if not DEST.is_dir() or not MANIFEST.is_file():
        print(f"DIVERGED: {DEST} or its manifest is missing -- run --sync")
        return 1
    have = snapshot(DEST)
    if offline:
        want = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
        want = {k: v for k, v in want.items() if not k.startswith("_")}
        d = diff(want, have)
        label = "manifest"
    else:
        if svc is None or wm is None:
            print("COULD NOT JUDGE: no door / world_model source tree on this box "
                  "(set AITHER_WM_SVC_DIR / AITHER_WORLD_MODEL_PKG, or use --offline)")
            return 2
        with tempfile.TemporaryDirectory(prefix="door-vendor-") as td:
            derived = derive(svc, wm, prices, Path(td) / "door_local")
            derived = {k: v for k, v in derived.items() if not k.startswith("_")}
        d = diff(derived, have)
        label = "derive(sources)"
    if d:
        print(f"DIVERGED from {label} ({len(d)}):\n  " + "\n  ".join(d[:40]))
        return 1
    print(f"in step with {label}: {len(have)} files")
    return 0


def self_test() -> int:
    # the scrub removes every declared shape and leaves code compilable
    sample = ('URL = "https://aitheros-microscheduler:8150/v1"  # see D-1234 and PQ010\n'
              'P = r"C:\\AitherOS-Fresh\\x"; M = "aither-orchestrator"\n'
              'E = "aither-code-embed-0.6b"\n')
    out = scrub(sample)
    for bad in ("aitheros-", "D-1234", "PQ010", "AitherOS-Fresh", "aither-orchestrator",
                "aither-code-embed"):
        if bad in out:
            print(f"SELF-TEST FAILED: scrub left {bad!r} in {out!r}")
            return 1
    try:
        compile(out, "<scrubbed>", "exec")
    except SyntaxError as exc:
        print(f"SELF-TEST FAILED: scrubbed sample does not compile: {exc}")
        return 1
    # the detector: a mutated byte in the vendored copy is a diff
    a = {"x.py": "1", "y.py": "2"}
    b = {"x.py": "1", "y.py": "3"}
    if diff(a, b) != ["differs: y.py"] or diff(a, {"x.py": "1"}) != ["only in derived: y.py"]:
        print("SELF-TEST FAILED: diff did not report the mutation")
        return 1
    print("SELF-TEST PASSED: scrub removes every declared shape, compiles, and drift is detected")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    svc = _first_dir(SVC_CANDIDATES, "decide.py")
    wm = _first_dir(WM_CANDIDATES, "world_model/__init__.py")
    prices = _first_file(PRICES_CANDIDATES)
    if a.sync:
        if svc is None or wm is None:
            print("COULD NOT SYNC: door tree or world_model package not found "
                  "(AITHER_WM_SVC_DIR / AITHER_WORLD_MODEL_PKG)")
            return 2
        return sync(svc, wm, prices)
    return check(svc, wm, prices, a.offline)


if __name__ == "__main__":
    sys.exit(main())
