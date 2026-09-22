"""The door without a server: run its benches from the installed package.

`awdecide door-bench <name>` runs one of the vendored bench scripts
(`awdecide/door_local/tools/*_bench.py`) exactly as the write-up ran them, with the
shipped price table and a scratch journal dir, so a reader with nothing but
`pip install awdecide` gets the same table -- and the same exit code: 0 when the
bench's own verdict holds, 1 when it does not, 2 when it could not run. What it
measures on the SHIPPED dataset is the generated half (408 rows); the transcribed
rows of one fleet's logs are not carried, and the numbers say so in their row counts.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .door import VENDORED_DIR

TOOLS = VENDORED_DIR / "tools"
PRICES = TOOLS / "model_token_prices.yaml"

#: name -> (script, extra args). Every entry is a bench with its own verdict.
BENCHES: Dict[str, List[str]] = {
    "judge-holdout": ["judge_holdout_bench.py", "--seeds", "5", "--neural"],
    "judge-heldout": ["judge_heldout_bench.py"],
    "compact": ["compact_bench.py"],
    "cache-cost": ["cache_cost_bench.py"],
    "calibration": ["calibration_bench.py"],
    "judge": ["judge_bench.py", "--skip-llm"],
}


def _has_numpy() -> bool:
    try:
        import numpy  # noqa: F401
    except Exception:  # noqa: BLE001 -- absent or broken both mean "not measured"
        return False
    return True


def available() -> List[str]:
    return [n for n, (script, *_a) in BENCHES.items() if (TOOLS / script).is_file()]


def run_bench(name: str, extra: Optional[List[str]] = None, *, ckpt_dir: Optional[str] = None,
              json_out: bool = False) -> int:
    if name not in BENCHES:
        print(f"unknown bench {name!r}; have: {', '.join(sorted(BENCHES))}", file=sys.stderr)
        return 2
    script, *args = list(BENCHES[name])
    if "--neural" in args and not _has_numpy():
        # Without numpy the trainer cannot fit the rung and the leg would silently
        # tie the floor and read as "the rung adds nothing". Say what is missing and
        # run the engine-only measurement, whose bar is the tie.
        print("note: numpy is not installed, so the neural rung is not measured "
              "(pip install 'awdecide[door]'); running the engine-only leg", file=sys.stderr)
        args = [a for a in args if a != "--neural"]
    path = TOOLS / script
    if not path.is_file():
        print(f"COULD NOT RUN: {path} is not in this install (vendor_door.py --sync)",
              file=sys.stderr)
        return 2
    env = dict(os.environ)
    env.setdefault("AITHER_WM_CKPT_DIR", ckpt_dir or str(Path.home() / ".awdecide" / "bench-ckpt"))
    Path(env["AITHER_WM_CKPT_DIR"]).mkdir(parents=True, exist_ok=True)
    if PRICES.is_file():
        env.setdefault("AITHER_MODEL_PRICES", str(PRICES))
    cmd = [sys.executable, str(path), *args]
    if name in ("cache-cost", "judge") and PRICES.is_file() and "--prices" not in (extra or []):
        cmd += ["--prices", str(PRICES)]
    if json_out and "--json" not in cmd:
        cmd.append("--json")
    cmd += list(extra or [])
    try:
        return subprocess.call(cmd, cwd=str(VENDORED_DIR), env=env)
    except OSError as exc:
        print(f"COULD NOT RUN: {exc}", file=sys.stderr)
        return 2
