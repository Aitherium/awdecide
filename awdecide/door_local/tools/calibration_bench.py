#!/usr/bin/env python3
"""calibration_bench -- the unfair-coin test the public ran on Jev, run on our door.

The critique (2026-09-19): "an unfair coin that comes up heads 60% of the time" --
Jev's `choice` says 99/1; scanning p from 5% to 95%, its probabilities collapse to
the winner instead of following the coin. Its `noul` (P(yes)) was closer but
undershot by up to 9 points.

Here every p in the scan gets its own fork state; the door is asked
`yesno` "the next flip will be heads" and `choice` heads/tails; each answer is
followed by a REAL flip (Bernoulli(p), seeded) posted back as the outcome
(+1 if the answer matched the flip, -1 if not). We record the door's reported
probability at three points: cold (before any flip -- the LLM's stated number,
uncalibrated by construction), after 20 flips, after 100 flips. Expected
calibration error (mean |reported - p| over the scan) is the verdict.

In-process by default (the same engine the service runs; the LLM stubbed to a
Jev-shaped collapser that always says the likelier side with 0.99), or against a
live door with --url/--token (then cold answers come from the real brain).

Exit 0 when ECE after 100 flips <= 0.06 for yesno AND choice (Jev's noul was
0.09 at its worst, its choice ~0.4); 1 when not; 2 when it could not judge.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import ssl
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

PS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def coin_state(p: float) -> str:
    return f"We have an unfair coin that comes up heads {p * 100:.1f}% of the time."


class CollapsingLLM:
    """Jev's measured shape: the likelier side, stated at 0.99."""

    def __call__(self, prompt: str):
        pct = float(prompt.split("heads ", 1)[1].split("%", 1)[0])
        if 'Allowed answers: "yes"' in prompt or '"yes"' in prompt:
            return {"answer": "yes" if pct >= 50 else "no", "confidence": 0.99}
        return {"answer": "heads" if pct >= 50 else "tails", "confidence": 0.99}


class Live:
    def __init__(self, url: str, token: str):
        self.url, self.h = (
            url.rstrip("/"),
            {"Content-Type": "application/json", **({"X-WM-Token": token} if token else {})},
        )
        self.ctx = ssl._create_unverified_context() if url.startswith("https") else None

    def post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.url + path, data=json.dumps(body).encode(), headers=self.h
        )
        with urllib.request.urlopen(req, timeout=120, context=self.ctx) as r:
            return json.loads(r.read())

    def decide(self, item):
        return self.post("/decide", item)

    def outcome(self, body):
        return self.post("/decide/outcome", body)


def p_heads_from(answer: dict, kind: str) -> Optional[float]:
    """The door's reported P(heads) for this kind, whatever rung answered."""
    if kind == "yesno":
        return answer.get("p_yes")
    probs = answer.get("probabilities") or {}
    if "heads" in probs:
        return probs["heads"]
    if "tails" in probs:
        return round(1.0 - probs["tails"], 3)
    if answer.get("source") == "llm":  # stated number for the stated side
        c = float(answer.get("confidence") or 0.0)
        return c if answer.get("answer") == "heads" else 1.0 - c
    return None


def run(door, flips: int, seed: int) -> Dict[str, Dict[str, List[float]]]:
    rng = random.Random(seed)
    out = {
        "yesno": {"cold": [], "at20": [], "final": []},
        "choice": {"cold": [], "at20": [], "final": []},
    }
    for p in PS:
        for kind in ("yesno", "choice"):
            dom = f"decide.coin.{kind}.{int(p * 100)}"
            item = {
                "domain": dom,
                "state": coin_state(p),
                "kind": kind,
                "question": "Which side will come up on the next flip of this coin?"
                if kind == "choice"
                else "The next flip of this coin will come up heads.",
            }
            if kind == "choice":
                item["options"] = ["heads", "tails"]
            for i in range(flips + 1):
                a = door.decide(item)
                ph = p_heads_from(a, kind)
                if i == 0:
                    out[kind]["cold"].append(ph if ph is not None else 0.5)
                if i == 20:
                    out[kind]["at20"].append(ph if ph is not None else 0.5)
                if i == flips:
                    out[kind]["final"].append(ph if ph is not None else 0.5)
                    break
                flip_heads = rng.random() < p
                if kind == "yesno":
                    said_heads = a["answer"] == "yes"
                else:
                    said_heads = a["answer"] == "heads"
                reward = 1.0 if said_heads == flip_heads else -1.0
                door.outcome({"decision_id": a["decision_id"], "reward": reward})
                # the flip also grades the OTHER side (a coin reveals both outcomes)
                other = (
                    ("no" if said_heads else "yes")
                    if kind == "yesno"
                    else ("tails" if said_heads else "heads")
                )
                door.outcome(
                    {"domain": dom, "state": coin_state(p), "answer": other, "reward": -reward}
                )
    return out


def ece(reported: List[float]) -> float:
    return round(sum(abs(r - p) for r, p in zip(reported, PS)) / len(PS), 3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--flips", type=int, default=100)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--url")
    ap.add_argument("--token", default=os.environ.get("AITHER_WM_INTERNAL_TOKEN", ""))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.url:
        door = Live(a.url, a.token)
    else:
        os.environ["AITHER_WM_CKPT_DIR"] = tempfile.mkdtemp(prefix="coin-")
        import code_domains
        import decide

        if not code_domains._MLP_OK:
            print("COULD NOT JUDGE: world_model package not importable")
            return 2

        class InProc:
            def __init__(self):
                self.d = decide.Decider(
                    code_domains.DomainEngines(), llm=CollapsingLLM(), embed_enabled=False
                )

            def decide(self, item):
                return self.d.decide(item)

            def outcome(self, body):
                return self.d.outcome(body)

        door = InProc()

    res = run(door, a.flips, a.seed)
    report = {"p_scan": PS, "flips": a.flips}
    for kind in ("yesno", "choice"):
        report[kind] = {k: v for k, v in res[kind].items()}
        report[kind]["ece"] = {k: ece(v) for k, v in res[kind].items()}
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"unfair coin, p(heads) scanned over {PS}; {a.flips} flips per p\n")
        print(f"{'kind':8}{'stage':8}{'ECE':>7}   reported P(heads) per p")
        for kind in ("yesno", "choice"):
            for stage in ("cold", "at20", "final"):
                print(
                    f"{kind:8}{stage:8}{report[kind]['ece'][stage]:>7.3f}   "
                    + " ".join(f"{x:.2f}" for x in res[kind][stage])
                )
        print(
            "\n(Jev, per the public test: choice collapses to 0.99/0.01 -> ECE ~0.4; "
            "noul undershoots, worst 0.09)"
        )
    ok = report["yesno"]["ece"]["final"] <= 0.06 and report["choice"]["ece"]["final"] <= 0.06
    print(
        "\nVERDICT:",
        "calibrated -- reported probabilities follow the coin"
        if ok
        else "NOT calibrated after the flips",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
