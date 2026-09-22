#!/usr/bin/env python3
"""judge_bench -- our decision door vs an LLM judge, on the same grading task.

The public claim being answered (2026-09-19): an LLM-judge agent grading other
agents' work costs ~34 s and ~$0.08 per task with a frontier model, and a fast
classifier does it in 0.36 s for $0.00004. A classifier that cannot learn from a
correction is wrong the same way forever, so this measures BOTH halves: accuracy
against a labelled set, and cost per task.

Task: N agent-run outputs (pytest tails, build logs, tool errors, deploy output),
each with three criteria and a KNOWN answer per criterion. Three graders, same
items, same order:

  llm-judge    one model call per (output, criterion) through MicroScheduler,
               strict yes/no. The shape people run today.
  door cold    /judge with an empty store: the ladder falls to the same brain,
               so this measures the OVERHEAD of the door, not a different model.
  door warm    the same items after the labels have been taught once
               (/judge/outcome) -- what the second week of running evals costs.

Accuracy is against the labels; latency is wall clock; "model calls" is the
number of times a model was asked at all.

  python tools/judge_bench.py                         # in-process, needs --brain for the llm legs
  python tools/judge_bench.py --url https://127.0.0.1:8197 --token $AITHER_WM_INTERNAL_TOKEN

Exit 0 when the warm door is at least as accurate as the llm judge with strictly
fewer model calls, 1 when not, 2 when it could not judge.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

# (name, output, [(criterion, truth)])
CASES: List[Tuple[str, str, List[Tuple[str, bool]]]] = [
    (
        "pytest-green",
        "============================= test session starts =====================\n"
        "collected 42 items\n\ndev/tests/test_router.py ........................\n"
        "dev/tests/test_gate.py ..................\n\n"
        "======================== 42 passed, 1 warning in 4.02s ================\n",
        [("the test suite passed", True), ("an error or traceback is present", False),
         ("at least one test ran", True)],
    ),
    (
        "pytest-red",
        "collected 42 items\n\ndev/tests/test_router.py .......F................\n"
        "E   AssertionError: expected 'engine', got 'llm'\n"
        "======================== 1 failed, 41 passed in 4.55s =================\n",
        [("the test suite passed", False), ("an error or traceback is present", True),
         ("at least one test ran", True)],
    ),
    (
        "import-crash",
        'Traceback (most recent call last):\n  File "/app/svc/main.py", line 3, in <module>\n'
        "    from lib.core import AitherService\nModuleNotFoundError: No module named 'lib'\n"
        "exited with code 1\n",
        [("the command succeeded", False), ("an error or traceback is present", True),
         ("the failure names a missing dependency", True)],
    ),
    (
        "build-ok",
        "STEP 12/12: COMMIT localhost/example-service:latest\n"
        "Successfully tagged localhost/example-service:latest\nexit code: 0\n",
        [("the command succeeded", True), ("an error or traceback is present", False),
         ("an image was produced", True)],
    ),
    (
        "deploy-denied",
        "Error: permission denied while trying to connect to the Docker daemon socket\n"
        "exited with code 126\n",
        [("the command succeeded", False), ("an error or traceback is present", True),
         ("the failure is a permissions problem", True)],
    ),
    (
        "lint-clean",
        "ruff check .\nAll checks passed!\nexit code: 0\n",
        [("the command succeeded", True), ("an error or traceback is present", False),
         ("no violations were reported", True)],
    ),
    (
        "lint-dirty",
        "ruff check .\nservices/x.py:88:1: E402 module level import not at top of file\n"
        "Found 1 error.\nexit code: 1\n",
        [("the command succeeded", False), ("an error or traceback is present", True),
         ("no violations were reported", False)],
    ),
    (
        "probe-timeout",
        "curl: (28) Operation timed out after 30001 milliseconds with 0 bytes received\n"
        "exited with code 28\n",
        [("the command succeeded", False), ("an error or traceback is present", True),
         ("the failure is a timeout", True)],
    ),
]


def _ctx(url: str) -> Optional[ssl.SSLContext]:
    if not url.startswith("https"):
        return None
    ctx = ssl.create_default_context()
    for cand in (os.environ.get("SSL_CERT_FILE"), "/certs/ca-chain.pem"):
        if cand and os.path.isfile(cand):
            ctx.load_verify_locations(cand)
            return ctx
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def post(url: str, body: dict, headers: dict, timeout: float = 180.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers}
    )
    with urllib.request.urlopen(req, timeout=timeout, context=_ctx(url)) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


PRICES_DEFAULT = os.environ.get(
    "AITHER_MODEL_PRICES", r"C:\source\AitherOS\config\model_token_prices.yaml"
)


def load_prices(path: str) -> Dict[str, Dict[str, Any]]:
    """Published $/1M token prices. A model that is not listed is UNPRICED, never
    free -- a silent zero is how a cost claim becomes a lie."""
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        return {}
    try:
        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except OSError:
        return {}
    return doc.get("models") or {}


def cost_usd(model: str, prompt_tok: int, completion_tok: int,
             prices: Dict[str, Dict[str, Any]]) -> Optional[float]:
    row = prices.get(model)
    if not row or row.get("local"):
        return None
    try:
        return (
            prompt_tok * float(row["input_usd_per_1m"])
            + completion_tok * float(row["output_usd_per_1m"])
        ) / 1_000_000.0
    except (KeyError, TypeError, ValueError):
        return None


_CLOUD_PROVIDERS = {"deepseek", "anthropic", "openai", "google", "moonshot", "openrouter",
                    "bedrock"}


def _misrouted(price_row: Dict[str, Any], served_by: Dict[str, int]) -> Tuple[bool, Optional[str]]:
    """Did the backend that answered match the provider the price row is for?

    served_by names are MicroScheduler backend labels (`deepseek_api`, `anthropic`,
    `vllm_gemma4_12b`); the price table names providers (`deepseek`, `anthropic`,
    `local`). A local row answered by any cloud provider, or a cloud row answered
    by a different provider, is MISROUTED: the leg ran, but its $/item would be
    priced for a model that never saw the prompt, so it stays unpriced and says why.
    """
    got = {k.replace("_api", "") for k in served_by if k != "?"}
    if not price_row or not got:
        return False, None
    want = str(price_row.get("provider") or "")
    if price_row.get("local"):
        bad = sorted(got & _CLOUD_PROVIDERS)
        if bad:
            return True, f"local row answered by cloud backend(s) {bad}; not priced as local"
        return False, None
    if want and got != {want}:
        return True, f"priced as {want} but answered by {sorted(got)}; not priced"
    return False, None


class LLMJudge:
    """One model call per (output, criterion) -- the shape being replaced."""

    def __init__(self, brain: str, model: str, max_tokens: int = 512):
        self.brain, self.model, self.calls, self.errors = brain.rstrip("/"), model, 0, 0
        self.max_tokens = max_tokens
        self.prompt_tokens = self.completion_tokens = 0
        # who ANSWERED, per MicroScheduler's `aither_route.served_by`. A request
        # for a down local model is answered by the cloud fallback with a 200,
        # and a request for a frontier name the plane does not know is answered
        # by deepseek (measured 2026-09-20: claude-sonnet-4-5 -> deepseek_api).
        # Pricing the REQUESTED name in either case is a lie, so run_llm refuses.
        self.served_by: Dict[str, int] = {}

    def judge(self, output: str, criteria: List[str]) -> List[Optional[bool]]:
        out: List[Optional[bool]] = []
        for c in criteria:
            self.calls += 1
            body = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [
                    {"role": "system", "content": "Answer with exactly one word: yes or no."},
                    {
                        "role": "user",
                        "content": f"Criterion: {c}\n\nOutput:\n{output[:4000]}\n\n"
                        "Is the criterion satisfied? yes or no.",
                    },
                ],
            }
            try:
                doc = post(
                    self.brain + "/chat/completions", body, {"Authorization": "Bearer local"}
                )
                usage = doc.get("usage") or {}
                route = doc.get("aither_route") or {}
                sb = str(route.get("served_by") or doc.get("model") or "?")
                self.served_by[sb] = self.served_by.get(sb, 0) + 1
                self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                self.completion_tokens += int(usage.get("completion_tokens") or 0)
                msg = (doc.get("choices") or [{}])[0].get("message", {}) or {}
                text = ((msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")).lower()
            except Exception:  # noqa: BLE001
                self.errors += 1
                out.append(None)
                continue
            yes, no = text.rfind("yes"), text.rfind("no")
            out.append(None if yes < 0 and no < 0 else yes > no)
        return out


class DoorLive:
    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.h = {"X-WM-Token": token} if token else {}

    def judge(self, output: str, criteria: List[str], domain: str) -> dict:
        return post(self.url + "/judge", {"output": output, "criteria": criteria, "domain": domain},
                    self.h)

    def teach(self, output: str, criterion: str, should_pass: bool, domain: str) -> dict:
        return post(
            self.url + "/judge/outcome",
            {"output": output, "criterion": criterion, "should_pass": should_pass,
             "domain": domain},
            self.h,
        )


class DoorInProc:
    def __init__(self, brain: Optional[str], model: str, neural_model: Optional[str] = None):
        import tempfile

        os.environ.setdefault("AITHER_WM_CKPT_DIR", tempfile.mkdtemp(prefix="judgebench-"))
        import code_domains  # type: ignore
        import decide  # type: ignore
        import judge as judge_mod  # type: ignore

        if not code_domains._MLP_OK:
            raise RuntimeError("world_model package not importable")
        self.brain = LLMJudge(brain, model) if brain else None
        llm = None
        if self.brain is not None:
            def llm(prompt: str):  # noqa: E306 -- the door's llm rung, same brain
                crit = prompt.split("Criterion:", 1)[-1].split("\n", 1)[0].strip()
                body = prompt.split("--- output", 1)[-1]
                got = self.brain.judge(body, [crit])[0]
                if got is None:
                    return None
                return {"answer": "yes" if got else "no", "confidence": 0.6}
        # The neural rung is OFF unless a model file is named: 'door cold' must
        # keep measuring what it always measured (the overhead of the ladder over
        # the same brain), or the two legs would not be comparable run to run.
        self.d = decide.Decider(
            code_domains.DomainEngines(),
            llm=llm,
            embed_enabled=False,
            neural_model=Path(neural_model) if neural_model else None,
            neural_enabled=bool(neural_model),
        )
        self.j = judge_mod.Judge(self.d)

    def judge(self, output: str, criteria: List[str], domain: str) -> dict:
        return self.j.judge(output, criteria, domain=domain)

    def teach(self, output: str, criterion: str, should_pass: bool, domain: str) -> dict:
        return self.j.teach(output, criterion, should_pass, domain=domain)


def run_llm(judge: LLMJudge, prices: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    t0, right, total, unknown = time.perf_counter(), 0, 0, 0
    for _name, output, crits in CASES:
        got = judge.judge(output, [c for c, _ in crits])
        for (_c, truth), g in zip(crits, got):
            total += 1
            if g is None:
                unknown += 1
            elif g == truth:
                right += 1
    wall = round(time.perf_counter() - t0, 1)
    row = (prices or {}).get(judge.model) or {}
    misrouted, why = _misrouted(row, judge.served_by)
    usd = None if misrouted else cost_usd(
        judge.model, judge.prompt_tokens, judge.completion_tokens, prices or {}
    )
    # A leg whose calls FAILED is not a baseline. Measured 2026-09-20: a second
    # claude-sonnet-4 run hit 13 errors out of 24 (only 11 answers reached it),
    # scored 45.8%, and the door "beat" it -- a win over a broken baseline is not
    # a win. `degraded` is what main() refuses to compare against.
    degraded = judge.errors > 0 or unknown > 0
    return {
        "model": judge.model,
        "served_by": judge.served_by,
        "misrouted": misrouted,
        "unpriced_reason": why,
        "degraded": degraded,
        "accuracy_pct": round(100.0 * right / max(1, total), 1),
        "unparsed": unknown,
        "model_calls": judge.calls,
        "errors": judge.errors,
        "wall_s": wall,
        "items": total,
        "prompt_tokens": judge.prompt_tokens,
        "completion_tokens": judge.completion_tokens,
        "usd_total": None if usd is None else round(usd, 6),
        "usd_per_item": None if usd is None else round(usd / max(1, total), 8),
        "s_per_item": round(wall / max(1, total), 2),
        "priced": usd is not None,
    }


def run_door(door, domain: str) -> Dict[str, Any]:
    t0, right, total, srcs = time.perf_counter(), 0, 0, {}
    for _name, output, crits in CASES:
        res = door.judge(output, [c for c, _ in crits], domain)
        for (_c, truth), v in zip(crits, res.get("verdicts") or []):
            total += 1
            # an `unknown` verdict is never scored as correct, even when the
            # truth happens to be False
            right += int(v.get("pass") is truth)
            srcs[v.get("source", "?")] = srcs.get(v.get("source", "?"), 0) + 1
    wall = round(time.perf_counter() - t0, 1)
    # An `unknown` verdict is scored WRONG above (it is not a pass, and pretending
    # it is would be the worst failure this thing can have). But a leg that
    # abstains on half the items and is right on the rest is not 29% accurate at
    # judging -- it has 50% coverage at 58% accuracy, and both numbers have to be
    # printed or the abstention reads as an error rate.
    answered = total - srcs.get("none", 0)
    return {
        "accuracy_pct": round(100.0 * right / max(1, total), 1),
        "answered": answered,
        "coverage_pct": round(100.0 * answered / max(1, total), 1),
        "accuracy_when_answered_pct": (
            round(100.0 * right / answered, 1) if answered else None
        ),
        "model_calls": srcs.get("llm", 0),
        "sources": srcs,
        "wall_s": wall,
        "items": total,
        "s_per_item": round(wall / max(1, total), 3),
        "usd_total": 0.0 if srcs.get("llm", 0) == 0 else None,
        "usd_per_item": 0.0 if srcs.get("llm", 0) == 0 else None,
    }


def teach_all(door, domain: str) -> int:
    n = 0
    for _name, output, crits in CASES:
        for c, truth in crits:
            door.teach(output, c, truth, domain)
            n += 1
    return n


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", help="live door (else in-process)")
    ap.add_argument("--token", default=os.environ.get("AITHER_WM_INTERNAL_TOKEN", ""))
    ap.add_argument("--brain", default=os.environ.get("AITHER_BRAIN_URL", "https://127.0.0.1:8150/v1"))
    ap.add_argument("--model", default=os.environ.get("AITHER_BRAIN_MODEL", "default"))
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument(
        "--baseline-model",
        help="a SECOND judge leg on a named model (e.g. deepseek-chat) -- the "
        "third-party baseline an LLM-judge setup actually pays for",
    )
    ap.add_argument("--prices", default=PRICES_DEFAULT, help="per-token price table")
    ap.add_argument(
        "--neural-model",
        default=os.environ.get("AITHER_DECIDE_NEURAL_MODEL"),
        help="add a 'door cold (neural)' leg served by this neural-rung.npz "
        "(tools/train_neural_rung.py writes it)",
    )
    ap.add_argument("--domain", default=f"decide.judge.bench{int(time.time())}")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    report: Dict[str, Any] = {"cases": len(CASES), "criteria_per_case": 3, "domain": a.domain}

    prices = load_prices(a.prices)
    report["prices_from"] = a.prices if prices else f"{a.prices} (UNREADABLE -- costs unpriced)"
    if not a.skip_llm:
        try:
            report["llm-judge"] = run_llm(LLMJudge(a.brain, a.model), prices)
        except Exception as e:  # noqa: BLE001
            print(f"COULD NOT JUDGE: llm judge unusable at {a.brain}: {type(e).__name__}: {e}")
            return 2
        if a.baseline_model:
            try:
                report["llm-judge baseline"] = run_llm(
                    LLMJudge(a.brain, a.baseline_model), prices
                )
            except Exception as e:  # noqa: BLE001
                report["llm-judge baseline"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        door = DoorLive(a.url, a.token) if a.url else DoorInProc(
            None if a.skip_llm else a.brain, a.model
        )
        report["door cold"] = run_door(door, a.domain)
        report["taught"] = teach_all(door, a.domain)
        report["door warm"] = run_door(door, a.domain)
    except Exception as e:  # noqa: BLE001
        print(f"COULD NOT JUDGE: door unusable: {type(e).__name__}: {e}")
        return 2

    # 'door cold (neural)': the SAME cold items, on a door whose neural rung is
    # loaded and whose store is empty -- no evidence, no brain. It is a real cold
    # test only because train_neural_rung.py forces every fork these CASES ask
    # about into its held-out split; without that this leg would be a replay.
    if a.neural_model:
        if not Path(a.neural_model).is_file():
            report["door cold (neural)"] = {"error": f"no model at {a.neural_model}"}
        else:
            try:
                nd = DoorInProc(None, a.model, neural_model=a.neural_model)
                report["door cold (neural)"] = run_door(nd, a.domain + ".neural")
                report["door cold (neural)"]["model_file"] = a.neural_model
            except Exception as e:  # noqa: BLE001 -- a missing leg is not a dead bench
                report["door cold (neural)"] = {"error": f"{type(e).__name__}: {e}"}

    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{len(CASES)} outputs x 3 criteria = {len(CASES) * 3} judgements\n")
        print(f"{'grader':20}{'accuracy':>10}{'answered':>14}{'calls':>7}{'wall':>9}"
              f"{'s/item':>9}{'$/item':>12}")
        for key in ("llm-judge", "llm-judge baseline", "door cold",
                    "door cold (neural)", "door warm"):
            r = report.get(key)
            if not r or "accuracy_pct" not in r:
                continue
            usd = r.get("usd_per_item")
            if usd is None:
                cell = "unpriced"
            elif usd == 0:
                cell = "$0"
            else:
                cell = ("$" + f"{usd:.8f}".rstrip("0").rstrip("."))
            label = key if not r.get("model") else f"{key} ({r['model']})"
            if r.get("answered") is None:
                ans = f"{r['items']}/{r['items']}"
            else:
                ans = f"{r['answered']}/{r['items']}"
                if r.get("accuracy_when_answered_pct") is not None and r["answered"] < r["items"]:
                    ans += f"@{r['accuracy_when_answered_pct']:.0f}%"
            print(
                f"{label[:20]:20}{r['accuracy_pct']:>9.1f}%{ans:>14}{r['model_calls']:>7}"
                f"{r['wall_s']:>8.1f}s{r.get('s_per_item', 0):>9.2f}{cell:>12}"
            )
        for key in ("llm-judge", "llm-judge baseline"):
            r = report.get(key) or {}
            if r.get("served_by"):
                flag = f"  MISROUTED: {r['unpriced_reason']}" if r.get("misrouted") else ""
                if r.get("degraded"):
                    flag += (f"  DEGRADED: {r.get('errors', 0)} failed calls, "
                             f"{r.get('unparsed', 0)} unparsed")
                print(f"{key}: served_by={r['served_by']}{flag}")
        if report.get("door warm"):
            print(f"\nwarm sources: {report['door warm']['sources']}")
        nr = report.get("door cold (neural)")
        if nr:
            if nr.get("error"):
                print(f"door cold (neural): NOT RUN -- {nr['error']}")
            else:
                print(f"cold-neural sources: {nr['sources']} (model {nr['model_file']}); "
                      "every fork here was held out of that model's training split")

    warm, llm = report.get("door warm"), report.get("llm-judge")
    if not warm:
        print("COULD NOT JUDGE: no warm result")
        return 2
    for key in ("llm-judge", "llm-judge baseline"):
        r = report.get(key) or {}
        if r.get("degraded"):
            print(
                f"COULD NOT JUDGE: {key} ({r.get('model')}) is DEGRADED -- "
                f"{r.get('errors', 0)} failed calls, {r.get('unparsed', 0)} unparsed of "
                f"{r.get('items', 0)}. Beating a broken baseline proves nothing; rerun it."
            )
            return 2
    if llm is None:
        # --skip-llm: there is no llm leg to beat, so "matches or beats" would be a
        # vacuous pass (the critic's blocker, 2026-09-21). The only claim a warm-only
        # run can make is RECALL: every taught item comes back right, with no model
        # call. Accuracy on NEW work is tools/judge_holdout_bench.py, not this.
        ok = warm["model_calls"] == 0 and warm["accuracy_pct"] >= 100.0
        print(
            "VERDICT:",
            "warm door recalls every taught item with no model calls "
            "(recall check only -- no llm leg ran; held-out accuracy is judge_holdout_bench)"
            if ok
            else f"warm door FAILED recall: {warm['accuracy_pct']}% with "
                 f"{warm['model_calls']} model calls",
        )
        return 0 if ok else 1
    ok = warm["model_calls"] == 0 and warm["accuracy_pct"] >= llm["accuracy_pct"]
    print(
        "\nVERDICT:",
        "the warm door matches or beats the llm judge with no model calls"
        if ok
        else "warm door did not clear the llm judge",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
