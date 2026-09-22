"""The vendored door: `pip install awdecide` runs the engine with no service tree.

Three things a stranger's machine has to be able to do, each proven here with
AITHER_WM_SVC_DIR pointed at nothing:

  * decide -> outcome -> decide again on the same fork: the second answer comes
    from `engine` (learned), not `none`;
  * judge -> teach -> judge: a taught criterion is answered from evidence;
  * compact a long tool output with recall of the lines that matter.

And two things the vendored copy must never carry: the shape of one deployment
(host names, ledger ids, the author's paths) and a drift from its declared sources
(scripts/vendor_door.py --offline).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
DOOR = PKG / "awdecide" / "door_local"


@pytest.fixture()
def fresh_door(tmp_path, monkeypatch):
    monkeypatch.setenv("AITHER_WM_SVC_DIR", str(tmp_path / "nowhere"))
    monkeypatch.setenv("AITHER_WM_CKPT_DIR", str(tmp_path / "ckpt"))
    from awdecide import door as door_mod

    code_domains, decide = door_mod.load_service(None, str(tmp_path / "ckpt"))
    return decide.Decider(code_domains.DomainEngines(), llm=None, embed_enabled=False)


def test_vendored_tree_is_present_and_declared():
    assert (DOOR / "decide.py").is_file() and (DOOR / "VENDORED.json").is_file()
    man = json.loads((DOOR / "VENDORED.json").read_text(encoding="utf-8"))
    assert man["files"]["decide.py"] and man["scrub_version"] >= 1
    assert man["cases_dropped_transcribed"] > 0, "the transcribed rows must NOT ship"


def test_vendored_copy_matches_its_manifest():
    r = subprocess.run([sys.executable, str(PKG / "scripts" / "vendor_door.py"), "--offline"],
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


# Built from parts so this test file itself carries none of the shapes it hunts
# (the sdist ships tests/, and the moat guard reads them too). No word boundaries:
# in a JSON row a name can follow a literal backslash-n, where \b does not fire.
_SHAPES = [
    (r"D-\d{3,5}\b", "ledger id"),
    (r"(NX|SEC|PQ|HYG|QS|EMC|ST|AWG|NAV|CSR|ACG|MCP|BW|ONB|AC|WT)\d{3}[A-Z]?\b", "rule id"),
    (r"[A-Za-z]:[\\/]Aither" + "OS-Fresh", "author's checkout"),
    ("aither" + "os-[a-z0-9-]+", "container name"),
    (r"aither-(orchestrator|code-embed)", "deployment alias"),
    ("arc-agi" + "-3", "author's data root"),
    ("g" + "arg[a-z0-9.-]*", "customer name"),
]


def test_vendored_copy_carries_no_deployment_shape():
    hits = []
    for p in sorted(DOOR.rglob("*")):
        if not p.is_file() or p.suffix not in {".py", ".md", ".jsonl", ".txt", ".yaml", ".json"}:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for rx, what in _SHAPES:
            for m in re.finditer(rx, text):
                hits.append(f"{p.relative_to(DOOR)}: {what} {m.group(0)!r}")
    assert not hits, "\n".join(hits[:20])


def test_decide_learns_from_outcome_without_a_service_tree(fresh_door):
    body = {"domain": "decide.test.retry", "kind": "yesno", "state": "exit code 1; 3 failed",
            "question": "retry the tests?", "options": ["yes", "no"]}
    cold = fresh_door.decide(body)
    assert cold["source"] in ("none", "prior"), cold
    # teach explicitly: at this state, "no" was right and "yes" was wrong
    for _ in range(3):
        fresh_door.outcome({"domain": body["domain"], "state": body["state"], "answer": "no",
                            "reward": 1.0})
        fresh_door.outcome({"domain": body["domain"], "state": body["state"], "answer": "yes",
                            "reward": -1.0})
    warm = fresh_door.decide(body)
    assert warm["source"] == "engine", warm
    assert warm["answer"] == "no"


def test_judge_teaches_and_recalls(fresh_door):
    import judge as judge_mod  # vendored, on sys.path via load_service

    j = judge_mod.Judge(fresh_door)
    out = "===== 40 passed in 1.2s =====\n"
    crit = "the test suite passed"
    j.teach(out, crit, True, domain="decide.judge.test")
    res = j.judge(out, [crit], domain="decide.judge.test")
    v = res["verdicts"][0]
    assert v["pass"] is True and v["source"] == "engine", v


def test_compact_keeps_the_lines_that_matter(fresh_door):
    import compact  # vendored

    lines = ["collected 300 items"] + [f"tests/test_x.py::test_{i} PASSED" for i in range(300)]
    lines += ["E   AssertionError: boom", "1 failed, 299 passed in 4.2s"]
    res = compact.compact("\n".join(lines), "pytest", decider=fresh_door)
    assert res["kept_lines"] < res["total_lines"] // 2
    assert "AssertionError: boom" in res["kept"] and "1 failed, 299 passed" in res["kept"]
