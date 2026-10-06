# awdecide for agents

Read this if you are an agent (or a human) editing this package. Short on
purpose: the commands, the traps that cost a session, and where the rest lives.
Nothing here is read at runtime — it is for you.

## What this is

PyPI distribution **`awdecide`** (version in `pyproject.toml`), import package
`awdecide`, Python >= 3.10. One typed-decision contract — choice / score / bool
with a probability — over a ladder of backends you already run (rules, tiny
local models, an LLM's logprobs), **fail-closed**, with a Brier ledger that
resolves every decision against its outcome.

This repository is a **synced mirror** of the AitherOS monorepo (lane
`.github/workflows/sync-awdecide.yml`). Hand edits made here are overwritten on
the next sync — change the source and let the lane publish.

## Build, test, verify

```bash
python -m pytest tests -q        # the suite: 51 tests, green at v0.4.0
pip install -e .                 # editable install for developing against it
```

The suite was run from a source checkout with no prior install. The publish
lane (`publish-brick.yml`) additionally builds the wheel, installs it and
imports it — a tree that tests green can still ship a broken wheel.

## Rules that keep this useful

- **The vendored door copy is manifest-hashed, byte for byte.**
  `scripts/vendor_door.py --offline` must report zero diverged files
  (`tests/test_door_local.py` runs it). Two ways this goes wrong, both
  measured: editing the vendored copy instead of re-vendoring, and letting a
  Windows checkout CRLF it — its `.py` files were covered by the global
  `*.py` rule but the corpus/eval/pricing text files were not, so 11 files
  "diverged from manifest" on Windows only (fixed 2026-10-06 with a
  `text eol=lf` rule on the tools tree). If the check fails, look at the
  manifest before the code.
- **Fail closed, always.** `test_door_backend.py` + `test_door_local.py` pin
  the ladder's refusal path: when no backend can answer, the decision is a
  refusal, never a defaulted choice. A decision tool that guesses is worse
  than one that stops.
- **Every decision resolves against its outcome.** The Brier ledger is the
  point of the brick (`test_loop.py`) — a new decision kind ships with the
  path that settles it, or the calibration quietly becomes fiction.
- **The registry drives the public surface.** This repo's README header,
  `llms.txt` and `aither-manifest.json` are generated from the ecosystem
  registry (one yaml in the AitherOS monorepo) and rewritten on every sync.
  Change the registry; do not hand-edit the generated blocks.

## Read next

- `llms.txt` — the install/use card written for an agent to execute
- `README.md` — the human front door
- `docs/` — the generated docs site source
