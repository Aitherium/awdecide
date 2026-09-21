# Changelog

## 0.3.0 -- 2026-09-21

- `Loop` (`awdecide/loop.py`): the rung a stateless decider cannot have. It wraps any
  `Ladder` with an **evidence** rung over the ledger -- a decision whose outcome was
  resolved is answered from that evidence on the next identical sighting, with no model
  call; an option resolved wrong is withheld from the ladder, so a reported mistake is
  not repeated. Probabilities are fitted to outcomes (a 60/40 coin reads 0.60).
- `ChatBackend`: any OpenAI-wire chat endpoint as a rung, for servers without logprobs.
  Its bare pick is not a probability, so the loop replaces it with the rung's measured
  hit rate at that key.
- `awdecide mcp`: a stdio MCP server (`decide`, `decide_outcome`, `decide_teach`,
  `decide_stats`) for Claude Code, Codex, Cursor. `awdecide serve`: the same over HTTP.
- `awdecide bench`: one imperfect brain called every time vs behind the loop --
  70.8% / 400 calls vs 97.0% / 52 calls at the default seed, Brier under the base rate.

## 0.2.0 -- 2026-09-20

- `DoorBackend` (`awdecide/door.py`): a learning decision door as a ladder rung.
  choice / score / bool map onto the door's choice / score / yesno; its answer comes back
  with the probability its own outcomes earned, `backend=door:<source>`, and `source=none`
  ABSTAINS instead of returning a placeholder option. In-process when the service tree is
  importable, otherwise HTTP (gateway `/v1` or the service base). `resolve()` posts the
  outcome back, so resolving in the ledger also teaches the door.
- A rung may now own its probability: `Ladder` takes a `decide(state, question) -> Decision`
  from a backend as-is, instead of renormalizing weights and destroying the calibration.
- `Ledger.ingest()` lands a decision made elsewhere under its own id, idempotently, and
  `reliability()` now breaks Brier out `by_backend`.
- `awdecide/bridge.py` + `awdecide ingest-door <ckpt_dir>`: the door's `decisions.jsonl`
  and per-fork transition journals into the ledger (journaled pairs, prequential replay,
  or `--generate` bench-generated pairs). Confidence-only rows from before 2026-09-20 are
  counted and skipped -- evidence strength is not P(right).
- Self-test arm 7 covers the rung and the bridge against a fake door; `tests/test_door_backend.py`
  pins parity between the contract and the door in-process.

## 0.1.0 -- 2026-09-18

- First release: the contract (choice / score / bool), the ladder (rules, callable, logprob), the Brier ledger, `awdecide --self-test`.
