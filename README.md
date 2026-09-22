# awdecide — Aither World Decide

**One typed-decision contract — choice / score / bool with a probability — over the
backends you already run, fail-closed, with a Brier ledger that resolves every
decision against its outcome.**

```bash
pip install -e AitherOS/packages/awdecide      # monorepo; no public mirror yet
awdecide --self-test
```

## The problem it exists for

Software makes the same bounded decision millions of times a day and asks a text
model each time, then parses prose. Hosted "decision models" fix the shape — a typed
question in, a typed answer with a probability out — but they send your program state
off-box on every call, emit a probability that is never resolved against what
happened, and cannot say what happens next if you act on it.

A decision is a function. The loop it sits in is the product. `awdecide` is the
function, written so the loop can be closed on your own machine:

- **the contract** — `choice` (one of N names), `score` (one of N *ordered* levels),
  `bool`; every answer carries `probability`, `confidence`, `backend`, `decided`.
- **the ladder** — rungs asked in order, first answer above `min_confidence` wins:
  `RulesBackend` (a matched rule answers at 1.0; no match abstains, never guesses),
  `CallableBackend` (any `fn(state, question) -> {option: weight}` — a nanoGPT, an
  sklearn model, a classify surface), `LogprobBackend` (an OpenAI-wire server with
  `logprobs`: the mass over the option labels *is* the distribution; no prose parsed).
- **fail-closed** — `decided=False` is a first-class answer. `value` is `None`, the
  reasons say which rung abstained and why, and the strongest sub-threshold evidence is
  kept in `probabilities` so a reviewer can see what was *not* acted on.
- **the ledger** — SQLite. `record` a decided answer, `resolve` it against the outcome,
  `reliability()` reports overall Brier, the climatology (base-rate) Brier it must beat,
  and the per-bucket table (mean confidence vs observed frequency). Run `awdecide
  reliability` against that database on a schedule and it tells you the moment the
  probabilities stop carrying information.

## Use

```python
from awdecide import Question, Ladder, RulesBackend, LogprobBackend, Ledger

questions = {
    "category": Question.choice(["billing", "technical", "sales"]),
    "urgency":  Question.score(["low", "medium", "high"]),
    "escalate": Question.bool(min_confidence=0.7),
}
ladder = Ladder([
    RulesBackend([(r"refund|invoice", "billing")]),
    LogprobBackend("http://127.0.0.1:8150", "orchestrator"),   # your own MicroScheduler
])
answers = ladder.decide(state_text, questions)
answers["category"].value            # "billing"
answers["escalate"].decided          # False if no rung cleared 0.7 -- act on that in code

led = Ledger()                        # ~/.aither/awdecide.db or $AWDECIDE_DB
did = led.record("escalate", state_text, answers["escalate"])
...
led.resolve(did, correct=True)       # later, when you know
led.reliability()                    # brier, climatology, beats_base_rate, buckets
```

```bash
awdecide ask --state "Customer emailed twice about a failed refund" \
    category:choice=billing,technical,sales urgent:bool@0.7 \
    --rule 'refund|invoice=billing' --record
# exit 0 = every question decided · 3 = at least one decided=False (the JSON says why)
awdecide resolve <id> --correct
awdecide reliability
```

## The loop: a decision you resolved is never paid for twice

`Ladder` answers a question. `Loop` remembers how the answer turned out. Ask, act,
resolve -- the next identical decision is answered from that evidence with no model
call, and an answer that was resolved wrong is never given again.

Add it to the agent harness you already use:

```bash
claude mcp add awdecide -- uvx awdecide mcp          # Claude Code
```

```toml
# Codex: ~/.codex/config.toml
[mcp_servers.awdecide]
command = "uvx"
args = ["awdecide", "mcp"]
```

Then one paragraph in your `CLAUDE.md` / `AGENTS.md`:

> Before a bounded decision you make repeatedly here (which command, which branch, retry
> or stop), call `decide` with a STABLE `state` string. Act on the answer, then ALWAYS call
> `decide_outcome`. If it returns `decided=false`, decide yourself and `decide_teach` it.

The harness's own model is the brain on the first sighting; the ledger is the memory on
every one after. To give the loop its own brain, point it at any OpenAI-wire endpoint:
`AWDECIDE_LLM_URL`, `AWDECIDE_LLM_MODEL`, optional `AWDECIDE_LLM_KEY`.

```python
from awdecide import Loop, Ladder, Ledger, Question, ChatBackend

loop = Loop(Ladder([ChatBackend("http://127.0.0.1:11434", "qwen3:8b")]), Ledger())
d = loop.decide("test-runner", "lang:py,changed:tests",
                Question.choice(["pytest -x", "pytest -n8", "tox"]))
loop.resolve(d.id, correct=run(d.value))     # d.backend is "evidence" next time
```

Measure it: `awdecide bench` runs one deliberately imperfect brain (right 70% of the
time) over 40 situations seen 10 times each, two ways.

| | accuracy | model calls |
|---|---|---|
| call the brain every time | 70.8% | 400 |
| **behind the loop** | **97.0%** | **52** |

Same brain. More accurate because a wrong answer is resolved and retired; cheaper
because a right one is never asked for again. The run also reports Brier against the
base rate, so the probabilities are graded too. Change `--seed` and rerun it.

## The learning backend: the decision door

A rung may be a door that LEARNS. `DoorBackend` asks a world-model decision door
(engine -> neighbor -> neural -> llm -> prior), which answers with the probability its
own posted outcomes earned; `resolve()` sends the outcome back, so the next time that
state -- or one like it -- arrives, the answer comes from evidence instead of a model.

```python
from awdecide import DoorBackend, Ladder, Ledger, Question

state = "kind:code,len:short"
door = DoorBackend(url="http://127.0.0.1:8299/v1", token=TOKEN, fork="router")
d = Ladder([door]).decide_one(state, Question.choice(["fast-local", "reasoner"]))
d.backend          # 'door:engine' -- which rung of the door answered
led = Ledger()
door.resolve(led.record("route", state, d), correct=True, ledger=led)   # teaches both
```

The door's own journals become ledger rows, idempotently by decision id:

```bash
awdecide ingest-door /path/to/ckpt-dir          # journaled pairs + prequential replay
awdecide ingest-door --generate                 # bench-generated pairs, labelled bench/
awdecide reliability                            # brier vs base rate, broken out by backend
```

A decision the door made with no probability, or with `source=none`, is not a claim and
never becomes a row.

## The door in-process: `pip install awdecide` is the whole install

Since 0.4.0 the package carries the door's engine (`awdecide.door_local`): the same
`decide.py` / `judge.py` / `compact.py` that run behind the HTTP door, the tabular
domain engine they answer from, the bench scripts, the compaction corpus and the
labelled judge dataset. `DoorBackend()` with no `url` and no service tree on the
machine uses it automatically, journaling under `~/.awdecide/door`. Nothing to run,
no fleet, stdlib only (`pip install awdecide[door]` adds numpy for the neural rung).

```python
from awdecide import DoorBackend, Ladder, Question
door = DoorBackend(fork="router")           # transport == "inprocess" on a bare machine
d = Ladder([door]).decide_one("kind:code,len:short", Question.choice(["fast", "slow"]))
door.resolve(d.id, correct=True)            # the next identical state is answered from evidence
```

Every number in the write-up reproduces from the installed package, each bench with
its own verdict in its exit code (0 holds, 1 does not, 2 could not run):

```bash
awdecide door-bench judge-holdout     # judge on rows it was NOT taught, 5 seeds, neural rung
awdecide door-bench compact           # line-shape compaction of real tool output, recall gated
awdecide door-bench cache-cost        # prompt-cache dollars: append-time vs history-edit
awdecide door-bench calibration       # a 60/40 coin reads 0.60
awdecide door-bench judge             # the 24-item judge bench, recall check (no brain needed)
```

Measured 2026-09-21 from the installed package on the SHIPPED dataset -- 408 rows /
1,567 criteria, all generated by running real commands on a scratch tree (the 55 rows
transcribed from one fleet's logs are not carried, so these differ from the write-up's
463-row table by exactly that): held out by row, 5 seeds, the neural rung fitted on the
training split only:

| leg | coverage | accuracy when answered | overall (unknown = wrong) |
|---|---|---|---|
| door, engine + neural | 94.0% | **98.4%** (97.2-99.3) | 92.6% |
| lookup floor (majority label per criterion + feature key) | 94.0% | 97.3% (94.9-98.7) | 91.4% |

Compaction on the shipped corpus: rules alone 53% of estimated tokens saved, the
taught door 74%, at 100% recall of the hand-labelled must-keep lines. Cache cost over
a modelled 40-turn session at published list prices: no compaction 1.21, history-edit
1.20 (66% more cache writes, 47 reasoning blocks invalidated), append-time 0.69 with
rules and 0.50 with the door, in USD.

`awdecide/door_local/` is written only by `scripts/vendor_door.py` (`--sync`), which
also checks it has not drifted from its sources (`VENDORED.json` records the source
commits, every file's hash, and the reasons for the small set of rewrites that remove
one deployment's host names and paths). Edit the source, never the copy.

## What it is not

It is not a model. It ships no weights and calls nothing you did not name; a
`LogprobBackend` with no URL is a refusal, not a default. It does not explain — a
probability is what you get, and the ledger is where you find out whether it meant
anything. Predicting what happens *after* the decision is `awpredict`'s contract;
classifying documents at the door is `awclassify`'s; proving a page did the right thing
is `awprove`'s.

## Self-test

`awdecide --self-test` runs seven arms and exits 0 only when all pass: rules answer,
an empty ladder fails closed, `min_confidence` demotes a weak answer and keeps the
evidence, the logprob rung turns a real logprobs payload (served in-process) into a
distribution over the option labels only, the ledger's calibrated set beats the base
rate and its overconfident set does not, the CLI grammar rejects a malformed spec, and the
door rung maps every primitive, keeps the door's calibrated probability unrenormalized,
abstains when the door has nothing, and ingests a door journal exactly once.

Apache-2.0. Python 3.10+. Standard library only.
## Sources and prediction

`awdecide.sources` adds where a decision's state comes from and a rung that tries to
predict the answer. `ReplState(session)` turns a live `awrepl` session into a STABLE
state descriptor -- variable names, types and bucketed sizes (`rows:list:1k-9k`), never
a repr and never a value, so the descriptor is safe to hash, log and resolve in the
ledger. `PredictBackend(env)` is a ladder rung that asks a value oracle what each option
is worth and ABSTAINS unless the top two are further apart than `margin`; the oracle is
anything exposing `value(state, option)`, including `AwpredictValueEnv`, which wraps an
`awpredict` engine's reward or value head. **The shipped default
(`default_predict_backend()`) is the self-updating last-outcome lookup, not the learned
model, and that is a measurement rather than a preference.**
A bench (`tool_outcome_predict_bench`) asked whether anything can predict
that the next run of a command shape will pass, well enough to skip the call, over
323,644 real tool outcomes (95.0% pass) with a temporal 80/20 split, scored on the
UNSEEN bucket only -- novel command shapes -- because a self-updating dictionary already
owns the seen rows and an aggregate is therefore structurally unable to move. On 37,668
UNSEEN rows the lookup, the online majority and "always run it" all score 0.9571, and
coarsening the key to the command family makes it worse (0.9291); on a 12,000-row window
with the learned arms in (1,785 UNSEEN) `awpredict`'s token-hash arm ties the base rate
exactly at 0.9434 and its whole-string arm loses to it at 0.9412 -- so the bench exits 1,
and exit 1 IS the finding: on a novel shape there is nothing in the history to learn
from, and every arm collapses onto the base rate. The number that decides the design is
not accuracy but skip-precision: 0.9434 on UNSEEN means roughly one in eighteen skipped
calls would really have failed, so the rung is wired to abstain by default and to answer
only where it has seen the option before. Re-run it when an engine with a real value head
is a candidate; the arm ships the day it beats the dictionary, not before.
