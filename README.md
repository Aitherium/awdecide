# awdecide — Aither World Decide

<!-- aither-header:start GENERATED from the ecosystem registry. Edits here are overwritten; change the registry instead. -->

[Source](https://github.com/Aitherium/awdecide)  ·  [The Aither World](https://aitherium.github.io/)

> **The Aither World** is an operating system for agents — a Linux you can hand to one, the runtimes it works in, and the tools it works with. [awnix](https://github.com/Aitherium/awnix) is the Linux underneath it; **awdecide** is one of its 67 bricks — each installs on its own, runs offline, and needs no account.
>
> **Start here:** Ask one typed question of one state and read back a decision, a probability, and decided=False when nothing earned it.

<!-- aither-header:end -->

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

<!-- aither-ecosystem:start GENERATED from the ecosystem registry. Edits here are overwritten; change the registry instead. -->

## The aw family

Standalone tools that share one idea: **replace something you would otherwise have to _trust_ with something you can _check_.**

Each installs on its own, works offline, and needs no account.

| | instead of trusting | you check |
|---|---|---|
| [awdk](https://github.com/Aitherium/awdk) | a framework's idea of how your agents should run | one loop you can read, pointed at a backend you already pay for |
| [awskills](https://github.com/Aitherium/awskills) | that an agent knows your procedure | the procedure written down, versioned, and loadable by any agent |
| [awpack](https://github.com/Aitherium/awpack) | that the pack you want shipped inside somebody's SDK, under whatever licence that SDK happens to carry | the pack as its own versioned artifact, with its own licence, that any agent runtime can install |
| [awm](https://github.com/Aitherium/awm) | that memory stayed in its lane | tenant:user:project scopes, so a write cannot cross a boundary |
| [awdesk](https://github.com/Aitherium/awdesk) | that the agent is somewhere behind a browser tab | a tray icon, a face on your desktop, and the decision card that pops when it needs you |
| [awnode](https://github.com/Aitherium/awnode) | a vendor's cloud with every prompt | a local gateway routing to backends you chose |
| [awgraph](https://github.com/Aitherium/awgraph) | that grep found everything | an AST + tree-sitter call graph an agent can traverse |
| [awgit](https://github.com/Aitherium/awgit) | that no one else is editing this file | a lease, refused at commit time if you do not hold it |
| [awdelphi](https://github.com/Aitherium/awdelphi) | one agent's confident take on a decision | the round trace, the anonymity, and who dissents |
| [awclassify](https://github.com/Aitherium/awclassify) | a filename, a folder, or whoever last touched it | doc_type, visibility, audience and topics, with the evidence lines that decided each |
| **awdecide** _(you are here)_ | a hosted classifier's probability that never learns whether it was right | the decision, its probability, and the calibration curve from your own resolved outcomes |
| [awtoll](https://github.com/Aitherium/awtoll) | that your tooling is saving you context | the measured token cost of each tool call, and what the alternative cost |
| [awseal](https://github.com/Aitherium/awseal) | that the artifact came from who you think | an Ed25519 seal — the key that verifies is not the key that forges |
| [awshare](https://github.com/Aitherium/awshare) | that the download is intact | content-addressed bundles, verified on fetch |
| [awsuite](https://github.com/Aitherium/awsuite) | that an agent holding your mailbox will not send on its own | every send, draft, upload and create returns a dry-run until confirm is true |
| [awnest](https://github.com/Aitherium/awnest) | that there is a person on the other end | a verdict with evidence, where "we could not tell" is not "yes" |
| [awrena](https://github.com/Aitherium/awrena) | a leaderboard someone can edit, and votes nobody counted | a scored duel with both answers kept, and a result bound to them |
| [awnboard](https://github.com/Aitherium/awnboard) | a share link anyone who sees it can use | an invitation addressed to one person, for one gate, revocable |
| [awnix](https://github.com/Aitherium/awnix) | that the box is what you left it as | an immutable image you built, with atomic rollback |
| [awrecover](https://github.com/Aitherium/awrecover) | that the restore worked | a restore that fully lands or does not land at all |
| [awstorage](https://github.com/Aitherium/awstorage) | a du you ran last month, and a peers file that says 3 TB free | an inventory snapshot per node with a diff since the last one, and each tree classified re-fetchable or not |
| [awrelay](https://github.com/Aitherium/awrelay) | a SaaS in the middle of your agents | findings, alerts and coordination over your own transport |
| [awask](https://github.com/Aitherium/awask) | that anyone read the paragraph where you asked | the ask itself, with a button that steers the session that raised it |
| [awmail](https://github.com/Aitherium/awmail) | a mailbox somebody else can read | mail your agents send and receive over your own server |
| [awswarm](https://github.com/Aitherium/awswarm) | that a model either fits your GPU or it doesn't run at all | a placement plan and an acquisition-probability estimate before you spend on a run |
| [awfind](https://github.com/Aitherium/awfind) | one vendor's idea of the web | results from whichever providers you configured |
| [awbrowse](https://github.com/Aitherium/awbrowse) | that the page said what you were told | the render, the DOM and the requests it made |
| [awvoice](https://github.com/Aitherium/awvoice) | that a cloud vendor may hold your audio | a transcript and a wav from a service you host |
| [awvision](https://github.com/Aitherium/awvision) | a filename and a caption somebody wrote | what a model actually reports about the pixels |
| [awscreen](https://github.com/Aitherium/awscreen) | a selector that was true when the page was written | the elements actually rendered, by what they look like |
| [awbeads](https://github.com/Aitherium/awbeads) | that a layout your users built survives the next deploy | the arrangement as data you can read back, diff, and hand to another surface |
| [awbonsai](https://github.com/Aitherium/awbonsai) | that inference always means a request left the machine | a WebGPU model answering on the tab's own GPU, with a consent record logged before it ever loaded |
| [gawbbonet](https://github.com/Aitherium/gawbbonet) | the model to keep a 300-message campaign coherent by itself | campaign facts recalled from scoped memory you can list and edit |
| [aitherkvcache](https://github.com/Aitherium/aitherkvcache) | a vendor's quantisation defaults | sub-byte KV cache kernels you can benchmark yourself |
| [awrtifact](https://github.com/Aitherium/awrtifact) | a hand-rolled split script and a hand-edited worker manifest | byte-verified parts in a release, served with Range + CORS, sizes asserted by a live gate |
| [AitherZero](https://github.com/Aitherium/AitherZero) | a pile of scripts nobody has numbered | numbered, discoverable automation with declarative playbooks |
| [AitherConnect](https://github.com/Aitherium/AitherConnect) | what a page tells your browser to do | a federated search and desktop bridge you host |
| [awreason](https://github.com/Aitherium/awreason) | a confident paragraph | the phases it went through, and every tool call it made to get there |
| [awrecurse](https://github.com/Aitherium/awrecurse) | that everything you pasted in was actually read | which slices it opened, and what it concluded from each |
| [awprism](https://github.com/Aitherium/awprism) | the first explanation that fits | the ranked alternatives, and the observation that separates them |
| [awrepl](https://github.com/Aitherium/awrepl) | what the agent believes the value is | the value, printed from the live session |
| [awreport](https://github.com/Aitherium/awreport) | that the report you pasted carried no token in it | a redacted report, and the duplicate it merged into instead of filing twice |
| [awresearch](https://github.com/Aitherium/awresearch) | a summary of pages nobody opened | every claim against the source it came from |
| [awfocus](https://github.com/Aitherium/awfocus) | twelve terminal tabs and a bad memory | one command that names every session, finds any transcript, and opens or steers the one you want |
| [awgym](https://github.com/Aitherium/awgym) | that a world model learned anything from the games it saw | transitions captured from real play, fed back, and the retrodiction score falling on grids it never saw |
| [awpredict](https://github.com/Aitherium/awpredict) | a model because it trained without erroring | its prediction against a self-updating lookup, on the rows that are actually novel |
| [awevolve](https://github.com/Aitherium/awevolve) | that your optimisation loop is finding anything | every version it kept, the score that version earned, and the edit that produced it |
| [awsh](https://github.com/Aitherium/awsh) | that you already know the name of the command | what it decided your line meant, before it acts on it |
| [awmine](https://github.com/Aitherium/awmine) | that a session's lesson survived the session | a row per outcome, a candidate per lesson, and the transcript line each one came from |
| [awrise](https://github.com/Aitherium/awrise) | that a scheduled agent ran at all, and ran exactly once | a durable record of every wake -- fired, skipped, overlapped or timed out -- each with its reason |
| [awkno](https://github.com/Aitherium/awkno) | that the docs site is up, or that you remember the family | the whole ecosystem in your terminal, with no network at all |
| [awwall](https://github.com/Aitherium/awwall) | that a service only talks to the hosts you think it talks to | an explicit egress allowlist, where a denial names the rule that denied it |
| [awembed](https://github.com/Aitherium/awembed) | a general-purpose embedder that has never seen your code | a held-out split of whole directories, scored teacher vs student vs int8 |
| [awtax](https://github.com/Aitherium/awtax) | a closed tax app's sealed file you can never read again | a plain, provider-neutral schema of every figure, with the page it came from |
| [awsettings](https://github.com/Aitherium/awsettings) | that you will remember to re-approve the same thing on every box you work from | one profile, unioned rather than overwritten, with the credentials left behind |
| [awavatar](https://github.com/Aitherium/awavatar) | a cloud 3D vendor's opaque task id | a manifest with a sha256, a licence and a rig-audit verdict per file |

[**awnix**](https://github.com/Aitherium/awnix) is the ground floor — A Linux you can hand to an agent — immutable base, capabilities included.

## The Aitherium ecosystem

Every repository here is public. Each publishes an `aither-manifest.json` beside its page, so any surface can read every sibling's — the network is browsable from any node in it.

| repo | what it is | pages |
|---|---|---|
| [awdk](https://github.com/Aitherium/awdk) | Build AI agent fleets — 3 lines, any backend, local or cloud | [docs](https://aitherium.github.io/awdk/) |
| [awskills](https://github.com/Aitherium/awskills) | Portable agent skills — self-contained procedures an agent loads on demand | [docs](https://aitherium.github.io/awskills/) |
| [awpack](https://github.com/Aitherium/awpack) | First-party agent packs — the ones we build, versioned and installable on their own | [docs](https://aitherium.github.io/awpack/) |
| [awm](https://github.com/Aitherium/awm) | A portable, scoped agent memory | [docs](https://aitherium.github.io/awm/) |
| [awdesk](https://github.com/Aitherium/awdesk) | Aither World Desk -- the desktop body of AitherOS Online: tray, avatars, decision cards, the Living Desktop as an overlay | [docs](https://aitherium.github.io/awdesk/) |
| [awnode](https://github.com/Aitherium/awnode) | A lightweight local gateway — bridges your apps to the AI backends you chose | [docs](https://aitherium.github.io/awnode/) |
| [awrun](https://github.com/Aitherium/awrun) | A priority-aware queue and dispatcher for agentic runs and ad-hoc CI builds. It also judges whether the runner pool is big enough for the queue it is draining, and can ask a host to grow it -- reserving capacity is zero-sum, so a saturated pool needs more of it, not a different share of it | [docs](https://aitherium.github.io/awrun/) |
| [awgraph](https://github.com/Aitherium/awgraph) | A semantic code graph for agents — AST + tree-sitter, call graphs | [docs](https://aitherium.github.io/awgraph/) |
| [awgit](https://github.com/Aitherium/awgit) | Semantic version control on top of git — edit-ops and leases | [docs](https://aitherium.github.io/awgit/) |
| [awdelphi](https://github.com/Aitherium/awdelphi) | Anonymous multi-round expert panels — a converged answer with a trace | [docs](https://aitherium.github.io/awdelphi/) |
| [awclassify](https://github.com/Aitherium/awclassify) | Classify any document -- what it is, who may read it, who it is for, what it is about | — |
| **awdecide** _(you are here)_ | One typed-decision contract -- choice / score / bool with a probability -- over a ladder of backends you already run (rules, tiny local models, an LLM's logprobs), fail-closed, with a Brier ledger that resolves every decision against its outcome | — |
| [awtoll](https://github.com/Aitherium/awtoll) | What every tool call costs you in context, measured from your own transcripts | [docs](https://aitherium.github.io/awtoll/) |
| [awseal](https://github.com/Aitherium/awseal) | Sign an artifact so a stranger can verify it | [docs](https://aitherium.github.io/awseal/) |
| [awshare](https://github.com/Aitherium/awshare) | Publish an artifact and fetch it back verified | [docs](https://aitherium.github.io/awshare/) |
| [awsuite](https://github.com/Aitherium/awsuite) | Your Google Workspace as agent tools, and no write happens without a yes | — |
| [awdit](https://github.com/Aitherium/awdit) | An append-only audit trail whose gaps are DETECTABLE | [docs](https://aitherium.github.io/awdit/) |
| [awbac](https://github.com/Aitherium/awbac) | Role-based access control that fails closed and explains itself | [docs](https://aitherium.github.io/awbac/) |
| [awiam](https://github.com/Aitherium/awiam) | Who is this caller? A directory and session store that fails honestly | [docs](https://aitherium.github.io/awiam/) |
| [awtunnel](https://github.com/Aitherium/awtunnel) | Reach a service that has no public address | [docs](https://aitherium.github.io/awtunnel/) |
| [awnest](https://github.com/Aitherium/awnest) | Prove there is a human before you let them into the nest | [docs](https://aitherium.github.io/awnest/) |
| [awrena](https://github.com/Aitherium/awrena) | Put two agents head to head and get a verdict you can check | [docs](https://aitherium.github.io/awrena/) |
| [awnboard](https://github.com/Aitherium/awnboard) | A front gate you can put in front of anything, and hand someone the key to | [docs](https://aitherium.github.io/awnboard/) |
| [awnix](https://github.com/Aitherium/awnix) | A Linux you can hand to an agent — immutable base, capabilities included | [docs](https://aitherium.github.io/awnix/) |
| [awrecover](https://github.com/Aitherium/awrecover) | Labelled snapshots with an all-or-nothing restore | [docs](https://aitherium.github.io/awrecover/) |
| [awstorage](https://github.com/Aitherium/awstorage) | Every drive on every node, indexed, classified and diffed -- so you can see what you own before you delete it | [docs](https://aitherium.github.io/awstorage/) |
| [awrelay](https://github.com/Aitherium/awrelay) | Portable agent messaging — findings, alerts, coordination | [docs](https://aitherium.github.io/awrelay/) |
| [awask](https://github.com/Aitherium/awask) | Your agent asks you a question — and acts on your answer | [docs](https://aitherium.github.io/awask/) |
| [awmail](https://github.com/Aitherium/awmail) | Give an agent an email address — send, and actually receive | [docs](https://aitherium.github.io/awmail/) |
| [awnet](https://github.com/Aitherium/awnet) | The agentic web — agents host a mesh, and agents join one | [docs](https://aitherium.github.io/awnet/) |
| [awswarm](https://github.com/Aitherium/awswarm) | Run one model too big for any single GPU across a pool of small ones | — |
| [awfind](https://github.com/Aitherium/awfind) | A portable search client — query, results, ranking | [docs](https://aitherium.github.io/awfind/) |
| [awbrowse](https://github.com/Aitherium/awbrowse) | A portable browser client — navigate, console, network, DOM, screenshot | [docs](https://aitherium.github.io/awbrowse/) |
| [awvoice](https://github.com/Aitherium/awvoice) | Hear and speak — transcribe audio, synthesize a voice | [docs](https://aitherium.github.io/awvoice/) |
| [awvision](https://github.com/Aitherium/awvision) | See an image — describe it, ask it a question, compare two | [docs](https://aitherium.github.io/awvision/) |
| [awscreen](https://github.com/Aitherium/awscreen) | See this machine — what is on screen, and where to click it | [docs](https://aitherium.github.io/awscreen/) |
| [awkit](https://github.com/Aitherium/awkit) | Render an agent panel from a tool result — one component, any React app | — |
| [awbeads](https://github.com/Aitherium/awbeads) | A spatial canvas for a page — arrange things, connect them, and keep the arrangement | — |
| [awbonsai](https://github.com/Aitherium/awbonsai) | Run a real model in the visitor's own browser — no server round trip, no upload | — |
| [awknowledge](https://github.com/Aitherium/awknowledge) | How to run a coding agent so the result survives — the laws, with evidence | [docs](https://aitherium.github.io/awknowledge/) |
| [awbrain](https://github.com/Aitherium/awbrain) | Your history as a wiki of linked markdown — claims pinned to the evidence | — |
| [gawbbonet](https://github.com/Aitherium/gawbbonet) | GobboNet campaigns with a real agent brain — scoped memory, graph recall | [docs](https://aitherium.github.io/gawbbonet/) |
| [aitherkvcache](https://github.com/Aitherium/aitherkvcache) | Near-optimal KV cache quantization for LLM inference — sub-byte compression | [docs](https://aitherium.github.io/aitherkvcache/) |
| [awrtifact](https://github.com/Aitherium/awrtifact) | Deliberately chunk artifacts into GitHub release assets — the productized aitherkvcache mirror lane | [docs](https://aitherium.github.io/awrtifact/) |
| [AitherZero](https://github.com/Aitherium/AitherZero) | PowerShell 7+ automation framework — numbered, self-describing scripts | [docs](https://aitherium.github.io/AitherZero/) |
| [AitherConnect](https://github.com/Aitherium/AitherConnect) | Browser extension — federated AI search, page context, and the Living OS overlay | [docs](https://aitherium.github.io/AitherConnect/) |
| [awreason](https://github.com/Aitherium/awreason) | A portable reasoning client — sessions, phases, thoughts, and the chain that produced the answer | [docs](https://aitherium.github.io/awreason/) |
| [awrecurse](https://github.com/Aitherium/awrecurse) | Answer a question over a context far larger than the window — recursively, with the trace kept | [docs](https://aitherium.github.io/awrecurse/) |
| [awprism](https://github.com/Aitherium/awprism) | Turn a failure into ranked hypotheses — and say what would confirm each one | [docs](https://aitherium.github.io/awprism/) |
| [awrepl](https://github.com/Aitherium/awrepl) | A REPL an agent can actually use — state that survives between turns | [docs](https://aitherium.github.io/awrepl/) |
| [awreport](https://github.com/Aitherium/awreport) | File a bug report that has already scrubbed your secrets and collapsed the duplicate | — |
| [awresearch](https://github.com/Aitherium/awresearch) | Ask a research question, get a cited report you can check | [docs](https://aitherium.github.io/awresearch/) |
| [awfocus](https://github.com/Aitherium/awfocus) | See, search and steer every Claude session from one command | [docs](https://aitherium.github.io/awfocus/) |
| [awgym](https://github.com/Aitherium/awgym) | An ARC training gym — a game a world model can watch, and six roles that play through it | [docs](https://aitherium.github.io/awgym/) |
| [awpredict](https://github.com/Aitherium/awpredict) | Predict what your environment does next, and how surprised you were | [docs](https://aitherium.github.io/awpredict/) |
| [awevolve](https://github.com/Aitherium/awevolve) | Point an agent at a file and a command that scores it, and let it improve | — |
| [awsh](https://github.com/Aitherium/awsh) | Your terminal answers you -- type a question where a command would go | [docs](https://aitherium.github.io/awsh/) |
| [awmine](https://github.com/Aitherium/awmine) | Mine what your agents did -- outcomes, lessons and procedures out of the transcripts they left behind | — |
| [awrise](https://github.com/Aitherium/awrise) | Wake an agent on a schedule, let it do one thing, and put it back to sleep | [docs](https://aitherium.github.io/awrise/) |
| [awkno](https://github.com/Aitherium/awkno) | The man page for the Aither World — every brick, stack and law, offline | [docs](https://aitherium.github.io/awkno/) |
| [awwall](https://github.com/Aitherium/awwall) | Say what a workload may reach, and watch everything else fail closed | [docs](https://aitherium.github.io/awwall/) |
| [awrouter](https://github.com/Aitherium/awrouter) | OpenRouter for your own fleet: pick a model backend by cost/latency/ capability, fail over, fit the context window, stream. Standalone, OpenAI-compatible, no Aither-specifics required to be valuable | — |
| [awembed](https://github.com/Aitherium/awembed) | Train an embedding model that knows your corpus, and prove it beats the big one | [docs](https://aitherium.github.io/awembed/) |
| [awtax](https://github.com/Aitherium/awtax) | Turn any tax PDF -- returns, W-2, 1099, statements, even scans -- into structured data you can check | [docs](https://aitherium.github.io/awtax/) |
| [awflow](https://github.com/Aitherium/awflow) | A deterministic workflow runtime — chain agent calls with journal replay and budget control | [docs](https://aitherium.github.io/awflow/) |
| [awsettings](https://github.com/Aitherium/awsettings) | Your agent's permissions and config, following you to the next machine | [docs](https://aitherium.github.io/awsettings/) |
| [awavatar](https://github.com/Aitherium/awavatar) | One character spec in, a rigged, animated, multi-style avatar pack out | [docs](https://aitherium.github.io/awavatar/) |

**Built on** [llama.cpp](https://github.com/ggml-org/llama.cpp) · [vLLM](https://github.com/vllm-project/vllm) · [ComfyUI](https://github.com/comfyanonymous/ComfyUI) · [CentOS Stream](https://www.centos.org/centos-stream/) · [Podman](https://github.com/containers/podman) · [Docker](https://github.com/moby/moby) · [LanceDB](https://github.com/lancedb/lancedb) · [WireGuard](https://www.wireguard.com/) · [FFmpeg](https://ffmpeg.org/) · [Blender + Rigify](https://www.blender.org/) · [headroom](https://github.com/headroomlabs-ai/headroom) · [SANA](https://github.com/NVlabs/Sana) · [Hunyuan3D](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) · [repowise](https://github.com/repowise-dev/repowise) · [Playwright](https://github.com/microsoft/playwright) · [Chromium](https://www.chromium.org/) · [Next.js](https://github.com/vercel/next.js) · [React](https://github.com/facebook/react).

<div id="aither-constellation" data-self="awdecide"></div>
<script src="aither-constellation.js"></script>

<!-- aither-ecosystem:end -->
