"""awdecide -- Aither World Decide.

One typed-decision contract -- choice / score / bool with a probability -- over
a ladder of backends you already run (rules, any local model as a callable, an
OpenAI-wire model's logprobs), fail-closed (decided=False is an answer), with a
Brier ledger that resolves every decision against its outcome and reports
whether the probabilities carry information beyond the base rate.

    from awdecide import Question, Ladder, RulesBackend, Ledger

    q = {"category": Question.choice(["billing", "technical", "sales"]),
         "urgent": Question.bool(min_confidence=0.7)}
    answers = Ladder([RulesBackend([(r"refund|invoice", "billing")])]).decide(state, q)
    answers["category"].value, answers["category"].probability, answers["urgent"].decided

Stdlib only. Nothing here sends the state anywhere you did not name.
"""
from .backends import (
    Backend,
    CallableBackend,
    Ladder,
    LogprobBackend,
    RulesBackend,
    default_ladder,
    parse_question_spec,
)
from .contract import Decision, Question, from_probabilities, normalize, undecided
from .door import DoorBackend, decision_from_door, door_request
from .ledger import Ledger
from .loop import ChatBackend, Loop

__version__ = "0.4.0"
__all__ = [
    "Backend", "CallableBackend", "Ladder", "LogprobBackend", "RulesBackend",
    "default_ladder", "parse_question_spec", "Decision", "Question",
    "from_probabilities", "normalize", "undecided", "Ledger", "DoorBackend",
    "decision_from_door", "door_request", "ChatBackend", "Loop", "__version__",
]
