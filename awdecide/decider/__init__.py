"""awdecide.decider -- the nerve: a small, fast head that decides react-now or think.

The Kev pattern (a few-billion-parameter model in front of a big one, deciding per
stimulus whether the big one is needed) at the size the door already runs: one
softmax head per decision TASK over a cached sentence embedding and/or a hashed bag
of the context, trained on typed rows every part of the stack already journals.

    python -m awdecide.decider.build_decider_dataset --library <dir> --out rows.jsonl
    python -m awdecide.decider.train_decider --rows rows.jsonl --out-dir <dir>
    python -m awdecide.decider.train_decider --self-test

Row shape (one per decision, typed):
    {context, kind, task, label, label_space, outcome, source, ts}
    kind  yesno | choice | route | reflex
    route labels are ``react`` or ``think:<brain>``

A head replaces what serves today only if its held-out Brier is LOWER and its
accuracy is not lower by more than one point (``heads.promotion``). numpy only.
"""

from __future__ import annotations

KINDS = ("yesno", "choice", "route", "reflex")
BRAINS = ("orchestrator", "bonsai2", "deepseek-pool", "gemma4-vision")
ROUTE_SPACE = ("react",) + tuple(f"think:{b}" for b in BRAINS)

__all__ = ["KINDS", "BRAINS", "ROUTE_SPACE"]
