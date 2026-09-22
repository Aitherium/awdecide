"""world_model — AitherOS canonical world-model package (JEPA + MLP engines).

One package, two engines, N environment adapters:

  * ``world_model.core.lewm.LeWorldModel`` — the LeWM-style JEPA
    (two-term loss: next-latent MSE + SIGReg; CEM planner). This file is the
    CANONICAL copy of the live ARC world-model service's model
    (D:\\awdecide\\arc-world-model-svc\\lewm.py); content drift between the
    two is a failure of AitherOS/dev/tools/check_wm_package_parity.py.
    KNOWN DIVERGENCE: the in-solver fork at
    D:\\awdecide\\ARC-AGI-3-Agents\\agents\\lewm.py additionally carries a
    value head (_ValueHead/_FsAdapter/value()/train_value_step) not yet
    merged here — that merge is its own gated slice of the world-model
    program, not silent drift.
  * ``world_model.core.mlp.MLPWorldModel`` — the embedding-MLP transition
    model extracted from lib/cognitive/LearnedWorldModel.py
    (tabular → hybrid → neural).

Contracts live in ``world_model.contracts`` (WorldModel, EnvironmentAdapter).
Torch and numpy are OPTIONAL at import time — engines degrade loudly
(ok == False), never raise into a caller.
"""

from world_model.contracts import EnvironmentAdapter, WorldModel, conforms

__version__ = "0.1.0"

__all__ = ["EnvironmentAdapter", "WorldModel", "conforms", "__version__"]
