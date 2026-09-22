"""awdecide.door_local -- the decision door in-process, no server.

Vendored by scripts/vendor_door.py from the door's service tree (see VENDORED.json
for the exact sources, hashes and the declared rewrites). Do not edit files here;
edit the source and re-run `--sync`.

The vendored modules import each other by their flat names (`import decide`,
`import judge`, `import code_domains`), exactly as they do in the service, so this
package puts its own directory first on `sys.path` when imported. Those names are
therefore taken while it is loaded -- if your project has a top-level `decide`
or `judge` module, import this before it or use `awdecide.local` only.

    from awdecide.door_local import decider, judge_for
    d = decider()                     # journals under $AITHER_WM_CKPT_DIR or ~/.awdecide/door
    j = judge_for(d)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
os.environ.setdefault("AITHER_WM_CKPT_DIR", str(Path.home() / ".awdecide" / "door"))
Path(os.environ["AITHER_WM_CKPT_DIR"]).mkdir(parents=True, exist_ok=True)

TOOLS = HERE / "tools"
PRICES = TOOLS / "model_token_prices.yaml"


def decider(llm=None, **kw):
    """A Decider over the vendored engines. `llm` is an optional callable
    prompt -> {"answer", "confidence"}; None means no model rung (cold forks read
    `source=none` and are never guessed at)."""
    import code_domains  # type: ignore
    import decide  # type: ignore

    if not code_domains._MLP_OK:
        raise RuntimeError("the vendored world_model engine failed to import")
    kw.setdefault("embed_enabled", False)
    return decide.Decider(code_domains.DomainEngines(), llm=llm, **kw)


def judge_for(d):
    import judge  # type: ignore

    return judge.Judge(d)
