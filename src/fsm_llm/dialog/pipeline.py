from __future__ import annotations

"""
fsm_llm.dialog.pipeline — back-compat shim over fsm_llm.dialog.turn.

The implementation moved to ``fsm_llm.dialog.turn`` per R13 (plan
plan_2026-04-27_32652286 step 13). The module rename reflects the
single-turn-of-dialog body's role under the post-R10 oracle-collapsed
architecture (pipeline-of-callbacks → single Leaf-emitting term).

# DECISION D-PIVOT-1-R13 — shim deprecation:
# - 0.5.0: emit DeprecationWarning at import time (THIS PR).
# - 0.6.0: remove the shim.
# Identity contract: import fsm_llm.dialog.pipeline as A;
# import fsm_llm.dialog.turn as B; A is B.
"""

import sys as _sys
import warnings as _warnings

_warnings.warn(
    "`fsm_llm.dialog.pipeline` is a deprecated alias for "
    "`fsm_llm.dialog.turn` (since 0.5.0; will be removed in 0.6.0). "
    "Update: `from fsm_llm.dialog.pipeline import …` → "
    "`from fsm_llm.dialog.turn import …`.",
    DeprecationWarning,
    stacklevel=2,
)

import fsm_llm.dialog.turn as _impl
from fsm_llm.dialog.turn import *  # noqa: F403

_sys.modules[__name__] = _impl
