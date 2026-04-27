from __future__ import annotations

"""
fsm_llm.fsm — back-compat shim over fsm_llm.dialog.fsm (R4, plan v3 step 22).

The implementation moved to ``fsm_llm.dialog.fsm`` per ``docs/lambda.md`` §11.
This module preserves the old import path and the identity contract
``import fsm_llm.fsm as A; import fsm_llm.dialog.fsm as B; A is B``.

# DECISION D-004 — silent shim policy:
# - 0.4.x: shim works silently (this PR).
# - 0.5.0: emit DeprecationWarning at import time.
# - 0.6.0: remove the shim.
"""

import sys as _sys

import fsm_llm.dialog.fsm as _impl
from fsm_llm.dialog.fsm import *  # noqa: F403

_sys.modules[__name__] = _impl
