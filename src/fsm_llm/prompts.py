from __future__ import annotations

"""
fsm_llm.prompts — back-compat shim over fsm_llm.dialog.prompts (R4, plan v3 step 22).

The implementation moved to ``fsm_llm.dialog.prompts`` per ``docs/lambda.md`` §11.
This module preserves the old import path and the identity contract
``import fsm_llm.prompts as A; import fsm_llm.dialog.prompts as B; A is B``.

# DECISION D-004 — silent shim policy:
# - 0.4.x: shim works silently (this PR).
# - 0.5.0: emit DeprecationWarning at import time.
# - 0.6.0: remove the shim.
"""

import sys as _sys

import fsm_llm.dialog.prompts as _impl
from fsm_llm.dialog.prompts import *  # noqa: F403

_sys.modules[__name__] = _impl
