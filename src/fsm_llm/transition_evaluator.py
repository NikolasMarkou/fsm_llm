from __future__ import annotations

"""
fsm_llm.transition_evaluator — back-compat shim over fsm_llm.dialog.transition_evaluator (R4, plan v3 step 22).

The implementation moved to ``fsm_llm.dialog.transition_evaluator`` per ``docs/lambda.md`` §11.
This module preserves the old import path and the identity contract
``import fsm_llm.transition_evaluator as A; import fsm_llm.dialog.transition_evaluator as B; A is B``.

# DECISION D-004 — silent shim policy:
# - 0.4.x: shim works silently (this PR).
# - 0.5.0: emit DeprecationWarning at import time.
# - 0.6.0: remove the shim.
"""

import sys as _sys

import fsm_llm.dialog.transition_evaluator as _impl
from fsm_llm.dialog.transition_evaluator import *  # noqa: F403

_sys.modules[__name__] = _impl
