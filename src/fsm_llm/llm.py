from __future__ import annotations

"""
fsm_llm.llm — back-compat shim over fsm_llm.runtime._litellm (R4, plan v3 step 23).

The implementation moved to ``fsm_llm.runtime._litellm`` per ``docs/lambda.md`` §11
(R3+R4: LiteLLMInterface becomes a private-by-convention runtime adapter, with
the Oracle protocol as the unified call shape). This module preserves
``from fsm_llm.llm import LiteLLMInterface, LLMInterface`` and the identity
contract ``import fsm_llm.llm as A; import fsm_llm.runtime._litellm as B; A is B``.

# DECISION D-004 — silent shim policy:
# - 0.4.x: shim works silently (this PR).
# - 0.5.0: emit DeprecationWarning at import time.
# - 0.6.0: remove the shim.
"""

import sys as _sys

import fsm_llm.runtime._litellm as _impl
from fsm_llm.runtime._litellm import *  # noqa: F403

_sys.modules[__name__] = _impl
