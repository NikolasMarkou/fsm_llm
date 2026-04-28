from __future__ import annotations

"""
fsm_llm.lam — back-compat shim over fsm_llm.runtime (R4, plan v3 step 20).

The kernel was renamed `lam` → `runtime` in plan v3 (D-PLAN-08, D-PLAN-10).
This module preserves the old import path: every existing
``from fsm_llm.lam import …`` and ``from fsm_llm.lam.<sub> import …`` keeps
working with no behaviour change. New code should import from
``fsm_llm.runtime``.

# DECISION D-004 / D-PIVOT-1-R13 — shim deprecation timeline:
# - 0.4.x: silent shim (was the prior behaviour).
# - 0.5.0: emit DeprecationWarning at import time (R13).
# - 0.6.0: remove the shim entirely.
# Anchored per D-PLAN-10 in plans/plan_2026-04-27_a426f667/decisions.md.
#
# DECISION D-001 (plan_2026-04-27_5d8a038b) — L6 early cleanup: this shim
# now sources `compile_fsm` / `compile_fsm_cached` and the `fsm_compile`
# submodule alias DIRECTLY from `fsm_llm.dialog.compile_fsm`, not via a
# back-reference on `fsm_llm.runtime`. The kernel is closed against
# `dialog/`. See plans/plan_2026-04-27_5d8a038b/decisions.md D-001.

The shim must cover both the top-level package and every submodule because
existing call sites use both `from fsm_llm.lam import Executor` (top-level
re-export) and `from fsm_llm.lam.executor import Executor` (deep submodule
import — see findings/r4-import-sites.md, 9 submodule paths in use).
"""

import importlib as _importlib
import sys as _sys
import warnings as _warnings

# DECISION D-PIVOT-1-R13: emit DeprecationWarning at import time.
# stacklevel=2 surfaces the warning at the user's import line.
_warnings.warn(
    "`fsm_llm.lam` is a deprecated alias for `fsm_llm.runtime` (since "
    "0.5.0; will be removed in 0.6.0). Update imports: "
    "`from fsm_llm.lam import …` → `from fsm_llm.runtime import …`. "
    "FSM compiler: `from fsm_llm.lam import compile_fsm` → "
    "`from fsm_llm import compile_fsm` (R11 promotion).",
    DeprecationWarning,
    stacklevel=2,
)

import fsm_llm.runtime as _runtime

# Re-export every public name from runtime so `from fsm_llm.lam import X`
# resolves identically to `from fsm_llm.runtime import X`.
from fsm_llm.runtime import *  # noqa: F403
from fsm_llm.runtime import __all__ as _runtime_all

# Identity contract: import fsm_llm.lam as A; import fsm_llm.runtime as B; A is B.
# Achieved by re-pointing this module entry in sys.modules to runtime.
# Must be done AFTER the `from … import *` above so that any in-progress
# import statements that already grabbed a reference to this module see the
# same object.
_sys.modules[__name__] = _runtime

# DECISION D-001 (plan_2026-04-27_5d8a038b): the runtime back-reference to
# `compile_fsm` / `compile_fsm_cached` was removed from
# `fsm_llm.runtime.__init__`. Because `fsm_llm.lam` is now identical to
# `_runtime` (sys.modules swap above), `from fsm_llm.lam import compile_fsm`
# resolves attribute access on the runtime module — so we must re-attach
# these names to `_runtime` directly. They are NOT added to
# `runtime.__all__`; they remain accessible to lam-shim consumers only.
from fsm_llm.dialog.compile_fsm import (
    compile_fsm as _compile_fsm,
)
from fsm_llm.dialog.compile_fsm import (
    compile_fsm_cached as _compile_fsm_cached,
)

_runtime.compile_fsm = _compile_fsm
_runtime.compile_fsm_cached = _compile_fsm_cached

# Submodule shimming — register each `fsm_llm.lam.<sub>` as an alias for
# `fsm_llm.runtime.<sub>`. Without this, `from fsm_llm.lam.executor import
# Executor` would raise ModuleNotFoundError because the file
# fsm_llm/lam/executor.py no longer exists (it was git-mv'd to runtime/).
for _sub in (
    "ast",
    "combinators",
    "constants",
    "cost",
    "dsl",
    "errors",
    "executor",
    "oracle",
    "planner",
):
    _sys.modules[f"fsm_llm.lam.{_sub}"] = getattr(_runtime, _sub)

# DECISION D-001 (plan_2026-04-27_5d8a038b): `fsm_compile` is sourced
# directly from `fsm_llm.dialog.compile_fsm` rather than via the kernel.
# Identity preserved: `sys.modules["fsm_llm.lam.fsm_compile"] is
# sys.modules["fsm_llm.dialog.compile_fsm"]` (asserted by
# tests/test_fsm_llm/test_module_shims.py).
_dialog_compile_fsm_mod = _importlib.import_module("fsm_llm.dialog.compile_fsm")
_sys.modules["fsm_llm.lam.fsm_compile"] = _dialog_compile_fsm_mod

# Also expose `fsm_compile` as a module attribute on `_runtime` (which is
# `fsm_llm.lam` after the sys.modules swap above) so that
# `from fsm_llm.lam import fsm_compile` keeps resolving to the dialog
# submodule. NOT added to `runtime.__all__` — visible only as a
# side-effect of importing the lam shim.
_runtime.fsm_compile = _dialog_compile_fsm_mod

__all__ = list(_runtime_all)
