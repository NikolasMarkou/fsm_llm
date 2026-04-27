from __future__ import annotations

"""
Constants for the λ (lam) subpackage.

Centralised defaults for the executor, planner, and oracle adapter. Kept
separate from the top-level ``fsm_llm.constants`` module so the lam kernel
stays self-contained (per D-002 — additive-only M1).
"""

# Executor safety cap — hard upper bound on Fix trampoline recursion depth.
# If planner computes ``d`` greater than this, executor raises
# ``TerminationError`` before the first Leaf call is issued.
DEFAULT_MAX_DEPTH: int = 32

# Default branching factor for SPLIT under linear cost (paper Thm 4).
DEFAULT_K_STAR: int = 2

# Default oracle context-window budget in tokens. Callers SHOULD override
# with the real ``K`` for the model in use; this is a conservative fallback.
DEFAULT_CONTEXT_WINDOW: int = 8192

# Default split-size threshold τ used when planner inputs omit it. Kept
# deliberately conservative so small inputs short-circuit to a single Leaf.
DEFAULT_TAU: int = 512

# Default toy accuracy constants used by planner theorems when real
# measurements are unavailable. These are placeholders — callers SHOULD
# pass measured values via ``PlanInputs``.
DEFAULT_LEAF_ACCURACY: float = 0.9
DEFAULT_COMBINE_ACCURACY: float = 0.95

# Default per-token cost proxy (arbitrary units). Only ratios matter for
# the planner; absolute numbers are reported by ``CostAccumulator`` using
# real usage metadata from the oracle.
DEFAULT_TOKEN_COST: float = 1.0

# Char-per-token fallback used when a real tokenizer is unavailable.
CHARS_PER_TOKEN_FALLBACK: int = 4
