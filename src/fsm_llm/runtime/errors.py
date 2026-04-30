from __future__ import annotations

"""
Exception hierarchy for the λ (lam) subpackage.

Rooted under ``fsm_llm.definitions.FSMError`` to stay consistent with the
rest of the framework (CLAUDE.md — "Exceptions: Core: FSMError -> ...").
Four leaf subclasses cover the four points where M1 can legitimately fail:

- ``ASTConstructionError`` — misuse of the builder DSL or structurally
  invalid AST (e.g., ``Var`` with no binding, ``Case`` with no matching
  branch and no default).
- ``TerminationError`` — runtime depth breach in the ``Fix`` trampoline,
  or a non-rank-reducing ``SPLIT`` detected at Leaf entry (I5 violation).
- ``PlanningError`` — planner input is infeasible (e.g., ``|P| > K`` with
  ``k = 1``, meaning no split can fit the budget).
- ``OracleError`` — any oracle call that fails the ``|P| ≤ K`` guard or
  that the underlying LLM interface raises ``LLMResponseError`` for.
"""

from fsm_llm._models import FSMError


class LambdaError(FSMError):
    """Base exception for all λ-kernel failures."""


class ASTConstructionError(LambdaError):
    """Raised when an AST is built incorrectly or evaluated with an
    environment that does not bind a referenced variable."""


class TerminationError(LambdaError):
    """Raised when the executor cannot guarantee termination: Fix depth
    exceeds ``max_depth`` or SPLIT fails to strictly reduce rank."""


class PlanningError(LambdaError):
    """Raised when the planner receives infeasible inputs (e.g., ``|P|``
    larger than any ``k``-split can fit under the context window)."""


class OracleError(LambdaError):
    """Raised when a Leaf invocation exceeds ``|P| ≤ K`` or when the
    underlying ``LLMInterface`` call fails."""
