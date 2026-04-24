from __future__ import annotations

"""
fsm_llm.lam — λ-calculus substrate (M1 kernel).

Additive subpackage introducing a typed λ-AST, Python builder DSL, pure
combinator library, planner, β-reduction executor, oracle adapter over
``LiteLLMInterface``, and per-leaf cost accumulator.

See ``docs/lambda.md`` for the full design and theorems. M1 delivers the
kernel only — the FSM-JSON → λ compiler (M2), stdlib reorg (M3), example
migration (M4), and Category-C benchmarks (M5) land in subsequent plans.

Public API is populated progressively as each module is implemented. Users
import via ``from fsm_llm.lam import X``; the top-level ``fsm_llm`` package
is intentionally not modified during M1.
"""

__all__: list[str] = []
