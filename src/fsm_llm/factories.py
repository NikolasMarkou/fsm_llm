from __future__ import annotations

"""fsm_llm.factories — named λ-term factory functions.

Thin re-export of every ``*_term`` factory from the stdlib subpackages.
Canonical import path for term-mode authoring since 0.9.0:

    from fsm_llm.factories import react_term, analytical_term, linear_term

The full per-domain surface (with subpackage-specific helpers like
``ReasoningEngine`` or ``WorkflowEngine``) remains importable from the
subpackages directly:

- ``fsm_llm.stdlib.agents``
- ``fsm_llm.stdlib.reasoning``
- ``fsm_llm.stdlib.workflows``
- ``fsm_llm.stdlib.long_context``

This module only surfaces the term builder functions — the ``*_term``
suffix is the canonical naming convention since 0.6.0.
"""

from .stdlib.agents import (
    memory_term,
    react_term,
    reflexion_term,
    rewoo_term,
)
from .stdlib.long_context import (
    aggregate_term,
    multi_hop_dynamic_term,
    multi_hop_term,
    niah_padded_term,
    niah_term,
    pairwise_term,
)
from .stdlib.reasoning.lam_factories import (
    abductive_term,
    analogical_term,
    analytical_term,
    calculator_term,
    classifier_term,
    creative_term,
    critical_term,
    deductive_term,
    hybrid_term,
    inductive_term,
    solve_term,
)
from .stdlib.workflows.lam_factories import (
    branch_term,
    linear_term,
    parallel_term,
    retry_term,
    switch_term,
)

__all__ = [
    # Agents (4)
    "react_term",
    "rewoo_term",
    "reflexion_term",
    "memory_term",
    # Reasoning (11)
    "analytical_term",
    "deductive_term",
    "inductive_term",
    "abductive_term",
    "analogical_term",
    "creative_term",
    "critical_term",
    "hybrid_term",
    "calculator_term",
    "classifier_term",
    "solve_term",
    # Workflows (5)
    "linear_term",
    "branch_term",
    "switch_term",
    "parallel_term",
    "retry_term",
    # Long-context (6)
    "niah_term",
    "aggregate_term",
    "pairwise_term",
    "multi_hop_term",
    "multi_hop_dynamic_term",
    "niah_padded_term",
]
