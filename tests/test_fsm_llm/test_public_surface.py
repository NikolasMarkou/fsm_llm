"""Tests for the R11 public-surface promotion.

Plan: plans/plan_2026-04-27_32652286/plan.md — Step 2 (Bundle A).

Coverage:

- SC1: ``from fsm_llm import Program, Term, leaf, fix, react_term, niah,
  compile_fsm, Executor, Oracle, Result, ProgramModeError`` succeeds.
- I6 identity: substrate names re-exported from `fsm_llm` are the *same
  object* as their canonical home in `fsm_llm.runtime` / `dialog`.
- ``__all__`` order: substrate names appear before FSM-front-end names.
- ``Handler`` alias points at ``FSMHandler``.
- ``compose`` exported.
- Stdlib factory terms (react_term, rewoo_term, reflexion_term,
  memory_term, niah, aggregate, pairwise, multi_hop) all reachable.
"""

from __future__ import annotations


class TestImportLineSucceeds:
    """SC1 — single import line covers the substrate + facade names."""

    def test_full_import_line(self):
        # The README-quality import line must not raise.
        from fsm_llm import (  # noqa: F401
            Executor,
            Oracle,
            Program,
            ProgramModeError,
            Result,
            Term,
            compile_fsm,
            fix,
            leaf,
            niah_term,
            react_term,
        )


class TestI6IdentityContracts:
    """I6: re-exports preserve object identity vs the canonical module."""

    def test_term_identity(self):
        import fsm_llm
        from fsm_llm.runtime.ast import Term as CanonicalTerm

        assert fsm_llm.Term is CanonicalTerm

    def test_leaf_identity(self):
        import fsm_llm
        from fsm_llm.runtime.dsl import leaf as canonical_leaf

        assert fsm_llm.leaf is canonical_leaf

    def test_fix_identity(self):
        import fsm_llm
        from fsm_llm.runtime.dsl import fix as canonical_fix

        assert fsm_llm.fix is canonical_fix

    def test_executor_identity(self):
        import fsm_llm
        from fsm_llm.runtime.executor import Executor as CanonicalExecutor

        assert fsm_llm.Executor is CanonicalExecutor

    def test_oracle_identity(self):
        import fsm_llm
        from fsm_llm.runtime.oracle import Oracle as CanonicalOracle

        assert fsm_llm.Oracle is CanonicalOracle

    def test_compile_fsm_identity(self):
        import fsm_llm
        from fsm_llm.dialog.compile_fsm import compile_fsm as canonical_compile

        assert fsm_llm.compile_fsm is canonical_compile

    def test_react_term_identity(self):
        import fsm_llm
        from fsm_llm.stdlib.agents import react_term as canonical_react

        assert fsm_llm.react_term is canonical_react

    def test_niah_term_identity(self):
        import fsm_llm
        from fsm_llm.stdlib.long_context import niah_term as canonical_niah_term

        assert fsm_llm.niah_term is canonical_niah_term


class TestAllOrdering:
    """Substrate names must appear in __all__ before FSM-front-end names.

    The ``API`` re-export was the canonical Legacy anchor through 0.6.x but
    was removed at 0.7.0. ``FSMManager`` is the next-most-prominent Legacy
    name and serves as the new ordering anchor.
    """

    def test_program_before_fsmmanager(self):
        import fsm_llm

        idx_program = fsm_llm.__all__.index("Program")
        idx_fsm = fsm_llm.__all__.index("FSMManager")
        assert idx_program < idx_fsm

    def test_result_before_fsmmanager(self):
        import fsm_llm

        assert fsm_llm.__all__.index("Result") < fsm_llm.__all__.index("FSMManager")

    def test_term_before_fsmmanager(self):
        import fsm_llm

        assert fsm_llm.__all__.index("Term") < fsm_llm.__all__.index("FSMManager")

    def test_leaf_before_fsmmanager(self):
        import fsm_llm

        assert fsm_llm.__all__.index("leaf") < fsm_llm.__all__.index("FSMManager")

    def test_executor_before_classifier(self):
        import fsm_llm

        assert fsm_llm.__all__.index("Executor") < fsm_llm.__all__.index("Classifier")


class TestHandlerAliasAndCompose:
    """`Handler` alias + `compose` exported (R11)."""

    def test_handler_alias_is_fsmhandler(self):
        import fsm_llm

        assert fsm_llm.Handler is fsm_llm.FSMHandler

    def test_handler_in_all(self):
        import fsm_llm

        assert "Handler" in fsm_llm.__all__

    def test_compose_in_all(self):
        import fsm_llm

        assert "compose" in fsm_llm.__all__

    def test_compose_callable(self):
        from fsm_llm import compose

        assert callable(compose)


class TestStdlibFactoryReexports:
    """Stdlib factory terms reachable at top level."""

    def test_agent_factory_terms(self):
        from fsm_llm import memory_term, react_term, reflexion_term, rewoo_term

        assert callable(react_term)
        assert callable(rewoo_term)
        assert callable(reflexion_term)
        assert callable(memory_term)

    def test_long_context_factory_terms(self):
        from fsm_llm import (
            aggregate_term,
            multi_hop_dynamic_term,
            multi_hop_term,
            niah_padded_term,
            niah_term,
            pairwise_term,
        )

        assert callable(niah_term)
        assert callable(aggregate_term)
        assert callable(pairwise_term)
        assert callable(multi_hop_term)
        assert callable(multi_hop_dynamic_term)
        assert callable(niah_padded_term)


class TestKernelExceptionsReexported:
    """Kernel exceptions surface at the top level (R11)."""

    def test_lambda_error_chain(self):
        from fsm_llm import (
            ASTConstructionError,
            LambdaError,
            OracleError,
            PlanningError,
            TerminationError,
        )

        assert issubclass(ASTConstructionError, LambdaError)
        assert issubclass(TerminationError, LambdaError)
        assert issubclass(PlanningError, LambdaError)
        assert issubclass(OracleError, LambdaError)


class TestPlannerReexported:
    """Planner names (PlanInputs, Plan, plan) at top level."""

    def test_planner_names(self):
        from fsm_llm import Plan, PlanInputs, plan

        assert callable(plan)
        # PlanInputs and Plan are dataclass-like; instantiation via
        # the public constructor is enough proof of import-line wiring.
        # Modest feasible inputs: n leaves easily within K-budget.
        pi = PlanInputs(n=4, K=4096)
        p = plan(pi)
        assert isinstance(p, Plan)


class TestNoRegressionsInLegacySurface:
    """All pre-R11 names still reachable (back-compat I3) — except those
    explicitly removed at 0.7.0 (the I5 epoch closure: top-level ``API``
    is gone, sibling shim packages are gone)."""

    def test_legacy_api_names(self):
        from fsm_llm import (  # noqa: F401
            Classifier,
            ContextMergeStrategy,
            FSMDefinition,
            FSMManager,
            HandlerBuilder,
            HandlerSystem,
            HandlerTiming,
            LLMInterface,
        )

    def test_api_class_still_importable_via_dialog(self):
        """``API`` was removed from the top-level convenience surface at
        0.7.0 but the class still lives at ``fsm_llm.dialog.api.API``."""
        from fsm_llm.dialog.api import API  # noqa: F401

    def test_litellm_interface_private_at_runtime(self):
        """``LiteLLMInterface`` was formalised as private at 0.7.0 (D-009).
        Top-level re-export removed; canonical path is the runtime adapter."""
        from fsm_llm.runtime._litellm import LiteLLMInterface  # noqa: F401
