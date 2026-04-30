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


class TestHandlerSurfaceAndCompose:
    """`FSMHandler` + `compose` are the canonical L2 names.

    0.8.0: the back-compat ``Handler`` alias was removed — ``FSMHandler``
    is the only protocol name. Importing ``Handler`` from the top level
    now raises ``ImportError``.
    """

    def test_handler_alias_removed_at_080(self):
        import fsm_llm

        assert not hasattr(fsm_llm, "Handler"), (
            "back-compat Handler alias should be removed at 0.8.0"
        )
        assert "Handler" not in fsm_llm.__all__

    def test_fsmhandler_in_all(self):
        import fsm_llm

        assert "FSMHandler" in fsm_llm.__all__

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
    """Legacy dialog-surface names still reachable (back-compat I3) —
    except those explicitly removed across the I5 (0.7.0) and 0.8.0
    deep-cleanup epochs.

    Removed at 0.7.0: top-level ``API``, sibling shim packages.
    Removed at 0.8.0: top-level ``LLMInterface`` (D-009 closure;
    canonical path is ``fsm_llm.runtime._litellm.LLMInterface``),
    ``Handler`` alias, ``BUILTIN_OPS`` (closed registry, internal-only),
    ``has_*`` / ``get_*`` extension-check helpers.
    """

    def test_legacy_api_names(self):
        from fsm_llm import (  # noqa: F401
            Classifier,
            ContextMergeStrategy,
            FSMDefinition,
            FSMManager,
            HandlerBuilder,
            HandlerSystem,
            HandlerTiming,
        )

    def test_api_class_still_importable_via_dialog(self):
        """``API`` was removed from the top-level convenience surface at
        0.7.0 but the class still lives at ``fsm_llm.dialog.api.API``."""
        from fsm_llm.dialog.api import API  # noqa: F401

    def test_litellm_interface_private_at_runtime(self):
        """``LiteLLMInterface`` was formalised as private at 0.7.0 (D-009).
        Top-level re-export removed; canonical path is the runtime adapter."""
        from fsm_llm.runtime._litellm import LiteLLMInterface  # noqa: F401

    def test_llm_interface_removed_from_top_level_at_080(self):
        """``LLMInterface`` was removed from the top-level surface at 0.8.0
        (D-009 closure). Canonical path is ``fsm_llm.runtime._litellm``."""
        import fsm_llm

        assert "LLMInterface" not in fsm_llm.__all__
        # Canonical path still works:
        from fsm_llm.runtime._litellm import LLMInterface  # noqa: F401

    def test_builtin_ops_removed_from_top_level_at_080(self):
        """``BUILTIN_OPS`` is a closed kernel registry — removed from the
        top-level surface at 0.8.0. Canonical path is ``fsm_llm.runtime``."""
        import fsm_llm

        assert "BUILTIN_OPS" not in fsm_llm.__all__
        assert not hasattr(fsm_llm, "BUILTIN_OPS")
        # Canonical path still works:
        from fsm_llm.runtime import BUILTIN_OPS  # noqa: F401

    def test_extension_check_helpers_removed_at_080(self):
        """``has_*`` / ``get_*`` extension-check helpers were removed at
        0.8.0. The stdlib subpackages ship with core since 0.7.0 (the
        sibling shim packages were deleted at the I5 epoch closure), so
        the helpers had become no-ops returning unconditional ``True``.
        Direct imports continue to work:
        ``from fsm_llm.stdlib import workflows, reasoning, agents``.
        """
        import fsm_llm

        for name in (
            "has_workflows",
            "has_reasoning",
            "has_agents",
            "get_workflows",
            "get_reasoning",
            "get_agents",
        ):
            assert name not in fsm_llm.__all__
            assert not hasattr(fsm_llm, name)
        # Direct subpackage imports still work:
        from fsm_llm.stdlib import agents, reasoning, workflows  # noqa: F401
