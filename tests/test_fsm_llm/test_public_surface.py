"""Tests for the 0.9.0 public surface — slim top-level + sub-namespaces.

Coverage:

- Top-level imports: ``Program``, ``Result``, ``ProgramModeError``,
  ``Executor``, ``compile_fsm``, ``Oracle``, factory roots.
- Sub-namespace identity: ``fsm_llm.ast``, ``fsm_llm.dsl``,
  ``fsm_llm.combinators``, ``fsm_llm.factories``, ``fsm_llm.errors``,
  ``fsm_llm.debug`` re-exports preserve object identity vs canonical homes.
- ``__all__`` order: ``Program`` appears before dialog-tier names.
- ``compose`` exported.
- Stdlib factory terms reachable via ``fsm_llm.factories`` and the
  per-domain stdlib paths.
"""

from __future__ import annotations


class TestTopLevelImports:
    """The slim top-level keeps the high-traffic API."""

    def test_facade_imports(self):
        from fsm_llm import (  # noqa: F401
            Executor,
            Oracle,
            Program,
            ProgramModeError,
            Result,
            compile_fsm,
        )

    def test_substrate_off_top_level(self):
        """0.9.0: AST and DSL primitives moved to sub-namespaces."""
        import fsm_llm

        for name in ("Term", "Var", "Abs", "App", "Let", "Case", "Leaf", "Fix"):
            assert name not in fsm_llm.__all__, (
                f"{name!r} should be in fsm_llm.ast, not top-level (0.9.0)"
            )
        for name in ("var", "abs_", "let", "case_", "fix", "leaf"):
            assert name not in fsm_llm.__all__, (
                f"{name!r} should be in fsm_llm.dsl, not top-level (0.9.0)"
            )

    def test_factories_off_top_level(self):
        """0.9.0: factory ``*_term`` functions moved to ``fsm_llm.factories``."""
        import fsm_llm

        for name in ("react_term", "analytical_term", "linear_term", "niah_term"):
            assert name not in fsm_llm.__all__, (
                f"{name!r} should be in fsm_llm.factories, not top-level"
            )


class TestSubNamespaceIdentity:
    """Sub-namespace re-exports preserve object identity vs canonical homes."""

    def test_ast_namespace_identity(self):
        from fsm_llm.ast import Term, Var, Abs, Leaf, Fix
        from fsm_llm.runtime.ast import Term as CT, Var as CV, Abs as CA, Leaf as CL, Fix as CF

        assert Term is CT
        assert Var is CV
        assert Abs is CA
        assert Leaf is CL
        assert Fix is CF

    def test_dsl_namespace_identity(self):
        from fsm_llm.dsl import leaf, fix, let, case_, var, abs_
        from fsm_llm.runtime.dsl import (
            leaf as cleaf,
            fix as cfix,
            let as clet,
            case_ as ccase,
            var as cvar,
            abs_ as cabs,
        )

        assert leaf is cleaf
        assert fix is cfix
        assert let is clet
        assert case_ is ccase
        assert var is cvar
        assert abs_ is cabs

    def test_combinators_namespace_identity(self):
        from fsm_llm.combinators import split, fmap, ffilter, reduce, ReduceOp
        from fsm_llm.runtime.dsl import (
            split as csplit,
            fmap as cfmap,
            ffilter as cffilter,
            reduce as creduce,
        )
        from fsm_llm.runtime.combinators import ReduceOp as CRO

        assert split is csplit
        assert fmap is cfmap
        assert ffilter is cffilter
        assert reduce is creduce
        assert ReduceOp is CRO

    def test_factories_namespace_identity(self):
        from fsm_llm.factories import react_term, niah_term, analytical_term
        from fsm_llm.stdlib.agents import react_term as cr
        from fsm_llm.stdlib.long_context import niah_term as cn
        from fsm_llm.stdlib.reasoning.lam_factories import analytical_term as ca

        assert react_term is cr
        assert niah_term is cn
        assert analytical_term is ca

    def test_errors_namespace_identity(self):
        from fsm_llm.errors import (
            FSMError,
            LambdaError,
            ProgramModeError,
            HandlerSystemError,
            ReasoningEngineError,
            WorkflowError,
            AgentError,
        )
        from fsm_llm._models import FSMError as CF
        from fsm_llm.runtime.errors import LambdaError as CL
        from fsm_llm.program import ProgramModeError as CP
        from fsm_llm.handlers import HandlerSystemError as CH
        from fsm_llm.stdlib.reasoning.exceptions import ReasoningEngineError as CR
        from fsm_llm.stdlib.workflows.exceptions import WorkflowError as CW
        from fsm_llm.stdlib.agents.exceptions import AgentError as CA

        assert FSMError is CF
        assert LambdaError is CL
        assert ProgramModeError is CP
        assert HandlerSystemError is CH
        assert ReasoningEngineError is CR
        assert WorkflowError is CW
        assert AgentError is CA

    def test_debug_namespace(self):
        from fsm_llm.debug import (
            BUFFER_METADATA,
            disable_warnings,
            enable_debug_logging,
        )

        assert callable(enable_debug_logging)
        assert callable(disable_warnings)
        assert isinstance(BUFFER_METADATA, str)

    def test_compile_fsm_identity(self):
        import fsm_llm
        from fsm_llm.dialog.compile_fsm import compile_fsm as canonical_compile

        assert fsm_llm.compile_fsm is canonical_compile


class TestAllOrdering:
    """Substrate / Program names must appear before dialog-tier names."""

    def test_program_before_fsmmanager(self):
        import fsm_llm

        idx_program = fsm_llm.__all__.index("Program")
        idx_fsm = fsm_llm.__all__.index("FSMManager")
        assert idx_program < idx_fsm

    def test_executor_before_classifier(self):
        import fsm_llm

        assert fsm_llm.__all__.index("Executor") < fsm_llm.__all__.index("Classifier")


class TestHandlerSurfaceAndCompose:
    """``FSMHandler`` + ``compose`` are the canonical L2 names at 0.9.0.

    The ``Handler`` alias was removed at 0.8.0; ``HandlerSystem`` was
    dropped from top-level at 0.9.0.
    """

    def test_handler_alias_removed(self):
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


class TestPlannerReexported:
    """Planner names (``PlanInputs``, ``Plan``, ``plan``) at top level."""

    def test_planner_names(self):
        from fsm_llm import Plan, PlanInputs, plan

        assert callable(plan)
        pi = PlanInputs(n=4, K=4096)
        p = plan(pi)
        assert isinstance(p, Plan)


class TestRootErrorsAtTopLevel:
    """0.9.0: only roots (FSMError + LambdaError) at top level."""

    def test_root_errors_top_level(self):
        from fsm_llm import FSMError, LambdaError  # noqa: F401

    def test_subclasses_off_top_level(self):
        import fsm_llm

        for name in (
            "ASTConstructionError",
            "TerminationError",
            "PlanningError",
            "OracleError",
            "StateNotFoundError",
            "InvalidTransitionError",
            "LLMResponseError",
            "HandlerSystemError",
        ):
            assert name not in fsm_llm.__all__, (
                f"{name!r} should be in fsm_llm.errors, not top-level (0.9.0)"
            )


class TestDialogTierStillReachable:
    """Dialog tier — FSM dialog front-end names still at top level."""

    def test_legacy_api_names(self):
        from fsm_llm import (  # noqa: F401
            Classifier,
            ContextMergeStrategy,
            FSMDefinition,
            FSMManager,
            HandlerBuilder,
            HandlerTiming,
        )

    def test_api_class_still_importable_via_dialog(self):
        from fsm_llm.dialog.api import API  # noqa: F401

    def test_litellm_interface_private_at_runtime(self):
        from fsm_llm.runtime._litellm import LiteLLMInterface  # noqa: F401

    def test_dialog_definitions_type_reexports_removed_at_080(self):
        """The 0.7.0 back-compat re-export block in dialog/definitions.py was
        removed at 0.8.0. Names that moved to ``fsm_llm._models`` are no
        longer importable via the legacy path."""
        import fsm_llm.dialog.definitions as defs
        from fsm_llm._models import (  # noqa: F401
            FSMError,
            LLMRequestType,
            ResponseGenerationResponse,
        )

        for name in (
            "FSMError",
            "StateNotFoundError",
            "InvalidTransitionError",
            "LLMResponseError",
            "TransitionEvaluationError",
            "ClassificationError",
            "SchemaValidationError",
            "ClassificationResponseError",
            "FieldExtractionRequest",
            "FieldExtractionResponse",
            "LLMRequestType",
        ):
            assert not hasattr(defs, name), (
                f"{name!r} should no longer be re-exported from "
                "dialog/definitions at 0.8.0; import it from fsm_llm._models"
            )
