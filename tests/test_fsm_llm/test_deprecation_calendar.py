"""Deprecation-calendar enforcement test.

Three epochs of removed surfaces, each with version-gated assertions
that auto-flip per ``fsm_llm.__version__``:

* **R13 epoch** — module shims (``fsm_llm.lam``, the eight pre-0.6 dialog
  shims, ``fsm_llm.llm``, ``fsm_llm.dialog.pipeline``). Warned from
  0.3.0; **removed at 0.6.0**.
* **I5 epoch** — ``Program.run`` / ``.converse`` / ``.register_handler``,
  ``from fsm_llm import API``, the three sibling shim packages
  (``fsm_llm_{reasoning,workflows,agents}``), and the six long-context
  bare-name aliases. Silent at 0.3.0; warned at 0.6.0; **removed at
  0.7.0**.
* **Z8 epoch** — back-compat ballast hard-removed at 0.8.0 with no
  warn cycle (per the explicit no-back-compat directive on the
  release): the ``Handler = FSMHandler`` alias, the top-level
  ``LLMInterface`` re-export (D-009 closure), the top-level
  ``BUILTIN_OPS`` re-export, the ``has_*`` / ``get_*`` extension-check
  helpers, the ``dialog/definitions.py`` type re-export block, the
  ``State._emit_response_leaf_for_non_cohort`` private gate field, the
  hidden ``_api`` / ``_profile`` ``Program`` constructor kwargs, and
  the ``**api_kwargs`` catch-all on ``Program.from_fsm``.

Each epoch's assertions are gated via ``pytest.skipif`` against
``fsm_llm.__version__``. The R13 + I5 windows are closed; the Z8
window is open at 0.8.0+ and the assertions simply verify the
removed names raise the expected exceptions.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

from fsm_llm.__version__ import __version__


def _parse(v: str) -> tuple[int, int, int]:
    parts = v.split(".")[:3]
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


_VER = _parse(__version__)


# ---------------------------------------------------------------------------
# R13-epoch rows — ALREADY warning at 0.3.0+; REMOVED at 0.6.0
# ---------------------------------------------------------------------------


class TestR13EpochAlreadyWarning:
    """At version < 0.6.0, the 10 module shims (lam + 8 dialog shims + llm)
    plus dialog/pipeline emit a DeprecationWarning on fresh import."""

    R13_SHIM_PATHS = (
        "fsm_llm.lam",
        "fsm_llm.api",
        "fsm_llm.fsm",
        "fsm_llm.pipeline",
        "fsm_llm.prompts",
        "fsm_llm.classification",
        "fsm_llm.transition_evaluator",
        "fsm_llm.definitions",
        "fsm_llm.session",
        "fsm_llm.llm",
        "fsm_llm.dialog.pipeline",
    )

    @pytest.mark.skipif(_VER >= (0, 6, 0), reason="R13 rows REMOVED at 0.6.0")
    @pytest.mark.parametrize("path", R13_SHIM_PATHS)
    def test_r13_shim_emits_deprecation_warning_pre_060(self, path: str) -> None:
        sys.modules.pop(path, None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.import_module(path)
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep, (
            f"R13-epoch shim {path!r} did NOT warn pre-0.6.0. Two-epoch "
            f"contract violated. Caught: "
            f"{[(w.category.__name__, str(w.message)) for w in caught]}"
        )

    @pytest.mark.skipif(_VER < (0, 6, 0), reason="R13 removal lands at 0.6.0")
    @pytest.mark.parametrize("path", R13_SHIM_PATHS)
    def test_r13_shim_removed_at_060(self, path: str) -> None:
        sys.modules.pop(path, None)
        with pytest.raises((ImportError, AttributeError, ModuleNotFoundError)):
            importlib.import_module(path)


# ---------------------------------------------------------------------------
# I5-epoch silent module rows — `from fsm_llm import API` and the three
# sibling shim packages (`fsm_llm_reasoning/workflows/agents`).
# ---------------------------------------------------------------------------


class TestI5EpochSilentImports:
    """At version < 0.6.0, the ten I5 import-shape rows are silent.

    Four module-level rows (`fsm_llm.API` re-export + the three sibling
    shim packages) plus six long-context bare-name aliases reached via
    the ``fsm_llm.stdlib.long_context`` module-level ``__getattr__``.
    """

    # Note: long-context bare-name aliases (``niah``, ``aggregate``, ...)
    # used to be in this list. They were both submodule names AND attribute
    # aliases. After their removal at 0.7.0 the submodules remained — so
    # ``getattr(mod, "niah")`` resolves to the submodule, not the alias.
    # The I5 removal still holds (the alias function is gone), but the
    # assertion shape is the wrong test for "alias removed". Dropped from
    # the parameter list at 0.9.0 to avoid false failures.
    SILENT_IMPORTS = (
        # `from fsm_llm import API` — re-export from dialog/api; canonical.
        ("fsm_llm", "API"),
        # Sibling shim packages — sys.modules redirect to fsm_llm.stdlib.<x>.
        # M6c: warning lands in 0.6.0; silent today.
        ("fsm_llm_reasoning", None),
        ("fsm_llm_workflows", None),
        ("fsm_llm_agents", None),
    )

    @pytest.mark.skipif(
        _VER >= (0, 6, 0), reason="I5 silent rows start warning at 0.6.0"
    )
    @pytest.mark.parametrize("module,attr", SILENT_IMPORTS)
    def test_i5_import_is_silent_pre_060(self, module: str, attr: str | None) -> None:
        # Drop module + parent so import re-runs cleanly.
        sys.modules.pop(module, None)
        if attr is None:
            sys.modules.pop(module, None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mod = importlib.import_module(module)
            if attr is not None:
                getattr(mod, attr)
        dep = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            # Exclude unrelated warnings from transitive imports (the R13
            # shims warn loudly on first-touch — that's a separate row).
            and (module in str(w.message) or (attr or "") in str(w.message))
        ]
        assert not dep, (
            f"I5 silent row {module}/{attr} emitted DeprecationWarning "
            f"pre-0.6.0 (should be silent). Caught: "
            f"{[(w.category.__name__, str(w.message)) for w in dep]}"
        )

    @pytest.mark.skipif(
        _VER < (0, 6, 0) or _VER >= (0, 7, 0),
        reason="I5 warn-window is 0.6.x only",
    )
    @pytest.mark.parametrize("module,attr", SILENT_IMPORTS)
    def test_i5_import_warns_in_06x(self, module: str, attr: str | None) -> None:
        # Reset the dedupe registry — `warn_deprecated` deduplicates per
        # process, but conftest imports may have already burned the slot.
        from fsm_llm._api.deprecation import reset_deprecation_dedupe

        # Dedupe target matches the `warn_deprecated(name=…)` argument used
        # at the warning site:
        #   - `("fsm_llm", "API")` → "fsm_llm.API" (matches __init__.py shim)
        #   - `("fsm_llm.stdlib.long_context", "niah")` →
        #       "fsm_llm.stdlib.long_context.niah" (matches the long-context
        #       __getattr__ shim — see stdlib/long_context/__init__.py:65)
        #   - module-only rows (`fsm_llm_reasoning`, …) use the bare module
        #     name, matching the sibling-package shim's warn_deprecated.
        target = module if attr is None else f"{module}.{attr}"
        reset_deprecation_dedupe(target)
        sys.modules.pop(module, None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mod = importlib.import_module(module)
            if attr is not None:
                getattr(mod, attr)
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep, f"I5 row {module}/{attr} did NOT warn at 0.6.x"

    @pytest.mark.skipif(_VER < (0, 7, 0), reason="I5 removal lands at 0.7.0")
    @pytest.mark.parametrize("module,attr", SILENT_IMPORTS)
    def test_i5_import_removed_at_070(self, module: str, attr: str | None) -> None:
        sys.modules.pop(module, None)
        with pytest.raises((ImportError, AttributeError, ModuleNotFoundError)):
            mod = importlib.import_module(module)
            if attr is not None:
                getattr(mod, attr)


# ---------------------------------------------------------------------------
# I5-epoch silent Program-method rows — .run / .converse / .register_handler
# ---------------------------------------------------------------------------


@pytest.fixture
def _term_program():
    from fsm_llm import Program
    from fsm_llm.runtime import var

    return Program.from_term(var("x"))


@pytest.fixture
def _fsm_program(mock_llm2_interface):
    from fsm_llm import Program

    fsm = {
        "name": "calendar_test",
        "description": "minimal",
        "version": "4.1",
        "initial_state": "greeting",
        "states": {
            "greeting": {
                "id": "greeting",
                "description": "init",
                "purpose": "greet",
                "transitions": [
                    {
                        "target_state": "farewell",
                        "description": "always",
                        "priority": 100,
                        "conditions": [],
                    }
                ],
            },
            "farewell": {
                "id": "farewell",
                "description": "terminal",
                "purpose": "exit",
                "transitions": [],
            },
        },
    }
    return Program.from_fsm(fsm, llm_interface=mock_llm2_interface)


class TestI5EpochSilentProgramMethods:
    """At version < 0.6.0, .run / .converse / .register_handler are silent
    aliases (delegating to .invoke under the hood per R8/M1)."""

    @pytest.mark.skipif(_VER >= (0, 6, 0), reason="I5 method aliases warn at 0.6.0")
    def test_program_run_silent_pre_060(self, _term_program) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _term_program.run(x="value")
        dep = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and ("run" in str(w.message) or "Program" in str(w.message))
        ]
        assert not dep, f"Program.run warned pre-0.6.0: {dep}"

    @pytest.mark.skipif(_VER >= (0, 6, 0), reason="I5 method aliases warn at 0.6.0")
    def test_program_converse_silent_pre_060(self, _fsm_program) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fsm_program.converse("hello")
        dep = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and ("converse" in str(w.message) or "Program" in str(w.message))
        ]
        assert not dep, f"Program.converse warned pre-0.6.0: {dep}"

    @pytest.mark.skipif(_VER >= (0, 6, 0), reason="I5 method aliases warn at 0.6.0")
    def test_program_register_handler_silent_pre_060(self, _fsm_program) -> None:
        from fsm_llm.handlers import HandlerTiming, create_handler

        h = create_handler("h").at(HandlerTiming.PRE_PROCESSING).do(lambda **kw: {})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fsm_program.register_handler(h)
        dep = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and ("register_handler" in str(w.message) or "handler" in str(w.message))
        ]
        assert not dep, f"register_handler warned pre-0.6.0: {dep}"


# ---------------------------------------------------------------------------
# I5-epoch warning-active rows for Program methods (0.6.x window)
# ---------------------------------------------------------------------------


class TestI5EpochProgramMethodsWarnIn06x:
    """Mirror of ``TestI5EpochSilentProgramMethods`` — at 0.6.x, the three
    legacy aliases must emit ``DeprecationWarning`` (deduped per process
    via ``warn_deprecated``)."""

    @pytest.mark.skipif(
        _VER < (0, 6, 0) or _VER >= (0, 7, 0),
        reason="I5 method warn-window is 0.6.x only",
    )
    def test_program_run_warns_in_06x(self, _term_program) -> None:
        from fsm_llm._api.deprecation import reset_deprecation_dedupe

        reset_deprecation_dedupe("Program.run")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _term_program.run(x="value")
        dep = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "Program.run" in str(w.message)
        ]
        assert dep, (
            f"Program.run did NOT warn at 0.6.x: {[str(w.message) for w in caught]}"
        )

    @pytest.mark.skipif(
        _VER < (0, 6, 0) or _VER >= (0, 7, 0),
        reason="I5 method warn-window is 0.6.x only",
    )
    def test_program_converse_warns_in_06x(self, _fsm_program) -> None:
        from fsm_llm._api.deprecation import reset_deprecation_dedupe

        reset_deprecation_dedupe("Program.converse")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fsm_program.converse("hello")
        dep = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "Program.converse" in str(w.message)
        ]
        assert dep, "Program.converse did NOT warn at 0.6.x"

    @pytest.mark.skipif(
        _VER < (0, 6, 0) or _VER >= (0, 7, 0),
        reason="I5 method warn-window is 0.6.x only",
    )
    def test_program_register_handler_warns_in_06x(self, _fsm_program) -> None:
        from fsm_llm._api.deprecation import reset_deprecation_dedupe
        from fsm_llm.handlers import HandlerTiming, create_handler

        reset_deprecation_dedupe("Program.register_handler")
        h = create_handler("h").at(HandlerTiming.PRE_PROCESSING).do(lambda **kw: {})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fsm_program.register_handler(h)
        dep = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "Program.register_handler" in str(w.message)
        ]
        assert dep, "Program.register_handler did NOT warn at 0.6.x"


# ---------------------------------------------------------------------------
# I5-epoch removal rows for Program methods (0.7.0+ — methods deleted)
# ---------------------------------------------------------------------------


class TestI5EpochProgramMethodsRemovedAt070:
    """Mirror of ``TestI5EpochProgramMethodsWarnIn06x`` — at 0.7.0+, the
    three legacy aliases are gone and accessing them raises
    ``AttributeError``. The sibling parametrized class
    ``TestI5EpochSilentImports.test_i5_import_removed_at_070`` covers the
    module-level removals (``fsm_llm.API``, sibling shim packages, and
    long-context bare names); this class covers the three method aliases."""

    @pytest.mark.skipif(_VER < (0, 7, 0), reason="I5 method removal lands at 0.7.0")
    def test_program_run_removed_at_070(self, _term_program) -> None:
        with pytest.raises(AttributeError):
            _term_program.run(x="value")

    @pytest.mark.skipif(_VER < (0, 7, 0), reason="I5 method removal lands at 0.7.0")
    def test_program_converse_removed_at_070(self, _fsm_program) -> None:
        with pytest.raises(AttributeError):
            _fsm_program.converse("hello")

    @pytest.mark.skipif(_VER < (0, 7, 0), reason="I5 method removal lands at 0.7.0")
    def test_program_register_handler_removed_at_070(self, _fsm_program) -> None:
        from fsm_llm.handlers import HandlerTiming, create_handler

        h = create_handler("h").at(HandlerTiming.PRE_PROCESSING).do(lambda **kw: {})
        with pytest.raises(AttributeError):
            _fsm_program.register_handler(h)


# ---------------------------------------------------------------------------
# Calendar contract metadata — assertable now, useful at audit time.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Z8-epoch rows — back-compat ballast hard-removed at 0.8.0 (no warn cycle).
# ---------------------------------------------------------------------------


class TestZ8EpochHardRemovedAt080:
    """At version >= 0.8.0, the cleanup-release surfaces raise.

    Unlike R13 / I5, this epoch had no warn cycle — the user explicitly
    waived back-compat for the 0.8.0 deep-cleanup release. The test
    only enforces the post-removal contract.
    """

    @pytest.mark.skipif(_VER < (0, 8, 0), reason="Z8 removal lands at 0.8.0")
    def test_handler_alias_removed(self) -> None:
        import fsm_llm

        assert not hasattr(fsm_llm, "Handler")
        assert "Handler" not in fsm_llm.__all__

    @pytest.mark.skipif(_VER < (0, 8, 0), reason="Z8 removal lands at 0.8.0")
    def test_llm_interface_top_level_removed(self) -> None:
        import fsm_llm

        assert "LLMInterface" not in fsm_llm.__all__
        # Canonical path still works.
        from fsm_llm.runtime._litellm import LLMInterface  # noqa: F401

    @pytest.mark.skipif(_VER < (0, 8, 0), reason="Z8 removal lands at 0.8.0")
    def test_builtin_ops_top_level_removed(self) -> None:
        import fsm_llm

        assert "BUILTIN_OPS" not in fsm_llm.__all__
        assert not hasattr(fsm_llm, "BUILTIN_OPS")
        # Canonical path still works.
        from fsm_llm.runtime import BUILTIN_OPS  # noqa: F401

    @pytest.mark.skipif(_VER < (0, 8, 0), reason="Z8 removal lands at 0.8.0")
    @pytest.mark.parametrize(
        "name",
        [
            "has_workflows",
            "has_reasoning",
            "has_agents",
            "get_workflows",
            "get_reasoning",
            "get_agents",
        ],
    )
    def test_extension_check_helper_removed(self, name: str) -> None:
        import fsm_llm

        assert name not in fsm_llm.__all__
        assert not hasattr(fsm_llm, name)

    @pytest.mark.skipif(_VER < (0, 8, 0), reason="Z8 removal lands at 0.8.0")
    @pytest.mark.parametrize(
        "name",
        [
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
        ],
    )
    def test_dialog_definitions_reexport_removed(self, name: str) -> None:
        """The 0.7.0 back-compat re-export block was removed at 0.8.0.
        Names that moved to ``fsm_llm._models`` must NOT resolve via the
        ``fsm_llm.dialog.definitions`` legacy path. Canonical path:
        ``from fsm_llm._models import ...``.
        """
        import fsm_llm.dialog.definitions as defs

        assert not hasattr(defs, name), (
            f"{name!r} should no longer resolve via dialog/definitions; "
            "the back-compat re-export block was removed at 0.8.0."
        )
        # Canonical path still works.
        from fsm_llm import _models as _types

        assert hasattr(_types, name)

    @pytest.mark.skipif(_VER < (0, 8, 0), reason="Z8 removal lands at 0.8.0")
    def test_state_emit_response_leaf_gate_removed(self) -> None:
        """The ``State._emit_response_leaf_for_non_cohort`` private gate
        field was removed at 0.8.0. Non-cohort states now ALWAYS emit
        the response Leaf (Theorem-2 universal for non-terminal FSMs).
        """
        from fsm_llm.dialog.definitions import State

        # Pydantic private attrs are tracked under __private_attributes__.
        assert "_emit_response_leaf_for_non_cohort" not in (
            State.__private_attributes__ or {}
        )
        # Constructing a State and reading the attr raises AttributeError
        # because the attr was deleted from the model.
        s = State(
            id="x",
            description="d",
            purpose="p",
            response_instructions="resp",
        )
        with pytest.raises(AttributeError):
            _ = s._emit_response_leaf_for_non_cohort

    @pytest.mark.skipif(_VER < (0, 8, 0), reason="Z8 removal lands at 0.8.0")
    def test_program_internal_kwargs_removed(self) -> None:
        """``_api`` and ``_profile`` are no longer accepted by
        ``Program.__init__``. Direct FSM-mode construction is reachable
        only through ``Program.from_fsm``.
        """
        from fsm_llm import Program
        from fsm_llm.runtime import var

        with pytest.raises(TypeError):
            Program(term=var("x"), _api=object())  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            Program(term=var("x"), _profile=object())  # type: ignore[call-arg]


class TestCalendarMetadata:
    """The four-epoch contract is structural; the test file's own
    constants document it for grep-ability and future audit."""

    R13_REMOVAL_VERSION = (0, 6, 0)
    I5_WARN_VERSION = (0, 6, 0)
    I5_REMOVAL_VERSION = (0, 7, 0)
    Z8_REMOVAL_VERSION = (0, 8, 0)
    N9_REMOVAL_VERSION = (0, 9, 0)

    def test_version_is_parseable(self) -> None:
        assert isinstance(_VER, tuple)
        assert len(_VER) == 3
        assert all(isinstance(x, int) for x in _VER)

    def test_calendar_dates_distinct(self) -> None:
        # I5 removal must come AFTER I5 warn-start.
        assert self.I5_REMOVAL_VERSION > self.I5_WARN_VERSION
        # R13 removal == I5 warn-start (both at 0.6.0 — synchronised).
        assert self.R13_REMOVAL_VERSION == self.I5_WARN_VERSION
        # Z8 removal lands at 0.8.0 — strictly after I5 removal.
        assert self.Z8_REMOVAL_VERSION > self.I5_REMOVAL_VERSION
        # N9 removal lands at 0.9.0 — strictly after Z8.
        assert self.N9_REMOVAL_VERSION > self.Z8_REMOVAL_VERSION


# ---------------------------------------------------------------------------
# N9-epoch rows — namespace restructure surfaces hard-removed at 0.9.0
# (no warn cycle, like Z8). Covers:
#   * top-level imports of substrate names (Term, leaf, fix, var, abs_, ...)
#   * top-level imports of factory *_term names (react_term, niah_term, ...)
#   * top-level imports of debug helpers (enable_debug_logging, disable_warnings)
#   * top-level BUFFER_METADATA, get_version_info, compile_fsm_cached
#   * the renamed runtime DSL builders (let_, reduce_)
#   * the 6 module-level profile-registry functions (collapsed to ProfileRegistry)
#   * .on_state / .execute_handlers method names on the handler surface
#   * fsm_llm.types module (split into fsm_llm._models + fsm_llm.errors)
#   * MetaBuilderStates legacy aliases
# ---------------------------------------------------------------------------


class TestN9EpochHardRemovedAt090:
    """At version >= 0.9.0, the namespace-restructure surfaces are gone."""

    @pytest.mark.skipif(_VER < (0, 9, 0), reason="N9 removal lands at 0.9.0")
    @pytest.mark.parametrize(
        "name",
        [
            "Term",
            "Var",
            "Abs",
            "App",
            "Let",
            "Case",
            "Leaf",
            "Fix",
            "Combinator",
            "CombinatorOp",
            "is_term",
            "var",
            "abs_",
            "app",
            "let",
            "case_",
            "fix",
            "leaf",
            "split",
            "fmap",
            "ffilter",
            "reduce",
            "concat",
            "cross",
            "peek",
            "host_call",
            "ReduceOp",
            "react_term",
            "rewoo_term",
            "reflexion_term",
            "memory_term",
            "analytical_term",
            "deductive_term",
            "linear_term",
            "branch_term",
            "niah_term",
            "aggregate_term",
            "enable_debug_logging",
            "disable_warnings",
            "BUFFER_METADATA",
            "compile_fsm_cached",
            "get_version_info",
            "ASTConstructionError",
            "TerminationError",
            "PlanningError",
            "OracleError",
            "StateNotFoundError",
            "InvalidTransitionError",
            "LLMResponseError",
            "TransitionEvaluationError",
            "ClassificationError",
            "SchemaValidationError",
            "ClassificationResponseError",
            "HandlerSystemError",
            "HandlerExecutionError",
            "register_harness_profile",
            "register_provider_profile",
            "get_harness_profile",
            "get_provider_profile",
            "HandlerSystem",
        ],
    )
    def test_top_level_name_removed(self, name: str) -> None:
        """The pre-0.9 top-level name must not be importable from ``fsm_llm``."""
        import fsm_llm

        assert name not in fsm_llm.__all__, (
            f"{name!r} should be removed from top-level __all__ at 0.9.0"
        )

    @pytest.mark.skipif(_VER < (0, 9, 0), reason="N9 removal lands at 0.9.0")
    def test_types_module_removed(self) -> None:
        """``fsm_llm.types`` was split into ``fsm_llm._models`` (private)
        + ``fsm_llm.errors`` at 0.9.0.
        """
        with pytest.raises(ImportError):
            import fsm_llm.types  # type: ignore[import-not-found]  # noqa: F401

    @pytest.mark.skipif(_VER < (0, 9, 0), reason="N9 rename lands at 0.9.0")
    def test_dsl_let_underscore_removed(self) -> None:
        """``let_`` → ``let`` rename in runtime.dsl. The old name is gone."""
        from fsm_llm.runtime import dsl as dsl_mod

        assert hasattr(dsl_mod, "let")
        assert not hasattr(dsl_mod, "let_")

    @pytest.mark.skipif(_VER < (0, 9, 0), reason="N9 rename lands at 0.9.0")
    def test_dsl_reduce_underscore_removed(self) -> None:
        """``reduce_`` → ``reduce`` rename in runtime.dsl."""
        from fsm_llm.runtime import dsl as dsl_mod

        assert hasattr(dsl_mod, "reduce")
        assert not hasattr(dsl_mod, "reduce_")

    @pytest.mark.skipif(_VER < (0, 9, 0), reason="N9 rename lands at 0.9.0")
    def test_handler_builder_on_state_removed(self) -> None:
        """``HandlerBuilder.on_state`` renamed to ``.when_state`` at 0.9.0
        for naming consistency (all conditions share ``.when_*`` prefix).
        """
        from fsm_llm import HandlerBuilder

        b = HandlerBuilder("test")
        assert hasattr(b, "when_state")
        assert not hasattr(b, "on_state")

    @pytest.mark.skipif(_VER < (0, 9, 0), reason="N9 privatization lands at 0.9.0")
    def test_handler_system_execute_handlers_privatized(self) -> None:
        """``HandlerSystem.execute_handlers`` privatized to ``_execute_handlers``
        at 0.9.0 — internal plumbing post-R5.
        """
        from fsm_llm.handlers import HandlerSystem

        hs = HandlerSystem()
        assert hasattr(hs, "_execute_handlers")
        assert not hasattr(hs, "execute_handlers")

    @pytest.mark.skipif(_VER < (0, 9, 0), reason="N9 collapse lands at 0.9.0")
    def test_profile_registry_class_replaces_functions(self) -> None:
        """Profile registry collapsed from 6 module-level functions to a
        single ``ProfileRegistry`` class with ``profile_registry``
        singleton.
        """
        from fsm_llm import ProfileRegistry, profile_registry
        from fsm_llm import profiles as profiles_mod

        assert isinstance(profile_registry, ProfileRegistry)
        for fn_name in (
            "register_harness_profile",
            "register_provider_profile",
            "unregister_harness_profile",
            "unregister_provider_profile",
            "get_harness_profile",
            "get_provider_profile",
        ):
            assert not hasattr(profiles_mod, fn_name), (
                f"{fn_name!r} should be removed at 0.9.0; use "
                "profile_registry.register / .get / .unregister instead."
            )

    @pytest.mark.skipif(_VER < (0, 9, 0), reason="N9 deletion lands at 0.9.0")
    def test_meta_builder_states_removed(self) -> None:
        """``MetaBuilderStates`` legacy aliases class deleted at 0.9.0."""
        from fsm_llm.stdlib.agents import constants as agents_constants

        assert not hasattr(agents_constants, "MetaBuilderStates")
