"""Deprecation-calendar enforcement test (M5 + M6c).

Per ``docs/lambda_fsm_merge.md`` §3 I5 (two-epoch reconciliation) and §6 G4.

The merge spec defines two parallel deprecation calendars:

- **R13 epoch** — `lam`, `dialog/{api,fsm,pipeline,prompts,classification,
  transition_evaluator,definitions,session}`, `llm`, and `dialog/pipeline`
  module shims. Already emit ``DeprecationWarning`` at HEAD (0.3.0); will be
  REMOVED at 0.6.0 per the warning text in `lam/__init__.py:34-47`.
- **I5 epoch** — `Program.run`, `Program.converse`, `Program.register_handler`,
  `from fsm_llm import API`, `import fsm_llm_{reasoning,workflows,agents}`.
  Silent at HEAD (0.3.0); will WARN at 0.6.0; REMOVED at 0.7.0.

This test reads ``fsm_llm.__version__`` and asserts the appropriate epoch
behaviour for each row. Future-version assertions (>=0.6.0, >=0.7.0) are
gated via ``pytest.skipif`` and will activate at the next two version bumps.

Plan: plans/plan_2026-04-28_f1003066/ — Step 4.
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
    """At version < 0.6.0, the four I5 import-shape rows are silent."""

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
    from fsm_llm.lam import var

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

    @pytest.mark.skipif(
        _VER >= (0, 6, 0), reason="I5 method aliases warn at 0.6.0"
    )
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

    @pytest.mark.skipif(
        _VER >= (0, 6, 0), reason="I5 method aliases warn at 0.6.0"
    )
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

    @pytest.mark.skipif(
        _VER >= (0, 6, 0), reason="I5 method aliases warn at 0.6.0"
    )
    def test_program_register_handler_silent_pre_060(self, _fsm_program) -> None:
        from fsm_llm.handlers import HandlerTiming, create_handler

        h = (
            create_handler("h")
            .at(HandlerTiming.PRE_PROCESSING)
            .do(lambda **kw: {})
        )
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
# Calendar contract metadata — assertable now, useful at audit time.
# ---------------------------------------------------------------------------


class TestCalendarMetadata:
    """The two-epoch contract is structural; the test file's own constants
    document it for grep-ability and future audit."""

    R13_REMOVAL_VERSION = (0, 6, 0)
    I5_WARN_VERSION = (0, 6, 0)
    I5_REMOVAL_VERSION = (0, 7, 0)

    def test_version_is_parseable(self) -> None:
        assert isinstance(_VER, tuple)
        assert len(_VER) == 3
        assert all(isinstance(x, int) for x in _VER)

    def test_calendar_dates_distinct(self) -> None:
        # I5 removal must come AFTER I5 warn-start.
        assert self.I5_REMOVAL_VERSION > self.I5_WARN_VERSION
        # R13 removal == I5 warn-start (both at 0.6.0 — synchronised).
        assert self.R13_REMOVAL_VERSION == self.I5_WARN_VERSION
