"""Oracle ownership tests (M4 — merge spec §3 I1+I2 / G2 gate).

Two flavours of assertion:

1. **AST-walk gate (G2)** — ``dialog/turn.py`` has zero ``LiteLLMOracle(...)``
   constructor calls in the message-processing hot path (i.e. anywhere
   except ``MessagePipeline.__init__``'s back-compat default-fallback).
   The fallback exists for tests that construct ``MessagePipeline``
   directly with only ``llm_interface``; production callers (FSMManager,
   API, Program) always thread an explicit oracle through, so the
   fallback path is never taken in the wire/dispatch hot path.

2. **Identity propagation** — when ``Program.from_fsm(defn, oracle=O)``
   is called, the same Oracle instance ``O`` is observable at every
   layer of the dispatch chain: ``Program._oracle is api._oracle is
   fsm_manager._oracle is pipeline._oracle``. This is the structural
   invariant that "Program owns one Oracle" depends on.

Both tests run as fast unit tests in the regular
``pytest -m 'not slow and not real_llm'`` collection. The AST-walk
test is pure static (no LLM, no network); the identity test
constructs a Program against a tiny FSM definition with a mock
``LLMInterface`` (no LLM call is actually issued — just construction
checks).
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import Mock

import pytest

import fsm_llm
from fsm_llm import LLMInterface, Program
from fsm_llm.runtime.oracle import LiteLLMOracle, Oracle

PKG_ROOT: Path = Path(fsm_llm.__file__).parent
TURN_PY: Path = PKG_ROOT / "dialog" / "turn.py"


# ---------------------------------------------------------------------------
# AST-walk gate (G2 — zero LiteLLMOracle constructions in the hot path)
# ---------------------------------------------------------------------------


def _find_litellmoracle_calls(tree: ast.AST) -> list[tuple[int, str]]:
    """Return (lineno, parent-function-name) for every ``LiteLLMOracle(...)``
    or ``_LiteLLMOracle(...)`` Call node found.

    The walker keeps a stack of enclosing function names so failures
    point back to the offending method.
    """
    out: list[tuple[int, str]] = []

    class _V(ast.NodeVisitor):
        def __init__(self) -> None:
            self._stack: list[str] = []

        def _enter(self, name: str) -> None:
            self._stack.append(name)

        def _leave(self) -> None:
            self._stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._enter(node.name)
            self.generic_visit(node)
            self._leave()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._enter(node.name)
            self.generic_visit(node)
            self._leave()

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            name: str | None = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in {"LiteLLMOracle", "_LiteLLMOracle"}:
                parent = self._stack[-1] if self._stack else "<module>"
                out.append((node.lineno, parent))
            self.generic_visit(node)

    _V().visit(tree)
    return out


class TestG2Gate:
    """G2 — dialog/turn.py message-processing hot path is Oracle-construction-free."""

    def test_no_litellmoracle_call_outside_init(self) -> None:
        """Every ``LiteLLMOracle(...)`` call in turn.py must live inside
        ``MessagePipeline.__init__`` (the back-compat default-fallback).

        Pre-M4: 7 calls scattered across hot-path methods. Post-M4: at
        most 1 call, inside ``__init__``. Any call inside any other
        method is a regression of the G2 gate.
        """
        tree = ast.parse(TURN_PY.read_text(encoding="utf-8"))
        calls = _find_litellmoracle_calls(tree)
        offenders = [(ln, fn) for (ln, fn) in calls if fn != "__init__"]
        assert not offenders, (
            "G2 violation — LiteLLMOracle(...) construction found outside "
            "MessagePipeline.__init__ in dialog/turn.py:\n"
            + "\n".join(f"  line {ln}: in function `{fn}`" for ln, fn in offenders)
            + "\n\nFix: replace with `self._oracle` field-read."
        )

    def test_no_local_litellmoracle_imports_outside_init(self) -> None:
        """Function-scoped ``from ..runtime.oracle import LiteLLMOracle``
        imports must only appear inside ``MessagePipeline.__init__``.

        Each pre-M4 call site had a paired function-scoped import; the
        field-read rewrite removes both. Asserting this together with
        the call-site count prevents partial migrations.

        Module-level (top-of-file) imports — including the TYPE_CHECKING
        guard for type annotations — are explicitly allowed. The check
        targets only imports nested inside non-``__init__`` functions.
        """
        tree = ast.parse(TURN_PY.read_text(encoding="utf-8"))
        offenders: list[tuple[int, str]] = []

        class _V(ast.NodeVisitor):
            def __init__(self) -> None:
                self._stack: list[str] = []

            def visit_FunctionDef(self, n: ast.FunctionDef) -> None:
                self._stack.append(n.name)
                self.generic_visit(n)
                self._stack.pop()

            def visit_AsyncFunctionDef(self, n: ast.AsyncFunctionDef) -> None:
                self._stack.append(n.name)
                self.generic_visit(n)
                self._stack.pop()

            def visit_ImportFrom(self, n: ast.ImportFrom) -> None:
                if (
                    n.module
                    and n.module.endswith("runtime.oracle")
                    and any(
                        a.name == "LiteLLMOracle" or a.asname == "_LiteLLMOracle"
                        for a in n.names
                    )
                ):
                    parent = self._stack[-1] if self._stack else "<module>"
                    # Allow module-level imports (including TYPE_CHECKING).
                    # Only function-scoped imports outside __init__ are
                    # the regression we're guarding against.
                    if parent != "__init__" and parent != "<module>":
                        offenders.append((n.lineno, parent))
                self.generic_visit(n)

        _V().visit(tree)
        assert not offenders, (
            "G2 violation — local `from ..runtime.oracle import LiteLLMOracle` "
            "found outside MessagePipeline.__init__ in dialog/turn.py:\n"
            + "\n".join(f"  line {ln}: in function `{fn}`" for ln, fn in offenders)
        )

    def test_messagepipeline_init_has_oracle_param(self) -> None:
        """The merge contract requires ``MessagePipeline.__init__`` to
        accept an ``oracle:`` keyword argument so the Program-owned
        Oracle can be threaded through.
        """
        tree = ast.parse(TURN_PY.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "MessagePipeline":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        names = [a.arg for a in item.args.args] + [
                            a.arg for a in item.args.kwonlyargs
                        ]
                        assert "oracle" in names, (
                            "MessagePipeline.__init__ must accept an `oracle:` "
                            f"kwarg. Found args: {names}"
                        )
                        return
        pytest.fail("MessagePipeline.__init__ not found in dialog/turn.py")


# ---------------------------------------------------------------------------
# Identity propagation
# ---------------------------------------------------------------------------


def _make_min_fsm_dict() -> dict:
    """Smallest valid FSM JSON for construction-only tests."""
    return {
        "name": "OracleIdentityTest",
        "description": "Two-state FSM for oracle-identity propagation tests.",
        "version": "4.1",
        "initial_state": "start",
        "persona": "test",
        "states": {
            "start": {
                "id": "start",
                "description": "start",
                "purpose": "test",
                "extraction_instructions": "",
                "response_instructions": "say hi",
                "transitions": [
                    {
                        "target_state": "end",
                        "description": "always",
                        "priority": 100,
                        "conditions": [],
                    }
                ],
                "required_context_keys": [],
                "field_extractions": [],
                "classification_extractions": [],
            },
            "end": {
                "id": "end",
                "description": "end",
                "purpose": "test",
                "extraction_instructions": "",
                "response_instructions": "say bye",
                "transitions": [],
                "required_context_keys": [],
                "field_extractions": [],
                "classification_extractions": [],
            },
        },
    }


class TestOracleIdentityPropagation:
    """Program owns exactly one Oracle; the same instance is observable
    at every layer of the dispatch chain."""

    def test_identity_propagates_program_to_pipeline(self) -> None:
        """Constructing Program.from_fsm(..., oracle=O) puts the same O
        on Program._oracle, API._oracle, FSMManager._oracle, and
        MessagePipeline._oracle.

        This is the structural invariant of "Program owns one Oracle"
        (merge spec §3 I2). Any layer that wraps or replaces the
        Oracle instance breaks the contract.
        """
        mock_llm = Mock(spec=LLMInterface)
        mock_llm.model = "mock-model"
        oracle = LiteLLMOracle(mock_llm)

        program = Program.from_fsm(_make_min_fsm_dict(), oracle=oracle)

        # Every layer holds the same instance.
        assert program._oracle is oracle, "Program._oracle is not the supplied Oracle"
        assert program._api is not None, "FSM-mode Program must have ._api"
        assert program._api._oracle is oracle, (
            "API._oracle diverged from the supplied Oracle"
        )
        assert program._api.fsm_manager._oracle is oracle, (
            "FSMManager._oracle diverged from the supplied Oracle"
        )
        assert program._api.fsm_manager._pipeline._oracle is oracle, (
            "MessagePipeline._oracle diverged from the supplied Oracle"
        )

    def test_default_oracle_propagates_when_none_supplied(self) -> None:
        """When ``oracle=`` is not supplied to ``API``, the API constructs
        one default LiteLLMOracle and threads it through. The same
        instance must reach FSMManager and MessagePipeline.
        """
        mock_llm = Mock(spec=LLMInterface)
        mock_llm.model = "mock-model"

        # Construct API directly (skipping Program) to exercise the
        # default-oracle branch on the API side.
        from fsm_llm.dialog.api import API

        api = API(_make_min_fsm_dict(), llm_interface=mock_llm)

        assert isinstance(api._oracle, Oracle), (
            "API._oracle must satisfy the Oracle protocol"
        )
        # Single instance threaded down.
        assert api.fsm_manager._oracle is api._oracle, (
            "FSMManager._oracle diverged from API._oracle"
        )
        assert api.fsm_manager._pipeline._oracle is api._oracle, (
            "MessagePipeline._oracle diverged from API._oracle"
        )
