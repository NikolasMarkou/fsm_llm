from __future__ import annotations

"""S1 scaffold tests for fsm_compile.

S1 only verifies that the module exists, exports the expected names, and
rejects malformed inputs. The body of ``compile_fsm`` raises
``NotImplementedError`` until S2 lands the base case.
"""

import pytest

from fsm_llm.definitions import FSMDefinition
from fsm_llm.lam import compile_fsm
from fsm_llm.lam import fsm_compile as fsc
from fsm_llm.lam.errors import ASTConstructionError


def _greeter_fsm_dict() -> dict:
    """Single-state FSM, no transitions — used across fsm_compile tests."""
    return {
        "name": "greeter",
        "description": "one-state greeter for compiler tests",
        "initial_state": "hello",
        "persona": "friendly test bot",
        "states": {
            "hello": {
                "id": "hello",
                "description": "say hi",
                "purpose": "greet",
                "response_instructions": "Say hello.",
                "transitions": [],
            },
        },
    }


def _two_state_fsm_dict() -> dict:
    """Minimal 2-state FSM with a single deterministic transition."""
    return {
        "name": "two_state",
        "description": "start → end",
        "initial_state": "start",
        "persona": "test",
        "states": {
            "start": {
                "id": "start",
                "description": "begin",
                "purpose": "start",
                "response_instructions": "Say hello.",
                "transitions": [
                    {
                        "target_state": "end",
                        "description": "always advance",
                        "conditions": [
                            {"description": "always", "logic": {"==": [1, 1]}}
                        ],
                    }
                ],
            },
            "end": {
                "id": "end",
                "description": "done",
                "purpose": "end",
                "response_instructions": "Say goodbye.",
                "transitions": [],
            },
        },
    }


class TestScaffold:
    def test_symbol_exported_from_package(self) -> None:
        from fsm_llm.lam import compile_fsm as exported  # noqa: F401

    def test_reserved_var_names_frozenset(self) -> None:
        # Contract with S8: pipeline must bind exactly these names in env.
        assert fsc.VAR_STATE_ID in fsc.RESERVED_VARS
        assert fsc.VAR_MESSAGE in fsc.RESERVED_VARS
        assert fsc.VAR_CONV_ID in fsc.RESERVED_VARS
        assert fsc.VAR_INSTANCE in fsc.RESERVED_VARS
        for cb in (
            fsc.CB_EXTRACT,
            fsc.CB_FIELD_EXTRACT,
            fsc.CB_CLASS_EXTRACT,
            fsc.CB_EVAL_TRANSIT,
            fsc.CB_RESOLVE_AMBIG,
            fsc.CB_TRANSIT,
            fsc.CB_RESPOND,
        ):
            assert cb in fsc.RESERVED_VARS
        assert isinstance(fsc.RESERVED_VARS, frozenset)

    def test_gensym_monotone(self) -> None:
        ctx = fsc._CompileCtx(fsm_name="t")
        a = ctx.gensym("x")
        b = ctx.gensym("x")
        assert a != b
        assert a.startswith("__x_")
        assert b.startswith("__x_")


class TestCompileFsmStub:
    def test_raises_not_implemented_for_valid_fsm(self) -> None:
        # S1 scaffold: the body of compile_fsm is not yet implemented.
        # S2 removes this NotImplementedError.
        defn = FSMDefinition.model_validate(_greeter_fsm_dict())
        with pytest.raises(NotImplementedError):
            compile_fsm(defn)

    def test_rejects_empty_states(self) -> None:
        # FSMDefinition.validate blocks empty states at model-validation
        # time, so we can't construct one via model_validate. We exercise
        # the compile-time guard by bypassing construction.
        defn = FSMDefinition.model_validate(_greeter_fsm_dict())
        object.__setattr__(defn, "states", {})
        with pytest.raises(ASTConstructionError, match="no states"):
            compile_fsm(defn)


@pytest.mark.xfail(reason="S2 will implement the base case", strict=True)
class TestCompileBaseCase:
    """Red tests for S2. Each assertion describes the S2 contract.

    Marked strict=True so S2 lands green and pytest flips these to pass;
    if S2 regresses any of them we fail loud.
    """

    def test_trivial_fsm_returns_term_shape(self) -> None:
        from fsm_llm.lam.ast import Abs, Case

        defn = FSMDefinition.model_validate(_greeter_fsm_dict())
        term = compile_fsm(defn)
        # Outermost: λ state_id. λ message. λ conv_id. λ instance. <body>
        assert isinstance(term, Abs) and term.param == fsc.VAR_STATE_ID
        inner1 = term.body
        assert isinstance(inner1, Abs) and inner1.param == fsc.VAR_MESSAGE
        inner2 = inner1.body
        assert isinstance(inner2, Abs) and inner2.param == fsc.VAR_CONV_ID
        inner3 = inner2.body
        assert isinstance(inner3, Abs) and inner3.param == fsc.VAR_INSTANCE
        body = inner3.body
        assert isinstance(body, Case)
        assert set(body.branches.keys()) == {"hello"}

    def test_two_state_fsm_has_two_branches(self) -> None:
        from fsm_llm.lam.ast import Abs, Case

        defn = FSMDefinition.model_validate(_two_state_fsm_dict())
        term = compile_fsm(defn)
        assert isinstance(term, Abs)
        # Walk four Abs layers to reach the Case.
        body = term.body.body.body.body
        assert isinstance(body, Case)
        assert set(body.branches.keys()) == {"start", "end"}
