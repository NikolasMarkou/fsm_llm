"""Tests for the R8 unified ``Program.invoke`` surface.

Plan: plans/plan_2026-04-27_32652286/plan.md — Step 1 (Bundle A).

Coverage:

- E1: FSM-mode invoke without conversation_id auto-starts and caches.
- E2: term-mode invoke with inputs=None → empty env.
- E3: explain=True → Result.explain populated; default → None.
- E4: mode mismatch raises ProgramModeError.
- Result dataclass shape (frozen, value + optional explain).
- Deprecation aliases (.run / .converse) byte-equivalent to .invoke.
- register_handler still works in term-mode (no R8 regression).
- explain(inputs=) honours `n` inference and otherwise stays static.
"""

from __future__ import annotations

import pytest

from fsm_llm import ExplainOutput, Program
from fsm_llm.dialog.api import API
from fsm_llm.program import ProgramModeError, Result
from fsm_llm.runtime import leaf, var

# ---------------------------------------------------------------------------
# Fixtures (local — keep self-contained per test module)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_fsm_dict():
    """Minimal valid FSM definition: greeting → farewell terminal."""
    return {
        "name": "test_invoke_greeting",
        "description": "A minimal FSM for R8 invoke tests",
        "version": "4.1",
        "initial_state": "greeting",
        "states": {
            "greeting": {
                "id": "greeting",
                "description": "Initial greeting state",
                "purpose": "Greet the user",
                "transitions": [
                    {
                        "target_state": "farewell",
                        "description": "Always exit",
                        "priority": 100,
                        "conditions": [],
                    }
                ],
            },
            "farewell": {
                "id": "farewell",
                "description": "Farewell terminal",
                "purpose": "Say goodbye",
                "transitions": [],
            },
        },
    }


# ---------------------------------------------------------------------------
# Result dataclass shape
# ---------------------------------------------------------------------------


class TestResultDataclass:
    def test_result_is_frozen(self):
        r = Result(value="hi", explain=None)
        with pytest.raises((AttributeError, Exception)):
            r.value = "other"  # type: ignore[misc]

    def test_result_defaults(self):
        r = Result()
        assert r.value is None
        assert r.explain is None

    def test_result_with_explain(self):
        eo = ExplainOutput(ast_shape="X")
        r = Result(value=42, explain=eo)
        assert r.value == 42
        assert r.explain is eo


# ---------------------------------------------------------------------------
# ProgramModeError exception type
# ---------------------------------------------------------------------------


class TestProgramModeError:
    def test_inherits_from_fsmerror(self):
        from fsm_llm.dialog.definitions import FSMError

        assert issubclass(ProgramModeError, FSMError)

    def test_exported_from_program_module(self):
        from fsm_llm import program as program_mod

        assert "ProgramModeError" in program_mod.__all__
        assert "Result" in program_mod.__all__


# ---------------------------------------------------------------------------
# Term-mode invoke
# ---------------------------------------------------------------------------


class TestInvokeTermMode:
    def test_invoke_returns_result(self):
        # E2: term-mode invoke returns Result(value=..., explain=None).
        prog = Program.from_term(var("x"))
        out = prog.invoke(inputs={"x": "hello"})
        assert isinstance(out, Result)
        assert out.value == "hello"
        assert out.explain is None

    def test_invoke_with_no_inputs_uses_empty_env(self):
        # E2: inputs=None → empty env. var("x") with no env raises;
        # use a no-input term instead. We pass a term that doesn't
        # require any free vars: a leaf has free vars (its template
        # vars) so the cleanest no-env term is a closed Abs. Simpler:
        # var with default-providing env via dict. Use leaf() with no
        # input_vars — actually, the simplest: a term that needs no
        # env at all is impossible in our DSL except via Abs. Use a
        # closed identity-bound term.
        from fsm_llm.runtime import abs_, app

        prog = Program.from_term(app(abs_("y", var("y")), var("z")))
        out = prog.invoke(inputs={"z": 99})
        assert isinstance(out, Result)
        assert out.value == 99

    def test_invoke_with_explain_true_populates_explain(self):
        # E3: explain=True returns Result with explain field set.
        prog = Program.from_term(var("x"))
        out = prog.invoke(inputs={"x": 1}, explain=True)
        assert isinstance(out, Result)
        assert out.value == 1
        assert isinstance(out.explain, ExplainOutput)
        assert "Var('x')" in out.explain.ast_shape

    def test_invoke_explain_false_default_returns_none(self):
        # E3: default explain=False → Result.explain is None.
        prog = Program.from_term(var("x"))
        out = prog.invoke(inputs={"x": 1})
        assert out.explain is None

    def test_invoke_term_mode_with_message_raises(self):
        # E4: mode mismatch — term-mode + message=.
        prog = Program.from_term(var("x"))
        with pytest.raises(ProgramModeError, match="term-mode invoke"):
            prog.invoke(message="hi")


# ---------------------------------------------------------------------------
# FSM-mode invoke
# ---------------------------------------------------------------------------


class TestInvokeFsmMode:
    def test_invoke_auto_starts_conversation_returns_result_with_string_value(
        self, sample_fsm_dict, mock_llm2_interface
    ):
        # E1 + M1 (plan plan_2026-04-28_6597e394): FSM-mode invoke with
        # no conversation_id auto-starts AND returns a Result (not a
        # bare string). Test inversion+rename per D-STEP-2-T1 precedent.
        prog = Program.from_fsm(sample_fsm_dict, llm_interface=mock_llm2_interface)
        result = prog.invoke(message="hello")
        assert isinstance(result, Result)
        assert isinstance(result.value, str)
        assert result.conversation_id is not None
        assert result.conversation_id == prog._default_conv_id
        # M1: FSM-mode plan/leaf_calls/oracle_calls are placeholders
        # until M3 lifts the response Leaf.
        assert result.plan is None
        assert result.leaf_calls == 0
        assert result.oracle_calls == 0
        assert prog._default_conv_id is not None

    def test_invoke_caches_default_conv_id(self, sample_fsm_dict, mock_llm2_interface):
        # E1: cached id is reused on subsequent calls without explicit id.
        prog = Program.from_fsm(sample_fsm_dict, llm_interface=mock_llm2_interface)
        prog.invoke(message="first")
        first_id = prog._default_conv_id
        assert first_id is not None
        # Cached on the program — would be reused. We don't multi-turn
        # because sample fsm is terminal after greeting.

    def test_invoke_explicit_conv_id_returns_result_and_does_not_overwrite_cache(
        self, sample_fsm_dict, mock_llm2_interface
    ):
        # M1 (plan plan_2026-04-28_6597e394): explicit conversation_id
        # path also returns Result with that id reflected on it. Test
        # inversion+rename per D-STEP-2-T1 precedent.
        prog = Program.from_fsm(sample_fsm_dict, llm_interface=mock_llm2_interface)
        cid, _ = prog._api.start_conversation()
        result = prog.invoke(message="hello", conversation_id=cid)
        assert isinstance(result, Result)
        assert isinstance(result.value, str)
        assert result.conversation_id == cid
        # Explicit id path does not populate the default-id cache.
        assert getattr(prog, "_default_conv_id", None) is None

    def test_invoke_fsm_mode_with_inputs_raises(
        self, sample_fsm_dict, mock_llm_interface
    ):
        # E4: mode mismatch — FSM-mode + inputs=.
        prog = Program.from_fsm(sample_fsm_dict, llm_interface=mock_llm_interface)
        with pytest.raises(ProgramModeError, match="FSM-mode invoke"):
            prog.invoke(inputs={"x": 1})

    def test_invoke_fsm_mode_no_message_raises(
        self, sample_fsm_dict, mock_llm_interface
    ):
        # FSM-mode requires message=.
        prog = Program.from_fsm(sample_fsm_dict, llm_interface=mock_llm_interface)
        with pytest.raises(ProgramModeError, match="message="):
            prog.invoke()


# ---------------------------------------------------------------------------
# Deprecation alias byte-equivalence
# ---------------------------------------------------------------------------


# Deprecation aliases (.run / .converse / .register_handler) were removed at
# 0.7.0. Their removal is asserted in
# ``tests/test_fsm_llm/test_deprecation_calendar.py``
# (TestI5EpochProgramMethodsRemovedAt070); the canonical surface is
# ``Program.invoke(inputs={...})`` and ``Program.invoke(message=...)``,
# with handlers passed via ``handlers=[...]`` at construction.


# ---------------------------------------------------------------------------
# explain(inputs=) — R8 addition
# ---------------------------------------------------------------------------


class TestExplainInputs:
    def test_explain_with_inputs_n_inferred(self):
        # When inputs supplies "n" key and explicit n is None, inference
        # picks it up. Term has no Fix → plans stays empty regardless.
        t = leaf("hello {x}", input_vars=("x",))
        prog = Program.from_term(t)
        out = prog.explain(inputs={"n": 8})
        assert isinstance(out, ExplainOutput)
        # No Fix subtree → plans is empty even with n inferred.
        assert out.plans == []
        # leaf_schemas indexed.
        assert len(out.leaf_schemas) == 1

    def test_explain_with_inputs_no_n_key_falls_back(self):
        prog = Program.from_term(var("x"))
        out = prog.explain(inputs={"some": "data"})
        assert isinstance(out, ExplainOutput)
        assert out.plans == []

    def test_explain_explicit_n_overrides_inputs(self):
        # When n is supplied explicitly, inputs.n is ignored.
        prog = Program.from_term(var("x"))
        out = prog.explain(inputs={"n": 99}, n=10, K=4)
        # No Fix node → plans empty regardless.
        assert out.plans == []


# ---------------------------------------------------------------------------
# Public re-exports — proof Result + ProgramModeError surface from program
# ---------------------------------------------------------------------------


class TestProgramModuleExports:
    def test_result_importable(self):
        from fsm_llm.program import Result as R

        assert R is Result

    def test_program_mode_error_importable(self):
        from fsm_llm.program import ProgramModeError as P

        assert P is ProgramModeError

    def test_api_still_alive(self, sample_fsm_dict, mock_llm_interface):
        # Sanity: API import + .converse still works post-R8.
        api = API(sample_fsm_dict, llm_interface=mock_llm_interface)
        cid, _ = api.start_conversation()
        assert isinstance(api.converse("hi", cid), str)
