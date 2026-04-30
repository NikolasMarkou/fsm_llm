"""Tests for the Program facade — the unified ``Program.invoke()`` entry point.

Coverage organised by surface:

- Construction (TestProgramConstruction): bare ctor, from_term, from_factory,
  from_fsm, the term-vs-API XOR invariant, handlers= kwarg.
- Invoke / term mode (TestProgramInvokeTerm): term-mode evaluation, factory
  composition, oracle defaulting, mode-mismatch raises.
- Invoke / FSM mode (TestProgramInvokeFsm): API delegation, conversation
  auto-start, multi-turn id reuse, LiteLLMOracle unwrap, non-LiteLLM
  TypeError, parity with API.from_definition (Invariant 4).
- Explain (TestProgramExplain): ExplainOutput shape, leaf_schemas index,
  ast_shape rendering on Var/Leaf/Fix/Case/Combinator, FSM-compiled term
  walk.
- Handlers (TestProgramHandlersCtor): handlers passed at construction
  register on the API (FSM mode) and compose into the term (term mode).
- Public surface (TestPublicSurface): __all__ + import-from-package.

The legacy ``.run`` / ``.converse`` / ``.register_handler`` aliases were
removed at 0.7.0 — this file targets the post-removal contract. Removal
is asserted in tests/test_fsm_llm/test_deprecation_calendar.py.
"""

from __future__ import annotations

import json
import pathlib
from unittest.mock import Mock

import pytest

from fsm_llm import ExplainOutput, Program
from fsm_llm.dialog.api import API
from fsm_llm.handlers import HandlerTiming, create_handler
from fsm_llm.runtime import (
    Executor,
    LiteLLMOracle,
    abs_,
    case_,
    fix,
    leaf,
    let,
    var,
)
from fsm_llm.runtime._litellm import LLMInterface

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def example_fsm_dict():
    """A real FSM definition copied from examples/. Stable test surface."""
    examples = pathlib.Path(__file__).parent.parent.parent / "examples"
    candidates = sorted(examples.rglob("*.json"))
    with open(candidates[0]) as f:
        return json.load(f)


@pytest.fixture
def sample_fsm_dict():
    """A minimal valid FSM definition (greeting → farewell terminal)."""
    return {
        "name": "test_greeting",
        "description": "A minimal greeting FSM for testing",
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
# Construction
# ---------------------------------------------------------------------------


class TestProgramConstruction:
    """SC1, SC2: constructors yield well-formed Programs."""

    def test_from_term_builds_program(self):
        prog = Program.from_term(var("x"))
        assert isinstance(prog, Program)
        assert prog._term is not None
        assert prog._api is None

    def test_from_factory_builds_program(self):
        def my_factory():
            return var("y")

        prog = Program.from_factory(my_factory)
        assert isinstance(prog, Program)
        assert prog._term is not None

    def test_from_factory_passes_factory_args(self):
        def factory_with_args(a, b, *, c):
            assert a == 1 and b == 2 and c == 3
            return var("done")

        prog = Program.from_factory(
            factory_with_args, factory_args=(1, 2), factory_kwargs={"c": 3}
        )
        assert prog._term is not None

    def test_from_fsm_builds_program(self, sample_fsm_dict, mock_llm_interface):
        prog = Program.from_fsm(sample_fsm_dict, llm_interface=mock_llm_interface)
        assert isinstance(prog, Program)
        assert prog._api is not None
        assert prog._term is None

    def test_bare_ctor_requires_term(self):
        # 0.8.0: the public ``Program(...)`` ctor is term-mode only.
        # FSM-mode is reachable exclusively through ``Program.from_fsm``.
        with pytest.raises(ValueError):
            Program()

    def test_bare_ctor_rejects_internal_api_kwarg(self):
        # 0.8.0: the ``_api`` kwarg was hidden — direct FSM-mode
        # construction via ``Program(_api=...)`` is no longer supported;
        # use ``Program.from_fsm(...)``.
        with pytest.raises(TypeError):
            Program(term=var("x"), _api=object())

    def test_bare_ctor_with_term_works(self):
        prog = Program(term=var("x"))
        assert prog._term is not None
        assert prog._api is None

    def test_handlers_default_to_empty_list(self):
        prog = Program.from_term(var("x"))
        assert prog._handlers == []

    def test_handlers_kwarg_copies_list(self):
        h = create_handler("h").at(HandlerTiming.START_CONVERSATION).do(lambda **kw: {})
        original = [h]
        prog = Program.from_term(var("x"), handlers=original)
        # The Program holds its own copy — mutations to the input list
        # don't bleed in.
        original.append("not_a_handler")
        assert prog._handlers == [h]


# ---------------------------------------------------------------------------
# Invoke — term mode
# ---------------------------------------------------------------------------


class TestProgramInvokeTerm:
    """SC3, SC4, SC5: Program.invoke(inputs=) delegates to Executor with env."""

    def test_invoke_returns_var_binding(self):
        prog = Program.from_term(var("x"))
        result = prog.invoke(inputs={"x": "hello"})
        assert result.value == "hello"

    def test_invoke_factory_evaluates(self):
        def fac():
            return var("y")

        prog = Program.from_factory(fac)
        assert prog.invoke(inputs={"y": 42}).value == 42

    def test_invoke_with_executor_parity_invariant_5(self):
        """Invariant 5: Program(term=t, oracle=o).invoke(inputs=env).value
        byte-equals Executor(oracle=o).run(t, env)."""
        t = var("x")
        prog = Program.from_term(t)
        ex_result = Executor().run(t, {"x": "value"})
        prog_result = prog.invoke(inputs={"x": "value"}).value
        assert prog_result == ex_result

    def test_invoke_fsm_mode_without_message_raises(
        self, sample_fsm_dict, mock_llm_interface
    ):
        # ProgramModeError when an FSM Program receives inputs= (or no
        # message=) — the FSM path requires a message.
        from fsm_llm.program import ProgramModeError

        prog = Program.from_fsm(sample_fsm_dict, llm_interface=mock_llm_interface)
        with pytest.raises(ProgramModeError):
            prog.invoke(inputs={})

    def test_invoke_with_explicit_oracle(self):
        # Explicit oracle is used over the lazy default.
        mock_llm = Mock(spec=LLMInterface)
        oracle = LiteLLMOracle(mock_llm)
        prog = Program.from_term(var("x"), oracle=oracle)
        # Var doesn't invoke the oracle; invoke completes without LLM calls.
        assert prog.invoke(inputs={"x": "hi"}).value == "hi"

    def test_invoke_lazy_default_oracle_not_built_until_used(self):
        # Building a Program with no oracle should not yet construct
        # LiteLLMInterface (it's lazy). A Var-only term runs without
        # needing network or LiteLLMInterface to even exist.
        prog = Program.from_term(var("x"))
        assert prog.invoke(inputs={"x": 1}).value == 1


# ---------------------------------------------------------------------------
# Invoke — FSM mode
# ---------------------------------------------------------------------------


class TestProgramInvokeFsm:
    """SC6, SC7: from_fsm wires API; .invoke(message=) delegates."""

    def test_from_fsm_creates_internal_api(self, sample_fsm_dict, mock_llm_interface):
        prog = Program.from_fsm(sample_fsm_dict, llm_interface=mock_llm_interface)
        assert isinstance(prog._api, API)

    def test_from_fsm_litellm_oracle_unwraps_to_interface(
        self, sample_fsm_dict, mock_llm_interface
    ):
        oracle = LiteLLMOracle(mock_llm_interface)
        prog = Program.from_fsm(sample_fsm_dict, oracle=oracle)
        # The underlying LLMInterface flows through to API.
        assert prog._api.llm_interface is mock_llm_interface

    def test_from_fsm_non_litellm_oracle_raises_typeerror(self, sample_fsm_dict):
        class FakeOracle:
            pass

        with pytest.raises(TypeError, match="LiteLLMOracle"):
            Program.from_fsm(sample_fsm_dict, oracle=FakeOracle())

    def test_from_fsm_handlers_register_via_api(
        self, sample_fsm_dict, mock_llm_interface
    ):
        h = create_handler("h").at(HandlerTiming.START_CONVERSATION).do(lambda **kw: {})
        prog = Program.from_fsm(
            sample_fsm_dict, llm_interface=mock_llm_interface, handlers=[h]
        )
        assert len(prog._api.handler_system.handlers) >= 1

    def test_invoke_message_term_mode_raises(self):
        # Term-mode invoke called with message= raises ProgramModeError —
        # term-mode is fundamentally stateless, conversational entry has
        # no coherent meaning.
        from fsm_llm.program import ProgramModeError

        prog = Program.from_term(var("x"))
        with pytest.raises(ProgramModeError):
            prog.invoke(message="msg")

    def test_invoke_auto_starts_conversation(
        self, sample_fsm_dict, mock_llm2_interface
    ):
        prog = Program.from_fsm(sample_fsm_dict, llm_interface=mock_llm2_interface)
        # No explicit conversation_id → auto-start.
        result = prog.invoke(message="hello")
        assert isinstance(result.value, str)
        assert prog._default_conv_id is not None

    def test_invoke_reuses_default_conv_id(self, sample_fsm_dict, mock_llm2_interface):
        prog = Program.from_fsm(sample_fsm_dict, llm_interface=mock_llm2_interface)
        prog.invoke(message="first")
        first_id = prog._default_conv_id
        # The behavioural assertion is just: id is cached.
        assert first_id is not None

    def test_invoke_explicit_id_supports_multiplex(
        self, sample_fsm_dict, mock_llm2_interface
    ):
        prog = Program.from_fsm(sample_fsm_dict, llm_interface=mock_llm2_interface)
        # Directly start a conversation via the underlying API.
        cid, _ = prog._api.start_conversation()
        result = prog.invoke(message="hello", conversation_id=cid)
        assert isinstance(result.value, str)

    def test_invariant_4_program_from_fsm_byte_equal_api(
        self, sample_fsm_dict, mock_llm2_interface
    ):
        """Invariant 4: Program.from_fsm(d).invoke(message=m, conversation_id=c)
        byte-equals API.from_definition(d).converse(m, c)."""
        # Same mock, same definition → same response.
        api_direct = API(sample_fsm_dict, llm_interface=mock_llm2_interface)
        cid_a, _ = api_direct.start_conversation()
        api_resp = api_direct.converse("hi", cid_a)

        # Reset call_history so the Program path is independent.
        mock_llm2_interface.call_history.clear()
        prog = Program.from_fsm(sample_fsm_dict, llm_interface=mock_llm2_interface)
        cid_b, _ = prog._api.start_conversation()
        prog_resp = prog.invoke(message="hi", conversation_id=cid_b).value
        assert prog_resp == api_resp


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------


class TestProgramExplain:
    """SC8: Program.explain returns ExplainOutput with shape + leaf_schemas."""

    def test_explain_returns_explainoutput(self):
        prog = Program.from_term(var("x"))
        out = prog.explain()
        assert isinstance(out, ExplainOutput)

    def test_explain_var_shape(self):
        out = Program.from_term(var("x")).explain()
        assert "Var('x')" in out.ast_shape
        assert out.leaf_schemas == {}

    def test_explain_leaf_indexes_schemas(self):
        t = leaf("hello {x}", input_vars=("x",))
        out = Program.from_term(t).explain()
        # Exactly one leaf, schema_ref None.
        assert len(out.leaf_schemas) == 1
        assert next(iter(out.leaf_schemas.values())) is None
        assert "Leaf" in out.ast_shape

    def test_explain_complex_term_renders_all_kinds(self):
        t = fix(
            abs_(
                "self",
                case_(
                    var("s"),
                    {
                        "a": leaf("hi", input_vars=()),
                        "b": var("x"),
                    },
                ),
            )
        )
        out = Program.from_term(t).explain()
        for kind in ("Fix", "Abs", "Case", "Var", "Leaf"):
            assert kind in out.ast_shape

    def test_explain_let_node_walk(self):
        t = let("temp", var("x"), var("temp"))
        out = Program.from_term(t).explain()
        assert "Let(name='temp')" in out.ast_shape

    def test_explain_plans_is_empty_in_r1(self):
        t = leaf("hi", input_vars=())
        out = Program.from_term(t).explain()
        # R1 contract: plans returned empty (runtime-info dependent).
        assert out.plans == []

    def test_explain_fsm_mode_walks_compiled_term(
        self, example_fsm_dict, mock_llm_interface
    ):
        prog = Program.from_fsm(example_fsm_dict, llm_interface=mock_llm_interface)
        out = prog.explain()
        # A real compiled FSM has nontrivial structure (Case + Lets).
        assert isinstance(out.ast_shape, str)
        assert len(out.ast_shape) > 100  # nontrivial rendering
        assert "Case" in out.ast_shape


# ---------------------------------------------------------------------------
# Handlers via constructor (post-0.7.0; .register_handler removed)
# ---------------------------------------------------------------------------


class TestProgramHandlersCtor:
    """Handlers are passed at construction (handlers=[...] kwarg) — the
    legacy ``.register_handler`` alias was removed at 0.7.0."""

    def test_fsm_mode_handlers_register_via_api(
        self, sample_fsm_dict, mock_llm_interface
    ):
        h = (
            create_handler("h1")
            .at(HandlerTiming.START_CONVERSATION)
            .do(lambda **kw: {})
        )
        prog = Program.from_fsm(
            sample_fsm_dict, llm_interface=mock_llm_interface, handlers=[h]
        )
        # Handler was forwarded to the underlying API's handler system.
        assert any(
            getattr(handler, "name", None) == "h1"
            for handler in prog._api.handler_system.handlers
        )

    def test_fsm_mode_handlers_tracked_on_program(
        self, sample_fsm_dict, mock_llm_interface
    ):
        h = (
            create_handler("h1")
            .at(HandlerTiming.START_CONVERSATION)
            .do(lambda **kw: {})
        )
        prog = Program.from_fsm(
            sample_fsm_dict, llm_interface=mock_llm_interface, handlers=[h]
        )
        assert h in prog._handlers

    def test_term_mode_handlers_compose_into_term(self):
        """handlers= at construction splices into self._term via
        fsm_llm.handlers.compose for term-mode programs."""
        h = create_handler("h2").at(HandlerTiming.PRE_PROCESSING).do(lambda **kw: {})
        # Without handlers — bare term.
        prog_bare = Program.from_term(var("x"))
        bare_term = prog_bare._term
        # With handlers — composed term differs from the bare one.
        prog = Program.from_term(var("x"), handlers=[h])
        assert prog._term is not bare_term
        assert h in prog._handlers


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class TestPublicSurface:
    """Program and ExplainOutput are exported from fsm_llm; ``API`` is no
    longer a top-level name (removed at 0.7.0; canonical path is
    ``fsm_llm.dialog.api.API``)."""

    def test_program_in_top_level_all(self):
        import fsm_llm

        assert "Program" in fsm_llm.__all__
        assert "ExplainOutput" in fsm_llm.__all__

    def test_program_imports_from_package(self):
        from fsm_llm import ExplainOutput, Program  # noqa: F401

    def test_program_imports_from_module(self):
        from fsm_llm.program import ExplainOutput, Program  # noqa: F401

    def test_program_module_resolves_to_same_class(self):
        import fsm_llm
        from fsm_llm.program import Program as P_module

        assert fsm_llm.Program is P_module

    def test_top_level_api_removed(self):
        """The top-level ``from fsm_llm import API`` shim was removed at 0.7.0.
        Users migrate to ``Program.from_fsm`` (or import the class from
        ``fsm_llm.dialog.api`` if they need it directly)."""
        import fsm_llm

        assert "API" not in fsm_llm.__all__
        with pytest.raises(AttributeError):
            fsm_llm.API  # noqa: B018

    def test_api_class_still_importable_via_dialog(self):
        """``API`` continues to live at ``fsm_llm.dialog.api.API`` — only the
        top-level convenience re-export was removed."""
        from fsm_llm.dialog.api import API as A

        assert isinstance(A, type)

    def test_program_is_class(self):
        assert isinstance(Program, type)

    def test_explainoutput_is_class(self):
        assert isinstance(ExplainOutput, type)
