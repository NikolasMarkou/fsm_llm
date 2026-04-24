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
    def test_rejects_empty_states(self) -> None:
        # FSMDefinition.validate blocks empty states at model-validation
        # time, so we can't construct one via model_validate. We exercise
        # the compile-time guard by bypassing construction.
        defn = FSMDefinition.model_validate(_greeter_fsm_dict())
        object.__setattr__(defn, "states", {})
        with pytest.raises(ASTConstructionError, match="no states"):
            compile_fsm(defn)


class TestCompileBaseCase:
    """S2 base case: response-only FSMs compile to an Abs-chain + Case."""

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

    def test_state_body_is_app_of_respond_cb(self) -> None:
        """S2 contract: each state's body is ``App(Var(CB_RESPOND), Var(VAR_INSTANCE))``."""
        from fsm_llm.lam.ast import App, Var

        defn = FSMDefinition.model_validate(_greeter_fsm_dict())
        term = compile_fsm(defn)
        case_body = term.body.body.body.body  # unwrap 4 Abs to reach Case
        state_body = case_body.branches["hello"]
        assert isinstance(state_body, App)
        assert isinstance(state_body.fn, Var) and state_body.fn.name == fsc.CB_RESPOND
        assert (
            isinstance(state_body.arg, Var) and state_body.arg.name == fsc.VAR_INSTANCE
        )


class TestCompileEndToEndExecutor:
    """S2 end-to-end: compiled term runs through the λ-executor via host
    callables (``executor.py:234`` accepts any Python callable under App).
    No oracle used; no Leaf involved in M2's FSM path."""

    def test_compiled_trivial_fsm_runs_through_executor(self) -> None:
        from fsm_llm.lam.executor import Executor

        defn = FSMDefinition.model_validate(_greeter_fsm_dict())
        term = compile_fsm(defn)

        captured = {"instance_seen": None}

        def fake_respond(instance: object) -> str:
            # Simulate what the pipeline's response-callback will do in S8:
            # receive the instance, return a response string.
            captured["instance_seen"] = instance
            return "hi there"

        sentinel_instance = object()
        env = {
            fsc.VAR_STATE_ID: "hello",
            fsc.VAR_MESSAGE: "whatever",
            fsc.VAR_CONV_ID: "c-1",
            fsc.VAR_INSTANCE: sentinel_instance,
            fsc.CB_RESPOND: fake_respond,
        }
        # Executor has no oracle — S2 uses no Leaf nodes (D-003).
        result = Executor().run(term, env)
        # The outermost term is λ state_id. λ message. λ conv_id. λ instance. ...
        # which evaluates to nested closures. To trigger the body we apply
        # it to all 4 inputs. But here we rely on the executor binding them
        # directly from env via Var lookups. Since the top-level term is an
        # Abs (not an App), running it produces a Closure, not the response.
        # The pipeline in S8 binds via env, not by applying the term.
        # Therefore this test exercises the "pre-applied via env" contract:
        # the compiled term expects its formal params to already be in env.
        # We assert the closure shape; a separate test invokes the Case body.
        from fsm_llm.lam.executor import _Closure

        assert isinstance(result, _Closure)
        assert result.param == fsc.VAR_STATE_ID

    def test_state_with_no_extraction_still_base_case(self) -> None:
        """Sanity: S2 base case must survive S3 additions. A state with no
        extraction and no field_extractions still compiles to just
        ``App(_cb_respond, instance)``."""
        from fsm_llm.lam.ast import App

        defn = FSMDefinition.model_validate(_greeter_fsm_dict())
        term = compile_fsm(defn)
        state_body = term.body.body.body.body.branches["hello"]
        assert isinstance(state_body, App)
        assert state_body.fn.name == fsc.CB_RESPOND

    def test_compiled_state_body_calls_respond_callback(self) -> None:
        """Extract the Case and evaluate it directly under an env that has
        the 4 inputs bound. This mirrors what S8 will do in pipeline.py:
        the pipeline binds inputs to env, then evaluates the Case scrutinee."""
        from fsm_llm.lam.executor import Executor

        defn = FSMDefinition.model_validate(_greeter_fsm_dict())
        term = compile_fsm(defn)
        case_body = term.body.body.body.body  # unwrap 4 Abs to reach Case

        calls: list[object] = []

        def fake_respond(instance: object) -> str:
            calls.append(instance)
            return "hi there"

        sentinel_instance = object()
        env = {
            fsc.VAR_STATE_ID: "hello",
            fsc.VAR_MESSAGE: "m",
            fsc.VAR_CONV_ID: "c",
            fsc.VAR_INSTANCE: sentinel_instance,
            fsc.CB_RESPOND: fake_respond,
        }
        result = Executor().run(case_body, env)
        assert result == "hi there"
        assert calls == [sentinel_instance]


# --------------------------------------------------------------
# S3: extraction-stage compilation
# --------------------------------------------------------------


def _extraction_fsm_dict(*, bulk: bool, fields: bool) -> dict:
    """Build an FSM whose single state optionally declares
    ``extraction_instructions`` and/or ``field_extractions``."""
    state: dict = {
        "id": "s0",
        "description": "d",
        "purpose": "p",
        "response_instructions": "respond",
        "transitions": [],
    }
    if bulk:
        state["extraction_instructions"] = "extract all the things"
    if fields:
        state["field_extractions"] = [
            {
                "field_name": "user_name",
                "field_type": "str",
                "extraction_instructions": "extract the user's name",
            }
        ]
    return {
        "name": "extr",
        "description": "extraction test",
        "initial_state": "s0",
        "persona": "t",
        "states": {"s0": state},
    }


class TestCompileExtractionStage:
    def test_extraction_only_emits_single_let(self) -> None:
        from fsm_llm.lam.ast import App, Let

        defn = FSMDefinition.model_validate(
            _extraction_fsm_dict(bulk=True, fields=False)
        )
        term = compile_fsm(defn)
        state_body = term.body.body.body.body.branches["s0"]
        assert isinstance(state_body, Let)
        # The Let's value is the extraction callback call.
        assert isinstance(state_body.value, App)
        assert state_body.value.fn.name == fsc.CB_EXTRACT
        # The body is the terminal respond call.
        assert isinstance(state_body.body, App)
        assert state_body.body.fn.name == fsc.CB_RESPOND

    def test_field_extract_only_emits_single_let(self) -> None:
        from fsm_llm.lam.ast import Let

        defn = FSMDefinition.model_validate(
            _extraction_fsm_dict(bulk=False, fields=True)
        )
        term = compile_fsm(defn)
        state_body = term.body.body.body.body.branches["s0"]
        assert isinstance(state_body, Let)
        assert state_body.value.fn.name == fsc.CB_FIELD_EXTRACT
        assert state_body.body.fn.name == fsc.CB_RESPOND

    def test_both_emit_nested_lets_bulk_outermost(self) -> None:
        """extraction runs BEFORE field extractions BEFORE respond.
        Therefore the bulk-extract Let must be the OUTER Let."""
        from fsm_llm.lam.ast import App, Let

        defn = FSMDefinition.model_validate(
            _extraction_fsm_dict(bulk=True, fields=True)
        )
        term = compile_fsm(defn)
        state_body = term.body.body.body.body.branches["s0"]
        assert isinstance(state_body, Let)
        assert state_body.value.fn.name == fsc.CB_EXTRACT  # outer = bulk
        inner = state_body.body
        assert isinstance(inner, Let)
        assert inner.value.fn.name == fsc.CB_FIELD_EXTRACT  # inner = fields
        final = inner.body
        assert isinstance(final, App)
        assert final.fn.name == fsc.CB_RESPOND

    def test_extraction_callback_invoked_before_respond(self) -> None:
        """End-to-end: extraction callback runs first, then respond."""
        from fsm_llm.lam.executor import Executor

        call_log: list[str] = []

        def fake_extract(instance: object) -> object:
            call_log.append("extract")
            return instance

        def fake_field_extract(instance: object) -> object:
            call_log.append("field_extract")
            return instance

        def fake_respond(instance: object) -> str:
            call_log.append("respond")
            return "done"

        defn = FSMDefinition.model_validate(
            _extraction_fsm_dict(bulk=True, fields=True)
        )
        term = compile_fsm(defn)
        case_body = term.body.body.body.body
        env = {
            fsc.VAR_STATE_ID: "s0",
            fsc.VAR_MESSAGE: "m",
            fsc.VAR_CONV_ID: "c",
            fsc.VAR_INSTANCE: object(),
            fsc.CB_EXTRACT: fake_extract,
            fsc.CB_FIELD_EXTRACT: fake_field_extract,
            fsc.CB_RESPOND: fake_respond,
        }
        result = Executor().run(case_body, env)
        assert result == "done"
        assert call_log == ["extract", "field_extract", "respond"]

    def test_empty_extraction_instructions_skipped(self) -> None:
        """A state whose extraction_instructions is '' or whitespace must
        NOT emit the extraction Let — skipping aligns with pipeline
        behavior that a blank template is a no-op."""
        from fsm_llm.lam.ast import App

        d = _extraction_fsm_dict(bulk=False, fields=False)
        d["states"]["s0"]["extraction_instructions"] = "   "
        defn = FSMDefinition.model_validate(d)
        term = compile_fsm(defn)
        state_body = term.body.body.body.body.branches["s0"]
        # No Let; straight to the respond App.
        assert isinstance(state_body, App)
        assert state_body.fn.name == fsc.CB_RESPOND
