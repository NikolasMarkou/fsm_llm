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

# --------------------------------------------------------------
# S4: classification-extraction stage
# --------------------------------------------------------------


def _classification_extract_block() -> dict:
    """A minimal valid ClassificationExtractionConfig dict."""
    return {
        "field_name": "intent",
        "intents": [
            {"name": "buy", "description": "user wants to purchase"},
            {"name": "browse", "description": "user is just looking"},
        ],
        "fallback_intent": "browse",
    }


def _cext_fsm_dict(*, bulk: bool, fields: bool, classes: bool) -> dict:
    state: dict = {
        "id": "s0",
        "description": "d",
        "purpose": "p",
        "response_instructions": "respond",
        "transitions": [],
    }
    if bulk:
        state["extraction_instructions"] = "extract"
    if fields:
        state["field_extractions"] = [
            {
                "field_name": "name",
                "field_type": "str",
                "extraction_instructions": "extract",
            }
        ]
    if classes:
        state["classification_extractions"] = [_classification_extract_block()]
    return {
        "name": "cext",
        "description": "classification test",
        "initial_state": "s0",
        "persona": "t",
        "states": {"s0": state},
    }


class TestCompileClassificationStage:
    def test_class_extract_only_emits_single_let(self) -> None:
        from fsm_llm.lam.ast import Let

        defn = FSMDefinition.model_validate(
            _cext_fsm_dict(bulk=False, fields=False, classes=True)
        )
        term = compile_fsm(defn)
        state_body = term.body.body.body.body.branches["s0"]
        assert isinstance(state_body, Let)
        assert state_body.value.fn.name == fsc.CB_CLASS_EXTRACT
        assert state_body.body.fn.name == fsc.CB_RESPOND

    def test_all_three_stages_nested_in_order(self) -> None:
        """bulk extract (outer) → field extract → class extract → respond."""
        from fsm_llm.lam.ast import App, Let

        defn = FSMDefinition.model_validate(
            _cext_fsm_dict(bulk=True, fields=True, classes=True)
        )
        term = compile_fsm(defn)
        layer0 = term.body.body.body.body.branches["s0"]
        assert isinstance(layer0, Let)
        assert layer0.value.fn.name == fsc.CB_EXTRACT
        layer1 = layer0.body
        assert isinstance(layer1, Let)
        assert layer1.value.fn.name == fsc.CB_FIELD_EXTRACT
        layer2 = layer1.body
        assert isinstance(layer2, Let)
        assert layer2.value.fn.name == fsc.CB_CLASS_EXTRACT
        tail = layer2.body
        assert isinstance(tail, App)
        assert tail.fn.name == fsc.CB_RESPOND

    def test_class_extract_runtime_ordering(self) -> None:
        """End-to-end: the three stages run in the documented order."""
        from fsm_llm.lam.executor import Executor

        call_log: list[str] = []

        def record(name: str):  # type: ignore[no-untyped-def]
            def _cb(instance: object) -> object:
                call_log.append(name)
                return instance

            return _cb

        defn = FSMDefinition.model_validate(
            _cext_fsm_dict(bulk=True, fields=True, classes=True)
        )
        term = compile_fsm(defn)
        case_body = term.body.body.body.body
        env = {
            fsc.VAR_STATE_ID: "s0",
            fsc.VAR_MESSAGE: "m",
            fsc.VAR_CONV_ID: "c",
            fsc.VAR_INSTANCE: object(),
            fsc.CB_EXTRACT: record("extract"),
            fsc.CB_FIELD_EXTRACT: record("field"),
            fsc.CB_CLASS_EXTRACT: record("class"),
            fsc.CB_RESPOND: lambda inst: (call_log.append("respond") or "ok"),
        }
        result = Executor().run(case_body, env)
        assert result == "ok"
        assert call_log == ["extract", "field", "class", "respond"]

    def test_no_classes_no_let(self) -> None:
        """If the state declares no classification_extractions, no Let
        for class extract is emitted."""

        defn = FSMDefinition.model_validate(
            _cext_fsm_dict(bulk=True, fields=False, classes=False)
        )
        term = compile_fsm(defn)
        layer0 = term.body.body.body.body.branches["s0"]
        # Only the bulk-extract Let should be present.
        inner = layer0.body
        from fsm_llm.lam.ast import App

        assert isinstance(inner, App)
        assert inner.fn.name == fsc.CB_RESPOND

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
        assert isinstance(state_body, App)
        assert state_body.fn.name == fsc.CB_RESPOND


# --------------------------------------------------------------
# S5: transition-evaluation dispatch
# --------------------------------------------------------------


def _transition_fsm_dict(
    *,
    extractions: bool = False,
    field_extractions: bool = False,
    class_extractions: bool = False,
) -> dict:
    """2-state FSM with a deterministic transition start → end.

    Optionally attach extraction stages to ``start`` to exercise ordering
    interactions between S3/S4 Let-chain and S5 Let+Case dispatch.
    """
    start_state: dict = {
        "id": "start",
        "description": "begin",
        "purpose": "start",
        "response_instructions": "respond_start",
        "transitions": [
            {
                "target_state": "end",
                "description": "always advance",
                "conditions": [
                    {"description": "always", "logic": {"==": [1, 1]}}
                ],
            }
        ],
    }
    if extractions:
        start_state["extraction_instructions"] = "extract all"
    if field_extractions:
        start_state["field_extractions"] = [
            {
                "field_name": "f1",
                "field_type": "string",
                "extraction_instructions": "x",
            }
        ]
    if class_extractions:
        start_state["classification_extractions"] = [
            {
                "field_name": "intent",
                "intents": [
                    {"name": "a", "description": "alpha"},
                    {"name": "b", "description": "beta"},
                ],
                "fallback_intent": "a",
                "confidence_threshold": 0.5,
            }
        ]
    return {
        "name": "transition_fsm",
        "description": "S5 test FSM",
        "initial_state": "start",
        "persona": "test",
        "states": {
            "start": start_state,
            "end": {
                "id": "end",
                "description": "done",
                "purpose": "done",
                "response_instructions": "respond_end",
                "transitions": [],
            },
        },
    }


class TestCompileTransitionStage:
    """S5: non-terminal states compile to Let+Case wrapping the respond call."""

    def test_terminal_state_unchanged(self) -> None:
        """Regression gate: terminal state (end) still compiles to bare
        App(CB_RESPOND, instance) — no Let/Case introduced by S5."""
        from fsm_llm.lam.ast import App

        defn = FSMDefinition.model_validate(_transition_fsm_dict())
        term = compile_fsm(defn)
        end_body = term.body.body.body.body.branches["end"]
        assert isinstance(end_body, App)
        assert end_body.fn.name == fsc.CB_RESPOND

    def test_nonterminal_state_is_let_of_eval_transit(self) -> None:
        """A state with transitions compiles to
        Let(__disc_*, App(CB_EVAL_TRANSIT, instance), Case(...))."""
        from fsm_llm.lam.ast import App, Case, Let, Var

        defn = FSMDefinition.model_validate(_transition_fsm_dict())
        term = compile_fsm(defn)
        start_body = term.body.body.body.body.branches["start"]

        assert isinstance(start_body, Let), (
            f"expected outer Let, got {type(start_body).__name__}"
        )
        assert start_body.name.startswith("__disc_"), (
            f"expected disc gensym, got {start_body.name!r}"
        )
        assert isinstance(start_body.value, App)
        assert isinstance(start_body.value.fn, Var)
        assert start_body.value.fn.name == fsc.CB_EVAL_TRANSIT
        assert isinstance(start_body.value.arg, Var)
        assert start_body.value.arg.name == fsc.VAR_INSTANCE
        assert isinstance(start_body.body, Case)

    def test_case_scrutinee_references_disc(self) -> None:
        from fsm_llm.lam.ast import Var

        defn = FSMDefinition.model_validate(_transition_fsm_dict())
        term = compile_fsm(defn)
        start_body = term.body.body.body.body.branches["start"]
        case_node = start_body.body
        assert isinstance(case_node.scrutinee, Var)
        assert case_node.scrutinee.name == start_body.name  # same disc binding

    def test_case_branches_cover_all_discriminants(self) -> None:
        """Branches: {advanced, blocked, ambiguous}, each body
        App(CB_RESPOND, instance). Default also App(CB_RESPOND, instance)."""
        from fsm_llm.lam.ast import App

        defn = FSMDefinition.model_validate(_transition_fsm_dict())
        term = compile_fsm(defn)
        start_body = term.body.body.body.body.branches["start"]
        case_node = start_body.body

        assert set(case_node.branches.keys()) == {"advanced", "blocked", "ambiguous"}
        for key, branch in case_node.branches.items():
            assert isinstance(branch, App), f"branch {key!r} not App"
            assert branch.fn.name == fsc.CB_RESPOND, (
                f"branch {key!r} fn is {branch.fn.name!r}, expected CB_RESPOND"
            )
            assert branch.arg.name == fsc.VAR_INSTANCE
        assert case_node.default is not None
        assert isinstance(case_node.default, App)
        assert case_node.default.fn.name == fsc.CB_RESPOND


class TestCompileTransitionEndToEnd:
    """S5 end-to-end: run the compiled Case body through Executor with
    synthesized callbacks; verify order + state-advance semantics."""

    def test_nonterminal_end_to_end_ordering(self) -> None:
        """start state with transitions: eval_transit fires after
        extractions (if any) and before respond. S5 minimal case has no
        extractions, so order is eval_transit → respond."""
        from fsm_llm.lam.executor import Executor

        call_log: list[str] = []

        def record(name: str, ret: object = None):  # type: ignore[no-untyped-def]
            def _cb(instance: object) -> object:
                call_log.append(name)
                return ret if ret is not None else instance

            return _cb

        defn = FSMDefinition.model_validate(_transition_fsm_dict())
        term = compile_fsm(defn)
        case_on_state_id = term.body.body.body.body  # outermost Case (state_id)

        env = {
            fsc.VAR_STATE_ID: "start",
            fsc.VAR_MESSAGE: "m",
            fsc.VAR_CONV_ID: "c",
            fsc.VAR_INSTANCE: object(),
            fsc.CB_EVAL_TRANSIT: record("eval_transit", ret="advanced"),
            fsc.CB_RESPOND: lambda inst: (call_log.append("respond") or "response"),
        }
        result = Executor().run(case_on_state_id, env)
        assert result == "response"
        assert call_log == ["eval_transit", "respond"]

    def test_deterministic_advance_mutates_current_state(self) -> None:
        """Fake eval_transit mutates instance.current_state; respond sees
        the mutated value. This is the load-bearing S5 behavior."""
        from types import SimpleNamespace

        from fsm_llm.lam.executor import Executor

        instance = SimpleNamespace(current_state="start")

        def fake_eval_transit(inst: SimpleNamespace) -> str:
            inst.current_state = "end"
            return "advanced"

        captured: dict[str, str] = {}

        def fake_respond(inst: SimpleNamespace) -> str:
            captured["seen"] = inst.current_state
            return "ok"

        defn = FSMDefinition.model_validate(_transition_fsm_dict())
        term = compile_fsm(defn)
        case_on_state_id = term.body.body.body.body

        env = {
            fsc.VAR_STATE_ID: "start",
            fsc.VAR_MESSAGE: "m",
            fsc.VAR_CONV_ID: "c",
            fsc.VAR_INSTANCE: instance,
            fsc.CB_EVAL_TRANSIT: fake_eval_transit,
            fsc.CB_RESPOND: fake_respond,
        }
        result = Executor().run(case_on_state_id, env)
        assert result == "ok"
        assert captured["seen"] == "end"
        assert instance.current_state == "end"

    def test_blocked_does_not_mutate_current_state(self) -> None:
        """eval_transit returns 'blocked' without mutating; respond
        observes the original current_state."""
        from types import SimpleNamespace

        from fsm_llm.lam.executor import Executor

        instance = SimpleNamespace(current_state="start")

        def blocking_eval_transit(inst: SimpleNamespace) -> str:
            return "blocked"

        captured: dict[str, str] = {}

        def fake_respond(inst: SimpleNamespace) -> str:
            captured["seen"] = inst.current_state
            return "blocked_response"

        defn = FSMDefinition.model_validate(_transition_fsm_dict())
        term = compile_fsm(defn)
        case_on_state_id = term.body.body.body.body

        env = {
            fsc.VAR_STATE_ID: "start",
            fsc.VAR_MESSAGE: "m",
            fsc.VAR_CONV_ID: "c",
            fsc.VAR_INSTANCE: instance,
            fsc.CB_EVAL_TRANSIT: blocking_eval_transit,
            fsc.CB_RESPOND: fake_respond,
        }
        result = Executor().run(case_on_state_id, env)
        assert result == "blocked_response"
        assert captured["seen"] == "start"
        assert instance.current_state == "start"
