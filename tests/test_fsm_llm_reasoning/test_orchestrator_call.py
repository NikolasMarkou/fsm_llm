"""The orchestrator pushes its strategy through a ``call`` state (plan baa37f40, step 15, D-031).

``execute_reasoning`` of ``strategies/orchestrator.json`` is a ``call`` state with ``choose``: the
strategy FSM is a frame the engine pushes and pops (``settle``), not a hook of the solve
(``_StrategyStack`` is gone). Rows hold: the file, the LLM call sequence of a solve (the one of
step 5b minus the calls that went with the push), the retry clear, the choice (classified, then
preferred, never a model reply), absent ``constraints``, the reply order, an interrupt inside the
pushed frame, the event log of a call frame, and the census rows D1 and D3 over the reasoning
flow. Offline: a scripted model, no network.
"""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

import fsm_llm
from fsm_llm.core.definitions import (
    FieldExtractionRequest,
    FieldExtractionResponse,
    FSMDefinition,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.core.verify import FSMValidator
from fsm_llm.logging import logger
from fsm_llm.reasoning import engine as engine_module
from fsm_llm.reasoning.constants import (
    ORCHESTRATOR_HANDLER_ONLY_KEYS,
    RETRY_CLEARED_KEYS,
    ContextKeys,
    Defaults,
    OrchestratorStates,
    solve_step_budget,
)
from fsm_llm.reasoning.strategy_files import STRATEGY_ORDER, load_fsm, strategy_id
from fsm_llm.runtime import EventLog
from fsm_llm.runtime.stores import fold_stack

from .test_engine_scripted import (
    _FAILING_SCRIPT,
    _HYBRID_SCRIPT,
    _PROBLEM,
    _VALID_SCRIPT,
    _ClaimingLLM,
    _engine,
    _ScriptedLLM,
    _the_call,
    _visited,
)
from .test_executables import _script

REPO = Path(__file__).resolve().parents[2]
CALL_STATE = OrchestratorStates.EXECUTE_REASONING


def test_the_module_under_test_is_this_checkout() -> None:
    assert Path(fsm_llm.__file__).resolve().is_relative_to(REPO / "src")
    assert Path(engine_module.__file__).resolve().is_relative_to(REPO / "src")


def _warnings(run: Any) -> str:
    buf = StringIO()
    logger.enable("fsm_llm")  # library logging is off by default
    sink = logger.add(buf, level="WARNING")
    try:
        run()
    finally:
        logger.remove(sink)
        logger.disable("fsm_llm")
    return buf.getvalue()


# ---- the file ----------------------------------------------------------------------------------


class TestTheOrchestratorFile:
    def test_execute_reasoning_is_a_call_with_the_closed_choice(self) -> None:
        state = load_fsm("orchestrator")["states"][CALL_STATE]
        assert state["call"] == {
            "choose": {
                "key": "classified_problem_type",
                "targets_from": {"capability": "strategy"},
                "targets": [strategy_id(name) for name in STRATEGY_ORDER],
                "fallback": "reasoning.analytical",
            },
            "inputs": [
                "problem_statement",
                "problem_components",
                "constraints",
                "problem_type",
            ],
            "on_failure": "synthesize_solution",
            "max_steps": 30,
        }
        # the reply text and the extraction prose went with the push
        assert state["response_instructions"] == ""
        assert "extraction_instructions" not in state

    def test_one_edge_reads_the_record_and_the_retry_edge_is_kept(self) -> None:
        states = load_fsm("orchestrator")["states"]
        (edge,) = states[CALL_STATE]["transitions"]
        assert edge["target_state"] == "synthesize_solution"
        (condition,) = edge["conditions"]
        assert condition["requires_context_keys"] == ["call_result"]
        assert condition["logic"] == {
            "in": [{"var": "call_result.status"}, ["ok", "failed", "aborted"]]
        }
        retry = [t["target_state"] for t in states["validate_refine"]["transitions"]]
        assert retry == ["final_answer", CALL_STATE]

    def test_the_definition_loads_and_validates_with_no_choose_lint(self) -> None:
        raw = load_fsm("orchestrator")
        definition = FSMDefinition(**raw)
        assert definition.states[CALL_STATE].call is not None
        report = FSMValidator(raw).validate()
        assert report.is_valid, report.errors
        assert not [w for w in report.warnings if "chooses its target" in w]

    def test_the_choice_key_is_handler_only_and_the_push_flag_is_gone(self) -> None:
        listed = load_fsm("orchestrator")["handler_only_keys"]
        assert ContextKeys.CLASSIFIED_PROBLEM_TYPE in listed
        assert "reasoning_push_pending" not in listed
        assert listed == list(ORCHESTRATOR_HANDLER_ONLY_KEYS)

    def test_the_call_record_joins_the_retry_clear(self) -> None:
        assert ContextKeys.CALL_RESULT in RETRY_CLEARED_KEYS

    def test_the_budget_of_a_solve_is_derived_from_the_file(self) -> None:
        max_steps = load_fsm("orchestrator")["states"][CALL_STATE]["call"]["max_steps"]
        assert solve_step_budget(max_steps) == (
            Defaults.MAX_TOTAL_STEPS + (Defaults.MAX_RETRIES + 1) * max_steps
        )


class TestTheDeletedNamesStayDeleted:
    """The push machinery went with its code (D-031); no solve pushes or pops a strategy by hand
    (the one `before_step` hook of the stream only marks the steps, plan baa37f40 D-036)."""

    def test_the_engine_module_has_none_of_it(self) -> None:
        for name in ("_StrategyStack", "_ordered_responses"):
            assert not hasattr(engine_module, name), name
        assert not hasattr(
            engine_module.ReasoningEngine, "_prepare_reasoning_execution"
        )
        source = Path(engine_module.__file__).read_text(encoding="utf-8")
        assert "push_fsm" not in source and "pop_fsm" not in source
        # the only hook is the stream's step marker: a hook that pushed or popped is the
        # `_StrategyStack` of D-053 again
        assert source.count("before_step=begin_step") == 1

    def test_an_engine_keeps_no_table_of_strategy_definitions(self) -> None:
        engine = _engine(_ScriptedLLM({}))
        assert not hasattr(engine, "reasoning_fsms")
        assert engine._call_config.choose is not None

    def test_the_context_manager_and_the_defaults_lost_their_names(self) -> None:
        from fsm_llm.reasoning.handlers import ContextManager

        assert not hasattr(ContextManager, "merge_reasoning_results")
        assert not hasattr(Defaults, "MAX_SUB_FSM_STEPS")

    def test_the_strategy_handlers_are_registered_once_on_the_orchestrator(
        self,
    ) -> None:
        from fsm_llm.reasoning.constants import HandlerNames

        engine = _engine(_ScriptedLLM({}))
        names = [h.name for h in engine.orchestrator.handler_system.handlers]
        for name in (
            HandlerNames.CALCULATOR_EXECUTOR,
            HandlerNames.CALCULATOR_FINALIZER,
            HandlerNames.HYBRID_KEY_CLEARER,
            HandlerNames.HYBRID_LOOP_COUNTER,
            HandlerNames.ORCHESTRATOR_SELECTOR,
        ):
            assert names.count(name) == 1, name


# ---- the model calls of a solve ----------------------------------------------------------------


def _label(request: Any) -> str:
    if isinstance(request, FieldExtractionRequest):
        return f"field:{request.field_name}"
    if isinstance(request, ResponseGenerationRequest):
        match = re.search(
            r"<current_state>(\w+)</current_state>", request.system_prompt
        )
        return f"reply:{match.group(1) if match else None}"
    return "bulk"


# The sequence of the scripted deductive solve on the parent commit (step 14, 44 calls) minus the
# two calls that went with the push: the Pass-2 acknowledgement of `execute_reasoning` and the
# bulk extraction of that state on the turn that left it. Everything else is in its place, the
# classification stays ONE `complete` request, and the child's greeting is `reply:identify_premises`.
_SEQUENCE = """
reply:problem_analysis field:problem_type field:problem_components bulk
reply:analyze_domain field:problem_domain field:domain_indicators bulk
field:problem_structure field:structural_elements reply:analyze_structure bulk
field:reasoning_requirements field:key_challenges reply:identify_reasoning_needs bulk
field:strategy_justification field:alternative_approaches reply:recommend_strategy
field:reasoning_strategy field:strategy_rationale reply:strategy_selection bulk
reply:identify_premises field:premises field:assumptions bulk
field:logical_steps field:intermediate_conclusions reply:apply_logic bulk
field:conclusion field:logical_validity reply:derive_conclusion
field:proposed_solution field:key_insights reply:synthesize_solution bulk
reply:validate_refine bulk field:final_solution reply:final_answer
""".split()


class TestTheModelCallsOfASolve:
    def test_the_sequence_is_the_one_of_the_push_minus_the_dropped_calls(self) -> None:
        llm = _ScriptedLLM(_VALID_SCRIPT)
        _engine(llm).solve_problem(_PROBLEM)

        assert [_label(r) for r in llm.requests] == _SEQUENCE
        assert [r.call_type for r in llm.completions] == ["classification"]

    def test_the_call_state_itself_makes_no_model_call(self) -> None:
        llm = _ScriptedLLM(_VALID_SCRIPT)
        _engine(llm).solve_problem(_PROBLEM)

        labels = [_label(r) for r in llm.requests]
        assert f"reply:{CALL_STATE}" not in labels
        assert not [r for r in llm.requests if "WAIT-MARKER" in str(r)]


# ---- the retry ---------------------------------------------------------------------------------


class _Spy(_ScriptedLLM):
    """Calls ``on_field(field_name)`` before it answers a per-field request."""

    def __init__(self, script: dict[str, Any], on_field: Any) -> None:
        super().__init__(script)
        self.on_field = on_field

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        self.on_field(request.field_name)
        return super().extract_field(request)


class TestTheRetryRunsTheStrategyAgain:
    def test_every_attempt_pushes_a_fresh_strategy_frame(self) -> None:
        """The clear runs on entry BEFORE the executor decides on arrival: a stale `ok`
        record would be skipped and the strategy would run once. Kills: `call_result`
        missing from `RETRY_CLEARED_KEYS`."""
        llm = _ScriptedLLM(_FAILING_SCRIPT)
        _solution, trace_info = _engine(llm).solve_problem(_PROBLEM)

        attempts = Defaults.MAX_RETRIES
        assert len(llm.field_requests(ContextKeys.CONCLUSION)) == attempts
        assert _visited(trace_info).count(CALL_STATE) == attempts
        assert _the_call(trace_info["final_context"])["status"] == "ok"


class _CalculatorLLM(_ScriptedLLM):
    """The model translates the problem to a refused expression first, then to a good one."""

    def __init__(self, script: dict[str, Any], expressions: list[str]) -> None:
        super().__init__(script)
        self.expressions = list(expressions)

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        if request.field_name == ContextKeys.CALCULATION_EXPRESSION:
            self.requests.append(request)
            value = self.expressions.pop(0)
            return FieldExtractionResponse(
                field_name=request.field_name, value=value, confidence=0.9
            )
        return super().extract_field(request)


_CALC_SCRIPT: dict[str, Any] = {
    "problem_type": "arithmetic",
    "problem_components": ["10", "/", "2"],
    "problem_domain": "math",
    "recommended_reasoning_type": "simple_calculator",
    "reasoning_strategy": "simple_calculator",
    "strategy_rationale": "plain division",
    "key_insights": ["division"],
    "final_solution": "5",
}


class TestARefusedCalculationThenASuccess:
    def test_the_retry_keeps_no_stale_error_and_the_answer_is_the_result(self) -> None:
        llm = _CalculatorLLM(_CALC_SCRIPT, ["10 / 0", "10 / 2"])
        solution, trace_info = _engine(llm).solve_problem("What is 10 / 2?")
        context = trace_info["final_context"]

        assert llm.expressions == []  # both attempts reached the model
        assert solution == "5"
        assert context[ContextKeys.RETRY_COUNT] == 1
        assert context[ContextKeys.CALCULATION_RESULT] == 5
        assert context.get(ContextKeys.CALCULATION_ERROR_DETAILS) is None
        assert context[ContextKeys.VALIDATION_RESULT] is True
        record = _the_call(context)
        assert (record["status"], record["target"]) == (
            "ok",
            "reasoning.simple_calculator",
        )


# ---- the choice --------------------------------------------------------------------------------


class TestTheChoice:
    def test_a_preferred_type_beats_the_classified_one_with_one_classification(
        self,
    ) -> None:
        llm = _ScriptedLLM(_HYBRID_SCRIPT)
        _solution, trace_info = _engine(llm).solve_problem(
            _PROBLEM, {ContextKeys.PREFERRED_REASONING_TYPE: "hybrid"}
        )
        context = trace_info["final_context"]

        assert [r.call_type for r in llm.completions] == ["classification"]
        assert context[ContextKeys.CLASSIFIED_PROBLEM_TYPE] == "reasoning.hybrid"
        assert _the_call(context)["target"] == "reasoning.hybrid"
        assert context[ContextKeys.REASONING_TYPE_SELECTED] == "hybrid"

    def test_an_unknown_preferred_type_warns_and_the_classified_one_runs(self) -> None:
        llm = _ScriptedLLM(_VALID_SCRIPT)
        engine = _engine(llm)
        result: list[Any] = []

        warned = _warnings(
            lambda: result.append(
                engine.solve_problem(
                    _PROBLEM, {ContextKeys.PREFERRED_REASONING_TYPE: "no such type"}
                )
            )
        )
        context = result[0][1]["final_context"]

        assert "names no known strategy" in warned
        assert _the_call(context)["target"] == "reasoning.deductive"

    def test_a_value_outside_the_list_takes_the_fallback(self) -> None:
        """A context that carries `problem_type` skips the classifier (a known limit), so the
        value of the choice key is whatever the caller seeded on the orchestrator itself."""
        script = {**_VALID_SCRIPT, **_script("analytical")}
        llm = _ScriptedLLM(script)
        engine = _engine(llm)
        cid, _ = engine.orchestrator.start_conversation(
            {
                ContextKeys.PROBLEM_STATEMENT: _PROBLEM,
                ContextKeys.PROBLEM_TYPE: "logic puzzle",
                ContextKeys.PROBLEM_COMPONENTS: ["a", "b"],
                ContextKeys.CLASSIFIED_PROBLEM_TYPE: "reasoning.no_such_strategy",
                ContextKeys.REASONING_TRACE: [],
                ContextKeys.RETRY_COUNT: 0,
            }
        )
        engine.orchestrator.run_until_terminal(cid, max_steps=engine._solve_steps)
        context = engine.orchestrator.get_data(cid)

        assert _the_call(context)["target"] == "reasoning.analytical"
        assert context[ContextKeys.REASONING_TYPE_SELECTED] == "analytical"
        assert context[ContextKeys.INTEGRATED_ANALYSIS]

    def test_a_model_reply_cannot_choose_the_strategy(self) -> None:
        """D-054: the choice key is handler-only, so a bulk reply that claims it (and the
        model's own `reasoning_strategy`) never picks the strategy. Kills: the key taken out
        of the orchestrator's `handler_only_keys`."""
        claims = {
            ContextKeys.CLASSIFIED_PROBLEM_TYPE: "reasoning.creative",
            ContextKeys.REASONING_STRATEGY: "creative",
        }
        llm = _ClaimingLLM(_VALID_SCRIPT, claims=claims)
        _solution, trace_info = _engine(llm).solve_problem(_PROBLEM)
        context = trace_info["final_context"]

        assert llm.field_requests(ContextKeys.CLASSIFIED_PROBLEM_TYPE) == []
        assert context[ContextKeys.CLASSIFIED_PROBLEM_TYPE] == "reasoning.deductive"
        assert _the_call(context)["target"] == "reasoning.deductive"
        assert ContextKeys.BEST_CREATIVE_SOLUTION not in context


class TestConstraints:
    def test_absent_constraints_run_the_strategy(self) -> None:
        """The executor refuses an absent declared input; the entry handler defaults it.
        Kills: no default (the call would end `failed/inputs_missing`)."""
        llm = _ScriptedLLM(_VALID_SCRIPT)
        _solution, trace_info = _engine(llm).solve_problem(_PROBLEM)
        context = trace_info["final_context"]

        record = _the_call(context)
        assert (record["status"], record["reason"]) == ("ok", None)
        assert context[ContextKeys.CONSTRAINTS] == []
        assert context[ContextKeys.DEDUCTIVE_CONCLUSION] == "Socrates is mortal"

    def test_a_callers_constraints_reach_the_strategy(self) -> None:
        seen: list[Any] = []
        llm = _Spy(_VALID_SCRIPT, lambda name: None)
        engine = _engine(llm)

        def watch(name: str) -> None:
            if name == "premises":
                (root,) = engine.orchestrator.list_active_conversations()
                top = engine.orchestrator.conversation_stacks[root][-1].conversation_id
                data = engine.orchestrator.fsm_manager.copy_raw_context_data(top)
                seen.append(data.get(ContextKeys.CONSTRAINTS))

        llm.on_field = watch
        engine.solve_problem(_PROBLEM, {ContextKeys.CONSTRAINTS: ["no loops"]})

        assert seen == [["no loops"]]


# ---- the answers -------------------------------------------------------------------------------


class _NamingLLM(_ScriptedLLM):
    """Every reply is the name of the state it was written from."""

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        self.requests.append(request)
        match = re.search(
            r"<current_state>(\w+)</current_state>", request.system_prompt
        )
        return ResponseGenerationResponse(
            message=match.group(1) if match else "?",
            message_type="response",
            reasoning="",
        )


class TestTheReplyOrder:
    def test_all_responses_are_in_the_order_they_were_produced(self) -> None:
        """The start greeting, then each step's reply; the greeting of the pushed strategy
        FSM rides on the step that pushed it. `execute_reasoning` says nothing."""
        llm = _NamingLLM(_VALID_SCRIPT)
        _solution, trace_info = _engine(llm).solve_problem(_PROBLEM)

        # the classifier runs on its own API: its replies are not the solve's
        assert trace_info["all_responses"] == [
            "problem_analysis",  # the start greeting
            "strategy_selection",
            "identify_premises",  # the greeting of the strategy FSM
            "apply_logic",
            "derive_conclusion",
            "synthesize_solution",
            "validate_refine",
            "final_answer",
        ]


# ---- cancel, interrupt, log --------------------------------------------------------------------


class TestAnInterruptInsideThePushedFrame:
    @pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
    def test_both_frames_end_and_the_root_mark_is_free(self, interrupt) -> None:
        from .test_engine_scripted import _InterruptingLLM

        llm = _InterruptingLLM(_VALID_SCRIPT, "conclusion", interrupt())
        engine = _engine(llm)
        llm.api = engine.orchestrator

        with pytest.raises(interrupt):
            engine.solve_problem(_PROBLEM)

        marks = engine.orchestrator._call_marks
        assert llm.frames_at_raise == 2
        assert marks.settling == set() and marks.pushed == {} and marks.ending == set()
        assert engine.orchestrator.list_active_conversations() == []
        assert not engine._solve_lock.locked()


class _Recording(EventLog):
    """A log that notes the type of every event it appends and the stack depth after a push."""

    def __init__(self, root: Any) -> None:
        super().__init__(root)
        self.types: list[str] = []
        self.depths: list[int] = []
        self.root_id: str | None = None

    def append(self, conversation_id: str, type: Any, **kwargs: Any) -> int:  # type: ignore[override]
        seq = super().append(conversation_id, type, **kwargs)
        kind = str(getattr(type, "value", type))
        self.types.append(kind)
        if kind == "stack/push":
            folded = fold_stack(self.read(conversation_id), self.get_json)
            self.depths.append(folded.depth)
        return seq


class TestTheEventLogOfACallFrame:
    def test_one_push_and_one_pop_and_the_fold_sees_the_frame(self, tmp_path) -> None:
        log = _Recording(tmp_path / "log")
        engine = fsm_llm.reasoning.ReasoningEngine(
            model="mock", llm_interface=_ScriptedLLM(_VALID_SCRIPT), event_log=log
        )
        engine.solve_problem(_PROBLEM)
        engine.close()
        log.close()

        assert log.types.count("stack/push") == 1
        assert log.types.count("stack/pop") == 1
        assert log.types.index("stack/push") < log.types.index("stack/pop")
        assert log.depths == [2]  # the fold rebuilt the strategy frame above the root


# ---- census D1 over the reasoning flow ---------------------------------------------------------


class TestTheStrategyReceivesOnlyTheDeclaredInputs:
    def test_a_grant_in_the_parent_context_never_reaches_the_strategy_frame(
        self,
    ) -> None:
        """Kills (D1): a push that inherits the parent's context, or hands over keys beyond
        the call's declared `inputs` (the child would hold `_approval_granted`)."""
        seen: list[dict[str, Any]] = []
        llm = _Spy(_VALID_SCRIPT, lambda name: None)
        engine = _engine(llm)

        def watch(name: str) -> None:
            if name == "premises":
                (root,) = engine.orchestrator.list_active_conversations()
                top = engine.orchestrator.conversation_stacks[root][-1].conversation_id
                seen.append(engine.orchestrator.fsm_manager.copy_raw_context_data(top))

        llm.on_field = watch
        engine.solve_problem(
            _PROBLEM,
            {"_approval_granted": {"tool": "x"}, "api_key": "sk-secret-value"},
        )

        (child,) = seen
        public = {k for k in child if not k.startswith("_")}
        assert "_approval_granted" not in child
        assert "api_key" not in child and "sk-secret-value" not in repr(child)
        assert public <= {
            "problem_statement",
            "problem_components",
            "constraints",
            "problem_type",
            "premises",
        } | set(_script("deductive"))
