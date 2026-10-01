"""Review round 1, agents and consumers fixes (plan 07ad3f8c step 22.2).

Each class pins one accepted finding of ``findings/review-iter-1-pass2.md``
or ``-pass3.md`` (D-044 items 6-11; item 12, ``WorkflowEngine``
keyword-only, is pinned in the workflows suite). The HITL tests drive the real ``API``
through ``_HitlProbe`` (typed gated ``transfer``, ungated ``check_balance``)
from ``test_advance_driver.py``.

- D-045: a refused call that the approver approves on a later ask and that
  then runs leaves no ``refused_actions`` entry; the entry is worded as a
  fact, never as final; two refused calls are two entries; nested and list
  parameters are redacted; only a strict ``True`` from the approver grants.
- ADaPT ``operator`` is a run output: caller context cannot seed it.
- D-046: ``final_answer`` is no answer source and no terminal state asks for
  it, so a model bulk reply cannot replace the final reply.
- D-047: every ``*States`` class names every state of its FSM.
- The pins the reviewer's mutation run found missing: the budget error cites
  the pattern's own ``max_iterations`` and chains core's error on both
  loops; a speaking initial state's greeting is part of both loops' output.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from fsm_llm import API
from fsm_llm.agents import ADaPTAgent, AgentConfig, ChainStep, ReactAgent
from fsm_llm.agents import constants as agent_constants
from fsm_llm.agents.base import BaseAgent
from fsm_llm.agents.constants import ContextKeys
from fsm_llm.agents.definitions import AgentResult
from fsm_llm.agents.exceptions import BudgetExhaustedError
from fsm_llm.agents.fsm_definitions import (
    build_adapt_fsm,
    build_debate_fsm,
    build_evalopt_fsm,
    build_maker_checker_fsm,
    build_orchestrator_fsm,
    build_plan_execute_fsm,
    build_prompt_chain_fsm,
    build_react_fsm,
    build_reflexion_fsm,
    build_rewoo_fsm,
    build_self_consistency_fsm,
)
from fsm_llm.agents.parallel_react import build_parallel_react_fsm
from fsm_llm.definitions import (
    BulkExtractionRequest,
    ClassificationResult,
    DataExtractionResponse,
    FieldExtractionResponse,
    ResponseGenerationResponse,
    RunBudgetExceededError,
)
from fsm_llm.llm import LLMInterface
from tests.conftest import PromptGroundedLLM
from tests.test_fsm_llm_agents.test_advance_driver import (
    _CALL,
    _HITL_BUILDERS,
    _HITL_TASK,
    _OTHER_CALL,
    _SECRET,
    _TASK,
    _conclude_prompt,
    _endless_fsm,
    _HitlLLM,
    _HitlProbe,
    _Probe,
    _registry,
)

_REFUSED_KEY = "refused_actions"
# Literal on purpose (the tests must fail on the parent for what the run
# does): the record of the gated _CALL and of _OTHER_CALL.
_RECORD = (
    "transfer({'account': 'A-17', 'amount': 250}): was refused by the human "
    "approver and was not performed."
)
_OTHER_RECORD = (
    "transfer({'account': 'X-99', 'amount': 9999}): was refused by the human "
    "approver and was not performed."
)


def _deny_first(asks: list[dict[str, Any]]) -> Any:
    """An approver that refuses the first ask and approves every later one."""

    def decide(request: Any) -> bool:
        asks.append(dict(request.parameters))
        return len(asks) > 1

    return decide


class _RetryLLM(_HitlLLM):
    """Re-selects the denied ``transfer`` call unchanged (live qwen3.5:4b
    behaviour, 3 asks per denied React run); concludes once it ran."""

    def _grounded(self, name: str, text: str) -> object | None:
        done = "Result: Transferred" in text
        if name == "tool_name":
            return None if done else "transfer"
        if name == "tool_input":
            return None if done else dict(_CALL)
        if name == "should_terminate":
            return True if done else None
        return None


class _SwitchCallLLM(_HitlLLM):
    """After a denial of ``_CALL`` selects ``transfer(_OTHER_CALL)``; after a
    denial of that one, ``check_balance``; concludes on a tool result."""

    def _grounded(self, name: str, text: str) -> object | None:
        done = "Result: Transferred" in text or "Result: Balance" in text
        if "denied the call transfer({'account': 'X-99'" in text:
            tool, call = "check_balance", {"account": "A-17"}
        elif "denied the call transfer" in text:
            tool, call = "transfer", dict(_OTHER_CALL)
        else:
            tool, call = "transfer", dict(_CALL)
        if name == "tool_name":
            return None if done else tool
        if name == "tool_input":
            return None if done else call
        if name == "should_terminate":
            return True if done else None
        return None


class _NestedSecretLLM(_HitlLLM):
    """``_HitlLLM`` whose gated call nests secrets in a mapping and a list."""

    def _grounded(self, name: str, text: str) -> object | None:
        value = super()._grounded(name, text)
        if name == "tool_input" and value == _CALL:
            return {
                **_CALL,
                "auth": {"api_key": _SECRET},
                "headers": [{"password": "hunter2"}, {"note": "keep me"}],
            }
        return value


class TestRefusalRecordStaysTrue:
    """D-045. RED on the parent 097cb6f: the record said "NOT performed ...
    the refusal is final for this run" and stayed after the same call was
    approved on the next ask and ran (``probe_k.py``)."""

    @pytest.mark.parametrize("build", _HITL_BUILDERS)
    def test_deny_then_approve_of_the_same_call_leaves_no_record(
        self, monkeypatch: pytest.MonkeyPatch, build: Any
    ):
        asks: list[dict[str, Any]] = []
        probe = _HitlProbe(
            monkeypatch,
            build,
            decide=_deny_first(asks),
            llm=_RetryLLM(default_response="Done."),
        )

        result = probe.agent.run(_HITL_TASK)

        assert asks == [_CALL, _CALL]
        assert probe.kinds("transfer") == [("transfer", "A-17", 250)]
        (final,) = probe.final
        assert _REFUSED_KEY not in final
        assert _REFUSED_KEY not in result.final_context
        prompt = _conclude_prompt(probe.llm)
        assert "was refused by the human approver" not in prompt
        assert "Transferred 250 from A-17" in prompt
        assert (result.success, result.stop_reason) == (True, "answered")

    @pytest.mark.parametrize("build", _HITL_BUILDERS)
    def test_approving_a_different_call_keeps_the_refused_ones_record(
        self, monkeypatch: pytest.MonkeyPatch, build: Any
    ):
        probe = _HitlProbe(
            monkeypatch,
            build,
            decide=lambda request: request.parameters != _CALL,
            llm=_SwitchCallLLM(default_response="Done."),
        )

        probe.agent.run(_HITL_TASK)

        assert probe.kinds("transfer") == [("transfer", "X-99", 9999)]
        (final,) = probe.final
        assert final[_REFUSED_KEY] == [_RECORD]
        assert _RECORD in _conclude_prompt(probe.llm)

    @pytest.mark.parametrize("build", _HITL_BUILDERS)
    def test_two_refused_calls_are_two_records_in_order(
        self, monkeypatch: pytest.MonkeyPatch, build: Any
    ):
        # Kills mutant R6 (the driver overwrote the list instead of appending).
        probe = _HitlProbe(
            monkeypatch,
            build,
            decide=lambda request: False,
            llm=_SwitchCallLLM(default_response="Done."),
        )

        probe.agent.run(_HITL_TASK)

        assert [event[1] for event in probe.kinds("ask")] == [_CALL, _OTHER_CALL]
        assert probe.kinds("transfer") == []
        (final,) = probe.final
        assert final[_REFUSED_KEY] == [_RECORD, _OTHER_RECORD]
        prompt = _conclude_prompt(probe.llm)
        assert _RECORD in prompt and _OTHER_RECORD in prompt

    @pytest.mark.parametrize("build", _HITL_BUILDERS)
    def test_nested_and_list_secrets_are_redacted_in_the_record(
        self, monkeypatch: pytest.MonkeyPatch, build: Any
    ):
        probe = _HitlProbe(
            monkeypatch,
            build,
            gate="policy",
            decide=lambda request: False,
            llm=_NestedSecretLLM(default_response="Done."),
        )

        probe.agent.run(_HITL_TASK)

        (final,) = probe.final
        (record,) = final[_REFUSED_KEY]
        assert record == (
            "transfer({'account': 'A-17', 'amount': 250, 'auth': {'api_key': "
            "'<redacted>'}, 'headers': [{'password': '<redacted>'}, {'note': "
            "'keep me'}]}): was refused by the human approver and was not "
            "performed."
        )
        prompt = _conclude_prompt(probe.llm)
        assert record in prompt
        assert _SECRET not in prompt and "hunter2" not in prompt

    @pytest.mark.parametrize("answer", ["yes", 1, {"approved": True}])
    @pytest.mark.parametrize("build", _HITL_BUILDERS)
    def test_a_truthy_non_bool_answer_is_a_denial(
        self, monkeypatch: pytest.MonkeyPatch, build: Any, answer: Any
    ):
        # Kills mutant S11 (`is True` relaxed to truthiness).
        probe = _HitlProbe(monkeypatch, build, decide=lambda request: answer)

        probe.agent.run(_HITL_TASK)

        assert probe.kinds("transfer") == []
        assert [event[1] for event in probe.kinds("ask")] == [_CALL]
        (final,) = probe.final
        assert final[_REFUSED_KEY] == [_RECORD]

    def test_record_wording_is_a_fact(self):
        from fsm_llm.agents.handlers import refusal_record

        record = refusal_record("transfer", json.dumps(_CALL))

        assert record == _RECORD
        assert "final" not in record and "NOT" not in record


# ---------------------------------------------------------------------------
# ADaPT operator
# ---------------------------------------------------------------------------


class _AdaptLLM(LLMInterface):
    """The attempt of the root task fails, it decomposes into a succeeding
    and a failing subtask, and the model answers ``operator`` AND."""

    def __init__(self) -> None:
        self.model = "mock-model"
        self.fields: list[str] = []

    def extract_field(self, request: Any) -> FieldExtractionResponse:
        name = request.field_name
        self.fields.append(name)
        task = str((request.context or {}).get("task", ""))
        value: Any = {
            "attempt_result": f"attempt at {task}",
            "attempt_succeeded": "good" in task,
            "subtasks": ["good sub", "bad sub"],
            "operator": "AND",
        }.get(name)
        return FieldExtractionResponse(
            field_name=name,
            value=value,
            confidence=0.9,
            reasoning="m",
            is_valid=value is not None,
        )

    def extract_bulk_data(self, request: Any) -> DataExtractionResponse:
        return DataExtractionResponse(extracted_data={})

    def generate_response(self, request: Any) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(
            message="Combined reply of the run, long enough.",
            message_type="response",
            reasoning="m",
        )


class TestAdaptOperatorIsARunOutput:
    """RED on the parent: a caller-seeded ``operator`` "OR" was kept, core
    never asked the model, the failing subtask never ran and the run
    reported ``(True, "evidence")`` (``probe_adapt.py``)."""

    @pytest.mark.parametrize("seed", ["OR", "or"])
    def test_caller_operator_cannot_change_the_outcome(self, seed: str):
        llm = _AdaptLLM()
        agent = ADaPTAgent(
            config=AgentConfig(max_iterations=6, model="mock/model"),
            max_depth=1,
            llm_interface=llm,
        )

        result = agent.run("root task", initial_context={"operator": seed})

        ran = [entry["subtask"] for entry in result.final_context["subtask_results"]]
        assert ran == ["good sub", "bad sub"]
        assert (result.success, result.stop_reason) == (False, "no_result")
        assert llm.fields.count("operator") == 1
        assert result.final_context["operator"] == "AND"

    def test_operator_is_in_the_patterns_run_outputs(self):
        assert ContextKeys.OPERATOR in ADaPTAgent._run_output_keys


# ---------------------------------------------------------------------------
# final_answer
# ---------------------------------------------------------------------------


class _PlantingAnswerLLM(_HitlLLM):
    """``_HitlLLM`` whose every bulk reply writes ``final_answer``."""

    def extract_bulk_data(
        self, request: BulkExtractionRequest
    ) -> DataExtractionResponse:
        self.requests.append(("extract_bulk_data", request))
        return DataExtractionResponse(
            extracted_data={ContextKeys.FINAL_ANSWER: "PLANTED ANSWER"}
        )


def _all_builder_fsms() -> list[tuple[str, dict[str, Any]]]:
    tools = _registry([])
    chain = [
        ChainStep(
            step_id="a",
            name="A",
            extraction_instructions="e",
            response_instructions="r",
        ),
        ChainStep(
            step_id="b",
            name="B",
            extraction_instructions="e",
            response_instructions="r",
        ),
    ]
    return [
        ("react", build_react_fsm(tools, include_approval_state=True)),
        ("reflexion", build_reflexion_fsm(tools, include_approval_state=True)),
        ("parallel_react", build_parallel_react_fsm(tools)),
        ("plan_execute", build_plan_execute_fsm(tools)),
        ("rewoo", build_rewoo_fsm(tools)),
        ("orchestrator", build_orchestrator_fsm()),
        ("adapt", build_adapt_fsm(tools)),
        ("debate", build_debate_fsm()),
        ("evalopt", build_evalopt_fsm()),
        ("maker_checker", build_maker_checker_fsm("make it", "check it")),
        ("prompt_chain", build_prompt_chain_fsm(chain)),
        ("self_consistency", build_self_consistency_fsm()),
    ]


class TestFinalAnswerCannotBePlanted:
    """D-046. RED on the parent: under ``use_classification=True`` the
    ``think`` bulk reply wrote ``final_answer``, which the answer seam
    preferred over the conclude reply (``probe_hitl.py B``), and four
    terminal states asked Pass 2 for ``final_answer``."""

    @pytest.fixture
    def planting_probe(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        def classify(message: Any, context: Any = None) -> ClassificationResult:
            return ClassificationResult(reasoning="m", intent="transfer", confidence=1)

        monkeypatch.setattr(
            "fsm_llm.pipeline.Classifier",
            lambda **_: SimpleNamespace(classify=classify),
        )

        def build(decide: Any) -> _HitlProbe:
            return _HitlProbe(
                monkeypatch,
                lambda **kwargs: ReactAgent(use_classification=True, **kwargs),
                decide=decide,
                llm=_PlantingAnswerLLM(default_response="The conclude reply."),
            )

        return build

    def test_bulk_reply_does_not_replace_the_answer_of_an_approved_run(
        self, planting_probe: Any
    ):
        probe = planting_probe(lambda request: True)

        result = probe.agent.run(_HITL_TASK)

        assert len(probe.llm.calls("extract_bulk_data")) >= 1
        assert probe.kinds("transfer") == [("transfer", "A-17", 250)]
        assert result.answer == "The conclude reply."
        assert (result.success, result.stop_reason) == (True, "answered")

    def test_bulk_reply_does_not_hide_the_refusal_of_a_denied_run(
        self, planting_probe: Any
    ):
        probe = planting_probe(lambda request: False)

        result = probe.agent.run(_HITL_TASK)

        assert probe.kinds("transfer") == []
        assert result.answer == "The conclude reply."
        assert _RECORD in _conclude_prompt(probe.llm)

    def test_conclude_request_does_not_ask_for_final_answer(self):
        probe = _Probe()

        probe.agent.run(_TASK)

        # Core renders required keys humanized ("- Final answer") under
        # <information_still_needed> with "work toward collecting them".
        prompt = _conclude_prompt(probe.llm)
        assert "<information_still_needed>" not in prompt
        assert "Final answer" not in prompt

    @pytest.mark.parametrize(
        ("pattern", "fsm"), _all_builder_fsms(), ids=[n for n, _ in _all_builder_fsms()]
    )
    def test_no_state_asks_for_final_answer(self, pattern: str, fsm: dict[str, Any]):
        for state in fsm["states"].values():
            assert ContextKeys.FINAL_ANSWER not in (
                state.get("required_context_keys") or []
            ), (pattern, state["id"])


# ---------------------------------------------------------------------------
# *States classes
# ---------------------------------------------------------------------------


def _members(owner: str) -> set[str]:
    cls = getattr(agent_constants, owner)
    return {
        value
        for name, value in vars(cls).items()
        if name.isupper() and isinstance(value, str)
    }


class TestStateClassesNameEveryState:
    """D-047. RED on the parent: 11 states had no member (the builders used
    string literals; step 20 removed the members only they would read)."""

    @pytest.mark.parametrize(
        ("owner", "pattern"),
        [
            ("AgentStates", "react"),
            ("AgentStates", "parallel_react"),
            ("ReflexionStates", "reflexion"),
            ("PlanExecuteStates", "plan_execute"),
            ("REWOOStates", "rewoo"),
            ("OrchestratorStates", "orchestrator"),
            ("ADaPTStates", "adapt"),
            ("DebateStates", "debate"),
            ("EvalOptStates", "evalopt"),
            ("MakerCheckerStates", "maker_checker"),
            ("SelfConsistencyStates", "self_consistency"),
        ],
    )
    def test_class_lists_every_state_of_its_fsm(self, owner: str, pattern: str):
        (fsm,) = [fsm for name, fsm in _all_builder_fsms() if name == pattern]
        states = set(fsm["states"])
        if pattern == "parallel_react":  # no approval state (no HITL)
            states.add(agent_constants.AgentStates.AWAIT_APPROVAL)
        assert _members(owner) == states

    def test_prompt_chain_states_are_the_prefix_and_output(self):
        (fsm,) = [fsm for name, fsm in _all_builder_fsms() if name == "prompt_chain"]
        prefix = agent_constants.PromptChainStates.STEP_PREFIX
        output = agent_constants.PromptChainStates.OUTPUT
        assert set(fsm["states"]) == {f"{prefix}0", f"{prefix}1", output}
        assert fsm["initial_state"] == f"{prefix}0"


# ---------------------------------------------------------------------------
# Run-loop pins (surviving mutants B5, B6, B6b, B9, B10)
# ---------------------------------------------------------------------------


class _OwnBudgetAgent(BaseAgent):
    """A pattern that passes its own ``max_iterations`` to the run loops,
    as Debate and PromptChain do."""

    def __init__(self, own_max: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.own_max = own_max

    def run(
        self, task: str, initial_context: dict[str, Any] | None = None
    ) -> AgentResult:
        return self._standard_run(
            task,
            _endless_fsm(),
            self._init_context(task, initial_context),
            "endless",
            max_iterations=self.own_max,
        )

    def run_stream(self, task: str) -> Any:
        return self._standard_run_stream(
            task,
            _endless_fsm(),
            self._init_context(task),
            "endless",
            max_iterations=self.own_max,
        )

    def _register_handlers(self, api: API) -> None:
        pass


class TestBudgetErrorOfTheRunLoops:
    @pytest.mark.parametrize("stream", [False, True], ids=["run", "run_stream"])
    def test_error_cites_the_patterns_own_max_iterations(self, stream: bool):
        agent = _OwnBudgetAgent(
            own_max=4,
            config=AgentConfig(max_iterations=2),
            llm_interface=PromptGroundedLLM(),
        )

        with pytest.raises(BudgetExhaustedError) as info:
            if stream:
                list(agent.run_stream(_TASK))
            else:
                agent.run(_TASK)

        assert info.value.limit == 12
        assert str(info.value) == (
            "Agent budget exhausted: iterations limit (12) reached "
            "(12 loop turns = max_iterations 4 x FSM_BUDGET_MULTIPLIER 3)"
        )

    @pytest.mark.parametrize("stream", [False, True], ids=["run", "run_stream"])
    def test_error_is_chained_to_cores_run_budget_error(self, stream: bool):
        agent = _OwnBudgetAgent(
            own_max=2,
            config=AgentConfig(max_iterations=2),
            llm_interface=PromptGroundedLLM(),
        )

        with pytest.raises(BudgetExhaustedError) as info:
            if stream:
                list(agent.run_stream(_TASK))
            else:
                agent.run(_TASK)

        cause = info.value.__cause__
        assert isinstance(cause, RunBudgetExceededError)
        assert (cause.budget, cause.limit, cause.steps_done) == ("steps", 6, 6)


def _greeting_fsm() -> dict[str, Any]:
    """A speaking initial state that hands over to a speaking terminal state."""
    return {
        "name": "greeter",
        "description": "Greets, then answers",
        "initial_state": "hello",
        "states": {
            "hello": {
                "id": "hello",
                "description": "Greeting",
                "purpose": "Greet",
                "response_instructions": "Greet the user",
                "transitions": [
                    {"target_state": "done", "description": "Always", "priority": 100}
                ],
            },
            "done": {
                "id": "done",
                "description": "Terminal",
                "purpose": "Answer",
                "response_instructions": "Give the answer",
            },
        },
    }


class _GreetingAgent(BaseAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.loop_replies: list[list[str]] = []

    def run(
        self, task: str, initial_context: dict[str, Any] | None = None
    ) -> AgentResult:
        return self._standard_run(
            task, _greeting_fsm(), self._init_context(task, initial_context), "greet"
        )

    def run_stream(self, task: str) -> Any:
        return self._standard_run_stream(
            task, _greeting_fsm(), self._init_context(task), "greet"
        )

    def _run_conversation_loop(self, *args: Any, **kwargs: Any) -> Any:
        out = super()._run_conversation_loop(*args, **kwargs)
        self.loop_replies.append(list(out[0]))
        return out

    def _register_handlers(self, api: API) -> None:
        pass


class TestGreetingOfASpeakingInitialState:
    """No shipped pattern has a speaking initial state on these loops, so a
    test pattern pins the branch (mutants B9, B10)."""

    def _agent(self) -> _GreetingAgent:
        llm = PromptGroundedLLM(
            responses={"hello": "Hello there.", "done": "All done here."}
        )
        return _GreetingAgent(config=AgentConfig(), llm_interface=llm)

    def test_run_loop_returns_the_greeting_first(self):
        agent = self._agent()

        result = agent.run(_TASK)

        assert agent.loop_replies == [["Hello there.", "All done here."]]
        assert result.answer == "All done here."

    def test_stream_yields_the_greeting_first(self):
        agent = self._agent()

        assert list(agent.run_stream(_TASK)) == ["Hello there.", "All done here."]
