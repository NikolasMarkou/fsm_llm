"""The agents run on core's bounded run loops, with no synthetic turns.

plan-2026-09-30T062855-07ad3f8c step 9 (D-014, D-028, D-030):
``BaseAgent._run_conversation_loop`` and ``_standard_run_stream`` call
``API.run_until_terminal`` / ``run_until_terminal_stream``. Every test drives
the real ``API`` with ``PromptGroundedLLM`` (a recording fake keyed by field
name) and a real tool function.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from fsm_llm import API
from fsm_llm.agents import AgentConfig, ReactAgent, ToolRegistry
from fsm_llm.agents.base import BaseAgent
from fsm_llm.agents.definitions import AgentResult
from fsm_llm.agents.exceptions import AgentTimeoutError, BudgetExhaustedError
from fsm_llm.agents.prompts import build_conclude_response_instructions
from tests.conftest import PromptGroundedLLM

_TASK = "What is the capital of France?"
_ANSWER = "The capital of France is Paris."

# should_terminate is grounded only by the tool observation ("is Paris"), so
# a run is: think (pick the tool), act (run it), think (conclude), conclude.
_FACTS: dict[str, tuple[object, str]] = {
    "tool_name": ("lookup", "capital"),
    "tool_input": ({"query": "capital of France"}, "capital"),
    "should_terminate": (True, "is Paris"),
}


def _registry(runs: list[str]) -> ToolRegistry:
    registry = ToolRegistry()

    def lookup(query: str) -> str:
        runs.append(query)
        return _ANSWER

    registry.register_function(lookup, name="lookup", description="Look up a fact")
    return registry


class _Probe:
    """A ReactAgent plus what its run did to the ``API`` it built.

    ``histories`` holds the conversation history read just before the run
    ends its conversation; ``hook_calls`` every ``_on_loop_iteration`` call as
    ``(api, conv_id, n, field_calls_so_far, state)``.
    """

    def __init__(self, **config: Any) -> None:
        self.runs: list[str] = []
        self.llm = PromptGroundedLLM(facts=_FACTS, default_response=_ANSWER)
        self.agent = ReactAgent(
            tools=_registry(self.runs),
            config=AgentConfig(**{"max_iterations": 6, **config}),
            llm_interface=self.llm,
        )
        self.apis: list[API] = []
        self.histories: list[list[dict[str, str]]] = []
        self.hook_calls: list[tuple[API, str, int, int, str]] = []
        create_api = self.agent._create_api
        hook = self.agent._on_loop_iteration

        def _create(fsm_def: dict[str, Any]) -> API:
            api = create_api(fsm_def)
            end = api.end_conversation

            def _end(conv_id: str) -> None:
                self.histories.append(api.get_conversation_history(conv_id))
                end(conv_id)

            api.end_conversation = _end  # type: ignore[method-assign]
            self.apis.append(api)
            return api

        def _hook(api: API, conv_id: str, iteration: int) -> None:
            self.hook_calls.append(
                (
                    api,
                    conv_id,
                    iteration,
                    len(self.llm.calls("extract_field")),
                    api.get_current_state(conv_id),
                )
            )
            hook(api, conv_id, iteration)

        self.agent._create_api = _create  # type: ignore[method-assign]
        self.agent._on_loop_iteration = _hook  # type: ignore[method-assign]

    def prompts(self) -> list[str]:
        """The system prompt and user message of every LLM request."""
        texts: list[str] = []
        for _, request in self.llm.requests:
            texts += [request.system_prompt, request.user_message]
        return texts


@pytest.fixture
def converse_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every ``API.converse`` / ``converse_stream`` call (calls pass through)."""
    calls: list[str] = []
    converse, converse_stream = API.converse, API.converse_stream

    def _converse(self: API, *args: Any, **kwargs: Any) -> str:
        calls.append("converse")
        return converse(self, *args, **kwargs)

    def _converse_stream(self: API, *args: Any, **kwargs: Any) -> Any:
        calls.append("converse_stream")
        return converse_stream(self, *args, **kwargs)

    monkeypatch.setattr(API, "converse", _converse)
    monkeypatch.setattr(API, "converse_stream", _converse_stream)
    return calls


class TestNoSyntheticTurns:
    """RED on the parent: each of these failed while the loops sent "Continue."."""

    def test_run_history_holds_no_user_exchange_and_no_state_marker(self):
        probe = _Probe()

        result = probe.agent.run(_TASK)

        assert (result.success, probe.runs) == (True, ["capital of France"])
        (history,) = probe.histories
        assert history == [{"system": _ANSWER}]
        assert not any("user" in entry for entry in history)

    def test_run_stream_history_holds_no_user_exchange_and_no_state_marker(self):
        probe = _Probe()

        out = list(probe.agent.run_stream(_TASK))

        assert "".join(out) == _ANSWER
        (history,) = probe.histories
        assert history == [{"system": _ANSWER}]

    def test_run_never_calls_converse(self, converse_calls: list[str]):
        probe = _Probe()

        result = probe.agent.run(_TASK)

        assert result.answer == _ANSWER
        assert converse_calls == []

    def test_run_stream_never_calls_converse(self, converse_calls: list[str]):
        probe = _Probe()

        assert "".join(probe.agent.run_stream(_TASK)) == _ANSWER
        assert converse_calls == []

    def test_a_tool_iteration_makes_no_skipped_response_call(self):
        probe = _Probe()

        probe.agent.run(_TASK)

        responses = probe.llm.calls("generate_response")
        skipped = [r for r in responses if r.skip_generation]
        # One skip request is the greeting of the silent `think` state (kept,
        # D-028); think, act and the second think make none. One real reply.
        assert len(skipped) == 1
        assert len(responses) == 2
        assert [r.field_name for r in probe.llm.calls("extract_field")] == [
            "tool_name",
            "tool_input",
            "should_terminate",
            "tool_name",
            "tool_input",
            "should_terminate",
        ]

    @pytest.mark.parametrize("stream", [False, True], ids=["run", "run_stream"])
    def test_no_prompt_contains_continue(self, stream: bool):
        probe = _Probe()

        if stream:
            list(probe.agent.run_stream(_TASK))
        else:
            probe.agent.run(_TASK)

        assert probe.llm.calls("extract_field") and probe.llm.calls("generate_response")
        # The conclude instructions still carry their own "ignore 'Continue.'"
        # sentence until plan step 10 rewords them; nothing else may name it
        # (no user message line, no history entry, no <original_input>).
        own_wording = build_conclude_response_instructions()
        texts = [text.replace(own_wording, "") for text in probe.prompts()]
        assert not [text for text in texts if "Continue." in text]
        # A step sends no user message at all.
        turn_requests = [
            request
            for kind, request in probe.llm.requests
            if kind == "extract_field" or not request.skip_generation
        ]
        assert {request.user_message for request in turn_requests} == {""}

    def test_loop_returns_only_the_replies_of_speaking_states(self):
        probe = _Probe()
        seen: list[list[str]] = []
        loop = probe.agent._run_conversation_loop

        def _loop(*args: Any, **kwargs: Any) -> tuple[list[str], dict[str, Any], int]:
            out = loop(*args, **kwargs)
            seen.append(out[0])
            return out

        probe.agent._run_conversation_loop = _loop  # type: ignore[method-assign]

        probe.agent.run(_TASK)

        assert seen == [[_ANSWER]]


class TestLoopHook:
    """``_on_loop_iteration`` is core's ``before_step``: same arguments, same
    moment (before the step), as the agents' own loop gave it."""

    def test_hook_runs_once_before_each_step(self):
        probe = _Probe()

        result = probe.agent.run(_TASK)

        (api,) = probe.apis
        assert {call[0] for call in probe.hook_calls} == {api}
        assert len({call[1] for call in probe.hook_calls}) == 1
        # (step number, field calls made before it, state it is about to run)
        assert [call[2:] for call in probe.hook_calls] == [
            (1, 0, "think"),
            (2, 3, "act"),
            (3, 3, "think"),
        ]
        assert result.trace.total_iterations == 2

    def test_hook_runs_once_before_each_streamed_step(self):
        probe = _Probe()

        list(probe.agent.run_stream(_TASK))

        assert [call[2:] for call in probe.hook_calls] == [
            (1, 0, "think"),
            (2, 3, "act"),
            (3, 3, "think"),
        ]


def _endless_fsm() -> dict[str, Any]:
    """Two silent states that hand over to each other; ``done`` never passes."""

    def _state(name: str, other: str) -> dict[str, Any]:
        return {
            "id": name,
            "description": f"{name} state",
            "purpose": "Keep going",
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": "done",
                    "description": "Never",
                    "priority": 10,
                    "conditions": [
                        {"description": "never true", "logic": {"==": [1, 2]}}
                    ],
                },
                {"target_state": other, "description": "Always", "priority": 100},
            ],
        }

    return {
        "name": "endless",
        "description": "A run that never ends",
        "initial_state": "ping",
        "states": {
            "ping": _state("ping", "pong"),
            "pong": _state("pong", "ping"),
            "done": {
                "id": "done",
                "description": "Terminal",
                "purpose": "Finish",
                "response_instructions": "Say the answer",
            },
        },
    }


class _EndlessAgent(BaseAgent):
    """A pattern whose FSM never reaches its terminal state."""

    def __init__(self, config: AgentConfig, **api_kwargs: Any) -> None:
        super().__init__(config, **api_kwargs)
        self.steps: list[tuple[int, str]] = []
        self.slow_step: int | None = None

    def run(
        self, task: str, initial_context: dict[str, Any] | None = None
    ) -> AgentResult:
        return self._standard_run(
            task, _endless_fsm(), self._init_context(task, initial_context), "endless"
        )

    def run_stream(self, task: str) -> Any:
        return self._standard_run_stream(
            task, _endless_fsm(), self._init_context(task), "endless"
        )

    def _register_handlers(self, api: API) -> None:
        pass

    def _on_loop_iteration(self, api: API, conv_id: str, iteration: int) -> None:
        self.steps.append((iteration, api.get_current_state(conv_id)))
        if iteration == self.slow_step:
            time.sleep(0.05)


class TestBudgetsAreCores:
    """Parity with the agents' own loop: same ceiling, same turn count, same
    public errors and messages. The budgets themselves are core's."""

    @pytest.mark.parametrize("stream", [False, True], ids=["run", "run_stream"])
    def test_endless_run_raises_budget_error_at_the_same_ceiling(
        self, stream: bool, converse_calls: list[str]
    ):
        agent = _EndlessAgent(
            AgentConfig(max_iterations=2), llm_interface=PromptGroundedLLM()
        )

        with pytest.raises(BudgetExhaustedError) as info:
            if stream:
                list(agent.run_stream(_TASK))
            else:
                agent.run(_TASK)

        # max_iterations 2 x FSM_BUDGET_MULTIPLIER 3 = 6 steps, then the error.
        assert agent.steps == [
            (1, "ping"),
            (2, "pong"),
            (3, "ping"),
            (4, "pong"),
            (5, "ping"),
            (6, "pong"),
        ]
        assert info.value.limit == 6
        assert str(info.value) == (
            "Agent budget exhausted: iterations limit (6) reached "
            "(6 loop turns = max_iterations 2 x FSM_BUDGET_MULTIPLIER 3)"
        )
        assert converse_calls == []

    @pytest.mark.parametrize("stream", [False, True], ids=["run", "run_stream"])
    def test_wall_clock_spent_between_steps_raises_timeout(self, stream: bool):
        agent = _EndlessAgent(
            AgentConfig(max_iterations=50, timeout_seconds=0.02),
            llm_interface=PromptGroundedLLM(),
        )
        agent.slow_step = 2

        with pytest.raises(AgentTimeoutError) as info:
            if stream:
                list(agent.run_stream(_TASK))
            else:
                agent.run(_TASK)

        # Step 2 ran (a started step is never cut short); step 3 never began.
        assert [n for n, _ in agent.steps] == [1, 2]
        assert info.value.timeout_seconds == 0.02
        assert str(info.value) == "Agent timed out after 0.0 seconds"

    def test_wall_clock_already_spent_raises_timeout_before_any_step(self):
        agent = _EndlessAgent(
            AgentConfig(timeout_seconds=5.0), llm_interface=PromptGroundedLLM()
        )
        api = agent._create_api(_endless_fsm())

        with pytest.raises(AgentTimeoutError) as info:
            agent._run_conversation_loop(
                api, agent._init_context(_TASK), time.monotonic() - 6.0, "endless"
            )

        assert agent.steps == []
        assert str(info.value) == "Agent timed out after 5.0 seconds"
        assert api.list_active_conversations() == []
