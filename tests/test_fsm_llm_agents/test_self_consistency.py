from __future__ import annotations

"""Tests for fsm_llm.agents.self_consistency module."""

import ast
import inspect
import textwrap
from typing import Any

import pytest

from fsm_llm import API
from fsm_llm.agents.constants import ContextKeys, Defaults, SelfConsistencyStates
from fsm_llm.agents.definitions import AgentConfig
from fsm_llm.agents.exceptions import AgentError
from fsm_llm.agents.fsm_definitions import build_self_consistency_fsm
from fsm_llm.agents.self_consistency import SelfConsistencyAgent, _majority_vote
from fsm_llm.definitions import FSMDefinition
from tests.conftest import PromptGroundedLLM


class TestSelfConsistencyCreation:
    """Tests for SelfConsistencyAgent initialization."""

    def test_create_with_defaults(self):
        agent = SelfConsistencyAgent()
        assert agent.num_samples == Defaults.NUM_SAMPLES
        assert agent.config is not None
        assert agent.aggregation_fn is _majority_vote

    def test_create_with_custom_num_samples(self):
        agent = SelfConsistencyAgent(num_samples=7)
        assert agent.num_samples == 7

    def test_create_with_custom_aggregation_fn(self):
        custom_fn = lambda samples: max(samples, key=len)  # noqa: E731
        agent = SelfConsistencyAgent(aggregation_fn=custom_fn)
        assert agent.aggregation_fn is custom_fn

    def test_create_with_num_samples_zero_raises(self):
        with pytest.raises(AgentError, match="at least 1"):
            SelfConsistencyAgent(num_samples=0)

    def test_create_with_num_samples_negative_raises(self):
        with pytest.raises(AgentError, match="at least 1"):
            SelfConsistencyAgent(num_samples=-5)

    def test_create_with_config_override(self):
        config = AgentConfig(max_iterations=3, model="gpt-4o-mini")
        agent = SelfConsistencyAgent(config=config, num_samples=3)
        assert agent.config.max_iterations == 3
        assert agent.config.model == "gpt-4o-mini"
        assert agent.num_samples == 3

    def test_create_with_num_samples_one(self):
        agent = SelfConsistencyAgent(num_samples=1)
        assert agent.num_samples == 1

    def test_no_tool_registry_needed(self):
        """SelfConsistencyAgent does not require a ToolRegistry."""
        agent = SelfConsistencyAgent()
        assert not hasattr(agent, "tools") or agent.__dict__.get("tools") is None

    def test_has_run_method(self):
        agent = SelfConsistencyAgent()
        assert callable(getattr(agent, "run", None))


class TestSelfConsistencyFSM:
    """Tests for build_self_consistency_fsm function."""

    def test_basic_fsm_structure(self):
        fsm = build_self_consistency_fsm()
        assert fsm["name"] == "self_consistency_sample"
        assert fsm["initial_state"] == "generate"
        assert "generate" in fsm["states"]

    def test_fsm_is_valid_definition(self):
        """The generated FSM should be parseable as an FSMDefinition."""
        fsm = build_self_consistency_fsm()
        fsm_def = FSMDefinition(**fsm)
        assert fsm_def.name == "self_consistency_sample"

    def test_generate_state_is_terminal(self):
        fsm = build_self_consistency_fsm()
        assert fsm["states"]["generate"]["transitions"] == []

    def test_only_one_state(self):
        fsm = build_self_consistency_fsm()
        assert len(fsm["states"]) == 1
        assert "generate" in fsm["states"]

    def test_custom_task_description(self):
        fsm = build_self_consistency_fsm(task_description="What is 2+2?")
        assert fsm["description"] == "What is 2+2?"

    def test_default_task_description(self):
        fsm = build_self_consistency_fsm()
        assert fsm["description"] == "Self-consistency single sample"

    def test_persona_mentions_precise(self):
        fsm = build_self_consistency_fsm()
        assert "precise" in fsm["persona"].lower()

    def test_generate_state_asks_for_an_answer_line_and_extracts_nothing(self):
        # PAT-05: a terminal initial state never extracts, so its bulk
        # instructions were dead; the reply's Answer: line is what is voted on.
        fsm = build_self_consistency_fsm()
        generate = fsm["states"]["generate"]
        assert generate["extraction_instructions"] == ""
        assert "Answer:" in generate["response_instructions"]


class TestSelfConsistencyConstants:
    """Tests for self-consistency related constants."""

    def test_self_consistency_states_generate(self):
        assert SelfConsistencyStates.GENERATE == "generate"

    def test_self_consistency_states_aggregate(self):
        assert SelfConsistencyStates.AGGREGATE == "aggregate"

    def test_context_keys_samples(self):
        assert ContextKeys.SAMPLES == "samples"

    def test_context_keys_aggregated_answer(self):
        assert ContextKeys.AGGREGATED_ANSWER == "aggregated_answer"

    def test_defaults_num_samples(self):
        assert Defaults.NUM_SAMPLES == 5

    def test_defaults_sample_temperature_range(self):
        assert isinstance(Defaults.SAMPLE_TEMPERATURE_RANGE, tuple)
        assert len(Defaults.SAMPLE_TEMPERATURE_RANGE) == 2
        low, high = Defaults.SAMPLE_TEMPERATURE_RANGE
        assert low < high


class TestMajorityVote:
    """Tests for the _majority_vote default aggregation function."""

    def test_majority_vote_simple(self):
        result = _majority_vote(["Paris", "Paris", "London"])
        assert result == "Paris"

    def test_majority_vote_all_same(self):
        result = _majority_vote(["yes", "yes", "yes"])
        assert result == "yes"

    def test_majority_vote_empty_list(self):
        result = _majority_vote([])
        assert result == ""

    def test_majority_vote_whitespace_normalization(self):
        result = _majority_vote(["  Paris ", "Paris", " Paris"])
        assert result == "Paris"

    def test_majority_vote_single_element(self):
        result = _majority_vote(["only"])
        assert result == "only"

    def test_majority_vote_skips_empty_strings(self):
        result = _majority_vote(["", "", "answer", "answer"])
        assert result == "answer"

    def test_majority_vote_all_empty(self):
        result = _majority_vote(["", "  ", ""])
        assert result == ""


# ---------------------------------------------------------------------------
# Plan 07ad3f8c step 13: a sample is one turn (the dead loop is gone)
# ---------------------------------------------------------------------------


class _SampleProbe:
    """A SelfConsistencyAgent on a recording fake, plus every ``API`` entry
    its samples called and each sample conversation's history."""

    REPLY = "Canberra is the capital.\nAnswer: Canberra"

    def __init__(self, monkeypatch: pytest.MonkeyPatch, num_samples: int = 3) -> None:
        self.llm = PromptGroundedLLM(default_response=self.REPLY)
        self.calls: list[str] = []
        self.histories: list[list[dict[str, str]]] = []
        for name in (
            "start_conversation",
            "converse",
            "converse_stream",
            "advance",
            "advance_stream",
            "run_until_terminal",
            "run_until_terminal_stream",
        ):
            self._record(monkeypatch, name)
        end = API.end_conversation
        probe = self

        def _end(api: API, conversation_id: str) -> None:
            probe.histories.append(api.get_conversation_history(conversation_id))
            end(api, conversation_id)

        monkeypatch.setattr(API, "end_conversation", _end)
        self.agent = SelfConsistencyAgent(
            config=AgentConfig(model="mock/model"),
            num_samples=num_samples,
            llm_interface=self.llm,
        )

    def _record(self, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
        real = getattr(API, name)

        def _entry(api: API, *args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            return real(api, *args, **kwargs)

        monkeypatch.setattr(API, name, _entry)


class TestSampleIsOneTurn:
    """A sample is the greeting reply of the terminal initial state: one
    ``start_conversation``, one reply request, and no further turn. On d4b1626
    a dead ``iteration < 5`` loop stood ready to send "Continue." turns."""

    def test_each_sample_makes_exactly_one_turn(self, monkeypatch):
        probe = _SampleProbe(monkeypatch, num_samples=3)

        result = probe.agent.run("What is the capital of Australia?")

        assert probe.calls == ["start_conversation"] * 3
        assert [kind for kind, _ in probe.llm.requests] == ["generate_response"] * 3
        assert result.final_context[ContextKeys.SAMPLES] == [_SampleProbe.REPLY] * 3
        assert result.trace.total_iterations == 3
        assert (result.success, result.stop_reason) == (True, "answered")
        assert result.answer == _SampleProbe.REPLY

    def test_sample_history_is_the_reply_alone(self, monkeypatch):
        probe = _SampleProbe(monkeypatch, num_samples=2)

        probe.agent.run("What is the capital of Australia?")

        assert probe.histories == [[{"system": _SampleProbe.REPLY}]] * 2

    def test_sample_request_carries_no_synthetic_message(self, monkeypatch):
        probe = _SampleProbe(monkeypatch, num_samples=1)

        probe.agent.run("What is the capital of Australia?")

        (request,) = probe.llm.calls("generate_response")
        assert request.user_message == ""
        assert "Continue" not in request.system_prompt
        assert "What is the capital of Australia?" in request.system_prompt

    def test_generate_single_holds_no_loop(self):
        source = textwrap.dedent(
            inspect.getsource(SelfConsistencyAgent._generate_single)
        )
        nodes = list(ast.walk(ast.parse(source)))

        assert not [n for n in nodes if isinstance(n, ast.While | ast.For)]
        called = {
            n.func.attr
            for n in nodes
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        assert {"start_conversation", "end_conversation"} <= called
        assert not called & {"converse", "advance", "run_until_terminal"}

    def test_builder_keeps_its_signature_and_loads(self):
        # examples/agents/workflow_agent/run.py builds the FSM by keyword.
        assert list(inspect.signature(build_self_consistency_fsm).parameters) == [
            "task_description"
        ]
        fsm = build_self_consistency_fsm(task_description="x")

        definition = FSMDefinition(**fsm)
        assert definition.initial_state == SelfConsistencyStates.GENERATE
        assert definition.states[SelfConsistencyStates.GENERATE].transitions == []
        api = API.from_definition(fsm, llm_interface=PromptGroundedLLM())
        conv_id, greeting = api.start_conversation({"task": "x"})
        assert greeting == "ok"
        assert api.has_conversation_ended(conv_id) is True
