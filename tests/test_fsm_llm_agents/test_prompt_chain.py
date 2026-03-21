from __future__ import annotations

"""Tests for fsm_llm_agents.prompt_chain module."""

import pytest

from fsm_llm.definitions import FSMDefinition
from fsm_llm_agents.constants import (
    ContextKeys,
    ErrorMessages,
    HandlerNames,
    PromptChainStates,
)
from fsm_llm_agents.definitions import AgentConfig, ChainStep
from fsm_llm_agents.exceptions import AgentError
from fsm_llm_agents.fsm_definitions import build_prompt_chain_fsm
from fsm_llm_agents.prompt_chain import PromptChainAgent


def _make_step(step_id: str, name: str, validation_fn=None) -> ChainStep:
    """Helper to create a ChainStep with minimal boilerplate."""
    return ChainStep(
        step_id=step_id,
        name=name,
        extraction_instructions=f"Extract data for {name}",
        response_instructions=f"Respond for {name}",
        validation_fn=validation_fn,
    )


def _make_chain(n: int) -> list[ChainStep]:
    """Create a chain of n steps."""
    return [_make_step(f"s{i}", f"Step {i}") for i in range(n)]


# -------------------------------------------------------------------------
# ChainStep model
# -------------------------------------------------------------------------


class TestChainStepModel:
    """Tests for ChainStep Pydantic model."""

    def test_basic_creation(self):
        step = ChainStep(
            step_id="outline",
            name="Generate outline",
            extraction_instructions="Extract outline",
            response_instructions="Present outline",
        )
        assert step.step_id == "outline"
        assert step.name == "Generate outline"
        assert step.extraction_instructions == "Extract outline"
        assert step.response_instructions == "Present outline"

    def test_validation_fn_default_none(self):
        step = _make_step("s0", "Step 0")
        assert step.validation_fn is None

    def test_with_validation_fn(self):
        fn = lambda ctx: len(ctx.get("result", "")) > 10  # noqa: E731
        step = ChainStep(
            step_id="check",
            name="Check step",
            extraction_instructions="Extract",
            response_instructions="Respond",
            validation_fn=fn,
        )
        assert step.validation_fn is fn
        assert step.validation_fn({"result": "long enough text"}) is True
        assert step.validation_fn({"result": "short"}) is False

    def test_validation_fn_excluded_from_dump(self):
        fn = lambda ctx: True  # noqa: E731
        step = ChainStep(
            step_id="s1",
            name="Step 1",
            extraction_instructions="Extract",
            response_instructions="Respond",
            validation_fn=fn,
        )
        data = step.model_dump()
        assert "validation_fn" not in data

    def test_all_fields_set(self):
        step = ChainStep(
            step_id="review",
            name="Review draft",
            extraction_instructions="Extract review comments",
            response_instructions="Present review feedback",
        )
        assert step.step_id == "review"
        assert step.name == "Review draft"


# -------------------------------------------------------------------------
# PromptChainAgent creation
# -------------------------------------------------------------------------


class TestPromptChainCreation:
    """Tests for PromptChainAgent initialization."""

    def test_create_with_valid_chain(self):
        chain = _make_chain(3)
        agent = PromptChainAgent(chain=chain)
        assert len(agent.chain) == 3
        assert agent.config is not None

    def test_create_with_single_step(self):
        chain = [_make_step("only", "Only step")]
        agent = PromptChainAgent(chain=chain)
        assert len(agent.chain) == 1

    def test_create_with_empty_chain_raises(self):
        with pytest.raises(AgentError, match=ErrorMessages.EMPTY_CHAIN):
            PromptChainAgent(chain=[])

    def test_create_with_custom_config(self):
        config = AgentConfig(max_iterations=20, model="gpt-4o-mini")
        chain = _make_chain(2)
        agent = PromptChainAgent(chain=chain, config=config)
        assert agent.config.max_iterations == 20
        assert agent.config.model == "gpt-4o-mini"

    def test_default_config(self):
        chain = _make_chain(2)
        agent = PromptChainAgent(chain=chain)
        assert agent.config.max_iterations == 10
        assert agent.config.temperature == 0.5

    def test_has_run_method(self):
        chain = _make_chain(2)
        agent = PromptChainAgent(chain=chain)
        assert hasattr(agent, "run")
        assert callable(agent.run)

    def test_chain_is_copied(self):
        """Agent should store its own copy of the chain list."""
        chain = _make_chain(2)
        agent = PromptChainAgent(chain=chain)
        chain.append(_make_step("extra", "Extra"))
        assert len(agent.chain) == 2

    def test_no_tool_registry_needed(self):
        """PromptChainAgent does not require a ToolRegistry."""
        chain = _make_chain(2)
        agent = PromptChainAgent(chain=chain)
        assert not hasattr(agent, "tools")

    def test_stores_api_kwargs(self):
        chain = _make_chain(1)
        agent = PromptChainAgent(chain=chain, some_kwarg="value")
        assert agent._api_kwargs == {"some_kwarg": "value"}

    def test_config_override_works(self):
        config = AgentConfig(
            model="gpt-4",
            max_iterations=15,
            timeout_seconds=60.0,
            temperature=0.9,
        )
        chain = _make_chain(2)
        agent = PromptChainAgent(chain=chain, config=config)
        assert agent.config.model == "gpt-4"
        assert agent.config.max_iterations == 15
        assert agent.config.timeout_seconds == 60.0
        assert agent.config.temperature == 0.9


# -------------------------------------------------------------------------
# Prompt Chain FSM definition
# -------------------------------------------------------------------------


class TestBuildPromptChainFsm:
    """Tests for build_prompt_chain_fsm function."""

    def test_single_step_state_count(self):
        chain = _make_chain(1)
        fsm = build_prompt_chain_fsm(chain)
        # 1 step state + output = 2 states
        assert len(fsm["states"]) == 2
        assert "step_0" in fsm["states"]
        assert "output" in fsm["states"]

    def test_three_step_state_count(self):
        chain = _make_chain(3)
        fsm = build_prompt_chain_fsm(chain)
        # 3 step states + output = 4 states
        assert len(fsm["states"]) == 4
        assert "step_0" in fsm["states"]
        assert "step_1" in fsm["states"]
        assert "step_2" in fsm["states"]
        assert "output" in fsm["states"]

    def test_dynamic_state_count_matches_chain_plus_one(self):
        for n in (1, 2, 5, 8):
            chain = _make_chain(n)
            fsm = build_prompt_chain_fsm(chain)
            assert len(fsm["states"]) == n + 1

    def test_transitions_form_correct_chain(self):
        chain = _make_chain(3)
        fsm = build_prompt_chain_fsm(chain)

        # step_0 -> step_1
        t0 = fsm["states"]["step_0"]["transitions"]
        assert len(t0) == 1
        assert t0[0]["target_state"] == "step_1"

        # step_1 -> step_2
        t1 = fsm["states"]["step_1"]["transitions"]
        assert len(t1) == 1
        assert t1[0]["target_state"] == "step_2"

        # step_2 -> output
        t2 = fsm["states"]["step_2"]["transitions"]
        assert len(t2) == 1
        assert t2[0]["target_state"] == "output"

    def test_single_step_transitions_to_output(self):
        chain = _make_chain(1)
        fsm = build_prompt_chain_fsm(chain)
        t0 = fsm["states"]["step_0"]["transitions"]
        assert len(t0) == 1
        assert t0[0]["target_state"] == "output"

    def test_output_is_terminal(self):
        chain = _make_chain(2)
        fsm = build_prompt_chain_fsm(chain)
        assert fsm["states"]["output"]["transitions"] == []

    def test_basic_structure(self):
        chain = _make_chain(2)
        fsm = build_prompt_chain_fsm(chain)
        assert fsm["name"] == "prompt_chain_agent"
        assert fsm["initial_state"] == "step_0"
        assert "persona" in fsm

    def test_fsm_is_valid_definition_single_step(self):
        chain = _make_chain(1)
        fsm = build_prompt_chain_fsm(chain)
        fsm_def = FSMDefinition(**fsm)
        assert fsm_def.name == "prompt_chain_agent"
        assert fsm_def.initial_state == "step_0"

    def test_fsm_is_valid_definition_multi_step(self):
        chain = _make_chain(4)
        fsm = build_prompt_chain_fsm(chain)
        fsm_def = FSMDefinition(**fsm)
        assert fsm_def.name == "prompt_chain_agent"
        assert len(fsm_def.states) == 5

    def test_custom_task_description(self):
        chain = _make_chain(2)
        fsm = build_prompt_chain_fsm(chain, task_description="Write an essay")
        assert fsm["description"] == "Write an essay"

    def test_default_description(self):
        chain = _make_chain(2)
        fsm = build_prompt_chain_fsm(chain)
        assert fsm["description"] == "Prompt chain agent"

    def test_step_uses_chain_instructions(self):
        step = ChainStep(
            step_id="my_step",
            name="My custom step",
            extraction_instructions="Custom extraction here",
            response_instructions="Custom response here",
        )
        fsm = build_prompt_chain_fsm([step])
        state = fsm["states"]["step_0"]
        assert state["extraction_instructions"] == "Custom extraction here"
        assert state["response_instructions"] == "Custom response here"

    def test_step_description_includes_name(self):
        step = _make_step("s0", "Generate outline")
        fsm = build_prompt_chain_fsm([step])
        state = fsm["states"]["step_0"]
        assert "Generate outline" in state["description"]

    def test_output_state_has_instructions(self):
        chain = _make_chain(1)
        fsm = build_prompt_chain_fsm(chain)
        output = fsm["states"]["output"]
        assert "extraction_instructions" in output
        assert "response_instructions" in output


# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------


class TestPromptChainConstants:
    """Tests for PromptChain-specific constants."""

    def test_prompt_chain_states_output(self):
        assert PromptChainStates.OUTPUT == "output"

    def test_prompt_chain_states_step_prefix(self):
        assert PromptChainStates.STEP_PREFIX == "step_"

    def test_prompt_chain_states_gate_prefix(self):
        assert PromptChainStates.GATE_PREFIX == "gate_"

    def test_context_keys_chain_step_index(self):
        assert hasattr(ContextKeys, "CHAIN_STEP_INDEX")
        assert isinstance(ContextKeys.CHAIN_STEP_INDEX, str)

    def test_context_keys_chain_results(self):
        assert hasattr(ContextKeys, "CHAIN_RESULTS")
        assert isinstance(ContextKeys.CHAIN_RESULTS, str)

    def test_context_keys_chain_step_result(self):
        assert hasattr(ContextKeys, "CHAIN_STEP_RESULT")
        assert isinstance(ContextKeys.CHAIN_STEP_RESULT, str)

    def test_context_keys_gate_passed(self):
        assert hasattr(ContextKeys, "GATE_PASSED")
        assert isinstance(ContextKeys.GATE_PASSED, str)

    def test_context_keys_should_terminate(self):
        assert hasattr(ContextKeys, "SHOULD_TERMINATE")
        assert isinstance(ContextKeys.SHOULD_TERMINATE, str)

    def test_handler_name_chain_gate_checker(self):
        assert HandlerNames.CHAIN_GATE_CHECKER == "ChainGateChecker"

    def test_error_message_empty_chain(self):
        assert "empty" in ErrorMessages.EMPTY_CHAIN.lower()


# -------------------------------------------------------------------------
# Internal handler logic (unit testable without LLM)
# -------------------------------------------------------------------------


class TestPromptChainHandlers:
    """Tests for internal handler methods without requiring LLM."""

    def test_gate_checker_sets_step_index(self):
        chain = _make_chain(3)
        agent = PromptChainAgent(chain=chain)
        gate_fn = agent._make_gate_checker(1)
        context = {}
        result = gate_fn(context)
        assert result[ContextKeys.CHAIN_STEP_INDEX] == 1

    def test_gate_checker_accumulates_step_result(self):
        chain = _make_chain(3)
        agent = PromptChainAgent(chain=chain)
        gate_fn = agent._make_gate_checker(1)
        context = {
            ContextKeys.CHAIN_STEP_RESULT: "Step 0 output",
            ContextKeys.CHAIN_RESULTS: [],
        }
        result = gate_fn(context)
        assert "Step 0 output" in result[ContextKeys.CHAIN_RESULTS]

    def test_gate_checker_validation_passes(self):
        chain = [
            _make_step("s0", "Step 0", validation_fn=lambda ctx: True),
            _make_step("s1", "Step 1"),
        ]
        agent = PromptChainAgent(chain=chain)
        gate_fn = agent._make_gate_checker(1)
        context = {}
        result = gate_fn(context)
        assert result.get(ContextKeys.GATE_PASSED) is True

    def test_gate_checker_validation_fails_terminates(self):
        chain = [
            _make_step("s0", "Step 0", validation_fn=lambda ctx: False),
            _make_step("s1", "Step 1"),
        ]
        agent = PromptChainAgent(chain=chain)
        gate_fn = agent._make_gate_checker(1)
        context = {}
        result = gate_fn(context)
        assert result.get(ContextKeys.GATE_PASSED) is False
        assert result.get(ContextKeys.SHOULD_TERMINATE) is True

    def test_gate_checker_no_validation_on_first_step(self):
        chain = _make_chain(3)
        agent = PromptChainAgent(chain=chain)
        gate_fn = agent._make_gate_checker(0)
        context = {}
        result = gate_fn(context)
        assert ContextKeys.GATE_PASSED not in result

    def test_iteration_limiter_increments(self):
        chain = _make_chain(2)
        agent = PromptChainAgent(chain=chain)
        limiter = agent._make_iteration_limiter()
        context = {ContextKeys.ITERATION_COUNT: 3}
        result = limiter(context)
        assert result[ContextKeys.ITERATION_COUNT] == 4

    def test_extract_answer_from_final_answer(self):
        chain = _make_chain(2)
        agent = PromptChainAgent(chain=chain)
        context = {ContextKeys.FINAL_ANSWER: "This is the final answer"}
        answer = agent._extract_answer(context, ["response1"])
        assert answer == "This is the final answer"

    def test_extract_answer_from_chain_results(self):
        chain = _make_chain(2)
        agent = PromptChainAgent(chain=chain)
        context = {ContextKeys.CHAIN_RESULTS: ["Step 0 result", "The last step result text"]}
        answer = agent._extract_answer(context, [""])
        assert answer == "The last step result text"

    def test_extract_answer_fallback_to_response(self):
        chain = _make_chain(2)
        agent = PromptChainAgent(chain=chain)
        context = {}
        answer = agent._extract_answer(context, ["", "A valid long response here"])
        assert answer == "A valid long response here"

    def test_extract_answer_default(self):
        chain = _make_chain(2)
        agent = PromptChainAgent(chain=chain)
        answer = agent._extract_answer({}, ["", ""])
        assert "could not" in answer.lower()
