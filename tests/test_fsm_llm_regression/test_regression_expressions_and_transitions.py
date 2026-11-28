"""Regression tests for plan 5 verified bugs in fsm_llm."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# ── B1: converse() catches all ValueError ─────────────────────


class TestConverseValueErrorCatchAll:
    """B1: converse() should not replace arbitrary ValueErrors with 'Conversation not found'."""

    def test_internal_valueerror_preserved(self):
        """A ValueError from process_message should propagate with original message."""
        from fsm_llm.engine.api.conversation import API

        api = API.__new__(API)
        api._stack_lock = __import__("threading").Lock()
        api.conversation_stacks = {"conv-1": [MagicMock(fsm_conversation_id="conv-1")]}
        api._last_accessed = {}

        mock_manager = MagicMock()
        mock_manager.process_message.side_effect = ValueError(
            "context_update must be a dictionary"
        )
        api.fsm_manager = mock_manager

        with pytest.raises(ValueError, match="context_update must be a dictionary"):
            api.converse("hello", "conv-1")


# ── B2: max_history_size=0 never trims ────────────────────────


class TestHistorySizeZero:
    """B2: Conversation with max_history_size=0 should keep exchanges empty."""

    def test_exchanges_dont_accumulate_at_size_zero(self):
        from fsm_llm.core.definitions import Conversation

        conv = Conversation(max_history_size=0)
        conv.add_user_message("hello")
        conv.add_system_message("hi")
        conv.add_user_message("how are you")
        conv.add_system_message("good")
        # With max_history_size=0, exchanges should not accumulate
        assert len(conv.exchanges) == 0


# ── B3: priority > 500 always BLOCKED ─────────────────────────


class TestHighPriorityTransition:
    """B3: A single unconditional transition with priority > 500 should not be BLOCKED."""

    def test_single_transition_priority_600_is_deterministic(self):
        from fsm_llm.core.definitions import State, Transition
        from fsm_llm.core.transition_evaluator import (
            TransitionEvaluator,
            TransitionEvaluatorConfig,
        )

        evaluator = TransitionEvaluator(config=TransitionEvaluatorConfig())
        state = State(
            id="s1",
            description="Test state",
            purpose="Testing",
            transitions=[
                Transition(target_state="s2", description="Go forward", priority=600)
            ],
        )
        from fsm_llm.core.definitions import FSMContext

        result = evaluator.evaluate_transitions(state, FSMContext())
        assert result.result_type.value == "deterministic", (
            f"Expected deterministic but got {result.result_type.value}"
        )

    def test_single_transition_priority_900_is_deterministic(self):
        from fsm_llm.core.definitions import State, Transition
        from fsm_llm.core.transition_evaluator import (
            TransitionEvaluator,
            TransitionEvaluatorConfig,
        )

        evaluator = TransitionEvaluator(config=TransitionEvaluatorConfig())
        state = State(
            id="s1",
            description="Test state",
            purpose="Testing",
            transitions=[Transition(target_state="s2", description="Go", priority=900)],
        )
        from fsm_llm.core.definitions import FSMContext

        result = evaluator.evaluate_transitions(state, FSMContext())
        assert result.result_type.value == "deterministic"


# ── B6: Empty message returns raw JSON ────────────────────────


class TestEmptyMessageFallback:
    """B6: Empty string message should not cause raw JSON to be returned."""

    def test_empty_message_does_not_return_raw_json(self):
        """data.get('message') == '' should NOT return raw JSON as the chat message."""
        from fsm_llm.llm import LiteLLMInterface

        interface = LiteLLMInterface.__new__(LiteLLMInterface)
        interface.model = "test"

        json_str = json.dumps({"message": "", "reasoning": "brief"})
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json_str

        result = interface._parse_response_generation_response(mock_response)
        assert result.message != json_str, (
            f"Message should not be raw JSON: {result.message!r}"
        )


# ── B7: and/or return bool instead of values ──────────────────


class TestAndOrReturnValues:
    """B7: and/or should return actual values per JsonLogic spec, not True/False."""

    def test_or_returns_first_truthy_value(self):
        from fsm_llm.core.expressions import evaluate_logic

        result = evaluate_logic({"or": [0, "default"]}, {})
        assert result == "default", f"Expected 'default', got {result!r}"

    def test_or_coalesce_pattern(self):
        from fsm_llm.core.expressions import evaluate_logic

        result = evaluate_logic({"or": [{"var": "x"}, "fallback"]}, {})
        assert result == "fallback", f"Expected 'fallback', got {result!r}"

    def test_and_returns_last_truthy_value(self):
        from fsm_llm.core.expressions import evaluate_logic

        result = evaluate_logic({"and": [1, 2, 3]}, {})
        assert result == 3, f"Expected 3, got {result!r}"

    def test_and_returns_first_falsy_value(self):
        from fsm_llm.core.expressions import evaluate_logic

        result = evaluate_logic({"and": [1, 0, 3]}, {})
        assert result == 0, f"Expected 0, got {result!r}"


# ── B8: > and >= crash on chained comparisons ─────────────────


class TestGreaterThanChaining:
    """B8: > and >= should support chained comparisons like < and <=."""

    def test_greater_than_three_operands(self):
        from fsm_llm.core.expressions import evaluate_logic

        result = evaluate_logic({">": [3, 2, 1]}, {})
        assert result is True, f"Expected True for 3 > 2 > 1, got {result!r}"

    def test_greater_or_equal_three_operands(self):
        from fsm_llm.core.expressions import evaluate_logic

        result = evaluate_logic({">=": [3, 2, 2]}, {})
        assert result is True, f"Expected True for 3 >= 2 >= 2, got {result!r}"

    def test_greater_than_chained_false(self):
        from fsm_llm.core.expressions import evaluate_logic

        result = evaluate_logic({">": [3, 2, 5]}, {})
        assert result is False, f"Expected False for 3 > 2 > 5, got {result!r}"


# ── B9: Handler system metadata leaks into LLM prompts ────────


class TestHandlerMetadataLeak:
    """B9: handler metadata under internal-prefix keys is filtered by get_user_visible_data.

    D-003 (plan 0068b055): the bare key ``system`` is no longer special-cased;
    it is an ordinary key. Metadata belongs under ``_``/``system_``/``internal_``.
    """

    def test_system_key_filtered_from_visible_data(self):
        from fsm_llm.core.definitions import FSMContext

        ctx = FSMContext()
        ctx.data["name"] = "Alice"
        ctx.data["system_x"] = {"handlers": {"PRE_PROCESSING": [{"name": "h1"}]}}
        ctx.data["_x"] = "hidden"
        ctx.data["internal_x"] = "hidden"
        ctx.data["system"] = "visible"

        visible = ctx.get_user_visible_data()
        for hidden in ("system_x", "_x", "internal_x"):
            assert hidden not in visible, (
                f"Handler metadata under {hidden!r} should not appear in user-visible data"
            )
        assert visible == {"name": "Alice", "system": "visible"}


# ── B14: Validator cycle normalization broken ─────────────────


class TestCycleNormalization:
    """B14: Same cycle from different starting points should be deduplicated."""

    def test_cycle_dedup_from_different_starts(self):
        from fsm_llm.core.definitions import FSMDefinition, State, Transition
        from fsm_llm.core.verify import FSMValidator

        # Create a simple cycle: A -> B -> C -> A
        fsm_def = FSMDefinition(
            name="test_cycle",
            description="Test cycle detection",
            initial_state="A",
            states={
                "A": State(
                    id="A",
                    description="State A",
                    purpose="A",
                    transitions=[Transition(target_state="B", description="to B")],
                ),
                "B": State(
                    id="B",
                    description="State B",
                    purpose="B",
                    transitions=[Transition(target_state="C", description="to C")],
                ),
                "C": State(
                    id="C",
                    description="State C",
                    purpose="C",
                    transitions=[
                        Transition(target_state="A", description="to A"),
                        Transition(target_state="end", description="to end"),
                    ],
                ),
                "end": State(
                    id="end", description="End", purpose="Done", state_type="terminal"
                ),
            },
        )
        validator = FSMValidator(fsm_def.model_dump())
        cycles = validator._find_cycles()
        # Should find exactly 1 unique cycle (A->B->C), not multiple duplicates
        assert len(cycles) == 1, f"Expected 1 unique cycle, got {len(cycles)}: {cycles}"


class TestEveryCycleIsFound:
    """plan 8525c797 step 10: `_find_cycles` lists every simple cycle reachable from
    the initial state, once each, and stops at its caps on a dense graph."""

    @staticmethod
    def _validator(edges, count):
        from fsm_llm.core.verify import FSMValidator

        states = {
            f"s{i}": {
                "id": f"s{i}",
                "transitions": [
                    {"target_state": f"s{j}", "description": "go"}
                    for j in edges.get(i, [])
                ],
            }
            for i in range(count)
        }
        return FSMValidator(
            {
                "name": "g",
                "description": "d",
                "initial_state": "s0",
                "states": states,
            }
        )

    @staticmethod
    def _canon(cycle):
        start = cycle.index(min(cycle))
        return tuple(cycle[start:] + cycle[:start])

    @classmethod
    def _brute_force(cls, edges):
        """Reference: every simple cycle reachable from s0, found from its smallest state."""
        reached, pending = set(), [0]
        while pending:
            node = pending.pop()
            if node not in reached:
                reached.add(node)
                pending.extend(edges.get(node, []))
        found = set()

        def walk(start, node, path):
            for target in edges.get(node, []):
                if target == start:
                    found.add(cls._canon([f"s{i}" for i in path]))
                elif target > start and target not in path:
                    walk(start, target, [*path, target])

        for start in sorted(reached):
            walk(start, start, [start])
        return found

    def test_the_smallest_shown_miss_is_found(self):
        # The old `finished` skip lost s0 -> s3 -> s2 -> s5 -> s0.
        edges = {0: [2, 3], 1: [], 2: [5], 3: [1, 2], 4: [1], 5: [0, 1, 2, 3]}
        cycles = self._validator(edges, 6)._find_cycles()
        assert ("s0", "s3", "s2", "s5") in {self._canon(c) for c in cycles}

    def test_the_cycles_match_a_brute_force_enumeration_on_random_graphs(self):
        import random

        rng = random.Random(1)
        for _ in range(600):
            count = rng.randint(3, 6)
            edges = {
                i: [j for j in range(count) if rng.random() < 0.3] for i in range(count)
            }
            cycles = self._validator(edges, count)._find_cycles()
            expected = self._brute_force(edges)
            assert len(cycles) == len(expected), (edges, cycles)
            assert {self._canon(c) for c in cycles} == expected, edges

    def test_parallel_transitions_and_self_loops_each_give_one_cycle(self):
        edges = {0: [1, 1], 1: [0, 1]}
        cycles = self._validator(edges, 2)._find_cycles()
        assert sorted(self._canon(c) for c in cycles) == [("s0", "s1"), ("s1",)]

    def test_a_long_chain_with_a_back_edge_stays_iterative(self):
        count = 5000
        edges = {i: [i + 1] for i in range(count - 1)}
        edges[count - 1] = [0]
        cycles = self._validator(edges, count)._find_cycles()
        assert len(cycles) == 1 and len(cycles[0]) == count

    def test_a_dense_graph_is_capped_fast_and_the_info_line_says_so(self):
        import time

        from fsm_llm.core.verify.graph import (
            MAX_CYCLE_SEARCH_STEPS,
            MAX_REPORTED_CYCLES,
        )

        count = 14  # a complete digraph: far more simple cycles than any cap
        edges = {i: [j for j in range(count) if j != i] for i in range(count)}
        validator = self._validator(edges, count)
        started = time.monotonic()
        cycles, capped = validator._search_cycles()
        assert time.monotonic() - started < 5
        assert capped is True
        assert len(cycles) <= MAX_REPORTED_CYCLES
        assert len({self._canon(c) for c in cycles}) == len(cycles)
        assert len(validator._find_cycles()) == len(cycles)
        validator._detect_cycles()
        infos = [m for m in validator.result.info if m.startswith("Cycle detected")]
        assert len(infos) <= MAX_REPORTED_CYCLES
        assert any(
            "Cycle search stopped at its cap" in m and str(MAX_CYCLE_SEARCH_STEPS) in m
            for m in validator.result.info
        )

    def test_exponentially_many_dead_paths_are_stopped_by_the_work_cap(self):
        import time

        # s0 <-> s1 is the one cycle. Past s1 sits a chain of 40 diamonds that all
        # lead back to s1, a state already on the path: 2**40 paths, none a cycle,
        # and the can-return pruning keeps them all. Only the work cap ends it.
        edges = {0: [1], 1: [0, 2]}
        diamonds = 40
        for i in range(diamonds):
            head = 2 + 3 * i
            edges[head] = [head + 1, head + 2]
            edges[head + 1] = [head + 3]
            edges[head + 2] = [head + 3]
        edges[2 + 3 * diamonds] = [1]
        validator = self._validator(edges, 3 + 3 * diamonds)
        started = time.monotonic()
        cycles, capped = validator._search_cycles()
        assert time.monotonic() - started < 10
        assert capped is True
        assert cycles == [["s0", "s1"]]

    def test_a_search_that_completes_is_not_flagged_as_capped(self):
        validator = self._validator({0: [1], 1: [0]}, 2)
        assert validator._search_cycles() == ([["s0", "s1"]], False)
        validator._detect_cycles()
        assert not any("stopped at its cap" in m for m in validator.result.info)


# ── B10: Transition info ignored in response prompt ───────────


class TestTransitionInfoInPrompt:
    """B10: _build_final_state_context_section should include transition info."""

    def test_transition_info_included_when_transition_occurred(self):
        from fsm_llm.context.layout import ResponseGenerationPromptBuilder
        from fsm_llm.core.definitions import State

        builder = ResponseGenerationPromptBuilder()
        state = State(id="confirm", description="Confirm details", purpose="Confirm")
        sections = builder._build_final_state_context_section(
            state, transition_occurred=True, previous_state="collect_info"
        )
        joined = "\n".join(sections)
        assert "collect_info" in joined, (
            "Previous state should appear in prompt when transition occurred"
        )


# ── B20: --version requires --fsm ────────────────────────────


class TestVersionFlag:
    """B20: --version should work without --fsm argument."""

    def test_version_without_fsm(self):
        """python -m fsm_llm --version should not require --fsm."""
        from fsm_llm.__main__ import main_cli

        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, "argv", ["prog", "--version"]):
                main_cli()
        # Should exit 0 (success) not 2 (argparse error)
        assert exc_info.value.code == 0, (
            f"Expected exit code 0 for --version, got {exc_info.value.code}"
        )


# ── B11: Incomplete XML tag sanitization ──────────────────────


class TestXMLTagSanitization:
    """B11: Critical prompt tags should be sanitized in user input."""

    def test_user_message_tag_sanitized(self):
        from fsm_llm.context.layout import BasePromptBuilder

        builder = BasePromptBuilder.__new__(BasePromptBuilder)
        # These tags are used in prompts and should be escaped
        text = "<user_message>injected</user_message>"
        result = builder._sanitize_text_for_prompt(text)
        assert "<user_message>" not in result, "user_message tag should be sanitized"

    def test_response_instructions_tag_sanitized(self):
        from fsm_llm.context.layout import BasePromptBuilder

        builder = BasePromptBuilder.__new__(BasePromptBuilder)
        text = "<response_instructions>evil</response_instructions>"
        result = builder._sanitize_text_for_prompt(text)
        assert "<response_instructions>" not in result

    def test_extracted_data_tag_sanitized(self):
        from fsm_llm.context.layout import BasePromptBuilder

        builder = BasePromptBuilder.__new__(BasePromptBuilder)
        text = "<extracted_data>injected</extracted_data>"
        result = builder._sanitize_text_for_prompt(text)
        assert "<extracted_data>" not in result
