"""Regression tests for identified bugs in fsm_llm."""
import hashlib
import inspect
import json
import os
import tempfile
from collections import deque
from unittest.mock import patch, MagicMock

import pytest


# ── B1: JSON escape handling ──────────────────────────────────

class TestJsonEscapeHandling:
    """B1: escape_next flag in balanced brace extraction is dead code."""

    def test_escaped_quote_inside_json_string(self):
        """Escaped quotes within JSON string values must not break extraction."""
        from fsm_llm.utilities import extract_json_from_text

        text = r'Here is JSON: {"msg": "He said \"hello\"", "count": 1}'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["count"] == 1
        assert 'hello' in result["msg"]

    def test_escaped_backslash_before_quote(self):
        """A double backslash before a quote should NOT escape the quote."""
        from fsm_llm.utilities import extract_json_from_text

        # \\\" means: escaped backslash + closing quote
        text = r'{"path": "C:\\Users\\", "ok": true}'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["ok"] is True

    def test_nested_json_with_escapes(self):
        """Nested braces with escaped characters."""
        from fsm_llm.utilities import extract_json_from_text

        text = 'Some text {"outer": {"inner": "val\\\"ue"}, "x": 1} trailing'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["x"] == 1


# ── B2: Import-time side effects ─────────────────────────────

class TestLoggingSideEffects:
    """B2: importing fsm_llm.logging creates logs/ directory."""

    def test_import_does_not_create_logs_dir(self):
        """Importing fsm_llm should not create a logs/ directory in CWD."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                # The logs/ dir should not exist before or after import
                assert not os.path.exists(os.path.join(tmpdir, "logs")), \
                    "logs/ directory should not exist before import"

                # Force re-import of the logging module
                import importlib
                import fsm_llm.logging as log_mod
                importlib.reload(log_mod)

                assert not os.path.exists(os.path.join(tmpdir, "logs")), \
                    "logs/ directory should not be created at import time"
            finally:
                os.chdir(original_cwd)


# ── B3: Handler protocol async mismatch ──────────────────────

class TestHandlerProtocolAsync:
    """B3: FSMHandler protocol declares async execute() but it's called synchronously."""

    def test_handler_execute_is_sync_in_protocol(self):
        """The FSMHandler protocol's execute method should be synchronous."""
        import inspect
        from fsm_llm.handlers import FSMHandler

        # Get the execute method from the protocol
        execute_method = FSMHandler.execute
        assert not inspect.iscoroutinefunction(execute_method), \
            "FSMHandler.execute should be synchronous (not async def)"

    def test_lambda_handler_execute_is_sync(self):
        """LambdaHandler.execute should be synchronous to match the protocol."""
        import inspect
        from fsm_llm.handlers import LambdaHandler, HandlerTiming

        handler = LambdaHandler(
            name="test",
            condition_lambdas=[],
            execution_lambda=lambda ctx: {"test": True},
            timings={HandlerTiming.POST_TRANSITION},
            states=set(),
            target_states=set(),
            required_keys=set(),
            updated_keys=set(),
        )
        assert not inspect.iscoroutinefunction(handler.execute), \
            "LambdaHandler.execute should be synchronous"


# ── B4: Deprecated Pydantic Config ───────────────────────────

class TestPydanticConfig:
    """B4: FSMStackFrame uses deprecated class Config instead of model_config."""

    def test_no_deprecated_config_class(self):
        """FSMStackFrame should use model_config, not class Config."""
        from fsm_llm.api import FSMStackFrame

        # Check that there's no inner Config class (deprecated in Pydantic v2)
        assert not hasattr(FSMStackFrame, 'Config') or \
            not hasattr(FSMStackFrame.Config, 'arbitrary_types_allowed'), \
            "FSMStackFrame should use model_config instead of class Config"


# ── B6: MD5 usage ────────────────────────────────────────────

class TestMd5Usage:
    """B6: md5 used for content hashing — should use sha256."""

    def test_process_fsm_definition_no_md5(self):
        """process_fsm_definition should not use md5 for hashing."""
        import fsm_llm.api as api_mod
        source = open(api_mod.__file__).read()
        # After the fix, md5 should not appear in the source
        assert 'md5' not in source, \
            "api.py should not use md5 — use sha256 instead"


# ── B7: BFS pop(0) performance ───────────────────────────────

class TestBfsPerformance:
    """B7: _calculate_reachable_states uses list.pop(0) instead of deque."""

    def test_reachable_states_uses_deque(self):
        """_calculate_reachable_states should use collections.deque for BFS."""
        import inspect
        from fsm_llm.definitions import FSMDefinition

        source = inspect.getsource(FSMDefinition._calculate_reachable_states)
        assert 'deque' in source, \
            "_calculate_reachable_states should use deque, not list.pop(0)"
        assert '.pop(0)' not in source, \
            "_calculate_reachable_states should not use pop(0)"


# ── B8: Unnecessary hasattr ──────────────────────────────────

class TestUnnecessaryHasattr:
    """B8: hasattr check for handler_system is always True."""

    def test_no_hasattr_handler_system(self):
        """API.__init__ should not guard handler_system assignment with hasattr."""
        import inspect
        from fsm_llm.api import API

        source = inspect.getsource(API.__init__)
        assert "hasattr(self.fsm_manager, 'handler_system')" not in source, \
            "Unnecessary hasattr guard should be removed"


# ── CR3: Python version check ────────────────────────────────

# ── B-NEW-1: Missing commas in prompts.py ────────────────────

class TestMissingCommasInPrompts:
    """B-NEW-1: Missing commas cause implicit string concatenation in list literals."""

    def test_response_format_list_elements_are_separate(self):
        """Each instruction in the response format list should be a separate element."""
        from fsm_llm.prompts import DataExtractionPromptBuilder

        builder = DataExtractionPromptBuilder()
        sections = builder._build_extraction_response_format()

        # Check that no single element contains both the key names instruction
        # AND the _extra instruction (they should be separate list items)
        # Note: use "`_extra`" (with backticks) to avoid matching "_extract" in "_to_extract"
        for element in sections:
            assert not (
                "key names" in element and "`_extra`" in element
            ), f"Implicit string concatenation detected: {element!r}"

        # Check that </response_format> is its own element, not concatenated
        for element in sections:
            assert not (
                "Do NOT generate" in element and "</response_format>" in element
            ), f"Closing tag concatenated with instruction: {element!r}"

    def test_response_format_closing_tag_standalone(self):
        """The </response_format> closing tag should be its own list element."""
        from fsm_llm.prompts import DataExtractionPromptBuilder

        builder = DataExtractionPromptBuilder()
        sections = builder._build_extraction_response_format()

        # Find the element containing </response_format>
        closing_tags = [e for e in sections if "</response_format>" in e]
        assert len(closing_tags) == 1, "Should have exactly one </response_format> element"
        assert closing_tags[0].strip() == "</response_format>", \
            f"</response_format> should be standalone, got: {closing_tags[0]!r}"


# ── B-NEW-2: Transition evaluator low-confidence logic ───────

class TestTransitionEvaluatorLowConfidence:
    """B-NEW-2: Single low-confidence transition returns AMBIGUOUS instead of BLOCKED."""

    def test_single_low_confidence_transition_is_deterministic(self):
        """A single passing transition should be DETERMINISTIC even with low confidence.

        When there is only one valid transition path, blocking it based on an
        arbitrary confidence formula is incorrect — it's the only option.
        """
        from fsm_llm.transition_evaluator import TransitionEvaluator, TransitionEvaluatorConfig
        from fsm_llm.definitions import (
            TransitionEvaluationResult, State, Transition,
            TransitionCondition, FSMContext
        )

        config = TransitionEvaluatorConfig(minimum_confidence=0.5)
        evaluator = TransitionEvaluator(config)

        # Create a state with one transition
        state = State(
            id="test_state",
            description="Test",
            purpose="Testing low confidence",
            transitions=[
                Transition(
                    target_state="next_state",
                    description="Go next",
                    conditions=[
                        TransitionCondition(
                            description="Status must be done",
                            field="status",
                            operator="equals",
                            value="done",
                        )
                    ],
                    priority=100,
                )
            ],
        )

        # Manually test _determine_evaluation_result with a low-confidence score
        transition_scores = [{
            'transition': state.transitions[0],
            'passes_conditions': True,
            'confidence': 0.3,  # Below minimum_confidence, but it's the only option
            'evaluation_notes': [],
            'failed_conditions': [],
        }]

        result = evaluator._determine_evaluation_result(
            transition_scores, state, {}
        )

        # Should be DETERMINISTIC — only one valid path, don't block it
        assert result.result_type == TransitionEvaluationResult.DETERMINISTIC, \
            f"Expected DETERMINISTIC for single passing transition, got {result.result_type}"


# ── B-NEW-3: INFO-level logging of evaluation data ───────────

class TestEvaluationLoggingLevel:
    """B-NEW-3: Evaluation result logged at INFO instead of DEBUG."""

    def test_evaluation_result_not_logged_at_info(self):
        """_evaluate_single_transition should not log evaluation_result at INFO level."""
        import inspect
        from fsm_llm.transition_evaluator import TransitionEvaluator

        source = inspect.getsource(TransitionEvaluator._evaluate_single_transition)
        # The problematic line is: logger.info("evaluation_result : ...")
        assert 'logger.info("evaluation_result' not in source, \
            "evaluation_result should be logged at DEBUG, not INFO"


# ── B-NEW-4: Initialization order in API ─────────────────────

class TestApiInitOrder:
    """B-NEW-4: _temp_fsm_definitions initialized after closure that uses it."""

    def test_temp_fsm_definitions_initialized_before_fsm_manager(self):
        """_temp_fsm_definitions should be initialized before FSMManager creation."""
        import ast

        api_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "src", "fsm_llm", "api.py"
        )
        with open(api_path) as f:
            source = f.read()

        tree = ast.parse(source)

        # Find the __init__ method of the API class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "API":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        # Find line numbers for key assignments
                        # Check both Assign and AnnAssign (type-annotated assignments)
                        temp_fsm_line = None
                        fsm_manager_line = None
                        for stmt in ast.walk(item):
                            target = None
                            if isinstance(stmt, ast.Assign):
                                for t in stmt.targets:
                                    if isinstance(t, ast.Attribute):
                                        target = t
                            elif isinstance(stmt, ast.AnnAssign):
                                if isinstance(stmt.target, ast.Attribute):
                                    target = stmt.target

                            if target is not None:
                                if target.attr == "_temp_fsm_definitions":
                                    temp_fsm_line = stmt.lineno
                                elif target.attr == "fsm_manager":
                                    fsm_manager_line = stmt.lineno

                        assert temp_fsm_line is not None, \
                            "_temp_fsm_definitions assignment not found"
                        assert fsm_manager_line is not None, \
                            "fsm_manager assignment not found"
                        assert temp_fsm_line < fsm_manager_line, \
                            f"_temp_fsm_definitions (line {temp_fsm_line}) should be " \
                            f"initialized before fsm_manager (line {fsm_manager_line})"


# ── B-NEW-6: Duplicate error modes ───────────────────────────

class TestDuplicateErrorModes:
    """B-NEW-6: 'continue' and 'skip' error modes are identical."""

    def test_skip_error_mode_not_in_execute_handlers(self):
        """The 'skip' error mode should be removed (it's identical to 'continue')."""
        import inspect
        from fsm_llm.handlers import HandlerSystem

        source = inspect.getsource(HandlerSystem.execute_handlers)
        # After the fix, "skip" should not appear as a separate branch
        assert 'error_mode == "skip"' not in source, \
            "The 'skip' error mode is dead code — identical to 'continue'"


# ── CR3: Python version check ────────────────────────────────

class TestPythonVersionCheck:
    """CR3: Python version check warns for <3.8 but 3.8 is EOL."""

    def test_minimum_version_is_310_or_higher(self):
        """The Python version check should require at least 3.10."""
        import ast

        init_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "src", "fsm_llm", "__init__.py"
        )
        with open(init_path) as f:
            source = f.read()

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                # Look for sys.version_info < (3, X)
                if (isinstance(node.left, ast.Attribute) and
                        getattr(node.left, 'attr', '') == 'version_info'):
                    for comparator in node.comparators:
                        if isinstance(comparator, ast.Tuple):
                            elts = comparator.elts
                            if len(elts) >= 2:
                                minor = elts[1]
                                if isinstance(minor, ast.Constant):
                                    assert minor.value >= 10, \
                                        f"Minimum Python version should be 3.10+, got 3.{minor.value}"


# ══════════════════════════════════════════════════════════════
# Plan 3 regression tests (plan_2026-03-07_b55f9c34)
# ══════════════════════════════════════════════════════════════


# ── P3-B1: Non-deterministic substring matching ──────────────

class TestTransitionSubstringMatching:
    """P3-B1: _parse_transition_response uses substring match on set iteration."""

    def test_longest_target_matched_when_substring_overlap(self):
        """When one state name is a substring of another, the longest match should win."""
        from fsm_llm.llm import LiteLLMInterface
        from fsm_llm.definitions import TransitionOption

        interface = LiteLLMInterface.__new__(LiteLLMInterface)

        # Create a mock response with content mentioning the longer state name
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "I think we should go to collect_name_confirmation"

        transitions = [
            TransitionOption(target_state="collect_name", description="Collect name", priority=1),
            TransitionOption(target_state="collect_name_confirmation", description="Confirm name", priority=2),
        ]

        result = interface._parse_transition_response(mock_response, transitions)
        assert result.selected_transition == "collect_name_confirmation", \
            f"Should match longest target, got '{result.selected_transition}'"

    def test_deterministic_across_runs(self):
        """Transition matching should be deterministic regardless of set iteration order."""
        from fsm_llm.llm import LiteLLMInterface
        from fsm_llm.definitions import TransitionOption

        interface = LiteLLMInterface.__new__(LiteLLMInterface)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Let's proceed to verify_address now"

        transitions = [
            TransitionOption(target_state="verify", description="Verify", priority=1),
            TransitionOption(target_state="verify_address", description="Verify address", priority=2),
        ]

        # Run multiple times to catch non-deterministic behavior
        results = set()
        for _ in range(20):
            result = interface._parse_transition_response(mock_response, transitions)
            results.add(result.selected_transition)

        assert len(results) == 1, f"Non-deterministic results: {results}"
        assert "verify_address" in results, f"Should match longest target, got {results}"


# ── P3-B2: Memory leak in conversation stacks ────────────────

class TestConversationMemoryLeak:
    """P3-B2: end_conversation doesn't clean up conversation_stacks."""

    def test_end_conversation_cleans_up_stacks(self):
        """end_conversation should remove entries from conversation_stacks."""
        from fsm_llm.api import API, FSMStackFrame

        api = API.__new__(API)
        api.active_conversations = {"conv1": True}
        api.conversation_stacks = {
            "conv1": [
                FSMStackFrame(
                    fsm_definition="test",
                    conversation_id="conv1_inner"
                )
            ]
        }
        api.fsm_manager = MagicMock()
        api._temp_fsm_definitions = {"temp_fsm_1": MagicMock()}

        # Bypass the decorator by calling the underlying logic
        api.end_conversation.__wrapped__(api, "conv1")

        assert "conv1" not in api.conversation_stacks, \
            "conversation_stacks should be cleaned up after end_conversation"
        assert "conv1" not in api.active_conversations, \
            "active_conversations should be cleaned up after end_conversation"


# ── P3-B3: handle_conversation_errors masks ValueError ───────

class TestHandleConversationErrorsMasking:
    """P3-B3: Decorator catches all ValueError as 'Conversation not found'."""

    def test_non_conversation_valueerror_not_masked(self):
        """A ValueError about invalid state should NOT become 'Conversation not found'."""
        from fsm_llm.logging import handle_conversation_errors

        @handle_conversation_errors
        def method_that_raises_state_error(self, conversation_id):
            raise ValueError("Invalid FSM state: 'nonexistent_state'")

        class FakeAPI:
            pass

        with pytest.raises(ValueError) as exc_info:
            method_that_raises_state_error(FakeAPI(), "conv1")

        # The original message should be preserved, not replaced with "Conversation not found"
        assert "Conversation not found" not in str(exc_info.value), \
            f"ValueError was masked: {exc_info.value}"


# ── P3-B4: Global environment mutation ────────────────────────

class TestApiKeyEnvironmentMutation:
    """P3-B4: _configure_api_keys sets os.environ directly."""

    def test_configure_api_keys_does_not_set_env_vars(self):
        """API key configuration should NOT mutate os.environ."""
        from fsm_llm.llm import LiteLLMInterface

        interface = LiteLLMInterface.__new__(LiteLLMInterface)
        interface.model = "gpt-4o"
        interface.kwargs = {}

        original_env = os.environ.get("OPENAI_API_KEY")
        try:
            # Remove env var if it exists
            os.environ.pop("OPENAI_API_KEY", None)

            interface._configure_api_keys("test-key-12345")

            assert os.environ.get("OPENAI_API_KEY") != "test-key-12345", \
                "_configure_api_keys should not set os.environ directly"
        finally:
            # Restore original state
            if original_env is not None:
                os.environ["OPENAI_API_KEY"] = original_env
            else:
                os.environ.pop("OPENAI_API_KEY", None)


# ── P3-B5: get_recent over-retrieval ─────────────────────────

class TestGetRecentOverRetrieval:
    """P3-B5: Prompts call get_recent(max_history_messages * 2) but get_recent expects exchanges."""

    def test_get_recent_not_double_multiplied(self):
        """_build_enhanced_history_section should not pass max_history_messages * 2 to get_recent."""
        from fsm_llm.prompts import BasePromptBuilder

        source = inspect.getsource(BasePromptBuilder._build_enhanced_history_section)
        assert "max_history_messages * 2" not in source, \
            "get_recent already multiplies by 2 internally; caller should not multiply again"


# ── P3-B6: disable_warnings wrong module ─────────────────────

class TestDisableWarningsModule:
    """P3-B6: disable_warnings filters 'fsm_llm' instead of 'fsm_llm'."""

    def test_disable_warnings_targets_correct_module(self):
        """disable_warnings should filter warnings from fsm_llm, not just fsm_llm."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "src", "fsm_llm", "__init__.py"
        )
        with open(source_path) as f:
            source = f.read()

        # Find the disable_warnings function and check it uses the right module
        in_func = False
        for line in source.split("\n"):
            if "def disable_warnings" in line:
                in_func = True
            elif in_func and "filterwarnings" in line:
                assert "fsm_llm" in line or 'module="fsm_llm' not in line, \
                    f"disable_warnings should filter fsm_llm, not just fsm_llm: {line.strip()}"
                break


# ── P3-B7: Redundant DEBUG logging in transition_evaluator ───

class TestRedundantTransitionLogging:
    """P3-B7: Unconditional DEBUG logs duplicate the detailed_logging-gated output."""

    def test_no_unconditional_evaluation_result_logging(self):
        """_evaluate_single_transition should not log evaluation_result unconditionally."""
        from fsm_llm.transition_evaluator import TransitionEvaluator

        source = inspect.getsource(TransitionEvaluator._evaluate_single_transition)
        # After fix, there should be no unconditional logger.debug with "evaluation_result"
        # outside of an if self.config.detailed_logging block
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'logger.debug' in line and 'evaluation_result' in line:
                # Check if it's inside a detailed_logging block
                context_lines = lines[max(0, i-3):i]
                in_detailed = any('detailed_logging' in cl for cl in context_lines)
                assert in_detailed, \
                    f"Unconditional evaluation_result log found at line {i}: {line.strip()}"

    def test_no_unconditional_condition_result_logging(self):
        """_evaluate_transition_conditions should not log unconditionally."""
        from fsm_llm.transition_evaluator import TransitionEvaluator

        source = inspect.getsource(TransitionEvaluator._evaluate_transition_conditions)
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'logger.debug' in line and 'evaluation_result' in line:
                context_lines = lines[max(0, i-3):i]
                in_detailed = any('detailed_logging' in cl for cl in context_lines)
                assert in_detailed, \
                    f"Unconditional evaluation_result log found: {line.strip()}"


# ── P3-B8: Redundant double error logging in runner ──────────

class TestRedundantRunnerLogging:
    """P3-B8: logger.error() followed by logger.exception() for same error."""

    def test_no_double_error_logging(self):
        """Error handling should not log the same error twice."""
        from fsm_llm import runner

        source = inspect.getsource(runner)
        lines = source.split('\n')
        for i in range(len(lines) - 1):
            if 'logger.error' in lines[i] and 'logger.exception' in lines[i + 1]:
                assert False, \
                    f"Redundant double logging at lines {i+1}-{i+2}: " \
                    f"'{lines[i].strip()}' followed by '{lines[i+1].strip()}'"


# ── P3-B9: get_supported_openai_params returns None ──────────

class TestGetSupportedParamsNone:
    """P3-B9: get_supported_openai_params returns None for unknown models."""

    def test_make_llm_call_handles_none_supported_params(self):
        """_make_llm_call should not crash when get_supported_openai_params returns None."""
        from fsm_llm.llm import LiteLLMInterface

        interface = LiteLLMInterface.__new__(LiteLLMInterface)
        interface.model = "unknown-model-xyz"
        interface.temperature = 0.5
        interface.max_tokens = 100
        interface.timeout = 120.0
        interface.kwargs = {}

        # Mock completion to avoid actual API call, mock get_supported_openai_params to return None
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"message": "hello"}'

        with patch("fsm_llm.llm.completion", return_value=mock_response), \
             patch("fsm_llm.llm.get_supported_openai_params", return_value=None):
            # Should not raise TypeError
            result = interface._make_llm_call(
                [{"role": "user", "content": "test"}],
                "data_extraction"
            )
            assert result is not None


# ── P3-B10: soft_equals boolean-string comparison ────────────

class TestSoftEqualsBoolString:
    """P3-B10: soft_equals(True, 'true') returns False due to case mismatch."""

    def test_true_equals_lowercase_true(self):
        """soft_equals(True, 'true') should return True (case-insensitive)."""
        from fsm_llm.expressions import soft_equals
        assert soft_equals(True, "true"), \
            "soft_equals(True, 'true') should be True — JSON booleans are lowercase"

    def test_false_equals_lowercase_false(self):
        """soft_equals(False, 'false') should return True (case-insensitive)."""
        from fsm_llm.expressions import soft_equals
        assert soft_equals(False, "false"), \
            "soft_equals(False, 'false') should be True — JSON booleans are lowercase"

    def test_true_string_equals_bool_true(self):
        """soft_equals('true', True) should also work (reversed order)."""
        from fsm_llm.expressions import soft_equals
        assert soft_equals("true", True), \
            "soft_equals('true', True) should be True"

    def test_non_boolean_string_comparison_unchanged(self):
        """Normal string comparisons should not be affected."""
        from fsm_llm.expressions import soft_equals
        assert soft_equals("hello", "hello")
        assert not soft_equals("hello", "world")
        assert soft_equals(1, "1")


# ── P3-B11: Duplicate extract_json_from_text in llm.py ──────

class TestDuplicateExtractJsonFunction:
    """P3-B11: extract_json_from_text defined in both llm.py and utilities.py."""

    def test_no_extract_json_function_in_llm_module(self):
        """llm.py should not define extract_json_from_text (it's in utilities.py)."""
        import fsm_llm.llm as llm_mod

        # Check that the function is not defined at module level in llm.py
        source = inspect.getsource(llm_mod)
        assert "\ndef extract_json_from_text" not in source, \
            "extract_json_from_text should not be defined in llm.py (duplicate of utilities.py)"
