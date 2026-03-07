"""Regression tests for identified bugs in llm_fsm_2."""
import hashlib
import json
import os
import tempfile
from collections import deque
from unittest.mock import patch

import pytest


# ── B1: JSON escape handling ──────────────────────────────────

class TestJsonEscapeHandling:
    """B1: escape_next flag in balanced brace extraction is dead code."""

    def test_escaped_quote_inside_json_string(self):
        """Escaped quotes within JSON string values must not break extraction."""
        from llm_fsm_2.utilities import extract_json_from_text

        text = r'Here is JSON: {"msg": "He said \"hello\"", "count": 1}'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["count"] == 1
        assert 'hello' in result["msg"]

    def test_escaped_backslash_before_quote(self):
        """A double backslash before a quote should NOT escape the quote."""
        from llm_fsm_2.utilities import extract_json_from_text

        # \\\" means: escaped backslash + closing quote
        text = r'{"path": "C:\\Users\\", "ok": true}'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["ok"] is True

    def test_nested_json_with_escapes(self):
        """Nested braces with escaped characters."""
        from llm_fsm_2.utilities import extract_json_from_text

        text = 'Some text {"outer": {"inner": "val\\\"ue"}, "x": 1} trailing'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["x"] == 1


# ── B2: Import-time side effects ─────────────────────────────

class TestLoggingSideEffects:
    """B2: importing llm_fsm_2.logging creates logs/ directory."""

    def test_import_does_not_create_logs_dir(self):
        """Importing llm_fsm_2 should not create a logs/ directory in CWD."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                # The logs/ dir should not exist before or after import
                assert not os.path.exists(os.path.join(tmpdir, "logs")), \
                    "logs/ directory should not exist before import"

                # Force re-import of the logging module
                import importlib
                import llm_fsm_2.logging as log_mod
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
        from llm_fsm_2.handlers import FSMHandler

        # Get the execute method from the protocol
        execute_method = FSMHandler.execute
        assert not inspect.iscoroutinefunction(execute_method), \
            "FSMHandler.execute should be synchronous (not async def)"

    def test_lambda_handler_execute_is_sync(self):
        """LambdaHandler.execute should be synchronous to match the protocol."""
        import inspect
        from llm_fsm_2.handlers import LambdaHandler, HandlerTiming

        handler = LambdaHandler(
            name="test",
            condition_lambdas=[],
            execution_lambda=lambda ctx: {"test": True},
            is_async=False,
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
        from llm_fsm_2.api import FSMStackFrame

        # Check that there's no inner Config class (deprecated in Pydantic v2)
        assert not hasattr(FSMStackFrame, 'Config') or \
            not hasattr(FSMStackFrame.Config, 'arbitrary_types_allowed'), \
            "FSMStackFrame should use model_config instead of class Config"


# ── B6: MD5 usage ────────────────────────────────────────────

class TestMd5Usage:
    """B6: md5 used for content hashing — should use sha256."""

    def test_process_fsm_definition_no_md5(self):
        """process_fsm_definition should not use md5 for hashing."""
        import llm_fsm_2.api as api_mod
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
        from llm_fsm_2.definitions import FSMDefinition

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
        from llm_fsm_2.api import API

        source = inspect.getsource(API.__init__)
        assert "hasattr(self.fsm_manager, 'handler_system')" not in source, \
            "Unnecessary hasattr guard should be removed"


# ── CR3: Python version check ────────────────────────────────

# ── B-NEW-1: Missing commas in prompts.py ────────────────────

class TestMissingCommasInPrompts:
    """B-NEW-1: Missing commas cause implicit string concatenation in list literals."""

    def test_response_format_list_elements_are_separate(self):
        """Each instruction in the response format list should be a separate element."""
        from llm_fsm_2.prompts import DataExtractionPromptBuilder

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
        from llm_fsm_2.prompts import DataExtractionPromptBuilder

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

    def test_single_low_confidence_transition_is_blocked(self):
        """A single transition below minimum_confidence should be BLOCKED, not AMBIGUOUS."""
        from llm_fsm_2.transition_evaluator import TransitionEvaluator, TransitionEvaluatorConfig
        from llm_fsm_2.definitions import (
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
            'confidence': 0.3,  # Below minimum_confidence of 0.5
            'evaluation_notes': [],
            'failed_conditions': [],
        }]

        result = evaluator._determine_evaluation_result(
            transition_scores, state, {}
        )

        # Should be BLOCKED because the only option doesn't meet confidence threshold
        assert result.result_type == TransitionEvaluationResult.BLOCKED, \
            f"Expected BLOCKED for low-confidence single transition, got {result.result_type}"


# ── B-NEW-3: INFO-level logging of evaluation data ───────────

class TestEvaluationLoggingLevel:
    """B-NEW-3: Evaluation result logged at INFO instead of DEBUG."""

    def test_evaluation_result_not_logged_at_info(self):
        """_evaluate_single_transition should not log evaluation_result at INFO level."""
        import inspect
        from llm_fsm_2.transition_evaluator import TransitionEvaluator

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
            "src", "llm_fsm_2", "api.py"
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


# ── B-NEW-5: AsyncExecutionLambda type alias ─────────────────

class TestAsyncExecutionLambdaType:
    """B-NEW-5: AsyncExecutionLambda is identical to sync ExecutionLambda."""

    def test_async_lambda_type_differs_from_sync(self):
        """AsyncExecutionLambda should have a different type than ExecutionLambda."""
        from llm_fsm_2.handlers import ExecutionLambda, AsyncExecutionLambda

        # The types should NOT be identical — async version should involve Awaitable
        assert ExecutionLambda != AsyncExecutionLambda, \
            "AsyncExecutionLambda should differ from ExecutionLambda (needs Awaitable return)"


# ── B-NEW-6: Duplicate error modes ───────────────────────────

class TestDuplicateErrorModes:
    """B-NEW-6: 'continue' and 'skip' error modes are identical."""

    def test_skip_error_mode_not_in_execute_handlers(self):
        """The 'skip' error mode should be removed (it's identical to 'continue')."""
        import inspect
        from llm_fsm_2.handlers import HandlerSystem

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
            "src", "llm_fsm_2", "__init__.py"
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
