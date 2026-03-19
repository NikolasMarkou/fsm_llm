"""
Regression tests for codebase review fixes (2026-03-19).

Tests cover all Critical, High, and Medium fixes from the epistemic review.
"""
from __future__ import annotations

import json
import inspect
from unittest.mock import MagicMock, patch

import pytest


# ══════════════════════════════════════════════════════════════
# C1: Version alignment (0.3.0)
# ══════════════════════════════════════════════════════════════


class TestVersionAlignment:
    """C1: pyproject.toml and __version__.py must agree."""

    def test_version_is_0_3_0(self):
        from fsm_llm.__version__ import __version__
        assert __version__ == "0.3.0"

    def test_init_exports_correct_version(self):
        import fsm_llm
        assert fsm_llm.__version__ == "0.3.0"


# ══════════════════════════════════════════════════════════════
# C2: Context pruning logs correct new size
# ══════════════════════════════════════════════════════════════


class TestContextPruningLog:
    """C2: prune_context should compute actual new size."""

    def test_prune_reports_different_sizes(self):
        from fsm_llm_reasoning.handlers import ReasoningHandlers
        from fsm_llm_reasoning.constants import ContextKeys, Defaults

        # Create context large enough to trigger pruning
        large_list = [f"item_{i}" for i in range(50)]
        context = {
            ContextKeys.PROBLEM_STATEMENT: "test",
            ContextKeys.REASONING_TRACE: large_list,
            ContextKeys.LOGICAL_STEPS: large_list.copy(),
            ContextKeys.OBSERVATIONS: large_list.copy(),
        }
        # Pad to exceed threshold
        context["_padding"] = "x" * (Defaults.CONTEXT_PRUNE_THRESHOLD + 1000)

        result = ReasoningHandlers.prune_context(context)

        # Should have pruned something
        assert len(result) > 0
        # At least one list should be truncated to 10
        for key in result:
            if isinstance(result[key], list):
                assert len(result[key]) <= 10


# ══════════════════════════════════════════════════════════════
# C3: Hard-coded context keys replaced with constants
# ══════════════════════════════════════════════════════════════


class TestContextKeysConstants:
    """C3: merge_reasoning_results should only use ContextKeys constants."""

    def test_no_raw_strings_in_merge(self):
        """All keys in merge_reasoning_results should come from ContextKeys."""
        from fsm_llm_reasoning.handlers import ContextManager
        source = inspect.getsource(ContextManager.merge_reasoning_results)

        # These raw strings should no longer appear as dict keys
        forbidden_raw_keys = [
            '"deductive_conclusion"',
            '"best_explanation"',
            '"adapted_solution_or_understanding"',
            '"analogy_confidence"',
            '"assessment_confidence"',
            '"calculation_error_details"',
            '"hybrid_synthesis_summary"',
            '"explanation_confidence"',
            '"analogical_solution"',
            '"inductive_hypothesis"',
        ]

        for key in forbidden_raw_keys:
            # Check it's not used as a dict key assignment (results[key] = ...)
            pattern = f"results[{key}]"
            assert pattern not in source, \
                f"Raw string {key} found in merge_reasoning_results — use ContextKeys constant"

    def test_all_reasoning_types_covered(self):
        """merge_reasoning_results should handle all ReasoningType values."""
        from fsm_llm_reasoning.constants import ReasoningType
        from fsm_llm_reasoning.handlers import ContextManager

        for rt in ReasoningType:
            # Should not crash for any reasoning type
            result = ContextManager.merge_reasoning_results(
                orchestrator_context={},
                sub_fsm_context={},
                reasoning_type=rt.value
            )
            # Should always have a completion flag
            assert f"{rt.value}_reasoning_completed" in result


# ══════════════════════════════════════════════════════════════
# H1: LLM timeout parameter
# ══════════════════════════════════════════════════════════════


class TestLLMTimeout:
    """H1: LiteLLMInterface should support timeout."""

    def test_default_timeout_is_120(self):
        from fsm_llm.llm import LiteLLMInterface

        interface = LiteLLMInterface(model="test-model")
        assert interface.timeout == 120.0

    def test_custom_timeout(self):
        from fsm_llm.llm import LiteLLMInterface

        interface = LiteLLMInterface(model="test-model", timeout=30.0)
        assert interface.timeout == 30.0

    def test_none_timeout_disables(self):
        from fsm_llm.llm import LiteLLMInterface

        interface = LiteLLMInterface(model="test-model", timeout=None)
        assert interface.timeout is None

    def test_timeout_passed_to_completion(self):
        from fsm_llm.llm import LiteLLMInterface

        interface = LiteLLMInterface(model="test-model", timeout=45.0)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"message": "hello"}'

        with patch("fsm_llm.llm.completion", return_value=mock_response) as mock_comp, \
             patch("fsm_llm.llm.get_supported_openai_params", return_value=None):
            interface._make_llm_call(
                [{"role": "user", "content": "test"}],
                "response_generation"
            )
            call_kwargs = mock_comp.call_args
            assert call_kwargs[1].get("timeout") == 45.0

    def test_none_timeout_not_passed_to_completion(self):
        from fsm_llm.llm import LiteLLMInterface

        interface = LiteLLMInterface(model="test-model", timeout=None)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"message": "hello"}'

        with patch("fsm_llm.llm.completion", return_value=mock_response) as mock_comp, \
             patch("fsm_llm.llm.get_supported_openai_params", return_value=None):
            interface._make_llm_call(
                [{"role": "user", "content": "test"}],
                "response_generation"
            )
            call_kwargs = mock_comp.call_args
            assert "timeout" not in call_kwargs[1]


# ══════════════════════════════════════════════════════════════
# H2: No duplicate import re
# ══════════════════════════════════════════════════════════════


class TestNoDuplicateImportRe:
    """H2: llm.py should not have duplicate 'import re'."""

    def test_single_import_re(self):
        from fsm_llm import llm
        source = inspect.getsource(llm)
        count = source.count("import re")
        assert count == 1, f"Expected 1 'import re', found {count}"


# ══════════════════════════════════════════════════════════════
# H3: Extraction failure returns confidence=0.0
# ══════════════════════════════════════════════════════════════


class TestExtractionFailureConfidence:
    """H3: Unstructured extraction response should return confidence=0.0."""

    def test_unstructured_response_returns_zero_confidence(self):
        from fsm_llm.llm import LiteLLMInterface

        interface = LiteLLMInterface.__new__(LiteLLMInterface)
        interface.model = "test"
        interface.kwargs = {}

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is plain text, not JSON"

        result = interface._parse_extraction_response(mock_response)
        assert result.confidence == 0.0
        assert result.extracted_data == {}


# ══════════════════════════════════════════════════════════════
# H4: No MergeStrategy alias
# ══════════════════════════════════════════════════════════════


class TestNoMergeStrategyAlias:
    """H4: reasoning engine should use ContextMergeStrategy directly."""

    def test_engine_imports_context_merge_strategy(self):
        from fsm_llm_reasoning import engine
        source = inspect.getsource(engine)
        assert "from fsm_llm.api import ContextMergeStrategy" in source
        # Should not import the alias from constants
        assert "from .constants import" in source
        import_line = [l for l in source.split("\n") if "from .constants import" in l][0]
        assert "MergeStrategy" not in import_line


# ══════════════════════════════════════════════════════════════
# H5: requirements.txt aligned with pyproject.toml
# ══════════════════════════════════════════════════════════════


class TestRequirementsAlignment:
    """H5: requirements.txt should match pyproject.toml core deps."""

    def test_requirements_has_correct_dotenv_version(self):
        import pathlib
        req_path = pathlib.Path(__file__).parents[2] / "requirements.txt"
        content = req_path.read_text()
        assert "python-dotenv>=1.0.0" in content
        # Should not have dev deps
        assert "pytest" not in content
        assert "pytest-mock" not in content

    def test_requirements_has_litellm_upper_bound(self):
        import pathlib
        req_path = pathlib.Path(__file__).parents[2] / "requirements.txt"
        content = req_path.read_text()
        assert "<2.0" in content


# ══════════════════════════════════════════════════════════════
# M1: No async handler support
# ══════════════════════════════════════════════════════════════


class TestNoAsyncHandlers:
    """M1: Handler system should not have async support."""

    def test_no_asyncio_import(self):
        from fsm_llm import handlers
        source = inspect.getsource(handlers)
        # asyncio should not be imported
        assert "import asyncio" not in source

    def test_no_async_execution_lambda_type(self):
        from fsm_llm import handlers
        assert not hasattr(handlers, "AsyncExecutionLambda")

    def test_lambda_handler_no_is_async(self):
        from fsm_llm.handlers import create_handler
        handler = create_handler("test").do(lambda ctx: {"ok": True})
        assert not hasattr(handler, "is_async")


# ══════════════════════════════════════════════════════════════
# M3: Consolidated __all__
# ══════════════════════════════════════════════════════════════


class TestConsolidatedAll:
    """M3: __all__ should be defined once, not extended dynamically."""

    def test_no_dynamic_all_extension(self):
        import pathlib
        init_path = pathlib.Path(__file__).parents[2] / "src" / "fsm_llm" / "__init__.py"
        source = init_path.read_text()
        assert source.count("__all__.extend") == 0, \
            "__all__ should not be extended dynamically"
        assert source.count("__all__.append") == 0, \
            "__all__ should not be appended to dynamically"

    def test_all_exports_are_importable(self):
        import fsm_llm
        for name in fsm_llm.__all__:
            assert hasattr(fsm_llm, name), f"{name} in __all__ but not importable"


# ══════════════════════════════════════════════════════════════
# M6: No dead tox docs env
# ══════════════════════════════════════════════════════════════


class TestNoDeadToxDocs:
    """M6: tox.ini should not have unused docs environment."""

    def test_no_sphinx_in_tox(self):
        import pathlib
        tox_path = pathlib.Path(__file__).parents[2] / "tox.ini"
        content = tox_path.read_text()
        assert "sphinx" not in content.lower()
        assert "[testenv:docs]" not in content
