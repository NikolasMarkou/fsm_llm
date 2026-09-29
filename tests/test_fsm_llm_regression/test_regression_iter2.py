"""
Regression tests for iter-2 fixes (2026-03-20).

Findings from the second epistemic deconstruction analysis (F-001 and F-002
are tested in other files, named below):
F-001: Workflow timer/event waiting broken by _ prefix filter (covered by
       test_fsm_llm_workflows/test_workflows.py::TestStepDataInternalKeyFilter)
F-002: solve_problem() mutates caller's dict (covered by
       test_engine_integration.py::TestReasoningEngineIntegration::
       test_solve_problem_does_not_mutate_initial_context)
F-003: Forbidden context pattern misses api_key
F-004: WaitForEventStep states not validated at definition time
F-005: FSM cache claims LRU but was FIFO
F-006: less() docstring claimed wrong result (code was correct)
"""

from __future__ import annotations

from collections import OrderedDict
from unittest.mock import MagicMock

# ══════════════════════════════════════════════════════════════
# F-003: Forbidden context pattern must catch api_key
# ══════════════════════════════════════════════════════════════


class TestForbiddenContextPatterns:
    """F-003: Security patterns must match both api_key and key_api orderings."""

    def test_api_key_matches_forbidden_pattern(self):
        """The pattern must match 'api_key' (api before key)."""
        from fsm_llm.constants import COMPILED_FORBIDDEN_CONTEXT_PATTERNS

        assert any(p.match("api_key") for p in COMPILED_FORBIDDEN_CONTEXT_PATTERNS)

    def test_key_api_matches_forbidden_pattern(self):
        """The pattern must match 'key_api' (key before api)."""
        from fsm_llm.constants import COMPILED_FORBIDDEN_CONTEXT_PATTERNS

        assert any(p.match("key_api") for p in COMPILED_FORBIDDEN_CONTEXT_PATTERNS)

    def test_api_key_value_matches(self):
        """The pattern must match 'api_key_value' (common variant)."""
        from fsm_llm.constants import COMPILED_FORBIDDEN_CONTEXT_PATTERNS

        assert any(
            p.match("api_key_value") for p in COMPILED_FORBIDDEN_CONTEXT_PATTERNS
        )

    def test_my_api_key_matches(self):
        """The pattern must match 'my_api_key' (prefixed variant)."""
        from fsm_llm.constants import COMPILED_FORBIDDEN_CONTEXT_PATTERNS

        assert any(p.match("my_api_key") for p in COMPILED_FORBIDDEN_CONTEXT_PATTERNS)

    def test_non_sensitive_key_does_not_match(self):
        """Normal keys must not match forbidden patterns."""
        from fsm_llm.constants import COMPILED_FORBIDDEN_CONTEXT_PATTERNS

        assert not any(
            p.match("user_name") for p in COMPILED_FORBIDDEN_CONTEXT_PATTERNS
        )

    def test_clean_context_keys_warns_on_api_key(self):
        """clean_context_keys must warn when api_key is in context."""
        from unittest.mock import patch

        from fsm_llm.context import clean_context_keys

        with patch("fsm_llm.context.logger") as mock_logger:
            result = clean_context_keys(
                {"api_key": "sk-123", "name": "Alice"}, conversation_id="test-conv"
            )
            # api_key should be kept (only warned, not removed)
            assert "api_key" in result
            assert "name" in result
            # Warning should have been issued
            mock_logger.bind.return_value.warning.assert_called_once()


# ══════════════════════════════════════════════════════════════
# F-004: WaitForEventStep states validated at definition time
# ══════════════════════════════════════════════════════════════


class TestWaitForEventStepValidation:
    """F-004: WaitForEventStep success/timeout states must be validated."""

    def test_wait_event_step_states_are_validated(self):
        """_get_referenced_states must include WaitForEventStep states."""
        try:
            from fsm_llm.workflows.definitions import WorkflowDefinition
            from fsm_llm.workflows.models import WaitEventConfig
            from fsm_llm.workflows.steps import WaitForEventStep

            step = WaitForEventStep(
                step_id="wait_payment",
                name="Wait for Payment",
                config=WaitEventConfig(
                    event_type="payment_received",
                    success_state="payment_done",
                    timeout_seconds=300,
                    timeout_state="payment_timeout",
                ),
            )

            referenced = WorkflowDefinition._get_referenced_states(
                WorkflowDefinition(
                    workflow_id="test",
                    name="test",
                    steps={"wait_payment": step},
                ),
                step,
            )
            assert "payment_done" in referenced
            assert "payment_timeout" in referenced
        except ImportError:
            import pytest

            pytest.skip("fsm_llm.workflows not installed")


# ══════════════════════════════════════════════════════════════
# F-005: FSM cache is real LRU (OrderedDict)
# ══════════════════════════════════════════════════════════════


class TestFSMCacheLRU:
    """F-005: Cache must use LRU eviction, not FIFO."""

    def test_cache_is_ordered_dict(self):
        """FSMManager.fsm_cache must be OrderedDict for LRU behavior."""
        from fsm_llm.fsm import FSMManager

        mock_llm = MagicMock()
        mgr = FSMManager(llm_interface=mock_llm, max_fsm_cache_size=3)
        assert isinstance(mgr.fsm_cache, OrderedDict)

    def test_lru_eviction_keeps_recently_accessed(self):
        """Accessing a cached entry should protect it from eviction."""
        from fsm_llm.definitions import FSMDefinition
        from fsm_llm.fsm import FSMManager

        def make_def(name):
            return FSMDefinition(
                name=name,
                description=f"{name} FSM",
                initial_state="start",
                states={
                    "start": {
                        "id": "start",
                        "description": "Start",
                        "purpose": "Begin",
                        "transitions": [{"target_state": "end", "description": "done"}],
                    },
                    "end": {
                        "id": "end",
                        "description": "End",
                        "purpose": "Finish",
                        "transitions": [],
                    },
                },
            )

        mock_llm = MagicMock()
        mgr = FSMManager(
            llm_interface=mock_llm,
            fsm_loader=make_def,
            max_fsm_cache_size=2,
        )

        # Load A and B (fills cache)
        mgr.get_fsm_definition("A")
        mgr.get_fsm_definition("B")

        # Access A to make it recently used
        mgr.get_fsm_definition("A")

        # Load C — should evict B (least recently used), not A
        mgr.get_fsm_definition("C")

        assert "A" in mgr.fsm_cache, "A was recently accessed and should not be evicted"
        assert "C" in mgr.fsm_cache
        assert "B" not in mgr.fsm_cache, "B should have been evicted as LRU"


# ══════════════════════════════════════════════════════════════
# F-006: less() docstring correctness (code was already correct)
# ══════════════════════════════════════════════════════════════


class TestLessDocstringCorrectness:
    """F-006: Verify less() behaves as now documented."""

    def test_less_with_numeric_strings(self):
        """less('10', '2') should return False (numeric coercion)."""
        from fsm_llm.expressions import less

        assert less("10", "2") is False

    def test_less_basic(self):
        """less(1, 2) should return True."""
        from fsm_llm.expressions import less

        assert less(1, 2) is True

    def test_less_chain(self):
        """less(1, 2, 3) should return True."""
        from fsm_llm.expressions import less

        assert less(1, 2, 3) is True
