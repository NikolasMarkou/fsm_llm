"""Unit tests for fsm_llm.context module."""

from __future__ import annotations

from fsm_llm.context import ContextCompactor, clean_context_keys


class TestCleanContextKeys:
    """Tests for clean_context_keys function."""

    def test_removes_none_values(self):
        """Keys with None values should be removed by default."""
        data = {"name": "Alice", "age": None, "city": "NYC"}
        result = clean_context_keys(data, "test-conv")
        assert result == {"name": "Alice", "city": "NYC"}

    def test_preserves_none_values_when_disabled(self):
        """None values should be preserved when remove_none_values=False."""
        data = {"name": "Alice", "age": None}
        result = clean_context_keys(data, "test-conv", remove_none_values=False)
        assert result == {"name": "Alice", "age": None}

    def test_removes_internal_prefix_underscore(self):
        """Keys starting with _ should be removed."""
        data = {"name": "Alice", "_internal": "hidden"}
        result = clean_context_keys(data, "test-conv")
        assert result == {"name": "Alice"}

    def test_removes_internal_prefix_system(self):
        """Keys starting with system_ should be removed."""
        data = {"name": "Alice", "system_state": "active"}
        result = clean_context_keys(data, "test-conv")
        assert result == {"name": "Alice"}

    def test_removes_internal_prefix_internal(self):
        """Keys starting with internal_ should be removed."""
        data = {"name": "Alice", "internal_id": "123"}
        result = clean_context_keys(data, "test-conv")
        assert result == {"name": "Alice"}

    def test_removes_internal_prefix_dunder(self):
        """Keys starting with __ should be removed."""
        data = {"name": "Alice", "__class": "user"}
        result = clean_context_keys(data, "test-conv")
        assert result == {"name": "Alice"}

    def test_preserves_empty_lists(self):
        """Empty lists are semantically meaningful and should be preserved."""
        data = {"allergies": [], "name": "Alice"}
        result = clean_context_keys(data, "test-conv")
        assert result == {"allergies": [], "name": "Alice"}

    def test_preserves_empty_strings(self):
        """Empty strings are preserved."""
        data = {"notes": "", "name": "Alice"}
        result = clean_context_keys(data, "test-conv")
        assert result == {"notes": "", "name": "Alice"}

    def test_preserves_zero_values(self):
        """Zero and False should be preserved."""
        data = {"count": 0, "active": False, "name": "Alice"}
        result = clean_context_keys(data, "test-conv")
        assert result == {"count": 0, "active": False, "name": "Alice"}

    def test_empty_dict(self):
        """Empty dict input returns empty dict."""
        result = clean_context_keys({}, "test-conv")
        assert result == {}

    def test_all_internal_keys(self):
        """Dict with only internal keys returns empty dict."""
        data = {"_a": 1, "system_b": 2, "__c": 3}
        result = clean_context_keys(data, "test-conv")
        assert result == {}

    def test_warns_on_forbidden_patterns(self):
        """Keys matching forbidden security patterns should trigger a warning."""
        data = {"password": "secret123", "name": "Alice"}
        # Should not raise, just warn
        result = clean_context_keys(data, "test-conv")
        # Forbidden keys are warned but NOT removed
        assert "password" in result
        assert "name" in result

    def test_warns_on_token_pattern(self):
        """Token-related keys should trigger a warning."""
        data = {"auth_token": "abc123"}
        result = clean_context_keys(data, "test-conv")
        assert "auth_token" in result

    def test_mixed_removal_and_preservation(self):
        """Combined scenario with internal, None, and valid keys."""
        data = {
            "name": "Alice",
            "age": None,
            "_internal": "hidden",
            "system_flag": True,
            "hobbies": [],
            "score": 0,
        }
        result = clean_context_keys(data, "test-conv")
        assert result == {"name": "Alice", "hobbies": [], "score": 0}


class TestContextCompactorCompact:
    """Tests for ContextCompactor.compact() — transient key cleanup."""

    def test_clears_transient_keys(self):
        compactor = ContextCompactor(transient_keys={"action", "action_result"})
        context = {"action": "add_state", "action_result": "ok", "name": "bot"}
        result = compactor.compact(context)
        assert result == {"action": None, "action_result": None}

    def test_ignores_missing_transient_keys(self):
        compactor = ContextCompactor(transient_keys={"action", "action_result"})
        context = {"name": "bot"}
        result = compactor.compact(context)
        assert result == {}

    def test_empty_transient_keys(self):
        compactor = ContextCompactor()
        context = {"action": "something"}
        result = compactor.compact(context)
        assert result == {}

    def test_partial_match(self):
        compactor = ContextCompactor(transient_keys={"a", "b", "c"})
        context = {"a": 1, "c": 3, "d": 4}
        result = compactor.compact(context)
        assert result == {"a": None, "c": None}


class TestContextCompactorPrune:
    """Tests for ContextCompactor.prune() — state-entry pruning."""

    def test_prunes_keys_on_state_entry(self):
        compactor = ContextCompactor(
            prune_on_entry={"review": {"done_flag", "old_data"}}
        )
        context = {
            "_current_state": "review",
            "done_flag": True,
            "old_data": "x",
            "keep": 1,
        }
        result = compactor.prune(context)
        assert result == {"done_flag": None, "old_data": None}

    def test_no_pruning_for_unmapped_state(self):
        compactor = ContextCompactor(prune_on_entry={"review": {"done_flag"}})
        context = {"_current_state": "design", "done_flag": True}
        result = compactor.prune(context)
        assert result == {}

    def test_no_current_state_in_context(self):
        compactor = ContextCompactor(prune_on_entry={"review": {"done_flag"}})
        context = {"done_flag": True}
        result = compactor.prune(context)
        assert result == {}

    def test_ignores_missing_keys(self):
        compactor = ContextCompactor(prune_on_entry={"review": {"a", "b", "c"}})
        context = {"_current_state": "review", "a": 1}
        result = compactor.prune(context)
        assert result == {"a": None}

    def test_multiple_states_configured(self):
        compactor = ContextCompactor(
            prune_on_entry={
                "design": {"phase1_data"},
                "review": {"phase2_data"},
            }
        )
        ctx_design = {
            "_current_state": "design",
            "phase1_data": "x",
            "phase2_data": "y",
        }
        assert compactor.prune(ctx_design) == {"phase1_data": None}

        ctx_review = {
            "_current_state": "review",
            "phase1_data": "x",
            "phase2_data": "y",
        }
        assert compactor.prune(ctx_review) == {"phase2_data": None}


# ==============================================================
# Conversation search tests
# ==============================================================


class TestConversationSearch:
    """Tests for Conversation.search()."""

    def _make_conversation(self, exchanges=None):
        from fsm_llm.definitions import Conversation

        conv = Conversation(max_history_size=100)
        for ex in exchanges or []:
            for role, msg in ex.items():
                if role == "user":
                    conv.add_user_message(msg)
                else:
                    conv.add_system_message(msg)
        return conv

    def test_search_finds_user_message(self):
        conv = self._make_conversation(
            [{"user": "My name is Alice"}, {"system": "Hello Alice!"}]
        )
        results = conv.search("Alice")
        assert len(results) == 2  # Both contain "Alice"

    def test_search_case_insensitive(self):
        conv = self._make_conversation([{"user": "Hello World"}])
        results = conv.search("hello")
        assert len(results) == 1

    def test_search_empty_query(self):
        conv = self._make_conversation([{"user": "Hello"}])
        assert conv.search("") == []

    def test_search_no_matches(self):
        conv = self._make_conversation([{"user": "Hello"}])
        assert conv.search("xyz") == []

    def test_search_respects_limit(self):
        conv = self._make_conversation([{"user": f"Message {i}"} for i in range(10)])
        results = conv.search("Message", limit=3)
        assert len(results) == 3

    def test_search_most_recent_first(self):
        conv = self._make_conversation(
            [{"user": "First message"}, {"user": "Second message"}]
        )
        results = conv.search("message")
        assert "Second" in results[0]["user"]

    def test_search_empty_conversation(self):
        from fsm_llm.definitions import Conversation

        conv = Conversation()
        assert conv.search("anything") == []


# ==============================================================
# Conversation summary tests
# ==============================================================


class TestConversationSummary:
    """Tests for Conversation.summary and _maintain_history_size."""

    def test_summary_none_by_default(self):
        from fsm_llm.definitions import Conversation

        conv = Conversation()
        assert conv.summary is None

    def test_summary_populated_on_trim(self):
        from fsm_llm.definitions import Conversation

        conv = Conversation(max_history_size=2)
        for i in range(5):
            conv.add_user_message(f"User message {i}")
            conv.add_system_message(f"System response {i}")
        # Should have trimmed older messages
        assert conv.summary is not None
        assert len(conv.summary) > 0

    def test_summary_contains_trimmed_content(self):
        from fsm_llm.definitions import Conversation

        conv = Conversation(max_history_size=1)
        conv.add_user_message("My name is Alice and I like pizza")
        conv.add_system_message("Nice to meet you Alice!")
        conv.add_user_message("What is the weather?")
        conv.add_system_message("It is sunny.")
        # First exchange should be in summary
        assert conv.summary is not None
        assert "Alice" in conv.summary

    def test_summary_capped_at_2000_chars(self):
        from fsm_llm.definitions import Conversation

        conv = Conversation(max_history_size=1)
        # Generate enough messages to exceed 2000 chars in summary
        for i in range(50):
            conv.add_user_message(f"{'x' * 100} message {i}")
            conv.add_system_message(f"{'y' * 100} response {i}")
        assert conv.summary is not None
        assert len(conv.summary) <= 2000

    def test_get_summary_and_recent(self):
        from fsm_llm.definitions import Conversation

        conv = Conversation(max_history_size=1)
        conv.add_user_message("Old message")
        conv.add_system_message("Old response")
        conv.add_user_message("New message")
        conv.add_system_message("New response")
        summary, recent = conv.get_summary_and_recent()
        assert summary is not None
        assert "Old" in summary
        assert len(recent) == 2  # 1 exchange = 2 messages

    def test_no_summary_when_not_trimmed(self):
        from fsm_llm.definitions import Conversation

        conv = Conversation(max_history_size=10)
        conv.add_user_message("Hello")
        conv.add_system_message("Hi!")
        assert conv.summary is None


# ==============================================================
# ContextCompactor.summarize tests
# ==============================================================


class TestContextCompactorSummarize:
    """Tests for ContextCompactor.summarize()."""

    def test_summarize_without_llm(self):
        from fsm_llm.definitions import Conversation

        compactor = ContextCompactor()
        conv = Conversation(max_history_size=100)
        conv.add_user_message("Hello, my name is Alice")
        conv.add_system_message("Nice to meet you, Alice!")

        result = compactor.summarize(conv)
        assert result is not None
        assert "Alice" in result
        assert conv.summary == result

    def test_summarize_empty_conversation(self):
        from fsm_llm.definitions import Conversation

        compactor = ContextCompactor()
        conv = Conversation()
        assert compactor.summarize(conv) is None

    def test_summarize_with_none_conversation(self):
        compactor = ContextCompactor()
        assert compactor.summarize(None) is None

    def test_summarize_caps_at_2000_chars(self):
        from fsm_llm.definitions import Conversation

        compactor = ContextCompactor()
        conv = Conversation(max_history_size=1000)
        for i in range(100):
            conv.add_user_message(f"{'x' * 50} message {i}")
            conv.add_system_message(f"{'y' * 50} response {i}")

        result = compactor.summarize(conv)
        assert result is not None
        assert len(result) <= 2000

    def test_summarize_with_mock_llm(self):
        from unittest.mock import MagicMock

        from fsm_llm.definitions import Conversation

        compactor = ContextCompactor()
        conv = Conversation(max_history_size=100)
        conv.add_user_message("Hello Alice")
        conv.add_system_message("Hi there!")

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.message = "User greeted as Alice."
        mock_llm.generate_response.return_value = mock_response

        result = compactor.summarize(conv, llm_interface=mock_llm)
        assert result == "User greeted as Alice."
        assert conv.summary == "User greeted as Alice."

    def test_summarize_llm_failure_falls_back(self):
        from unittest.mock import MagicMock

        from fsm_llm.definitions import Conversation

        compactor = ContextCompactor()
        conv = Conversation(max_history_size=100)
        conv.add_user_message("Hello Alice")
        conv.add_system_message("Hi there!")

        mock_llm = MagicMock()
        mock_llm.generate_response.side_effect = Exception("LLM down")

        result = compactor.summarize(conv, llm_interface=mock_llm)
        assert result is not None
        assert "Alice" in result  # Fallback text extraction
