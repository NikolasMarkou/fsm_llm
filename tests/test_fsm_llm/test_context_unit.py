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


# --------------------------------------------------------------
# F-12 — forbidden-pattern regexes (SC-9)
# --------------------------------------------------------------

# The negative set below is ADVERSARIAL by construction: every negative is
# maximally similar to a positive it must NOT be confused with
# (secretary/secretariat vs secrets, access_tokenizer vs access_tokens,
# private_keystone vs private_keys, passwordless_login vs password).
# A negative set of obviously-safe keys ("username", "email") would validate
# whatever the implementation happens to do. See constants.py D-009 anchor.

# Keys that MUST be stripped under strip_forbidden_keys=True.
FORBIDDEN_POSITIVES = [
    # plurals -- all four were KEPT before this fix (the under-match half)
    "secrets",
    "access_tokens",
    "private_keys",
    "oauth_tokens",
    "refresh_tokens",
    "client_secrets_file",
    # canonical singulars -- must not regress
    "password",
    "user_password",
    "password123",
    "auth_token",
    "api_key",
    "api_secret",
    "credentials",
    "private_key",
    "oauth_token",
]

# Keys that MUST survive. Each is a near-miss of a positive above.
FORBIDDEN_NEGATIVES = [
    # near-misses for the plural additions: a naive `secrets?.*` /
    # `tokens?.*` / `keys?.*` fix strips these.
    "secretary",
    "secretariat",
    "secretly",
    "secretsauce",
    "tokenizer",
    "access_tokenizer",
    "keystone",
    "private_keystone",
    # collateral damage of the old vacuous `password(?:.*|$)` -- these were
    # STRIPPED before this fix (the over-match half)
    "passwordless_login",
    "forgot_password_supported",
    "password_reset_flow_enabled",
    # plain safe keys
    "username",
    "email",
    "preference",
    "monkey_business",
]


class TestForbiddenPatternMatching:
    """F-12 / SC-9: the forbidden-key patterns over- and under-matched."""

    def test_forbidden_positives_are_stripped(self):
        """Every sensitive key, singular and plural, is removed."""
        for key in FORBIDDEN_POSITIVES:
            result = clean_context_keys(
                {key: "x", "name": "Alice"}, "test-conv", strip_forbidden_keys=True
            )
            assert key not in result, f"{key!r} should have been stripped"
            assert result["name"] == "Alice"

    def test_forbidden_negatives_are_kept(self):
        """Near-miss keys are legitimate user data and must survive."""
        for key in FORBIDDEN_NEGATIVES:
            result = clean_context_keys(
                {key: "x"}, "test-conv", strip_forbidden_keys=True
            )
            assert key in result, f"{key!r} should have been kept"

    def test_plural_secret_stripped_but_secretary_kept(self):
        """The exact discrimination the plural addition must achieve."""
        result = clean_context_keys(
            {"secrets": "s", "secretary": "Bob"},
            "test-conv",
            strip_forbidden_keys=True,
        )
        assert result == {"secretary": "Bob"}

    def test_password_flag_kept_but_password_stripped(self):
        """The exact discrimination the vacuous-group removal must achieve."""
        result = clean_context_keys(
            {"password": "hunter2", "passwordless_login": True},
            "test-conv",
            strip_forbidden_keys=True,
        )
        assert result == {"passwordless_login": True}

    def test_patterns_are_linear_on_adversarial_input(self):
        """No catastrophic backtracking: a nearly-matching long key is fast."""
        import time

        from fsm_llm.constants import COMPILED_FORBIDDEN_CONTEXT_PATTERNS

        for probe in ("password_" * 5000, "secret_" * 5000, "_" * 40000):
            start = time.perf_counter()
            any(p.match(probe) for p in COMPILED_FORBIDDEN_CONTEXT_PATTERNS)
            assert time.perf_counter() - start < 0.5, f"slow on {probe[:16]!r}..."


# --------------------------------------------------------------
# F-13 — case-insensitive internal-prefix matching (SC-10)
# --------------------------------------------------------------

# Mixed/upper-case spellings of every INTERNAL_KEY_PREFIXES entry.
MIXED_CASE_INTERNAL_KEYS = ["SYSTEM_foo", "Internal_x", "__Private", "_Hidden"]


class TestInternalPrefixCaseInsensitivity:
    """F-13 / SC-10: verified separately at each of the FIVE call sites."""

    def test_site_1_clean_context_keys(self):
        """context.py -- clean_context_keys."""
        data = dict.fromkeys(MIXED_CASE_INTERNAL_KEYS, "v")
        data["name"] = "Alice"
        result = clean_context_keys(data, "test-conv")
        assert result == {"name": "Alice"}

    def test_site_2_fsm_get_conversation_data(self):
        """fsm.py -- FSMManager.get_conversation_data."""
        from unittest.mock import Mock

        from fsm_llm.definitions import FSMContext, FSMInstance
        from fsm_llm.fsm import FSMManager
        from fsm_llm.llm import LLMInterface

        manager = FSMManager(llm_interface=Mock(spec=LLMInterface))
        context = FSMContext()
        context.data.update(dict.fromkeys(MIXED_CASE_INTERNAL_KEYS, "v"))
        context.data["name"] = "Alice"
        manager.instances["conv-1"] = FSMInstance(
            fsm_id="f", current_state="start", context=context
        )

        assert manager.get_conversation_data("conv-1") == {"name": "Alice"}

    def test_site_3_fsm_context_update_warns(self):
        """definitions.py:964 -- FSMContext.update warns on internal prefixes."""
        from fsm_llm.definitions import FSMContext
        from fsm_llm.logging import logger

        records: list[str] = []
        sink_id = logger.add(
            lambda m: records.append(m.record["message"]), level="WARNING"
        )
        logger.enable("fsm_llm")
        try:
            FSMContext().update({"SYSTEM_foo": "v"})
        finally:
            logger.remove(sink_id)

        assert any("SYSTEM_foo" in r for r in records), (
            "FSMContext.update should warn about a mixed-case internal key"
        )

    def test_site_4_get_user_visible_data(self):
        """definitions.py:991 -- FSMContext.get_user_visible_data."""
        from fsm_llm.definitions import FSMContext

        context = FSMContext()
        context.data.update(dict.fromkeys(MIXED_CASE_INTERNAL_KEYS, "v"))
        context.data["name"] = "Alice"

        assert context.get_user_visible_data() == {"name": "Alice"}

    def test_site_5_prompt_security_filter(self):
        """prompts.py:324 -- the FIFTH site, not named in the finding.

        This one is prompt-affecting: an unfiltered internal key leaks
        straight into the LLM prompt.
        """
        from fsm_llm.prompts import DataExtractionPromptBuilder

        builder = DataExtractionPromptBuilder()
        data = dict.fromkeys(MIXED_CASE_INTERNAL_KEYS, "v")
        data["name"] = "Alice"

        assert builder._filter_context_for_security(data) == {"name": "Alice"}

    def test_lowercase_prefixes_still_stripped(self):
        """Over-correction guard: the original lowercase behavior is intact."""
        data = {"_a": 1, "system_b": 2, "internal_c": 3, "__d": 4, "name": "Alice"}
        assert clean_context_keys(data, "test-conv") == {"name": "Alice"}

    def test_non_prefixed_keys_with_prefix_substring_survive(self):
        """Over-correction guard: the prefix must anchor at the start."""
        data = {"my_system_flag": 1, "the_internal_note": 2, "NAME": "Alice"}
        assert clean_context_keys(data, "test-conv") == data


class TestNestedContextCleaning:
    """F-11 / SC-11: the key filter must apply at EVERY depth, not just the top."""

    def test_nested_forbidden_key_is_stripped(self):
        """SC-11 headline case: a buried password must not reach the prompt."""
        data = {"user": {"password": "hunter2", "name": "bob"}}
        result = clean_context_keys(data, "test-conv", strip_forbidden_keys=True)
        assert result == {"user": {"name": "bob"}}

    def test_nested_forbidden_key_survives_when_not_stripping(self):
        """Warn-only mode must not start deleting nested keys."""
        data = {"user": {"password": "hunter2", "name": "bob"}}
        result = clean_context_keys(data, "test-conv", strip_forbidden_keys=False)
        assert result == data

    def test_deeply_nested_forbidden_key_is_stripped(self):
        """One level was the finding; the filter must hold all the way down."""
        data = {"a": {"b": {"c": {"api_key": "sk-1", "keep": "yes"}}}}
        result = clean_context_keys(data, "test-conv", strip_forbidden_keys=True)
        assert result == {"a": {"b": {"c": {"keep": "yes"}}}}

    def test_nested_internal_prefix_and_none_are_stripped(self):
        """The other two removal reasons must recurse too, case-insensitively."""
        data = {"outer": {"_hidden": 1, "SYSTEM_x": 2, "gone": None, "kept": 3}}
        result = clean_context_keys(data, "test-conv")
        assert result == {"outer": {"kept": 3}}

    def test_dicts_inside_lists_are_filtered(self):
        """In scope by decision: a list is not a bypass for the same leak."""
        data = {"users": [{"password": "x", "name": "bob"}, {"name": "eve"}]}
        result = clean_context_keys(data, "test-conv", strip_forbidden_keys=True)
        assert result == {"users": [{"name": "bob"}, {"name": "eve"}]}

    def test_dicts_inside_tuples_are_filtered_and_stay_tuples(self):
        data = {"users": ({"secrets": "x", "name": "bob"},)}
        result = clean_context_keys(data, "test-conv", strip_forbidden_keys=True)
        assert result == {"users": ({"name": "bob"},)}
        assert isinstance(result["users"], tuple)

    # -- [SOFT] falsy-survives contract, at depth --

    def test_falsy_values_survive_at_every_depth(self):
        """Empty list/str/0/False are semantically meaningful at ANY level."""
        leaf = {"lst": [], "s": "", "zero": 0, "flag": False, "d": {}}
        data = {"top": dict(leaf), "nest": {"mid": {"deep": dict(leaf)}}}
        data.update(leaf)
        result = clean_context_keys(data, "test-conv", strip_forbidden_keys=True)
        assert result == data

    def test_nested_dict_emptied_by_filtering_is_kept_as_empty_dict(self):
        """A dict that filters down to {} stays {}; the parent key is not dropped."""
        data = {"user": {"password": "x"}}
        result = clean_context_keys(data, "test-conv", strip_forbidden_keys=True)
        assert result == {"user": {}}

    def test_scalars_and_non_dict_values_pass_through_untouched(self):
        obj = object()
        data = {"o": obj, "n": 1.5, "b": b"raw", "l": [1, "", 0, False]}
        result = clean_context_keys(data, "test-conv", strip_forbidden_keys=True)
        assert result == data
        assert result["o"] is obj

    # -- recursion safety --

    def test_self_referential_dict_does_not_recurse_forever(self):
        """A cycle must terminate at the depth bound, not raise RecursionError."""
        data: dict = {"name": "bob"}
        data["self"] = data
        result = clean_context_keys(data, "test-conv", strip_forbidden_keys=True)
        assert result["name"] == "bob"

    def test_pathologically_deep_payload_does_not_blow_the_stack(self):
        """10,000-deep user-controlled data must not crash prompt construction."""
        data: dict = {"password": "leaf-secret"}
        for _ in range(10_000):
            data = {"n": data}
        result = clean_context_keys(data, "test-conv", strip_forbidden_keys=True)
        assert isinstance(result, dict)

    def test_container_past_the_depth_bound_is_dropped_not_passed_through(self):
        """Fail-CLOSED at the limit: burying a secret must not bypass the filter."""
        from fsm_llm.context import MAX_CONTEXT_FILTER_DEPTH

        data: dict = {"password": "leaf-secret"}
        for _ in range(MAX_CONTEXT_FILTER_DEPTH + 2):
            data = {"n": data}
        result = clean_context_keys(data, "test-conv", strip_forbidden_keys=True)
        assert "leaf-secret" not in repr(result)

    def test_payload_just_inside_the_depth_bound_is_still_filtered(self):
        """Vacuity guard for the test above: the bound is not so tight that
        everything is dropped."""
        from fsm_llm.context import MAX_CONTEXT_FILTER_DEPTH

        data: dict = {"password": "leaf-secret", "keep": "yes"}
        for _ in range(MAX_CONTEXT_FILTER_DEPTH - 2):
            data = {"n": data}
        result = clean_context_keys(data, "test-conv", strip_forbidden_keys=True)
        assert "leaf-secret" not in repr(result)
        assert "yes" in repr(result)

    def test_non_string_keys_do_not_crash_the_filter(self):
        """Non-str keys carry no prefix/pattern and must not raise on startswith."""
        data = {"outer": {1: "a", 2.5: "b", "password": "x"}}
        result = clean_context_keys(data, "test-conv", strip_forbidden_keys=True)
        assert result == {"outer": {1: "a", 2.5: "b"}}
