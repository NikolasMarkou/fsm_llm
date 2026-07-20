"""Unit tests for fsm_llm.context module."""

from __future__ import annotations

from fsm_llm.constants import has_internal_prefix, is_forbidden_context_entry
from fsm_llm.context import ContextCompactor, clean_context_keys
from tests.test_fsm_llm.fixtures.context_key_corpus import (
    CARVE_OUT_CREDENTIAL_ENTRIES,
    CARVE_OUT_KNOWN_FAIL_OPEN,
    CARVE_OUT_KNOWN_OVER_STRIPPED,
    CARVE_OUT_SAFE_ENTRIES,
    CRYPTO_KEY_KNOWN_OVER_STRIPPED,
    CRYPTO_KEY_SAFE_KEYS,
    CRYPTO_KEY_SAFE_VALUES,
    CRYPTO_KEY_SECRET_KEYS,
    CRYPTO_KEY_SECRET_VALUES,
    KNOWN_OVER_STRIPPED,
    SAFE_KEYS,
    SECRET_KEYS,
    TOKEN_KNOWN_OVER_STRIPPED,
    TOKEN_SAFE_KEYS,
    TOKEN_SAFE_VALUES,
    TOKEN_SECRET_KEYS,
    TOKEN_SECRET_SHORT_VALUE_ENTRIES,
    TOKEN_SECRET_VALUES,
)
from tests.test_fsm_llm.fixtures.holdout_key_corpus import (
    HOLDOUT,
    HOLDOUT_KNOWN_FAIL_OPEN,
    HOLDOUT_KNOWN_OVER_STRIPPED,
    arm_of,
)


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
    # `password_<secret-suffix>` -- the ~200-key class that the step-8
    # terminal-anchored pattern silently KEPT. This list had zero of these,
    # which is why it could only ever detect over-match. See D-016.
    "password_hash",
    "password_plaintext",
    "password_salt",
    "db_password_plaintext",
    "admin_password_encrypted",
    "user_password_salt",
    # a suffix in NO denylist -- pins that unrecognized suffixes fail CLOSED
    "password_pepper",
    "password_hash2",
    # kept by the policy allowlist via `reset`, stripped by the token pattern
    "password_reset_token",
    # The six keys measured reaching the LLM PROMPT under D-016's prefix-shaped
    # lookahead. D-016 found one member of this class (`password_reset_token`)
    # and closed it by naming it in the auth-token pattern; these are the rest.
    # See D-026 -- the allowlist now matches the WHOLE remaining suffix.
    "password_reset_code",
    "password_reset_otp",
    "password_reset_hash",
    "password_last_plaintext",
    "password_retrieval",
    "password_policy_key",
    # the excluded-token half of D-016, previously unpinned
    "password_confirmation",
    "password_hint",
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
    # further policy/status suffixes covered by the D-016 allowlist
    "passwordless",
    "password_policy",
    "password_strength_meter",
    "password_expiry_days",
    "password_min_length",
    "password_attempts",
    "reset_tokenizer",
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

        probes = (
            "password_" * 5000,
            "secret_" * 5000,
            "_" * 40000,
            # NON-matching inputs force the `(?:^|.*[\\W_])` prefix to backtrack
            # over every position -- the matching probes above return at
            # position 0 and so never exercise the expensive path.
            "a_" * 20000 + "passwordless",
            "x_" * 20000 + "access_tokenizer",
        )
        for probe in probes:
            start = time.perf_counter()
            any(p.match(probe) for p in COMPILED_FORBIDDEN_CONTEXT_PATTERNS)
            assert time.perf_counter() - start < 0.5, f"slow on {probe[:16]!r}..."


# --------------------------------------------------------------
# F-12 (remediation) — the password class as a SPACE, not a list
# --------------------------------------------------------------

# The step-8 pattern was checked against a hand-written list of key names, so a
# whole SHAPE of key (`password_<non-numeric-suffix>`) flipped STRIPPED->KEPT
# without any test noticing -- ~200 keys, including `password_hash` and
# `db_password_plaintext`, which were measured reaching the LLM prompt.
# This corpus enumerates the space as `prefix x stem x suffix` and classifies
# EVERY cell, so the next pattern edit cannot silently move a class again.
# See constants.py / decisions.md D-016.
_PW_PREFIXES = ["", "user_", "db_", "admin_", "forgot_", "old_", "new_", "root_"]

# Suffixes whose values are (or are derived from) the credential itself.
# Every `prefix + "password" + suffix` built from these MUST be stripped.
_PW_SECRET_SUFFIXES = [
    "",
    "s",
    "123",
    "_1",
    "_hash",
    "_hashes",
    "_digest",
    "_salt",
    "_plaintext",
    "_encrypted",
    "_enc",
    "_bcrypt",
    "_md5",
    "_sha256",
    "_blob",
    "_b64",
    "_ciphertext",
    "_value",
    "_secret",
    # deliberately absent from any denylist: these pin the FAIL-CLOSED default
    "_pepper",
    "_raw",
    "_clear",
    "_scrypt",
    "_argon2",
    "_pbkdf2",
    "_hash2",
    "_wibble",
    # --- COMPOUND `_<policy-token>_<secret>` suffixes (D-026) --------------
    # The list above is entirely SINGLE-token, which is why it could not see the
    # infix bypass: D-016's lookahead only inspected the token immediately after
    # `password`, so ANY allowlisted token inserted in front of a secret one
    # defeated the whole control (`password_plaintext` stripped but
    # `password_last_plaintext` was KEPT). These pin the SHAPE of the suffix, not
    # just its vocabulary — a fix that re-anchors only the tokens it was written
    # from will fail here.
    "_reset_hash",
    "_reset_code",
    "_reset_otp",
    "_reset_secret",
    "_last_plaintext",
    "_manager_dump",
    "_policy_key",
    "_retrieval",
    "_expiry_hash",
    "_login_plaintext",
    "_flow_token",
    "_status_raw",
    "_attempt_hash",
    "_setup_secret",
    "_help_plaintext",
    "_strategy_pepper",
    # depth 3: two policy tokens then a secret one
    "_reset_flow_hash",
    "_last_manager_salt",
    # --- tokens D-016 says it DELIBERATELY excluded from the allowlist -----
    # These are the load-bearing half of that decision and were unpinned: the
    # reviewer re-added them to the allowlist and 5 credential keys flipped
    # STRIP -> KEEP with ZERO test failures. `password_confirmation` is the
    # re-typed password itself.
    "_confirmation",
    "_confirm",
    "_hint",
    "_field",
    "_input",
    "_form",
]

# Suffixes naming a policy/status PROPERTY. These MUST be kept (SC-9).
_PW_POLICY_SUFFIXES = [
    "less_login",
    "less",
    "_reset_flow_enabled",
    "_reset_email_sent",
    "_policy",
    "_strength_meter",
    "_expiry_days",
    "_supported",
    "_min_length",
    "_max_length",
    "_attempts",
    "_last_changed",
    "_complexity_rules",
    "_validation_error",
]


def _password_corpus():
    """Yield (key, must_be_stripped) over the whole prefix x suffix space."""
    for prefix in _PW_PREFIXES:
        for suffix in _PW_SECRET_SUFFIXES:
            yield prefix + "password" + suffix, True
        for suffix in _PW_POLICY_SUFFIXES:
            yield prefix + "password" + suffix, False


class TestPasswordKeySpace:
    """F-12 remediation: assert the full differential over the key SPACE."""

    def test_corpus_covers_both_classes_and_is_not_trivial(self):
        """Vacuity guard: a corpus that is all-one-class proves nothing."""
        rows = list(_password_corpus())
        stripped = [k for k, must in rows if must]
        kept = [k for k, must in rows if not must]
        assert len(rows) == len(_PW_PREFIXES) * (
            len(_PW_SECRET_SUFFIXES) + len(_PW_POLICY_SUFFIXES)
        )
        assert len(stripped) >= 200, "secret class too small to be a space"
        assert len(kept) >= 100, "policy class too small to discriminate"

    def test_every_secret_shaped_password_key_is_stripped(self):
        """Fail-CLOSED: an unrecognized suffix must strip, not keep."""
        leaked = [
            key
            for key, must_strip in _password_corpus()
            if must_strip
            and key
            in clean_context_keys({key: "v"}, "test-conv", strip_forbidden_keys=True)
        ]
        assert leaked == [], f"{len(leaked)} secret-shaped keys KEPT: {leaked[:12]}"

    def test_every_policy_shaped_password_key_is_kept(self):
        """SC-9: the flag/policy class is legitimate user data."""
        destroyed = [
            key
            for key, must_strip in _password_corpus()
            if not must_strip
            and key
            not in clean_context_keys(
                {key: "v"}, "test-conv", strip_forbidden_keys=True
            )
        ]
        assert destroyed == [], (
            f"{len(destroyed)} policy keys STRIPPED: {destroyed[:12]}"
        )

    def test_differential_against_the_two_superseded_patterns(self):
        """Three-way differential: b00fade (vacuous) vs step-8 vs shipped.

        Pins the two failure MODES this seam has already shipped, so a future
        edit that re-creates either one fails here rather than in production.
        """
        import re

        from fsm_llm.constants import COMPILED_FORBIDDEN_CONTEXT_PATTERNS

        # b00fade: trailing `(?:.*|$)` is vacuous -> over-matches the flags.
        old = re.compile(r"(?:^|.*[\W_])password(?:.*|$)", re.IGNORECASE)
        # step 8: terminal-anchored -> under-matches every `password_<suffix>`.
        step8 = re.compile(r"(?:^|.*[\W_])password(?:s)?(?:[-_.]?\d+)?$", re.IGNORECASE)

        def shipped(key):
            return any(p.match(key) for p in COMPILED_FORBIDDEN_CONTEXT_PATTERNS)

        rows = list(_password_corpus())
        old_over = [k for k, must in rows if not must and old.match(k)]
        step8_under = [k for k, must in rows if must and not step8.match(k)]
        shipped_wrong = [k for k, must in rows if must is not bool(shipped(k))]

        # The two superseded patterns each fail on a large class ...
        assert len(old_over) >= 100, "old pattern's over-match not reproduced"
        assert len(step8_under) >= 150, "step-8 under-match not reproduced"
        # ... and the shipped pattern fails on neither.
        assert shipped_wrong == [], f"shipped pattern wrong on: {shipped_wrong[:12]}"

    def test_secret_password_keys_do_not_reach_the_llm_prompt(self):
        """The criterion is the PROMPT, not the regex.

        `_filter_context_for_security` always drops (unlike `clean_context_keys`,
        whose `strip_forbidden_keys` defaults to False), so it is the surface
        where the step-8 under-match actually leaked.
        """
        from fsm_llm.prompts import BasePromptBuilder

        builder = BasePromptBuilder()
        context = {
            "password_hash": "$2b$12$REAL",
            "db_password_plaintext": "hunter2",
            "user_password_salt": "NaCl",
            "admin_password_encrypted": "AAAA",
            "password_policy": "12 chars min",
        }
        filtered = builder._filter_context_for_security(context)
        assert list(filtered) == ["password_policy"]


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

    def test_falsy_non_string_keys_survive(self):
        """The previous test used only TRUTHY non-str keys (1, 2.5), so its
        failing branch was never taken. `if not key` used to run BEFORE the
        `isinstance(key, str)` guard, so these were destroyed as "empty key".
        See D-017."""
        # One key per case: `0`, `False` and `0.0` all hash equal, so putting
        # them in one dict would collapse them and test only the first.
        for falsy_key in (0, False, 0.0, ()):
            data = {"outer": {falsy_key: "kept"}}
            result = clean_context_keys(data, "test-conv", strip_forbidden_keys=True)
            assert result["outer"] == {falsy_key: "kept"}, (
                f"falsy non-str key {falsy_key!r} was destroyed"
            )

    def test_empty_string_key_is_still_removed(self):
        """Vacuity guard for the test above: reordering the guards must not
        stop `""` being dropped."""
        result = clean_context_keys(
            {"": "orphan", "name": "Alice"}, "test-conv", strip_forbidden_keys=True
        )
        assert result == {"name": "Alice"}

    def test_falsy_non_string_keys_agree_with_the_prompt_filter(self):
        """The two filters disagreed on exactly this input: `clean_context_keys`
        dropped `0`, `_filter_context_for_security` kept it."""
        from fsm_llm.prompts import BasePromptBuilder

        data = {0: "zero-key-val", "name": "Alice"}
        cleaned = clean_context_keys(dict(data), "test-conv", strip_forbidden_keys=True)
        prompted = BasePromptBuilder()._filter_context_for_security(dict(data))
        assert (0 in cleaned) == (0 in prompted) is True

    def test_non_string_key_skip_is_logged_not_silent(self):
        """A `bytes` key bypasses every name check, where it once raised
        TypeError. Fail-OPEN inside a fail-CLOSED control must at least be
        loud. See D-017 / concern 7."""
        from fsm_llm.logging import logger

        records = []
        sink_id = logger.add(lambda m: records.append(m.record), level="WARNING")
        try:
            result = clean_context_keys(
                {b"password": "BYTES-SECRET"}, "test-conv", strip_forbidden_keys=True
            )
        finally:
            logger.remove(sink_id)

        # Behaviour is unchanged (still kept) -- but it is now announced.
        assert b"password" in result
        assert any("skipped the security name checks" in r["message"] for r in records)


class TestIndependentVocabularyKeyCorpus:
    """D-030. Every previous corpus in this plan was ENUMERATED FROM the token
    list it was testing, so it could not contain a key the list omits, and for
    three consecutive rounds the residual defect hid in exactly that blind spot
    (round 1 an under-match, round 2 an over-match of the same class).

    This corpus is hand-authored in
    `tests/test_fsm_llm/fixtures/context_key_corpus.py` from real-world
    application vocabulary and never imports `fsm_llm.constants`. On its first
    run it caught two credential keys -- `signing_key` and `encryption_key` --
    that every generated corpus had missed, because `private[-_.]?key` covered
    exactly one qualifier (D-031).
    """

    @staticmethod
    def _kept(key):
        return key in clean_context_keys(
            {key: "v"}, "test-conv", strip_forbidden_keys=True
        )

    def test_the_corpus_is_independent_of_the_module_under_test(self):
        """The construction rule IS the control here, so pin it mechanically:
        a future edit that "helpfully" generates the corpus from the constants
        re-creates the exact blind spot this file exists to remove."""
        import pathlib

        from tests.test_fsm_llm.fixtures import context_key_corpus

        source = pathlib.Path(context_key_corpus.__file__).read_text()
        offending = [
            line
            for line in source.splitlines()
            if "import" in line and "fsm_llm" in line
        ]
        assert offending == [], (
            "the independent corpus imports the module it is testing, which "
            f"destroys its independence: {offending}"
        )

    def test_the_corpus_is_not_trivial(self):
        """Vacuity guard: an empty or single-class corpus proves nothing, and
        every 'safe' key must be genuinely reachable rather than all pinned as
        known-over-stripped."""
        assert len(SECRET_KEYS) >= 60, "secret class too small to be a class"
        assert len(SAFE_KEYS) >= 60, "safe class too small to discriminate"
        assert not (set(SECRET_KEYS) & set(SAFE_KEYS)), "a key claims both classes"
        assert KNOWN_OVER_STRIPPED <= set(SAFE_KEYS), (
            "KNOWN_OVER_STRIPPED names a key that is not in SAFE_KEYS"
        )
        assert len(KNOWN_OVER_STRIPPED) < len(SAFE_KEYS) // 2, (
            "most of the safe class is pinned as over-stripped, so the "
            "'is kept' assertion below has almost nothing left to check"
        )

    def test_no_credential_key_reaches_the_prompt(self):
        """The security assertion. Fail-CLOSED is not a defence here: every one
        of these names a credential VALUE outright."""
        leaked = [k for k in SECRET_KEYS if self._kept(k)]
        assert leaked == [], (
            f"{len(leaked)} credential keys reached the prompt: {leaked}"
        )

    def test_legitimate_metadata_keys_still_reach_the_prompt(self):
        """The usability assertion. Over-stripping is not dangerous but it
        silently degrades the model's context, so it is pinned, not ignored."""
        destroyed = [
            k for k in SAFE_KEYS if k not in KNOWN_OVER_STRIPPED and not self._kept(k)
        ]
        assert destroyed == [], (
            f"{len(destroyed)} legitimate keys were stripped: {destroyed}"
        )

    def test_the_known_over_strip_set_is_pinned_exactly(self):
        """Pinned in BOTH directions on purpose. A key leaving the set is an
        improvement that must be recorded rather than absorbed silently; a key
        joining it is a new usability regression of the class that went
        undisclosed for a whole round. Either way the cost stays visible."""
        actual = frozenset(k for k in SAFE_KEYS if not self._kept(k))
        assert actual == KNOWN_OVER_STRIPPED, (
            "the over-strict class drifted.\n"
            f"  newly stripped (regression): {sorted(actual - KNOWN_OVER_STRIPPED)}\n"
            f"  newly kept (fixed, update the pin): "
            f"{sorted(KNOWN_OVER_STRIPPED - actual)}"
        )


def _shipped_vocabulary_union() -> set[str]:
    """THE anti-tautology denominator: the UNION of every word list that ships.

    ONE definition, read back out of `fsm_llm.constants` at call time, shared by
    both anti-tautology tests. Do NOT inline a second copy: two hand-maintained
    denominators drift, and denominator SELECTION is itself the thing under
    audit at this seam (`plans/plan-2026-07-20T040150-876e7164/findings/
    review-iter-1-pass3.md`, "Adjudication of the 44%", defect 1 -- the prior
    plan reported a per-arm SPLIT figure while its own criterion said "absent
    from EVERY shipped alternation", and the looser of the two available
    denominators was the one reported).

    UNION, not SPLIT, and the difference is real. Measured on the shipped
    corpora at this commit:

        half         SPLIT    UNION
        key/SAFE     87.7%    85.8%
        key/CRED     18.1%    16.7%
        token/SAFE   56.2%    56.2%
        token/CRED   48.3%    41.7%

    The token credential half is the one that moves most (-6.6pp) and it is the
    half nothing used to gate at all.

    Returns: a set of lowercased vocabulary words, drawn from all SEVEN shipped
    lists. Never raises; never mutates `constants`.
    """
    from fsm_llm import constants as c

    words: set[str] = set()
    for blob in (
        c._CRYPTO_KEY_QUALIFIERS,
        c._KEY_MATERIAL_HEADS,
        c._SAFE_TOKEN_QUALIFIERS,
        # D-021 added a vocabulary to the token arm; SC-21 says "absent from
        # EVERY shipped alternation", so it belongs here. Omitting it would
        # inflate the figure by counting `csrf`, `jwt` and `session` as words
        # the author never enumerated.
        c._BEARER_TOKEN_QUALIFIERS,
        c._TOKEN_MATERIAL_HEADS,
    ):
        words |= {w for w in blob.split("|") if w}
    # D-003's identifier nouns gate the UUID/ULID carve-out. A frozenset rather
    # than an alternation blob, but still a list of words this codebase's author
    # enumerated. Omitting it inflates the key arm from 0.858 to 0.925.
    words |= set(c._IDENTIFIER_NOUN_VOCABULARY)
    # D-005's HTTP scheme words are the SEVENTH list and the newest. They are
    # value-side rather than name-side, which is the argument for leaving them
    # out -- and they are included anyway, because the stricter reading of the
    # denominator wins on this seam and this one is free. MEASURED: adding them
    # changes no half's figure by a single word (all 22 that matter, `bearer`,
    # `jwt`, `hmac`, `oauth`, `digest`, `basic`, `signature`, were already in
    # the union via the name lists). Recorded so a future reader does not
    # re-derive it: it is a no-op TODAY, not by construction.
    words |= set(c._AUTH_SCHEME_WORDS)
    return words


def _qualifier_words(names) -> set[str]:
    """The vocabulary a corpus of context-key names actually uses, minus the
    trigger words themselves (which every name in both arms carries by
    construction and which therefore cannot discriminate)."""
    import re as _re

    out: set[str] = set()
    for name in names:
        for word in _re.split(r"[^a-zA-Z0-9]+", name.lower()):
            if word and word not in ("key", "keys", "token", "tokens"):
                out.add(word)
    return out


class TestCryptoKeyAndTokenTriggers:
    """D-015 (correcting D-014). The two triggers now have DIFFERENT polarity,
    because their vocabularies have opposite structure:

      `key`   -- an ENUMERATED CRYPTO DENYLIST. Not a class control; it fails
                 OPEN on crypto vocabulary nobody listed. D-014 shipped a
                 fail-closed allowlist here and it destroyed 42% of an
                 independently-sourced safe corpus (31 names) on the live
                 prompt path.
      `token` -- a fail-CLOSED, START-ANCHORED allowlist for the `<qual>_token`
                 suffix shape, plus a value-head denylist for the `token_<head>`
                 prefix shape.

    Four controls, deliberately inseparable:

    (a) TWO-DIRECTION measurement. The secret half alone is satisfied by
        "strip everything", which is a strictly worse outcome than the leak it
        replaces (LESSONS [I:5]); the safe half alone is satisfied by the
        denylist that was already there. Both halves are asserted, and the
        residual over-strip is pinned two-sided so it cannot drift in either
        direction unannounced.
    (b) ANTI-TAUTOLOGY. D-014 passed (a) with 0/100 over-strip while shipping a
        live bypass, because 36/36 and 21/21 of its safe corpus's qualifiers
        were themselves members of its own allowlist. A corpus grown from the
        vocabulary under test cannot see what that vocabulary omits. The guard
        below measures that OVERLAP mechanically, which is the check the
        "no verbatim constant" guard structurally could not perform.
    (c) ANCHORING. D-014's lookaheads inspected only the qualifier immediately
        before the trigger, so `ssh_cache_key`, `csrf_max_token` and
        `signing_public_key` were all KEPT. The prefix-bypass shape is probed
        by name here.
    (d) ANTI-VACUITY and ReDoS. G-14; and these patterns run on the prompt path
        while the per-conversation lock is held, over context keys that are
        arbitrary consumer input.

    All of it operates on the EXACT SHIPPED PATTERNS read back out of
    `fsm_llm.constants`, never on a copy restated here.
    """

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _kept(key, value="v"):
        """Does this ENTRY survive the filter?

        D-021: the token arm decides on (name, value), not on the name alone,
        so the value is a real parameter now. It defaults to the inert ``"v"``
        -- short, single-character-class, no possible credential -- so a caller
        passing only a name is asking a well-defined question: *does the NAME
        alone strip this?* Every token-arm caller below passes a real value.
        """
        return key in clean_context_keys(
            {key: value}, "test-conv", strip_forbidden_keys=True
        )

    @staticmethod
    def _universe():
        return (
            tuple(CRYPTO_KEY_SECRET_KEYS)
            + tuple(CRYPTO_KEY_SAFE_KEYS)
            + tuple(TOKEN_SECRET_KEYS)
            + tuple(TOKEN_SAFE_KEYS)
            + tuple(SECRET_KEYS)
            + tuple(SAFE_KEYS)
        )

    @staticmethod
    def _vocabularies():
        """The four shipped vocabulary alternations, by name, each paired with
        the direction its removal must move the result set.

        `deny` groups make keys STRIP, so neutralising one must make keys
        newly KEPT. `allow` groups make keys KEEP, so neutralising one must
        make keys newly STRIPPED. Asserting the direction is what stops a
        future edit from inverting a polarity unnoticed -- the exact defect
        D-015 exists to correct.
        """
        from fsm_llm import constants as c

        return {
            "key/crypto-denylist": (c._CRYPTO_KEY_QUALIFIERS, "deny"),
            "key/material-head-denylist": (c._KEY_MATERIAL_HEADS, "deny"),
            # D-021: the token qualifier ALLOWLIST no longer lives in
            # `FORBIDDEN_CONTEXT_PATTERNS` -- it moved to
            # `_TOKEN_VALUE_SCAN_NAME_RE`, where a name outside it is referred
            # to the value layer instead of being stripped. Its polarity is
            # unchanged (it still makes names KEEP), so the surgery still
            # applies; only its address changed. `_shipped_sources` below is
            # what keeps this honest -- neutralising it must still move the
            # result set, or the allowlist has become decoration.
            "token/qualifier-allowlist": (c._SAFE_TOKEN_QUALIFIERS, "allow"),
            "token/bearer-denylist": (c._BEARER_TOKEN_QUALIFIERS, "deny"),
            "token/material-head-denylist": (c._TOKEN_MATERIAL_HEADS, "deny"),
        }

    @staticmethod
    def _shipped_sources():
        """Every regex source the filter's NAME layer decides on, in order.

        D-021 split this across two homes, so a guard that reads only
        `FORBIDDEN_CONTEXT_PATTERNS` silently stops covering the token
        allowlist. The referral regex is appended last and flagged, because
        neutralising it has the OPPOSITE effect of neutralising a shipped
        pattern: it removes a referral, not a strip.
        """
        from fsm_llm.constants import (
            _TOKEN_VALUE_SCAN_NAME_RE,
            FORBIDDEN_CONTEXT_PATTERNS,
        )

        return [*FORBIDDEN_CONTEXT_PATTERNS, _TOKEN_VALUE_SCAN_NAME_RE.pattern]

    @classmethod
    def _locate(cls, fragment):
        """Index of the single shipped regex source containing `fragment`."""
        sources = cls._shipped_sources()
        hits = [i for i, p in enumerate(sources) if fragment in p]
        assert len(hits) == 1, (
            "the anti-vacuity surgery could not find its target exactly once in "
            f"the SHIPPED patterns (found {len(hits)}). The guard is now testing "
            "nothing -- re-derive it from constants.py."
        )
        return hits[0]

    @classmethod
    def _result_set(cls, sources):
        """Names STRIPPED by this set of sources, under the real two-layer rule.

        The last source is the token referral: a name it matches is not
        stripped outright, it is handed to the value layer. The corpus values
        used here are the ones the corpus itself supplies, so this reflects the
        shipped decision rather than a name-only approximation of it.
        """
        import re

        from fsm_llm.constants import _token_value_is_credential

        *strip_sources, referral_source = sources
        strip = [re.compile(p, re.IGNORECASE) for p in strip_sources]
        referral = re.compile(referral_source, re.IGNORECASE)

        def stripped(name):
            if any(c.match(name) for c in strip):
                return True
            if referral.match(name):
                return _token_value_is_credential(cls._value_for(name))
            return False

        return frozenset(k for k in cls._universe() if stripped(k))

    @staticmethod
    def _value_for(name):
        """The corpus's own value for `name`, or the inert placeholder.

        Step 1 of plan-2026-07-20-b8a6b855 added the two `key`-arm maps. Until
        then this helper knew only the token arm, so every `key`-arm name in
        `_universe()` was surveyed at the inert `"v"` -- which is a
        well-defined question about the NAME and no question at all about the
        control, since `stripe_key` and `order_key` are the same string to the
        name layer and only the value separates them (LESSONS [I:5]).
        """
        if name in TOKEN_SECRET_VALUES:
            return TOKEN_SECRET_VALUES[name]
        if name in TOKEN_SAFE_VALUES:
            return TOKEN_SAFE_VALUES[name]
        if name in CRYPTO_KEY_SECRET_VALUES:
            return CRYPTO_KEY_SECRET_VALUES[name]
        if name in CRYPTO_KEY_SAFE_VALUES:
            return CRYPTO_KEY_SAFE_VALUES[name]
        return "v"

    @classmethod
    def _mutated(cls, index, old, new):
        sources = cls._shipped_sources()
        original = sources[index]
        sources[index] = original.replace(old, new)
        assert sources[index] != original
        return sources

    # -- corpus integrity ------------------------------------------------
    def test_the_new_corpora_are_not_trivial(self):
        """Vacuity guard on the corpus itself: the leak sets must be at least
        as large as the measured leaks they close (46 crypto, 27 token), the
        safe sets must be big enough to discriminate, and no name may claim
        both classes."""
        assert len(CRYPTO_KEY_SECRET_KEYS) >= 46
        assert len(TOKEN_SECRET_KEYS) >= 27
        # Raised for D-015: the safe halves are the measurement that D-014's
        # tautology defeated, so gutting them must fail loudly.
        assert len(CRYPTO_KEY_SAFE_KEYS) >= 90
        assert len(TOKEN_SAFE_KEYS) >= 55
        assert not (set(CRYPTO_KEY_SECRET_KEYS) & set(CRYPTO_KEY_SAFE_KEYS))
        assert not (set(TOKEN_SECRET_KEYS) & set(TOKEN_SAFE_KEYS))
        assert CRYPTO_KEY_KNOWN_OVER_STRIPPED <= set(CRYPTO_KEY_SAFE_KEYS)
        assert TOKEN_KNOWN_OVER_STRIPPED <= set(TOKEN_SAFE_KEYS)

    def test_the_corpus_does_not_paste_the_pattern_it_tests(self):
        """Strengthens the import-based independence guard, whose weakness was
        called out explicitly (`findings/crypto-key-class-control.md` Risk 5):
        pasting a literal copy of a `constants.py` qualifier string satisfies
        the import check while defeating its entire purpose. A corpus built
        from the vocabulary under test is structurally blind to what that
        vocabulary omits, which is the failure LESSONS [I:5] records firing
        three consecutive times on this exact seam."""
        import pathlib

        from fsm_llm import constants as c
        from tests.test_fsm_llm.fixtures import context_key_corpus

        source = pathlib.Path(context_key_corpus.__file__).read_text()
        pasted = [
            name
            for name in (
                "_CRYPTO_KEY_QUALIFIERS",
                "_KEY_MATERIAL_HEADS",
                "_SAFE_TOKEN_QUALIFIERS",
                "_TOKEN_MATERIAL_HEADS",
                "_PASSWORD_POLICY_SUFFIXES",
            )
            if getattr(c, name) in source
        ]
        assert pasted == [], (
            f"the corpus contains a verbatim copy of {pasted} from constants.py, "
            "so it can only confirm the pattern against its own vocabulary"
        )

    # -- (a) two-direction measurement -----------------------------------
    def test_no_crypto_key_material_reaches_the_prompt(self):
        """SC-4, direction 1. Every name here is private key material or the
        container it is read from.

        Measured with each name's REAL credential value since step 1 of
        plan-2026-07-20-b8a6b855. Before that this test passed `"v"` and so
        exercised layer 1 alone, on an arm whose whole design rests on layer 2
        deciding what the name cannot (defect 5, A-3).
        """
        missing = [
            k for k in CRYPTO_KEY_SECRET_KEYS if k not in CRYPTO_KEY_SECRET_VALUES
        ]
        assert missing == [], (
            f"{len(missing)} crypto-key names have no corpus value: {missing}. "
            "A name added to CRYPTO_KEY_SECRET_KEYS without a value would be "
            "measured against a placeholder, which is not a measurement."
        )
        leaked = [
            k
            for k in CRYPTO_KEY_SECRET_KEYS
            if self._kept(k, CRYPTO_KEY_SECRET_VALUES[k])
        ]
        assert leaked == [], f"{len(leaked)} crypto keys reached the prompt: {leaked}"

    def test_no_auth_token_reaches_the_prompt(self):
        """SC-4, direction 1. Possession of any of these values is sufficient
        to act as somebody.

        D-021: measured with each name's real credential VALUE, because that is
        now what the arm decides on for every qualifier outside
        `_BEARER_TOKEN_QUALIFIERS`. Measuring this list against a placeholder
        string would report 22 leaks and all 22 would be artefacts of the
        placeholder -- `user_token: "v"` genuinely is not a credential.
        """
        missing = [k for k in TOKEN_SECRET_KEYS if k not in TOKEN_SECRET_VALUES]
        assert missing == [], (
            f"{len(missing)} token names have no corpus value: {missing}. "
            "A name added to TOKEN_SECRET_KEYS without a value would be "
            "measured against a placeholder, which is not a measurement."
        )
        leaked = [k for k in TOKEN_SECRET_KEYS if self._kept(k, TOKEN_SECRET_VALUES[k])]
        assert leaked == [], f"{len(leaked)} auth tokens reached the prompt: {leaked}"

    def test_the_bearer_name_list_still_carries_short_credentials(self):
        """D-021. The value layer is structurally blind to a credential that is
        short or low-entropy, so `_BEARER_TOKEN_QUALIFIERS` -- the NAME layer --
        has to catch those. This test is what stops a future editor deleting
        that list on the reasonable-looking grounds that "the value layer
        handles tokens now": every entry here has a value the value layer
        cannot see, and every one must still strip."""
        leaked = [
            name
            for name, value in TOKEN_SECRET_SHORT_VALUE_ENTRIES
            if self._kept(name, value)
        ]
        assert leaked == [], (
            f"{len(leaked)} short-valued bearer credentials reached the prompt: "
            f"{leaked}. The name layer is no longer carrying them."
        )

    def test_ordinary_key_vocabulary_still_reaches_the_prompt(self):
        """SC-5, direction 2. Paired with the test above on purpose: a fix that
        stripped everything would pass that one and fail this one."""
        missing = [k for k in CRYPTO_KEY_SAFE_KEYS if k not in CRYPTO_KEY_SAFE_VALUES]
        assert missing == [], (
            f"{len(missing)} ordinary key names have no corpus value: {missing}"
        )
        destroyed = [
            k
            for k in CRYPTO_KEY_SAFE_KEYS
            if k not in CRYPTO_KEY_KNOWN_OVER_STRIPPED
            and not self._kept(k, CRYPTO_KEY_SAFE_VALUES[k])
        ]
        assert destroyed == [], f"{len(destroyed)} ordinary keys stripped: {destroyed}"

    def test_ordinary_token_vocabulary_still_reaches_the_prompt(self):
        """SC-5, direction 2. This is the tightest collateral constraint in the
        plan: this is an LLM framework and `*_tokens` metering vocabulary is
        pervasive, ordinary and non-secret."""
        missing = [k for k in TOKEN_SAFE_KEYS if k not in TOKEN_SAFE_VALUES]
        assert missing == [], f"{len(missing)} safe token names have no value"
        destroyed = [
            k
            for k in TOKEN_SAFE_KEYS
            if k not in TOKEN_KNOWN_OVER_STRIPPED
            and not self._kept(k, TOKEN_SAFE_VALUES[k])
        ]
        assert destroyed == [], f"{len(destroyed)} ordinary keys stripped: {destroyed}"

    def test_the_named_hard_invariants_hold_individually(self):
        """The corpus assertions are set-shaped, so a future corpus edit could
        drop one of these without any test noticing. Invariant I-7 and the LLM
        metering vocabulary are named here explicitly and separately."""
        must_keep = (
            "public_key",  # I-7, also pinned in tests/test_fsm_llm_regression/
            "max_tokens",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "token_count",
            "tokenizer",
            "cache_key",
            "primary_key",
            "foreign_key",
            "idempotency_key",
        )
        must_strip = (
            "ssh_key",
            "sshkey",
            "privkey",
            "private_key",
            "key_pair",
            "keyfile",
            "key_material",
            "kek",
            "dek",
            "id_rsa",
            "id_ed25519",
            "license_key",
            "csrf_token",
            "csrftoken",
            "session_token",
            "jwt_token",
            "id_token",
        )
        assert [k for k in must_keep if not self._kept(k)] == []
        assert [k for k in must_strip if self._kept(k)] == []

    def test_the_crypto_key_over_strip_set_is_pinned_exactly(self):
        """Two-sided, mirroring `test_the_known_over_strip_set_is_pinned_exactly`.
        A name joining this set is an undisclosed usability regression; a name
        leaving it is an improvement that must be recorded, not absorbed."""
        actual = frozenset(
            k
            for k in CRYPTO_KEY_SAFE_KEYS
            if not self._kept(k, CRYPTO_KEY_SAFE_VALUES[k])
        )
        assert actual == CRYPTO_KEY_KNOWN_OVER_STRIPPED, (
            "the crypto-key over-strip class drifted.\n"
            f"  newly stripped (regression): "
            f"{sorted(actual - CRYPTO_KEY_KNOWN_OVER_STRIPPED)}\n"
            f"  newly kept (update the pin): "
            f"{sorted(CRYPTO_KEY_KNOWN_OVER_STRIPPED - actual)}"
        )

    def test_the_token_over_strip_set_is_pinned_exactly(self):
        """Two-sided, same rule."""
        actual = frozenset(
            k for k in TOKEN_SAFE_KEYS if not self._kept(k, TOKEN_SAFE_VALUES[k])
        )
        assert actual == TOKEN_KNOWN_OVER_STRIPPED, (
            "the token over-strip class drifted.\n"
            f"  newly stripped (regression): "
            f"{sorted(actual - TOKEN_KNOWN_OVER_STRIPPED)}\n"
            f"  newly kept (update the pin): "
            f"{sorted(TOKEN_KNOWN_OVER_STRIPPED - actual)}"
        )

    # -- (a2) carve-out shape coverage and its two-sided verdicts ---------
    def test_every_carve_out_probe_is_decided_by_its_VALUE_not_its_name(self):
        """ANTI-VACUITY GUARD FOR THE SHAPE CORPUS ITSELF.

        A shape probe under a name layer 1 already strikes measures NOTHING
        about the shape -- the value is never consulted. A shape probe under a
        name that carries no trigger measures nothing either, for the opposite
        reason. Both mistakes are silent and both produce a green test.

        So attribution is pinned black-box, from both ends: every credential
        probe's NAME must be KEPT when paired with the inert ``"v"`` (proving
        the name alone does not decide), and every safe probe's NAME must be
        STRIPPED when paired with a real credential (proving the name alone
        does not rescue it). Whatever those probes then report is a statement
        about the VALUE, which is the only thing this layer reads.
        """
        strong = "9dR2pQ7xL4mZ8vN3bK6tY1wJ5hG0sF2aD8cE4rT7uI"
        name_decides = [
            entry_id
            for entry_id, name, _ in CARVE_OUT_CREDENTIAL_ENTRIES
            if not self._kept(name, "v")
        ]
        assert name_decides == [], (
            f"{len(name_decides)} credential shape probes sit under a name the "
            f"NAME layer already strips, so they measure nothing about the "
            f"value shape they claim to cover: {name_decides}"
        )
        name_rescues = [
            entry_id
            for entry_id, name, _ in CARVE_OUT_SAFE_ENTRIES
            if self._kept(name, strong)
        ]
        assert name_rescues == [], (
            f"{len(name_rescues)} safe shape probes sit under a name that is "
            f"kept even holding a real credential, so the filter never reads "
            f"their value: {name_rescues}"
        )

    def test_every_layer_two_carve_out_shape_has_both_a_credential_and_a_safe_instance(
        self,
    ):
        """SC-2. THE GUARD THAT WOULD HAVE PREVENTED THIS ENTIRE PLAN.

        Every one of the five defects being closed here survived multiple
        adversarial rounds for the same reason: the value layer's carve-outs
        were tuned against corpora containing ZERO instances of the shapes
        they carve out. `LESSONS [I:5]` records the question that was never
        asked -- *does my corpus contain instances of the shape THIS layer
        carves out?* -- and this test is that question, mechanised.

        The predicates below are DELIBERATELY re-derived here rather than
        imported from `fsm_llm.constants`. That duplication is the control:
        a coverage test that asks the module under test what its own shapes
        are can only ever confirm the module against itself, and would go
        green the moment a carve-out was deleted. Do NOT "centralise" these.
        """
        import math
        import re

        uuid_shape = re.compile(
            r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
            r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
        )
        ulid_shape = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")
        path_ext_shape = re.compile(r"^\S*/\S*\.[A-Za-z0-9]{1,6}$")
        percent_shape = re.compile(r"%[0-9A-Fa-f]{2}")
        pure_hex = re.compile(r"^[0-9a-fA-F]+$")
        # Independently written from vendor documentation, NOT read back out of
        # `_CREDENTIAL_VALUE_PREFIXES` -- see the docstring.
        vendor_prefixes = (
            "sk-",
            "sk_live_",
            "sk_test_",
            "ghp_",
            "gho_",
            "glpat-",
            "xoxb-",
            "xapp-",
            "akia",
            "shpat_",
            "eyj",
            "hf_",
        )

        def text(value):
            return value.strip() if isinstance(value, str) else None

        def entropy(sample):
            counts = {}
            for character in sample:
                counts[character] = counts.get(character, 0) + 1
            total = len(sample)
            return -sum((n / total) * math.log2(n / total) for n in counts.values())

        def classes(sample):
            return (
                any(c.islower() for c in sample)
                + any(c.isupper() for c in sample)
                + any(c.isdigit() for c in sample)
            )

        def is_pem(value):
            body = text(value)
            return body is not None and body.lower().startswith("-----begin")

        def colon_tail(value):
            body = text(value)
            if body is None or ":" not in body:
                return None
            return body.rsplit(":", 1)[1]

        shapes = {
            "canonical UUID": lambda v: bool(uuid_shape.match(text(v) or "")),
            "ULID": lambda v: bool(ulid_shape.match(text(v) or "")),
            "PEM armour": is_pem,
            "published vendor prefix": lambda v: (
                (text(v) or "").lower().startswith(vendor_prefixes)
            ),
            "<label>:<pure hex> composite": lambda v: (
                colon_tail(v) is not None and bool(pure_hex.match(colon_tail(v)))
            ),
            "<id>:<secret> composite": lambda v: (
                colon_tail(v) is not None and not pure_hex.match(colon_tail(v))
            ),
            "sub-24-character": lambda v: text(v) is not None and 0 < len(text(v)) < 24,
            "internal whitespace": lambda v: (
                text(v) is not None
                and not is_pem(v)
                and any(c.isspace() for c in text(v))
            ),
            "percent-encoded": lambda v: bool(percent_shape.search(text(v) or "")),
            "path with extension": lambda v: bool(path_ext_shape.match(text(v) or "")),
            'count("/") >= 3': lambda v: (text(v) or "").count("/") >= 3,
            "single character class": lambda v: (
                text(v) is not None and len(text(v)) >= 24 and classes(text(v)) < 2
            ),
            "sub-3.0-bit entropy": lambda v: (
                text(v) is not None and len(text(v)) >= 24 and entropy(text(v)) < 3.0
            ),
            "bool/int/float (token arm)": lambda v: isinstance(v, (bool, int, float)),
            "non-str non-numeric (token arm)": lambda v: (
                not isinstance(v, (str, bool, int, float))
            ),
        }

        credential_pool = (
            [value for _, _, value in CARVE_OUT_CREDENTIAL_ENTRIES]
            + list(CRYPTO_KEY_SECRET_VALUES.values())
            + list(TOKEN_SECRET_VALUES.values())
            + [value for _, value in TOKEN_SECRET_SHORT_VALUE_ENTRIES]
        )
        safe_pool = (
            [value for _, _, value in CARVE_OUT_SAFE_ENTRIES]
            + list(CRYPTO_KEY_SAFE_VALUES.values())
            + list(TOKEN_SAFE_VALUES.values())
        )

        uncovered = []
        for shape, matches in shapes.items():
            missing = [
                side
                for side, pool in (("credential", credential_pool), ("safe", safe_pool))
                if not any(matches(value) for value in pool)
            ]
            if missing:
                uncovered.append(f"{shape}: no {' and no '.join(missing)} instance")
        assert uncovered == [], (
            "the corpus cannot measure "
            f"{len(uncovered)} of the {len(shapes)} shapes the value layer "
            "carves out, so those carve-outs are untested branches with "
            "comments attached:\n  " + "\n  ".join(uncovered)
        )

    def test_the_carve_out_fail_open_set_is_pinned_exactly(self):
        """Two-sided pin on the shape probes whose CREDENTIAL side reaches the
        prompt today. Same mechanism as `TOKEN_KNOWN_OVER_STRIPPED`: this set
        is DISCLOSURE, not absolution.

        Measured at aa284b7 as 11 of 15 -- the first time this seam was ever
        probed with values of its own carve-out shapes. An id LEAVING this set
        is a fix (steps 4 and 5 own most of them) and must be recorded; an id
        JOINING it is a new leak that would otherwise go undisclosed for a
        whole round, which is exactly how this seam got to a fifth attempt.
        """
        actual = frozenset(
            entry_id
            for entry_id, name, value in CARVE_OUT_CREDENTIAL_ENTRIES
            if self._kept(name, value)
        )
        assert actual == CARVE_OUT_KNOWN_FAIL_OPEN, (
            "the carve-out fail-open class drifted.\n"
            f"  newly leaking (regression): "
            f"{sorted(actual - CARVE_OUT_KNOWN_FAIL_OPEN)}\n"
            f"  newly caught (fixed, update the pin): "
            f"{sorted(CARVE_OUT_KNOWN_FAIL_OPEN - actual)}"
        )

    def test_the_carve_out_over_strip_set_is_pinned_exactly(self):
        """Two-sided pin on the shape probes whose SAFE side is destroyed
        today. Paired with the test above on purpose: a fix that strips
        everything clears that one and fails this one, which is the single
        discipline every prior attempt on this seam lacked (`LESSONS [I:5]`)."""
        actual = frozenset(
            entry_id
            for entry_id, name, value in CARVE_OUT_SAFE_ENTRIES
            if not self._kept(name, value)
        )
        assert actual == CARVE_OUT_KNOWN_OVER_STRIPPED, (
            "the carve-out over-strip class drifted.\n"
            f"  newly stripped (regression): "
            f"{sorted(actual - CARVE_OUT_KNOWN_OVER_STRIPPED)}\n"
            f"  newly kept (fixed, update the pin): "
            f"{sorted(CARVE_OUT_KNOWN_OVER_STRIPPED - actual)}"
        )

    def test_the_carve_out_corpus_is_not_trivial(self):
        """Vacuity guard on the shape corpus. Entry ids must be unique (they
        are the pin keys), both sides must be populated, and neither pin set
        may swallow its whole side -- a corpus where every probe is disclosed
        as known-wrong asserts nothing."""
        credential_ids = [entry_id for entry_id, _, _ in CARVE_OUT_CREDENTIAL_ENTRIES]
        safe_ids = [entry_id for entry_id, _, _ in CARVE_OUT_SAFE_ENTRIES]
        assert len(set(credential_ids)) == len(credential_ids), (
            "duplicate credential entry id -- the two-sided pins key off these"
        )
        assert len(set(safe_ids)) == len(safe_ids), "duplicate safe entry id"
        assert CARVE_OUT_KNOWN_FAIL_OPEN <= set(credential_ids)
        assert CARVE_OUT_KNOWN_OVER_STRIPPED <= set(safe_ids)
        assert len(CARVE_OUT_KNOWN_OVER_STRIPPED) < len(safe_ids), (
            "every safe shape probe is pinned as over-stripped, so the "
            "usability direction has nothing left to assert"
        )

    # -- (b) anti-tautology ----------------------------------------------
    def test_the_safe_corpus_is_not_a_restatement_of_the_shipped_vocabulary(self):
        """THE GUARD THAT WOULD HAVE CAUGHT D-014.

        D-014's safe corpus reported 0/100 over-strip while the shipped pattern
        carried a live bypass, because every `*_key`/`*_token` qualifier in that
        corpus (36/36 and 21/21) was itself a member of the allowlist being
        tested. A corpus grown from the vocabulary under test is structurally
        blind to what that vocabulary omits, so its green is a restatement of
        the pattern rather than a measurement of it.

        `test_the_corpus_does_not_paste_the_pattern_it_tests` cannot see this:
        no constant is pasted verbatim, the words are merely the same words.
        This test measures the OVERLAP instead, and requires that a substantial
        majority of the safe corpus's vocabulary be words the pattern author
        never enumerated -- i.e. that the corpus can actually disagree with the
        implementation.

        DENOMINATOR: the UNION of all seven shipped lists, not a per-arm SPLIT.
        See `_shipped_vocabulary_union` for why, and for the measured gap
        between the two readings.
        """
        shipped_words = _shipped_vocabulary_union()
        qualifiers = _qualifier_words

        # D-021 raises the token floor from 0.25 to SC-21's stated 0.50. The
        # 0.25 was a concession to a corpus that could not do better: before
        # D-021 every non-allowlisted `*_token` name stripped, so any name that
        # would have raised this number was, by construction, an over-strip.
        # Referring those names to the value layer is what makes the real floor
        # reachable, and holding the guard at 0.25 afterwards would leave the
        # exact blind spot D-019 finding (2) was found in.
        measured = {}
        for label, corpus, floor in (
            ("key", CRYPTO_KEY_SAFE_KEYS, 0.50),
            ("token", TOKEN_SAFE_KEYS, 0.50),
        ):
            words = qualifiers(corpus)
            outside = {w for w in words if w not in shipped_words}
            fraction = len(outside) / len(words)
            measured[label] = (len(outside), len(words), round(fraction, 3))
            assert fraction >= floor, (
                f"{label}: only {len(outside)}/{len(words)} "
                f"({fraction:.0%}) of the safe corpus's qualifier words are "
                "absent from the shipped vocabulary. The corpus is drifting "
                "back into a restatement of the pattern it is supposed to "
                "audit -- re-source it from real tooling (cloud SDKs, DB docs, "
                "framework config) before touching the pattern. See D-015."
            )

        # And a corpus that cannot disagree is useless even if it is
        # independent, so pin that it still measures a real keep-rate.
        for label, corpus, values in (
            ("key", CRYPTO_KEY_SAFE_KEYS, {}),
            ("token", TOKEN_SAFE_KEYS, TOKEN_SAFE_VALUES),
        ):
            kept = sum(1 for k in corpus if self._kept(k, values.get(k, "v")))
            assert kept / len(corpus) >= 0.85, (
                f"{label}: independent corpus keep-rate is {kept}/{len(corpus)}. "
                "The Pre-Mortem stop trigger for this step is >15% over-strip; "
                "STOP and report rather than tuning the vocabulary against the "
                "corpus, which recreates the tautology in reverse."
            )

    def test_the_credential_corpus_independence_is_measured_and_disclosed(self):
        """THE CREDENTIAL SIDE OF THE ANTI-TAUTOLOGY GUARD -- gated for the
        first time here (pass-3 review, "Adjudication of the 44%", defect 2:
        "the credential-side floor is enforced by nothing", so the reported
        figures "are scratchpad-only and will drift silently").

        THIS TEST DELIBERATELY DOES NOT ASSERT THE 0.50 FLOOR, AND THAT IS THE
        POINT. Measured against the union at this commit:

            key/CRED    12/72 = 16.7%
            token/CRED  25/60 = 41.7%

        Both are far below the 0.50 the SAFE halves clear. That is NOT a defect
        to be fixed and the corpus MUST NOT be rewritten to clear it. It is the
        H-7 category distinction this plan has held throughout, and it is what
        `context_key_corpus.py`'s own banner says in the first line of the file:

            THIS IS A REGRESSION-PROBE + SHAPE-COVERAGE ARTIFACT.
            IT IS NOT AN INDEPENDENCE STATISTIC.

        A credential corpus is assembled from the vocabulary of real credentials
        -- `rsa`, `hmac`, `jwt`, `pem`, `signing`, `master`, `vault` -- which is
        necessarily the same vocabulary a credential DENYLIST enumerates. Low
        overlap-independence is what a *correct* credential regression probe
        looks like; a credential corpus scoring 90% independent would mean it had
        stopped naming credentials. The SAFE halves are the ones where the floor
        is meaningful, because there the shared vocabulary is evidence the corpus
        was grown from the allowlist under test (D-014, which passed at 0/100
        over-strip while shipping a live bypass).

        So what is pinned instead is a TWO-SIDED BAND, and it is a real pin:

          * a DROP below the band means the credential corpus has been diluted
            with names that are not credential vocabulary, weakening the
            regression probe;
          * a RISE above the band is the failure this whole plan is written to
            catch -- somebody "fixing" the number by rewriting the corpus rather
            than disclosing it, which is exactly what `LESSONS [I:4]` forbids and
            what SC-8 says must never happen.

        Both directions fail loudly. Neither is cleared by editing the corpus.
        """
        shipped_words = _shipped_vocabulary_union()

        # (measured, tolerance) -- +-0.05 absolute. Wide enough that adding a
        # handful of corpus names does not thrash the gate; narrow enough that
        # a rewrite of either credential half cannot pass unnoticed.
        for label, corpus, expected in (
            ("key", CRYPTO_KEY_SECRET_KEYS, 0.167),
            ("token", TOKEN_SECRET_KEYS, 0.417),
        ):
            words = _qualifier_words(corpus)
            outside = {w for w in words if w not in shipped_words}
            fraction = len(outside) / len(words)
            assert abs(fraction - expected) <= 0.05, (
                f"{label} credential half: vocabulary independence is "
                f"{len(outside)}/{len(words)} ({fraction:.1%}), pinned at "
                f"{expected:.1%} +-5pp.\n"
                "  If it ROSE: was the corpus rewritten to clear the 0.50 "
                "floor? SC-8 says a figure below the floor is DISCLOSED and "
                "pinned, NEVER cleared by rewriting the corpus. Revert.\n"
                "  If it FELL: the credential corpus is drifting away from real "
                "credential vocabulary and the regression probe is weakening.\n"
                "  If a shipped vocabulary genuinely grew, re-measure, update "
                "this pin, and say so in decisions.md."
            )

        # Vacuity guard: a band around a figure computed from an EMPTY word set
        # would pass trivially. Both halves must have real vocabulary.
        assert len(_qualifier_words(CRYPTO_KEY_SECRET_KEYS)) > 50
        assert len(_qualifier_words(TOKEN_SECRET_KEYS)) > 50

    # -- (c) anchoring ----------------------------------------------------
    def test_the_qualifier_scan_is_anchored_not_last_qualifier_only(self):
        """D-015 / the R1 finding, as a checked-in regression probe.

        D-014's negative lookaheads inspected ONLY the qualifier immediately
        before the trigger word, so prepending anything to an allowlisted
        qualifier defeated the strip. Every STRIP name in the second group
        below was KEPT by the shipped D-014 pattern.

        Both halves are asserted together on purpose: the anchoring fix is only
        correct if it leaves the ordinary vocabulary alone, and the `key` half
        of the fix is a polarity inversion whose whole justification is that it
        keeps names like `s3_key` and `config_key`.
        """
        must_keep = (
            # invariant I-7, also pinned in tests/test_fsm_llm_regression/
            "public_key",
            # the unbounded ordinary `<noun>_key` space that D-014 destroyed
            "s3_key",
            "user_key",
            "order_key",
            "config_key",
            "context_key",
            "state_key",
            "item_key",
            "cache_key",
            "primary_key",
            "foreign_key",
            "idempotency_key",
            # LLM metering and NLP vocabulary
            "max_tokens",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "token_count",
            "tokenizer",
            "next_page_token",
            "continuation_token",
        )
        must_strip = (
            # plain crypto/bearer shapes
            "private_key",
            "ssh_key",
            "master_key",
            "signing_key",
            "encryption_key",
            "api_key",
            "csrf_token",
            "jwt_token",
            "session_token",
            # THE BYPASS: an allowlisted qualifier with junk prepended. Every
            # one of these was KEPT before D-015.
            "ssh_public_key",
            "private_public_key",
            "signing_public_key",
            "ssh_cache_key",
            "master_index_key",
            "signing_search_key",
            "hmac_row_key",
            "csrf_max_token",
            "jwt_page_token",
            "session_sync_token",
            "bearer_cached_token",
        )
        # D-021 CHANGED WHICH LAYER STRIPS THE LAST FOUR, AND THAT IS DISCLOSED
        # HERE RATHER THAN ABSORBED. R1 made the bearer scan unbounded so that
        # `csrf_max_token` stripped on the NAME. That same unbounded scan also
        # strips `session_max_tokens`, `api_total_tokens` and
        # `auth_budget_tokens` -- plausible per-scope metering fields in an LLM
        # framework, which is the tightest collateral constraint on this
        # control -- so D-021 narrowed the bearer scan to qualifiers ADJACENT to
        # the trigger and referred the infixed shapes to the value layer. All 40
        # bullets below still resolve the same way; the last four now do it on
        # the VALUE, which is why they carry one.
        infixed_bearer = frozenset(
            {
                "csrf_max_token",
                "jwt_page_token",
                "session_sync_token",
                "bearer_cached_token",
            }
        )
        credential = "9dR2pQ7xL4mZ8vN3bK6tY1wJ5hG0sF2aD8cE4rT7uI"
        assert [k for k in must_keep if not self._kept(k)] == []
        assert [
            k
            for k in must_strip
            if self._kept(k, credential if k in infixed_bearer else "v")
        ] == []
        assert len(must_keep) + len(must_strip) == 40

    def test_the_infixed_bearer_residual_is_pinned(self):
        """D-021, the cost side of the adjacency narrowing above, pinned so it
        cannot be quietly forgotten.

        The residual is a CONJUNCTION: a bearer qualifier separated from the
        trigger by another word AND a value too short or too low-entropy for
        the value layer to see. Either alone is caught. Both together are not,
        and these names reach the prompt. They are pinned two-sided -- if a
        future change starts stripping them, that is a real improvement and the
        pin should be updated deliberately, not absorbed silently.
        """
        residual = ("csrf_max_token", "jwt_page_token", "bearer_cached_token")
        assert [k for k in residual if not self._kept(k, "abc")] == [], (
            "the disclosed infixed-bearer residual has changed. This is "
            "probably an improvement -- re-measure both axes and update D-021 "
            "and this pin together, rather than editing the pin alone."
        )
        # And the same names WITH a real credential still strip, so the
        # residual is bounded by value shape and not by the name at all.
        real = "9dR2pQ7xL4mZ8vN3bK6tY1wJ5hG0sF2aD8cE4rT7uI"
        assert [k for k in residual if self._kept(k, real)] == []

    # -- (d) anti-vacuity -------------------------------------------------
    def test_every_shipped_vocabulary_changes_the_result_set(self):
        """SC-6, second clause. A guard whose removal changes nothing is not a
        guard -- it is decoration that a future reader will delete, correctly
        observing that no test covers it. Each vocabulary is neutralised on its
        own, shown to matter on its own, AND shown to move the result set in
        the direction its polarity implies."""
        base = self._result_set(self._shipped_sources())
        vacuous = {}
        for label, (vocabulary, polarity) in self._vocabularies().items():
            index = self._locate(vocabulary)
            # `(?!)` can never match, so the group is neutralised in place
            # without disturbing the surrounding structure.
            got = self._result_set(self._mutated(index, vocabulary, "(?!)"))
            delta = got ^ base
            if not delta:
                vacuous[label] = "no change"
            elif polarity == "deny":
                assert delta <= base, (
                    f"{label}: neutralising a STRIP-list made keys newly "
                    "STRIPPED, which means its polarity is inverted somewhere"
                )
            else:
                assert delta <= got, (
                    f"{label}: neutralising a KEEP-list made keys newly KEPT, "
                    "which means its polarity is inverted somewhere"
                )
        assert vacuous == {}, (
            f"vacuous guards (removing them changes nothing): {vacuous}. "
            "This is the G-14 shape -- reject the pattern, do not ship it."
        )

    def test_the_token_qualifier_is_one_or_more_not_zero_or_more(self):
        """SC-6/G-14, stated twice because it is the single defect this seam has
        actually shipped. Structurally: `[a-z0-9]*` must not appear. Behaviourally:
        relaxing it must visibly destroy the allowlist, which proves the `+` is
        load-bearing rather than incidental. Only the `token` suffix arm still
        has an allowlist to defeat -- the `key` trigger no longer does."""
        base = self._result_set(self._shipped_sources())
        vocabulary, _ = self._vocabularies()["token/qualifier-allowlist"]
        index = self._locate(vocabulary)
        shipped = self._shipped_sources()[index]
        assert "[a-z0-9]*" not in shipped, (
            "token: a zero-or-more qualifier lets `.*[\\W_]` consume the safe "
            "qualifier and re-match a bare trigger, silently defeating the "
            "negative lookahead for EVERY allowlisted word (G-14)"
        )
        assert "[a-z0-9]+" in shipped

        # THE BEHAVIOURAL HALF IS MEASURED ON REFERRAL, NOT ON THE FINAL
        # VERDICT, AND D-021 IS WHY. Before D-021 this arm stripped on the name,
        # so relaxing `+` to `*` flipped 10+ safe names straight to STRIPPED and
        # the final verdict was the right thing to count. Now a name the
        # allowlist fails to protect is REFERRED to the value layer, which keeps
        # it anyway if its value is a metering count -- so counting final
        # verdicts under-reports the damage to 2 names and would let a real
        # regression through as "only two changed".
        #
        # What the `+` actually controls is the REFERRAL set: with `*`, the
        # `.*[\W_]` scan consumes the allowlisted qualifier and re-matches a
        # bare trigger, so every allowlisted name is referred and the allowlist
        # protects nothing. That is the G-14 defect, and it is still large.
        import re as _re

        # D-019 replaced the single optional separator with `_SEP_RUN`
        # (`[-_.]{0,4}`), so the surgery targets that spelling now.
        relaxed = _re.compile(
            shipped.replace("[a-z0-9]+[-_.]{0,4}", "[a-z0-9]*[-_.]{0,4}"),
            _re.IGNORECASE,
        )
        strict = _re.compile(shipped, _re.IGNORECASE)
        newly_referred = {
            k for k in TOKEN_SAFE_KEYS if relaxed.match(k) and not strict.match(k)
        }
        assert len(newly_referred) >= 10, (
            "token: relaxing the qualifier to zero-or-more referred only "
            f"{sorted(newly_referred)} of the safe corpus to the value layer. "
            "The `+` is supposed to be what keeps the whole allowlist alive; "
            "if it is not, the allowlist is vacuous."
        )
        # And the relaxation must still be visible in the final verdict too,
        # or the allowlist has no effect at all on what ships.
        newly_stripped = (
            self._result_set(
                self._mutated(index, "[a-z0-9]+[-_.]{0,4}", "[a-z0-9]*[-_.]{0,4}")
            )
            - base
        )
        assert newly_stripped, (
            "token: relaxing the qualifier changed no final verdict at all. "
            "The allowlist is decoration -- reject the pattern (G-14)."
        )

    def test_the_crypto_gap_is_bounded(self):
        """D-015 ReDoS control, asserted structurally as well as by timing.
        The gap scan sits inside an `(?:^|.*[\\W_])` scan, so an unbounded lazy
        gap makes the `key` pattern QUADRATIC on inputs like `"ssh_"*n` -- a
        denial of service on the prompt path, where this runs with the
        per-conversation lock held."""
        from fsm_llm import constants as c

        assert "{0,192}?" in c._CRYPTO_GAP, (
            "the crypto gap lost its bound; an unbounded lazy gap is quadratic "
            "here, not merely slower. See D-015 and the timing test below."
        )
        assert self._locate(c._CRYPTO_GAP) is not None

    # -- (c) ReDoS --------------------------------------------------------
    def test_the_patterns_are_linear_on_adversarial_input(self):
        """SC-6/I-9. These patterns run inside `_is_forbidden_context_key` on the
        prompt path, while the per-conversation lock is held, over context keys
        that are arbitrary consumer input -- so catastrophic backtracking here is
        a denial of service, not a slow test.

        Linearity is asserted on time-per-character rather than on the raw
        per-doubling ratio: for a linear pattern that quantity is flat, while
        for a quadratic one it doubles at every step (a 128x spread over this
        sweep). Using the spread rather than one adjacent pair keeps a single
        noisy sample from failing the run."""
        import re
        import time

        from fsm_llm.constants import COMPILED_FORBIDDEN_CONTEXT_PATTERNS

        assert all(
            isinstance(p, re.Pattern) for p in COMPILED_FORBIDDEN_CONTEXT_PATTERNS
        )

        adversarial = {
            'near-miss repeat "cach"*n+"_key"': lambda n: "cach" * n + "_key",
            'separator flood "a_"*n': lambda n: "a_" * n,
            'near-miss repeat "tok"*n+"_token"': lambda n: "tok" * n + "_token",
            # D-015: the `key` trigger is now a crypto DENYLIST reached through
            # a bounded lazy gap, so its worst case is a flood of crypto words
            # that never reaches a trigger. Unbounded, these two go quadratic.
            'crypto-word flood "ssh_"*n': lambda n: "ssh_" * n,
            'crypto near-miss "priv"*n+"_ke"': lambda n: "priv" * n + "_ke",
            'allowlisted repeat "cache_key_"*n': lambda n: "cache_key_" * n,
            'allowlisted repeat "max_tokens_"*n': lambda n: "max_tokens_" * n,
        }

        def measure(text):
            inner = max(1, 120_000 // len(text))
            best = None
            for _ in range(3):
                start = time.perf_counter()
                for _ in range(inner):
                    any(p.match(text) for p in COMPILED_FORBIDDEN_CONTEXT_PATTERNS)
                best = min(best or 1e9, (time.perf_counter() - start) / inner)
            return best

        for label, generate in adversarial.items():
            per_char = []
            for step in range(7):  # >= 6 doublings
                text = generate(64 << step)
                per_char.append(measure(text) / len(text))
            spread = max(per_char) / min(per_char)
            assert spread <= 2.5, (
                f"{label}: time per character varied {spread:.2f}x over 7 "
                f"doublings, which is not linear. ns/char by size: "
                f"{[round(t * 1e9, 1) for t in per_char]}"
            )


class TestValueShapeLayer:
    """D-019 / D-021 -- layer 2, the (name, value) half of the filter.

    D-017 recorded that `stripe_key` and `order_key` are inseparable as NAMES,
    and D-019 measured that they are trivially separable as PAIRS: a
    credential's value is high-entropy BY DEFINITION, because a low-entropy
    secret is already a broken secret. No name vocabulary can have that
    property. This class pins the consequences on both arms.
    """

    @staticmethod
    def _kept(key, value):
        return key in clean_context_keys(
            {key: value}, "test-conv", strip_forbidden_keys=True
        )

    # -- D-019: is the class claim true? ----------------------------------
    def test_the_generic_arm_carries_the_control(self):
        """THE LOAD-BEARING SECURITY CLAIM OF THIS ENTIRE LAYER, PINNED.

        `constants.py`'s D-019 block argues that layer 2 is a CLASS control and
        not an enumerated vendor list, on the grounds that the GENERIC arm
        (charset + character-class mix + length + Shannon entropy) does the work
        while the PREFIX arm (`_CREDENTIAL_VALUE_PREFIXES`: `sk_live_`, `ghp_`,
        `AKIA`, `AIza`, `xoxb-`, ...) is a disclosed denylist that "is NOT the
        arm the class claim rests on".

        That claim shipped UNPINNED for four attempts. A prior round cited a
        test of THIS NAME as pinning it; the test did not exist, and the citation
        was retracted at CLOSE of `plan-2026-07-20T040150-876e7164` (see the
        D-019 block). This is that test, written for real, now that step 1 has
        given the corpus credential VALUES to measure against.

        METHOD. Neutralise `_CREDENTIAL_VALUE_PREFIXES` to the empty tuple --
        `str.startswith(())` is always `False`, so the enumerated arm is
        switched off without touching any other rule -- and re-run every
        credential value in the shipped corpus. The question is what fraction of
        the detections the generic arm still makes on its own.

        MEASURED AT PLAN STEP 8: **99.4% (153/154)**. Every credential the
        filter detects bar one, it detects on shape alone. That is stronger than
        the 73%/27% the retracted citation claimed from a scratchpad.

        TWO CORRECTIONS TO WHAT THIS DOCSTRING SAID AT STEP 7, both found by
        step 8's audit and both recorded rather than quietly overwritten:

        1. Step 7 wrote "100% (148/148)". The share was right; the COUNT was
           wrong -- re-measured on step 7's own commit (`aee9753`) the corpus
           yields 153 head detections, not 148. A hand-transcribed denominator
           in a security anchor is exactly the class of claim
           `LESSONS [I:5]` says to grep before trusting.
        2. Step 7 wrote that 100% was "partly an artifact of WHICH vendor
           credentials this corpus holds", that the prefix arm is load-bearing
           only for a published vendor credential SHORTER than the 24-character
           floor, and that the corpus contained no such instance. All three were
           true. Step 8 CLOSED that gap rather than leaving it disclosed: the
           corpus now carries `vendor_prefix_short/credential`
           (`kiosk_key`, `AKIA...`, 20 chars). It is the single entry in
           `prefix_only` below, and it is why the share is no longer a clean
           100%. The number got worse because the instrument got better.

        So: do NOT read this test as licence to delete
        `_CREDENTIAL_VALUE_PREFIXES`. It buys the short tail, the corpus now
        contains that tail, and the second half of this test pins it besides.

        The floor is 0.50 -- a genuine majority claim, deliberately NOT a
        restatement of the measured 1.00. If it ever drops below half, the class
        argument in the D-019 block has become false and the block must be
        rewritten before the filter is.
        """
        from fsm_llm import constants as c

        credentials: list[tuple[str, object]] = []
        credentials += list(CRYPTO_KEY_SECRET_VALUES.items())
        credentials += list(TOKEN_SECRET_VALUES.items())
        credentials += list(TOKEN_SECRET_SHORT_VALUE_ENTRIES)
        credentials += [(n, v) for _, n, v in CARVE_OUT_CREDENTIAL_ENTRIES]

        head_detections = 0
        generic_detections = 0
        prefix_only: list[str] = []

        saved = c._CREDENTIAL_VALUE_PREFIXES
        try:
            for name, value in credentials:
                if type(value) is not str:
                    continue
                c._CREDENTIAL_VALUE_PREFIXES = saved
                at_head = c._looks_like_credential_value(value, name)
                c._CREDENTIAL_VALUE_PREFIXES = ()
                generic = c._looks_like_credential_value(value, name)
                if at_head:
                    head_detections += 1
                    if generic:
                        generic_detections += 1
                    else:
                        prefix_only.append(name)
        finally:
            c._CREDENTIAL_VALUE_PREFIXES = saved

        assert head_detections > 100, (
            "vacuity guard: the corpus must actually produce detections to "
            f"apportion, got {head_detections}"
        )
        share = generic_detections / head_detections
        assert share > 0.50, (
            f"THE CLASS CLAIM IS NOW FALSE. The generic shape arm carries only "
            f"{generic_detections}/{head_detections} ({share:.1%}) of layer 2's "
            "credential detections; the enumerated vendor-prefix denylist "
            f"carries the rest ({sorted(prefix_only)}).\n"
            "The D-019 block in constants.py argues this layer is a CLASS "
            "control because the generic arm does the work. Below 50% that "
            "argument is a vendor list wearing a class control's justification. "
            "Fix the WRITE-UP first -- retract or restate the class claim -- "
            "before touching the filter to make the number come back."
        )

        # The other side: the prefix arm is not dead weight. A published vendor
        # credential under the 24-char length floor is detected by it ALONE.
        # The corpus now carries one such instance itself
        # (`vendor_prefix_short/credential`); these two remain probed inline
        # because they are the PUBLISHED literals, and what a fixture may paste
        # is governed by `test_the_corpus_does_not_paste_the_pattern_it_tests`.
        for name, short_vendor_value in (
            ("aws_key", "AKIAIOSFODNN7EXAMPLE"),
            ("slack_token", "xoxb-1234"),
        ):
            assert c._looks_like_credential_value(short_vendor_value, name), (
                f"{name}: a published vendor credential stopped being detected"
            )
            c._CREDENTIAL_VALUE_PREFIXES = ()
            try:
                assert not c._looks_like_credential_value(short_vendor_value, name), (
                    f"{name}: this probe no longer demonstrates that the prefix "
                    "arm is load-bearing -- the generic arm now catches it too, "
                    "so pick a shorter/lower-entropy vendor format or retire "
                    "the second half of this test."
                )
            finally:
                c._CREDENTIAL_VALUE_PREFIXES = saved

    # -- D-021: the token arm's polarity ---------------------------------
    def test_a_metering_count_is_kept_and_a_bearer_string_is_stripped(self):
        """THE D-021 FINDING, as a test. The token arm's safe space is
        dominated by metering counts and its dangerous space by opaque
        strings, so on this arm the value TYPE is very nearly the whole
        signal -- far stronger than it is for `key`, where both sides are
        strings. The same NAME resolves both ways on the value alone."""
        for name in ("user_token", "billed_tokens", "service_token"):
            assert self._kept(name, 500), f"{name}: a count was stripped"
            assert self._kept(name, 4096.0), f"{name}: a float was stripped"
            assert not self._kept(name, "9dR2pQ7xL4mZ8vN3bK6tY1wJ5hG0sF2aD8cE4rT7uI"), (
                f"{name}: a bearer credential reached the prompt"
            )

    def test_the_token_arm_defaults_closed_on_an_unreadable_value(self):
        """The token arm's default is the OPPOSITE of the key arm's, and this
        is the test that says so. A value the layer cannot judge -- `None`,
        `bytes`, a container -- must STRIP, because that is the posture the
        name-only arm had before D-021 and relaxing an arm must not also
        relax its fallback. `_TOKEN_VALUE_SCAN_NAME_RE` is the exact set of
        names D-015 stripped unconditionally, so this is a strict statement
        that nothing became MORE permissive without a value to justify it."""
        from fsm_llm.constants import is_forbidden_context_entry

        for unreadable in (None, b"\x00\x01", {"nested": 1}, [1, 2], object()):
            assert is_forbidden_context_entry("harvest_token", unreadable), (
                f"{type(unreadable).__name__} value did not fail closed on the "
                "token arm"
            )
        # ... and the key arm keeps its own, opposite default.
        assert not is_forbidden_context_entry("order_key", None)

    def test_a_name_only_caller_gets_the_pre_d021_token_behaviour(self):
        """The documented asymmetry in `is_forbidden_context_entry`, pinned.
        Callers that pass no value still get the old unconditional strip on
        the token arm -- the safe direction -- so D-021 cannot silently widen
        the prompt surface for a caller that was never updated."""
        from fsm_llm.constants import is_forbidden_context_entry

        assert is_forbidden_context_entry("billed_tokens")
        assert not is_forbidden_context_entry("billed_tokens", 137)

    def test_pagination_cursors_are_carried_by_name_not_by_value(self):
        """The case that BREAKS the value signal, pinned so nobody deletes the
        allowlist believing layer 2 covers it. A cursor's value is an opaque
        high-entropy blob -- categorically indistinguishable from a bearer
        token -- so these survive only because their qualifier is allowlisted.
        The assertion pairs each cursor with a bearer credential carrying a
        value of the SAME shape, to show the name is doing all the work."""
        from fsm_llm.constants import _looks_like_credential_value

        cursor = "DXF1ZXJ5QW5kRmV0Y2gBAAAAAAAAB1ZaW5kZXgtMjAyNC0wNy0xOQ=="
        assert _looks_like_credential_value(cursor), (
            "this test is vacuous unless the cursor value really does look "
            "like a credential -- that is the whole point"
        )
        for allowlisted in ("next_page_token", "continuation_token", "sync_token"):
            assert self._kept(allowlisted, cursor)
        for not_allowlisted in ("harvest_token", "session_token"):
            assert not self._kept(not_allowlisted, cursor)

    # -- D-003: the fail-closed UUID/ULID identifier-noun carve-out -------
    def test_a_uuid_valued_credential_under_an_unlisted_name_now_strips(self):
        """THE FIX. Until D-003 the UUID/ULID carve-out was unconditional, so
        an API key generated as a canonical UUID was byte-identical to an
        idempotency key (A-2) and was KEPT. It is now conditioned on the NAME,
        and none of these names carries an identifier noun, so all of them
        fall through to the generic shape arm and strip.

        The three `*_token` names are the independently-authored holdout's
        entire token-arm UUID leak class -- the largest single fail-open class
        the holdout found, on a corpus that had never seen the pattern."""
        for name, value in (
            # both alphabets on the key arm, from the shipped shape corpus
            ("merchant_key", "7c3f1a92-4be8-4d17-9f60-2ab5c8e10d34"),
            ("partner_key", "01J9ZQ4T7XKD3M8VYB2NHF6CWE"),
            # the token arm, from the holdout
            ("consul_acl_token", "b41d7f6e-2c90-4a83-95e1-7fd0c6a3b82f"),
            ("nomad_management_token", "e07c4a19-83b5-4d62-a1f7-90cb2e5d4b18"),
            ("keycloak_offline_token", "5a9e2c71-64f8-4b03-8d1a-3ce7f096b25d"),
        ):
            assert not self._kept(name, value), (
                f"{name} carries a UUID/ULID-shaped credential and reached the "
                "prompt -- the D-003 carve-out has been widened back toward "
                "unconditional (disposition (c)), which leaks"
            )

    def test_a_uuid_valued_identifier_under_a_listed_noun_is_still_kept(self):
        """THE COST CONTROL, and the reason disposition (a) (delete the
        carve-out outright) was NOT shipped. (a) fixes exactly the same five
        credentials the test above pins and additionally destroys these -- real
        application data, silently removed from a prompt for no security gain.

        A failure here means the vocabulary shrank or the carve-out was
        deleted; either way over-strip has been spent buying nothing."""
        for name, value in (
            ("idempotency_key", "3f8b2c14-9d67-4a52-b0e3-7c1f5a94d208"),
            ("correlation_key", "3f8b2c14-9d67-4a52-b0e3-7c1f5a94d208"),
            ("dedupe_key", "01HZ8QK4PYRB6JT2WMXV3NCDGF"),
            ("request_key", "01HZ8QK4PYRB6JT2WMXV3NCDGF"),
            # the holdout's one independent safe UUID, matched via `replay` --
            # a noun that does NOT appear in the pre-D-003 code comment, and
            # therefore the only non-circular member of this list.
            ("replay_guard_key", "9c2e4f81-70a3-4b56-8e2d-1f45a9c70b63"),
        ):
            assert self._kept(name, value), (
                f"{name} is an ordinary identifier and was stripped -- this is "
                "disposition (a)'s measured cost, which D-003 declined to pay"
            )

    def test_the_identifier_noun_carve_out_is_fail_closed_for_unknown_names(self):
        """THE POLARITY, pinned on its own because it IS the design (plan S-2).

        A name that is neither a listed identifier noun nor an enumerated
        credential name gets NO carve-out and strips. That is what makes an
        omission in the vocabulary cost over-strip -- the axis with headroom --
        instead of fail-open. A future editor who flips this to fail-open (keep
        unless denylisted) reintroduces the `LESSONS [I:5]` failure shape that
        this seam has now hit five times.

        Matching is token-wise, not substring: `obatchment_key` must not ride
        `batch` into the carve-out."""
        uuid_value = "7c3f1a92-4be8-4d17-9f60-2ab5c8e10d34"
        for unknown in ("harvest_key", "widget_key", "obatchment_key"):
            assert not self._kept(unknown, uuid_value), (
                f"{unknown} matches no identifier noun and no credential "
                "vocabulary, so the fail-CLOSED default must strip it"
            )
        # And the listed-noun side of the same probe, so this is not vacuous.
        # `batch_key` is used rather than `session_key` because layer 1 strikes
        # `session_key` on the NAME and it never reaches this rule at all.
        assert self._kept("batch_key", uuid_value)

    # -- D-021: colon composites -----------------------------------------
    def test_a_colon_composite_credential_is_split_and_judged_on_its_tail(self):
        """The two fail-open leaks D-019 measured in its held-out slice. `:` is
        outside the credential charset because `sha256:` needs it excluded,
        which let `<numeric id>:<secret>` vendor formats through."""
        for name, value in (
            ("asana_key", "1/1201234567890123:aBcDeFgHiJkLmNoPqRsTuVwXyZ01"),
            ("cloudinary_key", "876543210987654:aBcDeFgHiJkLmNoPqRsTuVwXyZ0"),
        ):
            assert not self._kept(name, value), f"{name} still reaches the prompt"

    def test_a_labelled_hex_digest_is_still_kept(self):
        """The other side of the same rule, and an SC-5 named invariant.
        `cache_key` is pinned individually because the first cut of the colon
        rule keyed off the published digest LENGTHS and stripped it -- content
        addressing routinely truncates the digest, so length is not the
        signal and hex-ness is."""
        for value in (
            "sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822c",  # truncated
            "sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            "md5:9e107d9d372bb6826bd81d3542a419d6",
            "blake3:af1349b9f5f9a1a6a0404dea36dcc949",
        ):
            assert self._kept("cache_key", value), (
                f"cache_key: {value[:12]}... stripped"
            )
        assert self._kept("rate_limit_key", "ip:203.0.113.7")

    # -- D-005: the ONE charset/shape rule, and its ONE gaps list ---------
    def test_transport_wrapped_credentials_are_unwrapped_and_stripped(self):
        """D-005 fix 1. `Bearer <cred>` and `%2F`-encoded values used to fall
        outside `_CREDENTIAL_VALUE_CHARSET_RE` and were KEPT -- the value was
        dismissed as "structured application data" purely because it wore
        transport syntax. Both are normalised once, then judged."""
        for name, value in (
            ("gateway_key", "Bearer 9dR2pQ7xL4mZ8vN3bK6tY1wJ5hG0sF2aD8cE4rT7uI"),
            ("gateway_key", "Basic OWRSMnBRN3hMNG1aOHZOM2JLNnRZMXdKNWhHMHNGMmE"),
            ("upload_key", "Atzr%2FIQEBLjAsAhRmHNTV5xZ8pQwLmKjNbVcXsDfGhYtRe"),
            # The unwrap runs BEFORE the vendor-prefix arm on purpose, so a
            # wrapped self-identifying credential still reaches it.
            ("billing_key", "Bearer sk_live_9mK3aX7yB1nV5rD4jH2kM7"),
        ):
            assert not self._kept(name, value), (
                f"{name} carries a transport-wrapped credential and reached the "
                "prompt -- `_normalise_credential_value` has been bypassed"
            )

    def test_a_slash_bearing_base64_credential_is_no_longer_read_as_a_path(self):
        """D-005 fix 2, and the measured falsification of the old in-code claim
        that "the extension test alone carries almost all of it -- random base64
        does not end in `.pdf`". Standard base64 emits `/` and PASETO/JWT
        segments are `.`-joined, so both old path triggers fire on real
        credentials. The segment test is what separates them."""
        for name, value in (
            # extension-shaped tail: matched the old `_PATH_VALUE_RE`
            (
                "distribution_key",
                "9dR2pQ7xL4mZ8vN3bK6tY1wJ5hG0sF2/aD8cE4rT7uI.Xk3f9a",
            ),
            # four fields: matched the old bare `count("/") >= 3`
            ("mailer_key", "a8f3/d9c2b1e4/f7a0d3c6/b9e2f5a8d1c4e7b0a3d6f9Xk"),
        ):
            assert not self._kept(name, value), (
                f"{name} is base64 with slashes and reached the prompt -- the "
                "path carve-out no longer requires positive segment evidence"
            )

    def test_ordinary_paths_and_sentences_are_still_kept(self):
        """THE COST CONTROL for the whole D-005 edit, and the `LESSONS [I:4]`
        check it most needed: this edit pushes toward STRIP on three fronts at
        once, and fixing a filter's over-match is exactly how an under-match
        gets traded for a usability defect in the same commit. Every value here
        is ordinary application data whose shape overlaps a shape D-005 now
        catches. Over-strip did not move by one entry on either corpus, and this
        test is what keeps it that way."""
        for name, value in (
            ("s3_key", "invoices/2024/q3/invoice-10482.pdf"),
            ("object_key", "tenants/acme/exports/2024/09/orders"),
            ("route_key", "%2Fapi%2Fv2%2Forders%2Fsearch"),
            ("label_key", "Quarterly Revenue Report 2024"),
            ("cache_key", "sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2"),
            ("order_key", "ORD-10482"),
        ):
            assert self._kept(name, value), (
                f"{name} is ordinary application data and was stripped -- D-005 "
                "has over-corrected (Pre-Mortem 4)"
            )

    def test_the_unwrapping_is_two_shapes_deep_and_that_is_disclosed(self):
        """ACCEPTED GAP G2, pinned. Normalisation applies percent-decoding once
        and scheme-unwrapping once. Deeper or differently-shaped wrapping is
        still KEPT, and iterating to a fixed point was rejected because it turns
        a bounded normaliser into an unbounded one and reopens the decode
        surface `_VALUE_SCAN_LIMIT` exists to close.

        This test asserts the GAP, not the fix. It failing means the gap
        closed -- update the D-005 accepted-gaps list, do not delete this."""
        cred = "9dR2pQ7xL4mZ8vN3bK6tY1wJ5hG0sF2aD8cE4rT7uI"
        for label, value in (
            ("double percent-encoding", "Atzr%252FIQEBLjAsAhRmHNTV5xZ8pQwLmKj"),
            ("three fields", f"Bearer {cred} trailing"),
            ("unlisted scheme word", f"Sigv4Custom {cred}"),
        ):
            assert self._kept("gateway_key", value), (
                f"the {label} gap (G2) closed -- this is an improvement, but the "
                "D-005 accepted-gaps list now overstates the failure surface"
            )

    def test_a_credential_outside_the_charset_is_still_kept(self):
        """ACCEPTED GAP G3, pinned. A credential carrying any character outside
        `_CREDENTIAL_VALUE_CHARSET_RE` after normalisation is KEPT. Widening the
        charset is not available: `:` is excluded so `cache_key: "sha256:..."`
        survives (D-021), which `test_a_labelled_hex_digest_is_still_kept` pins.

        Asserts the GAP. `%E2%82%AC` is a VALID escape, so `errors="ignore"`
        does not drop it -- it decodes to a euro sign and lands outside the
        charset."""
        for value in (
            "Atzr%E2%82%ACIQEBLjAsAhRmHNTV5xZ8pQwLmKj",
            "9dR2pQ7xL4mZ8vN3bK6tY1wJ5hG0sF2a#D8cE4rT7uI",
            "9dR2pQ7xL4mZ8vN3bK6tY1wJ5hG0sF2a@D8cE4rT7uI",
        ):
            assert self._kept("upload_key", value), (
                "the charset gap (G3) closed -- update the D-005 accepted-gaps "
                "list and re-check `cache_key: sha256:...`"
            )

    def test_a_slash_credential_with_only_short_segments_is_still_kept(self):
        """ACCEPTED GAP G4, pinned -- the residual of the path fix. A
        slash-bearing credential whose segments are EACH below the length floor
        or single-class reads as a path and is KEPT. Strictly smaller than the
        `count("/") >= 3` gap it replaced, and disclosed rather than closed:
        closing it means judging the JOINED value, which strips `s3_key` and
        `object_key` (pinned above).

        Asserts the GAP."""
        for value in (
            "a8f3/d9c2/b1e4/f7a0/d3c6/b9e2/f5a8",
            "zqmwvjxrbtkdh/ngfplyscuearoibfe/xkdhngfply",
        ):
            assert self._kept("mailer_key", value), (
                "the path-segment gap (G4) closed -- re-check `s3_key` and "
                "`object_key` before updating the D-005 accepted-gaps list"
            )

    def test_the_length_floor_is_a_measured_refusal_not_an_omission(self):
        """ACCEPTED GAP G1, pinned TWO-SIDED, and the reason it is a gap.

        Credentials under `_MIN_CREDENTIAL_VALUE_LENGTH` are invisible to the
        generic arm; the NAME layer is their only control. D-005 swept the floor
        against both corpora and refused every lower value: the shortest pinned
        credential is 13 characters and any floor reaching it puts holdout
        over-strip at 18-22%, past the 15% bound. The safe side below is what
        that would have destroyed."""
        from fsm_llm.constants import _MIN_CREDENTIAL_VALUE_LENGTH

        assert _MIN_CREDENTIAL_VALUE_LENGTH == 24, (
            "the length floor moved. Re-run the two-axis floor sweep before "
            "accepting this: floor 22 costs 3 over-strips to catch 1 credential "
            "and floor 13 breaks the 15% bound on the holdout key arm"
        )
        assert self._kept("terminal_key", "aK9dQ2mZ7pL4x"), "G1 closed"
        for name, value in (
            ("field_key", "customer_email_v2"),
            ("natural_key", "SKU-8841-BLK-XL"),
            ("order_key", "ORD-10482"),
        ):
            assert self._kept(name, value), (
                f"{name} is what lowering the floor destroys -- it is the "
                "measured cost D-005 declined to pay"
            )

    # -- D-019 bypasses caught in adversarial review pass 2 ---------------
    def test_the_crypto_gap_reach_cliff_is_pinned_two_sided(self):
        """The `_CRYPTO_GAP` bound is a ReDoS control AND a hard limit on the
        control's REACH, and D-015 disclosed only the first role -- which let
        an exact, reproducible bypass cliff ship unmentioned. The cliff still
        EXISTS; it was moved from 64 to 192 characters, not removed. Pinned on
        BOTH sides so a future editor changing the number sees what it costs."""
        from fsm_llm.constants import is_forbidden_context_entry

        # The gap spans the two separators as well as the filler.
        assert is_forbidden_context_entry("ssh_" + "a" * 190 + "_key")
        assert not is_forbidden_context_entry("ssh_" + "a" * 191 + "_key")
        # The old 64-character cliff is genuinely gone.
        assert is_forbidden_context_entry("ssh_" + "a" * 65 + "_key")

    def test_a_digit_suffixed_crypto_word_still_strips(self):
        """`aes256_key` and `chacha20_key` differ from `aes_256_key` only by a
        missing separator, and the crypto word list holds the bare algorithm
        names, so without the `[0-9]*` the digit-suffixed spellings were kept."""
        from fsm_llm.constants import is_forbidden_context_entry

        for name in ("aes256_key", "aes_256_key", "chacha20_key", "rsa2048_key"):
            assert is_forbidden_context_entry(name), f"{name} reached the prompt"

    def test_a_doubled_separator_does_not_defeat_the_token_arm(self):
        """`[-_.]?` permitted at most ONE separator, so a second one left the
        qualifier match with nothing to consume and the arm silently kept
        `csrf__token`. Both the strip alternative and the allowlist lookahead
        use the bounded separator RUN, so `max__tokens` is still kept."""
        from fsm_llm.constants import is_forbidden_context_entry

        for name in ("csrf__token", "session__token", "jwt__token", "csrf--token"):
            assert is_forbidden_context_entry(name), f"{name} reached the prompt"
        assert not is_forbidden_context_entry("max__tokens", 4096)

    # -- security --------------------------------------------------------
    def test_the_value_is_never_written_into_a_log_or_an_exception(self):
        """THE PROHIBITION THAT MAKES THIS LAYER SAFE TO SHIP, enforced rather
        than documented. This layer exists to keep secrets OUT of prompts;
        logging the value it is inspecting would turn it into a mechanism that
        writes those same secrets to disk. Checked two ways -- no interpolation
        of a value-shaped name into any message in the layer's source, and no
        occurrence of a real secret in anything the filter emits."""
        import inspect
        import re as _re

        from fsm_llm import constants as c

        for function in (
            c._looks_like_credential_value,
            c._token_value_is_credential,
            c._colon_composite_tail,
            c._shannon_entropy,
            c.is_forbidden_context_entry,
        ):
            source = inspect.getsource(function)
            offenders = _re.findall(
                r"\{\s*(?:value|sample|stripped|tail|text|composite_tail)\b[^}]*\}",
                source,
            )
            assert offenders == [], (
                f"{function.__name__} interpolates the inspected value into a "
                f"message: {offenders}. See the prohibition in D-019/D-021 -- "
                "name the KEY and the reason, never the value."
            )

    def test_no_secret_value_appears_in_what_the_filter_emits(self):
        """The behavioural half of the prohibition above: drive real secrets
        through the live path and assert they appear in nothing it emits.

        The sink is asserted NON-EMPTY first. A test that captures no output
        passes this trivially, and "the secret was absent because nothing was
        logged" is exactly the false green this control cannot afford.
        """
        import io

        from fsm_llm.logging import logger

        secret = "9dR2pQ7xL4mZ8vN3bK6tY1wJ5hG0sF2aD8cE4rT7uI"
        sink = io.StringIO()
        logger.enable("fsm_llm")
        handler_id = logger.add(sink, format="{message}", level="DEBUG")
        try:
            clean_context_keys(
                {"harvest_token": secret, "stripe_key": secret, "keep_me": "ok"},
                "test-conv",
                strip_forbidden_keys=True,
            )
        finally:
            logger.remove(handler_id)

        emitted = sink.getvalue()
        assert "harvest_token" in emitted and "stripe_key" in emitted, (
            "the filter logged nothing about the keys it stripped, so this "
            f"test cannot see whether it also logged their values: {emitted!r}"
        )
        assert secret not in emitted, "the inspected value was written to a log record"


class TestHostileStrSubclasses:
    """D-006 -- the `str`-SUBCLASS rule, swept across the whole filter.

    `plans/LESSONS.md [I:4]` records that this codebase learned this rule at one
    call site (`memory.py` D-016) and then failed to generalise it, so the SAME
    hostile shape shipped live at three more sites in a security filter. These
    probes are deliberately one-per-DUNDER-or-METHOD rather than one-per-known-
    bug, so a future site that reintroduces any of them is caught here.

    Every assertion is on the VERDICT, not on the absence of an exception.
    `LESSONS [I:5]`: "no exception" is satisfied by silently KEEPING the value,
    which is strictly worse than the raise it replaced.
    """

    # A credential-shaped payload: 32 chars, mixed class, high entropy.
    SECRET = "9dR2pQ7mZ1xW4vB8nK6tL0sF2aD8cE4r"
    # Ordinary application data, kept at HEAD.
    SAFE = "checkout.button.submit"
    UUID = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"

    @staticmethod
    def _hostile_types():
        """One `str` subclass per hookable operation reachable from the filter.

        `__getitem__` and `.lower` are the two sites `plan.md` step 6 named;
        `__len__`/`__bool__` are the latent third site (the `if name`
        short-circuit step 4 added, reachable only under a UUID-shaped value);
        the rest are the sweep the brief asked for -- every operator the layer-2
        chain applies to a caller-supplied value.
        """

        class EvilSlice(str):
            def __getitem__(self, item):
                raise RuntimeError("boom-getitem")

        class EvilLower(str):
            def lower(self):
                raise RuntimeError("boom-lower")

        class EvilLen(str):
            def __len__(self):
                raise RuntimeError("boom-len")

        class EvilBool(str):
            def __bool__(self):
                raise RuntimeError("boom-bool")

        class EvilStrip(str):
            def strip(self, *args):
                raise RuntimeError("boom-strip")

            def lstrip(self, *args):
                raise RuntimeError("boom-lstrip")

        class EvilStartswith(str):
            def startswith(self, *args):
                raise RuntimeError("boom-startswith")

        class EvilContains(str):
            def __contains__(self, item):
                raise RuntimeError("boom-contains")

        class EvilSplit(str):
            def split(self, *args, **kwargs):
                raise RuntimeError("boom-split")

            def rsplit(self, *args, **kwargs):
                raise RuntimeError("boom-rsplit")

        class EvilIter(str):
            def __iter__(self):
                raise RuntimeError("boom-iter")

        class EvilEq(str):
            def __eq__(self, other):
                raise RuntimeError("boom-eq")

            def __hash__(self):
                return 0

        return [
            EvilSlice,
            EvilLower,
            EvilLen,
            EvilBool,
            EvilStrip,
            EvilStartswith,
            EvilContains,
            EvilSplit,
            EvilIter,
            EvilEq,
        ]

    def test_a_hostile_str_subclass_value_fails_closed(self):
        """Every hookable operation, on both arms, STRIPS -- never raises.

        Asserting `is True` and not merely "did not raise": a guard that
        swallowed the exception and returned False would satisfy the weaker
        claim while handing the credential to the prompt.
        """
        for hostile in self._hostile_types():
            for name in ("order_key", "harvest_token", "sk"):
                verdict = is_forbidden_context_entry(name, hostile(self.SECRET))
                assert verdict is True, (
                    f"{hostile.__name__} under {name!r} was not stripped "
                    f"({verdict!r}); a value whose shape cannot be trusted is "
                    "credential material (D-006, S-2)"
                )

    def test_a_benign_str_subclass_value_also_fails_closed(self):
        """The cost of D-006's polarity, pinned explicitly rather than implied.

        A subclass carrying ordinary data is stripped too. That is deliberate:
        the filter cannot tell a lying subclass from an honest one without
        running its code, and over-strip is the axis with headroom (S-1).
        """

        class Benign(str):
            pass

        assert is_forbidden_context_entry("order_key", self.SAFE) is False
        assert is_forbidden_context_entry("order_key", Benign(self.SAFE)) is True

    def test_a_hostile_str_subclass_key_is_still_decided_on_its_name(self):
        """The KEY arm reads the underlying buffer instead of failing closed.

        Failing closed on every subclass KEY would strip unrelated names
        wholesale, so this site is unbound-`str` (D-006). The verdict must
        therefore MATCH the plain-`str` verdict for the same name, not merely
        avoid raising -- otherwise the guard has silently changed the filter.
        """
        for hostile in self._hostile_types():
            for name, value in (
                ("order_key", self.SECRET),
                ("order_key", self.SAFE),
                ("harvest_token", self.SECRET),
                ("billed_tokens", 137),
                ("user_name", "alice"),
                ("api_password", "hunter2"),
            ):
                expected = is_forbidden_context_entry(name, value)
                actual = is_forbidden_context_entry(hostile(name), value)
                assert actual is expected, (
                    f"{hostile.__name__}({name!r}) got {actual!r} but a plain "
                    f"str got {expected!r}: the D-006 key guard changed a verdict"
                )

    def test_the_uuid_carve_out_name_read_is_not_a_latent_raise_site(self):
        """The third site: `if name` truthiness, reachable only under a UUID.

        Step 4 added a `if name else ()` short-circuit that read the subclass's
        `__len__`/`__bool__`. It survived step 5 because no probe supplied a
        UUID-shaped value, which is the only shape that reaches it.
        """
        for hostile in self._hostile_types():
            kept = is_forbidden_context_entry(hostile("idempotency_key"), self.UUID)
            stripped = is_forbidden_context_entry(hostile("merchant_key"), self.UUID)
            assert kept is False, (
                f"{hostile.__name__}('idempotency_key') lost its D-003 carve-out"
            )
            assert stripped is True, (
                f"{hostile.__name__}('merchant_key') leaked a UUID credential"
            )

    def test_the_internal_prefix_helper_survives_a_hostile_key(self):
        """`has_internal_prefix` is the fourth site, found by the D-006 sweep.

        It is outside `plan.md` step 6's stated two sites but inside the same
        file and the same defect class, and its own docstring claimed "Never
        raises". Same verdict-preservation contract as the key arm above.
        """
        for hostile in self._hostile_types():
            for name in ("system_flag", "SYSTEM_flag", "_hidden", "user_name"):
                expected = has_internal_prefix(name)
                actual = has_internal_prefix(hostile(name))
                assert actual is expected, (
                    f"{hostile.__name__}({name!r}) got {actual!r}, plain str got "
                    f"{expected!r}: the D-006 prefix guard changed a verdict"
                )

    def test_the_guard_leaks_nothing_into_a_log_or_a_traceback(self):
        """Invariant 2: no new path logs the value or re-raises carrying it.

        The hostile subclass's own exception message is the vehicle this step
        could most easily have leaked through -- a guard that caught and
        re-raised, or logged `repr(value)`, would put the payload in a
        traceback that reaches a log sink. Driven through the real
        `clean_context_keys` path with the sink asserted NON-EMPTY first.
        """
        import io

        from fsm_llm.logging import logger

        class EvilTalkative(str):
            def __getitem__(self, item):
                raise RuntimeError(f"payload-in-my-message: {str.__str__(self)}")

        sink = io.StringIO()
        logger.enable("fsm_llm")
        handler_id = logger.add(sink, format="{message}", level="DEBUG")
        try:
            cleaned = clean_context_keys(
                {
                    "order_key": EvilTalkative(self.SECRET),
                    "harvest_token": EvilTalkative(self.SECRET),
                    "keep_me": "ok",
                },
                "test-conv",
                strip_forbidden_keys=True,
            )
        finally:
            logger.remove(handler_id)

        emitted = sink.getvalue()
        assert "order_key" in emitted and "harvest_token" in emitted, (
            "the filter logged nothing about the keys it stripped, so this test "
            f"cannot see whether it also logged their values: {emitted!r}"
        )
        assert self.SECRET not in emitted, (
            "the hostile subclass's payload reached a log record"
        )
        assert "payload-in-my-message" not in emitted, (
            "the hostile subclass's own exception message reached a log record; "
            "a guard re-raised or logged with the value attached (invariant 2)"
        )
        assert "order_key" not in cleaned and "harvest_token" not in cleaned
        assert cleaned.get("keep_me") == "ok"


class TestBurnedHoldoutCorpus:
    """The independence corpus, after it was spent.

    `holdout_key_corpus.py` was authored in full BEFORE `constants.py` and
    `context_key_corpus.py` were opened in its authoring context, then measured
    against the shipped filter across steps 2-8 of
    `plan-2026-07-20T103203-b8a6b855`. That measurement BURNED it: it is a
    regression artifact now and its figures may never be re-quoted as an
    independence statistic (H-7, H-8; `plans/LESSONS.md` [I:5]).

    What ships here is therefore not an independence claim. It is (a) the
    mechanical guarantee that the two corpus artifacts stay separate, (b) the
    two-sided disclosure of every wrong verdict the corpus found, and (c) the
    bounds it was scored against, so that a future edit moving any of them has
    to say so.
    """

    @staticmethod
    def _stripped(key, value):
        return key not in clean_context_keys(
            {key: value}, "test-conv", strip_forbidden_keys=True
        )

    # -- SC-7: the two artifacts stay separate ----------------------------
    def test_the_burned_holdout_is_disjoint_from_the_regression_corpus(self):
        """SC-7. Collapsing a regression-probe set into an independence
        statistic is the metric category error that invalidated a prior plan's
        headline figure. The two files exist precisely so that cannot happen by
        accident, and this test is what stops it happening on purpose.

        NAME overlap must be EMPTY. VALUE overlap must be EXACTLY the one known,
        structurally-not-re-derivable scalar -- the literal ``True``. Arbitrary
        value overlap is NOT silently permitted: a second collision fails here
        and must be re-derived, not added to the exception.

        Values are compared by ``repr`` for non-``str`` so that ``True`` and
        ``1`` (equal and hash-equal in Python) do not silently merge, which
        would let a real integer collision hide behind the disclosed bool one.
        """
        from tests.test_fsm_llm.fixtures import context_key_corpus

        shipped_names: set[str] = set()
        shipped_values: set[str] = set()
        for attribute in dir(context_key_corpus):
            if attribute.startswith("__"):
                continue
            obj = getattr(context_key_corpus, attribute)
            if isinstance(obj, (tuple, list, frozenset, set)):
                for item in obj:
                    if isinstance(item, str):
                        shipped_names.add(item)
                    elif isinstance(item, tuple):
                        for sub in item:
                            if isinstance(sub, str):
                                shipped_names.add(sub)
                                shipped_values.add(sub)
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(key, str):
                        shipped_names.add(key)
                    shipped_values.add(value if isinstance(value, str) else repr(value))

        holdout_names = {name for name, _value, _truth in HOLDOUT}
        holdout_values = {
            value if isinstance(value, str) else repr(value)
            for _name, value, _truth in HOLDOUT
        }

        assert len(holdout_names) == len(HOLDOUT), (
            "the holdout contains duplicate names, so its denominators are wrong"
        )
        assert (holdout_names & shipped_names) == set(), (
            "the holdout shares names with the regression-probe corpus, so it is "
            "no longer a record of what an UNSIGHTED corpus found: "
            f"{sorted(holdout_names & shipped_names)}"
        )

        # The single disclosed exception, named explicitly rather than tolerated
        # as "a small overlap". `bool` has two inhabitants, so re-deriving this
        # value is impossible -- only the NAME could be (and was) re-derived.
        assert (holdout_values & shipped_values) == {"True"}, (
            "the value intersection is no longer exactly the one disclosed, "
            "not-re-derivable scalar `True`. Re-derive the new collision(s); do "
            "NOT widen this assertion: "
            f"{sorted(holdout_values & shipped_values)}"
        )

    def test_both_corpus_artifacts_declare_which_kind_of_artifact_they_are(self):
        """SC-7. A banner that drifts off a file is the same defect as never
        having written it. Both are asserted mechanically."""
        import pathlib

        from tests.test_fsm_llm.fixtures import context_key_corpus, holdout_key_corpus

        regression = pathlib.Path(context_key_corpus.__file__).read_text()
        assert "REGRESSION-PROBE + SHAPE-COVERAGE" in regression
        assert "NOT** AN INDEPENDENCE STATISTIC" in regression

        burned = pathlib.Path(holdout_key_corpus.__file__).read_text()
        assert "**BURNED**" in burned
        assert "MUST **NOT** BE USED AS AN INDEPENDENCE STATISTIC AGAIN" in burned
        assert "MUST derive a FRESH corpus" in burned

    # -- the disclosed residuals, two-sided -------------------------------
    def test_the_burned_holdout_fail_open_set_is_pinned_exactly(self):
        """Two-sided pin, same idiom as `TOKEN_KNOWN_OVER_STRIPPED`.

        A name LEAVING this set is a FIX and must be recorded in the banner. A
        name JOINING it is an undisclosed regression on a corpus that was
        authored without sight of the filter -- which is the only kind of
        regression evidence this seam has ever had.
        """
        actual = {
            name
            for name, value, truth in HOLDOUT
            if truth == "credential" and not self._stripped(name, value)
        }
        assert actual == HOLDOUT_KNOWN_FAIL_OPEN, (
            "the holdout's disclosed fail-open set moved.\n"
            "NEWLY LEAKING (undisclosed regression): "
            f"{sorted(actual - HOLDOUT_KNOWN_FAIL_OPEN)}\n"
            "NO LONGER LEAKING (a fix -- update the banner and the pin): "
            f"{sorted(HOLDOUT_KNOWN_FAIL_OPEN - actual)}"
        )

    def test_the_burned_holdout_over_strip_set_is_pinned_exactly(self):
        """The other axis, pinned with the same discipline. `LESSONS [I:4]`:
        fixing a filter's over-match silently creates an under-match in the same
        edit, so BOTH directions are pinned or neither measurement means
        anything."""
        actual = {
            name
            for name, value, truth in HOLDOUT
            if truth == "safe" and self._stripped(name, value)
        }
        assert actual == HOLDOUT_KNOWN_OVER_STRIPPED, (
            "the holdout's disclosed over-strip set moved.\n"
            "NEWLY OVER-STRIPPED: "
            f"{sorted(actual - HOLDOUT_KNOWN_OVER_STRIPPED)}\n"
            "NO LONGER OVER-STRIPPED: "
            f"{sorted(HOLDOUT_KNOWN_OVER_STRIPPED - actual)}"
        )

    # -- SC-3 / SC-4: the bounds, per-arm AND slice-total ------------------
    def test_the_burned_holdout_scores_inside_both_bounds_on_both_arms(self):
        """SC-3 and SC-4, mechanised in one test because they are one claim.

        Pre-Mortem 2 of the plan: *a one-axis improvement gets mistaken for a
        fix*. Every prior design on this seam moved error from one axis to the
        other with every gate green, because a stop trigger existed for one axis
        and not the other. So both axes are computed here, per arm AND
        slice-total -- six cells, all asserted, one failure message.

        The scoring is per-arm as well as total on purpose: at step 2 this
        corpus's token arm was at 12.5% fail-open while its key arm was at 2.4%,
        and a 7.4% slice-total would have masked exactly that.
        """
        bounds = {"over-strip": 15.0, "fail-open": 5.0}
        counts = {
            (axis, arm): [0, 0] for axis in bounds for arm in ("key", "token", "total")
        }

        for name, value, truth in HOLDOUT:
            axis = "fail-open" if truth == "credential" else "over-strip"
            stripped = self._stripped(name, value)
            wrong = (not stripped) if truth == "credential" else stripped
            for arm in (arm_of(name), "total"):
                counts[(axis, arm)][0] += 1
                counts[(axis, arm)][1] += int(wrong)

        vacuous = [key for key, (total, _) in counts.items() if total == 0]
        assert vacuous == [], f"empty cells make this test vacuous: {vacuous}"

        failures = []
        for (axis, arm), (total, wrong) in sorted(counts.items()):
            pct = 100.0 * wrong / total
            if pct > bounds[axis]:
                failures.append(
                    f"{axis} / {arm} arm: {pct:.1f}% ({wrong}/{total}) "
                    f"> {bounds[axis]:.0f}% bound"
                )
        assert failures == [], (
            "the BURNED holdout no longer scores inside its bounds:\n  "
            + "\n  ".join(failures)
            + "\n\nThis corpus is a regression artifact, so a cell moving out of "
            "bounds is a REGRESSION in the filter, not a new independence "
            "measurement. Do not re-tune a threshold to the specific entries "
            "named above (`plans/LESSONS.md` [I:4]); find what changed."
        )

    def test_the_burned_holdout_still_reaches_the_value_layer(self):
        """ANTI-VACUITY GUARD, and the reason the numbers above mean anything.

        A corpus whose credential half is struck at layer 1 on the NAME measures
        the NAME layer and says NOTHING about the value layer this plan changed.
        Step 2 measured this corpus at 58.0% value-attributable overall. The
        floor asserted here is a bare majority -- deliberately not a restatement
        of the measured figure -- because below half, the fail-open number above
        stops being a statement about layer 2 at all.

        For contrast, the shipped corpus's key arm is ~9% value-attributable and
        would fail this guard outright. That is correct for a regression-probe
        set built from credential-looking vendor names, and it is exactly why
        the two artifacts are not interchangeable.
        """
        inert = "v"
        value_attributable = 0
        credentials = 0
        for name, value, truth in HOLDOUT:
            if truth != "credential":
                continue
            credentials += 1
            if self._stripped(name, value) and not self._stripped(name, inert):
                value_attributable += 1

        assert credentials > 50, (
            f"vacuity: only {credentials} credential entries to apportion"
        )
        share = value_attributable / credentials
        assert share > 0.50, (
            f"only {value_attributable}/{credentials} ({share:.1%}) of the "
            "holdout's credentials are decided on their VALUE; the rest are "
            "struck on their NAME at layer 1. Below half, this corpus's "
            "fail-open figure is a statement about the name layer and must not "
            "be quoted as evidence about the value layer."
        )
