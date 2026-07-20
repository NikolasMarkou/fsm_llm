"""Unit tests for fsm_llm.context module."""

from __future__ import annotations

from fsm_llm.context import ContextCompactor, clean_context_keys
from tests.test_fsm_llm.fixtures.context_key_corpus import (
    CRYPTO_KEY_KNOWN_OVER_STRIPPED,
    CRYPTO_KEY_SAFE_KEYS,
    CRYPTO_KEY_SECRET_KEYS,
    KNOWN_OVER_STRIPPED,
    SAFE_KEYS,
    SECRET_KEYS,
    TOKEN_KNOWN_OVER_STRIPPED,
    TOKEN_SAFE_KEYS,
    TOKEN_SECRET_KEYS,
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
    def _kept(key):
        return key in clean_context_keys(
            {key: "v"}, "test-conv", strip_forbidden_keys=True
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
            "token/qualifier-allowlist": (c._SAFE_TOKEN_QUALIFIERS, "allow"),
            "token/material-head-denylist": (c._TOKEN_MATERIAL_HEADS, "deny"),
        }

    @staticmethod
    def _locate(fragment):
        """Index of the single shipped pattern containing `fragment`."""
        from fsm_llm.constants import FORBIDDEN_CONTEXT_PATTERNS

        hits = [i for i, p in enumerate(FORBIDDEN_CONTEXT_PATTERNS) if fragment in p]
        assert len(hits) == 1, (
            "the anti-vacuity surgery could not find its target exactly once in "
            f"the SHIPPED patterns (found {len(hits)}). The guard is now testing "
            "nothing -- re-derive it from constants.py."
        )
        return hits[0]

    @classmethod
    def _result_set(cls, patterns):
        import re

        compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
        return frozenset(
            k for k in cls._universe() if any(c.match(k) for c in compiled)
        )

    @classmethod
    def _mutated(cls, index, old, new):
        from fsm_llm.constants import FORBIDDEN_CONTEXT_PATTERNS

        patterns = list(FORBIDDEN_CONTEXT_PATTERNS)
        patterns[index] = patterns[index].replace(old, new)
        assert patterns[index] != FORBIDDEN_CONTEXT_PATTERNS[index]
        return patterns

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
        container it is read from."""
        leaked = [k for k in CRYPTO_KEY_SECRET_KEYS if self._kept(k)]
        assert leaked == [], f"{len(leaked)} crypto keys reached the prompt: {leaked}"

    def test_no_auth_token_reaches_the_prompt(self):
        """SC-4, direction 1. Possession of any of these values is sufficient
        to act as somebody."""
        leaked = [k for k in TOKEN_SECRET_KEYS if self._kept(k)]
        assert leaked == [], f"{len(leaked)} auth tokens reached the prompt: {leaked}"

    def test_ordinary_key_vocabulary_still_reaches_the_prompt(self):
        """SC-5, direction 2. Paired with the test above on purpose: a fix that
        stripped everything would pass that one and fail this one."""
        destroyed = [
            k
            for k in CRYPTO_KEY_SAFE_KEYS
            if k not in CRYPTO_KEY_KNOWN_OVER_STRIPPED and not self._kept(k)
        ]
        assert destroyed == [], f"{len(destroyed)} ordinary keys stripped: {destroyed}"

    def test_ordinary_token_vocabulary_still_reaches_the_prompt(self):
        """SC-5, direction 2. This is the tightest collateral constraint in the
        plan: this is an LLM framework and `*_tokens` metering vocabulary is
        pervasive, ordinary and non-secret."""
        destroyed = [
            k
            for k in TOKEN_SAFE_KEYS
            if k not in TOKEN_KNOWN_OVER_STRIPPED and not self._kept(k)
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
        actual = frozenset(k for k in CRYPTO_KEY_SAFE_KEYS if not self._kept(k))
        assert actual == CRYPTO_KEY_KNOWN_OVER_STRIPPED, (
            "the crypto-key over-strip class drifted.\n"
            f"  newly stripped (regression): "
            f"{sorted(actual - CRYPTO_KEY_KNOWN_OVER_STRIPPED)}\n"
            f"  newly kept (update the pin): "
            f"{sorted(CRYPTO_KEY_KNOWN_OVER_STRIPPED - actual)}"
        )

    def test_the_token_over_strip_set_is_pinned_exactly(self):
        """Two-sided, same rule."""
        actual = frozenset(k for k in TOKEN_SAFE_KEYS if not self._kept(k))
        assert actual == TOKEN_KNOWN_OVER_STRIPPED, (
            "the token over-strip class drifted.\n"
            f"  newly stripped (regression): "
            f"{sorted(actual - TOKEN_KNOWN_OVER_STRIPPED)}\n"
            f"  newly kept (update the pin): "
            f"{sorted(TOKEN_KNOWN_OVER_STRIPPED - actual)}"
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
        """
        import re as _re

        from fsm_llm import constants as c

        shipped_words = set()
        for blob in (
            c._CRYPTO_KEY_QUALIFIERS,
            c._KEY_MATERIAL_HEADS,
            c._SAFE_TOKEN_QUALIFIERS,
            c._TOKEN_MATERIAL_HEADS,
        ):
            shipped_words |= {w for w in blob.split("|") if w}

        def qualifiers(names):
            out = set()
            for name in names:
                for word in _re.split(r"[^a-zA-Z0-9]+", name.lower()):
                    if word and word not in ("key", "keys", "token", "tokens"):
                        out.add(word)
            return out

        measured = {}
        for label, corpus, floor in (
            ("key", CRYPTO_KEY_SAFE_KEYS, 0.50),
            ("token", TOKEN_SAFE_KEYS, 0.25),
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
        for label, corpus in (
            ("key", CRYPTO_KEY_SAFE_KEYS),
            ("token", TOKEN_SAFE_KEYS),
        ):
            kept = sum(1 for k in corpus if self._kept(k))
            assert kept / len(corpus) >= 0.85, (
                f"{label}: independent corpus keep-rate is {kept}/{len(corpus)}. "
                "The Pre-Mortem stop trigger for this step is >15% over-strip; "
                "STOP and report rather than tuning the vocabulary against the "
                "corpus, which recreates the tautology in reverse."
            )

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
        assert [k for k in must_keep if not self._kept(k)] == []
        assert [k for k in must_strip if self._kept(k)] == []

    # -- (d) anti-vacuity -------------------------------------------------
    def test_every_shipped_vocabulary_changes_the_result_set(self):
        """SC-6, second clause. A guard whose removal changes nothing is not a
        guard -- it is decoration that a future reader will delete, correctly
        observing that no test covers it. Each vocabulary is neutralised on its
        own, shown to matter on its own, AND shown to move the result set in
        the direction its polarity implies."""
        from fsm_llm.constants import FORBIDDEN_CONTEXT_PATTERNS

        base = self._result_set(FORBIDDEN_CONTEXT_PATTERNS)
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
        from fsm_llm.constants import FORBIDDEN_CONTEXT_PATTERNS

        base = self._result_set(FORBIDDEN_CONTEXT_PATTERNS)
        vocabulary, _ = self._vocabularies()["token/qualifier-allowlist"]
        index = self._locate(vocabulary)
        shipped = FORBIDDEN_CONTEXT_PATTERNS[index]
        assert "[a-z0-9]*" not in shipped, (
            "token: a zero-or-more qualifier lets `.*[\\W_]` consume the safe "
            "qualifier and re-match a bare trigger, silently defeating the "
            "negative lookahead for EVERY allowlisted word (G-14)"
        )
        assert "[a-z0-9]+" in shipped
        got = self._result_set(
            self._mutated(index, "[a-z0-9]+[-_.]?", "[a-z0-9]*[-_.]?")
        )
        newly_stripped = got - base
        assert len(newly_stripped) >= 10, (
            "token: relaxing the qualifier to zero-or-more changed only "
            f"{sorted(newly_stripped)}. The `+` is supposed to be what keeps "
            "the whole allowlist alive; if it is not, the pattern is vacuous."
        )

    def test_the_crypto_gap_is_bounded(self):
        """D-015 ReDoS control, asserted structurally as well as by timing.
        The gap scan sits inside an `(?:^|.*[\\W_])` scan, so an unbounded lazy
        gap makes the `key` pattern QUADRATIC on inputs like `"ssh_"*n` -- a
        denial of service on the prompt path, where this runs with the
        per-conversation lock held."""
        from fsm_llm import constants as c

        assert "{0,64}?" in c._CRYPTO_GAP, (
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
