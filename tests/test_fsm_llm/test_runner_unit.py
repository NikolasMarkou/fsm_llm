from __future__ import annotations

"""Unit tests for runner.py — the CLI runner module."""

import json
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


class TestRunnerUsesAPI:
    """Verify runner uses API class instead of FSMManager."""

    def test_runner_imports_api_not_fsm_manager(self):
        """runner.py should import API, not FSMManager."""
        import inspect

        from fsm_llm import runner

        source = inspect.getsource(runner)
        assert "API" in source
        # Should not import FSMManager (mentions in comments/docstrings are OK)
        import_lines = [
            ln for ln in source.split("\n") if ln.strip().startswith(("from", "import"))
        ]
        for line in import_lines:
            assert "FSMManager" not in line, f"runner.py imports FSMManager: {line}"

    def test_runner_main_requires_llm_model_env(self):
        """runner.main() should raise if LLM_MODEL env var is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("fsm_llm.runner.dotenv.load_dotenv"):
                with pytest.raises(
                    RuntimeError, match="Missing required environment variable"
                ):
                    from fsm_llm.runner import main

                    main(fsm_path=None, max_history_size=5, max_message_length=1000)

    def test_runner_main_creates_api_instance(self):
        """runner.main() should call API.from_file()."""
        mock_api = MagicMock()
        mock_api.start_conversation.return_value = ("conv-1", "Hello!")
        mock_api.has_conversation_ended.return_value = True
        mock_api.get_data.return_value = {}

        env = {
            "LLM_MODEL": "test-model",
            "LLM_TEMPERATURE": "0.5",
            "LLM_MAX_TOKENS": "100",
        }

        with patch.dict(os.environ, env, clear=True):
            with patch("fsm_llm.runner.dotenv.load_dotenv"):
                with patch(
                    "fsm_llm.runner.API.from_file", return_value=mock_api
                ) as mock_from_file:
                    with patch("fsm_llm.runner.setup_file_logging"):
                        from fsm_llm.runner import main

                        result = main(
                            fsm_path="/tmp/test.json",
                            max_history_size=5,
                            max_message_length=1000,
                        )

                        mock_from_file.assert_called_once_with(
                            "/tmp/test.json",
                            model="test-model",
                            temperature=0.5,
                            max_tokens=100,
                            max_history_size=5,
                            max_message_length=1000,
                        )
                        mock_api.start_conversation.assert_called_once()
                        mock_api.end_conversation.assert_called_once_with("conv-1")
                        assert result == 0

    def test_runner_uses_converse_not_process_message(self):
        """runner should use fsm.converse(), not fsm_manager.process_message()."""
        mock_api = MagicMock()
        mock_api.start_conversation.return_value = ("conv-1", "Hello!")
        mock_api.has_conversation_ended.side_effect = [False, True]
        mock_api.converse.return_value = "Response"
        mock_api.get_data.return_value = {}

        env = {"LLM_MODEL": "test-model"}

        with patch.dict(os.environ, env, clear=True):
            with patch("fsm_llm.runner.dotenv.load_dotenv"):
                with patch("fsm_llm.runner.API.from_file", return_value=mock_api):
                    with patch("fsm_llm.runner.setup_file_logging"):
                        with patch("builtins.input", return_value="test input"):
                            from fsm_llm.runner import main

                            result = main("/tmp/t.json", 5, 1000)

                            mock_api.converse.assert_called_once_with(
                                user_message="test input", conversation_id="conv-1"
                            )
                            assert result == 0

    def test_runner_handles_exit_command(self):
        """runner should exit cleanly on 'exit' command."""
        mock_api = MagicMock()
        mock_api.start_conversation.return_value = ("conv-1", "Hello!")
        mock_api.has_conversation_ended.return_value = False
        mock_api.get_data.return_value = {}

        env = {"LLM_MODEL": "test-model"}

        with patch.dict(os.environ, env, clear=True):
            with patch("fsm_llm.runner.dotenv.load_dotenv"):
                with patch("fsm_llm.runner.API.from_file", return_value=mock_api):
                    with patch("fsm_llm.runner.setup_file_logging"):
                        with patch("builtins.input", return_value="exit"):
                            from fsm_llm.runner import main

                            result = main("/tmp/t.json", 5, 1000)
                            assert result == 0
                            # converse should NOT have been called
                            mock_api.converse.assert_not_called()


class TestRedactContextRecurses:
    """D-014 part (b). `_redact_context` is the CLI log-redaction control
    (D-015). It matched only top-level keys, so a nested secret was written
    VERBATIM to the log at both of its call sites. Steps 9/10 closed the same
    top-level-only gap in `context.py` and `prompts.py`; this closes the third.
    """

    def test_nested_secret_is_redacted_and_the_key_stays_visible(self):
        from fsm_llm.runner import _REDACTED, _redact_context

        result = _redact_context({"user": {"password": "hunter2", "name": "bob"}})

        # SC-16b. The key is deliberately KEPT (unlike `clean_context_keys`,
        # which drops it) -- the log must show that a redacted key existed.
        assert result == {"user": {"password": _REDACTED, "name": "bob"}}
        assert "hunter2" not in json.dumps(result)

    def test_top_level_behavior_is_unchanged(self):
        """Control: the pre-existing D-015 semantics must survive the recursion."""
        from fsm_llm.runner import _REDACTED, _redact_context

        result = _redact_context({"api_key": "sk-live", "city": "Athens"})

        assert result == {"api_key": _REDACTED, "city": "Athens"}

    def test_secret_inside_a_list_of_dicts_is_redacted(self):
        """Same list-nesting scope as D-010: `{"users": [{"password": ...}]}`
        is the same leak as the nested-dict form."""
        from fsm_llm.runner import _REDACTED, _redact_context

        result = _redact_context({"users": [{"password": "p1"}, {"name": "bob"}]})

        assert result == {"users": [{"password": _REDACTED}, {"name": "bob"}]}
        assert "p1" not in json.dumps(result)

    def test_falsy_values_survive_at_every_depth(self):
        """The `[SOFT]` contract: `[]`/`""`/`0`/`False` are meaningful data."""
        from fsm_llm.runner import _redact_context

        payload = {"a": {"b": {"allergies": [], "count": 0, "ok": False, "s": ""}}}

        assert _redact_context(payload) == payload

    def test_container_past_the_depth_bound_is_redacted_not_passed_through(self):
        """Fail-CLOSED at the bound, matching D-010. Passing the sub-tree
        through unredacted would hand an attacker a one-line bypass: bury the
        secret 17 levels down and the CLI log prints it."""
        from fsm_llm.constants import MAX_CONTEXT_FILTER_DEPTH
        from fsm_llm.runner import _REDACTED, _redact_context

        payload: dict = {}
        cursor = payload
        for _ in range(MAX_CONTEXT_FILTER_DEPTH + 3):
            cursor["n"] = {}
            cursor = cursor["n"]
        cursor["password"] = "buried"

        rendered = json.dumps(_redact_context(payload))

        assert "buried" not in rendered
        assert _REDACTED in rendered

    def test_payload_just_inside_the_bound_is_still_redacted_normally(self):
        """Vacuity guard for the test above: a bound of 0 would satisfy it
        alone. This pins that legitimate depth is still walked, not truncated.
        """
        from fsm_llm.runner import _REDACTED, _redact_context

        payload = {"l1": {"l2": {"l3": {"password": "p", "keep": "yes"}}}}

        assert _redact_context(payload) == {
            "l1": {"l2": {"l3": {"password": _REDACTED, "keep": "yes"}}}
        }

    def test_non_string_keys_do_not_crash_the_filter(self):
        """Recursing into arbitrary nested data makes an int-keyed dict
        reachable; `pattern.match(1)` raises `TypeError`."""
        from fsm_llm.runner import _REDACTED, _redact_context

        assert _redact_context({"a": {1: "x", "password": "y"}}) == {
            "a": {1: "x", "password": _REDACTED}
        }

    def test_no_secret_pattern_is_inlined_in_the_runner(self):
        """D-015's single-sourcing constraint, still in force after the
        recursion was added: the REGEX list must stay in `security.py`
        (re-exported by `constants.py`)."""
        import inspect

        from fsm_llm import runner

        source = inspect.getsource(runner)
        # D-019 routed all three call sites through the one shared decision
        # point, which now takes the VALUE as well as the name. The intent of
        # this assertion is unchanged: runner must IMPORT the shared control,
        # never re-implement the match locally.
        assert "is_forbidden_context_entry" in source
        assert "re.compile" not in source
        for literal in ("password", "api_key", "secret", "token"):
            assert f'r"{literal}' not in source


class TestRedactContextNonStringKeys:
    """D-017 / concern 7: a non-str key cannot be pattern-matched, so it
    bypasses redaction entirely. Behaviour is unchanged -- but announced."""

    def test_non_string_key_value_still_recurses(self):
        from fsm_llm.runner import _redact_context

        result = _redact_context({0: {"password": "hunter2"}, "name": "Alice"})
        assert result[0]["password"] == "<redacted>"
        assert result["name"] == "Alice"

    def test_bytes_key_bypass_is_logged_at_warning(self):
        from fsm_llm.logging import logger
        from fsm_llm.runner import _redact_context

        records = []
        sink_id = logger.add(lambda m: records.append(m.record), level="WARNING")
        try:
            result = _redact_context({b"password": "BYTES-SECRET"})
        finally:
            logger.remove(sink_id)

        assert result[b"password"] == "BYTES-SECRET"
        assert any("Log-redaction skipped" in r["message"] for r in records)

    def test_string_keys_are_not_warned_about(self):
        """Vacuity guard: the warning must be specific to non-str keys."""
        from fsm_llm.logging import logger
        from fsm_llm.runner import _redact_context

        records = []
        sink_id = logger.add(lambda m: records.append(m.record), level="WARNING")
        try:
            _redact_context({"password": "p", "name": "Alice"})
        finally:
            logger.remove(sink_id)

        assert not any("Log-redaction skipped" in r["message"] for r in records)


class TestRunnerJsonDumpsSurvivesNonJsonNativeContextValues:
    """A handler storing a `datetime` (or other non-JSON-native value) via
    `update_context` must not crash the CLI's debug/dump JSON logging path
    (item 4 / plan step 8). `json.dumps` has no fallback for a `datetime`
    object and raises `TypeError: Object of type datetime is not JSON
    serializable` by default -- `logging.py:90`'s own `_record_to_json`
    already guards this with `default=str`; `runner.py`'s two CLI-path call
    sites (the per-turn debug dump and the final data dump) did not.

    Driven through the public `runner.main()` CLI entry point (mocked
    `API`), not the private `json.dumps` call directly, per this codebase's
    "every new regression test drives the public path" convention.
    """

    def _run_with_context(self, per_turn_data, final_data):
        mock_api = MagicMock()
        mock_api.start_conversation.return_value = ("conv-1", "Hello!")
        mock_api.has_conversation_ended.side_effect = [False, True]
        mock_api.converse.return_value = "Response"
        # First get_data() call happens inside the turn loop (the per-turn
        # debug dump); the second happens after the loop exits (the final
        # data dump) -- see runner.py's main().
        mock_api.get_data.side_effect = [per_turn_data, final_data]

        env = {"LLM_MODEL": "test-model"}
        with patch.dict(os.environ, env, clear=True):
            with patch("fsm_llm.runner.dotenv.load_dotenv"):
                with patch("fsm_llm.runner.API.from_file", return_value=mock_api):
                    with patch("fsm_llm.runner.setup_file_logging"):
                        with patch("builtins.input", return_value="hi"):
                            from fsm_llm.runner import main

                            return main("/tmp/t.json", 5, 1000)

    def test_datetime_in_final_dump_does_not_crash_the_cli(self):
        """RED (pre-fix): the final dump (outside the per-turn try/except)
        raised an uncaught TypeError all the way out of main(). GREEN
        (post-fix): main() returns 0, no exception."""
        result = self._run_with_context(
            per_turn_data={"turn": 1},
            final_data={"turn": 1, "seen_at": datetime.now(timezone.utc)},
        )
        assert result == 0

    def test_datetime_in_per_turn_dump_does_not_crash_the_cli(self):
        """The per-turn debug dump (line ~199) is inside a broad
        `except Exception` that swallows the crash into `return -1` even
        pre-fix -- still wrong (a serialization bug is not a converse
        failure) but not an uncaught crash. Post-fix it must return 0, not
        -1, proving the per-turn site is genuinely fixed rather than merely
        caught by the surrounding handler."""
        result = self._run_with_context(
            per_turn_data={"turn": 1, "seen_at": datetime.now(timezone.utc)},
            final_data={"turn": 1},
        )
        assert result == 0

    def test_secret_bearing_object_gets_safe_placeholder_not_repr(self):
        """D-014/D-015 / review-iter-2.md WARNING 4 regression: `default=str`
        stringifies an arbitrary object via `repr()`, so a secret-bearing
        object under a benign key used to reach persisted logs verbatim --
        on the CLI's own redaction path. Post-fix (`_json_default`), only a
        type-name placeholder is logged, and one WARNING is emitted."""
        from fsm_llm.logging import logger

        class Creds:
            """Stand-in for a caller's custom object whose repr leaks a
            secret field value -- exactly what `_json_default` must not
            call."""

            def __init__(self, api_key: str):
                self.api_key = api_key

            def __repr__(self) -> str:
                return f"Creds(api_key={self.api_key!r})"

        secret = "sk-live-SUPERSECRET"  # test fixture, not a real credential
        creds = Creds(secret)

        records: list = []
        logger.enable("fsm_llm")
        # level="DEBUG" (not "WARNING"): the assertions below need to see
        # the actual logged JSON dump line too, not just the WARNING.
        sink_id = logger.add(lambda m: records.append(m.record), level="DEBUG")
        try:
            result = self._run_with_context(
                per_turn_data={"turn": 1},
                final_data={"session": creds, "note": "hi"},
            )
        finally:
            logger.remove(sink_id)
            logger.disable("fsm_llm")

        assert result == 0

        # The secret's raw value, and the object's own repr(), must never
        # appear in any captured log line -- this is the leak the fix closes.
        for record in records:
            message = str(record["message"])
            assert secret not in message
            assert "Creds(api_key=" not in message

        # The type-name placeholder DOES reach the final dump line.
        assert any("<non-serializable: Creds>" in str(r["message"]) for r in records)

        # Exactly one WARNING fired for the one non-serializable object.
        warnings = [
            r
            for r in records
            if r["level"].name == "WARNING" and "non-JSON-native" in str(r["message"])
        ]
        assert len(warnings) == 1, (
            f"expected exactly 1 non-JSON-native WARNING, got "
            f"{len(warnings)}: {[str(r['message']) for r in records]}"
        )
