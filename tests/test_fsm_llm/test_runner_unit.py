from __future__ import annotations

"""Unit tests for runner.py — the CLI runner module."""

import json
import os
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from fsm_llm.core.definitions import ModelNotConfiguredError
from tests.conftest import cpu_timer
from tests.test_fsm_llm.support import _JOIN_TIMEOUT_S


@contextmanager
def _library_records(level: str):
    """Yield the loguru records of the library, with its logging switched on.

    The library disables its own loggers at import, so a sink added to
    ``fsm_llm.logging.logger`` sees nothing unless the test enables them
    itself. Relying on an earlier test having left them enabled made the
    result depend on test order. Ends with the library disabled again, the
    state it has at import.
    """
    from fsm_llm.logging import logger

    records: list[dict[str, Any]] = []
    logger.enable("fsm_llm")
    sink_id = logger.add(lambda m: records.append(m.record), level=level)
    try:
        yield records
    finally:
        logger.remove(sink_id)
        logger.disable("fsm_llm")


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
        """runner.main() should raise ModelNotConfiguredError if LLM_MODEL is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("fsm_llm.runner.load_env_file"):
                with pytest.raises(ModelNotConfiguredError, match="LLM_MODEL"):
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
            with patch("fsm_llm.runner.load_env_file"):
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
            with patch("fsm_llm.runner.load_env_file"):
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
            with patch("fsm_llm.runner.load_env_file"):
                with patch("fsm_llm.runner.API.from_file", return_value=mock_api):
                    with patch("fsm_llm.runner.setup_file_logging"):
                        with patch("builtins.input", return_value="exit"):
                            from fsm_llm.runner import main

                            result = main("/tmp/t.json", 5, 1000)
                            assert result == 0
                            # converse should NOT have been called
                            mock_api.converse.assert_not_called()


class TestRedactContextRecurses:
    """D-014 part (b). `redact_context` is the CLI log-redaction control
    (D-015). It matched only top-level keys, so a nested secret was written
    VERBATIM to the log at both of its call sites. Steps 9/10 closed the same
    top-level-only gap in `context.py` and `prompts.py`; this closes the third.
    """

    def test_nested_secret_is_redacted_and_the_key_stays_visible(self):
        from fsm_llm.core.redaction import REDACTED, redact_context

        result = redact_context({"user": {"password": "hunter2", "name": "bob"}})

        # SC-16b. The key is deliberately KEPT (unlike `clean_context_keys`,
        # which drops it) -- the log must show that a redacted key existed.
        assert result == {"user": {"password": REDACTED, "name": "bob"}}
        assert "hunter2" not in json.dumps(result)

    def test_top_level_behavior_is_unchanged(self):
        """Control: the pre-existing D-015 semantics must survive the recursion."""
        from fsm_llm.core.redaction import REDACTED, redact_context

        result = redact_context({"api_key": "sk-live", "city": "Athens"})

        assert result == {"api_key": REDACTED, "city": "Athens"}

    def test_secret_inside_a_list_of_dicts_is_redacted(self):
        """Same list-nesting scope as D-010: `{"users": [{"password": ...}]}`
        is the same leak as the nested-dict form."""
        from fsm_llm.core.redaction import REDACTED, redact_context

        result = redact_context({"users": [{"password": "p1"}, {"name": "bob"}]})

        assert result == {"users": [{"password": REDACTED}, {"name": "bob"}]}
        assert "p1" not in json.dumps(result)

    def test_falsy_values_survive_at_every_depth(self):
        """The `[SOFT]` contract: `[]`/`""`/`0`/`False` are meaningful data."""
        from fsm_llm.core.redaction import redact_context

        payload = {"a": {"b": {"allergies": [], "count": 0, "ok": False, "s": ""}}}

        assert redact_context(payload) == payload

    def test_container_past_the_depth_bound_is_redacted_not_passed_through(self):
        """Fail-CLOSED at the bound, matching D-010. Passing the sub-tree
        through unredacted would hand an attacker a one-line bypass: bury the
        secret 17 levels down and the CLI log prints it."""
        from fsm_llm.constants import MAX_CONTEXT_FILTER_DEPTH
        from fsm_llm.core.redaction import REDACTED, redact_context

        payload: dict = {}
        cursor = payload
        for _ in range(MAX_CONTEXT_FILTER_DEPTH + 3):
            cursor["n"] = {}
            cursor = cursor["n"]
        cursor["password"] = "buried"

        rendered = json.dumps(redact_context(payload))

        assert "buried" not in rendered
        assert REDACTED in rendered

    def test_payload_just_inside_the_bound_is_still_redacted_normally(self):
        """Vacuity guard for the test above: a bound of 0 would satisfy it
        alone. This pins that legitimate depth is still walked, not truncated.
        """
        from fsm_llm.core.redaction import REDACTED, redact_context

        payload = {"l1": {"l2": {"l3": {"password": "p", "keep": "yes"}}}}

        assert redact_context(payload) == {
            "l1": {"l2": {"l3": {"password": REDACTED, "keep": "yes"}}}
        }

    def test_non_string_keys_do_not_crash_the_filter(self):
        """Recursing into arbitrary nested data makes an int-keyed dict
        reachable; `pattern.match(1)` raises `TypeError`."""
        from fsm_llm.core.redaction import REDACTED, redact_context

        assert redact_context({"a": {1: "x", "password": "y"}}) == {
            "a": {1: "x", "password": REDACTED}
        }

    def test_no_secret_pattern_is_inlined_in_the_redactor(self):
        """D-015's single-sourcing constraint, still in force after the
        recursion was added: the REGEX list must stay in `security/patterns.py`
        (re-exported by `constants.py`)."""
        import inspect

        from fsm_llm import runner
        from fsm_llm.core import redaction

        # The walk lives in core.redaction; the runner only calls it.
        source = inspect.getsource(redaction.redact_context) + inspect.getsource(
            redaction._redact_mapping
        )
        assert "redact_context" in inspect.getsource(runner)
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
        from fsm_llm.core.redaction import redact_context

        result = redact_context({0: {"password": "hunter2"}, "name": "Alice"})
        assert result[0]["password"] == "<redacted>"
        assert result["name"] == "Alice"

    def test_bytes_key_bypass_is_logged_at_warning(self):
        from fsm_llm.core.redaction import redact_context

        with _library_records("WARNING") as records:
            result = redact_context({b"password": "BYTES-SECRET"})

        assert result[b"password"] == "BYTES-SECRET"
        assert any("Log-redaction skipped" in r["message"] for r in records)

    def test_string_keys_are_not_warned_about(self):
        """Vacuity guard: the warning must be specific to non-str keys."""
        from fsm_llm.core.redaction import redact_context

        with _library_records("WARNING") as records:
            redact_context({"password": "p", "name": "Alice"})

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

        # The per-turn dump is lazy: it runs (and takes the first get_data
        # value) only while a DEBUG sink is live, so the test turns one on
        # instead of inheriting it from an earlier test.
        env = {"LLM_MODEL": "test-model"}
        with _library_records("DEBUG"):
            with patch.dict(os.environ, env, clear=True):
                with patch("fsm_llm.runner.load_env_file"):
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
            final_data={"turn": 1, "seen_at": datetime.now(UTC)},
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
            per_turn_data={"turn": 1, "seen_at": datetime.now(UTC)},
            final_data={"turn": 1},
        )
        assert result == 0

    def test_datetime_in_final_dump_prints_as_its_string(self, capsys):
        """F-AF-11: the CLI dump uses the one shared hook, so a date in the
        context reads as its string (as in the prompt and the session file)
        instead of the `<non-serializable: datetime>` placeholder."""
        result = self._run_with_context(
            per_turn_data={"turn": 1},
            final_data={"seen_at": datetime(2026, 1, 2, 3, 4, 5)},
        )
        out = capsys.readouterr().out
        assert result == 0
        assert "2026-01-02 03:04:05" in out
        assert "non-serializable" not in out

    def test_secret_bearing_object_gets_safe_placeholder_not_repr(self):
        """D-014/D-015 / review-iter-2.md WARNING 4 regression: `default=str`
        stringifies an arbitrary object via `repr()`, so a secret-bearing
        object under a benign key used to reach persisted logs verbatim --
        on the CLI's own redaction path. Post-fix (the shared
        `redacting_json_default`), only a type-name placeholder is logged."""
        from fsm_llm.logging import logger

        class Creds:
            """Stand-in for a caller's custom object whose repr leaks a
            secret field value -- exactly what `redacting_json_default` must not
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
        assert any("<redacted:Creds>" in str(r["message"]) for r in records)


class TestRunnerPrintsToTerminal:
    """F-HM-01 / F-AF-01. `setup_file_logging()` swaps loguru's default sink
    for one that skips `fsm_llm` records, and `runner` records are `fsm_llm`
    records, so a reply written only through `logger.info` never reached the
    terminal. The CLI prints its product (reply, hint, data dump) to stdout
    and the failure to stderr."""

    def _run(self, api, *, patch_logging=True, inputs=("hi",)):
        env = {"LLM_MODEL": "test-model"}
        feed = iter(inputs)
        with patch.dict(os.environ, env, clear=True):
            with patch("fsm_llm.runner.load_env_file"):
                with patch("fsm_llm.runner.API.from_file", return_value=api):
                    with patch("builtins.input", side_effect=lambda _p="": next(feed)):
                        from fsm_llm import runner

                        if patch_logging:
                            with patch("fsm_llm.runner.setup_file_logging"):
                                return runner.main("/tmp/t.json", 5, 1000)
                        return runner.main("/tmp/t.json", 5, 1000)

    def _api(self, greeting="HELLO", reply="REPLY", data=None):
        api = MagicMock()
        api.start_conversation.return_value = ("conv-1", greeting)
        api.has_conversation_ended.side_effect = [False, True]
        api.converse.return_value = reply
        api.get_data.return_value = data if data is not None else {}
        return api

    def test_greeting_reply_and_hint_on_stdout(self, capsys):
        assert self._run(self._api()) == 0
        out = capsys.readouterr().out
        assert "System: HELLO" in out
        assert "System: REPLY" in out
        assert "Type 'exit' to end the conversation." in out

    def test_empty_reply_prints_no_system_line(self, capsys):
        assert self._run(self._api(greeting="", reply="")) == 0
        assert "System:" not in capsys.readouterr().out

    def test_final_data_dump_is_redacted_on_stdout(self, capsys):
        api = self._api(data={"name": "Ada", "api_key": "sk-live-abcdef0123456789"})
        assert self._run(api) == 0
        out = capsys.readouterr().out
        assert "Data:" in out
        assert '"name": "Ada"' in out
        assert "sk-live-abcdef0123456789" not in out
        assert "<redacted>" in out

    def test_failure_text_on_stderr_and_exit_1(self, capsys):
        api = self._api()
        api.converse.side_effect = RuntimeError("provider exploded")
        assert self._run(api) == 1
        assert "provider exploded" in capsys.readouterr().err

    def test_first_call_failure_is_an_error_line_and_exit_1_not_a_traceback(
        self, capsys
    ):
        """Live finding D3: the failure of the first model call
        (`start_conversation`) used to escape as a raw traceback while the same
        failure on a later turn printed `Error: ...` and exited 1."""
        api = self._api()
        api.start_conversation.side_effect = RuntimeError("model not found")
        assert self._run(api) == 1
        captured = capsys.readouterr()
        assert captured.err.strip() == "Error: model not found"
        assert "Traceback" not in captured.err
        assert captured.out.strip() == "Type 'exit' to end the conversation."
        api.converse.assert_not_called()
        api.end_conversation.assert_not_called()  # no conversation was started

    def test_first_call_failure_and_later_failure_share_one_report(self, capsys):
        first = self._api()
        first.start_conversation.side_effect = RuntimeError("boom")
        later = self._api()
        later.converse.side_effect = RuntimeError("boom")
        assert self._run(first) == self._run(later) == 1
        errs = capsys.readouterr().err.splitlines()
        assert errs == ["Error: boom", "Error: boom"]

    def test_ctrl_c_on_the_first_call_still_propagates_to_main_cli(self):
        """Exit 130 is `__main__`'s job (it maps KeyboardInterrupt); the REPL
        must not swallow it into a failure."""
        api = self._api()
        api.start_conversation.side_effect = KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            self._run(api)

    def test_reply_reaches_terminal_with_real_file_logging(
        self, capsys, tmp_path, monkeypatch
    ):
        """The real `setup_file_logging` (handler swap) is exercised: the
        reply must still be on stdout."""
        from fsm_llm.logging import reset_handlers

        monkeypatch.chdir(tmp_path)
        reset_handlers()
        try:
            assert self._run(self._api(), patch_logging=False) == 0
        finally:
            reset_handlers()
        assert "System: REPLY" in capsys.readouterr().out


class TestRunnerSurvivesUnencodableOutput:
    """A model reply the terminal cannot encode (a lone surrogate on UTF-8, an
    emoji on cp1252 or ascii) must not end the session with exit 1: the runner
    writes it through `cli_support.escape_unencodable_output` (backslashreplace),
    as the OKF and skills CLIs do."""

    _LONE = json.loads('"partial emoji \\ud83d"')

    def _streams(self, monkeypatch, encoding):
        import io
        import sys

        out = io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors="strict")
        err = io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors="strict")
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)
        return out, err

    @staticmethod
    def _text(stream):
        stream.flush()
        return stream.buffer.getvalue().decode(stream.encoding)

    def _run(self, api, inputs):
        helper = TestRunnerPrintsToTerminal()
        return helper._run(api, inputs=inputs)

    def _api(self, greeting, replies, ended):
        api = MagicMock()
        api.start_conversation.return_value = ("conv-1", greeting)
        api.has_conversation_ended.side_effect = ended
        api.converse.side_effect = replies
        api.get_data.return_value = {}
        return api

    def test_lone_surrogate_reply_keeps_session_going(self, monkeypatch):
        out, err = self._streams(monkeypatch, "utf-8")
        api = self._api("HELLO", [self._LONE, "SECOND"], [False, False, True])
        assert self._run(api, ("one", "two")) == 0
        assert api.converse.call_count == 2
        text = self._text(out)
        assert "partial emoji \\ud83d" in text
        assert "System: SECOND" in text
        assert "Error:" not in self._text(err)

    def test_lone_surrogate_greeting_keeps_session_going(self, monkeypatch):
        out, _ = self._streams(monkeypatch, "utf-8")
        api = self._api(self._LONE, ["REPLY"], [False, True])
        assert self._run(api, ("hi",)) == 0
        text = self._text(out)
        assert "partial emoji \\ud83d" in text
        assert "System: REPLY" in text

    @pytest.mark.parametrize("encoding", ["cp1252", "ascii"])
    def test_non_encodable_char_survives_on_narrow_stdout(self, monkeypatch, encoding):
        out, err = self._streams(monkeypatch, encoding)
        api = self._api("café ☃", ["snow ☃ done"], [False, True])
        assert self._run(api, ("hi",)) == 0
        text = self._text(out)
        assert "\\u2603" in text
        assert "System: snow" in text
        assert "Error:" not in self._text(err)

    def test_greeting_failure_still_ends_conversation(self, monkeypatch):
        """The greeting print sits inside the try/finally: a failure there
        still ends the conversation."""
        self._streams(monkeypatch, "utf-8")
        api = self._api("HELLO", ["x"], [True])

        def _fail_on_reply(*args, **kwargs):
            if str(args[0]).startswith("System:"):
                raise OSError("closed")

        with patch("fsm_llm.runner.print", _fail_on_reply, create=True):
            with pytest.raises(OSError):
                self._run(api, ())
        api.end_conversation.assert_called_once_with("conv-1")

    def test_error_line_survives_unencodable_message(self, monkeypatch):
        _out, err = self._streams(monkeypatch, "ascii")
        api = self._api("HELLO", [RuntimeError("bad ☃")], [False, True])
        assert self._run(api, ("hi",)) == 1
        assert "bad \\u2603" in self._text(err)


def _aliased_tower(levels: int, *, as_list: bool = False) -> dict:
    """A dict whose every level holds the SAME child three times: 3**levels
    paths over only ``levels + 1`` distinct containers."""
    node: Any = {"leaf": "x", "password": "buried"}
    for _ in range(levels):
        node = [node, node, node] if as_list else {"a": node, "b": node, "c": node}
    return {"root": node}


def _redact_in_thread(payload: Any, timing_factor: float) -> tuple[bool, float, Any]:
    """Run ``redact_context`` in a daemon thread so a super-linear walk FAILS
    the test at the join timeout (scaled by *timing_factor*) instead of hanging
    the suite."""

    from fsm_llm.core.redaction import redact_context

    box: dict[str, Any] = {}

    def _work() -> None:
        with cpu_timer(thread=True) as timer:
            box["result"] = redact_context(payload)
        box["elapsed"] = timer.elapsed

    worker = threading.Thread(target=_work, daemon=True)
    worker.start()
    worker.join(_JOIN_TIMEOUT_S * timing_factor)
    return worker.is_alive(), box.get("elapsed", float("inf")), box.get("result")


class TestStep13RunnerRedaction:
    """Step 13 (D-016): ``redact_context`` had no cycle guard and no
    memo. 3-way aliasing at depth 16 is 3**16 paths (more than 20 s), and a
    self-cycle recursed to the depth bound on every path."""

    @pytest.mark.parametrize("as_list", [False, True])
    def test_three_way_aliased_depth_16_finishes_under_one_second(
        self, as_list, timing_factor
    ):
        from fsm_llm.core.redaction import REDACTED

        payload = _aliased_tower(15, as_list=as_list)
        still_running, elapsed, result = _redact_in_thread(payload, timing_factor)

        assert not still_running, "redaction did not finish (super-linear walk)"
        assert elapsed < 1.0 * timing_factor
        # Do NOT json.dumps the result here: it re-expands all 3**15 paths.
        # Descend one path to the innermost level instead.
        node = result["root"]
        for _ in range(15):
            children = node if as_list else [node["a"], node["b"], node["c"]]
            assert children[0] is children[1] is children[2]  # shared result
            node = children[0]
        assert node == {"leaf": "x", "password": REDACTED}

    def test_aliased_output_equals_unaliased_output(self):
        from fsm_llm.core.redaction import redact_context

        payload = _aliased_tower(6)
        unaliased = json.loads(json.dumps(payload))
        assert redact_context(payload) == redact_context(unaliased)

    def test_shared_subtree_at_different_depths_truncates_per_depth(self):
        """The memo is keyed by (id, depth): a subtree reached at depth 1 and
        at depth 12 must be cut by the bound differently, exactly as its
        un-aliased copies are."""
        from fsm_llm.core.redaction import REDACTED, redact_context

        chain: dict = {"password": "deep-secret", "keep": "v"}
        for _ in range(8):
            chain = {"n": chain}
        wrapper: dict = {"shallow": chain}
        cursor = wrapper
        for _ in range(10):
            cursor["down"] = {}
            cursor = cursor["down"]
        cursor["deep"] = chain

        result = redact_context(wrapper)
        assert result == redact_context(json.loads(json.dumps(wrapper)))
        rendered = json.dumps(result)
        assert "deep-secret" not in rendered
        assert '"keep": "v"' in rendered  # shallow copy walked in full
        assert REDACTED in rendered

    def test_self_cycle_terminates_with_the_placeholder(self, timing_factor):
        from fsm_llm.core.redaction import CYCLE_PLACEHOLDER, REDACTED

        payload: dict = {"name": "Alice", "password": "p"}
        payload["self"] = payload
        still_running, _, result = _redact_in_thread(payload, timing_factor)

        assert not still_running
        assert result == {
            "name": "Alice",
            "password": REDACTED,
            "self": CYCLE_PLACEHOLDER,
        }
        json.dumps(result)  # no "Circular reference detected"

    def test_list_self_cycle_is_replaced_not_dropped(self):
        from fsm_llm.core.redaction import CYCLE_PLACEHOLDER, redact_context

        items: list = ["a"]
        items.append(items)
        assert redact_context({"items": items}) == {"items": ["a", CYCLE_PLACEHOLDER]}

    def test_cycle_dependent_subtree_is_not_memoised(self):
        """``shared`` is reached twice at depth 2: first on the path
        root -> a -> shared, where its ``back`` edge closes a cycle to ``a``;
        then on root -> w -> shared, where ``a`` is NOT active and must be
        walked. Reusing the first result would print the wrong placeholder."""
        from fsm_llm.core.redaction import (
            CYCLE_PLACEHOLDER,
            REDACTED,
            redact_context,
        )

        a: dict = {}
        shared: dict = {"back": a, "password": "s"}
        a["s"] = shared
        result = redact_context({"a": a, "w": {"s2": shared}})

        assert result["a"] == {"s": {"back": CYCLE_PLACEHOLDER, "password": REDACTED}}
        assert result["w"]["s2"] == {
            "back": {"s": CYCLE_PLACEHOLDER},
            "password": REDACTED,
        }

    def test_acyclic_sibling_of_a_cycle_is_still_memoised(self, timing_factor):
        """A cycle elsewhere must not switch the memo off for an unrelated
        aliased subtree (the D-052 lesson): the tower stays linear."""
        payload = _aliased_tower(15)
        payload["self"] = payload
        still_running, elapsed, _ = _redact_in_thread(payload, timing_factor)

        assert not still_running
        assert elapsed < 1.0 * timing_factor

    # -- lazy debug dumps ---------------------------------------------------

    def _run_main(self, per_turn_data, final_data):
        import os

        mock_api = MagicMock()
        mock_api.start_conversation.return_value = ("conv-1", "Hello!")
        mock_api.has_conversation_ended.side_effect = [False, True]
        mock_api.converse.return_value = "Response"
        mock_api.get_data.side_effect = [per_turn_data, final_data]

        with patch.dict(os.environ, {"LLM_MODEL": "test-model"}, clear=True):
            with patch("fsm_llm.runner.load_env_file"):
                with patch("fsm_llm.runner.API.from_file", return_value=mock_api):
                    with patch("fsm_llm.runner.setup_file_logging"):
                        with patch("builtins.input", return_value="hi"):
                            from fsm_llm.runner import main

                            return main("/tmp/t.json", 5, 1000)

    def test_no_redaction_runs_when_logging_is_disabled(self):
        import fsm_llm.runner as runner_module
        from fsm_llm.logging import logger

        logger.disable("fsm_llm")
        with (
            patch.object(logger, "enable"),
            patch.object(
                runner_module, "redact_context", wraps=runner_module.redact_context
            ) as spy,
        ):
            assert self._run_main({"turn": 1}, {"turn": 1}) == 0
        # The per-turn debug dump stays lazy (no redaction while logging is
        # disabled). The final dump is the CLI's printed product (F-HM-01), so
        # it is redacted once for the terminal whatever the log level is.
        assert spy.call_count == 1

    def test_debug_dump_redacts_when_debug_is_enabled(self):
        """Vacuity guard: with a DEBUG sink the dumps still run, redacted."""
        import fsm_llm.runner as runner_module
        from fsm_llm.logging import logger

        records: list = []
        # setup_file_logging (patched here) is what enables the library loggers
        logger.enable("fsm_llm")
        sink_id = logger.add(lambda m: records.append(m.record), level="DEBUG")
        try:
            with patch.object(
                runner_module, "redact_context", wraps=runner_module.redact_context
            ) as spy:
                assert (
                    self._run_main(
                        {"turn": 1, "password": "TURN-SECRET"},
                        {"password": "FINAL-SECRET"},
                    )
                    == 0
                )
        finally:
            logger.remove(sink_id)
            logger.disable("fsm_llm")

        assert spy.call_count == 2
        messages = [str(r["message"]) for r in records]
        assert any(m.startswith("Context data: ") for m in messages)
        assert any(m.startswith("Data: \n") for m in messages)
        assert not any("SECRET" in m for m in messages)


class TestRunnerPrintsNothingForASilentReply:
    """The runner CLI logs no empty ``System:`` line (consumers review)."""

    def _system_lines(self, greeting: str, reply: str) -> list[str]:
        mock_api = MagicMock()
        mock_api.start_conversation.return_value = ("conv-1", greeting)
        mock_api.has_conversation_ended.side_effect = [False, True]
        mock_api.converse.return_value = reply
        mock_api.get_data.return_value = {}
        lines: list[str] = []
        with (
            patch.dict(os.environ, {"LLM_MODEL": "test-model"}, clear=True),
            patch("fsm_llm.runner.load_env_file"),
            patch("fsm_llm.runner.API.from_file", return_value=mock_api),
            patch("fsm_llm.runner.setup_file_logging"),
            patch("builtins.input", return_value="hello"),
            patch("fsm_llm.runner.logger") as logger,
        ):
            from fsm_llm.runner import main

            assert main("/tmp/t.json", 5, 1000) == 0
            lines = [
                call.args[0]
                for call in logger.info.call_args_list
                if call.args and str(call.args[0]).startswith("System:")
            ]
        return lines

    def test_silent_greeting_and_silent_reply_print_no_system_line(self):
        assert self._system_lines("", "") == []

    def test_spoken_replies_are_still_printed(self):
        assert self._system_lines("Hello!", "Sure.") == [
            "System: Hello!",
            "System: Sure.",
        ]
