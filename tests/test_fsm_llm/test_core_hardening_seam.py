"""Integration-seam tests for the core-hardening set.

Each test in this file pins a defect that was reproduced by probe before the
fix landed. They exercise public seams (``FileSessionStore.save``,
``setup_logging``) rather than private helpers.
"""

import os
import re
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from fsm_llm.handlers import HandlerExecutionError, HandlerTiming, create_handler
from fsm_llm.logging import logger
from fsm_llm.prompts import (
    DataExtractionPromptBuilder,
    DataExtractionPromptConfig,
    ResponseGenerationPromptBuilder,
    ResponsePromptConfig,
)
from fsm_llm.session import FileSessionStore, SessionState

# Deliberate reuse: ``test_pipeline_handler_contract`` already owns the minimal
# two-state FSM plus the ``API.from_definition`` wiring these tests need. Copying
# a 60-line FSM dict here would be a second source of truth for the same fixture.
from tests.test_fsm_llm.test_pipeline_handler_contract import _make_api

# Deliberate reuse: test_prompts_unit.py already owns the minimal State /
# FSMInstance / FSMDefinition factories these tests need. Re-declaring them
# here would be a second source of truth for the same fixture shape.
# Aliased because this module's own ``_make_state`` builds a SessionState --
# importing unaliased silently shadows one of the two.
from tests.test_fsm_llm.test_prompts_unit import (
    _make_fsm_definition as _make_prompt_fsm_definition,
)
from tests.test_fsm_llm.test_prompts_unit import (
    _make_instance as _make_prompt_instance,
)
from tests.test_fsm_llm.test_prompts_unit import (
    _make_state as _make_prompt_state,
)

# Capturing loguru output MUST go through a subprocess. ``contextlib.redirect_stdout``
# lies here: loguru binds the stream object at ``logger.add()`` time, so a redirect
# installed afterwards is simply not the stream the handler writes to.
_SENTINEL = "SENTINEL_LOG_LINE_12345"


def _run_logging_snippet(body: str) -> subprocess.CompletedProcess:
    script = (
        "from fsm_llm.logging import setup_logging, logger\n"
        # loguru installs a default stderr handler at import that this library
        # never removes. Drop it first, or every stderr count is off by one for
        # reasons unrelated to what these tests are pinning.
        "logger.remove()\n"
        f"{body}\n"
        f"logger.info({_SENTINEL!r})\n"
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )


def _make_state(conversation_id: str = "conv-1") -> SessionState:
    return SessionState(
        conversation_id=conversation_id,
        fsm_id="fsm-1",
        current_state="start",
        context_data={"name": "Ada"},
    )


class TestSessionTempFileCleanup:
    """B2 — ``FileSessionStore.save`` must not leak a temp file on ANY failure.

    Before the fix the write+rename was wrapped in ``except OSError:`` only, so
    a non-OSError (RuntimeError, MemoryError, a TypeError out of json.dumps)
    skipped the unlink and left a ``*.tmp`` file behind permanently.
    """

    def test_non_oserror_failure_leaves_no_temp_file(self, tmp_path):
        store = FileSessionStore(tmp_path)

        with patch(
            "os.replace", side_effect=RuntimeError("simulated non-OSError failure")
        ):
            with pytest.raises(RuntimeError, match="simulated non-OSError failure"):
                store.save("sess-1", _make_state())

        assert list(tmp_path.glob("*.tmp")) == [], (
            "save() leaked a temp file after a non-OSError failure"
        )

    def test_oserror_failure_leaves_no_temp_file(self, tmp_path):
        """The pre-existing OSError path must keep working."""
        store = FileSessionStore(tmp_path)

        with patch("os.replace", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                store.save("sess-1", _make_state())

        assert list(tmp_path.glob("*.tmp")) == []

    def test_successful_save_writes_file_and_leaves_no_temp(self, tmp_path):
        """Positive control: the fix must not be satisfiable by writing nothing."""
        store = FileSessionStore(tmp_path)
        store.save("sess-1", _make_state())

        assert list(tmp_path.glob("*.tmp")) == []
        assert (tmp_path / "sess-1.json").exists()

        loaded = store.load("sess-1")
        assert loaded is not None
        assert loaded.conversation_id == "conv-1"
        assert loaded.current_state == "start"
        assert loaded.context_data == {"name": "Ada"}

    def test_write_failure_leaves_no_temp_file(self, tmp_path):
        """A failure BEFORE the rename must also clean up."""
        store = FileSessionStore(tmp_path)

        with patch("json.dumps", side_effect=TypeError("unserializable")):
            with pytest.raises(TypeError, match="unserializable"):
                store.save("sess-1", _make_state())

        assert list(tmp_path.glob("*.tmp")) == []
        assert not os.path.exists(tmp_path / "sess-1.json")


class TestStreamSinkReentrancyGuard:
    """B5 — repeated ``setup_logging(sink=...)`` must not duplicate stream handlers.

    The ``sink="file"`` branch has always been guarded by
    ``_file_handler_initialized``; the stderr/stdout branch was not, so every
    call appended another handler and each log line was emitted once per call.
    """

    def test_two_stdout_setups_emit_the_line_once(self):
        result = _run_logging_snippet(
            'setup_logging(sink="stdout")\nsetup_logging(sink="stdout")'
        )
        assert result.stdout.count(_SENTINEL) == 1, (
            f"expected exactly 1 sentinel, stdout was:\n{result.stdout}"
        )

    def test_single_stdout_setup_still_emits_the_line_once(self):
        """Positive control: the guard must not suppress the FIRST registration."""
        result = _run_logging_snippet('setup_logging(sink="stdout")')
        assert result.stdout.count(_SENTINEL) == 1, (
            f"guard suppressed the first registration, stdout was:\n{result.stdout}"
        )

    def test_five_stdout_setups_emit_the_line_once(self):
        body = "\n".join(['setup_logging(sink="stdout")'] * 5)
        result = _run_logging_snippet(body)
        assert result.stdout.count(_SENTINEL) == 1

    def test_distinct_sinks_both_register(self):
        """stdout then stderr are DIFFERENT targets — both must register.

        The guard is keyed on the resolved (sink, format, context) triple, not on
        a single global boolean, precisely so a legitimately different second sink
        is not wrongly suppressed.
        """
        result = _run_logging_snippet(
            'setup_logging(sink="stdout")\nsetup_logging(sink="stderr")'
        )
        assert result.stdout.count(_SENTINEL) == 1
        assert result.stderr.count(_SENTINEL) == 1

    def test_distinct_formats_both_register(self):
        """Human then JSON on the same stream are different handlers."""
        result = _run_logging_snippet(
            'setup_logging(sink="stdout")\nsetup_logging(sink="stdout", format="json")'
        )
        assert result.stdout.count(_SENTINEL) == 2

    def test_repeat_setup_returns_the_idempotency_sentinel(self):
        """Mirrors the file sink's contract: a suppressed call returns -1."""
        result = _run_logging_snippet(
            'first = setup_logging(sink="stdout")\n'
            'second = setup_logging(sink="stdout")\n'
            "assert first > 0, first\n"
            "assert second == -1, second"
        )
        assert result.returncode == 0


# ══════════════════════════════════════════════════════════════
# D-1 — LambdaHandler must not wrap its own failures
# ══════════════════════════════════════════════════════════════


def _critical_lambda_handler(name, execution):
    """Build a ``create_handler()`` handler that fires at PRE_PROCESSING.

    ``HandlerBuilder`` has no ``.critical()`` step, so the flag is set on the
    built instance -- which is exactly what ``execute_handlers`` reads via
    ``getattr(handler, "critical", False)``.
    """
    handler = create_handler(name).at(HandlerTiming.PRE_PROCESSING).do(execution)
    handler.critical = True
    return handler


class TestLambdaHandlerSingleWrapSite:
    """``create_handler().do()`` failures must have the same shape as ``BaseHandler`` ones.

    Driven through ``API.converse``, never through ``execute_handlers`` directly:
    a critical-handler suite once passed while production was broken precisely
    because it called the inner method.
    """

    def test_raising_lambda_is_wrapped_exactly_once(self, mock_llm2_interface):
        api, conv_id = _make_api(
            mock_llm2_interface,
            [_critical_lambda_handler("boom_handler", lambda ctx: 1 / 0)],
        )

        with pytest.raises(HandlerExecutionError) as exc_info:
            api.converse("hello", conv_id)

        message = str(exc_info.value)
        assert message.count("Error in handler") == 1, (
            f"handler failure was wrapped more than once: {message}"
        )
        assert isinstance(exc_info.value.original_error, ZeroDivisionError), (
            "original_error must be the raw user exception, got "
            f"{type(exc_info.value.original_error).__name__}"
        )

    def test_non_dict_return_is_wrapped_exactly_once(self, mock_llm2_interface):
        api, conv_id = _make_api(
            mock_llm2_interface,
            [_critical_lambda_handler("bad_handler", lambda ctx: "not a dict")],
        )

        with pytest.raises(HandlerExecutionError) as exc_info:
            api.converse("hello", conv_id)

        message = str(exc_info.value)
        assert message.count("Error in handler") == 1, (
            f"non-dict result was wrapped more than once: {message}"
        )
        assert isinstance(exc_info.value.original_error, TypeError), (
            "original_error must be the raw TypeError, got "
            f"{type(exc_info.value.original_error).__name__}"
        )

    def test_one_failure_emits_one_error_record_from_handlers(
        self, mock_llm2_interface
    ):
        """``handlers.py`` used to log the SAME failure twice.

        Scoped to ``fsm_llm.handlers`` on purpose: ``pipeline.py`` also logs the
        escaping error at its own layer, and that record is not this step's
        business. Counting every ERROR record would make the assertion depend on
        an unrelated module.
        """
        api, conv_id = _make_api(
            mock_llm2_interface,
            [_critical_lambda_handler("noisy_handler", lambda ctx: 1 / 0)],
        )

        records = []
        sink_id = logger.add(
            lambda message: records.append(message.record), level="ERROR"
        )
        # The package calls ``logger.disable("fsm_llm")`` at import so it stays
        # silent for hosts that never opt in. Without this, the sink above
        # receives nothing and the test passes for the wrong reason.
        logger.enable("fsm_llm")
        try:
            with pytest.raises(HandlerExecutionError):
                api.converse("hello", conv_id)
        finally:
            logger.disable("fsm_llm")
            logger.remove(sink_id)

        from_handlers = [r for r in records if r["name"] == "fsm_llm.handlers"]
        assert len(from_handlers) == 1, (
            f"expected exactly 1 handlers.py error record for 1 failure, got "
            f"{len(from_handlers)}:\n"
            + "\n---\n".join(str(r["message"]) for r in from_handlers)
        )


class TestRunnerContextLoggingRedaction:
    """B4 — ``fsm-llm --fsm ...`` wrote raw context values to a DEBUG log file.

    Driven through ``runner.main`` with a real file sink rather than through a
    private helper: the defect was that the *default* CLI path (unconditional
    ``setup_file_logging`` + ``LOG_DEFAULT_LEVEL == "DEBUG"``) persisted secrets
    to disk, and only the whole path proves that.
    """

    _SECRET_API_KEY = "sk-live-EXTREMELY-SECRET-1234567890"
    _SECRET_PASSWORD = "hunter2-DO-NOT-LOG"

    def _run_cli_and_read_log(self, tmp_path, context: dict) -> str:
        mock_api = MagicMock()
        mock_api.start_conversation.return_value = ("conv-1", "Hello!")
        mock_api.has_conversation_ended.side_effect = [False, True]
        mock_api.converse.return_value = "Response"
        mock_api.get_data.return_value = context

        log_file = tmp_path / "cli.log"
        # The library disables its own logger at import; without enabling it the
        # sink below collects nothing and the assertions pass vacuously.
        logger.enable("fsm_llm")
        sink_id = logger.add(str(log_file), level="DEBUG")
        try:
            with patch.dict(os.environ, {"LLM_MODEL": "test-model"}, clear=True):
                with patch("fsm_llm.runner.dotenv.load_dotenv"):
                    with patch("fsm_llm.runner.API.from_file", return_value=mock_api):
                        with patch("fsm_llm.runner.setup_file_logging"):
                            with patch("builtins.input", return_value="hi"):
                                from fsm_llm.runner import main

                                assert main("/tmp/t.json", 5, 1000) == 0
        finally:
            logger.remove(sink_id)
            logger.disable("fsm_llm")
        return log_file.read_text()

    def test_secret_shaped_context_values_never_reach_the_log_file(self, tmp_path):
        contents = self._run_cli_and_read_log(
            tmp_path,
            {
                "user_name": "Alice",
                "api_key": self._SECRET_API_KEY,
                "password": self._SECRET_PASSWORD,
            },
        )

        assert self._SECRET_API_KEY not in contents, (
            "api_key value was written to the DEBUG log file verbatim"
        )
        assert self._SECRET_PASSWORD not in contents, (
            "password value was written to the DEBUG log file verbatim"
        )

    def test_benign_context_values_are_still_logged(self, tmp_path):
        """Guards against 'fix by suppression'.

        Logging nothing at all would satisfy the redaction assertion above while
        being strictly worse: the CLI's diagnostic value would be gone.
        """
        contents = self._run_cli_and_read_log(
            tmp_path,
            {
                "user_name": "Alice",
                "api_key": self._SECRET_API_KEY,
                "password": self._SECRET_PASSWORD,
            },
        )

        assert "Alice" in contents, (
            "benign context value disappeared — the fix suppressed the log line "
            "instead of filtering it"
        )


# ============================================================================
# B3 -- <current_context> must be bounded the way <conversation_history> is
# ============================================================================


def _context_block(prompt: str) -> str:
    """Return the CDATA payload of the prompt's <current_context> section.

    Asserting against the whole prompt would be ambiguous: state descriptions
    and extraction instructions can echo context key names, so a bare
    ``"field_007" in prompt`` does not prove the key survived the cap.
    """
    start = prompt.index("<current_context><![CDATA[")
    end = prompt.index("]]></current_context>", start)
    return prompt[start:end]


class TestCurrentContextIsBounded:
    """Pins B3: the context section serialized every key with no cap.

    Probe evidence pre-fix: 500 of 500 context keys reached the prompt
    (113,784 chars) while conversation history was correctly capped to 2
    exchanges by ``_manage_conversation_history``.
    """

    _CAP = 10
    _TOTAL = 1000

    def _big_context(self) -> dict:
        return {f"field_{i:04d}": f"value_{i:04d}" for i in range(self._TOTAL)}

    def _build_extraction(self, context_data, **cfg):
        builder = DataExtractionPromptBuilder(DataExtractionPromptConfig(**cfg))
        return builder.build_extraction_prompt(
            _make_prompt_instance(context_data=context_data),
            _make_prompt_state(),
            _make_prompt_fsm_definition(),
        )

    def _build_response(self, context_data, **cfg):
        builder = ResponseGenerationPromptBuilder(ResponsePromptConfig(**cfg))
        return builder.build_response_prompt(
            _make_prompt_instance(context_data=context_data),
            _make_prompt_state(response_instructions="Reply politely"),
            _make_prompt_fsm_definition(),
        )

    def test_extraction_prompt_context_is_capped(self):
        block = _context_block(
            self._build_extraction(self._big_context(), max_context_keys=self._CAP)
        )
        assert block.count("field_") <= self._CAP, (
            f"context section carried {block.count('field_')} keys despite a "
            f"configured cap of {self._CAP}"
        )

    def test_response_prompt_context_is_capped(self):
        """The fix belongs at the shared base method, so BOTH builders inherit it.

        A fix applied only inside DataExtractionPromptBuilder would pass the
        test above and fail this one.
        """
        block = _context_block(
            self._build_response(self._big_context(), max_context_keys=self._CAP)
        )
        assert block.count("field_") <= self._CAP, (
            f"response-prompt context section carried {block.count('field_')} "
            f"keys despite a configured cap of {self._CAP}"
        )

    def test_retained_keys_carry_their_real_values(self):
        """Guards against 'fix by suppression'.

        Emitting an empty <current_context> block would satisfy every cap
        assertion above while being strictly worse than the unbounded bug: the
        model would lose the conversation state entirely.
        """
        block = _context_block(
            self._build_extraction(self._big_context(), max_context_keys=self._CAP)
        )
        retained = re.findall(r'"(field_\d{4})": "(value_\d{4})"', block)

        assert len(retained) == self._CAP, (
            f"expected exactly {self._CAP} fully-formed key/value pairs, "
            f"found {len(retained)}"
        )
        for key, value in retained:
            assert value == key.replace("field_", "value_"), (
                f"{key} was retained but its value was mangled: {value}"
            )

    def test_context_smaller_than_the_cap_is_untouched(self):
        """The cap must bound a pathological tail, not reshape the happy path."""
        small = {f"field_{i:04d}": f"value_{i:04d}" for i in range(5)}

        default_prompt = self._build_extraction(small)
        capped_prompt = self._build_extraction(small, max_context_keys=self._CAP)

        assert capped_prompt == default_prompt, (
            "a context smaller than the cap produced different output"
        )
        block = _context_block(default_prompt)
        for i in range(5):
            assert f'"field_{i:04d}": "value_{i:04d}"' in block

    def test_default_cap_does_not_fire_for_realistic_context_sizes(self):
        """The default is deliberately high: the 95.3% eval baseline is not
        being re-run in this plan, so the default must not alter any prompt a
        real FSM produces."""
        realistic = {f"field_{i:04d}": f"value_{i:04d}" for i in range(50)}
        block = _context_block(self._build_extraction(realistic))
        assert block.count("field_") == 50
