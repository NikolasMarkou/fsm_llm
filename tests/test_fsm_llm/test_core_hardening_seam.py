"""Integration-seam tests for the core-hardening set.

Each test in this file pins a defect that was reproduced by probe before the
fix landed. They exercise public seams (``FileSessionStore.save``,
``setup_logging``) rather than private helpers.
"""

import concurrent.futures
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from fsm_llm.api import API
from fsm_llm.definitions import FieldExtractionConfig, FSMError, LLMResponseError
from fsm_llm.handlers import HandlerExecutionError, HandlerTiming, create_handler
from fsm_llm.logging import logger
from fsm_llm.prompts import (
    DataExtractionPromptBuilder,
    DataExtractionPromptConfig,
    FieldExtractionPromptBuilder,
    FieldExtractionPromptConfig,
    ResponseGenerationPromptBuilder,
    ResponsePromptConfig,
)
from fsm_llm.session import FileSessionStore, SessionState
from tests.conftest import MockLLM2Interface

# Deliberate reuse: ``test_pipeline_handler_contract`` already owns the minimal
# two-state FSM, the ``API.from_definition`` wiring, and the "always raises at
# this timing" handler these tests need. Copying a 60-line FSM dict or a second
# exploding handler here would be a second source of truth for the same fixture.
from tests.test_fsm_llm.test_pipeline_handler_contract import (
    _ExplodingHandler,
    _fsm_definition,
    _make_api,
)

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

    Built entirely through the public fluent chain: ``.critical()`` sets the flag
    that ``execute_handlers`` reads via ``getattr(handler, "critical", False)``.
    This used to require reaching into the built instance post-hoc (F-14).
    """
    return (
        create_handler(name).at(HandlerTiming.PRE_PROCESSING).critical().do(execution)
    )


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


class TestFieldExtractionContextIsBounded:
    """The THIRD context-emitting site, and the one that most needed the cap.

    ``FieldExtractionPromptBuilder.build_field_extraction_prompt`` assembles its
    own ``Already extracted: {...}`` section instead of calling
    ``_build_enhanced_context_section``, so it did not inherit the cap the two
    builders above got. Probe evidence pre-fix: 500 of 500 keys landed, against
    a configured cap of 200 -- and this path runs once PER FIELD, unlike the
    once-per-turn sites above.
    """

    _CAP = 10
    _TOTAL = 500

    def _build(self, context_data, **cfg):
        builder = FieldExtractionPromptBuilder(FieldExtractionPromptConfig(**cfg))
        return builder.build_field_extraction_prompt(
            instance=_make_prompt_instance(),
            user_message="I want to go to Paris",
            field_config=FieldExtractionConfig(
                field_name="destination",
                field_type="str",
                extraction_instructions="the city the user named",
            ),
            dynamic_context=context_data,
        )

    @staticmethod
    def _already_extracted(prompt: str) -> str:
        """The builder's own context section, isolated.

        Asserting against the whole prompt would be ambiguous -- the field
        instructions echo key-like text.
        """
        line = next(
            ln for ln in prompt.splitlines() if ln.startswith("Already extracted: ")
        )
        return line

    def test_field_extraction_context_is_capped(self):
        big = {f"field_{i:04d}": f"value_{i:04d}" for i in range(self._TOTAL)}

        section = self._already_extracted(self._build(big, max_context_keys=self._CAP))

        assert section.count("field_") <= self._CAP, (
            f"field-extraction context carried {section.count('field_')} keys "
            f"despite a configured cap of {self._CAP}"
        )

    def test_retained_keys_carry_their_real_values(self):
        """Guards against 'fix by suppression'.

        Emitting an empty section would satisfy the cap assertion above while
        being strictly worse: the per-field extractor would lose the fields
        already collected and start re-asking for them.
        """
        big = {f"field_{i:04d}": f"value_{i:04d}" for i in range(self._TOTAL)}

        section = self._already_extracted(self._build(big, max_context_keys=self._CAP))
        retained = re.findall(r'"(field_\d{4})": "(value_\d{4})"', section)

        assert len(retained) == self._CAP, (
            f"expected exactly {self._CAP} fully-formed key/value pairs, "
            f"found {len(retained)}"
        )
        for key, value in retained:
            assert value == key.replace("field_", "value_"), (
                f"{key} was retained but its value was mangled: {value}"
            )

    def test_context_smaller_than_the_cap_is_untouched(self):
        small = {f"field_{i:04d}": f"value_{i:04d}" for i in range(5)}

        section = self._already_extracted(
            self._build(small, max_context_keys=self._CAP)
        )

        for i in range(5):
            assert f'"field_{i:04d}": "value_{i:04d}"' in section


# ══════════════════════════════════════════════════════════════
# F-07 — critical END_CONVERSATION handler swallowed in
#        start_conversation's two cleanup arms
# ══════════════════════════════════════════════════════════════


def _failing_start(failure, handlers, error_mode="continue"):
    """Drive ``start_conversation`` to failure with ``handlers`` registered.

    ``failure`` is raised from the initial-response LLM call, which is what
    selects the cleanup arm under test: an ``FSMError`` subclass takes the
    ``except FSMError`` arm, anything else takes the general ``except Exception``
    arm.

    Returns:
        ``(manager, raised)`` -- the ``FSMManager`` (so the caller can assert on
        ``instances`` / ``_conversation_locks``) and the exception that reached
        the caller, or ``None`` if ``start_conversation`` returned.
    """
    llm = MockLLM2Interface()

    def _explode(_request):
        raise failure

    llm.generate_response = _explode
    api = API.from_definition(
        _fsm_definition(),
        llm_interface=llm,
        handler_error_mode=error_mode,
        handlers=list(handlers),
    )

    raised = None
    try:
        api.start_conversation()
    except BaseException as exc:  # the exception IS the assertion subject
        raised = exc
    return api.fsm_manager, raised


def _critical_end_handler():
    """Reuse: ``_ExplodingHandler`` already is 'raises at this timing'."""
    return _ExplodingHandler(
        "critical_end", HandlerTiming.END_CONVERSATION, critical=True
    )


class TestCriticalEndHandlerDuringFailedStart:
    """A ``critical=True`` END_CONVERSATION handler must not be swallowed.

    ``start_conversation`` fires a best-effort END_CONVERSATION pass from both
    of its failure arms. Both used to log the handler's exception at WARNING and
    discard it -- the only one of the eight handler firing sites where the
    "a critical handler always raises" contract did not hold.

    Every test asserts three properties, not just "it raised": the handler's
    exception SURFACES, the original failure is still diagnosable via the
    exception chain, and both resource dicts are still emptied. Asserting only
    the raise would be satisfied by the strictly worse outcome of losing the
    original diagnosis or leaking the conversation lock.
    """

    _ORIGINAL_FSM_ERROR = "simulated LLM outage on initial response"
    _ORIGINAL_PLAIN = "simulated non-FSM outage on initial response"

    def test_fsm_error_arm_surfaces_the_critical_handler_failure(self):
        """The arm probed as SUSPECTED-by-shape; CONFIRMED identical to its twin."""
        manager, raised = _failing_start(
            LLMResponseError(self._ORIGINAL_FSM_ERROR), [_critical_end_handler()]
        )

        assert isinstance(raised, HandlerExecutionError), (
            "the critical handler's failure must reach the caller, got "
            f"{type(raised).__name__}: {raised}"
        )
        assert "critical_end" in str(raised)
        assert isinstance(raised.__cause__, LLMResponseError), (
            "the original failure must stay diagnosable through the chain, got "
            f"{type(raised.__cause__).__name__}"
        )
        assert self._ORIGINAL_FSM_ERROR in str(raised.__cause__)
        assert manager.instances == {}
        assert manager._conversation_locks == {}

    def test_general_exception_arm_surfaces_the_critical_handler_failure(self):
        manager, raised = _failing_start(
            RuntimeError(self._ORIGINAL_PLAIN), [_critical_end_handler()]
        )

        assert isinstance(raised, HandlerExecutionError), (
            "the critical handler's failure must reach the caller, got "
            f"{type(raised).__name__}: {raised}"
        )
        assert "critical_end" in str(raised)
        assert isinstance(raised.__cause__, RuntimeError), (
            "the original failure must stay diagnosable through the chain, got "
            f"{type(raised.__cause__).__name__}"
        )
        assert self._ORIGINAL_PLAIN in str(raised.__cause__)
        assert manager.instances == {}
        assert manager._conversation_locks == {}

    @pytest.mark.parametrize(
        ("failure", "expected"),
        [
            (LLMResponseError(_ORIGINAL_FSM_ERROR), LLMResponseError),
            (RuntimeError(_ORIGINAL_PLAIN), RuntimeError),
        ],
        ids=["fsm_error_arm", "general_exception_arm"],
    )
    def test_non_critical_cleanup_failure_is_still_swallowed(self, failure, expected):
        """Over-correction guard: only failures the handler system propagates win.

        Under ``handler_error_mode="continue"`` a non-critical handler's failure
        never escapes ``execute_handlers`` at all, so the caller must still see
        the ORIGINAL failure. If this test starts seeing a
        ``HandlerExecutionError``, the fix stopped honoring the swallow decision
        that already happened inside the handler system.
        """
        manager, raised = _failing_start(
            failure,
            [_ExplodingHandler("noisy_end", HandlerTiming.END_CONVERSATION)],
        )

        assert not isinstance(raised, HandlerExecutionError), (
            f"a non-critical cleanup failure must not surface, got {raised}"
        )
        # The general-Exception arm re-wraps into FSMError; the FSMError arm
        # re-raises bare. Either way the ORIGINAL is what the caller can see.
        original = raised if isinstance(raised, expected) else raised.__cause__
        assert isinstance(original, expected)
        assert manager.instances == {}
        assert manager._conversation_locks == {}

    def test_keyboard_interrupt_propagates_bare_and_still_frees_resources(self):
        """HARD invariant: never wrapped -- and the lock must not leak either."""

        class _InterruptingHandler(_ExplodingHandler):
            def execute(self, context):
                raise KeyboardInterrupt("operator interrupt during cleanup")

        manager, raised = _failing_start(
            RuntimeError(self._ORIGINAL_PLAIN),
            [
                _InterruptingHandler(
                    "interrupt_end", HandlerTiming.END_CONVERSATION, critical=True
                )
            ],
        )

        assert type(raised) is KeyboardInterrupt, (
            f"KeyboardInterrupt must propagate bare, got {type(raised).__name__}"
        )
        assert manager.instances == {}
        assert manager._conversation_locks == {}


# ══════════════════════════════════════════════════════════════
# C1 — read accessors must snapshot under conv_lock, not race the
#      locked write path's in-place mutation of context.data
# ══════════════════════════════════════════════════════════════


class _SlowMockLLM2(MockLLM2Interface):
    """Mock that sleeps mid-call to widen the in-flight write window.

    A converse turn holds the conversation's ``conv_lock`` for its whole
    duration; sleeping inside the LLM calls keeps that window open ~10ms so a
    concurrent reader (pre-fix: unlocked) has a wide window to iterate
    ``instance.context.data`` WHILE the write path resizes it.
    """

    def __init__(self, *args, delay: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self._delay = delay

    def generate_response(self, request):
        time.sleep(self._delay)
        return super().generate_response(request)

    def extract_field(self, request):
        time.sleep(self._delay)
        return super().extract_field(request)


def _self_loop_fsm() -> dict:
    """A self-looping state that never terminates in practice.

    The writer can call ``converse`` indefinitely because ``loop`` always takes
    its unconditional self-transition (so ``has_conversation_ended`` stays
    False and no turn is rejected for being terminal). The ``done`` terminal
    state exists only to satisfy the FSMDefinition validator ("must have at
    least one terminal state"); its transition is gated on a ``finished`` key
    the test never sets, so it is unreachable here.
    """
    return {
        "name": "read_concurrency_fsm",
        "description": "self-looping state; unreachable terminal for the validator",
        "version": "4.1",
        "initial_state": "loop",
        "states": {
            "loop": {
                "id": "loop",
                "description": "Self-looping state",
                "purpose": "Stay so the writer can converse indefinitely",
                "response_instructions": "Reply politely",
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "Never fires — 'finished' key is never set",
                        "priority": 200,
                        "conditions": [
                            {
                                "description": "finished flag present",
                                "requires_context_keys": ["finished"],
                            }
                        ],
                    },
                    {
                        "target_state": "loop",
                        "description": "Always stay in loop",
                        "priority": 100,
                    },
                ],
            },
            "done": {
                "id": "done",
                "description": "Unreachable terminal state (validator only)",
                "purpose": "Never entered",
                "transitions": [],
            },
        },
    }


class TestReadConcurrency:
    """C1 — read accessors must snapshot under ``conv_lock``.

    Pre-fix, ``get_data`` / ``get_conversation_history`` read
    ``instance.context.data`` / ``.exchanges`` with NO per-conversation lock
    while an in-flight ``process_message`` on the SAME conversation resizes
    ``context.data`` (a PRE_PROCESSING handler delta merged key-by-key via
    ``merge_delta`` + the extraction commit). A reader iterating the dict then
    raises ``RuntimeError: dictionary changed size during iteration``.

    This test was authored RED-first: run against the unfixed ``fsm.py`` it
    reliably surfaces that ``RuntimeError`` in the reader threads. Against the
    fixed code (reads snapshot under ``conv_lock``) it must pass with zero
    ``RuntimeError`` and structurally intact reads.
    """

    _READERS = 8
    _WRITER_TURNS = 60
    _SEED_KEYS = 300
    _GROWTH_PER_TURN = 40
    _MIN_READS = 500
    _DEADLINE_S = 60.0

    def _make_api(self):
        api = API.from_definition(
            _self_loop_fsm(),
            llm_interface=_SlowMockLLM2(delay=0.01),
        )
        # PRE_PROCESSING handler that GROWS context.data every turn. Each unique
        # key is merged one-by-one via merge_delta (pipeline.py), so a single
        # turn resizes the dict _GROWTH_PER_TURN times, spread across its ~10ms
        # window — maximizing the chance a resize lands while a reader is
        # mid-iteration over the (already large, seeded) dict.
        grow = (
            create_handler("grow")
            .at(HandlerTiming.PRE_PROCESSING)
            .do(
                lambda ctx: {
                    f"g_{uuid.uuid4().hex}": "v" for _ in range(self._GROWTH_PER_TURN)
                }
            )
        )
        api.register_handler(grow)
        seed = {f"seed_{i:04d}": f"value_{i:04d}" for i in range(self._SEED_KEYS)}
        conv_id, _ = api.start_conversation(seed)
        return api, conv_id

    def test_concurrent_reads_never_tear_or_crash(self):
        api, conv_id = self._make_api()
        errors: list[tuple[str, BaseException]] = []
        errors_lock = threading.Lock()
        stop = threading.Event()
        # Live read-cycle tally shared with the writer so the concurrency
        # window is ADAPTIVE: on a slow/oversubscribed runner a fixed
        # _WRITER_TURNS can complete before the readers reach the _MIN_READS
        # coverage floor, failing the final assert as "window too narrow"
        # with zero actual tears (measured: CI 3.11 hit 349/500). The writer
        # keeps writing until the floor is met (bounded by _DEADLINE_S), so
        # the floor asserts coverage, not runner speed.
        reads_tally = [0]
        tally_lock = threading.Lock()

        def _record(who: str, exc: BaseException) -> None:
            with errors_lock:
                errors.append((who, exc))

        def writer() -> int:
            turns = 0
            deadline = time.monotonic() + self._DEADLINE_S
            try:
                while (
                    turns < self._WRITER_TURNS or reads_tally[0] < self._MIN_READS
                ) and time.monotonic() < deadline:
                    try:
                        api.converse("hello", conv_id)
                        turns += 1
                    except FSMError as exc:
                        # Documented, bounded trade-off (plan Failure Modes): a
                        # writer's non-blocking conv_lock acquire can collide
                        # with a reader's sub-ms snapshot window and be rejected.
                        # This is a retryable spurious rejection, NOT a defect.
                        if "already being processed" in str(exc):
                            continue
                        raise
            except BaseException as exc:  # a writer crash is its own signal
                _record("writer", exc)
            finally:
                stop.set()
            return 0

        def reader() -> int:
            local = 0
            while not stop.is_set():
                try:
                    data = api.get_data(conv_id)
                    # Structural validity: a real mapping whose never-mutated
                    # seed keys survived intact (a torn dict rebuild would raise
                    # RuntimeError rather than silently corrupt — assert anyway).
                    assert isinstance(data, dict)
                    assert data.get("seed_0000") == "value_0000"
                    hist = api.get_conversation_history(conv_id)
                    assert isinstance(hist, list)
                    local += 1
                    with tally_lock:
                        reads_tally[0] += 1
                except BaseException as exc:
                    _record("reader", exc)
                    break
            return local

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self._READERS + 1
            ) as pool:
                reader_futures = [pool.submit(reader) for _ in range(self._READERS)]
                writer_future = pool.submit(writer)
                concurrent.futures.wait([*reader_futures, writer_future])

            reads_done = sum(f.result() for f in reader_futures)

            # RED signal first: the torn-read `RuntimeError: dictionary changed
            # size during iteration` is the defect this test pins. The accessor
            # decorator re-wraps it into an FSMError, so detect it through the
            # message / cause chain, not just isinstance. Assert on it BEFORE the
            # coverage check so an unfixed run reports the race, not a "window
            # too narrow" red herring.
            def _is_torn_read(exc: BaseException) -> bool:
                cur: BaseException | None = exc
                while cur is not None:
                    if isinstance(cur, RuntimeError):
                        return True
                    if "changed size during iteration" in str(cur):
                        return True
                    cur = cur.__cause__ or cur.__context__
                return False

            torn = [exc for _, exc in errors if _is_torn_read(exc)]
            assert not torn, (
                f"{len(torn)} torn-read error(s) under concurrent read/write "
                f"(C1 race): {torn[0]!r}"
            )
            assert not errors, (
                "unexpected error(s) during concurrent read/write: "
                f"{[(who, repr(exc)) for who, exc in errors][:5]}"
            )
            # Beat the non-determinism: the readers must actually have cycled
            # many times against in-flight writes, not exited early. The
            # writer holds the window open until this floor is met (or the
            # _DEADLINE_S wall-clock bound trips), so a miss here means a
            # genuinely pathological environment, not an unlucky fast writer.
            assert reads_done >= self._MIN_READS, (
                f"readers only completed {reads_done} cycles within "
                f"{self._DEADLINE_S}s — the window was too narrow to prove "
                "anything; widen READERS/WRITER_TURNS/_DEADLINE_S"
            )
        finally:
            api.close()


# ══════════════════════════════════════════════════════════════
# L11 — an instance present with NO per-conversation lock is a
#       broken create-together/remove-together invariant. Both the
#       read path (_read_under_lock) and update_conversation_context
#       must FAIL LOUD rather than silently proceed unlocked.
# ══════════════════════════════════════════════════════════════


class TestBrokenLockInvariantFailsLoud:
    """L11 — verify the fail-loud guard on the missing per-conversation lock.

    Both accessors look up ``_conversation_locks.get(cid)``; if an instance is
    present but its lock is ``None`` the create-together/remove-together
    invariant is broken and proceeding would do an UNLOCKED read/write that
    races the 2-pass write path. The guard raises ``FSMError`` instead. This
    test drives that artificial state directly: inject a live conversation,
    then delete its lock entry.
    """

    def _make_api_with_conv(self):
        api = API.from_definition(
            _self_loop_fsm(),
            llm_interface=MockLLM2Interface(),
        )
        conv_id, _ = api.start_conversation()
        return api, conv_id

    def test_read_accessor_raises_when_lock_missing(self):
        api, conv_id = self._make_api_with_conv()
        try:
            manager = api.fsm_manager
            # Instance stays; its lock is removed — the broken invariant.
            assert conv_id in manager.instances
            manager._conversation_locks.pop(conv_id, None)

            with pytest.raises(FSMError, match="broken invariant"):
                manager.get_conversation_data(conv_id)
        finally:
            # Restore a lock so end_conversation/close does not re-trip the guard.
            manager._conversation_locks[conv_id] = threading.RLock()
            api.close()

    def test_update_context_raises_when_lock_missing(self):
        api, conv_id = self._make_api_with_conv()
        try:
            manager = api.fsm_manager
            assert conv_id in manager.instances
            manager._conversation_locks.pop(conv_id, None)

            with pytest.raises(FSMError, match="broken invariant"):
                manager.update_conversation_context(conv_id, {"k": "v"})
        finally:
            manager._conversation_locks[conv_id] = threading.RLock()
            api.close()
