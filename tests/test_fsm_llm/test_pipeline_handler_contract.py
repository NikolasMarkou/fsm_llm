"""Integration-seam regression tests for the handler failure contract (S1).

These tests deliberately drive every assertion through ``API.converse`` /
``MessagePipeline`` rather than calling ``HandlerSystem.execute_handlers``
directly.  That distinction is the whole point of this file: the pre-existing
``critical=True`` tests (``test_logging_unit.py::TestHandlerCriticalFlag``) call
``HandlerSystem`` directly and therefore passed while the production path --
``MessagePipeline.execute_handlers`` -- silently swallowed every critical
handler failure under the default ``handler_error_mode="continue"``.

Contract under test:
  1. Anything escaping ``HandlerSystem.execute_handlers`` propagates all the way
     out of ``API.converse``, regardless of ``handler_error_mode``.
  2. Context written by handlers that ran and *succeeded before* the failing
     handler at the same timing point survives the raise.
  3. A non-critical handler raising under ``"continue"`` is still swallowed
     (no behavior change for the common case).
"""

import pytest

from fsm_llm.api import API
from fsm_llm.definitions import FSMDefinition
from fsm_llm.handlers import BaseHandler, HandlerExecutionError, HandlerTiming

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class _RecordingHandler(BaseHandler):
    """Succeeds and writes ``result`` into the context at ``timing``."""

    def __init__(self, name, timing, result, priority=100, critical=False):
        super().__init__(name=name, priority=priority, critical=critical)
        self._timing = timing
        self._result = result

    def should_execute(
        self, timing, current_state, target_state, context, updated_keys=None
    ):
        return timing == self._timing

    def execute(self, context):
        return dict(self._result)


class _ExplodingHandler(BaseHandler):
    """Always raises at ``timing``."""

    def __init__(self, name, timing, priority=100, critical=False):
        super().__init__(name=name, priority=priority, critical=critical)
        self._timing = timing

    def should_execute(
        self, timing, current_state, target_state, context, updated_keys=None
    ):
        return timing == self._timing

    def execute(self, context):
        raise RuntimeError(f"boom in {self.name}")


def _fsm_definition():
    """Two-state FSM with no field/classification extractions (keeps the mock trivial)."""
    return FSMDefinition.model_validate(
        {
            "name": "handler_contract_fsm",
            "description": "FSM for handler failure contract tests",
            "version": "4.1",
            "initial_state": "start",
            "states": {
                "start": {
                    "id": "start",
                    "description": "Start state",
                    "purpose": "Collect nothing; exercise handler timings",
                    "response_instructions": "Reply politely",
                    "transitions": [
                        {
                            "target_state": "done",
                            "description": "User is finished",
                            "priority": 100,
                            "conditions": [
                                {
                                    "description": "finished flag present",
                                    "requires_context_keys": ["finished"],
                                }
                            ],
                        }
                    ],
                },
                "done": {
                    "id": "done",
                    "description": "Terminal state",
                    "purpose": "Say goodbye",
                    "transitions": [],
                },
            },
        }
    )


def _make_api(mock_llm2_interface, handlers, error_mode="continue"):
    api = API.from_definition(
        _fsm_definition(),
        llm_interface=mock_llm2_interface,
        handler_error_mode=error_mode,
        handlers=list(handlers),
    )
    conversation_id, _ = api.start_conversation()
    return api, conversation_id


# ----------------------------------------------------------------------
# (a) critical handler failure must escape API.converse
# ----------------------------------------------------------------------


class TestCriticalHandlerPropagatesThroughPipeline:
    def test_critical_pre_processing_failure_raises_out_of_converse(
        self, mock_llm2_interface
    ):
        """SC-1: critical=True + error_mode='continue' -> converse raises."""
        api, conv_id = _make_api(
            mock_llm2_interface,
            [
                _ExplodingHandler(
                    "critical_pre",
                    HandlerTiming.PRE_PROCESSING,
                    critical=True,
                )
            ],
        )

        with pytest.raises(HandlerExecutionError, match="critical_pre"):
            api.converse("hello", conv_id)

    def test_critical_post_processing_failure_raises_out_of_converse(
        self, mock_llm2_interface
    ):
        """SC-1 (d): the shared wrapper covers more than PRE_PROCESSING."""
        api, conv_id = _make_api(
            mock_llm2_interface,
            [
                _ExplodingHandler(
                    "critical_post",
                    HandlerTiming.POST_PROCESSING,
                    critical=True,
                )
            ],
        )

        with pytest.raises(HandlerExecutionError, match="critical_post"):
            api.converse("hello", conv_id)

    def test_non_critical_handler_raises_under_raise_mode(self, mock_llm2_interface):
        """Pre-existing behavior: error_mode='raise' still propagates."""
        api, conv_id = _make_api(
            mock_llm2_interface,
            [_ExplodingHandler("plain_pre", HandlerTiming.PRE_PROCESSING)],
            error_mode="raise",
        )

        with pytest.raises(HandlerExecutionError, match="plain_pre"):
            api.converse("hello", conv_id)


# ----------------------------------------------------------------------
# (b) partial results from already-succeeded handlers survive the raise
# ----------------------------------------------------------------------


class TestPartialHandlerResultsPreserved:
    def test_earlier_handler_context_survives_critical_failure(
        self, mock_llm2_interface
    ):
        """SC-2: 'continue' means 'keep what worked', not 'discard the batch'."""
        api, conv_id = _make_api(
            mock_llm2_interface,
            [
                _RecordingHandler(
                    "validator",
                    HandlerTiming.PRE_PROCESSING,
                    {"validated": True},
                    priority=10,
                ),
                _ExplodingHandler(
                    "critical_pre",
                    HandlerTiming.PRE_PROCESSING,
                    priority=20,
                    critical=True,
                ),
            ],
        )

        with pytest.raises(HandlerExecutionError, match="critical_pre"):
            api.converse("hello", conv_id)

        assert api.get_data(conv_id)["validated"] is True

    def test_partial_context_empty_when_first_handler_fails(self, mock_llm2_interface):
        """Edge case: failure in the first handler -> merge is a no-op, raise still fires."""
        api, conv_id = _make_api(
            mock_llm2_interface,
            [
                _ExplodingHandler(
                    "critical_first",
                    HandlerTiming.PRE_PROCESSING,
                    priority=10,
                    critical=True,
                ),
                _RecordingHandler(
                    "never_runs",
                    HandlerTiming.PRE_PROCESSING,
                    {"validated": True},
                    priority=20,
                ),
            ],
        )

        with pytest.raises(HandlerExecutionError, match="critical_first"):
            api.converse("hello", conv_id)

        assert "validated" not in api.get_data(conv_id)


# ----------------------------------------------------------------------
# (c) non-regression: non-critical failures under "continue" stay swallowed
# ----------------------------------------------------------------------


class TestNonCriticalContinueUnchanged:
    def test_non_critical_failure_still_completes_the_turn(self, mock_llm2_interface):
        """SC-3: the common case must be byte-identical to pre-fix behavior."""
        api, conv_id = _make_api(
            mock_llm2_interface,
            [_ExplodingHandler("plain_pre", HandlerTiming.PRE_PROCESSING)],
        )

        response = api.converse("hello", conv_id)

        assert isinstance(response, str)
        assert response

    def test_successful_handlers_are_unaffected(self, mock_llm2_interface):
        """A timing point where nothing raises behaves exactly as before."""
        api, conv_id = _make_api(
            mock_llm2_interface,
            [
                _RecordingHandler(
                    "ok_pre",
                    HandlerTiming.PRE_PROCESSING,
                    {"validated": True},
                    priority=10,
                )
            ],
        )

        response = api.converse("hello", conv_id)

        assert isinstance(response, str)
        assert api.get_data(conv_id)["validated"] is True
