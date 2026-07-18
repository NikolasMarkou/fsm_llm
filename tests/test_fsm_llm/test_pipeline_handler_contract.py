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

Two boundaries of that contract are pinned explicitly below, because both were
asserted in the iteration-1 artifacts before any test covered them:

  * ``TestPostTransitionHandlerFailure`` pins the ACTUAL POST_TRANSITION
    semantics: guarantee 2 does NOT extend across a transition rollback.  The
    rollback wins, on purpose.  This test documents truth, not an aspiration --
    do not "fix" the rollback to make it pass differently.
  * ``TestCriticalHandlerAtPostTransitionExtraction`` pins CONTEXT_UPDATE at the
    *second* of its two call sites (post-transition re-extraction), which sits
    inside a broad ``except Exception`` that used to swallow handler failures.
"""

import pytest

from fsm_llm.api import API
from fsm_llm.definitions import FSMDefinition
from fsm_llm.handlers import BaseHandler, HandlerExecutionError, HandlerTiming
from tests.conftest import MockLLM2Interface

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


# ----------------------------------------------------------------------
# (d) transition-path helpers
# ----------------------------------------------------------------------


class _StateScopedExplodingHandler(BaseHandler):
    """Raises at ``timing``, but only when ``current_state`` matches ``only_state``.

    CONTEXT_UPDATE fires at two distinct call sites in one turn
    (``_execute_extraction_and_transition_pass``: once before the transition with
    ``current_state`` still the OLD state, once after it during post-transition
    re-extraction with ``current_state`` already the NEW state).  Scoping on
    ``current_state`` is what lets a test target the second site specifically.
    """

    def __init__(self, name, timing, only_state, priority=100, critical=False):
        super().__init__(name=name, priority=priority, critical=critical)
        self._timing = timing
        self._only_state = only_state
        self.seen_states = []

    def should_execute(
        self, timing, current_state, target_state, context, updated_keys=None
    ):
        if timing != self._timing:
            return False
        self.seen_states.append(current_state)
        return current_state == self._only_state

    def execute(self, context):
        raise RuntimeError(f"boom in {self.name}")


def _transitioning_fsm_definition():
    """FSM that transitions on turn 1 AND has an unfilled field in the target state.

    ``start`` needs ``finished``; supplying it fires the transition to ``done``.
    ``done`` needs ``followup``, which is still missing at transition time, so the
    pipeline runs post-transition re-extraction in ``done`` -- the code path that
    owns the second CONTEXT_UPDATE call site.
    """
    return FSMDefinition.model_validate(
        {
            "name": "handler_contract_transition_fsm",
            "description": "FSM that transitions and then re-extracts",
            "version": "4.1",
            "initial_state": "start",
            "states": {
                "start": {
                    "id": "start",
                    "description": "Start state",
                    "purpose": "Collect the finished flag",
                    "required_context_keys": ["finished"],
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
                    "purpose": "Collect the followup",
                    "required_context_keys": ["followup"],
                    "response_instructions": "Say goodbye",
                    "transitions": [],
                },
            },
        }
    )


def _make_transitioning_api(handlers):
    """API whose mock LLM supplies BOTH fields, so turn 1 transitions and re-extracts."""
    llm = MockLLM2Interface(extraction_data={"finished": "yes", "followup": "abc"})
    api = API.from_definition(
        _transitioning_fsm_definition(),
        llm_interface=llm,
        handler_error_mode="continue",
        handlers=list(handlers),
    )
    conversation_id, _ = api.start_conversation()
    return api, conversation_id


# ----------------------------------------------------------------------
# (e) POST_TRANSITION: the rollback overrides partial-result preservation
# ----------------------------------------------------------------------


class TestPostTransitionHandlerFailure:
    """Pins the ACTUAL semantics at POST_TRANSITION, which are NOT SC-2's.

    ``MessagePipeline.execute_handlers`` merges the failing batch's partial
    context into the instance, but ``_execute_state_transition`` then clears the
    context and restores the snapshot it took BEFORE POST_TRANSITION ran.  So the
    rollback wins and the partial delta is discarded.

    That is deliberate: a half-applied transition (new state kept, or context
    mutated by handlers that ran for a transition which was then undone) is worse
    than a lost partial delta.  These assertions therefore encode the real
    behavior.  If a future change makes ``saved`` survive here, that is a
    behavior change to argue for explicitly -- not a test to relax.
    """

    def test_critical_post_transition_failure_raises_and_rolls_back(self):
        api, conv_id = _make_transitioning_api(
            [
                _RecordingHandler(
                    "saver",
                    HandlerTiming.POST_TRANSITION,
                    {"saved": True},
                    priority=10,
                ),
                _ExplodingHandler(
                    "critical_post_transition",
                    HandlerTiming.POST_TRANSITION,
                    priority=20,
                    critical=True,
                ),
            ]
        )

        with pytest.raises(HandlerExecutionError, match="critical_post_transition"):
            api.converse("hello", conv_id)

        # 1. State rolled back to the pre-transition state.
        assert api.get_current_state(conv_id) == "start"

        # 2. Context restored to the pre-POST_TRANSITION snapshot: data extracted
        #    earlier in the turn survives (it predates the snapshot) ...
        data = api.get_data(conv_id)
        assert data["finished"] == "yes"

        # 3. ... but the successful sibling handler's delta does NOT. The rollback
        #    intentionally overrides the partial-result merge at this timing point.
        assert "saved" not in data


# ----------------------------------------------------------------------
# (f) CONTEXT_UPDATE at the post-transition re-extraction call site
# ----------------------------------------------------------------------


class TestCriticalHandlerAtPostTransitionExtraction:
    """The critical-handler contract must not depend on FSM topology.

    The post-transition re-extraction block is wrapped in a broad
    ``except Exception`` logging "Post-transition extraction failed (non-fatal)".
    That tolerance is for EXTRACTION failures; before the D-012 fix it also
    swallowed HANDLER failures, so the same critical CONTEXT_UPDATE handler
    failed the turn when no transition occurred and was silently ignored when one
    did.
    """

    def test_critical_context_update_at_second_call_site_propagates(self):
        handler = _StateScopedExplodingHandler(
            "critical_context_update_post",
            HandlerTiming.CONTEXT_UPDATE,
            only_state="done",
            critical=True,
        )
        api, conv_id = _make_transitioning_api([handler])

        with pytest.raises(HandlerExecutionError, match="critical_context_update_post"):
            api.converse("hello", conv_id)

        # Both CONTEXT_UPDATE call sites were exercised in this single turn: the
        # pre-transition one (still in "start") and the post-transition
        # re-extraction one (already in "done"). Without both, this test would be
        # passing for the wrong reason.
        assert handler.seen_states == ["start", "done"]

    def test_non_critical_context_update_failure_still_non_fatal(self):
        """Non-regression: the "non-fatal" tolerance is preserved for everything else."""
        handler = _StateScopedExplodingHandler(
            "plain_context_update_post",
            HandlerTiming.CONTEXT_UPDATE,
            only_state="done",
            critical=False,
        )
        api, conv_id = _make_transitioning_api([handler])

        response = api.converse("hello", conv_id)

        assert isinstance(response, str)
        assert handler.seen_states == ["start", "done"]
