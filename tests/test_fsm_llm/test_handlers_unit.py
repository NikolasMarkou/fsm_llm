"""
Comprehensive unit tests for the FSM-LLM handler system.

Tests cover: HandlerSystem, HandlerBuilder, LambdaHandler, BaseHandler,
HandlerExecutionError, priority ordering, context cascading, and metadata tracking.
"""

from __future__ import annotations

import copy
import json
import threading
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from fsm_llm import API
from fsm_llm.definitions import FSMDefinition, FSMError, State, Transition
from fsm_llm.handlers import (
    BaseHandler,
    HandlerBuilder,
    HandlerErrorMode,
    HandlerExecutionError,
    HandlerSystem,
    HandlerSystemError,
    HandlerTiming,
    LambdaHandler,
    clear_keys_delta,
    clear_keys_on_entry,
    create_handler,
)
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.testing import PromptGroundedLLM
from tests.test_fsm_llm.support import (
    _c3_fsm,
    _c_log_capture,
    _correction_fsm,
    _fake_response,
    _greeter_fsm,
    _mock_llm,
    _Prov,
    _raw_data,
    _reentry_fsm,
    _Session,
    _stream_chunk,
)

# ── Helpers ───────────────────────────────────────────────────


class AlwaysRunHandler(BaseHandler):
    """A handler that always runs and returns a fixed dict."""

    def __init__(self, name="always_run", priority=100, result=None):
        super().__init__(name=name, priority=priority)
        self._result = result or {}

    def should_execute(
        self, timing, current_state, target_state, context, updated_keys=None
    ):
        return True

    def execute(self, context):
        return self._result.copy()


class FailingHandler(BaseHandler):
    """A handler whose execute() always raises."""

    def __init__(self, name="failing", priority=100, error=None):
        super().__init__(name=name, priority=priority)
        self._error = error or RuntimeError("boom")

    def should_execute(
        self, timing, current_state, target_state, context, updated_keys=None
    ):
        return True

    def execute(self, context):
        raise self._error


class NeverRunHandler(BaseHandler):
    """A handler that never runs (default BaseHandler behavior)."""

    pass


# ══════════════════════════════════════════════════════════════
# 1. HandlerSystem initialization
# ══════════════════════════════════════════════════════════════


class TestHandlerSystemInit:
    """Tests for HandlerSystem.__init__ with valid and invalid handler_error_mode."""

    def test_init_continue_mode(self):
        hs = HandlerSystem(handler_error_mode="continue")
        assert hs.handler_error_mode == "continue"
        assert hs.handlers == []

    def test_init_raise_mode(self):
        hs = HandlerSystem(handler_error_mode="raise")
        assert hs.handler_error_mode == "raise"

    def test_init_default_mode_is_continue(self):
        hs = HandlerSystem()
        assert hs.handler_error_mode == "continue"

    def test_init_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid handler_error_mode"):
            HandlerSystem(handler_error_mode="skip")

    def test_init_invalid_mode_nonsense(self):
        with pytest.raises(ValueError, match="Invalid handler_error_mode"):
            HandlerSystem(handler_error_mode="explode")


class TestHandlerErrorMode:
    """HandlerErrorMode is the one closed vocabulary for handler_error_mode."""

    def test_members_and_values(self):
        assert [m.value for m in HandlerErrorMode] == ["continue", "raise"]
        assert HandlerErrorMode.CONTINUE == "continue"
        assert HandlerErrorMode.RAISE == "raise"

    @pytest.mark.parametrize(
        "given, expected",
        [
            (HandlerErrorMode.CONTINUE, HandlerErrorMode.CONTINUE),
            (HandlerErrorMode.RAISE, HandlerErrorMode.RAISE),
            ("continue", HandlerErrorMode.CONTINUE),
            ("raise", HandlerErrorMode.RAISE),
        ],
    )
    def test_accepts_members_and_strings(self, given, expected):
        assert HandlerErrorMode.from_string(given) is expected
        assert HandlerSystem(handler_error_mode=given).handler_error_mode is expected

    @pytest.mark.parametrize("bad", ["bogus", "", None, 3, 1.5])
    def test_rejects_everything_else(self, bad):
        with pytest.raises(ValueError, match="Invalid handler_error_mode") as exc:
            HandlerSystem(handler_error_mode=bad)
        assert repr(bad) in str(exc.value)

    def test_message_lists_values_not_member_reprs(self):
        with pytest.raises(ValueError) as exc:
            HandlerSystem(handler_error_mode="bogus")
        assert "['continue', 'raise']" in str(exc.value)
        assert "HandlerErrorMode." not in str(exc.value)


# ══════════════════════════════════════════════════════════════
# 2. HandlerSystem.register_handler
# ══════════════════════════════════════════════════════════════


class TestHandlerSystemRegister:
    """Registration and priority-sorted ordering."""

    def test_register_single_handler(self):
        hs = HandlerSystem()
        h = AlwaysRunHandler(name="h1")
        hs.register_handler(h)
        assert len(hs.handlers) == 1
        assert hs.handlers[0].name == "h1"

    def test_register_multiple_handlers_sorted_by_priority(self):
        hs = HandlerSystem()
        hs.register_handler(AlwaysRunHandler(name="low", priority=200))
        hs.register_handler(AlwaysRunHandler(name="high", priority=10))
        hs.register_handler(AlwaysRunHandler(name="mid", priority=100))
        names = [h.name for h in hs.handlers]
        assert names == ["high", "mid", "low"]

    def test_register_preserves_insertion_order_for_equal_priority(self):
        hs = HandlerSystem()
        hs.register_handler(AlwaysRunHandler(name="first", priority=50))
        hs.register_handler(AlwaysRunHandler(name="second", priority=50))
        # Python's sort is stable, so insertion order is preserved for equal keys
        names = [h.name for h in hs.handlers]
        assert names == ["first", "second"]


# ══════════════════════════════════════════════════════════════
# 3-4. HandlerSystem.execute_handlers — continue & raise modes
# ══════════════════════════════════════════════════════════════


class TestHandlerSystemExecuteContinue:
    """execute_handlers with handler_error_mode='continue'."""

    def test_handler_failure_continues_to_next(self):
        hs = HandlerSystem(handler_error_mode="continue")
        hs.register_handler(FailingHandler(name="fail1", priority=10))
        hs.register_handler(AlwaysRunHandler(name="ok", priority=20, result={"x": 1}))

        result = hs.execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        assert result["x"] == 1

    def test_context_propagation_across_handlers(self):
        hs = HandlerSystem(handler_error_mode="continue")
        hs.register_handler(AlwaysRunHandler(name="h1", priority=10, result={"a": 1}))
        hs.register_handler(AlwaysRunHandler(name="h2", priority=20, result={"b": 2}))

        result = hs.execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        assert result["a"] == 1
        assert result["b"] == 2


class TestHandlerSystemExecuteRaise:
    """execute_handlers with handler_error_mode='raise'."""

    def test_handler_failure_raises_handler_execution_error(self):
        hs = HandlerSystem(handler_error_mode="raise")
        hs.register_handler(FailingHandler(name="fail_handler"))

        with pytest.raises(HandlerExecutionError) as exc_info:
            hs.execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        assert exc_info.value.handler_name == "fail_handler"
        assert isinstance(exc_info.value.original_error, RuntimeError)

    def test_raise_mode_stops_at_first_failure(self):
        """After a raise, subsequent handlers should not execute."""
        hs = HandlerSystem(handler_error_mode="raise")
        tracker = []
        h1 = AlwaysRunHandler(name="h1", priority=10, result={"a": 1})
        h1_orig_execute = h1.execute

        def h1_tracked(ctx):
            tracker.append("h1")
            return h1_orig_execute(ctx)

        h1.execute = h1_tracked

        hs.register_handler(h1)
        hs.register_handler(FailingHandler(name="fail", priority=20))

        h3 = AlwaysRunHandler(name="h3", priority=30, result={"c": 3})
        h3_orig_execute = h3.execute

        def h3_tracked(ctx):
            tracker.append("h3")
            return h3_orig_execute(ctx)

        h3.execute = h3_tracked

        hs.register_handler(h3)

        with pytest.raises(HandlerExecutionError):
            hs.execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        # h1 should have run, h3 should NOT have run
        assert "h1" in tracker
        assert "h3" not in tracker


# ══════════════════════════════════════════════════════════════
# 5. execute_handlers — empty handler list
# ══════════════════════════════════════════════════════════════


class TestHandlerSystemExecuteEmpty:
    """execute_handlers with no handlers registered."""

    def test_empty_handlers_returns_empty_dict(self):
        hs = HandlerSystem()
        result = hs.execute_handlers(
            HandlerTiming.PRE_PROCESSING, "s1", None, {"existing": True}
        )
        assert result == {}

    def test_no_matching_handlers_returns_empty_dict(self):
        """All handlers return should_execute=False."""
        hs = HandlerSystem()
        hs.register_handler(NeverRunHandler(name="never"))
        result = hs.execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        assert result == {}


# ══════════════════════════════════════════════════════════════
# 5b. execute_handlers — deferred deep-copy (D-020)
# ══════════════════════════════════════════════════════════════


class TestExecuteHandlersDeepCopyDeferral:
    """D-020 / fresh-audit-sweep.md finding. ``execute_handlers`` used to
    ``copy.deepcopy(context)`` unconditionally, BEFORE checking whether any
    registered handler at this timing would actually pass its own
    ``should_execute()`` filter for the current state/target. ``should_execute``
    never mutates ``context`` (confirmed by reading ``BaseHandler``'s default
    and ``LambdaHandler``'s implementation -- both read-only), so the copy can
    be deferred until the first handler confirmed to run, or skipped
    entirely when none will.
    """

    def test_deep_copy_skipped_when_no_handler_passes_should_execute(self):
        """The plan's exact scenario: 2+ registered handlers, none of which
        pass ``should_execute`` for the current state."""
        hs = HandlerSystem()
        hs.register_handler(NeverRunHandler(name="never1", priority=10))
        hs.register_handler(NeverRunHandler(name="never2", priority=20))

        with patch("fsm_llm.handlers.copy.deepcopy", wraps=copy.deepcopy) as spy:
            result = hs.execute_handlers(
                HandlerTiming.PRE_PROCESSING, "s1", None, {"existing": True}
            )

        assert result == {}, "behavior changed for the no-match case"
        assert spy.call_count == 0, (
            f"copy.deepcopy was called {spy.call_count} time(s) even though "
            "no registered handler at this timing would ever run"
        )

    def test_zero_registered_handlers_also_skips_deep_copy(self):
        """A DIFFERENT code path than the test above: no handler registered
        at this timing AT ALL (the pre-existing early-return `candidates`
        guard), vs. handlers registered but all failing their own filter."""
        hs = HandlerSystem()

        with patch("fsm_llm.handlers.copy.deepcopy", wraps=copy.deepcopy) as spy:
            result = hs.execute_handlers(
                HandlerTiming.PRE_PROCESSING, "s1", None, {"existing": True}
            )

        assert result == {}
        assert spy.call_count == 0

    def test_deep_copy_happens_exactly_once_when_a_handler_will_run(self):
        """Control: when SOME handler passes, the deep copy must still
        happen -- exactly once, not once per probed candidate (which would
        silently reintroduce the removed cost under a different shape)."""
        hs = HandlerSystem()
        hs.register_handler(NeverRunHandler(name="never", priority=10))
        hs.register_handler(AlwaysRunHandler(name="runs", priority=20, result={"x": 1}))

        with patch("fsm_llm.handlers.copy.deepcopy", wraps=copy.deepcopy) as spy:
            result = hs.execute_handlers(
                HandlerTiming.PRE_PROCESSING, "s1", None, {"existing": True}
            )

        assert result == {"x": 1}
        assert spy.call_count == 1, (
            f"copy.deepcopy was called {spy.call_count} time(s), expected exactly 1"
        )

    def test_should_execute_sees_cascaded_updates_from_earlier_handler(self):
        """Self-check for the refactor itself (not just the plan's own
        stated scenario): a LATER handler's OWN ``should_execute()`` probe --
        not just its ``execute()`` body, which the pre-existing
        ``TestContextCascading`` class already covers -- must still see
        updates merged by an EARLIER handler. This exercises the
        `probe_context` hand-off from the original, uncopied `context` to
        the lazily-created `updated_context`; a bug that left the probe
        pinned to the stale original `context` would make ``gated`` below
        never run, and this test would catch it where the simpler
        no-should_execute-filter cascading tests would not.
        """
        hs = HandlerSystem()
        hs.register_handler(
            create_handler("setter").with_priority(10).do(lambda ctx: {"flag": True})
        )
        hs.register_handler(
            create_handler("gated")
            .with_priority(20)
            .when_context_has("flag")
            .do(lambda ctx: {"saw_flag": ctx.get("flag")})
        )

        result = hs.execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        assert result["flag"] is True
        assert result["saw_flag"] is True


class TestExecuteHandlersDeepCopyFailureIsNotSwallowed:
    """D-025 (iter-3 completion-fix) / review-iter-3.md WARNING 2.

    D-020 moved ``copy.deepcopy(context)`` INSIDE the per-handler ``try:``
    block that was meant to catch handler EXECUTION errors. That meant a
    non-deep-copyable context value (e.g. a live ``threading.Lock`` a caller
    left in ``context``) with a handler that WOULD qualify:

    * under the default ``handler_error_mode="continue"``: silently produced ``{}``
      -- no exception, the handler simply never ran -- instead of the bare
      ``TypeError`` it raised pre-D-020.
    * under ``handler_error_mode="raise"``: raised ``HandlerExecutionError``
      attributed to the qualifying handler, even though that handler was
      NEVER actually invoked -- the deep-copy failed before ``execute()``
      was ever called.

    D-025 hoists the deep-copy back out of the handler-execution ``try:``
    (it still only runs once ``should_execute()`` has confirmed a qualifying
    handler was found, preserving D-020's deferral benefit) so a
    non-deep-copyable context value raises loudly, in both error modes,
    out of ``execute_handlers`` itself.
    """

    def _context_with_non_deep_copyable_value(self):
        return {"lock": threading.Lock(), "a": 1}

    def test_continue_mode_raises_instead_of_silently_swallowing(self):
        hs = HandlerSystem(handler_error_mode="continue")
        hs.register_handler(AlwaysRunHandler(name="qualifies", result={"x": 1}))

        with pytest.raises(TypeError):
            hs.execute_handlers(
                HandlerTiming.PRE_PROCESSING,
                "s1",
                None,
                self._context_with_non_deep_copyable_value(),
            )

    def test_raise_mode_raises_a_copy_failure_not_a_handler_attribution(self):
        hs = HandlerSystem(handler_error_mode="raise")
        hs.register_handler(AlwaysRunHandler(name="qualifies", result={"x": 1}))

        # The deep-copy failure must surface as the plain TypeError it is,
        # not get wrapped into a HandlerExecutionError that names "qualifies"
        # as the failing handler -- that handler never actually ran.
        with pytest.raises(TypeError) as excinfo:
            hs.execute_handlers(
                HandlerTiming.PRE_PROCESSING,
                "s1",
                None,
                self._context_with_non_deep_copyable_value(),
            )
        assert "qualifies" not in str(excinfo.value)

    def test_handler_genuinely_never_ran_when_copy_fails(self):
        """The qualifying handler's ``execute()`` must never be reached."""
        ran = []
        hs = HandlerSystem(handler_error_mode="continue")
        hs.register_handler(
            create_handler("would_run").do(lambda ctx: ran.append(True) or {})
        )

        with pytest.raises(TypeError):
            hs.execute_handlers(
                HandlerTiming.PRE_PROCESSING,
                "s1",
                None,
                self._context_with_non_deep_copyable_value(),
            )

        assert ran == [], "handler's execute() ran despite the deep-copy failure"


# ══════════════════════════════════════════════════════════════
# 6. Handler returning None, empty dict, non-dict
# ══════════════════════════════════════════════════════════════


class TestHandlerReturnValues:
    """LambdaHandler.execute normalizes non-dict returns."""

    def test_handler_returning_none(self):
        handler = create_handler("none_handler").do(lambda ctx: None)
        result = handler.execute({"key": "value"})
        assert result == {}

    def test_handler_returning_empty_dict(self):
        handler = create_handler("empty_handler").do(lambda ctx: {})
        result = handler.execute({"key": "value"})
        assert result == {}

    def test_handler_returning_non_dict(self):
        """A non-dict return raises a RAW TypeError.

        ``LambdaHandler.execute`` no longer wraps: ``HandlerSystem.execute_handlers``
        is the single wrapping site, so the ``HandlerExecutionError`` a caller
        sees is produced there. See ``test_core_hardening_seam.py``
        ``TestLambdaHandlerSingleWrapSite`` for the user-visible shape.
        """
        handler = create_handler("non_dict_handler").do(lambda ctx: "not a dict")
        with pytest.raises(TypeError, match="Handler returned non-dict result: str"):
            handler.execute({"key": "value"})

    def test_handler_returning_valid_dict(self):
        handler = create_handler("dict_handler").do(lambda ctx: {"new_key": 42})
        result = handler.execute({"key": "value"})
        assert result == {"new_key": 42}

    def test_none_result_does_not_update_output_context(self):
        """When a handler returns None, the HandlerSystem should not update output."""
        hs = HandlerSystem()
        hs.register_handler(
            create_handler("nil").at(HandlerTiming.PRE_PROCESSING).do(lambda ctx: None)
        )
        result = hs.execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        # Only metadata, no other keys (None result is skipped)
        assert "new_key" not in result


# ══════════════════════════════════════════════════════════════
# 7. HandlerBuilder — fluent API
# ══════════════════════════════════════════════════════════════


class TestHandlerBuilderFluentAPI:
    """Fluent API methods: .at(), .on_state(), .on_states(), .with_priority(), .do(), .build()."""

    def test_at_single_timing(self):
        handler = (
            create_handler("t1").at(HandlerTiming.PRE_PROCESSING).do(lambda ctx: {})
        )
        assert HandlerTiming.PRE_PROCESSING in handler.timings

    def test_at_multiple_timings(self):
        handler = (
            create_handler("t2")
            .at(HandlerTiming.PRE_PROCESSING, HandlerTiming.POST_PROCESSING)
            .do(lambda ctx: {})
        )
        assert HandlerTiming.PRE_PROCESSING in handler.timings
        assert HandlerTiming.POST_PROCESSING in handler.timings

    def test_on_state_single(self):
        handler = create_handler("s1").on_state("greeting").do(lambda ctx: {})
        assert "greeting" in handler.states

    def test_on_state_multiple(self):
        handler = (
            create_handler("s2").on_state("greeting", "farewell").do(lambda ctx: {})
        )
        assert handler.states == {"greeting", "farewell"}

    def test_with_priority(self):
        handler = create_handler("p1").with_priority(42).do(lambda ctx: {})
        assert handler.priority == 42

    def test_default_priority_is_100(self):
        handler = create_handler("dp").do(lambda ctx: {})
        assert handler.priority == 100

    def test_do_returns_handler_not_builder(self):
        result = create_handler("done").do(lambda ctx: {})
        assert isinstance(result, BaseHandler)
        assert isinstance(result, LambdaHandler)

    def test_when_context_has(self):
        handler = (
            create_handler("ctx")
            .when_context_has("user_name", "email")
            .do(lambda ctx: {})
        )
        assert handler.required_keys == {"user_name", "email"}

    def test_when_keys_updated(self):
        handler = create_handler("upd").when_keys_updated("score").do(lambda ctx: {})
        assert handler.updated_keys == {"score"}

    def test_on_state_entry_shorthand(self):
        handler = create_handler("entry").on_state_entry("completed").do(lambda ctx: {})
        assert HandlerTiming.POST_TRANSITION in handler.timings
        assert "completed" in handler.target_states

    def test_on_state_exit_shorthand(self):
        handler = create_handler("exit").on_state_exit("collecting").do(lambda ctx: {})
        assert HandlerTiming.PRE_TRANSITION in handler.timings
        assert "collecting" in handler.states

    def test_on_context_update_shorthand(self):
        handler = create_handler("cu").on_context_update("user_name").do(lambda ctx: {})
        assert HandlerTiming.CONTEXT_UPDATE in handler.timings
        assert "user_name" in handler.updated_keys

    def test_not_on_state(self):
        handler = create_handler("ns").not_on_state("error").do(lambda ctx: {})
        assert "error" in handler.not_states

    def test_not_on_target_state(self):
        handler = create_handler("nts").not_on_target_state("error").do(lambda ctx: {})
        assert "error" in handler.not_target_states

    def test_on_target_state(self):
        handler = create_handler("ts").on_target_state("farewell").do(lambda ctx: {})
        assert "farewell" in handler.target_states

    def test_chaining_returns_same_builder(self):
        builder = create_handler("chain")
        same = (
            builder.at(HandlerTiming.PRE_PROCESSING)
            .on_state("s1")
            .with_priority(10)
            .when_context_has("k1")
        )
        assert same is builder

    def test_handler_builder_build_copies_collections(self):
        builder = (
            create_handler("copies")
            .when(lambda *a: True)
            .on_state("s1")
            .on_target_state("t1")
            .not_on_state("n1")
            .not_on_target_state("nt1")
            .at(HandlerTiming.PRE_PROCESSING)
            .when_context_has("k1")
            .when_keys_updated("u1")
        )
        builder.execution_lambda = lambda ctx: {}
        h = builder.build()
        # every collection build() copies into the handler
        names = (
            "condition_lambdas",
            "timings",
            "states",
            "target_states",
            "required_keys",
            "updated_keys",
            "not_states",
            "not_target_states",
        )
        for attr in names:
            assert getattr(h, attr) is not getattr(builder, attr), attr

        builder.when(lambda *a: False)
        builder.on_state("s2")
        builder.on_target_state("t2")
        builder.not_on_state("n2")
        builder.not_on_target_state("nt2")
        builder.at(HandlerTiming.POST_PROCESSING)
        builder.when_context_has("k2")
        builder.when_keys_updated("u2")
        assert len(h.condition_lambdas) == 1
        assert h.states == {"s1"}
        assert h.target_states == {"t1"}
        assert h.not_states == {"n1"}
        assert h.not_target_states == {"nt1"}
        assert h.timings == {HandlerTiming.PRE_PROCESSING}
        assert h.required_keys == {"k1"}
        assert h.updated_keys == {"u1"}

        h.states.add("only_h")
        h.target_states.add("only_h")
        h.not_states.add("only_h")
        h.not_target_states.add("only_h")
        h.required_keys.add("only_h")
        h.updated_keys.add("only_h")
        h.timings.add(HandlerTiming.PRE_TRANSITION)
        h.condition_lambdas.append(lambda *a: True)
        assert builder.states == {"s1", "s2"}
        assert builder.target_states == {"t1", "t2"}
        assert builder.not_states == {"n1", "n2"}
        assert builder.not_target_states == {"nt1", "nt2"}
        assert builder.required_keys == {"k1", "k2"}
        assert builder.updated_keys == {"u1", "u2"}
        assert HandlerTiming.PRE_TRANSITION not in builder.timings
        assert len(builder.condition_lambdas) == 2

        second = builder.build()
        assert second.states == {"s1", "s2"}
        assert second.target_states == {"t1", "t2"}
        assert second.not_states == {"n1", "n2"}
        assert second.not_target_states == {"nt1", "nt2"}
        assert second.required_keys == {"k1", "k2"}
        assert second.updated_keys == {"u1", "u2"}
        assert second.timings == {
            HandlerTiming.PRE_PROCESSING,
            HandlerTiming.POST_PROCESSING,
        }
        assert len(second.condition_lambdas) == 2


# ══════════════════════════════════════════════════════════════
# 8. HandlerBuilder — missing .do() raises ValueError
# ══════════════════════════════════════════════════════════════


class TestHandlerBuilderMissingDo:
    """build() without .do() must raise ValueError."""

    def test_build_without_do_raises(self):
        builder = create_handler("no_do")
        with pytest.raises(ValueError, match="Execution lambda is required"):
            builder.build()

    def test_build_with_do_succeeds(self):
        builder = create_handler("with_do")
        builder.execution_lambda = lambda ctx: {}
        handler = builder.build()
        assert isinstance(handler, LambdaHandler)


# ══════════════════════════════════════════════════════════════
# 9. HandlerBuilder — all timing points
# ══════════════════════════════════════════════════════════════


class TestAllTimingPoints:
    """Every HandlerTiming value can be used with the builder."""

    @pytest.mark.parametrize("timing", list(HandlerTiming))
    def test_timing_point_accepted(self, timing):
        handler = create_handler(f"t_{timing.name}").at(timing).do(lambda ctx: {})
        assert timing in handler.timings

    @pytest.mark.parametrize("timing", list(HandlerTiming))
    def test_handler_executes_at_matching_timing(self, timing):
        handler = (
            create_handler(f"t_{timing.name}").at(timing).do(lambda ctx: {"ran": True})
        )
        assert handler.should_execute(timing, "s1", None, {})

    @pytest.mark.parametrize("timing", list(HandlerTiming))
    def test_handler_rejects_non_matching_timing(self, timing):
        # Pick a different timing
        other = next(t for t in HandlerTiming if t != timing)
        handler = create_handler(f"t_{timing.name}").at(timing).do(lambda ctx: {})
        assert not handler.should_execute(other, "s1", None, {})


# ══════════════════════════════════════════════════════════════
# 10. LambdaHandler.should_execute — filters
# ══════════════════════════════════════════════════════════════


class TestLambdaHandlerShouldExecute:
    """Tests for LambdaHandler.should_execute condition filtering."""

    def test_timing_filter(self):
        handler = (
            create_handler("tf").at(HandlerTiming.PRE_PROCESSING).do(lambda ctx: {})
        )
        assert handler.should_execute(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        assert not handler.should_execute(HandlerTiming.POST_PROCESSING, "s1", None, {})

    def test_state_filter(self):
        handler = create_handler("sf").on_state("active").do(lambda ctx: {})
        assert handler.should_execute(HandlerTiming.PRE_PROCESSING, "active", None, {})
        assert not handler.should_execute(
            HandlerTiming.PRE_PROCESSING, "inactive", None, {}
        )

    def test_not_state_filter(self):
        handler = create_handler("nsf").not_on_state("error").do(lambda ctx: {})
        assert handler.should_execute(HandlerTiming.PRE_PROCESSING, "active", None, {})
        assert not handler.should_execute(
            HandlerTiming.PRE_PROCESSING, "error", None, {}
        )

    def test_target_state_filter(self):
        handler = create_handler("tsf").on_target_state("farewell").do(lambda ctx: {})
        assert handler.should_execute(
            HandlerTiming.POST_TRANSITION, "s1", "farewell", {}
        )
        assert not handler.should_execute(
            HandlerTiming.POST_TRANSITION, "s1", "greeting", {}
        )
        assert not handler.should_execute(HandlerTiming.POST_TRANSITION, "s1", None, {})

    def test_not_target_state_filter(self):
        handler = create_handler("ntsf").not_on_target_state("error").do(lambda ctx: {})
        assert handler.should_execute(HandlerTiming.POST_TRANSITION, "s1", "ok", {})
        assert not handler.should_execute(
            HandlerTiming.POST_TRANSITION, "s1", "error", {}
        )
        # None target_state does not match not_target_states, so should pass
        assert handler.should_execute(HandlerTiming.POST_TRANSITION, "s1", None, {})

    def test_required_keys_filter(self):
        handler = create_handler("rkf").when_context_has("user_name").do(lambda ctx: {})
        assert handler.should_execute(
            HandlerTiming.PRE_PROCESSING, "s1", None, {"user_name": "Alice"}
        )
        assert not handler.should_execute(
            HandlerTiming.PRE_PROCESSING, "s1", None, {"email": "a@b.c"}
        )

    def test_updated_keys_filter(self):
        handler = create_handler("ukf").when_keys_updated("score").do(lambda ctx: {})
        assert handler.should_execute(
            HandlerTiming.CONTEXT_UPDATE, "s1", None, {}, updated_keys={"score", "name"}
        )
        assert not handler.should_execute(
            HandlerTiming.CONTEXT_UPDATE, "s1", None, {}, updated_keys={"name"}
        )
        assert not handler.should_execute(
            HandlerTiming.CONTEXT_UPDATE, "s1", None, {}, updated_keys=None
        )

    def test_no_filters_means_always_execute(self):
        handler = create_handler("open").do(lambda ctx: {})
        assert handler.should_execute(
            HandlerTiming.PRE_PROCESSING, "any_state", "any_target", {}
        )

    def test_combined_filters_all_must_pass(self):
        handler = (
            create_handler("combo")
            .at(HandlerTiming.POST_TRANSITION)
            .on_state("active")
            .on_target_state("done")
            .when_context_has("key1")
            .do(lambda ctx: {})
        )
        # All conditions met
        assert handler.should_execute(
            HandlerTiming.POST_TRANSITION, "active", "done", {"key1": True}
        )
        # Wrong timing
        assert not handler.should_execute(
            HandlerTiming.PRE_PROCESSING, "active", "done", {"key1": True}
        )
        # Wrong state
        assert not handler.should_execute(
            HandlerTiming.POST_TRANSITION, "inactive", "done", {"key1": True}
        )
        # Wrong target
        assert not handler.should_execute(
            HandlerTiming.POST_TRANSITION, "active", "pending", {"key1": True}
        )
        # Missing key
        assert not handler.should_execute(
            HandlerTiming.POST_TRANSITION, "active", "done", {}
        )


# ══════════════════════════════════════════════════════════════
# 11. LambdaHandler.execute — sync and async
# ══════════════════════════════════════════════════════════════


class TestLambdaHandlerExecute:
    """Sync and async execution paths."""

    def test_sync_execution(self):
        handler = create_handler("sync").do(lambda ctx: {"result": ctx.get("x", 0) + 1})
        result = handler.execute({"x": 5})
        assert result == {"result": 6}

    def test_none_return_becomes_empty_dict(self):
        handler = create_handler("none_return").do(lambda ctx: None)
        result = handler.execute({})
        assert result == {}

    def test_execute_propagates_the_raw_exception(self):
        """The lambda's exception escapes ``execute`` unwrapped.

        Wrapping is ``HandlerSystem.execute_handlers``'s job -- the same single
        site that has always served plain ``BaseHandler`` subclasses.
        """

        def bad_fn(ctx):
            raise ValueError("bad value")

        handler = create_handler("bad").do(bad_fn)
        with pytest.raises(ValueError, match="bad value"):
            handler.execute({})

    def test_execute_failure_is_wrapped_once_by_the_handler_system(self):
        """The caller-facing contract: one wrap, raw cause preserved."""

        def bad_fn(ctx):
            raise ValueError("bad value")

        system = HandlerSystem(handler_error_mode="raise")
        system.register_handler(
            create_handler("bad").at(HandlerTiming.PRE_PROCESSING).do(bad_fn)
        )

        with pytest.raises(HandlerExecutionError) as exc_info:
            system.execute_handlers(HandlerTiming.PRE_PROCESSING, "start", None, {})

        assert exc_info.value.handler_name == "bad"
        assert isinstance(exc_info.value.original_error, ValueError)
        assert str(exc_info.value).count("Error in handler") == 1


# ══════════════════════════════════════════════════════════════
# 12. LambdaHandler — with custom condition lambda
# ══════════════════════════════════════════════════════════════


class TestLambdaHandlerCustomCondition:
    """Custom condition lambda via .when()."""

    def test_custom_condition_passes(self):
        handler = (
            create_handler("cond")
            .when(lambda t, s, ts, ctx, uk: ctx.get("score", 0) > 50)
            .do(lambda ctx: {"bonus": True})
        )
        assert handler.should_execute(
            HandlerTiming.PRE_PROCESSING, "s1", None, {"score": 80}
        )

    def test_custom_condition_fails(self):
        handler = (
            create_handler("cond")
            .when(lambda t, s, ts, ctx, uk: ctx.get("score", 0) > 50)
            .do(lambda ctx: {"bonus": True})
        )
        assert not handler.should_execute(
            HandlerTiming.PRE_PROCESSING, "s1", None, {"score": 30}
        )

    def test_multiple_conditions_all_must_pass(self):
        handler = (
            create_handler("multi")
            .when(lambda t, s, ts, ctx, uk: ctx.get("a", 0) > 0)
            .when(lambda t, s, ts, ctx, uk: ctx.get("b", 0) > 0)
            .do(lambda ctx: {})
        )
        assert handler.should_execute(
            HandlerTiming.PRE_PROCESSING, "s1", None, {"a": 1, "b": 2}
        )
        assert not handler.should_execute(
            HandlerTiming.PRE_PROCESSING, "s1", None, {"a": 1, "b": -1}
        )
        assert not handler.should_execute(
            HandlerTiming.PRE_PROCESSING, "s1", None, {"a": -1, "b": 2}
        )

    def test_condition_exception_raises_handler_error(self):
        """A failing condition lambda should raise HandlerExecutionError."""
        handler = (
            create_handler("exc_cond")
            .when(lambda t, s, ts, ctx, uk: 1 / 0)  # ZeroDivisionError
            .do(lambda ctx: {})
        )
        with pytest.raises(HandlerExecutionError):
            handler.should_execute(HandlerTiming.PRE_PROCESSING, "s1", None, {})


# ══════════════════════════════════════════════════════════════
# 13. HandlerExecutionError wrapping
# ══════════════════════════════════════════════════════════════


class TestHandlerExecutionError:
    """HandlerExecutionError preserves handler name and original error."""

    def test_error_attributes(self):
        orig = ValueError("some detail")
        err = HandlerExecutionError("my_handler", orig)
        assert err.handler_name == "my_handler"
        assert err.original_error is orig

    def test_error_message_format(self):
        orig = RuntimeError("connection refused")
        err = HandlerExecutionError("api_handler", orig)
        assert "api_handler" in str(err)
        assert "connection refused" in str(err)

    def test_is_subclass_of_handler_system_error(self):
        err = HandlerExecutionError("h", Exception("x"))
        assert isinstance(err, HandlerSystemError)

    def test_is_catchable_as_exception(self):
        err = HandlerExecutionError("h", Exception("x"))
        assert isinstance(err, Exception)


# ══════════════════════════════════════════════════════════════
# 14. Priority ordering — lower number executes first
# ══════════════════════════════════════════════════════════════


class TestPriorityOrdering:
    """Handlers execute in priority order (lower numbers first)."""

    def test_execution_order_follows_priority(self):
        execution_log = []

        def make_fn(label):
            def fn(ctx):
                execution_log.append(label)
                return {}

            return fn

        hs = HandlerSystem()
        hs.register_handler(create_handler("low").with_priority(200).do(make_fn("low")))
        hs.register_handler(create_handler("high").with_priority(1).do(make_fn("high")))
        hs.register_handler(create_handler("mid").with_priority(100).do(make_fn("mid")))

        hs.execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        assert execution_log == ["high", "mid", "low"]

    def test_priority_zero_runs_first(self):
        execution_log = []

        def make_fn(label):
            def fn(ctx):
                execution_log.append(label)
                return {}

            return fn

        hs = HandlerSystem()
        hs.register_handler(
            create_handler("default").do(make_fn("default"))  # priority=100
        )
        hs.register_handler(
            create_handler("urgent").with_priority(0).do(make_fn("urgent"))
        )

        hs.execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        assert execution_log[0] == "urgent"
        assert execution_log[1] == "default"


# ══════════════════════════════════════════════════════════════
# 15. Context cascading — later handlers see earlier updates
# ══════════════════════════════════════════════════════════════


class TestContextCascading:
    """Later handlers see context updates from earlier handlers."""

    def test_second_handler_sees_first_handlers_output(self):
        def first(ctx):
            return {"step": 1}

        def second(ctx):
            # Should see step=1 from the first handler
            return {"saw_step": ctx.get("step")}

        hs = HandlerSystem()
        hs.register_handler(create_handler("first").with_priority(10).do(first))
        hs.register_handler(create_handler("second").with_priority(20).do(second))

        result = hs.execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        assert result["step"] == 1
        assert result["saw_step"] == 1

    def test_three_handler_cascade(self):
        def h1(ctx):
            return {"counter": 1}

        def h2(ctx):
            return {"counter": ctx["counter"] + 1}

        def h3(ctx):
            return {"counter": ctx["counter"] + 1}

        hs = HandlerSystem()
        hs.register_handler(create_handler("h1").with_priority(10).do(h1))
        hs.register_handler(create_handler("h2").with_priority(20).do(h2))
        hs.register_handler(create_handler("h3").with_priority(30).do(h3))

        result = hs.execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        assert result["counter"] == 3

    def test_original_context_not_mutated(self):
        original = {"original_key": "original_value"}

        hs = HandlerSystem()
        hs.register_handler(create_handler("mutator").do(lambda ctx: {"added": True}))

        hs.execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, original)
        assert "added" not in original


# ══════════════════════════════════════════════════════════════
# 16. Handler execution metadata (tracked internally, not in user context)
# ══════════════════════════════════════════════════════════════


class TestHandlerMetadata:
    """Handler internal metadata must not leak into user-facing context output."""

    def test_metadata_not_leaked_to_output_context(self):
        hs = HandlerSystem()
        hs.register_handler(
            create_handler("meta_test")
            .at(HandlerTiming.POST_TRANSITION)
            .do(lambda ctx: {"x": 1})
        )

        result = hs.execute_handlers(HandlerTiming.POST_TRANSITION, "s1", "s2", {})
        # Metadata must NOT appear in user-facing context output
        assert "_handler_metadata" not in result
        assert result == {"x": 1}

    def test_no_metadata_when_no_handler_executes(self):
        hs = HandlerSystem()
        hs.register_handler(NeverRunHandler(name="never"))
        result = hs.execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        assert "_handler_metadata" not in result

    def test_multiple_handlers_return_merged_context(self):
        hs = HandlerSystem()
        hs.register_handler(
            create_handler("a").with_priority(10).do(lambda ctx: {"from_a": 1})
        )
        hs.register_handler(
            create_handler("b").with_priority(20).do(lambda ctx: {"from_b": 2})
        )

        result = hs.execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        assert "_handler_metadata" not in result
        assert result == {"from_a": 1, "from_b": 2}

    def test_none_result_handler_does_not_pollute_context(self):
        hs = HandlerSystem()
        hs.register_handler(create_handler("nil").do(lambda ctx: None))
        result = hs.execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        assert "_handler_metadata" not in result
        assert result == {}


# ══════════════════════════════════════════════════════════════
# Bonus: BaseHandler defaults
# ══════════════════════════════════════════════════════════════


class TestBaseHandler:
    """BaseHandler default behavior."""

    def test_default_should_execute_returns_false(self):
        h = BaseHandler(name="base")
        assert h.should_execute(HandlerTiming.PRE_PROCESSING, "s1", None, {}) is False

    def test_default_execute_returns_empty_dict(self):
        h = BaseHandler(name="base")
        assert h.execute({}) == {}

    def test_name_defaults_to_class_name(self):
        h = BaseHandler()
        assert h.name == "BaseHandler"

    def test_custom_name(self):
        h = BaseHandler(name="custom")
        assert h.name == "custom"

    def test_default_priority(self):
        h = BaseHandler()
        assert h.priority == 100

    def test_custom_priority(self):
        h = BaseHandler(priority=5)
        assert h.priority == 5


# ══════════════════════════════════════════════════════════════
# Bonus: LambdaHandler.__str__
# ══════════════════════════════════════════════════════════════


class TestLambdaHandlerStr:
    """String representation."""

    def test_str_contains_name(self):
        handler = create_handler("my_handler").do(lambda ctx: {})
        assert "my_handler" in str(handler)
        assert "Lambda Handler" in str(handler)


# ══════════════════════════════════════════════════════════════
# Bonus: create_handler convenience function
# ══════════════════════════════════════════════════════════════


class TestCreateHandlerFunction:
    """create_handler() returns a HandlerBuilder."""

    def test_returns_handler_builder(self):
        builder = create_handler("test")
        assert isinstance(builder, HandlerBuilder)

    def test_default_name(self):
        builder = create_handler()
        assert builder.name == "LambdaHandler"

    def test_custom_name(self):
        builder = create_handler("custom_name")
        assert builder.name == "custom_name"


# ══════════════════════════════════════════════════════════════
# F-14: HandlerBuilder.critical()
# ══════════════════════════════════════════════════════════════


class TestHandlerBuilderCritical:
    """``.critical()`` is the fluent path to a critical handler (F-14).

    Before this existed, the only way to get one from ``create_handler()`` was to
    mutate ``handler.critical`` on the built instance.
    """

    def test_critical_sets_the_flag(self):
        handler = (
            create_handler("c1")
            .at(HandlerTiming.PRE_PROCESSING)
            .critical()
            .do(lambda ctx: {})
        )
        assert handler.critical is True

    def test_default_is_still_not_critical(self):
        """Adding the method must not change what existing builder callers get."""
        handler = (
            create_handler("c2").at(HandlerTiming.PRE_PROCESSING).do(lambda ctx: {})
        )
        assert handler.critical is False

    def test_critical_false_is_explicitly_supported(self):
        handler = (
            create_handler("c3")
            .at(HandlerTiming.PRE_PROCESSING)
            .critical(False)
            .do(lambda ctx: {})
        )
        assert handler.critical is False

    def test_critical_can_be_cleared_again(self):
        handler = (
            create_handler("c4")
            .at(HandlerTiming.PRE_PROCESSING)
            .critical()
            .critical(False)
            .do(lambda ctx: {})
        )
        assert handler.critical is False

    def test_critical_returns_self_for_chaining(self):
        builder = create_handler("c5")
        assert builder.critical() is builder

    @pytest.mark.parametrize(
        "position",
        ["first", "middle", "last"],
    )
    def test_critical_composes_in_any_order(self, position):
        """``.critical()`` must not disturb the other chain steps, wherever it sits."""
        builder = create_handler("c6")
        if position == "first":
            builder = builder.critical()
        builder = builder.at(HandlerTiming.PRE_PROCESSING).on_state("greeting")
        if position == "middle":
            builder = builder.critical()
        builder = builder.when_context_has("user_name").with_priority(42)
        if position == "last":
            builder = builder.critical()
        handler = builder.do(lambda ctx: {})

        assert handler.critical is True
        # The other steps are unchanged regardless of where .critical() landed.
        assert handler.timings == {HandlerTiming.PRE_PROCESSING}
        assert handler.states == {"greeting"}
        assert handler.required_keys == {"user_name"}
        assert handler.priority == 42

    def test_critical_works_via_build_not_only_do(self):
        builder = create_handler("c7").at(HandlerTiming.PRE_PROCESSING).critical()
        builder.execution_lambda = lambda ctx: {}
        assert builder.build().critical is True

    # ── The flag must be WIRED, not merely set ────────────────

    def test_built_critical_handler_actually_raises_in_continue_mode(self):
        """End-to-end through the real firing site.

        An attribute-only assertion would be satisfied by a builder that sets the
        flag and wires it to nothing, so this drives ``execute_handlers`` in
        ``handler_error_mode="continue"`` -- where a NON-critical failure is swallowed.
        """
        system = HandlerSystem(handler_error_mode="continue")
        system.register_handler(
            create_handler("boom")
            .at(HandlerTiming.PRE_PROCESSING)
            .critical()
            .do(lambda ctx: 1 / 0)
        )

        with pytest.raises(HandlerExecutionError) as exc_info:
            system.execute_handlers(
                HandlerTiming.PRE_PROCESSING, "greeting", None, {}, None
            )

        assert "boom" in str(exc_info.value)

    def test_non_critical_twin_is_still_swallowed_in_continue_mode(self):
        """Over-correction guard: without ``.critical()`` the same failure is skipped.

        Without this, the test above would also pass under a build() that marked
        EVERY handler critical.
        """
        system = HandlerSystem(handler_error_mode="continue")
        system.register_handler(
            create_handler("boom")
            .at(HandlerTiming.PRE_PROCESSING)
            .do(lambda ctx: 1 / 0)
        )

        result = system.execute_handlers(
            HandlerTiming.PRE_PROCESSING, "greeting", None, {}, None
        )

        assert result == {}

    def test_lambda_handler_accepts_critical_directly(self):
        """``LambdaHandler`` dropped ``critical`` on its way to ``BaseHandler`` too."""
        handler = LambdaHandler(
            name="direct",
            condition_lambdas=[],
            execution_lambda=lambda ctx: {},
            timings={HandlerTiming.PRE_PROCESSING},
            states=set(),
            target_states=set(),
            required_keys=set(),
            updated_keys=set(),
            critical=True,
        )
        assert handler.critical is True


# ── clear_keys_on_entry / clear_keys_delta (moved from agents) ─────────


def _clear_fsm() -> dict:
    """intro (silent, unconditional) -> collect (silent, blocked) -> done."""
    return {
        "name": "clearing",
        "description": "clear_keys_on_entry through a real turn",
        "initial_state": "intro",
        "states": {
            "intro": {
                "id": "intro",
                "description": "Intro",
                "purpose": "Move on",
                "response_instructions": "",
                "transitions": [
                    {"target_state": "collect", "description": "Go", "priority": 100}
                ],
            },
            "collect": {
                "id": "collect",
                "description": "Collect",
                "purpose": "Wait",
                "response_instructions": "",
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "Never",
                        "priority": 100,
                        "conditions": [
                            {
                                "description": "never",
                                "logic": {"==": [{"var": "go"}, "now"]},
                            }
                        ],
                    }
                ],
            },
            "done": {
                "id": "done",
                "description": "Done",
                "purpose": "End",
                "response_instructions": "Say done",
            },
        },
    }


class TestClearKeysDelta:
    def test_set_keys_become_none_unset_keys_left_out(self):
        context = {"a": 1, "b": None, "d": "", "e": False}
        assert clear_keys_delta(["a", "b", "c", "d", "e"], context) == {
            "a": None,
            "d": None,
            "e": None,
        }

    def test_nothing_to_clear_is_an_empty_delta(self):
        assert clear_keys_delta(["a", "b"], {"other": 1}) == {}

    def test_order_follows_keys(self):
        assert list(clear_keys_delta(["b", "a"], {"a": 1, "b": 2})) == ["b", "a"]

    @pytest.mark.parametrize("keys", ["draft", None, ["draft", 1]])
    def test_a_non_collection_of_names_is_refused(self, keys):
        # A bare str was iterated as characters: "draft" cleared "d" and "r"
        # and left "draft" set.
        with pytest.raises(ValueError, match="keys"):
            clear_keys_delta(keys, {"draft": "x", "d": 1, "r": 2})


class TestClearKeysOnEntry:
    def test_returns_a_handler_with_name_and_priority(self):
        handler = clear_keys_on_entry(
            ["draft"], state="produce", name="fresh_draft", priority=7
        )
        assert isinstance(handler, BaseHandler)
        assert (handler.name, handler.priority) == ("fresh_draft", 7)

    @pytest.mark.parametrize("keys", ["draft", None, ["draft", 1]])
    def test_a_non_collection_of_names_is_refused_at_build_time(self, keys):
        with pytest.raises(ValueError, match="keys"):
            clear_keys_on_entry(keys, state="produce")

    def test_default_name_and_priority(self):
        handler = clear_keys_on_entry(["draft"], state="produce")
        assert (handler.name, handler.priority) == ("clear_keys_on_produce", 100)

    def test_fires_only_on_entry_to_the_named_state(self):
        handler = clear_keys_on_entry(["draft"], state="produce")
        ctx = {"draft": "x"}

        assert handler.should_execute(
            HandlerTiming.POST_TRANSITION, "check", "produce", ctx
        )
        assert not handler.should_execute(
            HandlerTiming.POST_TRANSITION, "produce", "check", ctx
        )
        assert not handler.should_execute(
            HandlerTiming.PRE_TRANSITION, "check", "produce", ctx
        )
        for timing in HandlerTiming:
            if timing is not HandlerTiming.POST_TRANSITION:
                assert not handler.should_execute(timing, "check", "produce", ctx)

    def test_delta_sets_each_held_key_to_none(self):
        handler = clear_keys_on_entry(["draft", "verdict", "draft"], state="p")
        assert handler.execute({"draft": "x", "verdict": True, "keep": 1}) == {
            "draft": None,
            "verdict": None,
        }
        assert handler.execute({"keep": 1}) == {}

    @pytest.mark.parametrize("keys", [[], ()])
    def test_empty_keys_rejected(self, keys):
        with pytest.raises(ValueError, match="at least one key"):
            clear_keys_on_entry(keys, state="produce")

    def test_priority_orders_it_among_entry_handlers(self):
        system = HandlerSystem()
        order: list[str] = []
        system.register_handler(
            create_handler("late")
            .on_state_entry("produce")
            .with_priority(50)
            .do(lambda ctx: order.append(f"late:{ctx.get('draft')}") or {})
        )
        system.register_handler(
            clear_keys_on_entry(["draft"], state="produce", priority=10)
        )

        result = system.execute_handlers(
            HandlerTiming.POST_TRANSITION, "check", "produce", {"draft": "x"}, None
        )

        assert result == {"draft": None}
        assert order == ["late:None"]

    def test_clears_on_entry_through_a_real_turn(self):
        api = API.from_definition(_clear_fsm(), llm_interface=PromptGroundedLLM())
        api.register_handler(clear_keys_on_entry(["draft", "verdict"], state="collect"))
        conv_id, _ = api.start_conversation({"draft": "old", "verdict": True, "n": 1})

        api.advance(conv_id)  # intro -> collect

        data = api.get_data(conv_id)
        assert api.get_current_state(conv_id) == "collect"
        assert "draft" not in data and "verdict" not in data
        assert data["n"] == 1

    def test_other_state_entry_leaves_keys(self):
        api = API.from_definition(_clear_fsm(), llm_interface=PromptGroundedLLM())
        api.register_handler(clear_keys_on_entry(["draft"], state="done"))
        conv_id, _ = api.start_conversation({"draft": "old"})

        api.advance(conv_id)  # intro -> collect

        assert api.get_data(conv_id)["draft"] == "old"


class _ErrorHandlerHarness:
    """`API` over a completion that fails on demand; records ERROR handler calls.

    ``fail`` is one of: ``"sync"`` (the non-stream call raises), ``"first"``
    (the stream call raises before any chunk), ``"mid"`` (the stream yields one
    chunk then raises), ``None`` (healthy).
    """

    def __init__(self, fail: str | None, exc: BaseException | None = None):
        self._mode = fail
        self.fail: str | None = None  # armed by ``start`` (greeting must succeed)
        self.exc = exc if exc is not None else RuntimeError("provider down")
        self.error_ctxs: list[dict] = []
        self.handler_raises: BaseException | None = None

    def _completion(self, **kwargs):
        if kwargs.get("stream"):
            if self.fail == "first":
                raise self.exc
            if self.fail == "mid":

                def _gen():
                    yield _stream_chunk("partial")
                    raise self.exc

                return _gen()
            return iter([_stream_chunk("ok")])
        if self.fail == "sync":
            raise self.exc
        return _fake_response(json.dumps({"message": "ok", "reasoning": ""}))

    def _on_error(self, ctx):
        self.error_ctxs.append(dict(ctx))
        if self.handler_raises is not None:
            raise self.handler_raises
        return {}

    def start(self, api):
        """Start the conversation on a healthy LLM, then arm the failure."""
        cid, _ = api.start_conversation()
        self.fail = self._mode
        return cid

    def make_api(self):
        from fsm_llm.handlers import HandlerTiming

        api = API.from_definition(_greeter_fsm(False), model="gpt-4o", api_key="test")
        builder = api.create_handler("err_spy").at(HandlerTiming.ERROR)
        if self.handler_raises is not None:
            builder = builder.critical()
        api.register_handler(builder.do(self._on_error))
        return api

    def patches(self):
        return (
            patch("fsm_llm.llm.completion", side_effect=self._completion),
            patch(
                "fsm_llm.llm.get_supported_openai_params",
                return_value=["response_format"],
            ),
        )


class TestErrorHandlersFireForFSMErrorAndStream:
    """ERROR handlers run for LLMResponseError and on the stream path only."""

    def test_converse_llm_outage_fires_error_handler_once_and_reraises_fsmerror(self):
        from fsm_llm.definitions import FSMError, LLMResponseError

        h = _ErrorHandlerHarness("sync")
        p1, p2 = h.patches()
        with p1, p2:
            api = h.make_api()
            cid = h.start(api)
            with pytest.raises(FSMError) as ei:
                api.converse("hi", cid)
        assert isinstance(ei.value, LLMResponseError)
        assert len(h.error_ctxs) == 1
        assert "provider down" in h.error_ctxs[0]["_error"]
        assert "_traceback" in h.error_ctxs[0]

    def test_converse_failure_still_rolls_back_the_user_message(self):
        from fsm_llm.definitions import FSMError

        h = _ErrorHandlerHarness("sync")
        p1, p2 = h.patches()
        with p1, p2:
            api = h.make_api()
            cid = h.start(api)
            before = len(api.get_conversation_history(cid))
            with pytest.raises(FSMError):
                api.converse("hi", cid)
            assert len(api.get_conversation_history(cid)) == before

    def test_critical_handler_failure_replaces_fsmerror_with_original_as_cause(self):
        from fsm_llm.definitions import LLMResponseError
        from fsm_llm.handlers import HandlerExecutionError

        h = _ErrorHandlerHarness("sync")
        h.handler_raises = ValueError("handler exploded")
        p1, p2 = h.patches()
        with p1, p2:
            api = h.make_api()
            cid = h.start(api)
            with pytest.raises(HandlerExecutionError) as ei:
                api.converse("hi", cid)
        assert "handler exploded" in str(ei.value)
        assert isinstance(ei.value.__cause__, LLMResponseError)

    def test_stream_failure_at_first_token_fires_error_handler(self):
        from fsm_llm.definitions import FSMError

        h = _ErrorHandlerHarness("first")
        p1, p2 = h.patches()
        with p1, p2:
            api = h.make_api()
            cid = h.start(api)
            with pytest.raises(FSMError):
                list(api.converse_stream("hi", cid))
        assert len(h.error_ctxs) == 1
        assert "provider down" in h.error_ctxs[0]["_error"]

    def test_stream_failure_mid_stream_fires_error_handler(self):
        h = _ErrorHandlerHarness("mid")
        p1, p2 = h.patches()
        got: list[str] = []
        with p1, p2:
            api = h.make_api()
            cid = h.start(api)
            try:
                for tok in api.converse_stream("hi", cid):
                    got.append(tok)
            except Exception:
                pass
        assert got[:1] == ["partial"]
        assert len(h.error_ctxs) == 1

    def test_stream_critical_handler_failure_replaces_error_with_cause(self):
        from fsm_llm.handlers import HandlerExecutionError

        h = _ErrorHandlerHarness("first")
        h.handler_raises = ValueError("handler exploded")
        p1, p2 = h.patches()
        with p1, p2:
            api = h.make_api()
            cid = h.start(api)
            with pytest.raises(HandlerExecutionError) as ei:
                list(api.converse_stream("hi", cid))
        assert ei.value.__cause__ is not None

    @pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
    def test_interrupts_fire_no_error_handler_sync_and_stream(self, exc_type):
        h = _ErrorHandlerHarness("sync", exc=exc_type())
        p1, p2 = h.patches()
        with p1, p2:
            api = h.make_api()
            cid = h.start(api)
            with pytest.raises(exc_type):
                api.converse("hi", cid)
            h.fail = "first"
            with pytest.raises(exc_type):
                list(api.converse_stream("hi", cid))
        assert h.error_ctxs == []

    def test_generator_close_fires_no_error_handler_and_releases_lock(self):
        h = _ErrorHandlerHarness(None)
        p1, p2 = h.patches()
        with p1, p2:
            api = h.make_api()
            cid = h.start(api)
            gen = api.converse_stream("hi", cid)
            next(gen)
            gen.close()
            # lock released: a following turn is accepted, not "already processed"
            assert api.converse("again", cid)
        assert h.error_ctxs == []

    def test_healthy_turns_fire_no_error_handler(self):
        h = _ErrorHandlerHarness(None)
        p1, p2 = h.patches()
        with p1, p2:
            api = h.make_api()
            cid = h.start(api)
            api.converse("hi", cid)
            list(api.converse_stream("hi again", cid))
        assert h.error_ctxs == []


def _collect_fsm() -> dict:
    """ra05: Pass 1 commits `name` and moves a -> b, Pass 2 can then fail."""
    return {
        "name": "Collect",
        "description": "d",
        "initial_state": "a",
        "persona": "p",
        "states": {
            "a": {
                "id": "a",
                "description": "d",
                "purpose": "collect",
                "response_instructions": "Reply.",
                "field_extractions": [
                    {
                        "field_name": "name",
                        "field_type": "str",
                        "extraction_instructions": "n",
                        "required": True,
                    }
                ],
                "transitions": [
                    {
                        "target_state": "b",
                        "description": "have name",
                        "priority": 1,
                        "conditions": [
                            {
                                "description": "n",
                                "requires_context_keys": ["name"],
                                "logic": {"!=": [{"var": "name"}, None]},
                            }
                        ],
                    }
                ],
            },
            "b": {
                "id": "b",
                "description": "d",
                "purpose": "p",
                "response_instructions": "Reply b.",
                "transitions": [
                    {"target_state": "b", "description": "loop", "priority": 100},
                    {
                        "target_state": "z",
                        "description": "end",
                        "priority": 1,
                        "conditions": [
                            {
                                "description": "never",
                                "requires_context_keys": ["zzz"],
                                "logic": {"==": [{"var": "zzz"}, 1]},
                            }
                        ],
                    },
                ],
            },
            "z": {
                "id": "z",
                "description": "d",
                "purpose": "p",
                "response_instructions": "bye",
                "transitions": [],
            },
        },
    }


class TestErrorHandlerReturnIsNotMerged:
    def test_returned_delta_from_error_handler_is_dropped_sync_and_stream(self):
        """ra05 port: Pass-2 failure after a committed Pass 1."""
        from fsm_llm.definitions import FSMError

        fired: list[int] = []
        with _Session(_collect_fsm()) as s:

            def on_error(ctx):
                fired.append(1)
                return {"handler_marker": "set-by-ERROR-handler"}

            s.on(HandlerTiming.ERROR, on_error)
            s.provider.fail_pass2 = True
            with pytest.raises(FSMError):
                s.api.converse("I am Bob", s.cid)
            assert len(fired) == 1
            data = s.api.get_data(s.cid)
            assert "handler_marker" not in data
            # the turn was rolled back: nothing from Pass 1 survived
            assert "name" not in data
            assert s.api.get_current_state(s.cid) == "a"

            fired.clear()
            s.provider.fail_pass2 = False
            s.provider.fail_stream = True
            # advance to b first so the stream turn has a non-terminal state
            s.api.converse("I am Bob", s.cid)
            assert s.api.get_current_state(s.cid) == "b"
            with pytest.raises(FSMError):
                list(s.api.converse_stream("hi again", s.cid))
            assert len(fired) == 1
            assert "handler_marker" not in s.api.get_data(s.cid)

    def test_non_error_handler_delta_is_still_merged(self):
        """Vacuity guard: only ERROR timing stopped merging."""
        with _Session(_reentry_fsm()) as s:
            s.on(HandlerTiming.PRE_PROCESSING, lambda ctx: {"pre_marker": 1})
            s.api.converse("hello", s.cid)
            assert s.api.get_data(s.cid).get("pre_marker") == 1


def _three_field_advance_fsm() -> dict:
    fields = ["alpha", "beta", "gamma"]
    return {
        "name": "ThreeField",
        "description": "advance-turn deepcopy seam",
        "version": "4.1",
        "initial_state": "collect",
        "persona": "x",
        "states": {
            "collect": {
                "id": "collect",
                "description": "collect three fields",
                "purpose": "collect",
                "required_context_keys": fields,
                "field_extractions": [
                    {
                        "field_name": f,
                        "field_type": "str",
                        "extraction_instructions": f"the {f}",
                        "required": True,
                    }
                    for f in fields
                ],
                "response_instructions": "Reply briefly.",
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "all collected",
                        "priority": 10,
                        "conditions": [
                            {
                                "description": "all present",
                                "requires_context_keys": fields,
                                "logic": {"!=": [{"var": "alpha"}, None]},
                            }
                        ],
                    }
                ],
            },
            "done": {
                "id": "done",
                "description": "end",
                "purpose": "end",
                "response_instructions": "Bye.",
                "transitions": [],
            },
        },
    }


class _DeepcopyCounter:
    """Counts TOP-LEVEL ``copy.deepcopy`` calls (the recursive internals of one
    call re-enter the patched module global and are not counted)."""

    def __init__(self):
        import copy

        self._real = copy.deepcopy
        self.count = 0
        self._depth = 0

    def __call__(self, obj, memo=None, _nil=[]):  # noqa: B006
        if self._depth == 0:
            self.count += 1
        self._depth += 1
        try:
            return self._real(obj, memo) if memo is not None else self._real(obj)
        finally:
            self._depth -= 1


class TestZeroHandlerTurnSkipsHandlerCopies:
    def test_advance_turn_deepcopy_count_drops_with_no_handlers(self):
        counter = _DeepcopyCounter()
        with _Prov(_three_field_advance_fsm()) as p:
            # _Prov registers a CONTEXT_UPDATE spy; drop it so the FSM has
            # genuinely zero handlers.
            p.api.fsm_manager.handler_system.handlers.clear()
            p.field = {"alpha": "a", "beta": "b", "gamma": "c"}
            with patch("copy.deepcopy", counter):
                p.api.converse("a b c", p.cid)
            assert p.api.get_current_state(p.cid) == "done"
        # Baseline before the fix: 14 (10 handler-timing/transition copies plus
        # the D-012 pre-turn snapshots). The snapshots must remain.
        assert 1 <= counter.count <= 5, counter.count

    def test_non_copyable_value_no_longer_breaks_start_conversation(self):
        import threading

        with _Prov(_correction_fsm()) as p:
            p.api.fsm_manager.handler_system.handlers.clear()
            cid, _ = p.api.start_conversation({"lock": threading.Lock()})
            assert p.api.get_current_state(cid) == "profile"

    def test_registered_handler_still_runs_and_sees_context(self):
        seen: list[dict] = []
        with _Prov(_correction_fsm()) as p:
            p.api.register_handler(
                p.api.create_handler("pre")
                .at(HandlerTiming.PRE_PROCESSING)
                .do(lambda ctx: seen.append(dict(ctx)) or {"from_handler": 1})
            )
            p.api.converse("hello", p.cid)
            assert seen
            assert p.api.get_data(p.cid).get("from_handler") == 1

    def test_handlers_at_filters_by_timing(self):
        with _Prov(_correction_fsm()) as p:
            hs = p.api.fsm_manager.handler_system
            assert hs.handlers_at(HandlerTiming.PRE_PROCESSING) == []
            handler = (
                p.api.create_handler("only_post")
                .at(HandlerTiming.POST_PROCESSING)
                .do(lambda ctx: {})
            )
            p.api.register_handler(handler)
            assert hs.handlers_at(HandlerTiming.PRE_PROCESSING) == []
            assert hs.handlers_at(HandlerTiming.POST_PROCESSING) == [handler]


_DUCK_FSM = {
    "name": "S",
    "description": "d",
    "version": "4.1",
    "initial_state": "a",
    "persona": "p",
    "states": {
        "a": {
            "id": "a",
            "description": "d",
            "purpose": "p",
            "response_instructions": "r",
            "transitions": [
                {
                    "target_state": "b",
                    "description": "never",
                    "conditions": [
                        {"description": "never", "logic": {"==": [1, 2]}},
                    ],
                }
            ],
        },
        "b": {
            "id": "b",
            "description": "d",
            "purpose": "p",
            "response_instructions": "r",
            "transitions": [],
        },
    },
}


class _DuckHandlerSystem:
    """A handler system with only ``execute_handlers`` (no ``handlers_at``)."""

    def __init__(self):
        self.calls = 0

    def register_handler(self, handler):
        pass

    def close(self):
        pass

    def execute_handlers(
        self,
        timing,
        current_state,
        target_state,
        context,
        updated_keys=None,
        error_context=None,
    ):
        self.calls += 1
        return {}


def _manager_with(handler_system):
    from fsm_llm.definitions import FSMDefinition
    from fsm_llm.fsm import FSMManager

    definition = FSMDefinition(**_DUCK_FSM)
    with (
        patch(
            "fsm_llm.llm.completion",
            return_value=_fake_response(json.dumps({"message": "ok"})),
        ),
        patch(
            "fsm_llm.llm.get_supported_openai_params",
            return_value=["response_format"],
        ),
    ):
        llm = LiteLLMInterface(model="gpt-4o", api_key="x")
        manager = FSMManager(
            fsm_loader=lambda fid: definition,
            llm_interface=llm,
            handler_system=handler_system,
        )
        cid, _ = manager.start_conversation("S")
        response = manager.process_message(cid, "hello")
    return response


class TestHandlersAtIsOptional:
    def test_duck_typed_handler_system_starts_and_converses(self):
        duck = _DuckHandlerSystem()
        response = _manager_with(duck)
        assert response == "ok"
        assert duck.calls > 0

    def test_a_spec_mock_handler_system_keeps_the_old_path(self):
        from unittest.mock import Mock

        from fsm_llm.handlers import HandlerSystem

        system = Mock(spec=HandlerSystem)
        system.execute_handlers.return_value = {}
        response = _manager_with(system)
        assert response == "ok"
        assert system.execute_handlers.call_count > 0

    def test_a_non_callable_handlers_at_takes_the_old_path(self):
        duck = _DuckHandlerSystem()
        duck.handlers_at = None  # type: ignore[attr-defined]
        response = _manager_with(duck)
        assert response == "ok"
        assert duck.calls > 0


class _C1SortProbe:
    """A handler whose ``priority`` read can block once, inside a sort.

    ``HandlerSystem.register_handler`` reads ``priority`` from its sort key.
    Armed, the first read signals ``in_sort`` and waits for ``release``, so a
    reader thread observes ``handlers`` from inside the registration's sort.
    """

    name = "c1_probe"
    timings = None

    def __init__(self) -> None:
        self._in_sort: threading.Event | None = None
        self._release: threading.Event | None = None

    def arm(self, in_sort: threading.Event, release: threading.Event) -> None:
        self._in_sort, self._release = in_sort, release

    @property
    def priority(self) -> int:
        in_sort, release = self._in_sort, self._release
        if in_sort is not None and release is not None:
            self._in_sort = self._release = None
            in_sort.set()
            release.wait(2.0)
        return 100

    def should_execute(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        return {}


class _C2Handler(BaseHandler):
    """Always runs; ``action(context)`` supplies the returned delta."""

    def __init__(self, name: str, priority: int, action: Any) -> None:
        super().__init__(name=name, priority=priority)
        self._action = action

    def should_execute(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        return self._action(context)


class _C2CopyOnce:
    """Deep-copyable exactly once (the second ``deepcopy`` raises)."""

    copies = 0

    def __deepcopy__(self, memo: dict[int, Any]) -> _C2CopyOnce:
        type(self).copies += 1
        if type(self).copies > 1:
            raise TypeError("cannot copy twice")
        return _C2CopyOnce()


def _c3_api(delta_fn: Any, *, state: str = "start") -> tuple[API, str]:
    """An API whose one PRE_PROCESSING handler, in ``state``, returns
    ``delta_fn(context)``. The conversation is advanced to ``state`` first."""
    api = API.from_definition(_c3_fsm(), llm_interface=_mock_llm())
    api.register_handler(
        create_handler("c3_handler")
        .at(HandlerTiming.PRE_PROCESSING)
        .on_state(state)
        .do(delta_fn)
    )
    conv_id, _ = api.start_conversation(initial_context={"note": "keep me"})
    if state != "start":
        api.converse("advance", conv_id)
        assert api.get_current_state(conv_id) == state
    return api, conv_id


class TestStep12C1C3:
    def test_c1_concurrent_register_never_yields_empty_view(self):
        # Straddles the race window: the reader runs while registration is
        # inside its sort. The old in-place ``list.sort`` empties the list for
        # the duration of the sort, so the reader saw zero handlers.
        for _ in range(20):
            system = HandlerSystem()
            probe = _C1SortProbe()
            system.register_handler(probe)
            system.register_handler(_C2Handler("second", 200, lambda ctx: {}))
            in_sort, release = threading.Event(), threading.Event()
            probe.arm(in_sort, release)
            views: list[int] = []

            def _reader(in_sort=in_sort, release=release, system=system, views=views):
                entered = in_sort.wait(2.0)
                views.append(
                    len(system.handlers_at(HandlerTiming.PRE_PROCESSING))
                    if entered
                    else -1
                )
                release.set()

            reader = threading.Thread(target=_reader, daemon=True)
            reader.start()
            system.register_handler(_C2Handler("third", 300, lambda ctx: {}))
            reader.join(5.0)
            assert views == [2]
            assert len(system.handlers) == 3

    def test_c1_register_rebinds_list(self):
        system = HandlerSystem()
        first = _C2Handler("first", 50, lambda ctx: {})
        system.register_handler(first)
        before = system.handlers
        second = _C2Handler("second", 10, lambda ctx: {})
        system.register_handler(second)
        assert system.handlers is not before
        assert before == [first]
        assert system.handlers == [second, first]

    def test_c2_straggler_write_not_seen_by_next_handler(self):
        may_write, wrote = threading.Event(), threading.Event()

        def _straggler(context: dict[str, Any]) -> dict[str, Any]:
            may_write.wait(5.0)
            context["leak"] = True
            wrote.set()
            return {"straggler": True}

        def _reader(context: dict[str, Any]) -> dict[str, Any]:
            may_write.set()
            wrote.wait(5.0)
            return {"saw_leak": "leak" in context}

        system = HandlerSystem(handler_timeout=0.5)
        system.register_handler(_C2Handler("straggler", 1, _straggler))
        system.register_handler(_C2Handler("reader", 2, _reader))
        result = system.execute_handlers(
            timing=HandlerTiming.PRE_PROCESSING,
            current_state="start",
            target_state=None,
            context={"x": 1},
        )
        assert wrote.wait(5.0)
        assert result == {"saw_leak": False}

    def test_c2_timeout_thread_is_daemon(self):
        seen: list[tuple[bool, bool]] = []

        def _record(context: dict[str, Any]) -> dict[str, Any]:
            current = threading.current_thread()
            seen.append((current is not threading.main_thread(), current.daemon))
            return {"ran": True}

        system = HandlerSystem(handler_timeout=5.0)
        system.register_handler(_C2Handler("record", 1, _record))
        result = system.execute_handlers(
            timing=HandlerTiming.PRE_PROCESSING,
            current_state="start",
            target_state=None,
            context={},
        )
        assert result == {"ran": True}
        assert seen == [(True, True)]

    def test_c2_uncopyable_context_falls_back_to_shallow_copy(self):
        _C2CopyOnce.copies = 0

        def _write(context: dict[str, Any]) -> dict[str, Any]:
            context["scratch"] = True
            return {"saw_value": isinstance(context.get("value"), _C2CopyOnce)}

        def _after(context: dict[str, Any]) -> dict[str, Any]:
            return {"saw_scratch": "scratch" in context}

        system = HandlerSystem(handler_timeout=5.0)
        system.register_handler(_C2Handler("write", 1, _write))
        system.register_handler(_C2Handler("after", 2, _after))
        result = system.execute_handlers(
            timing=HandlerTiming.PRE_PROCESSING,
            current_state="start",
            target_state=None,
            context={"value": _C2CopyOnce()},
        )
        # Step 12.1 (D-049): a timed handler that completes has its in-place
        # writes adopted, so the shallow-copy fallback's top-level write is
        # visible to the next handler, as it is without a timeout.
        assert result == {"saw_value": True, "saw_scratch": True}

    def test_c3_handler_cannot_overwrite_conversation_id(self):
        api, conv_id = _c3_api(
            lambda ctx: {
                "_conversation_id": "hijacked",
                "_fsm_id": "other_fsm",
                "normal": "written",
            }
        )
        api.converse("hello", conv_id)
        data = _raw_data(api, conv_id)
        assert data["_conversation_id"] == conv_id
        assert data["_fsm_id"] != "other_fsm"
        assert data["normal"] == "written"

    def test_c3_handler_cannot_delete_current_state(self):
        api, conv_id = _c3_api(
            lambda ctx: {"_current_state": None, "_previous_state": "forged"},
            state="mid",
        )
        api.converse("again", conv_id)
        data = _raw_data(api, conv_id)
        assert data["_current_state"] == "mid"
        assert data["_previous_state"] == "start"

    def test_c3_handler_owned_internal_key_still_merges(self):
        api, conv_id = _c3_api(
            lambda ctx: {"_replan_count": ctx.get("_replan_count", 0) + 1}
        )
        api.converse("hello", conv_id)
        assert _raw_data(api, conv_id)["_replan_count"] == 1

    def test_c3_none_delete_of_normal_key_unchanged(self):
        api, conv_id = _c3_api(lambda ctx: {"note": None})
        assert _raw_data(api, conv_id)["note"] == "keep me"
        api.converse("hello", conv_id)
        assert "note" not in _raw_data(api, conv_id)

    def test_c3_reserved_set_is_internal_and_closed(self):
        from fsm_llm.constants import RESERVED_CONTEXT_KEYS, has_internal_prefix

        assert "_transition_classification_result" not in RESERVED_CONTEXT_KEYS
        assert all(has_internal_prefix(key) for key in RESERVED_CONTEXT_KEYS)
        assert "_replan_count" not in RESERVED_CONTEXT_KEYS
        assert isinstance(RESERVED_CONTEXT_KEYS, frozenset)


def _c2_run(system: HandlerSystem, context: dict[str, Any]) -> dict[str, Any]:
    return system.execute_handlers(
        timing=HandlerTiming.PRE_PROCESSING,
        current_state="start",
        target_state=None,
        context=context,
    )


class TestStep12_1:
    """Review-iter-1 W3 (straggler bound) and W4 (in-place write parity)."""

    def test_c2_straggler_threads_are_capped(self):
        from fsm_llm.constants import MAX_TIMED_HANDLER_STRAGGLERS

        release = threading.Event()
        started: list[threading.Thread] = []

        def _hang(context: dict[str, Any]) -> dict[str, Any]:
            started.append(threading.current_thread())
            release.wait(30.0)
            return {}

        system = HandlerSystem(handler_timeout=0.02)
        system.register_handler(_C2Handler("hang", 1, _hang))
        base = threading.active_count()
        peak = 0
        try:
            with _c_log_capture() as logs:
                for _ in range(200):
                    assert _c2_run(system, {"k": 1}) == {}
                    peak = max(peak, threading.active_count() - base)
            assert peak <= MAX_TIMED_HANDLER_STRAGGLERS + 1
            assert len(started) <= MAX_TIMED_HANDLER_STRAGGLERS + 1
            assert any(
                m.startswith("WARNING|") and str(MAX_TIMED_HANDLER_STRAGGLERS) in m
                for m in logs
            )
            # A refused call fails the timeout way, honouring handler_error_mode.
            from fsm_llm.handlers import HandlerExecutionError

            strict = HandlerSystem(handler_error_mode="raise", handler_timeout=0.02)
            strict.register_handler(_C2Handler("hang", 1, _hang))
            errors = []
            for _ in range(MAX_TIMED_HANDLER_STRAGGLERS + 1):
                with pytest.raises(HandlerExecutionError) as info:
                    _c2_run(strict, {})
                errors.append(info.value.original_error)
            assert all(isinstance(err, TimeoutError) for err in errors)
            assert "straggler" in str(errors[-1])
            assert len(started) <= 2 * MAX_TIMED_HANDLER_STRAGGLERS + 1
        finally:
            release.set()
            for worker in started:
                worker.join(5.0)
        # Once the stragglers finish, timed handlers run again.
        system_ok = HandlerSystem(handler_timeout=5.0)
        system_ok.register_handler(_C2Handler("ok", 1, lambda ctx: {"ok": True}))
        assert _c2_run(system_ok, {}) == {"ok": True}
        assert _c2_run(system, {}) == {}

    def test_c2_completed_timed_handler_in_place_writes_visible(self):
        def _a(context: dict[str, Any]) -> dict[str, Any]:
            context["shared_scratch"] = "from_a"
            context["nested"]["x"] = 2
            del context["gone"]
            return {}

        def _b(context: dict[str, Any]) -> dict[str, Any]:
            return {
                "b_saw": context.get("shared_scratch"),
                "b_nested": context["nested"]["x"],
                "b_gone": "gone" in context,
            }

        results = []
        for timeout in (None, 5.0):
            system = HandlerSystem(handler_timeout=timeout)
            system.register_handler(_C2Handler("a", 1, _a))
            system.register_handler(_C2Handler("b", 2, _b))
            original = {"nested": {"x": 1}, "gone": True}
            results.append(_c2_run(system, original))
            # The caller's dict is never mutated; only the returned delta leaves.
            assert original == {"nested": {"x": 1}, "gone": True}
        untimed, timed = results
        assert timed == untimed == {"b_saw": "from_a", "b_nested": 2, "b_gone": False}

    def test_c2_timed_out_handler_writes_still_invisible(self):
        release, wrote = threading.Event(), threading.Event()

        def _slow(context: dict[str, Any]) -> dict[str, Any]:
            context["early"] = True  # written before the timeout
            release.wait(5.0)
            context["late"] = True
            wrote.set()
            return {"slow": True}

        def _reader(context: dict[str, Any]) -> dict[str, Any]:
            release.set()
            wrote.wait(5.0)
            return {"saw": sorted(k for k in ("early", "late") if k in context)}

        system = HandlerSystem(handler_timeout=0.2)
        system.register_handler(_C2Handler("slow", 1, _slow))
        system.register_handler(_C2Handler("reader", 2, _reader))
        assert _c2_run(system, {"x": 1}) == {"saw": []}
        assert wrote.wait(5.0)


class _C6Handler(BaseHandler):
    """Runs everywhere and returns whatever it was given."""

    def __init__(self, name: str, priority: Any, result: Any = None) -> None:
        super().__init__(name=name, priority=priority)
        self._result = result

    def should_execute(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def execute(self, context: dict[str, Any]) -> Any:
        return self._result


def _c9_fsm_data(name: str = "C9", end_transitions: Any = None) -> dict[str, Any]:
    return {
        "name": name,
        "description": "C9",
        "initial_state": "start",
        "states": {
            "start": {
                "description": "Start",
                "purpose": "Begin",
                "transitions": [{"target_state": "end", "description": "Finish"}],
            },
            "end": {
                "description": "End",
                "purpose": "Stop",
                "transitions": end_transitions,
            },
        },
    }


def _c8_mock_api(converse_effect: Any) -> MagicMock:
    api = MagicMock()
    api.start_conversation.return_value = ("conv-1", "Hello!")
    api.has_conversation_ended.side_effect = [False, True]
    api.converse.side_effect = converse_effect
    api.get_data.return_value = {}
    return api


def _c8_run(api: MagicMock) -> Any:
    import os

    from fsm_llm.runner import main

    with (
        patch.dict(os.environ, {"LLM_MODEL": "test-model"}, clear=True),
        patch("fsm_llm.runner.load_env"),
        patch("fsm_llm.runner.API.from_file", return_value=api),
        patch("fsm_llm.runner.setup_file_logging"),
        patch("builtins.input", return_value="hi"),
    ):
        return main("/tmp/c8.json", 5, 1000)


class _C10RaisingFinder:
    """A meta-path finder that fails the import of one (dotted) module."""

    def __init__(self, target: str, missing: str) -> None:
        self.target, self.missing = target, missing

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname == self.target:
            raise ModuleNotFoundError(
                f"No module named {self.missing!r}", name=self.missing
            )
        return None


@contextmanager
def _c10_failing_import(missing: str):
    import sys

    import fsm_llm.workflows  # noqa: F401  (ensure the key exists to pop)

    saved = sys.modules.pop("fsm_llm.workflows")
    finder = _C10RaisingFinder("fsm_llm.workflows", missing)
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        sys.modules["fsm_llm.workflows"] = saved


def _c11_sub_fsm() -> FSMDefinition:
    return FSMDefinition(
        name="c11_sub",
        description="C11 sub FSM with a secret-looking default",
        initial_state="ask",
        persona="internal persona text",
        states={
            "ask": State(
                id="ask",
                description="Ask",
                purpose="Ask",
                response_instructions="Respond",
                transitions=[Transition(target_state="done", description="Done")],
            ),
            "done": State(id="done", description="Done", purpose="Done"),
        },
    )


class TestStep14C6C12:
    # -- C6 ---------------------------------------------------------------
    def test_c6_non_dict_handler_return_warns(self):
        system = HandlerSystem()
        system.register_handler(_C6Handler("listy", 10, result=["x"]))
        with _c_log_capture() as messages:
            out = system.execute_handlers(HandlerTiming.PRE_PROCESSING, "s", None, {})
        assert out == {}
        warnings = [m for m in messages if m.startswith("WARNING|")]
        assert any("listy" in m and "list" in m for m in warnings), messages

    def test_c6_unorderable_priority_rejected_at_registration(self):
        system = HandlerSystem()
        with pytest.raises(TypeError, match="priority"):
            system.register_handler(_C6Handler("stringy", "high"))
        with pytest.raises(ValueError, match="priority"):
            system.register_handler(_C6Handler("nan", float("nan")))
        assert system.handlers == []
        ok = _C6Handler("ok", 5)
        system.register_handler(ok)
        system.register_handler(_C6Handler("ok2", 1.5))
        assert [h.name for h in system.handlers] == ["ok2", "ok"]

    def test_c6_condition_lambda_error_wrapped_once(self):
        from fsm_llm.handlers import HandlerExecutionError

        def _boom(*args: Any) -> bool:
            raise ValueError("cond boom")

        system = HandlerSystem(handler_error_mode="raise")
        system.register_handler(create_handler("cond").when(_boom).do(lambda c: {}))
        with pytest.raises(HandlerExecutionError) as exc_info:
            system.execute_handlers(HandlerTiming.PRE_PROCESSING, "s", None, {})
        err = exc_info.value
        assert not isinstance(err.original_error, HandlerExecutionError)
        assert str(err).count("Error in handler") == 1
        assert "cond boom" in str(err)

    def test_c6_handler_execution_error_pickles(self):
        import pickle

        from fsm_llm.handlers import HandlerExecutionError

        err = HandlerExecutionError("h", ValueError("bad"))
        err.partial_context = {"a": 1}
        back = pickle.loads(pickle.dumps(err))
        assert type(back) is HandlerExecutionError
        assert back.handler_name == "h"
        assert isinstance(back.original_error, ValueError)
        assert back.original_error.args == ("bad",)
        assert back.partial_context == {"a": 1}
        assert str(back) == str(err)

    # -- C7 ---------------------------------------------------------------
    def test_c7_json_log_never_str_of_unknown_objects(self):
        import datetime as dt

        from fsm_llm.logging import _record_to_json

        class _Creds:
            def __str__(self) -> str:
                return "SECRET-TOKEN-VALUE"

        record = {
            "extra": {
                "creds": _Creds(),
                "when": dt.datetime(2026, 9, 22, 1, 2, 3),
            },
            "time": dt.datetime(2026, 9, 22),
            "level": MagicMock(name="INFO"),
            "message": "m",
            "name": "mod",
            "function": "f",
            "line": 1,
            "exception": None,
        }
        record["level"].name = "INFO"
        line = _record_to_json(record)
        assert "SECRET-TOKEN-VALUE" not in line
        entry = json.loads(line)
        assert "_Creds" in entry["creds"]
        assert entry["when"] == "2026-09-22T01:02:03"

    # -- C8 ---------------------------------------------------------------
    def test_c8_turn_error_exit_code_is_1(self):
        api = _c8_mock_api(RuntimeError("turn failed"))
        assert _c8_run(api) == 1
        api.end_conversation.assert_called_once_with("conv-1")

    def test_c8_ctrl_c_mid_converse_exits_130_after_cleanup(self):
        api = _c8_mock_api(KeyboardInterrupt())
        assert _c8_run(api) == 130
        api.end_conversation.assert_called_once_with("conv-1")

    def test_c8_cli_entry_catches_keyboard_interrupt(self):
        import sys

        from fsm_llm.__main__ import main_cli

        with (
            patch.object(sys, "argv", ["fsm-llm", "--fsm", "x.json"]),
            patch("fsm_llm.runner.main", side_effect=KeyboardInterrupt()),
        ):
            assert main_cli() == 130

    def test_c8_validate_report_on_stdout(self, tmp_path, capsys):
        from fsm_llm.validator import main

        path = tmp_path / "c8.json"
        path.write_text(json.dumps(_c9_fsm_data(end_transitions=[])), encoding="utf-8")
        code = main(str(path))
        report = json.loads(capsys.readouterr().out)
        assert report["is_valid"] is (code == 0)
        assert report["fsm_name"] == "C9"

    def test_c8_visualize_diagram_on_stdout(self, tmp_path, capsys):
        from fsm_llm.visualizer import main

        path = tmp_path / "c8.json"
        path.write_text(
            json.dumps(_c9_fsm_data(name="C8Viz", end_transitions=[])), encoding="utf-8"
        )
        assert main(str(path)) == 0
        out = capsys.readouterr().out
        assert "C8Viz" in out and "─" in out

    # -- C9 ---------------------------------------------------------------
    @pytest.mark.parametrize("style", ["full", "compact", "minimal"])
    def test_c9_null_transitions_render(self, style):
        from fsm_llm.visualizer import visualize_fsm_ascii

        output = visualize_fsm_ascii(_c9_fsm_data(), style)
        assert "Could not generate diagram" not in output
        assert "end" in output

    @pytest.mark.parametrize("style", ["full", "compact", "minimal"])
    def test_c9_newline_in_name_keeps_header_one_line(self, style):
        from fsm_llm.visualizer import visualize_fsm_ascii

        output = visualize_fsm_ascii(
            _c9_fsm_data(name="Evil\nName\r\x0bX", end_transitions=[]), style
        )
        lines = output.split("\n")
        if style == "minimal":
            assert lines[0] == "FSM: Evil Name X"
        else:
            assert "Evil Name X" in lines[1]
            assert len(lines[1]) == len(lines[0]) == 62

    # -- C10 --------------------------------------------------------------
    def test_c10_inner_import_error_not_rewritten(self):
        from fsm_llm import get_workflows

        with _c10_failing_import("yaml"):
            with pytest.raises(ImportError) as exc_info:
                get_workflows()
        assert exc_info.value.name == "yaml"
        assert "workflows extra" not in str(exc_info.value)

    def test_c10_missing_package_still_gets_install_hint(self):
        from fsm_llm import get_workflows

        with _c10_failing_import("fsm_llm.workflows"):
            with pytest.raises(ImportError, match=r"pip install -e \."):
                get_workflows()

    def test_c10_disable_warnings_scoped_to_fsm_llm(self):
        import warnings

        from fsm_llm import disable_warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            disable_warnings()
            # D-007: the subpackages are part of the framework now, so
            # fsm_llm.agents.react is suppressed; a lookalike sibling
            # (fsm_llm_contrib) must still not be (no prefix bleed).
            for module in (
                "fsm_llm",
                "fsm_llm.pipeline",
                "fsm_llm.agents.react",
                "fsm_llm_contrib.x",
            ):
                warnings.warn_explicit(
                    f"from {module}", UserWarning, "f.py", 1, module=module
                )
        assert [str(w.message) for w in caught] == ["from fsm_llm_contrib.x"]

    # -- C11 --------------------------------------------------------------
    def test_c11_sub_conversation_summary_uses_fsm_name(self):
        api = API.from_definition(_c3_fsm(), llm_interface=_mock_llm())
        conv_id, _ = api.start_conversation()
        api.push_fsm(conv_id, _c11_sub_fsm(), preserve_history=True)
        api.pop_fsm(conv_id)
        data = api.fsm_manager.instances[conv_id].context.data
        assert data["_sub_conversation_summary"]["fsm_type"] == "c11_sub"

    # -- C12 --------------------------------------------------------------
    def test_c12_lock_timeout_refuses_cleanup(self):
        api = API.from_definition(_c3_fsm(), llm_interface=_mock_llm())
        conv_id, _ = api.start_conversation()
        manager = api.fsm_manager
        busy = MagicMock()
        busy.acquire.return_value = False
        manager._conversation_locks[conv_id] = busy
        with _c_log_capture() as messages:
            with pytest.raises(FSMError, match="still"):
                manager.end_conversation(conv_id)
        assert conv_id in manager.instances
        busy.release.assert_not_called()
        assert any(m.startswith("ERROR|") and conv_id in m for m in messages)


class TestStep4LiveDictProbe:
    """D-039 reverts the P1-3 read-only probe: the uncopied-phase probe gets
    the caller's live dict again, per the pure-predicate contract (D-020)."""

    @pytest.mark.parametrize("handler_error_mode", ["continue", "raise"])
    @pytest.mark.parametrize(
        "condition",
        [
            lambda t, s, ts, ctx, uk: len(json.dumps(ctx)) > 0,
            lambda t, s, ts, ctx, uk: copy.deepcopy(ctx) == {"existing": 1},
            lambda t, s, ts, ctx, uk: isinstance(ctx, dict),
        ],
        ids=["json_dumps", "deepcopy", "isinstance_dict"],
    )
    def test_step_4_1_serialising_read_only_condition_fires(
        self, condition, handler_error_mode
    ):
        from fsm_llm.handlers import HandlerSystem, HandlerTiming, create_handler

        fired: list[int] = []

        def _action(ctx):
            fired.append(1)
            return {"logged": True}

        system = HandlerSystem(handler_error_mode=handler_error_mode)
        system.register_handler(create_handler("reader").when(condition).do(_action))
        context = {"existing": 1}
        result = system.execute_handlers(
            HandlerTiming.PRE_PROCESSING, "s", None, context
        )
        assert result == {"logged": True}
        assert fired == [1]
        assert context == {"existing": 1}


class TestLambdaHandlerNormalisesBareStrings:
    """F-HV-09: a bare ``str`` is one name for all six set arguments."""

    @staticmethod
    def _handler(**sets) -> LambdaHandler:
        args = {
            "states": set(),
            "target_states": set(),
            "required_keys": set(),
            "updated_keys": set(),
        }
        args.update(sets)
        return LambdaHandler(
            name="h",
            condition_lambdas=[],
            execution_lambda=lambda ctx: None,
            timings=None,
            **args,
        )

    def test_states_is_not_a_substring_test(self):
        h = self._handler(states="abc")
        t = HandlerTiming.PRE_PROCESSING
        assert h.should_execute(t, "abc", None, {}) is True
        assert h.should_execute(t, "b", None, {}) is False

    def test_target_states_is_not_a_substring_test(self):
        h = self._handler(target_states="abc")
        t = HandlerTiming.PRE_TRANSITION
        assert h.should_execute(t, "x", "abc", {}) is True
        assert h.should_execute(t, "x", "b", {}) is False

    def test_required_keys_is_not_a_character_set(self):
        h = self._handler(required_keys="name")
        t = HandlerTiming.PRE_PROCESSING
        assert h.should_execute(t, "x", None, {"name": 1}) is True
        assert h.should_execute(t, "x", None, {"n": 1, "a": 1, "m": 1, "e": 1}) is False

    def test_updated_keys_is_not_a_character_set(self):
        h = self._handler(updated_keys="name")
        t = HandlerTiming.CONTEXT_UPDATE
        assert h.should_execute(t, "x", None, {}, {"name"}) is True
        assert h.should_execute(t, "x", None, {}, {"n"}) is False

    def test_the_two_negative_sets_still_take_a_bare_string(self):
        h = self._handler(not_states="abc", not_target_states="abc")
        assert h.not_states == {"abc"}
        assert h.not_target_states == {"abc"}
