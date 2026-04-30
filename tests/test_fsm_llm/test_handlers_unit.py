"""
Comprehensive unit tests for the FSM-LLM handler system.

Tests cover: HandlerSystem, HandlerBuilder, LambdaHandler, BaseHandler,
HandlerExecutionError, priority ordering, context cascading, and metadata tracking.
"""

import pytest

from fsm_llm.handlers import (
    BaseHandler,
    HandlerBuilder,
    HandlerExecutionError,
    HandlerSystem,
    HandlerSystemError,
    HandlerTiming,
    LambdaHandler,
    create_handler,
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
    """Tests for HandlerSystem.__init__ with valid and invalid error_mode."""

    def test_init_continue_mode(self):
        hs = HandlerSystem(error_mode="continue")
        assert hs.error_mode == "continue"
        assert hs.handlers == []

    def test_init_raise_mode(self):
        hs = HandlerSystem(error_mode="raise")
        assert hs.error_mode == "raise"

    def test_init_default_mode_is_continue(self):
        hs = HandlerSystem()
        assert hs.error_mode == "continue"

    def test_init_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid error_mode"):
            HandlerSystem(error_mode="skip")

    def test_init_invalid_mode_nonsense(self):
        with pytest.raises(ValueError, match="Invalid error_mode"):
            HandlerSystem(error_mode="explode")


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
# 3-4. HandlerSystem._execute_handlers — continue & raise modes
# ══════════════════════════════════════════════════════════════


class TestHandlerSystemExecuteContinue:
    """execute_handlers with error_mode='continue'."""

    def test_handler_failure_continues_to_next(self):
        hs = HandlerSystem(error_mode="continue")
        hs.register_handler(FailingHandler(name="fail1", priority=10))
        hs.register_handler(AlwaysRunHandler(name="ok", priority=20, result={"x": 1}))

        result = hs._execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        assert result["x"] == 1

    def test_context_propagation_across_handlers(self):
        hs = HandlerSystem(error_mode="continue")
        hs.register_handler(AlwaysRunHandler(name="h1", priority=10, result={"a": 1}))
        hs.register_handler(AlwaysRunHandler(name="h2", priority=20, result={"b": 2}))

        result = hs._execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        assert result["a"] == 1
        assert result["b"] == 2


class TestHandlerSystemExecuteRaise:
    """execute_handlers with error_mode='raise'."""

    def test_handler_failure_raises_handler_execution_error(self):
        hs = HandlerSystem(error_mode="raise")
        hs.register_handler(FailingHandler(name="fail_handler"))

        with pytest.raises(HandlerExecutionError) as exc_info:
            hs._execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        assert exc_info.value.handler_name == "fail_handler"
        assert isinstance(exc_info.value.original_error, RuntimeError)

    def test_raise_mode_stops_at_first_failure(self):
        """After a raise, subsequent handlers should not execute."""
        hs = HandlerSystem(error_mode="raise")
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
            hs._execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
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
        result = hs._execute_handlers(
            HandlerTiming.PRE_PROCESSING, "s1", None, {"existing": True}
        )
        assert result == {}

    def test_no_matching_handlers_returns_empty_dict(self):
        """All handlers return should_execute=False."""
        hs = HandlerSystem()
        hs.register_handler(NeverRunHandler(name="never"))
        result = hs._execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        assert result == {}


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
        handler = create_handler("non_dict_handler").do(lambda ctx: "not a dict")
        with pytest.raises(HandlerExecutionError):
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
        result = hs._execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        # Only metadata, no other keys (None result is skipped)
        assert "new_key" not in result


# ══════════════════════════════════════════════════════════════
# 7. HandlerBuilder — fluent API
# ══════════════════════════════════════════════════════════════


class TestHandlerBuilderFluentAPI:
    """Fluent API methods: .at(), .when_state(), .on_states(), .with_priority(), .do(), .build()."""

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
        handler = create_handler("s1").when_state("greeting").do(lambda ctx: {})
        assert "greeting" in handler.states

    def test_on_state_multiple(self):
        handler = (
            create_handler("s2").when_state("greeting", "farewell").do(lambda ctx: {})
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
        handler = (
            create_handler("entry").when_state_entry("completed").do(lambda ctx: {})
        )
        assert HandlerTiming.POST_TRANSITION in handler.timings
        assert "completed" in handler.target_states

    def test_on_state_exit_shorthand(self):
        handler = (
            create_handler("exit").when_state_exit("collecting").do(lambda ctx: {})
        )
        assert HandlerTiming.PRE_TRANSITION in handler.timings
        assert "collecting" in handler.states

    def test_on_context_update_shorthand(self):
        handler = create_handler("cu").on_context_update("user_name").do(lambda ctx: {})
        assert HandlerTiming.CONTEXT_UPDATE in handler.timings
        assert "user_name" in handler.updated_keys

    def test_not_on_state(self):
        handler = create_handler("ns").not_when_state("error").do(lambda ctx: {})
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
            .when_state("s1")
            .with_priority(10)
            .when_context_has("k1")
        )
        assert same is builder


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
        handler = create_handler("sf").when_state("active").do(lambda ctx: {})
        assert handler.should_execute(HandlerTiming.PRE_PROCESSING, "active", None, {})
        assert not handler.should_execute(
            HandlerTiming.PRE_PROCESSING, "inactive", None, {}
        )

    def test_not_state_filter(self):
        handler = create_handler("nsf").not_when_state("error").do(lambda ctx: {})
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
            .when_state("active")
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

    def test_execute_raises_handler_execution_error_on_failure(self):
        def bad_fn(ctx):
            raise ValueError("bad value")

        handler = create_handler("bad").do(bad_fn)
        with pytest.raises(HandlerExecutionError) as exc_info:
            handler.execute({})
        assert exc_info.value.handler_name == "bad"
        assert isinstance(exc_info.value.original_error, ValueError)


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

        hs._execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
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

        hs._execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
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

        result = hs._execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
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

        result = hs._execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        assert result["counter"] == 3

    def test_original_context_not_mutated(self):
        original = {"original_key": "original_value"}

        hs = HandlerSystem()
        hs.register_handler(create_handler("mutator").do(lambda ctx: {"added": True}))

        hs._execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, original)
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

        result = hs._execute_handlers(HandlerTiming.POST_TRANSITION, "s1", "s2", {})
        # Metadata must NOT appear in user-facing context output
        assert "_handler_metadata" not in result
        assert result == {"x": 1}

    def test_no_metadata_when_no_handler_executes(self):
        hs = HandlerSystem()
        hs.register_handler(NeverRunHandler(name="never"))
        result = hs._execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        assert "_handler_metadata" not in result

    def test_multiple_handlers_return_merged_context(self):
        hs = HandlerSystem()
        hs.register_handler(
            create_handler("a").with_priority(10).do(lambda ctx: {"from_a": 1})
        )
        hs.register_handler(
            create_handler("b").with_priority(20).do(lambda ctx: {"from_b": 2})
        )

        result = hs._execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
        assert "_handler_metadata" not in result
        assert result == {"from_a": 1, "from_b": 2}

    def test_none_result_handler_does_not_pollute_context(self):
        hs = HandlerSystem()
        hs.register_handler(create_handler("nil").do(lambda ctx: None))
        result = hs._execute_handlers(HandlerTiming.PRE_PROCESSING, "s1", None, {})
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
