"""
Unit tests for handler timeout functionality in the HandlerSystem.

Tests cover: handler_timeout parameter, DEFAULT_HANDLER_TIMEOUT constant,
timeout with error_mode interaction, and backward compatibility with
handler_timeout=None.
"""

import time
import pytest

from fsm_llm.handlers import (
    HandlerTiming,
    HandlerSystem,
    HandlerExecutionError,
    BaseHandler,
)
from fsm_llm.constants import DEFAULT_HANDLER_TIMEOUT


# ── Helpers ───────────────────────────────────────────────────


class FastHandler(BaseHandler):
    """A handler that completes quickly."""

    def __init__(self, name="fast", result=None):
        super().__init__(name=name, priority=100)
        self._result = result or {"fast": True}

    def should_execute(
        self, timing, current_state, target_state, context, updated_keys=None
    ):
        return True

    def execute(self, context):
        return self._result.copy()


class SlowHandler(BaseHandler):
    """A handler that sleeps for a configurable duration."""

    def __init__(self, name="slow", sleep_seconds=1.0, result=None):
        super().__init__(name=name, priority=100)
        self._sleep_seconds = sleep_seconds
        self._result = result or {"slow": True}

    def should_execute(
        self, timing, current_state, target_state, context, updated_keys=None
    ):
        return True

    def execute(self, context):
        time.sleep(self._sleep_seconds)
        return self._result.copy()


# ══════════════════════════════════════════════════════════════
# 1. DEFAULT_HANDLER_TIMEOUT constant
# ══════════════════════════════════════════════════════════════


class TestDefaultHandlerTimeout:
    """Verify the DEFAULT_HANDLER_TIMEOUT constant exists and is sensible."""

    def test_constant_exists(self):
        assert DEFAULT_HANDLER_TIMEOUT is not None

    def test_constant_is_positive_float(self):
        assert isinstance(DEFAULT_HANDLER_TIMEOUT, float)
        assert DEFAULT_HANDLER_TIMEOUT > 0

    def test_constant_value(self):
        assert DEFAULT_HANDLER_TIMEOUT == 30.0


# ══════════════════════════════════════════════════════════════
# 2. HandlerSystem with handler_timeout=None (backward compat)
# ══════════════════════════════════════════════════════════════


class TestNoTimeout:
    """handler_timeout=None means no timeout — backward compatible default."""

    def test_default_timeout_is_none(self):
        hs = HandlerSystem()
        assert hs.handler_timeout is None

    def test_explicit_none_timeout(self):
        hs = HandlerSystem(handler_timeout=None)
        assert hs.handler_timeout is None

    def test_no_timeout_fast_handler_succeeds(self):
        hs = HandlerSystem(handler_timeout=None)
        hs.register_handler(FastHandler())

        result = hs.execute_handlers(
            timing=HandlerTiming.PRE_PROCESSING,
            current_state="start",
            target_state=None,
            context={},
        )

        assert result.get("fast") is True

    def test_no_timeout_slow_handler_completes(self):
        """Without timeout, even slow handlers should run to completion."""
        hs = HandlerSystem(handler_timeout=None)
        hs.register_handler(SlowHandler(sleep_seconds=0.3))

        result = hs.execute_handlers(
            timing=HandlerTiming.PRE_PROCESSING,
            current_state="start",
            target_state=None,
            context={},
        )

        assert result.get("slow") is True


# ══════════════════════════════════════════════════════════════
# 3. Handler completes within timeout
# ══════════════════════════════════════════════════════════════


class TestHandlerWithinTimeout:
    """Handlers that finish before the timeout should succeed normally."""

    def test_fast_handler_with_generous_timeout(self):
        hs = HandlerSystem(handler_timeout=5.0)
        hs.register_handler(FastHandler())

        result = hs.execute_handlers(
            timing=HandlerTiming.PRE_PROCESSING,
            current_state="start",
            target_state=None,
            context={},
        )

        assert result.get("fast") is True

    def test_slightly_slow_handler_within_timeout(self):
        hs = HandlerSystem(handler_timeout=0.5)
        hs.register_handler(SlowHandler(sleep_seconds=0.05, result={"completed": True}))

        result = hs.execute_handlers(
            timing=HandlerTiming.PRE_PROCESSING,
            current_state="start",
            target_state=None,
            context={},
        )

        assert result.get("completed") is True

    def test_multiple_fast_handlers_with_timeout(self):
        hs = HandlerSystem(handler_timeout=1.0)
        hs.register_handler(FastHandler(name="h1", result={"h1": True}))
        hs.register_handler(FastHandler(name="h2", result={"h2": True}))

        result = hs.execute_handlers(
            timing=HandlerTiming.PRE_PROCESSING,
            current_state="start",
            target_state=None,
            context={},
        )

        assert result.get("h1") is True
        assert result.get("h2") is True


# ══════════════════════════════════════════════════════════════
# 4. Handler exceeds timeout
# ══════════════════════════════════════════════════════════════


class TestHandlerExceedsTimeout:
    """Handler that exceeds timeout should raise HandlerExecutionError."""

    @pytest.mark.slow
    def test_slow_handler_raises_handler_execution_error(self):
        hs = HandlerSystem(handler_timeout=0.1, error_mode="raise")
        hs.register_handler(SlowHandler(sleep_seconds=2.0))

        with pytest.raises(HandlerExecutionError) as exc_info:
            hs.execute_handlers(
                timing=HandlerTiming.PRE_PROCESSING,
                current_state="start",
                target_state=None,
                context={},
            )

        # The original error should be a TimeoutError
        assert isinstance(exc_info.value.original_error, TimeoutError)

    @pytest.mark.slow
    def test_timeout_error_includes_handler_name(self):
        hs = HandlerSystem(handler_timeout=0.1, error_mode="raise")
        hs.register_handler(SlowHandler(name="my_slow_handler", sleep_seconds=2.0))

        with pytest.raises(HandlerExecutionError) as exc_info:
            hs.execute_handlers(
                timing=HandlerTiming.PRE_PROCESSING,
                current_state="start",
                target_state=None,
                context={},
            )

        assert exc_info.value.handler_name == "my_slow_handler"

    @pytest.mark.slow
    def test_timeout_error_message_includes_duration(self):
        hs = HandlerSystem(handler_timeout=0.1, error_mode="raise")
        hs.register_handler(SlowHandler(name="timed_out", sleep_seconds=2.0))

        with pytest.raises(HandlerExecutionError) as exc_info:
            hs.execute_handlers(
                timing=HandlerTiming.PRE_PROCESSING,
                current_state="start",
                target_state=None,
                context={},
            )

        assert "0.1" in str(exc_info.value.original_error)


# ══════════════════════════════════════════════════════════════
# 5. Timeout with error_mode interaction
# ══════════════════════════════════════════════════════════════


class TestTimeoutErrorModeInteraction:
    """Timeout behavior depends on error_mode."""

    @pytest.mark.slow
    def test_timeout_with_continue_mode_swallows_error(self):
        """In continue mode, timeout should be caught and logged, not raised."""
        hs = HandlerSystem(handler_timeout=0.1, error_mode="continue")
        hs.register_handler(SlowHandler(name="slow_swallowed", sleep_seconds=2.0))

        # Should not raise
        result = hs.execute_handlers(
            timing=HandlerTiming.PRE_PROCESSING,
            current_state="start",
            target_state=None,
            context={},
        )

        # The slow handler's result should not be in output (it timed out)
        assert result.get("slow") is not True

    @pytest.mark.slow
    def test_timeout_with_raise_mode_propagates_error(self):
        """In raise mode, timeout should propagate as HandlerExecutionError."""
        hs = HandlerSystem(handler_timeout=0.1, error_mode="raise")
        hs.register_handler(SlowHandler(sleep_seconds=2.0))

        with pytest.raises(HandlerExecutionError):
            hs.execute_handlers(
                timing=HandlerTiming.PRE_PROCESSING,
                current_state="start",
                target_state=None,
                context={},
            )

    @pytest.mark.slow
    def test_continue_mode_runs_subsequent_handlers_after_timeout(self):
        """After one handler times out in continue mode, remaining handlers should still run."""
        hs = HandlerSystem(handler_timeout=0.1, error_mode="continue")
        hs.register_handler(
            SlowHandler(name="slow_first", sleep_seconds=2.0, result={"slow_ran": True})
        )
        hs.register_handler(FastHandler(name="fast_second", result={"fast_ran": True}))

        result = hs.execute_handlers(
            timing=HandlerTiming.PRE_PROCESSING,
            current_state="start",
            target_state=None,
            context={},
        )

        # The fast handler should have run despite the slow one timing out
        assert result.get("fast_ran") is True
        assert result.get("slow_ran") is not True

    @pytest.mark.slow
    def test_timeout_with_critical_handler_raises_even_in_continue_mode(self):
        """Critical handlers should always raise, even in continue mode."""

        class CriticalSlowHandler(BaseHandler):
            def __init__(self):
                super().__init__(name="critical_slow", priority=100, critical=True)

            def should_execute(
                self, timing, current_state, target_state, context, updated_keys=None
            ):
                return True

            def execute(self, context):
                time.sleep(2.0)
                return {}

        hs = HandlerSystem(handler_timeout=0.1, error_mode="continue")
        hs.register_handler(CriticalSlowHandler())

        with pytest.raises(HandlerExecutionError):
            hs.execute_handlers(
                timing=HandlerTiming.PRE_PROCESSING,
                current_state="start",
                target_state=None,
                context={},
            )


# ══════════════════════════════════════════════════════════════
# 6. handler_timeout parameter validation
# ══════════════════════════════════════════════════════════════


class TestHandlerTimeoutParameter:
    """Test that handler_timeout is stored and used correctly."""

    def test_handler_timeout_stored_on_system(self):
        hs = HandlerSystem(handler_timeout=5.0)
        assert hs.handler_timeout == 5.0

    def test_handler_timeout_accepts_int(self):
        hs = HandlerSystem(handler_timeout=10)
        assert hs.handler_timeout == 10

    def test_handler_timeout_accepts_default_constant(self):
        hs = HandlerSystem(handler_timeout=DEFAULT_HANDLER_TIMEOUT)
        assert hs.handler_timeout == DEFAULT_HANDLER_TIMEOUT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
