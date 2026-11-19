"""Native async of ``LiteLLMInterface`` (plan 51fde95b, part A, step 4).

The provider is the patched ``completion`` (sync names) and ``acompletion`` (async names)
bindings of ``fsm_llm.llm.providers.litellm``; the observer is a recording double. The
parity rows drive the same scripted reply through the sync and the async name and compare
the result, the observer events and the usage meter. Nothing here reaches a network.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from litellm.types.utils import Choices, Message, ModelResponse

from fsm_llm.core.definitions import CompletionRequest
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.llm import observer as obs
from fsm_llm.logging import logger

_PROVIDER = "fsm_llm.llm.providers.litellm"
_PARAMS = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}


def _llm(**kwargs: Any) -> LiteLLMInterface:
    # a closed port: a provider call that escaped a patch fails locally, never on a network
    return LiteLLMInterface(model="gpt-4o", api_base="http://127.0.0.1:9", **kwargs)


def _reply(text: str = "Hello", finish: str = "stop") -> ModelResponse:
    return ModelResponse(
        choices=[Choices(message=Message(content=text), finish_reason=finish)]
    )


class _Recorder:
    """A request observer that records every hook it is called with."""

    def __init__(self, raises: tuple[str, ...] = ()) -> None:
        self.events: list[tuple[str, Any]] = []
        self.raises = raises

    def _hit(self, name: str, payload: Any) -> None:
        self.events.append((name, payload))
        if name in self.raises:
            raise RuntimeError(f"{name} boom")

    def begin_send(self, call: dict[str, Any], *, call_type: str) -> object:
        self._hit("begin", (call, call_type))
        return "token"

    def end_send(self, token: object, response: Any) -> None:
        self._hit("end", (token, response))

    def fail_send(self, token: object, error: BaseException) -> None:
        self._hit("fail", (token, error))

    def shape(self) -> list[Any]:
        """The events with the reply and the error reduced to what parity compares."""
        out: list[Any] = []
        for name, payload in self.events:
            if name == "begin":
                out.append((name, payload[0], payload[1]))
            elif name == "fail":
                out.append((name, type(payload[1]).__name__, str(payload[1])))
            else:
                out.append((name,))
        return out


@contextmanager
def _providers(
    script: Callable[..., Any], async_script: Callable[..., Any] | None = None
):
    """Both provider bindings run ``script`` (``acompletion`` may run its own coroutine
    function): a call that lands on either is visible."""
    with (
        patch(f"{_PROVIDER}.completion", side_effect=script) as sync,
        patch(f"{_PROVIDER}.acompletion", new_callable=AsyncMock) as async_,
        patch(f"{_PROVIDER}.get_supported_openai_params", return_value=[]),
    ):
        async_.side_effect = async_script or script
        yield sync, async_


@contextmanager
def _observed(recorder: _Recorder):
    token = obs.install_observer(recorder)
    try:
        yield recorder
    finally:
        obs.reset_observer(token)


async def _send(llm: LiteLLMInterface, transport: str, params: dict[str, Any]) -> Any:
    if transport == "sync":
        return llm._send(params, call_type="response_generation")
    return await llm._asend(params, call_type="response_generation")


async def _outcome(awaitable: Awaitable[Any]) -> tuple[Any, ...]:
    """What an awaited call did, comparable across transports (results by identity)."""
    try:
        value = await awaitable
    except Exception as exc:
        return ("raised", type(exc).__name__, str(exc))
    return ("returned", value)


@contextmanager
def _warnings():
    """The WARNING texts the library logs inside the block (its logger is off by default)."""
    records: list[str] = []
    sink = logger.add(lambda m: records.append(m.record["message"]), level="WARNING")
    logger.enable("fsm_llm")
    try:
        yield records
    finally:
        logger.disable("fsm_llm")
        logger.remove(sink)


class TestTheSendFamilyOnBothTransports:
    """``_send`` and ``_asend`` share one family: the same feeding on every outcome."""

    @pytest.mark.parametrize("scenario", ["success", "outage", "cut", "no_choices"])
    async def test_the_two_transports_leave_the_same_meter_and_events(
        self, scenario: str
    ) -> None:
        reply = {
            "success": _reply(),
            "cut": _reply("partial", "length"),
            "no_choices": ModelResponse(choices=[]),
        }.get(scenario)

        def script(**_: Any) -> Any:
            if reply is None:
                raise RuntimeError("provider down")
            return reply

        seen: dict[str, tuple[Any, ...]] = {}
        for transport in ("sync", "async"):
            llm = _llm()
            with _providers(script) as (sync, async_), _observed(_Recorder()) as rec:
                result = await _outcome(_send(llm, transport, _PARAMS))
                (landed, other) = (
                    (sync, async_) if transport == "sync" else (async_, sync)
                )
                assert landed.call_count == 1 and other.call_count == 0
            seen[transport] = (result, rec.shape(), llm.usage())
        assert seen["sync"] == seen["async"]
        outcome, events, usage = seen["async"]
        assert usage.calls == 1 and usage.errors == (1 if scenario == "outage" else 0)
        assert [e[0] for e in events] == [
            "begin",
            "fail" if scenario == "outage" else "end",
        ]
        if scenario == "outage":
            assert outcome == ("raised", "RuntimeError", "provider down")
        else:
            assert outcome[0] == "returned" and outcome[1] is reply

    async def test_a_cut_reply_is_warned_once_on_either_transport(self) -> None:
        for transport in ("sync", "async"):
            llm = _llm()

            with (
                _warnings() as warnings,
                _providers(lambda **_: _reply("partial", "length")),
            ):
                await _send(llm, transport, {**_PARAMS, "max_tokens": 7})
            assert sum("was cut off at max_tokens=7" in w for w in warnings) == 1

    async def test_a_streamed_request_is_a_call_with_usage_missing_on_both(
        self,
    ) -> None:
        for transport in ("sync", "async"):
            llm = _llm()
            with _providers(lambda **_: iter(["a", "b"])):
                await _send(llm, transport, {**_PARAMS, "stream": True})
            usage = llm.usage()
            assert (usage.calls, usage.usage_missing) == (1, 1)
            assert set(usage.by_kind) == {"stream"}

    async def test_a_refusing_begin_send_stops_the_call_on_either_transport(
        self,
    ) -> None:
        for transport in ("sync", "async"):
            llm = _llm()
            with _providers(lambda **_: _reply()) as (sync, async_):
                with _observed(_Recorder(raises=("begin",))):
                    with pytest.raises(RuntimeError, match="begin boom"):
                        await _send(llm, transport, _PARAMS)
            assert sync.call_count == async_.call_count == 0
            assert llm.usage().calls == 0

    async def test_a_raising_end_or_fail_never_changes_the_outcome(self) -> None:
        llm = _llm()
        with _providers(lambda **_: _reply()), _observed(_Recorder(raises=("end",))):
            assert (await _send(llm, "async", _PARAMS)).choices
        with _providers(_boom), _observed(_Recorder(raises=("fail",))):
            with pytest.raises(RuntimeError, match="provider down"):
                await _send(llm, "async", _PARAMS)
        assert (llm.usage().calls, llm.usage().errors) == (2, 1)


def _boom(**_: Any) -> Any:
    raise RuntimeError("provider down")


def _hanging(started: asyncio.Event) -> Callable[..., Awaitable[Any]]:
    """An ``acompletion`` that signals it is in flight and then never answers."""

    async def hang(**_: Any) -> Any:
        started.set()
        await asyncio.Event().wait()

    return hang


class TestACancelledRequest:
    """D-008: a cancelled in-flight request is reported to the observer and the meter."""

    async def test_a_cancel_is_a_failed_send_and_is_raised_again(self) -> None:
        llm, started = _llm(), asyncio.Event()
        with _providers(_boom, _hanging(started)), _observed(_Recorder()) as rec:
            task = asyncio.ensure_future(_send(llm, "async", _PARAMS))
            await asyncio.wait_for(started.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert rec.shape()[0][0] == "begin"
        assert rec.shape()[1][:2] == ("fail", "CancelledError")
        usage = llm.usage()
        assert (usage.calls, usage.errors) == (1, 1)

    @pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
    async def test_an_interrupt_passes_through_as_it_does_in_send(
        self, exc: type[BaseException]
    ) -> None:
        for transport in ("sync", "async"):
            llm = _llm()

            def script(**_: Any) -> Any:
                raise exc()

            with _providers(script), _observed(_Recorder()) as rec:
                with pytest.raises(exc):
                    await _send(llm, transport, _PARAMS)
            assert [e[0] for e in rec.shape()] == ["begin"]
            assert llm.usage().calls == 0


class TestIsObservedReadsBothSends:
    def test_a_subclass_that_replaces_asend_is_observed_only_if_it_says_so(
        self,
    ) -> None:
        class Own(LiteLLMInterface):
            async def _asend(self, call_params: Any, *, call_type: str) -> Any:
                return None

        class Restates(Own):
            observes_requests = True

        class Inherits(Restates):
            pass

        assert obs.is_observed(_llm())
        assert not obs.is_observed(Own(model="m"))
        assert obs.is_observed(Restates(model="m")) and obs.is_observed(Inherits("m"))


# --- one parity table over the methods: the same scripted reply through the sync name and
# the async name (a row per method grows with each converted family) -------------------

_METHODS: dict[str, Callable[[], Any]] = {
    "complete": lambda: CompletionRequest(messages=[{"role": "user", "content": "hi"}]),
}
# scenario -> (what the provider does, the call raises, calls, errors, events, cut warnings)
_SCENARIOS: dict[str, tuple[Any, bool, int, int, tuple[str, ...], int]] = {
    "success": (_reply(), False, 1, 0, ("begin", "end"), 0),
    "cut": (_reply("partial", "length"), False, 1, 0, ("begin", "end"), 1),
    "malformed": (ModelResponse(choices=[]), True, 1, 0, ("begin", "end"), 0),
    "outage": (None, True, 1, 1, ("begin", "fail"), 0),
}


async def _invoke(llm: LiteLLMInterface, name: str, request: Any) -> Any:
    value = getattr(llm, name)(request)
    return await value if inspect.isawaitable(value) else value


async def _drive(method: str, transport: str, scenario: str) -> tuple[Any, ...]:
    reply = _SCENARIOS[scenario][0]
    llm = _llm()

    def script(**_: Any) -> Any:
        if reply is None:
            raise RuntimeError("provider down")
        return reply

    name = method if transport == "sync" else f"a{method}"
    with _providers(script) as (sync, async_), _observed(_Recorder()) as rec:
        with _warnings() as warned:
            result = await _outcome(_invoke(llm, name, _METHODS[method]()))
        landed = (sync.call_count, async_.call_count)
    assert landed == ((1, 0) if transport == "sync" else (0, 1))
    return result, rec.shape(), llm.usage(), warned


class TestEveryMethodGivesTheSameOnBothNames:
    @pytest.mark.parametrize("scenario", list(_SCENARIOS))
    @pytest.mark.parametrize("method", list(_METHODS))
    async def test_result_events_meter_and_warnings_are_equal(
        self, method: str, scenario: str
    ) -> None:
        sync = await _drive(method, "sync", scenario)
        async_ = await _drive(method, "async", scenario)
        assert sync == async_
        # non-vacuity: what the scenario is
        _, raises, calls, errors, events, cut = _SCENARIOS[scenario]
        result, shape, usage, warned = async_
        assert (result[0] == "raised") is raises
        if raises:
            assert result[1] == "LLMResponseError"
        assert (usage.calls, usage.errors) == (calls, errors)
        assert tuple(e[0] for e in shape) == events
        assert sum("was cut off" in w for w in warned) == cut

    async def test_a_garbled_tool_call_is_data_on_both_names(self) -> None:
        request = CompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}],
        )

        def script(**_: Any) -> Any:
            raise RuntimeError("Invalid tool call: xml syntax error")

        with _providers(script):
            sync = await _invoke(_llm(), "complete", request)
            async_ = await _invoke(_llm(), "acomplete", request)
        assert sync == async_ and sync.kind == "malformed"

    async def test_a_cancel_reaches_the_observer_and_the_meter_of_every_method(
        self,
    ) -> None:
        for method, request in _METHODS.items():
            llm, started = _llm(), asyncio.Event()
            with _providers(_boom, _hanging(started)), _observed(_Recorder()) as rec:
                task = asyncio.ensure_future(_invoke(llm, f"a{method}", request()))
                await asyncio.wait_for(started.wait(), 5)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert [e[0] for e in rec.shape()] == ["begin", "fail"], method
            assert rec.shape()[1][1] == "CancelledError"
            assert (llm.usage().calls, llm.usage().errors) == (1, 1), method

    async def test_the_async_names_do_not_land_on_the_sync_binding(self) -> None:
        # a patched `completion` is NOT intercepted by an async name, and a patched
        # `acompletion` is not intercepted by a sync name (each mode has its own binding)
        for method, request in _METHODS.items():
            with _providers(lambda **_: _reply()) as (sync, async_):
                await _invoke(_llm(), f"a{method}", request())
                assert (sync.call_count, async_.call_count) == (0, 1)
                await _invoke(_llm(), method, request())
                assert (sync.call_count, async_.call_count) == (1, 1)
