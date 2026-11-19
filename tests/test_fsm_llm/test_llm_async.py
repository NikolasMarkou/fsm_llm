"""Native async of ``LiteLLMInterface`` (plan 51fde95b, part A, step 4).

The provider is the patched ``completion`` (sync names) and ``acompletion`` (async names)
bindings of ``fsm_llm.llm.providers.litellm``; the observer is a recording double. The
parity rows drive the same scripted reply through the sync and the async name and compare
the result, the observer events and the usage meter. Nothing here reaches a network.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from litellm.types.utils import (
    Choices,
    EmbeddingResponse,
    Message,
    ModelResponse,
    Usage,
)

from fsm_llm.aio import async_name
from fsm_llm.core.definitions import (
    BulkExtractionRequest,
    ClassificationSchema,
    CompletionRequest,
    CompletionResponse,
    FieldExtractionRequest,
    HierarchicalSchema,
    IntentDefinition,
    ResponseGenerationRequest,
)
from fsm_llm.engine.classification import Classifier, HierarchicalClassifier
from fsm_llm.llm import LiteLLMEmbedder, LiteLLMInterface
from fsm_llm.llm import observer as obs
from fsm_llm.logging import logger
from tests.test_fsm_llm.event_log_support import _chunk

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


def _scripted(*replies: Any) -> Callable[..., Any]:
    """A provider script that answers each call with the next reply (an exception is raised)."""
    queue = iter(replies)

    def script(**_: Any) -> Any:
        reply = next(queue)
        if isinstance(reply, Exception):
            raise reply
        return reply

    return script


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

# method -> (a request, the text of a reply that method can read)
_METHODS: dict[str, tuple[Callable[[], Any], str]] = {
    "generate_response": (
        lambda: ResponseGenerationRequest(system_prompt="Greet.", user_message="hi"),
        "Hello",
    ),
    "extract_field": (
        lambda: FieldExtractionRequest(
            system_prompt="Extract.",
            user_message="I am Bob",
            field_name="name",
            field_type="str",
        ),
        '{"name": "Bob", "confidence": 0.9}',
    ),
    "extract_bulk_data": (
        lambda: BulkExtractionRequest(
            system_prompt="Extract.", user_message="I am Bob"
        ),
        '{"extracted_data": {"name": "Bob"}, "confidence": 0.9}',
    ),
    "complete": (
        lambda: CompletionRequest(messages=[{"role": "user", "content": "hi"}]),
        "Hello",
    ),
}
# scenario -> (the call raises, calls, errors, observer events, cut-off warnings)
_SCENARIOS: dict[str, tuple[bool, int, int, tuple[str, ...], int]] = {
    "success": (False, 1, 0, ("begin", "end"), 0),
    "cut": (False, 1, 0, ("begin", "end"), 1),
    "malformed": (True, 1, 0, ("begin", "end"), 0),
    "outage": (True, 1, 1, ("begin", "fail"), 0),
}


def _reply_for(method: str, scenario: str) -> Any:
    """The provider reply of a scenario (``None``: the provider raises) for one method."""
    text = _METHODS[method][1]
    return {
        "success": lambda: _reply(text),
        "cut": lambda: _reply(text, "length"),
        "malformed": lambda: ModelResponse(choices=[]),
        "outage": lambda: None,
    }[scenario]()


async def _invoke(llm: LiteLLMInterface, name: str, request: Any) -> Any:
    value = getattr(llm, name)(request)
    return await value if inspect.isawaitable(value) else value


async def _drive(method: str, transport: str, scenario: str) -> tuple[Any, ...]:
    reply = _reply_for(method, scenario)
    llm = _llm()

    def script(**_: Any) -> Any:
        if reply is None:
            raise RuntimeError("provider down")
        return reply

    name = method if transport == "sync" else f"a{method}"
    with _providers(script) as (sync, async_), _observed(_Recorder()) as rec:
        with _warnings() as warned:
            result = await _outcome(_invoke(llm, name, _METHODS[method][0]()))
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
        raises, calls, errors, events, cut = _SCENARIOS[scenario]
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
        for method, (request, _) in _METHODS.items():
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
        for method, (request, text) in _METHODS.items():
            with _providers(_scripted(*[_reply(text)] * 2)) as (sync, async_):
                await _invoke(_llm(), f"a{method}", request())
                assert (sync.call_count, async_.call_count) == (0, 1)
                await _invoke(_llm(), method, request())
                assert (sync.call_count, async_.call_count) == (1, 1)


class TestAsyncRequestsOverlap:
    @pytest.mark.parametrize("method", list(_METHODS))
    async def test_every_concurrent_call_is_in_flight_at_once(
        self, method: str
    ) -> None:
        n, inside, all_in, release = 20, 0, asyncio.Event(), asyncio.Event()

        async def provider(**_: Any) -> Any:
            nonlocal inside
            inside += 1
            if inside == n:
                all_in.set()
            await (
                release.wait()
            )  # a call that is serialised behind another never gets here
            return _reply(_METHODS[method][1])

        llm = _llm()
        with _providers(_boom, provider):
            calls = [
                asyncio.ensure_future(_invoke(llm, f"a{method}", _METHODS[method][0]()))
                for _ in range(n)
            ]
            await asyncio.wait_for(all_in.wait(), 2)
            release.set()
            await asyncio.gather(*calls)
        assert llm.usage().calls == n


class TestTheApologyRetryIsOneBody:
    """21cd7f8e/D-020: an empty reply is retried once, whichever name made the call."""

    @pytest.mark.parametrize("second", ["recovers", "fails"])
    async def test_it_fires_once_and_equally_on_either_name(self, second: str) -> None:
        results = []
        for name in ("generate_response", "agenerate_response"):
            llm = _llm()
            retry = _reply("Hello") if second == "recovers" else RuntimeError("down")
            script = _scripted(_reply('{"message": ""}'), retry)
            with _providers(script) as (sync, async_), _observed(_Recorder()) as rec:
                out = await _invoke(llm, name, _METHODS["generate_response"][0]())
            # both attempts, the retry too, land on the binding of the name's own mode
            assert (sync.call_count, async_.call_count) in {(2, 0), (0, 2)}
            assert (async_.call_count == 2) is (name == "agenerate_response")
            results.append((out, llm.apology_retry_count, rec.shape(), llm.usage()))
        assert results[0] == results[1]
        out, count, events, usage = results[1]
        assert count == 1 and usage.calls == 2
        assert [e[0] for e in events] == [
            "begin",
            "end",
            "begin",
            "end" if second == "recovers" else "fail",
        ]
        assert (out.message == "Hello") is (second == "recovers")


# --- the stream: lazy, equal on both names, closing it closes the provider stream ------


def _stream_script(*items: Any, closed: list[str] | None = None) -> tuple[Any, Any]:
    """Provider scripts for ``completion`` and ``acompletion`` that stream ``items``.

    An item is a chunk or an exception to raise at that point; ``closed`` gets one entry
    when the provider's generator is closed (finished, failed or closed early).
    """

    def run() -> Iterator[Any]:
        try:
            for item in items:
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            if closed is not None:
                closed.append("closed")

    async def arun() -> AsyncIterator[Any]:
        try:
            for item in items:
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            if closed is not None:
                closed.append("closed")

    return (lambda **_: run()), (lambda **_: arun())


_STREAM_REQUEST = _METHODS["generate_response"][0]
_STREAMS: dict[str, tuple[Any, ...]] = {
    "success": (_chunk("Hel"), _chunk("lo")),
    "empty": (_chunk(""), _chunk(None)),
    "failure_midway": (_chunk("Hel"), RuntimeError("boom")),
}


async def _stream(llm: LiteLLMInterface, transport: str) -> list[Any]:
    """The chunks the stream gave, then the error it ended with (if any)."""
    out: list[Any] = []
    try:
        if transport == "sync":
            out.extend(llm.generate_response_stream(_STREAM_REQUEST()))
        else:
            async for chunk in llm.agenerate_response_stream(_STREAM_REQUEST()):
                out.append(chunk)
    except Exception as exc:
        out.append((type(exc).__name__, str(exc)))
    return out


class TestTheStream:
    @pytest.mark.parametrize("scenario", [*_STREAMS, "outage"])
    async def test_both_names_give_the_same_chunks_events_and_meter(
        self, scenario: str
    ) -> None:
        seen = []
        for transport in ("sync", "async"):
            llm = _llm()
            if scenario == "outage":
                scripts: tuple[Any, Any] = (_boom, _boom)
            else:
                scripts = _stream_script(*_STREAMS[scenario])
            with _providers(*scripts) as (sync, async_), _observed(_Recorder()) as rec:
                chunks = await _stream(llm, transport)
            assert (sync.call_count, async_.call_count) == (
                (1, 0) if transport == "sync" else (0, 1)
            )
            seen.append((chunks, rec.shape(), llm.usage()))
        assert seen[0] == seen[1]
        chunks, events, usage = seen[1]
        assert usage.calls == 1 and set(usage.by_kind) == {"stream"}
        assert [e[0] for e in events] == [
            "begin",
            "fail" if scenario == "outage" else "end",
        ]
        assert (usage.errors, usage.usage_missing) == (
            (1, 0) if scenario == "outage" else (0, 1)
        )
        errors = [c[0] for c in chunks if isinstance(c, tuple)]
        assert errors == ([] if scenario == "success" else ["LLMResponseError"])

    async def test_a_wrapper_with_a_sync_and_an_async_side_is_read_on_the_modes_side(
        self,
    ) -> None:
        # litellm's stream wrapper iterates both ways; its async side needs a running loop
        used: list[str] = []

        class Both:
            def __iter__(self) -> Iterator[Any]:
                used.append("sync")
                return iter(_STREAMS["success"])

            async def __aiter__(self) -> AsyncIterator[Any]:
                used.append("async")
                for chunk in _STREAMS["success"]:
                    yield chunk

        got = {}
        for transport in ("sync", "async"):
            with _providers(lambda **_: Both(), lambda **_: Both()):
                got[transport] = await _stream(_llm(), transport)
        assert got["sync"] == got["async"] == ["Hel", "lo"]
        assert used == ["sync", "async"]

    async def test_nothing_is_sent_before_the_first_next(self) -> None:
        for transport in ("sync", "async"):
            llm = _llm()
            with _providers(*_stream_script(*_STREAMS["success"])) as (sync, async_):
                if transport == "sync":
                    it = llm.generate_response_stream(_STREAM_REQUEST())
                    assert sync.call_count == 0
                    assert next(it) == "Hel" and sync.call_count == 1
                    it.close()
                else:
                    agen = llm.agenerate_response_stream(_STREAM_REQUEST())
                    assert async_.call_count == 0
                    assert await anext(agen) == "Hel" and async_.call_count == 1
                    await agen.aclose()

    async def test_closing_early_closes_the_provider_stream(self) -> None:
        for transport in ("sync", "async"):
            closed: list[str] = []
            llm = _llm()
            with _providers(*_stream_script(*_STREAMS["success"], closed=closed)):
                if transport == "sync":
                    it = llm.generate_response_stream(_STREAM_REQUEST())
                    assert next(it) == "Hel"
                    assert closed == []
                    it.close()
                else:
                    agen = llm.agenerate_response_stream(_STREAM_REQUEST())
                    assert await anext(agen) == "Hel"
                    assert closed == []
                    await agen.aclose()
            assert closed == ["closed"], transport


# --- the real library: litellm's own ``mock_response`` answers, no network, no patch -----


class TestTheRealLibrary:
    """Unit doubles cannot show how litellm's wrappers behave (its stream is both sync and
    async, and its async side needs a loop): one pass through the real ``completion`` and
    ``acompletion`` for every kind of call."""

    async def test_both_names_read_the_same_reply_the_same_stream_and_the_same_meter(
        self,
    ) -> None:
        def real() -> LiteLLMInterface:
            return LiteLLMInterface(
                "gpt-4o", api_key="sk-none", mock_response="Hello there"
            )

        sync, async_ = real(), real()
        request, complete = (
            _METHODS["generate_response"][0](),
            _METHODS["complete"][0](),
        )
        assert sync.generate_response(request) == await async_.agenerate_response(
            request
        )
        streamed = [chunk async for chunk in async_.agenerate_response_stream(request)]
        assert list(sync.generate_response_stream(request)) == streamed
        assert "".join(streamed) == "Hello there" and len(streamed) > 1
        assert sync.complete(complete) == await async_.acomplete(complete)
        assert sync.usage() == async_.usage() and sync.usage().calls == 3

    async def test_the_embedder_reads_litellms_own_mock_embedding_on_both_names(
        self,
    ) -> None:
        def real() -> LiteLLMEmbedder:
            return LiteLLMEmbedder(
                "text-embedding-3-small", api_key="sk-none", mock_response=[0.5, 0.25]
            )

        sync, async_ = real(), real()
        assert sync.embed(["a"]) == await async_.aembed(["a"]) == [[0.5, 0.25]]
        assert sync.usage() == async_.usage() and sync.usage().calls == 1


# --- the subclass guard (D-009): it warns, it never changes dispatch -------------------

_TWINS = (
    "generate_response",
    "generate_response_stream",
    "extract_field",
    "extract_bulk_data",
    "complete",
    "_send",
    "_make_llm_call",
)


def _define(**members: Any) -> tuple[type, list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cls = type("Sub", (LiteLLMInterface,), members)
    return cls, caught


async def _a(self: Any, *args: Any, **kwargs: Any) -> None:
    return None


class TestTheSubclassGuard:
    @pytest.mark.parametrize("name", _TWINS)
    def test_a_sync_override_without_its_twin_warns_once_at_the_class_statement(
        self, name: str
    ) -> None:
        _, caught = _define(**{name: lambda self, *a, **k: None})
        assert [w.category for w in caught] == [RuntimeWarning]
        message = str(caught[0].message)
        assert f"overrides {name} but not {async_name(name)}" in message
        assert caught[0].filename == __file__

    @pytest.mark.parametrize("name", _TWINS)
    def test_nothing_warns_when_the_twin_is_overridden_too_or_alone(
        self, name: str
    ) -> None:
        _, both = _define(**{name: lambda self, *a, **k: None, async_name(name): _a})
        _, only_async = _define(**{async_name(name): _a})
        _, nothing = _define()
        assert both == only_async == nothing == []

    def test_the_warning_reaches_the_class_statement_through_an_init_subclass_hook(
        self,
    ) -> None:
        class Mid(LiteLLMInterface):
            def __init_subclass__(cls, **kwargs: Any) -> None:
                super().__init_subclass__(**kwargs)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            line = inspect.currentframe().f_lineno + 2  # type: ignore[union-attr]

            class Sub(Mid):
                complete = lambda self, request: None  # noqa: E731

        assert [(w.filename, w.lineno) for w in caught] == [(__file__, line)]

    async def test_it_warns_and_changes_no_dispatch(self) -> None:
        sub, _ = _define(generate_response=lambda self, request: "override")
        llm = sub(model="gpt-4o", api_base="http://127.0.0.1:9")
        request = _METHODS["generate_response"][0]()
        with _providers(lambda **_: _reply()) as (sync, async_):
            assert llm.generate_response(request) == "override"
            result = await llm.agenerate_response(request)
        # the async name reaches the inherited native twin: the override is bypassed
        assert (
            result.message == "Hello"
            and async_.call_count == 1
            and sync.call_count == 0
        )


# --- one body per method, one send family: read from the source, so a copy cannot hide ---

_PROVIDERS_DIR = Path(__file__).resolve().parents[2] / "src/fsm_llm/llm/providers"
# public sync name -> the providers file that holds it; each is only `return run_sync|iterate_sync(self.a<name>(...))`
_WRAPPERS = {
    "generate_response": "litellm.py",
    "extract_field": "litellm.py",
    "extract_bulk_data": "litellm.py",
    "complete": "litellm.py",
    "generate_response_stream": "litellm.py",
    "_make_llm_call": "call.py",
}


def _defs(file: str) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse((_PROVIDERS_DIR / file).read_text(encoding="utf-8"))
    return {
        n.name: n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _calls(node: ast.AST) -> list[str]:
    return [
        n.func.id if isinstance(n.func, ast.Name) else n.func.attr
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name | ast.Attribute)
    ]


class TestOneBodyPerMethod:
    @pytest.mark.parametrize("name", list(_WRAPPERS))
    def test_a_sync_method_is_only_a_wrapper_over_its_async_twin(
        self, name: str
    ) -> None:
        defs = _defs(_WRAPPERS[name])
        twin = async_name(name)
        assert isinstance(defs[twin], ast.AsyncFunctionDef)
        body = [
            n
            for n in defs[name].body
            if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
        ]
        assert len(body) == 1 and isinstance(body[0], ast.Return)
        driver = "iterate_sync" if name == "generate_response_stream" else "run_sync"
        # `return <driver>(self.<twin>(...))` and no other call: no parse, no retry, no send
        assert [c for c in _calls(body[0]) if c != twin] == [driver]
        assert twin in _calls(body[0])

    def test_the_parse_and_retry_bodies_exist_once(self) -> None:
        source = (_PROVIDERS_DIR / "litellm.py").read_text(encoding="utf-8")
        defs = _defs("litellm.py")
        assert source.count("self.apology_retry_count += 1") == 1
        for sync_name in ("generate_response", "extract_field", "extract_bulk_data"):
            assert not [c for c in _calls(defs[sync_name]) if c.startswith("_parse")]
        parsers = [
            c for c in _calls(defs["agenerate_response"]) if c.startswith("_parse")
        ]
        assert (
            parsers == ["_parse_response_generation_response"] * 2
        )  # first read, retry

    def test_the_meter_and_the_observer_are_fed_only_in_the_send_family(self) -> None:
        defs = _defs("litellm.py")
        family = {"_begin_send", "_failed_send", "_finished_send"}
        fed = {"record", "record_error", "begin_send", "end_send", "fail_send"}
        for name, node in defs.items():
            if name not in family:
                assert not fed & set(_calls(node)), name
        source = (_PROVIDERS_DIR / "litellm.py").read_text(encoding="utf-8")
        assert all(source.count(f"def {name}(") == 1 for name in family)
        assert source.count("meter.record(") == 1
        for sender in ("_send", "_asend"):
            calls = _calls(defs[sender])
            assert all(calls.count(helper) == 1 for helper in family), sender


_SRC = Path(__file__).resolve().parents[2] / "src/fsm_llm"


@pytest.mark.parametrize(
    ("file", "cls", "name"),
    [
        ("llm/providers/embedder.py", "LiteLLMEmbedder", "embed"),
        ("engine/classification.py", "Classifier", "classify"),
        ("engine/classification.py", "Classifier", "classify_multi"),
        ("engine/classification.py", "Classifier", "_call_llm"),
        ("engine/classification.py", "HierarchicalClassifier", "classify"),
    ],
)
def test_an_embedder_or_classifier_sync_name_is_only_a_run_sync_wrapper(
    file: str, cls: str, name: str
) -> None:
    tree = ast.parse((_SRC / file).read_text(encoding="utf-8"))
    owner = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls
    )
    defs = {
        n.name: n
        for n in owner.body
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    twin = async_name(name)
    assert isinstance(defs[twin], ast.AsyncFunctionDef)
    body = [
        n
        for n in defs[name].body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
    ]
    assert len(body) == 1 and isinstance(body[0], ast.Return)
    assert [c for c in _calls(body[0]) if c != twin] == ["run_sync"]
    assert twin in _calls(body[0])


# --- the embedder twin (step 6.1): one body, two transports ----------------------------

_EMBEDDER = "fsm_llm.llm.providers.embedder"
_TEXTS = ["a", "b"]


def _vectors(**params: Any) -> EmbeddingResponse:
    data = [
        {"object": "embedding", "index": i, "embedding": [float(i), 1.0]}
        for i in range(len(params["input"]))
    ]
    return EmbeddingResponse(
        model="m", data=data, usage=Usage(prompt_tokens=4, total_tokens=4)
    )


def _one_vector_for_two_texts(**params: Any) -> EmbeddingResponse:
    return _vectors(input=["only"])


def _down(**params: Any) -> Any:
    raise ConnectionError("down")


def _cancelled(**params: Any) -> Any:
    raise asyncio.CancelledError()


class TestTheEmbedderTwin:
    @pytest.mark.parametrize(
        ("script", "errors"),
        [(_vectors, 0), (_one_vector_for_two_texts, 0), (_down, 1)],
    )
    async def test_both_names_give_one_result_one_request_and_one_count(
        self, script: Callable[..., Any], errors: int
    ) -> None:
        seen: dict[str, tuple[Any, ...]] = {}
        for mode in ("sync", "async"):
            embedder = LiteLLMEmbedder("text-embedding-3-small", api_key="sk-none")
            with (
                patch(f"{_EMBEDDER}.embedding", side_effect=script) as sync,
                patch(
                    f"{_EMBEDDER}.aembedding",
                    new_callable=AsyncMock,
                    side_effect=script,
                ) as async_,
                _observed(_Recorder()) as recorder,
            ):
                try:
                    if mode == "sync":
                        out: Any = embedder.embed(_TEXTS)
                    else:
                        out = await embedder.aembed(_TEXTS)
                except (Exception, asyncio.CancelledError) as exc:
                    out = (type(exc).__name__, str(exc))
            usage = embedder.usage()
            assert (usage.calls, usage.errors) == (1, errors)  # counted once, not twice
            assert (
                recorder.events == []
            )  # the embedder is not a request-observer client
            assert (sync.call_count, async_.call_count) == (
                (1, 0) if mode == "sync" else (0, 1)
            )
            call = sync.call_args if mode == "sync" else async_.call_args
            seen[mode] = (out, usage, call)
        assert seen["sync"] == seen["async"]

    async def test_a_cancel_counts_as_an_error_in_async_mode_only(self) -> None:
        started = asyncio.Event()

        async def hang(**_: Any) -> Any:
            started.set()
            await asyncio.Event().wait()

        for mode in ("sync", "async"):
            embedder = LiteLLMEmbedder("text-embedding-3-small", api_key="sk-none")
            with (
                patch(f"{_EMBEDDER}.embedding", side_effect=_cancelled),
                patch(f"{_EMBEDDER}.aembedding", new=hang),
                pytest.raises(asyncio.CancelledError),
            ):
                if mode == "sync":
                    embedder.embed(_TEXTS)  # a CancelledError out of the sync binding
                else:
                    task = asyncio.ensure_future(embedder.aembed(_TEXTS))
                    await asyncio.wait_for(started.wait(), 5)
                    task.cancel()
                    await task
            usage = embedder.usage()
            assert (usage.calls, usage.errors) == ((0, 0) if mode == "sync" else (1, 1))


# --- the classifier twins (step 6.2): one body per method ------------------------------


def _schema(*names: str) -> ClassificationSchema:
    intents = [IntentDefinition(name=n, description=f"The {n} intent") for n in names]
    return ClassificationSchema(intents=intents, fallback_intent=names[-1])


def _one(intent: str) -> str:
    return json.dumps({"reasoning": "r", "intent": intent, "confidence": 0.9})


_MANY = json.dumps({"intents": [{"intent": "buy", "confidence": 0.9}]})
_HIERARCHY = HierarchicalSchema(
    domain_schema=_schema("sales", "support"),
    intent_schemas={"sales": _schema("buy", "browse")},
)
# (method, the replies of a success, the requests a success sends)
_CLASSIFIERS = {
    "classify": (_one("buy"),),
    "classify_multi": (_MANY,),
    "hierarchical": (_one("sales"), _one("buy")),
}


def _classify_call(
    method: str, llm: LiteLLMInterface, transport: str
) -> Awaitable[Any]:
    if method == "hierarchical":
        hier = HierarchicalClassifier(_HIERARCHY, llm_interface=llm)
        name, owner = "classify", hier
    else:
        name, owner = method, Classifier(_schema("buy", "browse"), llm_interface=llm)
    if transport == "async":
        return getattr(owner, "a" + name)("hi")

    async def now() -> Any:
        return getattr(owner, name)("hi")

    return now()


class TestTheClassifierTwins:
    @pytest.mark.parametrize("method", list(_CLASSIFIERS))
    @pytest.mark.parametrize("scenario", ["success", "outage", "malformed"])
    async def test_both_names_give_one_result_one_request_path_and_one_meter(
        self, method: str, scenario: str
    ) -> None:
        seen: dict[str, tuple[Any, ...]] = {}
        for transport in ("sync", "async"):
            replies = iter(_CLASSIFIERS[method])

            def script(*, queue: Iterator[str] = replies, **_: Any) -> Any:
                if scenario == "outage":
                    raise RuntimeError("provider down")
                return _reply("not json" if scenario == "malformed" else next(queue))

            llm = _llm()
            with _providers(script) as (sync, async_), _observed(_Recorder()) as rec:
                out = await _outcome(_classify_call(method, llm, transport))
                landed, other = (
                    (sync, async_) if transport == "sync" else (async_, sync)
                )
                sent = len(_CLASSIFIERS[method]) if scenario == "success" else 1
                assert (landed.call_count, other.call_count) == (sent, 0)
            assert llm.usage().calls == sent  # the meter is fed once per request
            prompt = rec.events[0][1][0]["messages"][0]["content"]
            assert ("ranked by confidence" in prompt) == (method == "classify_multi")
            seen[transport] = (out, rec.shape(), llm.usage())
        assert seen["sync"] == seen["async"]
        out = seen["async"][0]
        failure = {
            "outage": "ClassificationError",
            "malformed": "ClassificationResponseError",
        }
        assert (
            out[1] == failure[scenario] if scenario in failure else out[0] == "returned"
        )

    async def test_a_patched_private_or_public_method_still_intercepts_each_mode(
        self,
    ) -> None:
        llm = _llm()
        classifier = Classifier(_schema("buy", "browse"), llm_interface=llm)
        raw = {"intent": "buy", "confidence": 0.9}
        reply = CompletionResponse(kind="final", text=_one("buy"))
        with (
            patch.object(Classifier, "_call_llm", return_value=raw) as call_llm,
            patch.object(
                Classifier, "_acall_llm", new_callable=AsyncMock, return_value=raw
            ) as acall_llm,
        ):
            assert classifier.classify("hi").intent == "buy"
            assert (call_llm.call_count, acall_llm.call_count) == (1, 0)
            assert (await classifier.aclassify("hi")).intent == "buy"
            assert (call_llm.call_count, acall_llm.call_count) == (1, 1)
        with (
            patch.object(llm, "complete", return_value=reply) as complete,
            patch.object(
                llm, "acomplete", new_callable=AsyncMock, return_value=reply
            ) as acomplete,
        ):
            assert classifier.classify("hi").intent == "buy"
            assert (complete.call_count, acomplete.call_count) == (1, 0)
            assert (await classifier.aclassify("hi")).intent == "buy"
            assert (complete.call_count, acomplete.call_count) == (1, 1)
