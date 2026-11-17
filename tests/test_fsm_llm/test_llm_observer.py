"""The request observer of ``LiteLLMInterface._send`` (plan 21e6f00d, step 4, D-002).

The provider is the patched ``fsm_llm.llm.providers.litellm.completion`` binding; the
observer is a recording double. Nothing here reads a clock or reaches a network.
"""

from __future__ import annotations

import ast
import contextvars
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
from litellm.types.utils import Choices, Message, ModelResponse

from fsm_llm.core.definitions import (
    CompletionRequest,
    LLMResponseError,
    ResponseGenerationRequest,
)
from fsm_llm.llm import LiteLLMInterface, LLMInterface
from fsm_llm.llm import observer as obs
from fsm_llm.testing import ScriptedLLM

_SRC = Path(__file__).resolve().parents[2] / "src" / "fsm_llm"
_LITELLM = _SRC / "llm" / "providers" / "litellm.py"
_SECRETS = {
    "api_key": "sk-SECRET",
    "api_base": "http://secret-host:1",
    "extra_headers": {"Authorization": "Bearer SECRET"},
    "custom_kwarg": "custom",
}


class _Recorder:
    """Records every call; ``raises`` names the hooks that raise."""

    def __init__(self, raises: tuple[str, ...] = ()) -> None:
        self.events: list[tuple[str, Any]] = []
        self.raises = raises

    def _hit(self, name: str, payload: Any) -> None:
        self.events.append((name, payload))
        if name in self.raises:
            raise RuntimeError(f"{name} boom")

    def begin_send(self, call: dict[str, Any], *, call_type: str) -> object:
        self._hit("begin", (call, call_type))
        return ("token", len(self.events))

    def end_send(self, token: object, response: Any) -> None:
        self._hit("end", (token, response))

    def fail_send(self, token: object, error: BaseException) -> None:
        self._hit("fail", (token, error))

    def names(self) -> list[str]:
        return [name for name, _ in self.events]


def _reply(text: str = "Hello") -> ModelResponse:
    return ModelResponse(
        choices=[Choices(message=Message(content=text), finish_reason="stop")]
    )


def _llm(**kwargs: Any) -> LiteLLMInterface:
    return LiteLLMInterface(model="gpt-4o", **{**_SECRETS, **kwargs})


def _generate(llm: LiteLLMInterface) -> Any:
    return llm.generate_response(
        ResponseGenerationRequest(system_prompt="Greet.", user_message="hi")
    )


def _call(llm: LiteLLMInterface, provider: Any, fn: Any = _generate) -> Any:
    with (
        patch("fsm_llm.llm.providers.litellm.completion", side_effect=provider),
        patch("fsm_llm.llm.providers.litellm.get_supported_openai_params") as params,
    ):
        params.return_value = []
        return fn(llm)


def _observed(recorder: Any, llm: LiteLLMInterface, provider: Any, fn: Any = _generate):
    token = obs.install_observer(recorder)
    try:
        return _call(llm, provider, fn)
    finally:
        obs.reset_observer(token)


class TestTheAllowList:
    def test_the_observer_sees_the_allow_listed_request_and_the_reply(self) -> None:
        recorder, seen, llm = _Recorder(), [], _llm()
        reply = _reply()

        def provider(**kw: Any) -> ModelResponse:
            seen.append(kw)
            return reply

        _observed(recorder, llm, provider)
        assert recorder.names() == ["begin", "end"]
        call, call_type = recorder.events[0][1]
        assert call_type == "response_generation"
        assert call == {k: seen[0][k] for k in obs.OBSERVED_CALL_KEYS if k in seen[0]}
        assert {"model", "messages", "temperature", "max_tokens"} <= set(call)
        token, response = recorder.events[1][1]
        assert token == ("token", 1) and response is reply

    def test_credentials_headers_and_kwargs_never_reach_the_observer(self) -> None:
        recorder, seen = _Recorder(), []

        def provider(**kw: Any) -> ModelResponse:
            seen.append(kw)
            return _reply()

        _observed(recorder, _llm(timeout=5.0, retries=2), provider)
        sent = set(seen[0])
        # non-vacuity: the provider DID get every one of them
        assert (set(_SECRETS) | {"timeout", "max_retries"}) <= sent
        call = recorder.events[0][1][0]
        assert not (set(call) & (set(_SECRETS) | {"timeout", "max_retries"}))
        assert "sk-SECRET" not in repr(recorder.events)

    @pytest.mark.parametrize("model", ["gpt-4o", "ollama_chat/qwen3:4b"])
    def test_every_key_the_builder_sets_is_allowed_or_named_as_excluded(
        self, model: str
    ) -> None:
        # The drift guard: a key the request builder starts to set must be decided
        # (allow-listed or excluded here), never reach a log by default.
        excluded = set(_SECRETS) | {"timeout", "max_retries"}
        llm = LiteLLMInterface(model=model, timeout=5.0, retries=1, **_SECRETS)
        msgs = [{"role": "user", "content": "hi"}]
        tool = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
        fmt = {"type": "json_object"}
        shapes: list[dict[str, Any]] = [
            {},
            {"stream": True},
            {"tools": tool},
            {"tools": tool, "tool_choice": "required"},
            {"response_format": fmt, "temperature": 0.1, "max_tokens": 9},
        ]
        keys: set[str] = set()
        with patch.object(
            LiteLLMInterface,
            "_supported_openai_params",
            return_value=["response_format"],
        ):
            for call_type in ("response_generation", "data_extraction"):
                for shape in shapes:
                    if "tools" in shape and call_type == "data_extraction":
                        continue
                    built = llm._build_call_params(msgs, call_type, **shape)
                    keys |= set(built)
        assert keys - excluded <= set(obs.OBSERVED_CALL_KEYS), keys
        assert {"tools", "tool_choice", "stream", "response_format"} <= keys

    def test_a_planted_key_outside_the_allow_list_is_dropped(self) -> None:
        recorder = _Recorder()
        token = obs.install_observer(recorder)
        try:
            handle = obs.begin_send(
                {"model": "m", "messages": [], "api_key": "k", "new_kwarg": 1},
                call_type="x",
            )
        finally:
            obs.reset_observer(token)
        assert handle is not None
        assert recorder.events[0][1][0] == {"model": "m", "messages": []}


class TestFailClosedAndNeverBreak:
    def test_a_raising_begin_means_no_provider_call_and_no_count(self) -> None:
        recorder, llm, provider = _Recorder(raises=("begin",)), _llm(), Mock()
        with pytest.raises(LLMResponseError, match="begin boom"):
            _observed(recorder, llm, provider)
        provider.assert_not_called()
        assert llm.usage().calls == 0 and llm.usage().errors == 0

    def test_a_raising_end_changes_neither_the_reply_nor_the_count(self) -> None:
        plain, llm = _llm(), _llm()
        expected = _call(plain, lambda **kw: _reply("same"))
        got = _observed(_Recorder(raises=("end",)), llm, lambda **kw: _reply("same"))
        assert got == expected
        assert llm.usage().calls == plain.usage().calls == 1

    def test_a_provider_error_reaches_fail_send_and_propagates_unchanged(self) -> None:
        boom = RuntimeError("provider down")
        for recorder in (_Recorder(), _Recorder(raises=("fail",))):
            llm = _llm()

            def provider(**kw: Any) -> None:
                raise boom

            token = obs.install_observer(recorder)
            try:
                with patch("fsm_llm.llm.providers.litellm.completion", provider):
                    with pytest.raises(RuntimeError) as caught:
                        llm._send({"model": "m"}, call_type="x")
            finally:
                obs.reset_observer(token)
            assert caught.value is boom
            assert recorder.names() == ["begin", "fail"]
            assert recorder.events[1][1] == (("token", 1), boom)
            assert llm.usage().errors == 1 and llm.usage().calls == 1

    def test_complete_sees_tools_and_a_failure_is_the_documented_error(self) -> None:
        recorder, tool = _Recorder(), [{"type": "function", "function": {"name": "t"}}]
        request = CompletionRequest(
            messages=[{"role": "user", "content": "hi"}], tools=tool
        )

        def fn(llm: LiteLLMInterface) -> Any:
            return llm.complete(request)

        _observed(recorder, _llm(), lambda **kw: _reply(), fn)
        call, call_type = recorder.events[0][1]
        assert call["tools"] == tool and call["tool_choice"] == "auto"
        assert call_type == request.call_type

        failing = _Recorder()
        with pytest.raises(LLMResponseError):
            _observed(failing, _llm(), Mock(side_effect=OSError("x")), fn)
        assert failing.names() == ["begin", "fail"]


class TestStreamingAndRetry:
    def test_a_streamed_call_begins_once_and_ends_with_the_unread_iterator(
        self,
    ) -> None:
        reads: list[str] = []

        class _Stream:
            def __iter__(self) -> _Stream:
                reads.append("iter")
                return self

            def __next__(self) -> Any:
                reads.append("next")
                raise StopIteration

        stream, recorder, llm = _Stream(), _Recorder(), _llm()
        token = obs.install_observer(recorder)
        try:
            with patch("fsm_llm.llm.providers.litellm.completion", lambda **kw: stream):
                assert (
                    llm._send({"model": "m", "stream": True}, call_type="x") is stream
                )
        finally:
            obs.reset_observer(token)
        assert recorder.names() == ["begin", "end"] and reads == []
        assert recorder.events[1][1][1] is stream
        assert recorder.events[0][1][0]["stream"] is True
        assert llm.usage().by_kind["stream"].calls == 1

    def test_generate_response_stream_reports_a_streamed_request(self) -> None:
        recorder, chunks = _Recorder(), []
        for text in ("a", "b"):
            delta = SimpleNamespace(content=text)
            chunks.append(SimpleNamespace(choices=[SimpleNamespace(delta=delta)]))

        def fn(llm: LiteLLMInterface) -> list[str]:
            return list(
                llm.generate_response_stream(
                    ResponseGenerationRequest(system_prompt="s", user_message="u")
                )
            )

        assert _observed(recorder, _llm(), lambda **kw: iter(chunks), fn) == ["a", "b"]
        assert recorder.names() == ["begin", "end"]
        assert recorder.events[0][1][0]["stream"] is True

    def test_the_apology_retry_is_its_own_request_pair(self) -> None:
        replies = iter([_reply('{"message": ""}'), _reply("Hello")])
        recorder, llm = _Recorder(), _llm()
        _observed(recorder, llm, lambda **kw: next(replies))
        assert llm.apology_retry_count == 1
        assert recorder.names() == ["begin", "end", "begin", "end"]


class TestNoObserverAndScope:
    def test_without_an_observer_nothing_is_built(self) -> None:
        class _Tripwire(tuple):
            def __iter__(self):
                raise AssertionError("the allow-listed copy was built")

        with patch.object(obs, "OBSERVED_CALL_KEYS", _Tripwire(obs.OBSERVED_CALL_KEYS)):
            assert obs.begin_send({"model": "m"}, call_type="x") is None
            assert _call(_llm(), lambda **kw: _reply()) is not None
        obs.end_send(None, object())
        obs.fail_send(None, RuntimeError())

    def test_the_variable_is_reset_after_the_scope(self) -> None:
        recorder = _Recorder()
        token = obs.install_observer(recorder)
        assert obs.begin_send({"model": "m"}, call_type="x") is not None
        obs.reset_observer(token)
        assert obs.begin_send({"model": "m"}, call_type="x") is None
        assert len(recorder.events) == 1

    def test_a_copied_context_carries_the_observer_to_a_thread_and_a_bare_thread_not(
        self,
    ) -> None:
        recorder, seen = _Recorder(), {}
        token = obs.install_observer(recorder)
        try:

            def probe(name: str) -> None:
                seen[name] = obs.begin_send({"model": "m"}, call_type="x") is not None

            copied = contextvars.copy_context()
            threads = [
                threading.Thread(target=copied.run, args=(probe, "copied")),
                threading.Thread(target=probe, args=("bare",)),
            ]
            for thread in threads:
                thread.start()
                thread.join()
        finally:
            obs.reset_observer(token)
        assert seen == {"copied": True, "bare": False}

    def test_none_suppresses_an_installed_observer(self) -> None:
        recorder = _Recorder()
        outer = obs.install_observer(recorder)
        inner = obs.install_observer(None)
        assert obs.begin_send({"model": "m"}, call_type="x") is None
        obs.reset_observer(inner)
        assert obs.begin_send({"model": "m"}, call_type="x") is not None
        obs.reset_observer(outer)

    def test_is_observed_is_true_only_for_interfaces_that_reach_send(self) -> None:
        class _Sub(LiteLLMInterface):
            pass

        assert obs.is_observed(_llm()) and obs.is_observed(_Sub(model="gpt-4o"))
        assert not obs.is_observed(Mock(spec=LLMInterface))
        assert not obs.is_observed(Mock())
        assert not obs.is_observed(ScriptedLLM())


class TestTheSharedSendPathStaysIntact:
    def test_litellm_is_still_imported_in_one_module_and_its_size_has_room(
        self,
    ) -> None:
        importers = []
        for path in _SRC.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                    if isinstance(node, ast.ImportFrom) and not node.level
                    else []
                )
                if any(n == "litellm" or n.startswith("litellm.") for n in names):
                    importers.append(path.relative_to(_SRC).as_posix())
        assert set(importers) == {"llm/providers/litellm.py"}
        assert len(_LITELLM.read_text(encoding="utf-8").splitlines()) <= 790

    def test_send_hooks_surround_the_one_provider_call_and_the_meter_order_holds(
        self,
    ) -> None:
        source = _LITELLM.read_text(encoding="utf-8")
        send = next(
            n
            for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.FunctionDef) and n.name == "_send"
        )
        calls = [
            (n.lineno, n.func.id if isinstance(n.func, ast.Name) else n.func.attr)
            for n in ast.walk(send)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name | ast.Attribute)
            and (n.func.id if isinstance(n.func, ast.Name) else n.func.attr)
            in {
                "begin_send",
                "completion",
                "record_error",
                "fail_send",
                "record",
                "end_send",
            }
        ]
        order = [name for _, name in sorted(calls)]
        assert order == [
            "begin_send",
            "completion",
            "record_error",
            "fail_send",
            "record",
            "end_send",
        ]
        # D-004 of plan 944e2692: the anchor text is untouched; counted only here
        flat = " ".join(source.replace("# ", " ").split())
        assert (
            "DECISION plan-2026-10-01T093600-944e2692/D-004: the meter is per "
            "instance and fed here only. Do NOT add a process-global counter "
            "(shared state across conversations and tests), do NOT count in the "
            "public methods (a second count site drifts from the one send path), "
            "and do NOT read usage off a stream (its chunks carry none on Ollama). "
            "Readers that want counts own (inject) the interface. See D-004."
        ) in flat
        assert source.count("meter.record(") == 1
        assert "observes_requests = True" in source
