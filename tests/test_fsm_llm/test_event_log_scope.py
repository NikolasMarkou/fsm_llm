"""The turn scope of the runtime layer (plan 21e6f00d step 6): ``emit``, the request observer,
the engine fallback ``llm_call`` and ``scoped_iter``.

The provider is the patched ``fsm_llm.llm.providers.litellm.completion`` binding; the model
double is ``ScriptedLLM``. Bytes and counts, never timing. Unverified until the Windows run:
nothing here opens a file itself (the files are ``stores.jsonl``'s and ``blobs``').
"""

from __future__ import annotations

import contextvars
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from litellm.types.utils import Choices, Message, ModelResponse

from fsm_llm.core.definitions import (
    CompletionRequest,
    LLMResponseError,
    ResponseGenerationRequest,
)
from fsm_llm.llm import LiteLLMInterface, LLMInterface
from fsm_llm.llm import observer as obs
from fsm_llm.runtime import (
    EventLog,
    EventType,
    LogWriteError,
    bind,
    canonical_json,
    current_scope,
    emit,
    llm_call,
    scoped_iter,
)
from fsm_llm.runtime.events import EventPayloadError
from fsm_llm.runtime.scope import TurnScope
from fsm_llm.runtime.stores import jsonl
from fsm_llm.testing import ScriptedLLM, final_reply

ROOT = "root-1"
COMPLETION = "fsm_llm.llm.providers.litellm.completion"
_SECRETS = {
    "api_key": "sk-SECRET-key-0001",
    "api_base": "http://secret-host:1",
    "extra_headers": {"Authorization": "Bearer SECRET-bearer"},
}
_SECRET_TEXT = ("sk-SECRET-key-0001", "secret-host", "SECRET-bearer")


@pytest.fixture
def log(tmp_path: Path) -> Any:
    with EventLog(tmp_path) as opened:
        yield opened


def _reply(text: str = "Hello") -> ModelResponse:
    return ModelResponse(
        choices=[Choices(message=Message(content=text), finish_reason="stop")]
    )


def _llm() -> LiteLLMInterface:
    return LiteLLMInterface(model="gpt-4o", **_SECRETS)


def _request(text: str = "hi") -> ResponseGenerationRequest:
    return ResponseGenerationRequest(system_prompt="Greet.", user_message=text)


def _events(log: EventLog, *types: str) -> list[dict[str, Any]]:
    return list(log.query(ROOT, types=types or None))


def _tree_bytes(root: Path) -> bytes:
    return b"".join(p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file())


class _Wrapper(LLMInterface):
    """A custom interface around a real one: it never reaches ``_send`` by itself."""

    def __init__(self, inner: LiteLLMInterface) -> None:
        self.inner = inner

    def generate_response(self, request: Any) -> Any:
        return self.inner.generate_response(request)


class _Unserializable:
    def model_dump(self, mode: str = "python") -> Any:
        return {"bad": object()}


class TestWithoutAScope:
    def test_emit_writes_nothing_and_creates_no_file(self, tmp_path: Path) -> None:
        assert current_scope() is None
        assert emit(EventType.CONTEXT_RECORD, source="x") is None
        assert list(tmp_path.iterdir()) == []

    def test_llm_call_costs_nothing(self) -> None:
        class Boom:
            def model_dump(self, mode: str = "python") -> Any:
                raise AssertionError("the request must not be read without a scope")

        with llm_call(ScriptedLLM(), "completion", Boom()) as call:
            call.response(object())
            call.stream_done("text", abandoned=True)
        assert obs._ACTIVE.get() is None


class TestBinding:
    def test_bind_sets_and_resets_the_scope_and_the_observer(self, log: Any) -> None:
        with bind(log, ROOT, frame=2) as scope:
            assert current_scope() is scope
            assert (scope.log, scope.root_id, scope.frame) == (log, ROOT, 2)
            assert obs._ACTIVE.get() is scope.observer
        assert current_scope() is None and obs._ACTIVE.get() is None

    def test_bind_resets_on_an_exception_and_restores_an_outer_scope(
        self, log: Any
    ) -> None:
        with bind(log, ROOT) as outer:
            with pytest.raises(RuntimeError):
                with bind(log, ROOT, frame=1):
                    raise RuntimeError("inside")
            assert current_scope() is outer and obs._ACTIVE.get() is outer.observer
        assert current_scope() is None and obs._ACTIVE.get() is None

    def test_emit_appends_under_the_root_id_and_frame(self, log: Any) -> None:
        with bind(log, ROOT, frame=3):
            seq = emit("context/record", source="start")
        (event,) = _events(log)
        assert (event["seq"], event["frame"], event["source"]) == (seq, 3, "start")
        assert event["conversation_id"] == ROOT and event["type"] == "context/record"

    def test_emit_refuses_an_unknown_type_and_a_reserved_field(self, log: Any) -> None:
        with bind(log, ROOT):
            with pytest.raises(ValueError):
                emit("llm_request")
            with pytest.raises(EventPayloadError):
                emit(EventType.CONTEXT_RECORD, seq=9)

    def test_a_context_copy_thread_sees_the_scope_and_a_plain_thread_does_not(
        self, log: Any
    ) -> None:
        seen: dict[str, Any] = {}

        def work(name: str) -> None:
            seen[name] = current_scope()
            emit(EventType.CONTEXT_RECORD, source=name)

        with bind(log, ROOT) as scope:
            copied = contextvars.copy_context()
            for _name, run in (
                ("copied", lambda: copied.run(work, "copied")),
                ("plain", lambda: work("plain")),
            ):
                thread = threading.Thread(target=run)
                thread.start()
                thread.join()
        assert seen == {"copied": scope, "plain": None}
        assert [e["source"] for e in _events(log)] == ["copied"]


class TestScopedIter:
    def _generator(self, seen: list[Any]) -> Any:
        try:
            for item in range(3):
                seen.append(current_scope())
                yield item
        finally:
            seen.append(("finally", current_scope()))
            emit(EventType.CONTEXT_RECORD, source="closing")

    def test_the_scope_is_bound_inside_each_step_and_never_between(
        self, log: Any
    ) -> None:
        seen: list[Any] = []
        scope = TurnScope(log, ROOT)
        outside: list[Any] = []
        for item in scoped_iter(scope, self._generator(seen)):
            outside.append((item, current_scope(), obs._ACTIVE.get()))
        assert [o[1:] for o in outside] == [(None, None)] * 3
        assert seen[:3] == [scope] * 3 and seen[3] == ("finally", scope)
        assert [e["source"] for e in _events(log)] == ["closing"]

    def test_close_runs_the_inner_finally_inside_the_scope(self, log: Any) -> None:
        seen: list[Any] = []
        scope = TurnScope(log, ROOT)
        stream = scoped_iter(scope, self._generator(seen))
        assert next(stream) == 0
        stream.close()
        assert seen[-1] == ("finally", scope)
        assert [e["source"] for e in _events(log)] == ["closing"]
        assert current_scope() is None

    def test_a_stream_resumed_on_another_thread_raises_nothing(self, log: Any) -> None:
        seen: list[Any] = []
        stream = scoped_iter(TurnScope(log, ROOT), self._generator(seen))
        results: list[Any] = []

        def step() -> None:
            try:
                results.append((next(stream), current_scope()))
            except BaseException as exc:
                results.append(exc)

        for _ in range(2):
            thread = threading.Thread(target=step)
            thread.start()
            thread.join()
        assert results == [(0, None), (1, None)]
        stream.close()
        assert current_scope() is None

    def test_an_exception_of_the_inner_iterator_propagates_and_leaves_no_scope(
        self, log: Any
    ) -> None:
        def broken() -> Any:
            yield 1
            raise ValueError("inner")

        stream = scoped_iter(TurnScope(log, ROOT), broken())
        assert next(stream) == 1
        with pytest.raises(ValueError, match="inner"):
            next(stream)
        assert current_scope() is None and obs._ACTIVE.get() is None


class TestObserverPath:
    def _run(self, log: Any, reply: Any = None) -> tuple[LiteLLMInterface, dict]:
        llm, captured = _llm(), {}

        def completion(**kwargs: Any) -> Any:
            captured.update(kwargs)
            captured["seen"] = [e["type"] for e in log.read(ROOT)]
            captured["fsyncs"] = len(fsyncs)
            return reply or _reply()

        fsyncs: list[int] = []
        real = jsonl._fsync
        with (
            patch.object(jsonl, "_fsync", lambda fd: (fsyncs.append(fd), real(fd))[1]),
            patch(COMPLETION, completion),
            bind(log, ROOT),
        ):
            llm.generate_response(_request())
        return llm, captured

    def test_the_request_blob_is_the_allow_listed_provider_input(
        self, log: Any
    ) -> None:
        _, sent = self._run(log)
        request, response = _events(log)
        assert (request["type"], response["type"]) == ("llm/request", "llm/response")
        expected = {k: sent[k] for k in obs.OBSERVED_CALL_KEYS if k in sent}
        assert log.get_blob(request["prompt_ref"]) == canonical_json(expected)
        assert (request["source"], request["model"]) == ("provider", "gpt-4o")
        assert (request["call_type"], request["stream"]) == (
            "response_generation",
            False,
        )
        assert response["request_seq"] == request["seq"]
        assert response["finish_reason"] == "stop" and response["source"] == "provider"
        body = log.get_json(response["response_ref"])
        assert body["choices"][0]["message"]["content"] == "Hello"

    def test_the_request_is_durable_before_the_provider_call(self, log: Any) -> None:
        _, sent = self._run(log)
        assert sent["seen"] == ["llm/request"] and sent["fsyncs"] >= 1

    def test_credentials_never_reach_the_log_or_a_blob(
        self, log: Any, tmp_path: Path
    ) -> None:
        self._run(log)
        data = _tree_bytes(tmp_path)
        assert data and not [s for s in _SECRET_TEXT if s.encode() in data]

    def test_a_failing_request_checkpoint_means_no_provider_call_and_no_count(
        self, log: Any
    ) -> None:
        llm, called = _llm(), []
        with (
            patch.object(jsonl, "_fsync", side_effect=OSError("disk")),
            patch(COMPLETION, lambda **kw: called.append(kw)),
            bind(log, ROOT),
            pytest.raises(LLMResponseError, match="model call was not made"),
        ):
            llm.generate_response(_request())
        assert called == [] and llm.usage().calls == 0
        assert [e["type"] for e in _events(log)] == ["llm/request"]

    def test_a_provider_error_is_logged_by_type_and_status_only(
        self, log: Any, tmp_path: Path
    ) -> None:
        class Down(Exception):
            status_code = 503

        def completion(**kwargs: Any) -> Any:
            raise Down("provider said sk-SECRET-key-0001 is wrong")

        with patch(COMPLETION, completion), bind(log, ROOT):
            with pytest.raises(LLMResponseError):
                _llm().generate_response(_request())
        request, error = _events(log)
        assert error["type"] == "provider/error"
        assert (error["error_type"], error["status_code"]) == ("Down", 503)
        assert error["request_seq"] == request["seq"]
        assert b"provider said" not in _tree_bytes(tmp_path)
        assert b"sk-SECRET" not in _tree_bytes(tmp_path)

    @pytest.mark.parametrize("kind", ["deep dump", "raising dump"])
    def test_an_unrecordable_response_is_flagged_and_the_call_still_succeeds(
        self, log: Any, kind: str
    ) -> None:
        class Reply:
            def model_dump(self, mode: str = "python") -> Any:
                if kind == "raising dump":
                    raise RuntimeError("no dump")
                deep: Any = []
                for _ in range(40):
                    deep = [deep]
                return deep

        reply = Reply()
        with patch(COMPLETION, lambda **kw: reply), bind(log, ROOT):
            assert _llm()._send({"model": "m"}, call_type="x") is reply
        response = _events(log, "llm/response")[0]
        assert response["unrecorded"] is True and "response_ref" not in response

    def test_a_stream_logs_one_request_and_one_joined_response(self, log: Any) -> None:
        reads: list[str] = []

        class Chunks:
            def __init__(self) -> None:
                self.items = [
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content=t))]
                    )
                    for t in ("a", "b")
                ]

            def __iter__(self) -> Any:
                reads.append("iter")
                return iter(self.items)

        llm = _llm()
        with patch(COMPLETION, lambda **kw: Chunks()), bind(log, ROOT):
            with llm_call(llm, "response_generation", _request(), stream=True) as call:
                stream = llm._send(
                    {"model": "m", "messages": [], "stream": True}, call_type="x"
                )
                assert reads == []
                joined = "".join(c.choices[0].delta.content for c in stream)
                call.stream_done(joined, abandoned=False)
        request, response = _events(log)
        assert request["stream"] is True and response["stream"] is True
        assert (response["abandoned"], response["request_seq"]) == (
            False,
            request["seq"],
        )
        assert log.get_json(response["response_ref"])["text"] == "ab"
        assert [e["type"] for e in _events(log)] == ["llm/request", "llm/response"]

    def test_an_abandoned_stream_is_flagged(self, log: Any) -> None:
        llm = _llm()
        with patch(COMPLETION, lambda **kw: iter(())), bind(log, ROOT):
            with llm_call(llm, "response_generation", _request(), stream=True) as call:
                llm._send({"model": "m", "stream": True}, call_type="x")
                call.stream_done("par", abandoned=True)
        assert _events(log, "llm/response")[0]["abandoned"] is True


class TestEngineFallback:
    def _complete(self, llm: ScriptedLLM) -> Any:
        request = CompletionRequest(messages=[{"role": "user", "content": "hi"}])
        with llm_call(llm, "completion", request) as call:
            reply = llm.complete(request)
            call.response(reply)
        return request, reply

    def test_a_double_gets_one_request_and_one_response_from_the_engine(
        self, log: Any
    ) -> None:
        llm = ScriptedLLM(replies=[final_reply("done")])
        with bind(log, ROOT):
            request, _ = self._complete(llm)
        sent, got = _events(log)
        assert (sent["type"], got["type"]) == ("llm/request", "llm/response")
        assert (sent["source"], sent["call_type"], sent["stream"]) == (
            "engine",
            "completion",
            False,
        )
        assert log.get_json(sent["prompt_ref"]) == {
            "source": "engine",
            "request_type": "CompletionRequest",
            "request": request.model_dump(mode="json"),
        }
        assert got["request_seq"] == sent["seq"] and got["source"] == "engine"
        assert log.get_json(got["response_ref"])["response"]["text"] == "done"

    def test_the_request_is_durable_before_the_double_runs(self, log: Any) -> None:
        seen: list[Any] = []
        llm = ScriptedLLM(replies=[final_reply("x")])
        real = llm.complete

        def spy(request: Any) -> Any:
            seen.append([e["type"] for e in log.read(ROOT)])
            return real(request)

        llm.complete = spy  # type: ignore[method-assign]
        with (
            patch.object(jsonl, "_fsync", wraps=jsonl._fsync) as fsync,
            bind(log, ROOT),
        ):
            self._complete(llm)
            assert fsync.call_count >= 1
        assert seen == [["llm/request"]]

    def test_a_failing_checkpoint_means_the_double_is_not_called(
        self, log: Any
    ) -> None:
        llm = ScriptedLLM(replies=[final_reply("x")])
        with (
            patch.object(jsonl, "_fsync", side_effect=OSError("disk")),
            bind(log, ROOT),
            pytest.raises(LogWriteError, match="model call was not made"),
        ):
            self._complete(llm)
        assert llm.requests == [] and obs._ACTIVE.get() is None

    def test_an_unserializable_request_fails_closed(self, log: Any) -> None:
        ran: list[int] = []
        with bind(log, ROOT), pytest.raises(LogWriteError, match="not made"):
            with llm_call(ScriptedLLM(), "completion", _Unserializable()):
                ran.append(1)
        assert ran == [] and _events(log) == []

    def test_an_error_is_logged_by_type_name_only_and_propagates(
        self, log: Any, tmp_path: Path
    ) -> None:
        llm = ScriptedLLM(replies=[LLMResponseError("provider text sk-LEAK-0001")])
        with bind(log, ROOT), pytest.raises(LLMResponseError):
            self._complete(llm)
        request, error = _events(log)
        assert error["type"] == "provider/error"
        assert error["error_type"] == "LLMResponseError"
        assert error["request_seq"] == request["seq"] and error["source"] == "engine"
        assert b"sk-LEAK" not in _tree_bytes(tmp_path)

    def test_a_response_after_the_reply_is_written_once(self, log: Any) -> None:
        llm = ScriptedLLM(replies=[final_reply("x")])
        with bind(log, ROOT):
            request = CompletionRequest(messages=[{"role": "user", "content": "hi"}])
            with llm_call(llm, "completion", request) as call:
                call.response(llm.complete(request))
                call.response(object())
                call.stream_done("late")
        assert [e["type"] for e in _events(log)] == ["llm/request", "llm/response"]

    def test_a_failing_response_write_never_hides_the_reply(self, log: Any) -> None:
        real = log.append

        def append(conversation_id: str, kind: Any, **fields: Any) -> int:
            if EventType.from_string(kind) is EventType.LLM_RESPONSE:
                raise LogWriteError("disk")
            return real(conversation_id, kind, **fields)

        llm = ScriptedLLM(replies=[final_reply("x")])
        with patch.object(log, "append", append), bind(log, ROOT):
            _, reply = self._complete(llm)
        assert reply.text == "x"
        assert [e["type"] for e in _events(log)] == ["llm/request"]

    def test_an_observed_interface_is_logged_once_by_the_observer(
        self, log: Any
    ) -> None:
        llm = _llm()
        with patch(COMPLETION, lambda **kw: _reply()), bind(log, ROOT):
            with llm_call(llm, "response_generation", _request()) as call:
                call.response(llm.generate_response(_request()))
        events = _events(log)
        assert [e["type"] for e in events] == ["llm/request", "llm/response"]
        assert events[0]["source"] == "provider"

    def test_a_wrapper_around_a_real_interface_logs_one_pair_not_two(
        self, log: Any
    ) -> None:
        calls: list[Any] = []
        wrapper = _Wrapper(_llm())

        def completion(**kwargs: Any) -> Any:
            calls.append(obs._ACTIVE.get())
            return _reply()

        with patch(COMPLETION, completion), bind(log, ROOT):
            with llm_call(wrapper, "response_generation", _request()) as call:
                call.response(wrapper.generate_response(_request()))
            assert obs._ACTIVE.get() is current_scope().observer
        events = _events(log)
        assert [e["type"] for e in events] == ["llm/request", "llm/response"]
        assert events[0]["source"] == "engine" and calls == [None]
        assert wrapper.inner.usage().calls == 1
