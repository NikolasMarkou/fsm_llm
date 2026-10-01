"""Usage counters of ``LiteLLMInterface`` (plan 944e2692, step 3, D-004).

Every provider request goes through the one send path (``_send``), which
counts it once on the interface instance, per call kind. The provider is the
``fsm_llm.llm.completion`` binding, patched; replies are real litellm
``ModelResponse`` objects unless a test needs another usage shape.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from litellm.types.utils import Choices, Message, ModelResponse
from pydantic import ValidationError

import fsm_llm
from fsm_llm.definitions import (
    BulkExtractionRequest,
    CompletionRequest,
    FieldExtractionRequest,
    LLMCallCounts,
    LLMResponseError,
    LLMUsage,
    ResponseGenerationRequest,
)
from fsm_llm.llm import LiteLLMInterface

_COUNTER_FIELDS = (
    "calls",
    "errors",
    "usage_missing",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


def _reply(content: str, usage: Any = None) -> ModelResponse:
    kwargs: dict[str, Any] = {}
    if usage is not None:
        kwargs["usage"] = usage
    return ModelResponse(
        choices=[Choices(message=Message(content=content), finish_reason="stop")],
        **kwargs,
    )


def _usage(prompt: int, completion: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _patched(provider: Any):
    """Patch the provider binding and the supported-params lookup."""
    return (
        patch("fsm_llm.llm.completion", side_effect=provider),
        patch("fsm_llm.llm.get_supported_openai_params", return_value=[]),
    )


def _run(llm: LiteLLMInterface, provider: Any, fn: Any) -> Any:
    send, params = _patched(provider)
    with send, params:
        return fn(llm)


def _generate(llm: LiteLLMInterface) -> Any:
    return llm.generate_response(
        ResponseGenerationRequest(system_prompt="Greet.", user_message="hi")
    )


def _complete(llm: LiteLLMInterface, call_type: str = "completion") -> Any:
    return llm.complete(
        CompletionRequest(
            messages=[{"role": "user", "content": "hi"}], call_type=call_type
        )
    )


def _counts(usage: LLMUsage | LLMCallCounts) -> dict[str, int]:
    return {name: getattr(usage, name) for name in _COUNTER_FIELDS}


class TestOneCallOneCount:
    def test_a_fresh_interface_has_zero_usage(self):
        usage = LiteLLMInterface(model="gpt-4o").usage()
        assert usage == LLMUsage()
        assert usage.by_kind == {}

    def test_generate_counts_one_call_with_tokens(self):
        llm = LiteLLMInterface(model="gpt-4o")
        _run(llm, lambda **kw: _reply("Hello there", _usage(11, 4)), _generate)
        usage = llm.usage()
        assert _counts(usage) == {
            "calls": 1,
            "errors": 0,
            "usage_missing": 0,
            "prompt_tokens": 11,
            "completion_tokens": 4,
            "total_tokens": 15,
        }
        assert list(usage.by_kind) == ["generate"]
        assert _counts(usage.by_kind["generate"]) == _counts(usage)

    def test_every_call_path_has_its_kind(self):
        llm = LiteLLMInterface(model="gpt-4o")
        replies = {
            "field_extraction": '{"field_name": "name", "value": "Ada", '
            '"confidence": 0.9}',
            "data_extraction": '{"extracted_data": {"name": "Ada"}, "confidence": 0.9}',
            "classification": '{"intent": "buy", "confidence": 0.9}',
        }

        def provider(**kwargs: Any) -> ModelResponse:
            system = kwargs["messages"][0]["content"]
            for marker, text in replies.items():
                if system == marker:
                    return _reply(text, _usage(2, 1))
            return _reply("Hello", _usage(2, 1))

        def calls(llm: LiteLLMInterface) -> None:
            _generate(llm)
            llm.extract_field(
                FieldExtractionRequest(
                    system_prompt="field_extraction",
                    user_message="I am Ada",
                    field_name="name",
                    field_type="str",
                )
            )
            llm.extract_bulk_data(
                BulkExtractionRequest(
                    system_prompt="data_extraction", user_message="I am Ada"
                )
            )
            llm.complete(
                CompletionRequest(
                    messages=[
                        {"role": "system", "content": "classification"},
                        {"role": "user", "content": "buy it"},
                    ],
                    call_type="classification",
                )
            )
            _complete(llm)

        _run(llm, provider, calls)
        usage = llm.usage()
        assert {kind: c.calls for kind, c in usage.by_kind.items()} == {
            "classify": 1,
            "complete": 1,
            "extract": 2,
            "generate": 1,
        }
        assert usage.calls == 5
        assert usage.total_tokens == 15

    def test_the_apology_retry_is_a_second_counted_call(self):
        llm = LiteLLMInterface(model="gpt-4o")
        replies = iter([_reply('{"message": ""}'), _reply("Hello")])
        _run(llm, lambda **kw: next(replies), _generate)
        assert llm.apology_retry_count == 1
        assert llm.usage().by_kind["generate"].calls == 2


class TestErrorsAndStreams:
    def test_a_raising_call_counts_as_a_call_and_an_error(self):
        llm = LiteLLMInterface(model="gpt-4o")

        def provider(**kwargs: Any) -> Any:
            raise ConnectionError("provider down")

        with pytest.raises(LLMResponseError) as info:
            _run(llm, provider, _complete)
        assert isinstance(info.value.__cause__, ConnectionError)
        usage = llm.usage()
        assert _counts(usage) == {
            "calls": 1,
            "errors": 1,
            "usage_missing": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        assert usage.by_kind["complete"].errors == 1

    def test_a_stream_counts_as_a_call_with_usage_missing(self):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Hi"
        chunk.choices[0].delta.reasoning_content = None
        llm = LiteLLMInterface(model="gpt-4o")
        request = ResponseGenerationRequest(system_prompt="Greet.", user_message="hi")

        class _Stream:
            """A stream wrapper that exposes a ``usage`` attribute: the send
            path must not read it (no stream carries usage before it is
            consumed)."""

            usage = _usage(9, 9)

            def __iter__(self) -> Any:
                return iter([chunk])

        assert _run(
            llm,
            lambda **kw: _Stream(),
            lambda llm: list(llm.generate_response_stream(request)),
        ) == ["Hi"]
        usage = llm.usage()
        assert list(usage.by_kind) == ["stream"]
        assert usage.calls == 1
        assert usage.usage_missing == 1
        assert usage.errors == 0
        assert usage.total_tokens == 0

    def test_a_stream_that_fails_to_start_is_an_error(self):
        llm = LiteLLMInterface(model="gpt-4o")
        request = ResponseGenerationRequest(system_prompt="Greet.", user_message="hi")

        def provider(**kwargs: Any) -> Any:
            raise TimeoutError("slow")

        with pytest.raises(LLMResponseError):
            _run(llm, provider, lambda llm: list(llm.generate_response_stream(request)))
        stream = llm.usage().by_kind["stream"]
        assert (stream.calls, stream.errors, stream.usage_missing) == (1, 1, 0)


class TestUsageShapes:
    @pytest.mark.parametrize(
        ("usage", "expected"),
        [
            pytest.param(_usage(7, 3), (7, 3, 10), id="dict"),
            pytest.param(
                SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7),
                (5, 2, 7),
                id="object",
            ),
            pytest.param(
                {"prompt_tokens": 4, "completion_tokens": 6}, (4, 6, 10), id="no-total"
            ),
            pytest.param(
                {"prompt_tokens": "4", "completion_tokens": None, "total_tokens": True},
                (0, 0, 0),
                id="not-ints",
            ),
            pytest.param(
                {"prompt_tokens": -3, "completion_tokens": 2}, (0, 2, 2), id="negative"
            ),
        ],
    )
    def test_usage_is_read_defensively(self, usage, expected):
        reply = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )
        llm = LiteLLMInterface(model="gpt-4o")
        _run(llm, lambda **kw: reply, _complete)
        snapshot = llm.usage()
        assert (
            snapshot.prompt_tokens,
            snapshot.completion_tokens,
            snapshot.total_tokens,
        ) == expected
        assert (snapshot.calls, snapshot.usage_missing) == (1, 0)

    def test_a_reply_without_usage_counts_as_usage_missing(self):
        llm = LiteLLMInterface(model="gpt-4o")
        _run(llm, lambda **kw: _reply("ok"), _complete)
        usage = llm.usage()
        assert (usage.calls, usage.usage_missing, usage.total_tokens) == (1, 1, 0)

    def test_a_dict_reply_with_dict_usage(self):
        reply = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": _usage(1, 1),
        }
        llm = LiteLLMInterface(model="gpt-4o")
        send, params = _patched(lambda **kw: reply)
        with send, params, pytest.raises(LLMResponseError):
            # The reply reader refuses a dict reply; the call is still metered.
            _complete(llm)
        usage = llm.usage()
        assert (usage.calls, usage.errors, usage.total_tokens) == (1, 0, 2)


class TestSnapshotAndReset:
    def test_the_snapshot_is_a_frozen_copy(self):
        llm = LiteLLMInterface(model="gpt-4o")
        _run(llm, lambda **kw: _reply("ok", _usage(1, 1)), _complete)
        before = llm.usage()
        with pytest.raises(ValidationError):
            before.calls = 9
        _run(llm, lambda **kw: _reply("ok", _usage(1, 1)), _complete)
        assert before.calls == 1
        assert before.by_kind["complete"].calls == 1
        assert llm.usage().calls == 2

    def test_reset_returns_the_cleared_counts(self):
        llm = LiteLLMInterface(model="gpt-4o")
        _run(llm, lambda **kw: _reply("ok", _usage(2, 3)), _complete)
        cleared = llm.reset_usage()
        assert (cleared.calls, cleared.total_tokens) == (1, 5)
        assert llm.usage() == LLMUsage()
        _run(llm, lambda **kw: _reply("ok", _usage(2, 3)), _complete)
        assert llm.usage().calls == 1

    def test_instances_count_separately(self):
        first = LiteLLMInterface(model="gpt-4o")
        second = LiteLLMInterface(model="gpt-4o")
        _run(first, lambda **kw: _reply("ok"), _complete)
        assert first.usage().calls == 1
        assert second.usage().calls == 0

    def test_an_instance_built_with_new_counts(self):
        llm = LiteLLMInterface.__new__(LiteLLMInterface)
        llm.model = "gpt-4o"
        llm.temperature = 0.5
        llm.max_tokens = 100
        llm.timeout = None
        llm.retries = 0
        llm.kwargs = {}
        assert llm.usage().calls == 0
        _run(llm, lambda **kw: _reply("ok"), _complete)
        assert llm.usage().calls == 1

    def test_usage_types_are_public(self):
        assert fsm_llm.LLMUsage is LLMUsage
        assert fsm_llm.LLMCallCounts is LLMCallCounts
        assert {"LLMUsage", "LLMCallCounts"} <= set(fsm_llm.__all__)


class TestConcurrency:
    THREADS = 8
    CALLS = 50

    def test_counts_sum_exactly_under_concurrent_calls(self):
        """One interface shared by 8 threads, 50 calls each, two kinds."""
        llm = LiteLLMInterface(model="gpt-4o")
        barrier = threading.Barrier(self.THREADS)
        failures: list[BaseException] = []

        def provider(**kwargs: Any) -> ModelResponse:
            return _reply("Hello", _usage(3, 2))

        def worker(index: int) -> None:
            try:
                barrier.wait()
                for _ in range(self.CALLS):
                    if index % 2:
                        _generate(llm)
                    else:
                        _complete(llm)
            except BaseException as exc:  # pragma: no cover - reported below
                failures.append(exc)

        send, params = _patched(provider)
        with send, params:
            threads = [
                threading.Thread(target=worker, args=(i,)) for i in range(self.THREADS)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        assert failures == []
        usage = llm.usage()
        total = self.THREADS * self.CALLS
        assert usage.calls == total
        assert usage.prompt_tokens == 3 * total
        assert usage.completion_tokens == 2 * total
        assert usage.total_tokens == 5 * total
        assert usage.by_kind["generate"].calls == total // 2
        assert usage.by_kind["complete"].calls == total // 2
        for name in _COUNTER_FIELDS:
            assert getattr(usage, name) == sum(
                getattr(kind, name) for kind in usage.by_kind.values()
            )

    def test_reset_under_concurrent_calls_loses_nothing(self):
        llm = LiteLLMInterface(model="gpt-4o")
        harvested: list[int] = []
        done = threading.Event()

        def reader() -> None:
            while not done.is_set():
                harvested.append(llm.reset_usage().calls)

        def worker() -> None:
            for _ in range(self.CALLS):
                _complete(llm)

        send, params = _patched(lambda **kw: _reply("ok"))
        with send, params:
            watcher = threading.Thread(target=reader)
            watcher.start()
            workers = [threading.Thread(target=worker) for _ in range(self.THREADS)]
            for thread in workers:
                thread.start()
            for thread in workers:
                thread.join()
            done.set()
            watcher.join()

        assert sum(harvested) + llm.usage().calls == self.THREADS * self.CALLS
