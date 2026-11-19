"""The async bodies of the pipeline answer like the sync ones (plan f150ec28, step 2).

Each body is written once as ``async def`` and the sync name is a ``run_sync`` wrapper, so a
row runs the same scripted turn twice: sync, and async on a double whose ``a*`` twins really
suspend (``_Suspending``, one loop turn per call). A row compares what a caller can see: the
reply, the data, the history, the state, the bytes of every request in order and the usage
meter. ``suspensions`` equal to the number of calls proves every model call went through its
``a`` twin in async mode (a body that called the sync name there would not suspend). The
mutants each row kills are named in its docstring.
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from fsm_llm import API
from fsm_llm.aio import run_sync
from fsm_llm.core.definitions import DataExtractionResponse
from fsm_llm.engine.kinds import CompletionExecutor, ConversationExecutor, StateExecutor
from fsm_llm.engine.kinds.conversation.executor import ConversationPass1
from fsm_llm.testing import ScriptedLLM, final_reply
from tests.test_fsm_llm import kind_corpus as kc

_QUESTION = [{"role": "user", "content": "What is 2 + 3?"}]


class _Suspending(ScriptedLLM):
    """A ``ScriptedLLM`` whose ``a*`` twins suspend once, then answer like the sync method."""

    suspensions = 0

    async def _hop(self) -> None:
        self.suspensions += 1
        await asyncio.sleep(0)

    async def acomplete(self, request):
        await self._hop()
        return await super().acomplete(request)

    async def agenerate_response(self, request):
        await self._hop()
        return await super().agenerate_response(request)

    async def aextract_field(self, request):
        await self._hop()
        return await super().aextract_field(request)

    async def aextract_bulk_data(self, request):
        await self._hop()
        return await super().aextract_bulk_data(request)


def _requests(llm: ScriptedLLM) -> list[tuple[str, str]]:
    """Every call in order with its request bytes (what ``test_kind_equality`` pins)."""
    return [
        (name, json.dumps(req.model_dump(mode="json"), sort_keys=True))
        for name, req in llm.call_history
    ]


def _bind(fsm: dict, llm: ScriptedLLM, context=None, advance: int = 0):
    """A started conversation: its pipeline, instance, id and current state."""
    api = API.from_definition(fsm, llm_interface=llm)
    cid, _ = api.start_conversation(initial_context=context)
    for _ in range(advance):
        api.advance(cid)
    manager = api.fsm_manager
    instance = manager.instances[cid]
    pipeline = manager._pipeline
    state = pipeline.get_state(instance, cid)
    return SimpleNamespace(
        llm=llm,
        pipeline=pipeline,
        instance=instance,
        cid=cid,
        state=state,
        calls0=len(llm.call_history),
    )


def _conversation(cls: type[ScriptedLLM]):
    llm = cls(extraction_data={"name": "Ada"}, responses={"greet": "Hello"})
    return _bind(kc._plain_fsm(), llm), "I am Ada"


def _completion(cls: type[ScriptedLLM]):
    llm = cls([final_reply("5")], default_response="Fine.")
    bound = _bind(
        kc._tool_fsm(), llm, {"_completion_messages": list(_QUESTION)}, advance=1
    )
    return bound, None


_KINDS = {
    "conversation": (_conversation, ConversationExecutor),
    "completion": (_completion, CompletionExecutor),
}


class TestExecutorTwins:
    @pytest.mark.parametrize("kind", sorted(_KINDS))
    async def test_aextract_answers_like_extract(self, kind):
        """Kills: a native ``aextract`` that differs from ``extract``, a model call that
        skips its ``a`` twin (suspensions), a call outside the request order."""
        make, executor_class = _KINDS[kind]
        want, message = make(ScriptedLLM)
        got, _ = make(_Suspending)
        expected = executor_class().extract(
            want.pipeline, want.instance, message, want.cid, want.state
        )
        actual = await executor_class().aextract(
            got.pipeline, got.instance, message, got.cid, got.state
        )
        assert actual.model_dump() == expected.model_dump()
        assert got.instance.last_extraction_response is actual
        assert _requests(got.llm) == _requests(want.llm)
        assert got.llm.usage() == want.llm.usage()
        assert got.llm.suspensions == len(got.llm.call_history) - got.calls0 > 0

    def test_a_patch_of_the_sync_name_still_intercepts_in_sync_mode(self):
        """Kills: a body that awaits the ``a`` twin directly instead of ``acall`` (sync mode
        would reach the poisoned twin and the field would come back failed)."""
        bound, message = _conversation(ScriptedLLM)
        with patch.object(bound.llm, "aextract_field", side_effect=AssertionError):
            response = ConversationPass1(bound.pipeline)._execute_data_extraction(
                bound.instance, message, bound.cid
            )
        assert response.extracted_data == {"name": "Ada"}
        sent = bound.llm.call_history[bound.calls0 :]
        assert [name for name, _ in sent] == ["extract_field"]


class _SyncOnly(StateExecutor):
    """The shape of the custom executors of the suite: only the sync methods."""

    def __init__(self) -> None:
        self.threads: list[int] = []

    def extract(self, pipeline, instance, user_message, conversation_id, state):
        self.threads.append(threading.get_ident())
        return DataExtractionResponse(extracted_data={"k": 1})

    def after_transition(self, *args):
        self.threads.append(threading.get_ident())


class TestDefaultTwinsOfAnExecutor:
    """Kills: no default (AttributeError), a default that never leaves the loop thread
    (the worker row), a default that always hops (the sync-mode row)."""

    async def test_async_mode_runs_the_sync_methods_on_a_worker_thread(self):
        executor = _SyncOnly()
        args = (None, None, "m", "c")
        got = await executor.aextract(*args, None)
        await executor.aafter_transition(*args, DataExtractionResponse(), "a", None)
        assert got.extracted_data == {"k": 1}
        assert len(executor.threads) == 2
        assert threading.get_ident() not in executor.threads

    def test_sync_mode_runs_them_inline_with_no_loop(self):
        executor = _SyncOnly()
        args = (None, None, "m", "c")
        got = run_sync(executor.aextract(*args, None))
        run_sync(executor.aafter_transition(*args, DataExtractionResponse(), "a", None))
        assert got.extracted_data == {"k": 1}
        assert executor.threads == [threading.get_ident()] * 2
