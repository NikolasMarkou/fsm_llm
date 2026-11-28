"""An interrupt or a cancel after Pass 2 added the reply rolls the history back whole (plan
8525c797, step 7.1, concern 5 of the review round).

The snapshot of a turn holds the exchanges and the summary, so the history after a failed turn is
the pre-turn history and the manager arm only pops the user message that was added before the
snapshot. The window is injected the way a real signal lands in it: ``add_system_message`` adds
the reply and then raises, so the exception comes from the statements that follow the reply.
Every row patches inside its body with ``patch.object`` (restored on exit even when the patch
raises). Mutants each row kills are named in its docstring.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest

from fsm_llm import API
from fsm_llm.core.definitions import Conversation
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm.test_async_engine import _CHAT, _collect


def _history(api: API, cid: str) -> list[dict[str, str]]:
    instance = api.fsm_manager.instances[
        api.conversation_stacks[cid][-1].conversation_id
    ]
    return [dict(e) for e in instance.context.conversation.exchanges]


def _summary(api: API, cid: str) -> str | None:
    instance = api.fsm_manager.instances[
        api.conversation_stacks[cid][-1].conversation_id
    ]
    return instance.context.conversation.summary


def _api(**kwargs: Any) -> tuple[API, str]:
    llm = ScriptedLLM(default_response="reply", stream_chunks=["re", "ply"])
    api = API.from_definition(_CHAT(), llm_interface=llm, **kwargs)
    cid, _ = api.start_conversation()
    return api, cid


@contextmanager
def _signal_after_the_reply(signal: type[BaseException]) -> Iterator[None]:
    """The next ``add_system_message`` adds its reply, then raises *signal* once; the original
    method is back when the block exits, whatever happens inside it."""
    real = Conversation.add_system_message
    fired = False

    def add_then_signal(self: Conversation, message: str) -> None:
        nonlocal fired
        real(self, message)
        if not fired:
            fired = True
            raise signal()

    with patch.object(Conversation, "add_system_message", add_then_signal):
        yield


# One signal per row (`KeyboardInterrupt`): dropping `SystemExit` from the driver arms is killed
# by the retry rows of `test_api_elaborate.py` and the atomic rows of `test_pipeline.py`.


class TestAnInterruptAfterTheReplyIsAdded:
    def test_sync_converse_leaves_the_pre_turn_history_and_a_retry_adds_one_exchange(
        self,
    ) -> None:
        """The repro of the review: state back, history gained ``[user, system]``. Kills: the
        snapshot without the history (the reply stays), a restore without the user message pop
        by the manager (the user message stays), a retry on a stale reply."""
        api, cid = _api()
        api.converse("first", cid)
        before = _history(api, cid)
        assert [next(iter(e)) for e in before] == ["system", "user", "system"]

        with _signal_after_the_reply(KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                api.converse("hello", cid)

        assert _history(api, cid) == before
        assert _summary(api, cid) is None
        api.converse("hello", cid)  # the same signal does not fire again
        assert _history(api, cid) == [*before, {"user": "hello"}, {"system": "reply"}]

    async def test_async_converse_leaves_the_pre_turn_history(self) -> None:
        api, cid = _api()
        await api.aconverse("first", cid)
        before = _history(api, cid)

        with _signal_after_the_reply(KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                await api.aconverse("hello", cid)

        assert _history(api, cid) == before
        await api.aconverse("hello", cid)
        assert _history(api, cid) == [*before, {"user": "hello"}, {"system": "reply"}]

    def test_a_sync_stream_leaves_the_pre_turn_history(self) -> None:
        """The stream persists the joined reply in a ``finally``; a signal raised right after
        it is not an abandonment, so the whole turn is rolled back."""
        api, cid = _api()
        api.converse("first", cid)
        before = _history(api, cid)

        with _signal_after_the_reply(KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                list(api.converse_stream("hello", cid))

        assert _history(api, cid) == before
        assert "".join(api.converse_stream("hello", cid)) == "reply"
        assert _history(api, cid) == [*before, {"user": "hello"}, {"system": "reply"}]

    async def test_an_async_stream_leaves_the_pre_turn_history(self) -> None:
        api, cid = _api()
        await api.aconverse("first", cid)
        before = _history(api, cid)

        with _signal_after_the_reply(KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                await _collect(api.aconverse_stream("hello", cid))

        assert _history(api, cid) == before
        assert "".join(await _collect(api.aconverse_stream("hello", cid))) == "reply"

    def test_a_message_free_advance_pops_nothing_of_an_earlier_turn(self) -> None:
        """No user message was added, so the manager pops nothing: the history must end as it
        was (the greeting and the first turn), not lose an exchange and not gain the reply."""
        api, cid = _api()
        api.converse("first", cid)
        before = _history(api, cid)

        with _signal_after_the_reply(KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                api.advance(cid)

        assert _history(api, cid) == before

    def test_the_trim_a_reply_causes_is_undone_with_the_summary(self) -> None:
        """Adding a reply trims the oldest exchanges into ``summary``; the restore puts both
        back, exchange by exchange. Kills: truncating to a captured length (the trimmed
        exchanges stay in the summary and the cut lands on the wrong entries), a restore of the
        exchanges without the summary."""
        api, cid = _api(max_history_size=2)
        api.converse("first", cid)
        api.converse(
            "second", cid
        )  # four entries after it: at the limit, already trimmed once
        before = _history(api, cid)
        before_summary = _summary(api, cid)

        with _signal_after_the_reply(KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                api.converse("third", cid)

        assert _history(api, cid) == before
        assert _summary(api, cid) == before_summary
        api.converse("third", cid)
        assert _history(api, cid)[-2:] == [{"user": "third"}, {"system": "reply"}]


class TestACancelAtTheSamePoint:
    async def test_a_cancel_after_the_reply_is_added_leaves_the_pre_turn_history(
        self,
    ) -> None:
        """The cancel arm restores the same snapshot: the existing cancel rows pin the awaits,
        this one pins the statements after the last await."""
        api, cid = _api()
        await api.aconverse("first", cid)
        before = _history(api, cid)

        with _signal_after_the_reply(asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await api.aconverse("hello", cid)

        assert _history(api, cid) == before
        await api.aconverse("hello", cid)
        assert _history(api, cid) == [*before, {"user": "hello"}, {"system": "reply"}]
