"""Acceptance proofs of the event log, in-process (plan 21e6f00d step 10, reduced by D-023).

Only what the other event-log suites do not already prove (the matrix is in progress.md): the
roadmap's "Log test" (every ``llm/request`` of a multi-turn conversation rebuilt byte for byte
from the log) on the provider path and, per call kind, on the engine path; strangler parity of a
real interface (replies, data, history, snapshot, ``usage()`` and the exact provider requests,
log on versus off, with a failed Pass 2 and its retry); and the privacy corpus run end to end
over a real interface (key, base, header, context secrets). Bytes and counts, never timing.
Unverified until the Windows run: nothing here opens a file itself (the files are
``stores.jsonl``'s and ``blobs``').
"""

from __future__ import annotations

import contextlib
import copy
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from fsm_llm import API, FileSessionStore
from fsm_llm.core.definitions import FSMError
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.llm.observer import OBSERVED_CALL_KEYS
from fsm_llm.runtime import PRODUCED_TYPES, EventLog, canonical_json
from tests.test_fsm_llm.event_log_support import _chunk, _reply, tree_bytes
from tests.test_fsm_llm.test_event_log_producers import (
    _CHILD,
    _FSM,
    _run,
    _script,
)

COMPLETION = "fsm_llm.llm.providers.litellm.completion"
_KEY = "sk-SECRET-accept-0001"
_BASE = "https://secret-host.invalid/v1"
_HEADERS = {"X-Org": "hdr-SECRET-0003"}
_CONTEXT_SECRETS = ("pw-SECRET-ctx-0004", "tok-SECRET-ctx-0005", "sk-SECRET-ctx-0006")
_REPLY_BY_PROMPT = (
    ("Left reply", "left reply"),
    ("Child bye", "child bye"),
    ("Child hello", "child hello"),
    ("Working on it", "working"),
    ("Done reply", "all done"),
    ("Greet the user", "hi"),
)
_PROVIDER_KINDS = {
    "response_generation",
    "field_extraction",
    "data_extraction",
    "classification",
    "completion",
}
#: engine call kind -> the interface method that serves it
_ENGINE_KINDS = {
    "greeting": "generate_response",
    "response_generation": "generate_response",
    "response_stream": "generate_response_stream",
    "data_extraction": "extract_bulk_data",
    "field_extraction": "extract_field",
    "classification": "complete",
    "completion": "complete",
}


class _Provider:
    """A patched ``completion``: answers by prompt, records a copy of every request it gets.

    ``fail_on`` names a reply text whose FIRST request raises (a Pass-2 failure and its retry).
    """

    def __init__(self, fail_on: str | None = None) -> None:
        self.received: list[dict[str, Any]] = []
        self.fail_on = fail_on

    def __call__(self, **kwargs: Any) -> Any:
        self.received.append(copy.deepcopy(kwargs))
        system = kwargs["messages"][0]["content"] if kwargs["messages"] else ""
        if kwargs.get("tools"):
            return _reply("4")
        if "Extract the field" in system:
            return _reply(
                json.dumps(
                    {"value": "Ann", "confidence": 0.9, "reasoning": "r", "is_valid": 1}
                )
            )
        if "intent classification" in system:
            return _reply(
                json.dumps({"reasoning": "r", "intent": "left", "confidence": 0.9})
            )
        if kwargs.get("response_format"):
            return _reply(
                json.dumps({"extracted_data": {"mood": "happy"}, "confidence": 0.9})
            )
        text = next((t for needle, t in _REPLY_BY_PROMPT if needle in system), "?")
        if text == self.fail_on:
            self.fail_on = None
            raise RuntimeError("provider down")
        return (
            iter([_chunk(text[:2]), _chunk(text[2:])])
            if kwargs.get("stream")
            else _reply(text)
        )


def _scenario(log: EventLog | None, store_dir: Path) -> SimpleNamespace:
    """Start, a tie, a push and pop, a stream, a transcript write, a failed advance and its retry."""
    provider = _Provider(fail_on="all done")
    llm = LiteLLMInterface(
        model="gpt-4o", api_key=_KEY, api_base=_BASE, extra_headers=_HEADERS
    )
    api = API(
        copy.deepcopy(_FSM),
        llm_interface=llm,
        session_store=FileSessionStore(store_dir),
        event_log=log,
    )
    with patch(COMPLETION, provider):
        cid, greeting = api.start_conversation(
            {
                "api_key": _CONTEXT_SECRETS[2],
                "profile": {"password": _CONTEXT_SECRETS[0], "city": "Oslo"},
            }
        )
        replies = [greeting, api.converse("Ann", cid)]
        api.push_fsm(cid, copy.deepcopy(_CHILD))
        replies.append(api.converse("hello child", cid))
        api.pop_fsm(cid)
        replies.append("".join(api.converse_stream("go", cid)))
        api.update_context(
            cid,
            {
                "_completion_messages": [{"role": "user", "content": "2+2?"}],
                "db_password": _CONTEXT_SECRETS[1],
            },
        )
        with pytest.raises(FSMError):
            api.advance(cid)
        replies.append(api.advance(cid).response)
        api.save_session(cid)
    saved = json.loads((store_dir / f"{cid}.json").read_text("utf-8"))
    for volatile in ("saved_at", "conversation_id"):
        saved.pop(volatile)
    return SimpleNamespace(
        cid=cid,
        provider=provider,
        replies=replies,
        data=api.get_data(cid),
        state=api.get_current_state(cid),
        history=api.get_conversation_history(cid),
        usage=llm.usage().model_dump(),
        snapshot=saved,
    )


class TestTheLogTest:
    """The roadmap's done-when for row 1.1: any past prompt is rebuilt byte-identical."""

    def test_log_test_every_llm_request_is_rebuilt_byte_identical(
        self, log: EventLog, tmp_path: Path
    ) -> None:
        run = _scenario(log, tmp_path / "store")
        requests = [e for e in log.read(run.cid) if e["type"] == "llm/request"]
        # ALL provider calls, the root greeting first (plan 3f4530d3 step 2 dropped the `[1:]`)
        sent = run.provider.received
        assert len(requests) == len(sent) > 8  # every call checked, and not vacuously
        first_turn = next(
            e["seq"] for e in log.read(run.cid) if e["type"] == "turn/start"
        )
        assert requests[0]["seq"] < first_turn and requests[0]["frame"] == 0
        assert {e["call_type"] for e in requests} == _PROVIDER_KINDS
        assert {e["frame"] for e in requests} == {0, 1}  # a pushed child's calls too
        assert any(e["stream"] for e in requests)
        for event, kwargs in zip(requests, sent, strict=True):
            expected = canonical_json(
                {k: kwargs[k] for k in OBSERVED_CALL_KEYS if k in kwargs}
            )
            assert log.get_blob(event["prompt_ref"]) == expected
            assert canonical_json(log.get_json(event["prompt_ref"])) == expected
            assert (event["source"], event["model"]) == ("provider", "gpt-4o")
            assert event["stream"] is bool(kwargs.get("stream"))
        seen = {e["type"] for e in log.read(run.cid)}
        assert seen <= {t.value for t in PRODUCED_TYPES}

    def test_the_failed_call_is_an_error_event_and_its_retry_a_new_request(
        self, log: EventLog, tmp_path: Path
    ) -> None:
        run = _scenario(log, tmp_path / "store")
        events = list(log.read(run.cid))
        (error,) = [e for e in events if e["type"] == "provider/error"]
        assert error["error_type"] == "RuntimeError"
        failed = next(e for e in events if e["seq"] == error["request_seq"])
        retry = next(
            e
            for e in events
            if e["type"] == "llm/request"
            and e["seq"] > error["seq"]
            and e["call_type"] == failed["call_type"]
        )
        assert log.get_blob(retry["prompt_ref"]) == log.get_blob(failed["prompt_ref"])


@pytest.fixture(scope="module")
def engine_pairs(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[EventLog, list[tuple[dict[str, Any], str, Any]]]]:
    """Every logged ``llm/request`` of the engine-path scenario paired with the call the double got."""
    with EventLog(tmp_path_factory.mktemp("engine") / "events") as opened:
        llm = _script()
        _, cid, _ = _run(opened, llm)
        requests = [e for e in opened.read(cid) if e["type"] == "llm/request"]
        calls = llm.call_history  # every call, the root greeting included
        assert len(requests) == len(calls) == 10
        yield opened, [(e, m, r) for e, (m, r) in zip(requests, calls, strict=True)]


def _wire(method: str, request: Any) -> list[dict[str, Any]]:
    """The provider requests a real interface builds for ``request`` (replies are throwaway)."""
    seen: list[dict[str, Any]] = []

    def completion(**kwargs: Any) -> Any:
        seen.append(copy.deepcopy(kwargs))
        return iter([_chunk("x")]) if kwargs.get("stream") else _reply("{}")

    interface = LiteLLMInterface(model="gpt-4o", api_key="sk-unused")
    with patch(COMPLETION, completion), contextlib.suppress(Exception):
        result = getattr(interface, method)(request)
        if method == "generate_response_stream":
            list(result)
    return seen


class TestTheEngineFallbackRebuild:
    @pytest.mark.parametrize("kind", sorted(_ENGINE_KINDS))
    def test_each_call_kind_rebuilds_the_request_the_double_received(
        self, kind: str, engine_pairs: Any
    ) -> None:
        log, pairs = engine_pairs
        chosen = [(e, m, r) for e, m, r in pairs if e["call_type"] == kind]
        # the greeting kind has two rows: the root's and the pushed child's
        assert len(chosen) == (2 if kind == "greeting" else len(chosen)) and chosen, (
            kind
        )
        for event, method, request in chosen:
            assert method == _ENGINE_KINDS[kind]
            blob = log.get_json(event["prompt_ref"])
            assert (blob["source"], blob["request_type"]) == (
                "engine",
                type(request).__name__,
            )
            assert blob["request"] == request.model_dump(mode="json")
            rebuilt = type(request).model_validate(blob["request"])
            assert rebuilt == request
            wire = _wire(method, rebuilt)
            assert wire and wire == _wire(method, request)


class TestStranglerParityOfARealInterface:
    def test_log_on_and_off_agree_on_replies_state_snapshot_usage_and_provider_requests(
        self, log: EventLog, tmp_path: Path
    ) -> None:
        off = _scenario(None, tmp_path / "off")
        on = _scenario(log, tmp_path / "on")
        assert (
            on.replies
            == off.replies
            == [
                "hi",
                "left reply",
                "child bye",
                "working",
                "all done",
            ]
        )
        for field in ("data", "state", "history", "usage", "snapshot"):
            assert getattr(on, field) == getattr(off, field), field
        assert on.usage["calls"] == len(on.provider.received) > 10
        # the provider got the same requests, in the same order, with the log on and off
        assert on.provider.received == off.provider.received
        # D5: the greeting is the first provider call, and its bytes are the same log on or off
        assert on.provider.received[0] == off.provider.received[0]
        assert len(log.conversation_ids()) == 1
        assert not list((tmp_path / "off").glob("*.jsonl"))


class TestThePrivacyCorpusEndToEnd:
    def test_no_credential_or_context_secret_is_in_any_log_or_blob_byte(
        self, log: EventLog, tmp_path: Path
    ) -> None:
        run = _scenario(log, tmp_path / "store")
        # non-vacuity: the provider really got the credentials, and no prompt held a secret
        assert all(
            (c["api_key"], c["api_base"], c["extra_headers"]) == (_KEY, _BASE, _HEADERS)
            for c in run.provider.received
        )
        assert not [
            s for s in _CONTEXT_SECRETS if s in json.dumps(run.provider.received)
        ]
        data = tree_bytes(tmp_path / "events")
        assert b'"llm/request"' in data
        needles = (_KEY, _BASE, "secret-host", "hdr-SECRET", *_CONTEXT_SECRETS)
        assert [n for n in needles if n.encode() in data] == []
        # user text is stored as typed: a documented residual, asserted so the doc stays true
        assert b"hello child" in data and b"Oslo" in data
