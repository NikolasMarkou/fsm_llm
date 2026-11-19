"""Plan 51fde95b step 12.3b: the read-only event log browse (whitelisted fields, no refs, no links)."""

from __future__ import annotations

import shutil
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from fsm_llm.monitor import eventlog
from fsm_llm.monitor.server import app, configure
from fsm_llm.runtime import EventLog, EventType
from fsm_llm.runtime.privacy import SHAPE_FIELDS
from fsm_llm.runtime.usage import usage_from_events
from tests.conftest import symlink_or_skip
from tests.test_fsm_llm_monitor.support import _manager

_ENVELOPE = {"seq", "ts", "type", "frame"}


def _write(root, privacy: str = "values") -> None:
    log = EventLog(root, privacy=privacy)
    log.append(
        "abc",
        "turn/start",
        state="greet",
        has_user_message=True,
        user_message_ref="sha256:aa",
        note="hunter2 free text",
    )
    log.append(
        "abc",
        "llm/request",
        source="s",
        call_type="c",
        stream=False,
        model="m",
        prompt_ref="sha256:bb",
    )
    log.append(
        "abc",
        "llm/response",
        source="s",
        call_type="c",
        stream=False,
        model="m",
        prompt_tokens=3,
        completion_tokens=4,
        total_tokens=7,
        response_ref="sha256:cc",
    )
    log.append("abc", "handler", name="h", secret="hunter2")  # no allow-list entry
    log.append(
        "abc",
        "transition",
        from_state="a",
        to_state="b",
        outcome="DETERMINISTIC",
        executed=True,
        priority=1,
        candidates=[{"target": "b", "api_key": "sk-abcdefghijklmnopqrstuvwxyz012345"}],
    )
    log.append("abc", "turn/end", outcome="committed")
    log.append("zzz", "turn/end", outcome="committed")
    log.close()


def _client(tmp_path, *, api_key=None):
    _write(tmp_path)
    _write(tmp_path / "shape", privacy="shape")
    # a log file that is a link to a file outside the root
    outside = tmp_path / "outside.jsonl"
    shutil.copy(tmp_path / "logs" / "abc.jsonl", outside)
    symlink_or_skip(tmp_path / "logs" / "lnk.jsonl", outside)
    (tmp_path / "logs" / "Foreign Name.txt").write_text("x", encoding="utf-8")
    mgr = _manager()
    mgr.attach_event_log(EventLog(tmp_path))
    configure(manager=mgr, api_key=api_key)
    return TestClient(app), mgr


def test_the_list_shows_regular_logs_only_and_none_without_a_log(tmp_path):
    client, mgr = _client(tmp_path)
    listed = client.get("/api/eventlog").json()
    assert listed == {"available": True, "conversations": ["abc", "zzz"]}
    with mock.patch.object(eventlog, "MAX_EVENTLOG_CONVERSATIONS", 1):
        assert client.get("/api/eventlog").json()["conversations"] == ["abc"]
    mgr._event_log = None
    assert client.get("/api/eventlog").json() == {
        "available": False,
        "conversations": [],
    }
    assert client.get("/api/eventlog/abc").status_code == 404


def test_an_attached_apis_own_log_is_used_and_an_explicit_one_wins(tmp_path):
    _write(tmp_path)
    mgr = _manager()
    mgr._attached_api = SimpleNamespace(event_log=EventLog(tmp_path))
    assert mgr.event_log is mgr._attached_api.event_log
    other = EventLog(tmp_path / "other")
    mgr.attach_event_log(other)
    assert mgr.event_log is other


def test_events_are_cut_to_the_whitelist_and_hold_no_reference_or_text(tmp_path):
    client, _ = _client(tmp_path)
    body = client.get("/api/eventlog/abc").json()
    assert [e["type"] for e in body["events"]] == [
        "turn/start",
        "llm/request",
        "llm/response",
        "handler",
        "transition",
        "turn/end",
    ]
    for event in body["events"]:
        allowed = _ENVELOPE | SHAPE_FIELDS.get(
            EventType.from_string(event["type"]), set()
        )
        assert set(event) <= allowed
    assert body["events"][0]["state"] == "greet"
    assert body["events"][2]["total_tokens"] == 7
    assert set(body["events"][3]) == _ENVELOPE  # no allow-list entry: envelope only
    assert body["events"][4]["candidates"] == [
        {"target": "b"}
    ]  # a nested secret is dropped
    text = client.get("/api/eventlog/abc").text
    assert "hunter2" not in text and "_ref" not in text and "sha256" not in text
    assert "sk-abc" not in text
    reader = EventLog(tmp_path)
    assert body["usage"] == usage_from_events(reader.read("abc")).model_dump()
    assert body["usage"]["total_tokens"] == 7


def test_a_shape_log_is_readable_and_paging_continues(tmp_path):
    client, _ = _client(tmp_path)
    shape = EventLog(tmp_path / "shape")
    assert "abc" in shape.conversation_ids()
    mgr = _manager()
    mgr.attach_event_log(shape)
    configure(manager=mgr)
    page = client.get("/api/eventlog/abc", params={"limit": 2}).json()
    assert [e["seq"] for e in page["events"]] == [1, 2] and page["next_seq"] == 2
    rest = client.get("/api/eventlog/abc", params={"since_seq": 2, "limit": 50}).json()
    assert [e["seq"] for e in rest["events"]] == [3, 4, 5, 6] and rest[
        "next_seq"
    ] is None
    assert (
        client.get("/api/eventlog/abc", params={"limit": 0}).json()["events"][0]["seq"]
        == 1
    )


def test_the_event_count_is_capped(tmp_path):
    client, _ = _client(tmp_path)
    with mock.patch.object(eventlog, "MAX_EVENTLOG_EVENTS", 3):
        body = client.get("/api/eventlog/abc", params={"limit": 100}).json()
    assert len(body["events"]) == 3 and body["next_seq"] == 3


@pytest.mark.parametrize("name", ["lnk", "missing", "..", "A", "a b", "..%2Fx", "%00"])
def test_a_link_an_unknown_and_a_malformed_id_are_404(tmp_path, name):
    client, _ = _client(tmp_path)
    assert client.get(f"/api/eventlog/{name}").status_code == 404


def test_the_routes_need_the_api_key(tmp_path):
    client, _ = _client(tmp_path, api_key="k")
    for path in ("/api/eventlog", "/api/eventlog/abc"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"X-API-Key": "k"}).status_code == 200


def test_an_event_type_this_version_does_not_know_shows_its_envelope_only():
    event = {"seq": 1, "ts": "t", "type": "future/x", "frame": 0, "secret": "hunter2"}
    assert eventlog._shown(event) == {
        "seq": 1,
        "ts": "t",
        "type": "future/x",
        "frame": 0,
    }


def test_the_cli_flag_attaches_the_directory(tmp_path, monkeypatch):
    import uvicorn

    from fsm_llm.monitor import __main__ as monitor_main
    from fsm_llm.monitor.server import get_manager

    monkeypatch.setattr(
        sys, "argv", ["fsm-llm-monitor", "--no-browser", "--event-log", str(tmp_path)]
    )
    monkeypatch.setattr(uvicorn.Server, "run", mock.Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(SystemExit):
        monitor_main.main_cli()
    assert get_manager().event_log.root == tmp_path
