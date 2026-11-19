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


# ---- plan 51fde95b step 15, D-056: bounded work per page, a damaged line, no whole-dir scan


def _long_log(root, *, count: int = 12, name: str = "long") -> EventLog:
    log = EventLog(root)
    for _ in range(count):
        log.append(
            name,
            "llm/response",
            source="s",
            call_type="c",
            stream=False,
            model="m",
            prompt_tokens=3,
            completion_tokens=4,
            total_tokens=7,
        )
    log.close()
    return EventLog(root)


def _serve(log: EventLog):
    mgr = _manager()
    mgr.attach_event_log(log)
    configure(manager=mgr)
    eventlog._USAGE_CACHE.clear()
    return TestClient(app)


class TestUsageIsTheWholeLogAndCached:
    def test_usage_covers_events_beyond_the_page(self, tmp_path):
        client = _serve(_long_log(tmp_path))
        page = client.get("/api/eventlog/long", params={"limit": 2}).json()
        assert len(page["events"]) == 2 and page["next_seq"] == 2
        assert page["usage"]["total_tokens"] == 12 * 7  # not the 2 events shown
        assert page["usage"]["calls"] == 12
        assert (page["skipped"], page["corrupt_after_seq"]) == (0, None)

    def test_an_unchanged_log_is_folded_once_and_a_grown_one_again(
        self, tmp_path, monkeypatch
    ):
        log = _long_log(tmp_path)
        client = _serve(log)
        folds = []
        real = eventlog._fold
        monkeypatch.setattr(
            eventlog, "_fold", lambda *a: (folds.append(a[1]), real(*a))[1]
        )
        for _ in range(3):
            assert client.get("/api/eventlog/long").status_code == 200
        assert folds == ["long"]
        writer = EventLog(tmp_path)
        writer.append("long", "turn/end", outcome="committed")
        writer.close()
        body = client.get("/api/eventlog/long").json()
        assert folds == ["long", "long"] and body["usage"]["calls"] == 12

    def test_the_cache_is_bounded(self, tmp_path, monkeypatch):
        log = _long_log(tmp_path, count=1, name="aa")
        _long_log(tmp_path, count=1, name="bb")
        client = _serve(log)
        monkeypatch.setattr(eventlog, "MAX_EVENTLOG_USAGE_CACHE", 1)
        for name in ("aa", "bb", "aa"):
            assert client.get(f"/api/eventlog/{name}").status_code == 200
        assert len(eventlog._USAGE_CACHE) == 1


class TestADamagedLine:
    def _damage(self, tmp_path, line: int = 11):
        log = _long_log(tmp_path, count=20)
        path = tmp_path / "logs" / "long.jsonl"
        lines = path.read_bytes().split(b"\n")
        lines[line - 1] = b"{not json"
        path.write_bytes(b"\n".join(lines))
        return log

    def test_a_page_before_the_damage_and_the_page_across_it_answer_200(self, tmp_path):
        client = _serve(self._damage(tmp_path))
        first = client.get("/api/eventlog/long", params={"limit": 5})
        assert first.status_code == 200, first.text
        body = first.json()
        assert [e["seq"] for e in body["events"]] == [1, 2, 3, 4, 5]
        assert (body["skipped"], body["corrupt_after_seq"]) == (1, 10)
        assert body["usage"]["calls"] == 10  # the readable prefix
        across = client.get("/api/eventlog/long", params={"since_seq": 8}).json()
        assert [e["seq"] for e in across["events"]] == [9, 10]
        assert across["next_seq"] is None and across["skipped"] == 1

    def test_a_damaged_log_is_still_listed(self, tmp_path):
        client = _serve(self._damage(tmp_path))
        assert client.get("/api/eventlog").json()["conversations"] == ["long"]


class TestNoWholeDirectoryWorkPerPage:
    def test_a_page_lists_and_lstats_nothing_but_its_own_log(
        self, tmp_path, monkeypatch
    ):
        log = _long_log(tmp_path, count=2, name="aa")
        _long_log(tmp_path, count=2, name="bb")
        client = _serve(log)
        monkeypatch.setattr(
            log, "conversation_ids", mock.Mock(side_effect=AssertionError("listed"))
        )
        lstats = []
        real = eventlog.os.lstat
        monkeypatch.setattr(
            eventlog.os, "lstat", lambda p, *a, **k: (lstats.append(str(p)), real(p))[1]
        )
        assert client.get("/api/eventlog/aa").status_code == 200
        assert [p.rsplit("/", 1)[1] for p in lstats] == ["aa.jsonl"]

    def test_the_list_stops_checking_at_its_cap(self, tmp_path, monkeypatch):
        log = _long_log(tmp_path, count=1, name="aa")
        for name in ("bb", "cc", "dd"):
            _long_log(tmp_path, count=1, name=name)
        checked = []
        real = eventlog._is_regular
        monkeypatch.setattr(
            eventlog, "_is_regular", lambda p: (checked.append(p.name), real(p))[1]
        )
        monkeypatch.setattr(eventlog, "MAX_EVENTLOG_CONVERSATIONS", 2)
        assert eventlog.list_logs(log) == ["aa", "bb"] and len(checked) == 2

    def test_the_timeline_route_is_time_bounded(self, tmp_path):
        from fsm_llm.monitor.server import routes_monitoring

        client = _serve(_long_log(tmp_path, count=1))
        with mock.patch.object(
            routes_monitoring, "_bounded", wraps=routes_monitoring._bounded
        ) as bounded:
            assert client.get("/api/eventlog/long").status_code == 200
        assert bounded.call_count == 1
