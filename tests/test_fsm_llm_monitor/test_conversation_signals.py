"""Plan 51fde95b step 12.3a: classification results and the cut-off flag on the conversation snapshot."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from fsm_llm import API
from fsm_llm.monitor.instance_manager.snapshots import snapshot_from_api
from fsm_llm.monitor.server import app, configure
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm_monitor.support import _manager

_FSM = {
    "name": "T",
    "description": "d",
    "initial_state": "a",
    "persona": "p",
    "states": {
        "a": {
            "id": "a",
            "description": "a",
            "purpose": "p",
            "transitions": [{"target_state": "b", "description": "go", "priority": 1}],
        },
        "b": {"id": "b", "description": "b", "purpose": "p"},
    },
}


def _started():
    api = API.from_definition(_FSM, llm_interface=ScriptedLLM(default_response="hi"))
    conversation_id, _ = api.start_conversation()
    return api, conversation_id, api.fsm_manager.instances[conversation_id].context


class TestClassificationAndCutOff:
    def test_records_and_the_cut_are_shown_and_hidden_things_stay_hidden(self):
        api, cid, ctx = _started()
        ctx.metadata["classification_results"] = {
            "intent": {"intent": "buy", "confidence": 0.9, "system_password": "hunter2"}
        }
        ctx.metadata["transition_classification"] = {
            "chosen": "go",
            "nested": {"api_key": "sk-leak"},
        }
        ctx.data["_cut_off"] = {"reply": True, "fields": ["intent", 7]}
        ctx.data["visible"] = 1

        snap = snapshot_from_api(api, cid)

        assert snap is not None
        assert snap.classification_results == {
            "intent": {"intent": "buy", "confidence": 0.9}
        }
        assert snap.transition_classification == {"chosen": "go", "nested": {}}
        assert snap.cut_off == {"reply": True, "fields": ["intent", "7"]}
        assert snap.context_data == {"visible": 1}  # the internal key stays hidden
        assert "hunter2" not in json.dumps(snap.model_dump(), default=str)
        assert "sk-leak" not in json.dumps(snap.model_dump(), default=str)

    def test_a_turn_with_nothing_to_show_has_none(self):
        api, cid, _ = _started()
        snap = snapshot_from_api(api, cid)
        assert snap is not None
        assert (snap.classification_results, snap.cut_off) == (None, None)
        assert snap.transition_classification is None

    @pytest.mark.parametrize(
        ("record", "expected"),
        [
            ("oops", None),
            ({"reply": "yes"}, {"reply": False, "fields": []}),
            ({"fields": "abc"}, {"reply": False, "fields": []}),
            (
                {"fields": [f"f{i}" for i in range(25)] + ["x" * 300]},
                {"reply": False, "fields": [f"f{i}" for i in range(20)]},
            ),
            ({"fields": ["x" * 300]}, {"reply": False, "fields": ["x" * 100]}),
        ],
    )
    def test_the_cut_off_record_is_shaped_and_capped(self, record, expected):
        api, cid, ctx = _started()
        ctx.data["_cut_off"] = record
        snap = snapshot_from_api(api, cid)
        assert snap is not None and snap.cut_off == expected

    def test_the_conversation_route_carries_them(self):
        api, cid, ctx = _started()
        ctx.data["_cut_off"] = {"reply": True}
        mgr = _manager()
        mgr._attached_api = api
        configure(manager=mgr)

        body = TestClient(app).get(f"/api/conversations/{cid}").json()

        assert body["cut_off"] == {"reply": True, "fields": []}
        assert {"classification_results", "transition_classification"} <= set(body)
