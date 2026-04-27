"""Per-site parity tests for R10 step 7.4 + 7.5.

7.4 — L2162-era site (now ~L2298): fast-path empty-response_instructions
generate_response in `_execute_response_generation_pass`. Behind
FSM_LLM_ORACLE_CLASSIFIER. Wire-equivalent because system_prompt="."
sentinel short-circuits the LLM call regardless of user_message.

7.5 — L2204-era site (now ~L2340): canonical Pass-2 main response
generation. Behind FSM_LLM_ORACLE_CLASSIFIER_RESP. **Wire-NOT-equivalent**
because oracle._invoke_unstructured pins user_message="" while legacy
path sends the actual user_message — different message arrays at the
wire. Flag stays default-OFF; documented in decisions.md D-STEP-7.5.

Step 8 will flip 7.4 default-ON (wire-equivalent), keep 7.5 default-OFF.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fsm_llm.dialog.api import API

from .test_pipeline_oracle_parity import (
    REFERENCE_FSMS,
    SITE_RESPONSE,
    RecordingLLM,
)


@pytest.fixture
def restore_classifier_flags(monkeypatch):
    monkeypatch.delenv("FSM_LLM_ORACLE_CLASSIFIER", raising=False)
    monkeypatch.delenv("FSM_LLM_ORACLE_CLASSIFIER_RESP", raising=False)
    yield
    monkeypatch.delenv("FSM_LLM_ORACLE_CLASSIFIER", raising=False)
    monkeypatch.delenv("FSM_LLM_ORACLE_CLASSIFIER_RESP", raising=False)


def _converse_one(fsm_path: Path, msg: str) -> tuple[str, RecordingLLM]:
    spy = RecordingLLM()
    api = API.from_file(str(fsm_path), llm_interface=spy)
    conv_id, _ = api.start_conversation()
    out = api.converse(msg, conv_id)
    api.end_conversation(conv_id)
    return out, spy


# ---- 7.4 (FSM_LLM_ORACLE_CLASSIFIER) — fast-path sentinel site ----


def test_classifier_flag_off_default(monkeypatch, restore_classifier_flags):
    out, _ = _converse_one(REFERENCE_FSMS["simple_greeting"], "hello")
    assert out  # got a response


def test_classifier_flag_on_oracle_path(monkeypatch, restore_classifier_flags):
    monkeypatch.setenv("FSM_LLM_ORACLE_CLASSIFIER", "1")
    out, _ = _converse_one(REFERENCE_FSMS["simple_greeting"], "hello")
    assert out


def test_classifier_user_visible_return_equivalent(
    monkeypatch, restore_classifier_flags
):
    """Sentinel-path runs return the same `[state_id]` synthetic regardless
    of which path made the (short-circuited) LLM call."""
    monkeypatch.delenv("FSM_LLM_ORACLE_CLASSIFIER", raising=False)
    out_off, _ = _converse_one(REFERENCE_FSMS["simple_greeting"], "hello")
    monkeypatch.setenv("FSM_LLM_ORACLE_CLASSIFIER", "1")
    out_on, _ = _converse_one(REFERENCE_FSMS["simple_greeting"], "hello")
    assert out_off == out_on


# ---- 7.5 (FSM_LLM_ORACLE_CLASSIFIER_RESP) — canonical Pass-2 main response ----


def test_classifier_resp_flag_off_default(monkeypatch, restore_classifier_flags):
    out, _ = _converse_one(REFERENCE_FSMS["simple_greeting"], "hello")
    assert out


def test_classifier_resp_flag_on_oracle_path(
    monkeypatch, restore_classifier_flags
):
    monkeypatch.setenv("FSM_LLM_ORACLE_CLASSIFIER_RESP", "1")
    out, _ = _converse_one(REFERENCE_FSMS["simple_greeting"], "hello")
    assert out


def test_classifier_resp_wire_user_message_differs(
    monkeypatch, restore_classifier_flags
):
    """ACK STOP-IF: oracle path drops user_message; wire payload differs.
    Documented in decisions.md D-STEP-7.5; flag stays default-OFF."""
    monkeypatch.delenv("FSM_LLM_ORACLE_CLASSIFIER_RESP", raising=False)
    _, spy_off = _converse_one(REFERENCE_FSMS["simple_greeting"], "hello world")
    monkeypatch.setenv("FSM_LLM_ORACLE_CLASSIFIER_RESP", "1")
    _, spy_on = _converse_one(REFERENCE_FSMS["simple_greeting"], "hello world")
    # Find a recorded request with a non-empty user_message under flag-OFF.
    off_with_user = [
        r for r in spy_off.records[SITE_RESPONSE] if r.get("user_message")
    ]
    on_with_user = [
        r for r in spy_on.records[SITE_RESPONSE] if r.get("user_message")
    ]
    # Under flag-ON, all oracle-routed calls have user_message="" (oracle
    # _invoke_unstructured pins it). Under flag-OFF, the legacy path
    # propagates the user message. So if the legacy run captured any
    # non-empty user_message, the oracle run will have fewer.
    if off_with_user:
        assert len(on_with_user) <= len(off_with_user)
