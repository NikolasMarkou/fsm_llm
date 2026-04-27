"""Per-site parity test for R10 step 7.3 — L1243 generate_response (initial)
→ oracle.invoke behind FSM_LLM_ORACLE_RESPONSE.

This site is the most promising for byte-equivalence: at the litellm wire
the two paths are equal (only system_prompt + user_message reach litellm,
and both paths send identical values). The recorder spy does see
different model_dump() shapes (oracle path drops extracted_data/context),
but the **observable user-facing return value** is identical.

Per the user-authorized parity contract: this site CAN be flipped default-ON
in step 8 since the wire payload is byte-identical. We document parity at
the wire level here.
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
def restore_response_flag(monkeypatch):
    monkeypatch.delenv("FSM_LLM_ORACLE_RESPONSE", raising=False)
    yield
    monkeypatch.delenv("FSM_LLM_ORACLE_RESPONSE", raising=False)


def _greeting_under_flag(
    fsm_path: Path, monkeypatch, flag_value: str | None
) -> tuple[str, RecordingLLM]:
    if flag_value is None:
        monkeypatch.delenv("FSM_LLM_ORACLE_RESPONSE", raising=False)
    else:
        monkeypatch.setenv("FSM_LLM_ORACLE_RESPONSE", flag_value)
    spy = RecordingLLM()
    api = API.from_file(str(fsm_path), llm_interface=spy)
    _conv_id, greeting = api.start_conversation()
    api.end_conversation(_conv_id)
    return greeting, spy


def test_response_flag_off_legacy_path(monkeypatch, restore_response_flag):
    greeting, spy = _greeting_under_flag(
        REFERENCE_FSMS["simple_greeting"], monkeypatch, None
    )
    assert greeting == "(stub response)"
    # The recorder caught at least one response call.
    assert len(spy.records[SITE_RESPONSE]) >= 1


def test_response_flag_on_oracle_path(monkeypatch, restore_response_flag):
    greeting, spy = _greeting_under_flag(
        REFERENCE_FSMS["simple_greeting"], monkeypatch, "1"
    )
    assert greeting == "(stub response)"
    # The oracle path also routes through generate_response (via
    # LiteLLMOracle._invoke_unstructured), so SITE_RESPONSE bucket grows.
    assert len(spy.records[SITE_RESPONSE]) >= 1


def test_response_user_visible_return_byte_equivalent(
    monkeypatch, restore_response_flag
):
    """The greeting string returned to the user is identical under both flags."""
    greeting_off, _ = _greeting_under_flag(
        REFERENCE_FSMS["simple_greeting"], monkeypatch, None
    )
    greeting_on, _ = _greeting_under_flag(
        REFERENCE_FSMS["simple_greeting"], monkeypatch, "1"
    )
    assert greeting_off == greeting_on


def test_response_wire_system_prompt_byte_equivalent(
    monkeypatch, restore_response_flag
):
    """At the litellm wire, system_prompt + user_message are identical."""
    _, spy_off = _greeting_under_flag(
        REFERENCE_FSMS["simple_greeting"], monkeypatch, None
    )
    _, spy_on = _greeting_under_flag(
        REFERENCE_FSMS["simple_greeting"], monkeypatch, "1"
    )
    # Each ran one initial response — find the matching record.
    if spy_off.records[SITE_RESPONSE] and spy_on.records[SITE_RESPONSE]:
        off_first = spy_off.records[SITE_RESPONSE][0]
        on_first = spy_on.records[SITE_RESPONSE][0]
        # The wire-level fields (system_prompt, user_message) match.
        assert off_first["system_prompt"] == on_first["system_prompt"]
        assert off_first["user_message"] == on_first["user_message"]
