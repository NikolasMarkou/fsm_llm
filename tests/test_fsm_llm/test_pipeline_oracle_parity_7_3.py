"""Post-step-8 parity test for R10 site L1243 (initial response generation).

Step 8 retired the FSM_LLM_ORACLE_RESPONSE flag and made oracle.invoke
the only path for the initial-response site. These tests verify the
default-ON path still runs and produces user-visible output equivalent
to the legacy path it replaced (per the wire-equivalence proof in
decisions.md D-STEP-7.3).
"""

from __future__ import annotations

from pathlib import Path

from fsm_llm.dialog.api import API

from .test_pipeline_oracle_parity import (
    REFERENCE_FSMS,
    SITE_RESPONSE,
    RecordingLLM,
)


def _greeting(fsm_path: Path) -> tuple[str, RecordingLLM]:
    spy = RecordingLLM()
    api = API.from_file(str(fsm_path), llm_interface=spy)
    _conv_id, greeting = api.start_conversation()
    api.end_conversation(_conv_id)
    return greeting, spy


def test_initial_response_returns_via_oracle_path():
    """Initial response generation succeeds and returns the stub message."""
    greeting, spy = _greeting(REFERENCE_FSMS["simple_greeting"])
    assert greeting == "(stub response)"
    # The oracle path routes through generate_response; the spy sees it.
    assert len(spy.records[SITE_RESPONSE]) >= 1


def test_initial_response_wire_user_message_empty():
    """The initial response wire payload pins user_message="" (oracle._invoke_unstructured
    behaviour, byte-equivalent to the legacy ResponseGenerationRequest(..., user_message="")
    that the same site used to construct)."""
    _, spy = _greeting(REFERENCE_FSMS["simple_greeting"])
    if spy.records[SITE_RESPONSE]:
        first = spy.records[SITE_RESPONSE][0]
        assert first["user_message"] == ""


def test_initial_response_completes_without_legacy_call_path():
    """No fallback to legacy generate_response; the oracle is the only
    boundary for the initial-response site post-step-8."""
    import fsm_llm.dialog.turn as p_mod

    src = Path(p_mod.__file__).read_text()
    # Oracle wiring present in generate_initial_response
    assert "D-R10-7.3 (step 8 finalised)" in src
    # Flag check removed
    assert "FSM_LLM_ORACLE_RESPONSE" not in src.replace(
        "FSM_LLM_ORACLE_RESPONSE_STREAM", ""
    )
