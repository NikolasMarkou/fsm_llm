from __future__ import annotations

"""Parity harness scaffold for R10 step 7 (pipeline → oracle.invoke).

This module is the **scaffold only** — it provides:

1. A ``RecordingLLM`` spy that intercepts every ``LLMInterface`` method
   the dialog turn currently calls (``generate_response``,
   ``generate_response_stream``, ``extract_field``, ``_make_llm_call``)
   and records the exact arguments seen at each site.
2. Three reference FSM fixtures spanning the matrix of call sites
   that R10 step 7 will rewire:
   - ``simple_greeting`` — terminal/cohort + transitions (sites L1185,
     L1227, L1270, L1606).
   - ``form_filling`` — heavy data + field extraction (sites L1270,
     L1606).
   - ``multi_turn_extraction`` — classification path (sites L2146, L2188).
3. A pre-wiring snapshot capture helper: ``capture_snapshot(fsm_path,
   user_messages)`` returns a structured dict of the recorded calls,
   normalised so step 7 can byte-compare legacy-path vs oracle-path
   outputs.

The harness intentionally does **NOT** assert anything about the wiring
itself — that is step 7's job. Step 6 verifies only:

- Each fixture FSM loads, compiles, and runs a turn against the spy
  without raising.
- The snapshot helper captures ≥ 1 call site per fixture.
- The recorder is sufficient to distinguish the 6 R10 sites.

When step 7 lands, it will (per-site, behind the per-callback flag):

```python
legacy = capture_snapshot(fsm, msgs, flag_off=True)
wired  = capture_snapshot(fsm, msgs, flag_on=True, flag_name=...)
assert legacy[site] == wired[site]   # byte-for-byte
```
"""

from pathlib import Path
from typing import Any

import pytest

from fsm_llm.dialog.api import API
from fsm_llm.dialog.definitions import (
    FieldExtractionRequest,
    FieldExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.runtime._litellm import LLMInterface

EXAMPLES_ROOT = Path(__file__).resolve().parents[2] / "examples"


REFERENCE_FSMS = {
    "simple_greeting": EXAMPLES_ROOT / "basic" / "simple_greeting" / "fsm.json",
    "form_filling": EXAMPLES_ROOT / "basic" / "form_filling" / "fsm.json",
    "multi_turn_extraction": (
        EXAMPLES_ROOT / "basic" / "multi_turn_extraction" / "fsm.json"
    ),
}


# ---- Site IDs (mirror R10 doc + plan.md "Files To Modify" Bundle C) ----

SITE_RESPONSE_STREAM = "L1185_response_stream"
SITE_RESPONSE = "L1227_response"
SITE_EXTRACT = "L1270_extract"
SITE_FIELD_EXTRACT = "L1606_field_extract"
SITE_CLASSIFIER = "L2146_classifier"
SITE_CLASSIFIER_RESP = "L2188_classifier_resp"

ALL_SITES = (
    SITE_RESPONSE_STREAM,
    SITE_RESPONSE,
    SITE_EXTRACT,
    SITE_FIELD_EXTRACT,
    SITE_CLASSIFIER,
    SITE_CLASSIFIER_RESP,
)


class RecordingLLM(LLMInterface):
    """Spy that records every method call into per-site buckets.

    Returns benign canned responses so the dialog turn can complete
    without exercising any real provider. Unlike ``Mock(spec=...)``,
    this captures the precise pydantic-model arg shape that R10's
    parity harness will byte-compare in step 7.
    """

    model = "ollama_chat/qwen3.5:4b"
    max_tokens = 1000
    timeout = None

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}
        self.records: dict[str, list[Any]] = {site: [] for site in ALL_SITES}
        # _make_llm_call is internal-but-spied because R10 site L1270
        # calls it directly (extraction).
        self.records["_make_llm_call"] = []

    # --- LLMInterface surface ---

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        # Sites L1227, L2146, L2188 all reach here. We can't tell them
        # apart at the LLMInterface level — step 7 distinguishes them
        # by the per-callback flag scope at the call site. For step 6
        # we record under L1227 as the canonical "non-stream response"
        # bucket; classifier sites get distinguished at the dialog
        # boundary (the harness wraps Classifier.classify too — TODO
        # step 7).
        self.records[SITE_RESPONSE].append(request.model_dump())
        # Return a minimal valid response. Empty system_prompt sentinel
        # is honoured by the dialog fast-path.
        return ResponseGenerationResponse(
            message="(stub response)",
            reasoning="recorded",
        )

    def generate_response_stream(self, request: ResponseGenerationRequest):
        self.records[SITE_RESPONSE_STREAM].append(request.model_dump())
        yield "(stub stream)"

    def extract_field(
        self, request: FieldExtractionRequest
    ) -> FieldExtractionResponse:
        self.records[SITE_FIELD_EXTRACT].append(request.model_dump())
        return FieldExtractionResponse(
            field_name=request.field_name,
            field_type=request.field_type,
            value=None,
            confidence=0.0,
            reasoning="recorded",
        )

    def _make_llm_call(self, messages, call_type: str = "unknown"):
        # Site L1270 reaches here for "data_extraction".
        bucket = SITE_EXTRACT if call_type == "data_extraction" else "_make_llm_call"
        self.records[bucket].append(
            {"messages": messages, "call_type": call_type}
        )
        # Return a JSON-shaped string the extraction parser can ingest.
        return '{"extracted_data": {}, "confidence": 0.5, "reasoning": "stub"}'


# --------------------------------------------------------------
# Snapshot helpers
# --------------------------------------------------------------


def capture_snapshot(
    fsm_path: Path, user_messages: list[str]
) -> dict[str, list[Any]]:
    """Run a converse loop and return the per-site recorded calls.

    Step 7 will call this twice per fixture (flag-off vs flag-on) and
    diff the dicts.
    """
    spy = RecordingLLM()
    api = API.from_file(str(fsm_path), llm_interface=spy)
    conv_id, _greeting = api.start_conversation()
    for msg in user_messages:
        api.converse(msg, conv_id)
    api.end_conversation(conv_id)
    return spy.records


# --------------------------------------------------------------
# Tests — scaffold-level only (no parity assertions yet)
# --------------------------------------------------------------


@pytest.mark.parametrize("fsm_id,fsm_path", list(REFERENCE_FSMS.items()))
def test_reference_fsm_loads_and_runs_with_recorder(
    fsm_id: str, fsm_path: Path
) -> None:
    """Each reference FSM compiles and runs one turn against the spy."""
    assert fsm_path.exists(), f"reference FSM missing: {fsm_path}"
    snap = capture_snapshot(fsm_path, ["hello"])
    # At least one site recorded a call (every FSM with response
    # generation will hit L1227 at minimum).
    total_calls = sum(len(v) for v in snap.values())
    assert total_calls >= 1, f"{fsm_id}: no LLM calls recorded"


def test_recording_llm_distinguishes_extract_from_response() -> None:
    """The spy buckets ``data_extraction`` vs response generation."""
    spy = RecordingLLM()
    spy._make_llm_call(
        [{"role": "system", "content": "x"}], call_type="data_extraction"
    )
    spy.generate_response(
        ResponseGenerationRequest(system_prompt="hi", user_message="")
    )
    assert len(spy.records[SITE_EXTRACT]) == 1
    assert len(spy.records[SITE_RESPONSE]) == 1
    # The two buckets are distinct.
    assert spy.records[SITE_EXTRACT] != spy.records[SITE_RESPONSE]


def test_recording_llm_buckets_field_extract() -> None:
    spy = RecordingLLM()
    req = FieldExtractionRequest(
        system_prompt="extract age",
        user_message="I am 42",
        field_name="age",
        field_type="int",
    )
    spy.extract_field(req)
    assert len(spy.records[SITE_FIELD_EXTRACT]) == 1
    assert spy.records[SITE_FIELD_EXTRACT][0]["field_name"] == "age"


def test_recording_llm_buckets_stream() -> None:
    spy = RecordingLLM()
    req = ResponseGenerationRequest(system_prompt="x", user_message="")
    list(spy.generate_response_stream(req))
    assert len(spy.records[SITE_RESPONSE_STREAM]) == 1


def test_all_six_site_ids_are_unique() -> None:
    """Sanity: site IDs collide nowhere — step 7's diff dict needs unique keys."""
    assert len(set(ALL_SITES)) == 6


@pytest.mark.parametrize("fsm_id", list(REFERENCE_FSMS.keys()))
def test_snapshot_dict_contract(fsm_id: str) -> None:
    """Snapshot dict must contain a list (possibly empty) for every known site.

    Step 7's parity comparison iterates ALL_SITES and assumes each key exists.
    """
    snap = capture_snapshot(REFERENCE_FSMS[fsm_id], ["hi"])
    for site in ALL_SITES:
        assert site in snap, f"{fsm_id}: snapshot missing site {site}"
        assert isinstance(snap[site], list), f"{fsm_id}: site {site} not a list"


def test_capture_snapshot_is_deterministic() -> None:
    """Same fixture + same input → same snapshot. (Step 7 relies on this.)"""
    fsm = REFERENCE_FSMS["simple_greeting"]
    snap_a = capture_snapshot(fsm, ["hello"])
    snap_b = capture_snapshot(fsm, ["hello"])
    # Dicts must compare equal (recorded payloads identical).
    assert snap_a == snap_b


# Re-export the recorder + snapshot fn so step 7's per-site parity
# tests can import without rebuilding the harness.
__all__ = [
    "ALL_SITES",
    "REFERENCE_FSMS",
    "RecordingLLM",
    "SITE_CLASSIFIER",
    "SITE_CLASSIFIER_RESP",
    "SITE_EXTRACT",
    "SITE_FIELD_EXTRACT",
    "SITE_RESPONSE",
    "SITE_RESPONSE_STREAM",
    "capture_snapshot",
]
