"""Post-step-8 parity test for R10 site L1201 (streaming Pass-2 response).

Step 8 retired the FSM_LLM_ORACLE_RESPONSE_STREAM flag and made
oracle.invoke_stream the only path for the streaming site. These tests
verify the default-ON path still routes through the spy correctly and
preserves the wire-level user_message kwarg semantics.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fsm_llm.dialog.definitions import ResponseGenerationRequest

from .test_pipeline_oracle_parity import RecordingLLM


def test_invoke_stream_preserves_user_message():
    """LiteLLMOracle.invoke_stream forwards user_message verbatim — this is
    why this site was wire-equivalent in the step-7 parity check and
    eligible for default-ON in step 8."""
    from fsm_llm.runtime.oracle import LiteLLMOracle

    captured: list[ResponseGenerationRequest] = []
    spy = RecordingLLM()
    real_stream = spy.generate_response_stream

    def _capture_stream(request):
        captured.append(request)
        return real_stream(request)

    with patch.object(spy, "generate_response_stream", side_effect=_capture_stream):
        oracle = LiteLLMOracle(spy)
        chunks = list(oracle.invoke_stream("hi system", user_message="hi user"))

    assert len(chunks) >= 1
    assert len(captured) == 1
    assert captured[0].user_message == "hi user"
    assert captured[0].system_prompt == "hi system"


def test_streaming_site_is_oracle_only_post_step_8():
    """Step 8: legacy generate_response_stream branch removed from pipeline.py.

    Updated post-A.M3d-narrowed (plan_2026-04-29_0f87b9c4): the
    ``_stream_response_generation_pass`` method that carried the
    ``D-R10-7.6 (step 8 finalised)`` marker has been retired entirely
    along with ``_make_cb_respond_stream``. Streaming now flows through
    the executor's stream-mode branch on the D2 ``Leaf(streaming=True)``
    chain via ``StreamingOracle.invoke_stream``; pipeline.py no longer
    touches streaming directly. The contract this test enforces (no
    legacy direct ``self.llm_interface.generate_response_stream`` call
    in the dialog turn) holds even more strongly now."""
    import fsm_llm.dialog.pipeline as p_mod

    src = Path(p_mod.__file__).read_text()
    # Flag check removed
    assert "FSM_LLM_ORACLE_RESPONSE_STREAM" not in src
    # Legacy direct generate_response_stream call removed.
    assert "self.llm_interface.generate_response_stream" not in src
    # The retired streaming sibling and its support method are gone.
    assert "_make_cb_respond_stream" not in src
    assert "_stream_response_generation_pass" not in src
