"""S9 — tests for the `use_compiled` routing flag on FSMManager.

Plan: plans/plan_2026-04-24_b00b890f/plan.md
- SC1 / SC2: flag defaults True; opt-out routes to legacy.
- SC6: prompt-string equivalence between legacy and compiled paths.
- SC7: opt-out behavior covered here too.

Step 1 (scaffold): flag existence + default. Routing-behavior tests land in step 2.
"""

from __future__ import annotations

import pytest

from fsm_llm.definitions import (
    FieldExtractionResponse,
    FSMDefinition,
    ResponseGenerationResponse,
)
from fsm_llm.fsm import FSMManager
from fsm_llm.llm import LLMInterface


def _simple_fsm_dict() -> dict:
    """Two-state FSM: start → done. Deterministic transition, no extractions.

    `start` has an unconditional transition to `done`. That keeps the state
    non-terminal (so process_message is accepted) without requiring
    extractions/classifications.
    """
    return {
        "name": "S9RoutingFSM",
        "description": "Minimal FSM for S9 routing tests",
        "initial_state": "start",
        "version": "4.1",
        "states": {
            "start": {
                "id": "start",
                "description": "Start state",
                "purpose": "Greet the user",
                "response_instructions": "Greet the user briefly.",
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "Always advance on any message",
                        "priority": 100,
                        "conditions": [],
                    }
                ],
            },
            "done": {
                "id": "done",
                "description": "Terminal state",
                "purpose": "Conversation over",
                "response_instructions": "Say goodbye.",
                "transitions": [],
            },
        },
    }


@pytest.fixture
def simple_fsm_def() -> FSMDefinition:
    return FSMDefinition.model_validate(_simple_fsm_dict())


@pytest.fixture
def fsm_manager(mock_llm2_interface, simple_fsm_def) -> FSMManager:
    def _loader(fsm_id: str) -> FSMDefinition:
        return simple_fsm_def

    return FSMManager(fsm_loader=_loader, llm_interface=mock_llm2_interface)


# ---------------------------------------------------------------------------
# Step 1 — flag surface
# ---------------------------------------------------------------------------


class TestUseCompiledFlagSurface:
    def test_flag_defaults_true(self, fsm_manager: FSMManager) -> None:
        assert fsm_manager.use_compiled is True

    def test_flag_can_be_disabled_at_construction(
        self, mock_llm2_interface, simple_fsm_def
    ) -> None:
        def _loader(fsm_id: str) -> FSMDefinition:
            return simple_fsm_def

        mgr = FSMManager(
            fsm_loader=_loader,
            llm_interface=mock_llm2_interface,
            use_compiled=False,
        )
        assert mgr.use_compiled is False


# ---------------------------------------------------------------------------
# Step 2 — routing behavior (RED at step 1, GREEN at step 2)
# ---------------------------------------------------------------------------


class TestProcessMessageRouting:
    def test_default_routes_compiled(
        self, fsm_manager: FSMManager, monkeypatch
    ) -> None:
        """With use_compiled=True (default), process_message must call
        MessagePipeline.process_compiled, NOT legacy `process`.
        """
        calls: list[str] = []
        pipeline = fsm_manager._pipeline

        orig_compiled = pipeline.process_compiled
        orig_legacy = pipeline.process

        def spy_compiled(*args, **kwargs):
            calls.append("compiled")
            return orig_compiled(*args, **kwargs)

        def spy_legacy(*args, **kwargs):
            calls.append("legacy")
            return orig_legacy(*args, **kwargs)

        monkeypatch.setattr(pipeline, "process_compiled", spy_compiled)
        monkeypatch.setattr(pipeline, "process", spy_legacy)

        conv_id, _ = fsm_manager.start_conversation("s9-test")
        fsm_manager.process_message(conv_id, "hello")

        assert calls == ["compiled"], (
            f"expected compiled-only routing, got {calls}"
        )

    def test_opt_out_uses_legacy(
        self, mock_llm2_interface, simple_fsm_def, monkeypatch
    ) -> None:
        """With use_compiled=False, process_message must call legacy `process`."""

        def _loader(fsm_id: str) -> FSMDefinition:
            return simple_fsm_def

        mgr = FSMManager(
            fsm_loader=_loader,
            llm_interface=mock_llm2_interface,
            use_compiled=False,
        )
        calls: list[str] = []
        pipeline = mgr._pipeline

        orig_compiled = pipeline.process_compiled
        orig_legacy = pipeline.process

        def spy_compiled(*args, **kwargs):
            calls.append("compiled")
            return orig_compiled(*args, **kwargs)

        def spy_legacy(*args, **kwargs):
            calls.append("legacy")
            return orig_legacy(*args, **kwargs)

        monkeypatch.setattr(pipeline, "process_compiled", spy_compiled)
        monkeypatch.setattr(pipeline, "process", spy_legacy)

        conv_id, _ = mgr.start_conversation("s9-test")
        mgr.process_message(conv_id, "hello")

        assert calls == ["legacy"], (
            f"expected legacy-only routing with use_compiled=False, got {calls}"
        )


# ---------------------------------------------------------------------------
# Step 5 — prompt-string equivalence smoke (SC6)
# ---------------------------------------------------------------------------


class _RecordingLLM(LLMInterface):
    """Mock LLM that records every request; returns deterministic stubs.

    Used by the prompt-string equivalence smoke to capture the exact
    system_prompt + user_message strings produced by each path.
    """

    def __init__(self, extract_values: dict[str, str] | None = None) -> None:
        self.extract_values = extract_values or {}
        self.response_requests: list[dict] = []
        self.extract_requests: list[dict] = []

    def generate_response(self, request) -> ResponseGenerationResponse:
        self.response_requests.append(
            {
                "system_prompt": request.system_prompt,
                "user_message": request.user_message,
                "extracted_data": dict(request.extracted_data),
                "context": dict(request.context),
            }
        )
        return ResponseGenerationResponse(
            message="stub",
            message_type="response",
            reasoning="stub",
        )

    def extract_field(self, request) -> FieldExtractionResponse:
        self.extract_requests.append(
            {
                "field_name": request.field_name,
                "system_prompt": request.system_prompt,
                "user_message": request.user_message,
            }
        )
        value = self.extract_values.get(request.field_name)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0 if value is not None else 0.0,
            reasoning="stub",
            is_valid=value is not None,
        )


def _multi_state_fsm_dict() -> dict:
    """3-state FSM exercising transition + extraction. Mirrors the canonical
    `greeting_fsm_dict` fixture used by the regression suite, so the
    equivalence smoke covers the same surface the T5 gate exercised.
    """
    return {
        "name": "s9_prompt_equivalence_fsm",
        "description": "3-state FSM for prompt equivalence smoke",
        "version": "4.1",
        "initial_state": "greeting",
        "states": {
            "greeting": {
                "id": "greeting",
                "description": "Initial greeting",
                "purpose": "Greet the user",
                "response_instructions": "Say hello.",
                "transitions": [
                    {
                        "target_state": "collect_name",
                        "description": "Move to collect name",
                        "priority": 100,
                    }
                ],
            },
            "collect_name": {
                "id": "collect_name",
                "description": "Collect user name",
                "purpose": "Ask for and collect user name",
                "response_instructions": "Ask for the user's name.",
                "required_context_keys": ["user_name"],
                "transitions": [
                    {
                        "target_state": "farewell",
                        "description": "Name collected, say goodbye",
                        "priority": 100,
                        "conditions": [
                            {
                                "description": "User name collected",
                                "requires_context_keys": ["user_name"],
                            }
                        ],
                    }
                ],
            },
            "farewell": {
                "id": "farewell",
                "description": "Farewell state",
                "purpose": "Say goodbye",
                "response_instructions": "Wish them well.",
                "transitions": [],
            },
        },
    }


class TestPromptStringEquivalence:
    """SC6 — byte-equal prompts under both paths closes the S8b devil's
    advocate gap (mock-equivalence of extracted_data ≠ prompt equivalence).
    """

    def test_prompt_string_equivalence_two_turn(self) -> None:
        fsm_dict = _multi_state_fsm_dict()
        fsm_def = FSMDefinition.model_validate(fsm_dict)

        def loader(_id: str) -> FSMDefinition:
            return fsm_def

        extract_values = {"user_name": "Alice"}
        llm_compiled = _RecordingLLM(extract_values=extract_values)
        llm_legacy = _RecordingLLM(extract_values=extract_values)

        mgr_compiled = FSMManager(
            fsm_loader=loader, llm_interface=llm_compiled, use_compiled=True
        )
        mgr_legacy = FSMManager(
            fsm_loader=loader, llm_interface=llm_legacy, use_compiled=False
        )

        for mgr in (mgr_compiled, mgr_legacy):
            conv_id, _ = mgr.start_conversation("prompt_eq")
            mgr.process_message(conv_id, "hi")
            mgr.process_message(conv_id, "My name is Alice")

        # Response-generation prompts: every system_prompt + user_message
        # must be byte-equal across paths, in the same order.
        assert len(llm_compiled.response_requests) == len(
            llm_legacy.response_requests
        ), (
            f"response-request count diverges: compiled="
            f"{len(llm_compiled.response_requests)} legacy="
            f"{len(llm_legacy.response_requests)}"
        )
        for i, (c, g) in enumerate(
            zip(
                llm_compiled.response_requests,
                llm_legacy.response_requests,
                strict=False,
            )
        ):
            assert c["system_prompt"] == g["system_prompt"], (
                f"response[{i}] system_prompt diverges"
            )
            assert c["user_message"] == g["user_message"], (
                f"response[{i}] user_message diverges"
            )
            assert c["extracted_data"] == g["extracted_data"], (
                f"response[{i}] extracted_data diverges"
            )

        # Extraction prompts: same surface, same order.
        assert len(llm_compiled.extract_requests) == len(
            llm_legacy.extract_requests
        ), (
            f"extract-request count diverges: compiled="
            f"{len(llm_compiled.extract_requests)} legacy="
            f"{len(llm_legacy.extract_requests)}"
        )
        for i, (c, g) in enumerate(
            zip(
                llm_compiled.extract_requests,
                llm_legacy.extract_requests,
                strict=False,
            )
        ):
            assert c["field_name"] == g["field_name"], (
                f"extract[{i}] field_name diverges"
            )
            assert c["system_prompt"] == g["system_prompt"], (
                f"extract[{i}] system_prompt diverges"
            )
            assert c["user_message"] == g["user_message"], (
                f"extract[{i}] user_message diverges"
            )


# ---------------------------------------------------------------------------
# S10 — compiled streaming path
# ---------------------------------------------------------------------------


class _StreamingRecordingLLM(LLMInterface):
    """Recording LLM that also serves as a deterministic chunk source.

    `generate_response_stream` yields a fixed chunk list per call; each call's
    `request` is recorded so prompt-string equivalence can be asserted
    across legacy / compiled streaming paths.
    """

    def __init__(
        self,
        chunks: list[str] | None = None,
        extract_values: dict[str, str] | None = None,
    ) -> None:
        self.chunks = chunks if chunks is not None else ["Hi ", "there ", "Alice!"]
        self.extract_values = extract_values or {}
        self.stream_requests: list[dict] = []
        self.response_requests: list[dict] = []
        self.extract_requests: list[dict] = []

    def generate_response(self, request) -> ResponseGenerationResponse:
        self.response_requests.append(
            {
                "system_prompt": request.system_prompt,
                "user_message": request.user_message,
                "extracted_data": dict(request.extracted_data),
                "context": dict(request.context),
            }
        )
        return ResponseGenerationResponse(
            message="".join(self.chunks),
            message_type="response",
            reasoning="stub",
        )

    def generate_response_stream(self, request):
        self.stream_requests.append(
            {
                "system_prompt": request.system_prompt,
                "user_message": request.user_message,
                "extracted_data": dict(request.extracted_data),
                "context": dict(request.context),
            }
        )
        yield from self.chunks

    def extract_field(self, request) -> FieldExtractionResponse:
        self.extract_requests.append(
            {
                "field_name": request.field_name,
                "system_prompt": request.system_prompt,
                "user_message": request.user_message,
            }
        )
        value = self.extract_values.get(request.field_name)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0 if value is not None else 0.0,
            reasoning="stub",
            is_valid=value is not None,
        )


class TestProcessMessageStreamRouting:
    """S10 SC1 / SC2 — process_message_stream routes by use_compiled flag."""

    def test_default_routes_compiled_stream(
        self, fsm_manager: FSMManager, monkeypatch
    ) -> None:
        calls: list[str] = []
        pipeline = fsm_manager._pipeline

        orig_compiled = pipeline.process_stream_compiled
        orig_legacy = pipeline.process_stream

        def spy_compiled(*args, **kwargs):
            calls.append("compiled")
            yield from orig_compiled(*args, **kwargs)

        def spy_legacy(*args, **kwargs):
            calls.append("legacy")
            yield from orig_legacy(*args, **kwargs)

        monkeypatch.setattr(pipeline, "process_stream_compiled", spy_compiled)
        monkeypatch.setattr(pipeline, "process_stream", spy_legacy)

        conv_id, _ = fsm_manager.start_conversation("s10-stream")
        list(fsm_manager.process_message_stream(conv_id, "hello"))

        assert calls == ["compiled"], (
            f"expected compiled-stream routing, got {calls}"
        )

    def test_stream_opt_out_uses_legacy(
        self, mock_llm2_interface, simple_fsm_def, monkeypatch
    ) -> None:
        def _loader(fsm_id: str) -> FSMDefinition:
            return simple_fsm_def

        mgr = FSMManager(
            fsm_loader=_loader,
            llm_interface=mock_llm2_interface,
            use_compiled=False,
        )
        calls: list[str] = []
        pipeline = mgr._pipeline

        orig_compiled = pipeline.process_stream_compiled
        orig_legacy = pipeline.process_stream

        def spy_compiled(*args, **kwargs):
            calls.append("compiled")
            yield from orig_compiled(*args, **kwargs)

        def spy_legacy(*args, **kwargs):
            calls.append("legacy")
            yield from orig_legacy(*args, **kwargs)

        monkeypatch.setattr(pipeline, "process_stream_compiled", spy_compiled)
        monkeypatch.setattr(pipeline, "process_stream", spy_legacy)

        conv_id, _ = mgr.start_conversation("s10-stream")
        list(mgr.process_message_stream(conv_id, "hello"))

        assert calls == ["legacy"], (
            f"expected legacy-stream routing with use_compiled=False, got {calls}"
        )


class TestStreamEquivalence:
    """S10 SC4 / SC5 — chunk + prompt-string equivalence across paths."""

    def _run(self, mgr: FSMManager) -> list[str]:
        conv_id, _ = mgr.start_conversation("stream_eq")
        chunks_turn1 = list(mgr.process_message_stream(conv_id, "hi"))
        chunks_turn2 = list(
            mgr.process_message_stream(conv_id, "My name is Alice")
        )
        return chunks_turn1 + chunks_turn2

    def test_stream_chunk_equivalence_two_turn(self) -> None:
        fsm_dict = _multi_state_fsm_dict()
        fsm_def = FSMDefinition.model_validate(fsm_dict)

        def loader(_id: str) -> FSMDefinition:
            return fsm_def

        chunks = ["Hi ", "Alice", "! Bye."]
        extract_values = {"user_name": "Alice"}

        llm_compiled = _StreamingRecordingLLM(
            chunks=chunks, extract_values=extract_values
        )
        llm_legacy = _StreamingRecordingLLM(
            chunks=chunks, extract_values=extract_values
        )

        mgr_compiled = FSMManager(
            fsm_loader=loader, llm_interface=llm_compiled, use_compiled=True
        )
        mgr_legacy = FSMManager(
            fsm_loader=loader, llm_interface=llm_legacy, use_compiled=False
        )

        out_compiled = self._run(mgr_compiled)
        out_legacy = self._run(mgr_legacy)

        assert out_compiled == out_legacy, (
            f"stream chunks diverge: compiled={out_compiled!r} "
            f"legacy={out_legacy!r}"
        )

    def test_stream_prompt_string_equivalence_two_turn(self) -> None:
        """S10 analog of S9 D-S9-06 smoke: Pass-2 stream prompt must match
        legacy on turn 2 (post-transition, with re-extracted context).
        """
        fsm_dict = _multi_state_fsm_dict()
        fsm_def = FSMDefinition.model_validate(fsm_dict)

        def loader(_id: str) -> FSMDefinition:
            return fsm_def

        chunks = ["ok"]
        extract_values = {"user_name": "Alice"}

        llm_compiled = _StreamingRecordingLLM(
            chunks=chunks, extract_values=extract_values
        )
        llm_legacy = _StreamingRecordingLLM(
            chunks=chunks, extract_values=extract_values
        )

        mgr_compiled = FSMManager(
            fsm_loader=loader, llm_interface=llm_compiled, use_compiled=True
        )
        mgr_legacy = FSMManager(
            fsm_loader=loader, llm_interface=llm_legacy, use_compiled=False
        )

        self._run(mgr_compiled)
        self._run(mgr_legacy)

        assert len(llm_compiled.stream_requests) == len(llm_legacy.stream_requests), (
            f"stream-request count diverges: compiled="
            f"{len(llm_compiled.stream_requests)} legacy="
            f"{len(llm_legacy.stream_requests)}"
        )
        for i, (c, g) in enumerate(
            zip(
                llm_compiled.stream_requests,
                llm_legacy.stream_requests,
                strict=False,
            )
        ):
            assert c["system_prompt"] == g["system_prompt"], (
                f"stream[{i}] system_prompt diverges"
            )
            assert c["user_message"] == g["user_message"], (
                f"stream[{i}] user_message diverges"
            )
            assert c["extracted_data"] == g["extracted_data"], (
                f"stream[{i}] extracted_data diverges"
            )


class TestStreamGeneratorInterruption:
    """S10 SC6 — caller-break persists partial chunks via `finally`."""

    def test_stream_generator_interruption_persists_chunks(self) -> None:
        fsm_dict = _simple_fsm_dict()
        fsm_def = FSMDefinition.model_validate(fsm_dict)

        def loader(_id: str) -> FSMDefinition:
            return fsm_def

        chunks = ["a", "b", "c", "d", "e"]
        llm = _StreamingRecordingLLM(chunks=chunks)

        mgr = FSMManager(fsm_loader=loader, llm_interface=llm, use_compiled=True)
        conv_id, _ = mgr.start_conversation("interrupt")

        instance = mgr.instances[conv_id]

        collected: list[str] = []
        gen = mgr.process_message_stream(conv_id, "hi")
        for i, chunk in enumerate(gen):
            collected.append(chunk)
            if i == 1:
                # Break after 2/5 chunks → GeneratorExit unwinds `finally`.
                gen.close()
                break

        assert collected == ["a", "b"]

        # The streaming closure's `finally` block must have persisted the
        # partial prefix as a system message on the conversation history.
        system_texts = [
            exc["system"]
            for exc in instance.context.conversation.exchanges
            if "system" in exc
        ]
        assert any("ab" in t for t in system_texts), (
            f"expected 'ab' in system messages, got {system_texts!r}"
        )
