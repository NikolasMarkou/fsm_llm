"""S9 — tests for the `use_compiled` routing flag on FSMManager.

Plan: plans/plan_2026-04-24_b00b890f/plan.md
- SC1 / SC2: flag defaults True; opt-out routes to legacy.
- SC6: prompt-string equivalence between legacy and compiled paths.
- SC7: opt-out behavior covered here too.

Step 1 (scaffold): flag existence + default. Routing-behavior tests land in step 2.
"""

from __future__ import annotations

import pytest

from fsm_llm.definitions import FSMDefinition
from fsm_llm.fsm import FSMManager


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
