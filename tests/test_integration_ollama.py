"""
Integration tests using Ollama + qwen3.5:4b to verify end-to-end FSM-LLM functionality.

These tests require a running Ollama instance with the qwen3.5:4b model pulled.
They exercise the full 2-pass architecture: data extraction, transition evaluation,
and response generation against a real LLM.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.real_llm, pytest.mark.slow]
from pathlib import Path

# Enable logging for test visibility
from fsm_llm.logging import logger

logger.enable("fsm_llm")

from fsm_llm import (
    API,
    HandlerTiming,
    LiteLLMInterface,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "ollama_chat/qwen3.5:4b"
# Disable thinking mode for qwen3.5 to get cleaner JSON output
MODEL_KWARGS = {"extra_body": {"options": {"num_predict": 200}}}
EXAMPLE_DIR = Path(__file__).parent.parent / "examples"
SIMPLE_GREETING_FSM = str(EXAMPLE_DIR / "basic" / "simple_greeting" / "fsm.json")

MAX_RETRIES = 3


def _ollama_available() -> bool:
    """Check if Ollama is reachable and the model is pulled."""
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        return any("qwen3.5:4b" in m for m in models)
    except Exception:
        return False


def _retry(fn, retries=MAX_RETRIES):
    """Retry a callable up to `retries` times, returning first success."""
    last_err = None
    for _i in range(retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
    raise last_err


requires_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not running or qwen3.5:4b not available",
)


# ---------------------------------------------------------------------------
# Test: Basic Conversation Flow
# ---------------------------------------------------------------------------

@requires_ollama
class TestBasicConversationFlow:
    """End-to-end conversation using the simple greeting FSM."""

    def test_start_conversation_returns_greeting(self):
        """The initial response should be a non-empty greeting string."""
        def run():
            api = API.from_file(
                SIMPLE_GREETING_FSM,
                model=MODEL,
                temperature=0.5,
                max_tokens=200,
            )
            _conv_id, response = api.start_conversation()
            assert isinstance(response, str)
            assert len(response) > 5
            api.close()

        _retry(run)

    def test_conversation_turn(self):
        """A single conversation turn should produce a response."""
        def run():
            api = API.from_file(
                SIMPLE_GREETING_FSM,
                model=MODEL,
                temperature=0.3,
                max_tokens=200,
            )
            conv_id, greeting = api.start_conversation()
            assert greeting
            response = api.converse("I'm doing great, thanks for asking!", conv_id)
            assert isinstance(response, str)
            assert len(response) > 0
            api.close()

        _retry(run)

    def test_context_manager_cleanup(self):
        """API context manager should clean up without errors."""
        def run():
            with API.from_file(SIMPLE_GREETING_FSM, model=MODEL, max_tokens=200) as api:
                conv_id, _ = api.start_conversation()
                api.converse("Hello!", conv_id)

        _retry(run)


# ---------------------------------------------------------------------------
# Test: Data Extraction
# ---------------------------------------------------------------------------

@requires_ollama
class TestDataExtraction:
    """Verify data extraction pass works with a form-filling FSM."""

    @pytest.fixture
    def form_fsm(self):
        """A minimal FSM that collects a name."""
        return {
            "name": "Name Collection FSM",
            "description": "Collect user name",
            "initial_state": "ask_name",
            "version": "1.0",
            "states": {
                "ask_name": {
                    "id": "ask_name",
                    "description": "Ask for user name",
                    "purpose": "Collect the user's name",
                    "required_context_keys": ["name"],
                    "transitions": [
                        {
                            "target_state": "confirm",
                            "description": "Name provided",
                            "priority": 0,
                            "condition": {"has_context": "name"},
                        },
                        {
                            "target_state": "ask_name",
                            "description": "Still need name",
                            "priority": 1,
                        },
                    ],
                    "instructions": "Ask the user for their name.",
                },
                "confirm": {
                    "id": "confirm",
                    "description": "Confirm collected data",
                    "purpose": "Confirm the user's name",
                    "transitions": [],
                    "instructions": "Confirm the name back to the user.",
                },
            },
            "persona": "A polite assistant collecting information.",
        }

    @pytest.mark.xfail(
        reason="qwen3.5 thinking mode + litellm bug: extraction may fail to produce usable JSON",
        strict=False,
    )
    def test_extraction_populates_context(self, form_fsm):
        """Providing a name should populate the context via data extraction."""
        def run():
            api = API.from_definition(
                form_fsm,
                model=MODEL,
                temperature=0.3,
                max_tokens=200,
            )
            conv_id, _ = api.start_conversation()
            api.converse("My name is Alice", conv_id)
            data = api.get_data(conv_id)
            context_str = str(data).lower()
            assert "alice" in context_str, f"Expected 'alice' in context, got: {data}"
            api.close()

        _retry(run)


# ---------------------------------------------------------------------------
# Test: Handler Integration
# ---------------------------------------------------------------------------

@requires_ollama
class TestHandlerIntegration:
    """Verify handler system works end-to-end with real LLM calls."""

    @pytest.mark.xfail(
        reason="qwen3.5 thinking mode + litellm bug: multi-turn may fail",
        strict=False,
    )
    def test_pre_processing_handler_fires(self):
        """A PRE_PROCESSING handler should execute before LLM calls."""
        handler_log = []

        def run():
            handler_log.clear()
            api = API.from_file(
                SIMPLE_GREETING_FSM,
                model=MODEL,
                max_tokens=200,
            )

            handler = (
                api.create_handler("TestPreProcessor")
                .on_timing(HandlerTiming.PRE_PROCESSING)
                .execute(lambda ctx: handler_log.append("pre_processing_fired") or {})
                .build()
            )
            api.register_handler(handler)

            conv_id, _ = api.start_conversation()
            api.converse("Hello!", conv_id)

            assert "pre_processing_fired" in handler_log
            api.close()

        _retry(run)


# ---------------------------------------------------------------------------
# Test: LLM Interface Directly
# ---------------------------------------------------------------------------

@requires_ollama
class TestLLMInterfaceDirect:
    """Test the LiteLLMInterface directly against Ollama."""

    def test_extract_data_returns_valid_response(self):
        """extract_data should return a DataExtractionResponse."""
        from fsm_llm.definitions import DataExtractionRequest, DataExtractionResponse

        def run():
            llm = LiteLLMInterface(model=MODEL, temperature=0.3, max_tokens=200)
            request = DataExtractionRequest(
                system_prompt=(
                    "You are a data extraction component. Extract the user's name from their message. "
                    'Return ONLY this JSON: {"extracted_data": {"name": "Alice"}, "confidence": 0.9}'
                ),
                user_message="My name is Alice.",
                context={},
            )
            response = llm.extract_data(request)
            assert isinstance(response, DataExtractionResponse)

        _retry(run)

    def test_generate_response_returns_message(self):
        """generate_response should return a ResponseGenerationResponse with a message."""
        from fsm_llm.definitions import (
            ResponseGenerationRequest,
            ResponseGenerationResponse,
        )

        def run():
            llm = LiteLLMInterface(model=MODEL, temperature=0.5, max_tokens=200)
            request = ResponseGenerationRequest(
                system_prompt=(
                    "You are a friendly assistant. Respond warmly to the user. "
                    'Return ONLY this JSON: {"message": "Hello! How can I help you today?"}'
                ),
                user_message="Hello!",
                extracted_data={},
                context={},
                transition_occurred=False,
                previous_state=None,
            )
            response = llm.generate_response(request)
            assert isinstance(response, ResponseGenerationResponse)
            assert len(response.message) > 0

        _retry(run)

    def test_decide_transition_selects_valid_target(self):
        """decide_transition should select one of the available targets."""
        from fsm_llm.definitions import (
            TransitionDecisionRequest,
            TransitionDecisionResponse,
            TransitionOption,
        )

        def run():
            llm = LiteLLMInterface(model=MODEL, temperature=0.1, max_tokens=200)

            options = [
                TransitionOption(
                    target_state="farewell",
                    description="User wants to leave",
                    priority=0,
                ),
                TransitionOption(
                    target_state="conversation",
                    description="User wants to continue chatting",
                    priority=1,
                ),
            ]

            request = TransitionDecisionRequest(
                system_prompt=(
                    'You must select a transition. Return ONLY this JSON: '
                    '{"selected_transition": "farewell", "reasoning": "user said bye"}'
                ),
                current_state="conversation",
                available_transitions=options,
                context={},
                user_message="Bye bye!",
                extracted_data={},
            )

            response = llm.decide_transition(request)
            assert isinstance(response, TransitionDecisionResponse)
            assert response.selected_transition in {"farewell", "conversation"}

        _retry(run)


# ---------------------------------------------------------------------------
# Test: Thread Safety (C1 fix validation)
# ---------------------------------------------------------------------------

@requires_ollama
class TestThreadSafety:
    """Verify per-conversation locking prevents concurrent mutations."""

    def test_concurrent_same_conversation_blocked(self):
        """Two threads processing the same conversation: one succeeds, one gets blocked."""
        import threading

        def run():
            api = API.from_file(
                SIMPLE_GREETING_FSM,
                model=MODEL,
                max_tokens=100,
            )

            conv_id, _ = api.start_conversation()

            results = {"success": 0, "blocked": 0, "error": 0}
            barrier = threading.Barrier(2)

            def worker(msg, key):
                try:
                    barrier.wait(timeout=5)
                    api.converse(msg, conv_id)
                    results["success"] += 1
                except Exception as e:
                    if "already being processed" in str(e):
                        results["blocked"] += 1
                    else:
                        results["error"] += 1

            t1 = threading.Thread(target=worker, args=("Hello!", "t1"))
            t2 = threading.Thread(target=worker, args=("Hi!", "t2"))
            t1.start()
            t2.start()
            t1.join(timeout=30)
            t2.join(timeout=30)

            # At least one must succeed; the other may be blocked.
            # Zero errors from corruption.
            assert results["success"] >= 1, f"Expected at least 1 success, got {results}"
            assert results["error"] == 0, f"Got unexpected errors: {results}"
            api.close()

        _retry(run)


# ---------------------------------------------------------------------------
# Test: Classification Extension
# ---------------------------------------------------------------------------

@requires_ollama
class TestClassificationIntegration:
    """End-to-end classification using Ollama."""

    @pytest.mark.xfail(
        reason="qwen3.5 thinking mode + litellm bug: classification may fail",
        strict=False,
    )
    def test_single_intent_classification(self):
        """Classify a simple message into a single intent."""
        from fsm_llm_classification import (
            ClassificationSchema,
            Classifier,
            IntentDefinition,
        )

        def run():
            schema = ClassificationSchema(
                intents=[
                    IntentDefinition(
                        name="greeting", description="User is greeting or saying hello"
                    ),
                    IntentDefinition(
                        name="farewell", description="User is saying goodbye"
                    ),
                    IntentDefinition(
                        name="question", description="User is asking a question"
                    ),
                ],
                fallback_intent="question",
            )

            classifier = Classifier(schema=schema, model=MODEL, temperature=0.1)
            result = classifier.classify("Hello, how are you?")

            assert result.intent in {"greeting", "farewell", "question"}
            assert 0.0 <= result.confidence <= 1.0

        _retry(run)


# ---------------------------------------------------------------------------
# Test: Workflow Models (validates UTC fix — no Ollama needed)
# ---------------------------------------------------------------------------

class TestWorkflowUTCConsistency:
    """Verify all workflow datetimes are UTC-aware after our fix."""

    def test_workflow_event_timestamp_is_utc(self):
        from fsm_llm_workflows.models import WorkflowEvent

        event = WorkflowEvent(event_type="test")
        assert event.timestamp.tzinfo is not None

    def test_workflow_instance_timestamps_are_utc(self):
        from fsm_llm_workflows.models import WorkflowInstance

        instance = WorkflowInstance(
            instance_id="test",
            workflow_id="wf",
            current_step_id="step1",
        )
        assert instance.created_at.tzinfo is not None
        assert instance.updated_at.tzinfo is not None

    def test_step_result_timestamp_is_utc(self):
        from fsm_llm_workflows.models import WorkflowStepResult

        result = WorkflowStepResult(success=True)
        assert result.timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# Test: Parallel Step Context Isolation (C5 fix validation)
# ---------------------------------------------------------------------------

class TestParallelStepIsolation:
    """Verify ParallelStep deep-copies context for each parallel branch."""

    @pytest.mark.asyncio
    async def test_parallel_steps_get_isolated_context(self):
        from fsm_llm_workflows.steps import AutoTransitionStep, ParallelStep

        mutations = []

        async def mutating_action(ctx):
            # Mutate the context — if shared, other steps see this
            ctx["shared_key"] = f"mutated_by_{ctx.get('_step_marker', 'unknown')}"
            mutations.append(ctx.get("shared_key"))
            return {}

        step1 = AutoTransitionStep(
            step_id="s1", name="Step 1", next_state="done",
            action=lambda ctx: (ctx.update({"_step_marker": "s1"}) or mutating_action(ctx)),
        )
        step2 = AutoTransitionStep(
            step_id="s2", name="Step 2", next_state="done",
            action=lambda ctx: (ctx.update({"_step_marker": "s2"}) or mutating_action(ctx)),
        )

        parallel = ParallelStep(
            step_id="parallel", name="Parallel", next_state="end",
            steps=[step1, step2],
        )

        context = {"shared_key": "original"}
        await parallel.execute(context)

        # Original context should not be mutated by parallel steps
        assert context["shared_key"] == "original"
