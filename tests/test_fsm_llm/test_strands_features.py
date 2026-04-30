"""Tests for Strands-adapted features: schema enforcement, invocation state, streaming, session persistence."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from fsm_llm.dialog.api import API
from fsm_llm import FileSessionStore, SessionState, WorkingMemory
from fsm_llm.dialog.definitions import (
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.memory import BUFFER_METADATA, DEFAULT_HIDDEN_BUFFERS
from fsm_llm.runtime._litellm import LiteLLMInterface, LLMInterface

# ================================================================
# Helpers
# ================================================================


def _minimal_fsm_dict(name: str = "test") -> dict:
    """Minimal valid FSM definition for testing."""
    return {
        "name": name,
        "description": "Test FSM",
        "initial_state": "start",
        "persona": "Test bot",
        "states": {
            "start": {
                "id": "start",
                "description": "Start state",
                "purpose": "Begin conversation",
                "response_instructions": "Say hello",
                "transitions": [
                    {
                        "target_state": "end",
                        "description": "Move to end",
                        "conditions": [
                            {
                                "description": "Always",
                                "logic": {"==": [1, 1]},
                            }
                        ],
                    }
                ],
            },
            "end": {
                "id": "end",
                "description": "End state",
                "purpose": "End conversation",
                "response_instructions": "Say goodbye",
                "transitions": [],
            },
        },
    }


class MockLLM(LLMInterface):
    """Mock LLM that returns canned responses."""

    def __init__(self, response_text: str = "Hello!"):
        self.response_text = response_text
        self.last_request = None

    def generate_response(self, request):
        self.last_request = request
        return ResponseGenerationResponse(
            message=self.response_text,
            message_type="response",
        )

    def generate_response_stream(self, request):
        self.last_request = request
        for word in self.response_text.split():
            yield word + " "

    def extract_field(self, request):
        from fsm_llm.dialog.definitions import FieldExtractionResponse

        return FieldExtractionResponse(
            field_name=request.field_name,
            value="test",
            confidence=1.0,
        )


# ================================================================
# Feature #12: Schema-Enforced Output
# ================================================================


class TestSchemaEnforcedOutput:
    """Tests for response_format on ResponseGenerationRequest."""

    def test_response_generation_request_has_response_format(self):
        """ResponseGenerationRequest accepts response_format field."""
        req = ResponseGenerationRequest(
            system_prompt="Test",
            user_message="Hi",
            response_format={"type": "json_object"},
        )
        assert req.response_format == {"type": "json_object"}

    def test_response_generation_request_default_none(self):
        """response_format defaults to None."""
        req = ResponseGenerationRequest(
            system_prompt="Test",
            user_message="Hi",
        )
        assert req.response_format is None

    def test_generate_response_passes_format_to_call(self):
        """When response_format is set, it's passed through to _make_llm_call."""
        llm = LiteLLMInterface(model="gpt-4o", api_key="test")

        schema_format = {
            "type": "json_schema",
            "json_schema": {"name": "Test", "schema": {"type": "object"}},
        }
        request = ResponseGenerationRequest(
            system_prompt="Test prompt",
            user_message="Hello",
            response_format=schema_format,
        )

        # Mock _make_llm_call to verify response_format is passed
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"message": "hi"}'

        with patch.object(
            llm, "_make_llm_call", return_value=mock_response
        ) as mock_call:
            llm.generate_response(request)
            mock_call.assert_called_once()
            _, kwargs = mock_call.call_args
            assert kwargs["response_format"] == schema_format

    def test_make_llm_call_applies_response_format(self):
        """_make_llm_call applies explicit response_format for response_generation."""
        llm = LiteLLMInterface(model="gpt-4o", api_key="test")

        fmt = {"type": "json_object"}
        with (
            patch("fsm_llm.runtime._litellm.completion") as mock_comp,
            patch(
                "fsm_llm.runtime._litellm.get_supported_openai_params",
                return_value=["response_format"],
            ),
        ):
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = '{"message": "ok"}'
            mock_comp.return_value = mock_resp

            llm._make_llm_call(
                [{"role": "user", "content": "hi"}],
                "response_generation",
                response_format=fmt,
            )

            call_kwargs = mock_comp.call_args[1]
            assert call_kwargs["response_format"] == fmt

    def test_make_llm_call_no_format_when_unsupported(self):
        """response_format is NOT applied when provider doesn't support it."""
        llm = LiteLLMInterface(model="some-model", api_key="test")

        with (
            patch("fsm_llm.runtime._litellm.completion") as mock_comp,
            patch(
                "fsm_llm.runtime._litellm.get_supported_openai_params", return_value=[]
            ),
        ):
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "ok"
            mock_comp.return_value = mock_resp

            llm._make_llm_call(
                [{"role": "user", "content": "hi"}],
                "response_generation",
                response_format={"type": "json_object"},
            )

            call_kwargs = mock_comp.call_args[1]
            assert "response_format" not in call_kwargs

    def test_agent_sets_output_response_format_in_context(self):
        """When output_schema is set, agents store response_format in context."""
        from fsm_llm.stdlib.agents.base import BaseAgent
        from fsm_llm.stdlib.agents.definitions import AgentConfig, AgentResult

        class DummyAgent(BaseAgent):
            def run(self, task, initial_context=None):
                return AgentResult(answer="done", success=True)

            def _register_handlers(self, api):
                pass

        class MyOutput(BaseModel):
            summary: str
            score: float

        agent = DummyAgent(config=AgentConfig(output_schema=MyOutput))
        context = agent._init_context("test task")

        assert "_output_response_format" in context
        fmt = context["_output_response_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "MyOutput"
        assert "properties" in fmt["json_schema"]["schema"]


# ================================================================
# Feature #11: Invocation State (Hidden Metadata Buffer)
# ================================================================


class TestInvocationState:
    """Tests for hidden metadata buffer in WorkingMemory."""

    def test_metadata_buffer_constant_exists(self):
        """BUFFER_METADATA constant is defined."""
        assert BUFFER_METADATA == "metadata"

    def test_default_hidden_buffers(self):
        """Default hidden buffers includes metadata."""
        assert "metadata" in DEFAULT_HIDDEN_BUFFERS

    def test_metadata_excluded_from_get_all_data(self):
        """Metadata buffer data does NOT appear in get_all_data()."""
        mem = WorkingMemory(
            buffers=("core", "scratch", "metadata"),
            hidden_buffers={"metadata"},
        )
        mem.set("core", "name", "Alice")
        mem.set("metadata", "user_tier", "premium")
        mem.set("metadata", "billing_id", "B-123")

        all_data = mem.get_all_data()
        assert "name" in all_data
        assert "user_tier" not in all_data
        assert "billing_id" not in all_data

    def test_metadata_excluded_from_search(self):
        """Metadata buffer data does NOT appear in search results."""
        mem = WorkingMemory(
            buffers=("core", "metadata"),
            hidden_buffers={"metadata"},
        )
        mem.set("core", "name", "Alice")
        mem.set("metadata", "secret_key", "sk-12345")

        results = mem.search("12345")
        assert len(results) == 0

        results = mem.search("Alice")
        assert len(results) == 1
        assert results[0] == ("core", "name", "Alice")

    def test_metadata_excluded_from_scoped_view(self):
        """Metadata buffer data does NOT appear in to_scoped_view()."""
        mem = WorkingMemory(
            buffers=("core", "metadata"),
            hidden_buffers={"metadata"},
        )
        mem.set("core", "name", "Alice")
        mem.set("metadata", "tier", "premium")

        view = mem.to_scoped_view(["name", "tier"])
        assert "name" in view
        assert "tier" not in view

    def test_metadata_directly_accessible(self):
        """Hidden buffer data IS accessible via direct get/set."""
        mem = WorkingMemory(
            buffers=("core", "metadata"),
            hidden_buffers={"metadata"},
        )
        mem.set("metadata", "user_tier", "premium")
        assert mem.get("metadata", "user_tier") == "premium"

    def test_metadata_in_to_dict(self):
        """Hidden buffers ARE included in serialization (to_dict)."""
        mem = WorkingMemory(
            buffers=("core", "metadata"),
            hidden_buffers={"metadata"},
        )
        mem.set("metadata", "tier", "premium")

        data = mem.to_dict()
        assert "metadata" in data
        assert data["metadata"]["tier"] == "premium"

    def test_from_dict_preserves_hidden(self):
        """from_dict can restore hidden buffers."""
        data = {
            "core": {"name": "Alice"},
            "metadata": {"tier": "premium"},
        }
        mem = WorkingMemory.from_dict(data, hidden_buffers={"metadata"})

        assert mem.get("metadata", "tier") == "premium"
        all_data = mem.get_all_data()
        assert "tier" not in all_data


# ================================================================
# Feature #2: Response Streaming
# ================================================================


class TestResponseStreaming:
    """Tests for streaming response generation."""

    def test_llm_interface_has_stream_method(self):
        """LLMInterface ABC has generate_response_stream with default impl."""
        mock_llm = MockLLM("Hello world")
        chunks = list(
            mock_llm.generate_response_stream(
                ResponseGenerationRequest(
                    system_prompt="Test",
                    user_message="Hi",
                )
            )
        )
        # MockLLM overrides stream to yield word-by-word
        assert len(chunks) == 2
        assert "".join(chunks).strip() == "Hello world"

    def test_mock_llm_streams_chunks(self):
        """Mock LLM yields word-by-word chunks."""
        mock_llm = MockLLM("Hello beautiful world")
        req = ResponseGenerationRequest(system_prompt="Test", user_message="Hi")
        chunks = list(mock_llm.generate_response_stream(req))
        assert len(chunks) == 3
        assert "".join(chunks).strip() == "Hello beautiful world"

    def test_api_converse_stream_exists(self):
        """API.converse_stream method exists."""
        assert hasattr(API, "converse_stream")

    def test_api_converse_stream_yields_chunks(self):
        """converse_stream yields response chunks."""
        mock_llm = MockLLM("Streaming works great")
        api = API(
            fsm_definition=_minimal_fsm_dict(),
            llm_interface=mock_llm,
        )
        conv_id, _ = api.start_conversation()

        chunks = list(api.converse_stream("Hello", conv_id))
        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert "Streaming" in full_response

    def test_converse_stream_is_iterator(self):
        """converse_stream returns an Iterator."""
        mock_llm = MockLLM("Test")
        api = API(
            fsm_definition=_minimal_fsm_dict(),
            llm_interface=mock_llm,
        )
        conv_id, _ = api.start_conversation()

        result = api.converse_stream("Hello", conv_id)
        assert isinstance(result, Iterator)

    def test_litellm_stream_method_exists(self):
        """LiteLLMInterface has generate_response_stream."""
        llm = LiteLLMInterface(model="test", api_key="test")
        assert hasattr(llm, "generate_response_stream")

    def test_litellm_stream_sentinel_prompt(self):
        """Streaming with sentinel prompt '.' yields empty string immediately."""
        llm = LiteLLMInterface(model="test", api_key="test")
        req = ResponseGenerationRequest(system_prompt=".", user_message="test")
        chunks = list(llm.generate_response_stream(req))
        assert chunks == [""]


# ================================================================
# Feature #9: Session Persistence
# ================================================================


class TestSessionPersistence:
    """Tests for SessionStore and FileSessionStore."""

    def test_session_state_model(self):
        """SessionState is a valid Pydantic model."""
        state = SessionState(
            conversation_id="conv-1",
            fsm_id="fsm-1",
            current_state="start",
            context_data={"name": "Alice"},
            conversation_history=[{"user": "Hi", "system": "Hello!"}],
        )
        assert state.conversation_id == "conv-1"
        assert state.current_state == "start"
        assert state.saved_at  # auto-populated

    def test_session_state_serialization(self):
        """SessionState roundtrips through JSON."""
        state = SessionState(
            conversation_id="conv-1",
            fsm_id="fsm-1",
            current_state="start",
            context_data={"score": 42},
        )
        data = state.model_dump()
        restored = SessionState.model_validate(data)
        assert restored.conversation_id == state.conversation_id
        assert restored.context_data == state.context_data

    def test_file_session_store_save_load(self):
        """FileSessionStore saves and loads correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(tmpdir)
            state = SessionState(
                conversation_id="conv-1",
                fsm_id="fsm-1",
                current_state="start",
                context_data={"name": "Alice"},
            )
            store.save("conv-1", state)
            loaded = store.load("conv-1")

            assert loaded is not None
            assert loaded.conversation_id == "conv-1"
            assert loaded.context_data["name"] == "Alice"

    def test_file_session_store_load_missing(self):
        """Loading a non-existent session returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(tmpdir)
            assert store.load("nonexistent") is None

    def test_file_session_store_delete(self):
        """Deleting a session removes it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(tmpdir)
            state = SessionState(
                conversation_id="c1",
                fsm_id="f1",
                current_state="start",
            )
            store.save("c1", state)
            assert store.exists("c1")
            assert store.delete("c1")
            assert not store.exists("c1")

    def test_file_session_store_list(self):
        """list_sessions returns all saved session IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(tmpdir)
            for i in range(3):
                store.save(
                    f"session-{i}",
                    SessionState(
                        conversation_id=f"c-{i}",
                        fsm_id="f",
                        current_state="start",
                    ),
                )
            sessions = store.list_sessions()
            assert len(sessions) == 3
            assert set(sessions) == {"session-0", "session-1", "session-2"}

    def test_file_session_store_path_traversal_safety(self):
        """Path traversal attempts are rejected with ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(tmpdir)
            state = SessionState(
                conversation_id="test",
                fsm_id="f",
                current_state="start",
            )
            with pytest.raises(ValueError, match="Invalid session_id"):
                store.save("../etc/passwd", state)

    def test_api_session_store_parameter(self):
        """API accepts session_store parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(tmpdir)
            mock_llm = MockLLM()
            api = API(
                fsm_definition=_minimal_fsm_dict(),
                llm_interface=mock_llm,
                session_store=store,
            )
            assert api._session_store is store

    def test_api_save_session(self):
        """API.save_session persists state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(tmpdir)
            mock_llm = MockLLM()
            api = API(
                fsm_definition=_minimal_fsm_dict(),
                llm_interface=mock_llm,
                session_store=store,
            )
            conv_id, _ = api.start_conversation()
            api.save_session(conv_id)

            loaded = store.load(conv_id)
            assert loaded is not None
            assert loaded.conversation_id == conv_id

    def test_api_auto_save_on_converse(self):
        """When session_store is set, converse() auto-saves."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(tmpdir)
            mock_llm = MockLLM()
            api = API(
                fsm_definition=_minimal_fsm_dict(),
                llm_interface=mock_llm,
                session_store=store,
            )
            conv_id, _ = api.start_conversation()
            api.converse("Hello", conv_id)

            loaded = store.load(conv_id)
            assert loaded is not None

    def test_api_load_session(self):
        """API.load_session returns saved state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(tmpdir)
            mock_llm = MockLLM()
            api = API(
                fsm_definition=_minimal_fsm_dict(),
                llm_interface=mock_llm,
                session_store=store,
            )
            conv_id, _ = api.start_conversation()
            api.save_session(conv_id)

            state = api.load_session(conv_id)
            assert state is not None
            assert state.fsm_id == api.fsm_id

    def test_api_no_session_store_raises(self):
        """save_session raises when no store is configured."""
        mock_llm = MockLLM()
        api = API(
            fsm_definition=_minimal_fsm_dict(),
            llm_interface=mock_llm,
        )
        conv_id, _ = api.start_conversation()
        with pytest.raises(Exception, match="No session store"):
            api.save_session(conv_id)
