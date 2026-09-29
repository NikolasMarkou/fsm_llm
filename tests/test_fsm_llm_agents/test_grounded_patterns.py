"""Agent patterns driven through the real ``API`` by ``PromptGroundedLLM``.

The fake answers a field only when the prompt shows the evidence for it, so
these tests fail when a pattern asks the model for a value without putting
the task or the prior turns in front of it (the RC1 loop class).
"""

from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest

from fsm_llm.agents.constants import ContextKeys
from fsm_llm.agents.debate import DebateAgent
from fsm_llm.api import API
from fsm_llm.definitions import (
    BulkExtractionRequest,
    FieldExtractionRequest,
    ResponseGenerationRequest,
)
from tests.conftest import PromptGroundedLLM, network_exempt


def _field_request(**overrides: object) -> FieldExtractionRequest:
    fields: dict[str, object] = {
        "system_prompt": "Extract the city.",
        "user_message": "Continue.",
        "field_name": "city",
    }
    fields.update(overrides)
    return FieldExtractionRequest.model_validate(fields)


class TestPromptGroundedLLM:
    """The fake's own contract (``tests/conftest.py``)."""

    def test_field_found_in_system_prompt(self):
        llm = PromptGroundedLLM(facts={"city": ("Paris", "capital of France")})
        request = _field_request(system_prompt="Task: the capital of France.")

        response = llm.extract_field(request)

        assert (response.value, response.is_valid) == ("Paris", True)

    def test_field_found_in_user_message(self):
        llm = PromptGroundedLLM(facts={"city": ("Paris", "capital of France")})

        response = llm.extract_field(
            _field_request(user_message="What is the capital of France?")
        )

        assert response.value == "Paris"

    def test_field_found_in_context(self):
        llm = PromptGroundedLLM(facts={"city": ("Paris", "capital of France")})

        response = llm.extract_field(
            _field_request(context={"task": "Name the capital of France"})
        )

        assert response.value == "Paris"

    def test_field_without_evidence_is_null(self):
        llm = PromptGroundedLLM(facts={"city": ("Paris", "capital of France")})

        response = llm.extract_field(_field_request())

        assert (response.value, response.is_valid, response.confidence) == (
            None,
            False,
            0.0,
        )

    def test_unknown_field_is_null(self):
        llm = PromptGroundedLLM(facts={"city": ("Paris", "Continue.")})

        response = llm.extract_field(_field_request(field_name="country"))

        assert response.value is None

    def test_false_value_is_a_grounded_answer(self):
        llm = PromptGroundedLLM(facts={"done": (False, "not finished")})

        response = llm.extract_field(
            _field_request(
                field_name="done", field_type="bool", context={"s": "not finished"}
            )
        )

        assert (response.value, response.is_valid) == (False, True)

    def test_bulk_returns_named_keys_with_evidence(self):
        llm = PromptGroundedLLM(
            facts={
                "city": ("Paris", "capital of France"),
                "river": ("Seine", "which river"),
                "country": ("France", "capital of France"),
            }
        )
        request = BulkExtractionRequest(
            system_prompt=(
                'Task: the capital of France.\n- "city": the city\n- "river": the river'
            ),
            user_message="Continue.",
        )

        response = llm.extract_bulk_data(request)

        # river is named but ungrounded; country is grounded but not asked for.
        assert response.extracted_data == {"city": "Paris"}

    def test_context_free_bulk_prompt_yields_nothing(self):
        llm = PromptGroundedLLM(facts={"city": ("Paris", "capital of France")})
        request = BulkExtractionRequest(
            system_prompt='Extract the following as JSON:\n- "city": the city',
            user_message="Continue.",
        )

        assert llm.extract_bulk_data(request).extracted_data == {}

    def test_response_scripted_per_state(self):
        llm = PromptGroundedLLM(responses={"judge": "verdict"}, default_response="d")
        judge = ResponseGenerationRequest(
            system_prompt="<current_state>judge</current_state>", user_message=""
        )
        other = ResponseGenerationRequest(
            system_prompt="<current_state>propose</current_state>", user_message=""
        )

        assert llm.generate_response(judge).message == "verdict"
        assert llm.generate_response(other).message == "d"

    def test_records_every_request_in_order(self):
        llm = PromptGroundedLLM()
        field = _field_request()
        bulk = BulkExtractionRequest(system_prompt="x", user_message="y")
        reply = ResponseGenerationRequest(system_prompt="x", user_message="y")

        llm.extract_field(field)
        llm.extract_bulk_data(bulk)
        llm.generate_response(reply)

        assert llm.requests == [
            ("extract_field", field),
            ("extract_bulk_data", bulk),
            ("generate_response", reply),
        ]
        assert llm.calls("extract_field") == [field]

    def test_grounds_through_real_api(self):
        """Per-field extraction through the real pipeline sees the user turn."""
        fsm = {
            "name": "Grounded",
            "description": "Collect a city",
            "initial_state": "ask",
            "states": {
                "ask": {
                    "id": "ask",
                    "description": "Ask for the city",
                    "purpose": "Collect the city",
                    "required_context_keys": ["city"],
                    "response_instructions": "Ask for the city",
                    "transitions": [
                        {
                            "target_state": "done",
                            "description": "City known",
                            "priority": 100,
                            "conditions": [
                                {
                                    "description": "City collected",
                                    "requires_context_keys": ["city"],
                                }
                            ],
                        }
                    ],
                },
                "done": {
                    "id": "done",
                    "description": "Terminal",
                    "purpose": "Confirm",
                    "response_instructions": "Confirm the city",
                },
            },
        }
        llm = PromptGroundedLLM(
            facts={"city": ("Paris", "capital of France")},
            responses={"done": "Paris it is."},
        )
        api = API.from_definition(fsm, llm_interface=llm)
        conv_id, _ = api.start_conversation()

        reply = api.converse("I live in the capital of France.", conv_id)

        assert api.get_data(conv_id)["city"] == "Paris"
        assert api.get_current_state(conv_id) == "done"
        assert reply == "Paris it is."
        assert llm.calls("extract_field")


class TestDebateGrounding:
    """RC1 probe: debate's critique is asked of a prompt that lacks the proposition."""

    PROPOSITION = "Remote work raises output for focused engineering tasks."
    CRITIQUE = "It ignores the cost to team cohesion."

    # Step 19 moves debate to typed per-field extraction and removes the xfail.
    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="RC1 at HEAD: debate's bulk prompt is context-free ('Continue.')",
    )
    def test_critique_grounded_in_proposition(self):
        task = "Does remote work raise engineering output?"
        llm = PromptGroundedLLM(
            facts={
                ContextKeys.PROPOSITION: (self.PROPOSITION, task),
                ContextKeys.CRITIQUE: (self.CRITIQUE, self.PROPOSITION),
            }
        )
        agent = DebateAgent(num_rounds=1, llm_interface=llm)

        result = agent.run(task)

        assert result.final_context.get(ContextKeys.CRITIQUE) == self.CRITIQUE


def _loopback_listener() -> socket.socket:
    """A bound, listening TCP socket on 127.0.0.1 (bind is never blocked)."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    return server


class TestOfflineNetworkGuard:
    """E4: the autouse guard in ``conftest.py`` refuses TCP, loopback included.

    Each connect targets a live local listener, so without the guard it would
    succeed and these tests fail.
    """

    def test_connect_to_loopback_refused(self):
        server = _loopback_listener()
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(ConnectionRefusedError, match="network blocked"):
                client.connect(server.getsockname())
        finally:
            client.close()
            server.close()

    def test_connect_ex_to_loopback_refused(self):
        server = _loopback_listener()
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(ConnectionRefusedError, match="network blocked"):
                client.connect_ex(server.getsockname())
        finally:
            client.close()
            server.close()

    def test_unix_sockets_untouched(self):
        left, right = socket.socketpair()
        try:
            left.sendall(b"ok")
            assert right.recv(2) == b"ok"
        finally:
            left.close()
            right.close()

    @pytest.mark.parametrize(
        ("markers", "exempt"),
        [
            ((), False),
            (("slow",), False),
            (("real_llm",), True),
            (("integration",), True),
        ],
    )
    def test_live_markers_exempt(self, markers, exempt):
        node = SimpleNamespace(
            get_closest_marker=lambda name: name if name in markers else None
        )

        assert network_exempt(node) is exempt
