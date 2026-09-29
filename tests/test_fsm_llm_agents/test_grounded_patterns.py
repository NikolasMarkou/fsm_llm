"""Agent patterns driven through the real ``API`` by ``PromptGroundedLLM``.

The fake answers a field only when the prompt shows the evidence for it, so
these tests fail when a pattern asks the model for a value without putting
the task or the prior turns in front of it (the RC1 loop class).
"""

from __future__ import annotations

import itertools
import json
import socket
import time
from types import SimpleNamespace

import pytest

from fsm_llm.agents.constants import ContextKeys
from fsm_llm.agents.debate import DebateAgent
from fsm_llm.agents.fsm_definitions import _typed_field_extraction
from fsm_llm.agents.handlers import make_fresh_keys_handler
from fsm_llm.api import API
from fsm_llm.definitions import (
    BulkExtractionRequest,
    FieldExtractionConfig,
    FieldExtractionRequest,
    ResponseGenerationRequest,
)
from tests.conftest import _CURRENT_STATE_TAG, PromptGroundedLLM, network_exempt


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

    # Step 19 moved debate to typed per-field extraction (was xfail-strict).
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


_TASK = "Assess the claim that TASK-7731 cut latency."
_TRACE_MARKER = "TRACE-MARKER-9f2"


def _loop_fsm(field_extractions: list[dict]) -> dict:
    """``produce -> check -> produce`` loop; only ``produce`` extracts."""
    return {
        "name": "TypedLoop",
        "description": "Typed per-field loop probe",
        "initial_state": "produce",
        "persona": "An analyst",
        "states": {
            "produce": {
                "id": "produce",
                "description": "Produce the critique",
                "purpose": "Write a critique of the claim",
                "extraction_instructions": "",
                "response_instructions": "",
                "field_extractions": field_extractions,
                "transitions": [
                    {"target_state": "check", "description": "Next", "priority": 900}
                ],
            },
            "check": {
                "id": "check",
                "description": "Check the critique",
                "purpose": "Decide whether to stop",
                "response_instructions": "",
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "Stop",
                        "priority": 10,
                        "conditions": [
                            {
                                "description": "stop set",
                                "logic": {"==": [{"var": "stop"}, True]},
                            }
                        ],
                    },
                    {"target_state": "produce", "description": "Redo", "priority": 900},
                ],
            },
            "done": {
                "id": "done",
                "description": "Terminal",
                "purpose": "Report",
                "response_instructions": "Report the critique",
            },
        },
    }


def _start_loop(llm: PromptGroundedLLM, field_extractions: list[dict]):
    api = API.from_definition(_loop_fsm(field_extractions), llm_interface=llm)
    conv_id, _ = api.start_conversation(
        {
            ContextKeys.TASK: _TASK,
            ContextKeys.OBSERVATIONS: ["[Step 1] Tool: probe | Result: p99 fell"],
            ContextKeys.AGENT_TRACE: [{"note": _TRACE_MARKER}],
        }
    )
    return api, conv_id


class TestTypedFieldExtraction:
    """Step 14 helpers, and Pre-Mortem check 2: typed fields ride the per-field path."""

    def test_helper_shape_is_a_valid_core_config(self):
        field = _typed_field_extraction(
            "critique",
            "str",
            "Name the weakest point.",
            extra_context_keys=[ContextKeys.PROPOSITION, ContextKeys.TASK],
            required=False,
        )
        config = FieldExtractionConfig.model_validate(field)

        assert (config.field_type, config.required) == ("str", False)
        assert config.context_keys == [
            ContextKeys.TASK,
            ContextKeys.OBSERVATIONS,
            ContextKeys.PROPOSITION,
        ]
        assert ContextKeys.AGENT_TRACE not in config.context_keys
        assert "from the task" in config.extraction_instructions
        assert "Name the weakest point." in config.extraction_instructions

    @pytest.mark.parametrize("field_type", ["str", "float", "list", "bool", "any"])
    def test_supported_types(self, field_type):
        field = _typed_field_extraction("item", field_type, "x")
        assert FieldExtractionConfig.model_validate(field).field_type == field_type

    @pytest.mark.parametrize("field_type", ["dict", "int"])
    def test_other_types_rejected(self, field_type):
        with pytest.raises(ValueError, match="unsupported"):
            _typed_field_extraction("item", field_type, "x")  # type: ignore[arg-type]

    @pytest.mark.parametrize("key", [ContextKeys.AGENT_TRACE, "_max_iterations"])
    def test_trace_and_internal_context_keys_rejected(self, key):
        with pytest.raises(ValueError, match="not allowed"):
            _typed_field_extraction("item", "str", "x", extra_context_keys=[key])

    @pytest.mark.parametrize(
        "name", ["reasoning", "confidence", "value", "field_name", "extracted_data"]
    )
    def test_envelope_named_field_rejected(self, name):
        # D-034/D-035: an envelope-named field was filled with the model's own
        # meta-commentary (the envelope's `reasoning`), so it is refused.
        with pytest.raises(ValueError, match="envelope"):
            _typed_field_extraction(name, "str", "x")

    def test_envelope_named_field_never_reaches_the_model(self):
        # The grounded fake would answer `reasoning` (its evidence is in the
        # prompt); the helper refuses the config before any state is built.
        llm = PromptGroundedLLM(facts={"reasoning": ("meta text", "TASK-7731")})
        with pytest.raises(ValueError, match="envelope"):
            _start_loop(llm, [_typed_field_extraction("reasoning", "str", "x")])
        assert llm.requests == []

    def test_typed_only_state_makes_no_bulk_call(self):
        llm = PromptGroundedLLM(facts={"critique": ("Too few samples.", "TASK-7731")})
        api, conv_id = _start_loop(
            llm, [_typed_field_extraction("critique", "str", "Name the flaw.")]
        )

        api.converse("Continue.", conv_id)

        assert llm.calls("extract_bulk_data") == []
        assert api.get_data(conv_id)["critique"] == "Too few samples."

    def test_per_field_request_carries_the_task(self):
        llm = PromptGroundedLLM(facts={"critique": ("Too few samples.", "TASK-7731")})
        api, conv_id = _start_loop(
            llm, [_typed_field_extraction("critique", "str", "Name the flaw.")]
        )

        api.converse("Continue.", conv_id)

        (request,) = llm.calls("extract_field")
        assert _TASK in request.system_prompt
        assert request.context[ContextKeys.TASK] == _TASK
        assert "p99 fell" in request.system_prompt

    def test_agent_trace_absent_from_narrowed_prompt(self):
        llm = PromptGroundedLLM(facts={"critique": ("Too few samples.", "TASK-7731")})
        api, conv_id = _start_loop(
            llm, [_typed_field_extraction("critique", "str", "Name the flaw.")]
        )

        api.converse("Continue.", conv_id)

        (request,) = llm.calls("extract_field")
        assert ContextKeys.AGENT_TRACE not in request.context
        assert _TRACE_MARKER not in request.system_prompt
        assert ContextKeys.AGENT_TRACE not in request.system_prompt

    def test_unnarrowed_config_leaks_agent_trace(self):
        """Control: without ``context_keys`` core dumps the trace (LOOP-08)."""
        llm = PromptGroundedLLM(facts={"critique": ("Too few samples.", "TASK-7731")})
        api, conv_id = _start_loop(
            llm,
            [
                {
                    "field_name": "critique",
                    "field_type": "str",
                    "extraction_instructions": "Name the flaw.",
                }
            ],
        )

        api.converse("Continue.", conv_id)

        (request,) = llm.calls("extract_field")
        assert _TRACE_MARKER in request.system_prompt

    @pytest.mark.parametrize("refresh", [True, False])
    def test_fresh_keys_handler_reopens_the_loop_value(self, refresh):
        llm = PromptGroundedLLM(facts={"critique": ("Round one.", "TASK-7731")})
        api, conv_id = _start_loop(
            llm, [_typed_field_extraction("critique", "str", "Name the flaw.")]
        )
        if refresh:
            api.register_handler(
                api.create_handler("fresh_critique")
                .on_state_entry("produce")
                .do(make_fresh_keys_handler(["critique"]))
            )

        api.converse("Continue.", conv_id)  # produce -> check
        llm.facts["critique"] = ("Round two.", "TASK-7731")
        api.converse("Continue.", conv_id)  # check -> produce
        api.converse("Continue.", conv_id)  # produce -> check

        expected = "Round two." if refresh else "Round one."
        assert api.get_data(conv_id)["critique"] == expected
        assert len(llm.calls("extract_field")) == (2 if refresh else 1)


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


# ---------------------------------------------------------------------------
# Step 13: one success contract with ``stop_reason`` (D-011 of plan 06a5ec0a)
# ---------------------------------------------------------------------------

_HAIKU = "Rain on tin roofs sings"


def _failing_evaluation(output: str, context: dict) -> object:
    from fsm_llm.agents.definitions import EvaluationResult

    return EvaluationResult(passed=False, score=0.1, feedback="more imagery")


def _passing_evaluation(output: str, context: dict) -> object:
    from fsm_llm.agents.definitions import EvaluationResult

    return EvaluationResult(passed=True, score=0.9, feedback="good")


def _lookup_registry(runs: list[str]) -> object:
    from fsm_llm.agents import ToolRegistry

    registry = ToolRegistry()

    def lookup(query: str) -> str:
        runs.append(query)
        return "The capital of France is Paris."

    registry.register_function(lookup, name="lookup", description="Look up a fact")
    return registry


# The tool observation carries "is Paris", so should_terminate is grounded only
# after the lookup ran.
_REACT_FACTS: dict[str, tuple[object, str]] = {
    "tool_name": ("lookup", "capital"),
    "tool_input": ({"query": "capital of France"}, "capital"),
    "should_terminate": (True, "is Paris"),
}


class TestSuccessContract:
    """``success`` means the run reached its goal; ``stop_reason`` says why it
    ended. A forced stop or forced pass still ships its answer but reports
    ``success=False`` (c1d5bfbc D-022 forbids raising, not reporting)."""

    def test_stop_reason_is_an_optional_serialisable_field(self):
        from fsm_llm.agents.definitions import AgentResult

        legacy = AgentResult(answer="a", success=True)
        assert legacy.stop_reason is None
        dumped = AgentResult(answer="a", success=False, stop_reason="stalled")
        assert '"stop_reason":"stalled"' in dumped.model_dump_json()

    def test_evalopt_forced_pass_at_max_refinements_reports_failure(self):
        from fsm_llm.agents import AgentConfig, EvaluatorOptimizerAgent

        agent = EvaluatorOptimizerAgent(
            evaluation_fn=_failing_evaluation,
            max_refinements=1,
            config=AgentConfig(max_iterations=10),
            llm_interface=PromptGroundedLLM(
                facts={"generated_output": (_HAIKU, "haiku")}
            ),
        )
        result = agent.run("Write a haiku about rain")

        assert result.answer == _HAIKU  # the forced output still ships
        assert result.success is False
        assert result.stop_reason == "forced_pass"

    def test_evalopt_real_pass_is_answered(self):
        from fsm_llm.agents import AgentConfig, EvaluatorOptimizerAgent

        agent = EvaluatorOptimizerAgent(
            evaluation_fn=_passing_evaluation,
            max_refinements=1,
            config=AgentConfig(max_iterations=10),
            llm_interface=PromptGroundedLLM(
                facts={"generated_output": (_HAIKU, "haiku")}
            ),
        )
        result = agent.run("Write a haiku about rain")

        assert (result.success, result.stop_reason) == (True, "answered")

    def _maker_checker(self, facts: dict[str, tuple[object, str]]):
        from fsm_llm.agents import AgentConfig, MakerCheckerAgent

        return MakerCheckerAgent(
            maker_instructions="Write a haiku.",
            checker_instructions="Review the haiku.",
            max_revisions=1,
            config=AgentConfig(max_iterations=10),
            llm_interface=PromptGroundedLLM(facts=facts),
        )

    def test_maker_checker_forced_pass_at_max_revisions_reports_failure(self):
        agent = self._maker_checker(
            {
                "draft_output": (_HAIKU, "haiku"),
                "checker_passed": (False, "haiku"),
                "quality_score": (0.2, "haiku"),
            }
        )
        result = agent.run("Write a haiku about rain")

        assert result.answer == _HAIKU  # the forced draft still ships
        assert result.success is False
        assert result.stop_reason == "forced_pass"

    def test_maker_checker_quality_pass_is_answered(self):
        agent = self._maker_checker(
            {
                "draft_output": (_HAIKU, "haiku"),
                "quality_score": (0.95, "haiku"),
            }
        )
        result = agent.run("Write a haiku about rain")

        assert (result.success, result.stop_reason) == (True, "answered")

    def test_react_answer_is_answered(self):
        from fsm_llm.agents import AgentConfig, ReactAgent

        runs: list[str] = []
        agent = ReactAgent(
            tools=_lookup_registry(runs),
            config=AgentConfig(max_iterations=6),
            llm_interface=PromptGroundedLLM(facts=_REACT_FACTS),
        )
        result = agent.run("What is the capital of France?")

        assert runs == ["capital of France"]
        assert (result.success, result.stop_reason) == (True, "answered")

    def test_react_forced_stop_reports_max_iterations(self):
        from fsm_llm.agents import AgentConfig, ReactAgent

        runs: list[str] = []
        facts = {k: v for k, v in _REACT_FACTS.items() if k != "should_terminate"}
        agent = ReactAgent(
            tools=_lookup_registry(runs),
            config=AgentConfig(max_iterations=6),
            llm_interface=PromptGroundedLLM(facts=facts),
        )
        result = agent.run("What is the capital of France?")

        assert runs, "the tool never ran"
        assert result.final_context[ContextKeys.MAX_ITERATIONS_REACHED] is True
        assert result.answer  # the last answer still ships
        assert result.success is False
        assert result.stop_reason == "max_iterations"

    def test_react_stall_reports_stalled(self):
        from fsm_llm.agents import AgentConfig, ReactAgent

        runs: list[str] = []
        agent = ReactAgent(
            tools=_lookup_registry(runs),
            config=AgentConfig(max_iterations=10),
            llm_interface=PromptGroundedLLM(facts={}),
        )
        result = agent.run("What is the capital of France?")

        assert runs == []
        assert (result.success, result.stop_reason) == (False, "stalled")

    def test_caller_cannot_seed_the_forced_reason(self):
        from fsm_llm.agents import AgentConfig, ReactAgent

        runs: list[str] = []
        agent = ReactAgent(
            tools=_lookup_registry(runs),
            config=AgentConfig(max_iterations=6),
            llm_interface=PromptGroundedLLM(facts=_REACT_FACTS),
        )
        result = agent.run(
            "What is the capital of France?",
            initial_context={ContextKeys.FORCED_STOP_REASON: "forced_pass"},
        )

        assert (result.success, result.stop_reason) == (True, "answered")

    def test_native_fc_exhausted_loop_reports_max_iterations(self):
        from fsm_llm.agents import AgentConfig
        from fsm_llm.agents.native_fc import NativeFunctionCallingReactAgent

        runs: list[str] = []
        call = {
            "content": "",
            "tool_calls": [
                {"id": "c1", "name": "lookup", "arguments": {"query": "France"}}
            ],
        }
        agent = NativeFunctionCallingReactAgent(
            tools=_lookup_registry(runs),
            config=AgentConfig(model="mock/model", max_iterations=2),
            complete_fn=lambda model, messages, schemas: call,
        )
        result = agent.run("What is the capital of France?")

        assert len(runs) == 2
        assert (result.success, result.stop_reason) == (False, "max_iterations")

    def test_native_fc_answer_is_answered(self):
        from fsm_llm.agents import AgentConfig
        from fsm_llm.agents.native_fc import NativeFunctionCallingReactAgent

        agent = NativeFunctionCallingReactAgent(
            tools=_lookup_registry([]),
            config=AgentConfig(model="mock/model"),
            complete_fn=lambda model, messages, schemas: {
                "content": "Paris",
                "tool_calls": [],
            },
        )
        result = agent.run("What is the capital of France?")

        assert (result.success, result.stop_reason) == (True, "answered")

    def test_self_consistency_empty_aggregate_is_no_result(self):
        from fsm_llm.agents import AgentConfig
        from fsm_llm.agents.self_consistency import SelfConsistencyAgent

        agent = SelfConsistencyAgent(
            num_samples=2,
            aggregation_fn=lambda samples: "",
            config=AgentConfig(max_iterations=5),
            llm_interface=PromptGroundedLLM(default_response="Paris"),
        )
        result = agent.run("What is the capital of France?")

        assert (result.success, result.stop_reason) == (False, "no_result")

    def test_swarm_handoff_cap_is_a_forced_stop(self):
        from fsm_llm.agents.definitions import AgentResult
        from fsm_llm.agents.swarm import SwarmAgent

        class _Handoff:
            """Always asks to hand off to the other agent."""

            def __init__(self, target: str) -> None:
                self.target = target

            def run(self, task: str, initial_context: object = None) -> AgentResult:
                return AgentResult(
                    answer=f"to {self.target}",
                    success=True,
                    stop_reason="answered",
                    final_context={"next_agent": self.target},
                )

        swarm = SwarmAgent(
            agents={"a": _Handoff("b"), "b": _Handoff("a")},  # type: ignore[dict-item]
            entry_agent="a",
            max_handoffs=2,
        )
        result = swarm.run("ping")

        assert result.answer.startswith("to ")
        assert (result.success, result.stop_reason) == (False, "max_iterations")

    def test_agent_server_exposes_stop_reason(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from fsm_llm.agents.definitions import AgentResult
        from fsm_llm.agents.remote import AgentServer

        class _Forced:
            def run(self, task: str, initial_context: object = None) -> AgentResult:
                return AgentResult(
                    answer="best effort", success=False, stop_reason="forced_pass"
                )

        client = TestClient(AgentServer(agent=_Forced()).app)  # type: ignore[arg-type]
        body = client.post("/invoke", json={"task": "t"}).json()

        assert body["success"] is False
        assert body["stop_reason"] == "forced_pass"


def _act_entry_probe(flags: list[object]) -> object:
    """A handler recording ``max_iterations_reached`` as each ``act`` is entered.

    Priority 1 runs it before the tool executor (100), so ``flags`` holds the
    flag each executor call saw.
    """
    from fsm_llm.handlers import create_handler

    return (
        create_handler("ActEntryProbe")
        .on_state_entry("act")
        .with_priority(1)
        .do(lambda ctx: flags.append(ctx.get(ContextKeys.MAX_ITERATIONS_REACHED)) or {})
    )


class _TurnAwareLLM(PromptGroundedLLM):
    """``PromptGroundedLLM`` plus fields computed from the request.

    ``derived`` maps a field name to ``fn(text, context) -> value | None``;
    those fields ignore ``facts``. The request is recorded either way.
    """

    def __init__(self, derived: dict[str, object], **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.derived = derived

    def extract_field(self, request: FieldExtractionRequest):
        fn = self.derived.get(request.field_name)
        if fn is None:
            return super().extract_field(request)
        from fsm_llm.definitions import FieldExtractionResponse

        self.requests.append(("extract_field", request))
        context = request.context or {}
        text = f"{request.system_prompt}\n{json.dumps(context, default=str)}"
        value = fn(text, context)  # type: ignore[operator]
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0 if value is not None else 0.0,
            reasoning="turn-aware fake",
            is_valid=value is not None,
        )


def _field_requests(llm: PromptGroundedLLM, name: str) -> list:
    return [r for r in llm.calls("extract_field") if r.field_name == name]


class TestReactLoop:
    """Step 15: the ReAct loop extracts grounded typed fields each think turn,
    ``max_iterations`` counts think turns, no tool runs after the forced stop,
    and executor/HITL feedback reaches the next think turn."""

    def _agent(self, llm, runs, max_iterations=6, **kwargs):
        from fsm_llm.agents import AgentConfig, ReactAgent

        return ReactAgent(
            tools=_lookup_registry(runs),
            config=AgentConfig(max_iterations=max_iterations),
            llm_interface=llm,
            **kwargs,
        )

    def test_max_iterations_counts_think_turns(self):
        # LOOP-04: 6 think turns = 5 tool turns (was 3: every transition counted).
        runs: list[str] = []
        facts = {k: v for k, v in _REACT_FACTS.items() if k != "should_terminate"}
        agent = self._agent(PromptGroundedLLM(facts=facts), runs)
        result = agent.run("What is the capital of France?")

        assert len(runs) == 5
        assert result.final_context[ContextKeys.ITERATION_COUNT] == 6
        assert result.stop_reason == "max_iterations"

    def test_no_tool_runs_after_the_forced_stop(self):
        # LOOP-05: the flag used to land on think -> act and the tool still ran.
        runs: list[str] = []
        flags: list[object] = []
        facts = {k: v for k, v in _REACT_FACTS.items() if k != "should_terminate"}
        agent = self._agent(
            PromptGroundedLLM(facts=facts),
            runs,
            handlers=[_act_entry_probe(flags)],
        )
        agent.run("What is the capital of France?")

        assert flags, "act was never entered"
        assert len(runs) == sum(1 for flag in flags if flag is not True)
        assert True not in flags

    def test_think_prompt_contains_the_previous_executor_warning(self):
        # LOOP-06: the no-tool WARNING lived in tool_result, which the
        # compactor deleted before think read it; here the tool selection is
        # grounded ONLY on that warning.
        runs: list[str] = []
        facts: dict[str, tuple[object, str]] = {
            "tool_name": ("lookup", "No tool was called"),
            "tool_input": ({"query": "capital of France"}, "No tool was called"),
            "should_terminate": (True, "is Paris"),
        }
        llm = PromptGroundedLLM(facts=facts)
        result = self._agent(llm, runs).run("What is the capital of France?")

        assert runs == ["capital of France"]
        seen = [
            r
            for r in _field_requests(llm, "tool_name")
            if "WARNING: No tool was called"
            in str((r.context or {}).get(ContextKeys.AGENT_FEEDBACK))
        ]
        assert seen, "no think prompt carried the executor warning"
        assert (result.success, result.stop_reason) == (True, "answered")

    def test_no_field_prompt_contains_agent_trace(self):
        # LOOP-08: agent_trace is unbounded; it must stay out of every prompt.
        runs: list[str] = []
        llm = PromptGroundedLLM(facts=_REACT_FACTS)
        self._agent(llm, runs).run("What is the capital of France?")

        requests = llm.calls("extract_field")
        assert runs and requests
        for request in requests:
            assert ContextKeys.AGENT_TRACE not in (request.context or {})
            assert '"action": "lookup(' not in request.system_prompt
        assert not llm.calls("extract_bulk_data")  # no context-free bulk call

    def test_think_turn_makes_three_field_calls_and_no_bulk_call(self):
        # D-034/D-035: think extracts tool_name, tool_input, should_terminate
        # (no `reasoning` call, it collided with the envelope key) and, with
        # use_classification=False, no context-free bulk call.
        from fsm_llm.agents.fsm_definitions import build_react_fsm

        runs: list[str] = []
        facts = dict(_REACT_FACTS)
        facts["reasoning"] = ("meta text", "capital")
        llm = PromptGroundedLLM(facts=facts)
        api = API.from_definition(
            build_react_fsm(_lookup_registry(runs)), llm_interface=llm
        )
        conv_id, _ = api.start_conversation(
            {
                ContextKeys.TASK: "What is the capital of France?",
                ContextKeys.OBSERVATIONS: [],
            }
        )
        llm.requests.clear()

        api.converse("Continue.", conv_id)

        names = sorted(r.field_name for r in llm.calls("extract_field"))
        assert names == ["should_terminate", "tool_input", "tool_name"]
        assert llm.calls("extract_bulk_data") == []
        assert ContextKeys.REASONING not in api.get_data(conv_id)

    def test_thought_is_not_extracted_per_turn(self):
        # LOOP-16 known open: the per-turn thought is no longer a model call,
        # so trace steps carry an empty thought even when the model would
        # answer one.
        runs: list[str] = []
        facts = dict(_REACT_FACTS)
        facts["reasoning"] = ("meta text", "capital")
        llm = PromptGroundedLLM(facts=facts)
        result = self._agent(llm, runs).run("What is the capital of France?")

        assert runs == ["capital of France"]
        assert [call.reasoning for call in result.trace.tool_calls] == [""]
        assert _field_requests(llm, "reasoning") == []

    def test_caller_hint_stays_in_the_think_prompt(self):
        # Narrowed prompts must still list the caller's own context keys.
        runs: list[str] = []
        facts = dict(_REACT_FACTS)
        facts["tool_name"] = ("lookup", "prefer-lookup-7")
        llm = PromptGroundedLLM(facts=facts)
        self._agent(llm, runs).run(
            "What is the capital of France?",
            initial_context={"suggested_tool": "prefer-lookup-7"},
        )

        assert runs == ["capital of France"]

    def _hitl(self, decision: bool, asked: list[str]):
        from fsm_llm.agents import HumanInTheLoop

        def callback(request) -> bool:
            asked.append(request.tool_name)
            return decision

        return HumanInTheLoop(
            approval_policy=lambda call, ctx: call.tool_name == "danger",
            approval_callback=callback,
        )

    def _danger_registry(self, runs: list[str]):
        registry = _lookup_registry(runs)

        def danger(query: str) -> str:
            runs.append(f"danger:{query}")
            return "The capital of France is Paris."

        registry.register_function(danger, name="danger", description="Risky lookup")
        return registry

    def test_hitl_denial_reaches_the_next_think_turn_without_evidence(self):
        # LOOP-06: a denial wrote nothing the model could see, so it asked for
        # the same gated call again. It must not count as evidence either.
        from fsm_llm.agents import AgentConfig, ReactAgent

        def tool_name(text: str, context: dict) -> object:
            return "lookup" if "denied the call" in text else "danger"

        runs: list[str] = []
        asked: list[str] = []
        facts = {
            "tool_input": ({"query": "capital of France"}, "capital"),
            "should_terminate": (True, "is Paris"),
        }
        llm = _TurnAwareLLM({"tool_name": tool_name}, facts=facts)
        agent = ReactAgent(
            tools=self._danger_registry(runs),
            config=AgentConfig(max_iterations=6),
            hitl=self._hitl(False, asked),
            llm_interface=llm,
        )
        result = agent.run("What is the capital of France?")

        assert asked == ["danger"]
        assert runs == ["capital of France"]
        assert result.final_context[ContextKeys.OBSERVATION_COUNT] == 1
        assert not any("danger" in o for o in result.final_context["observations"])

    def test_approved_call_runs_before_an_early_terminate_concludes(self):
        # Step 5 gap: think set should_terminate with the gated call, and
        # await_approval -> conclude (p1) beat the approved -> act edge.
        from fsm_llm.agents import AgentConfig, ReactAgent

        runs: list[str] = []
        asked: list[str] = []
        facts: dict[str, tuple[object, str]] = {
            "tool_name": ("danger", "capital"),
            "tool_input": ({"query": "capital of France"}, "capital"),
            "should_terminate": (True, "capital"),
        }
        agent = ReactAgent(
            tools=self._danger_registry(runs),
            config=AgentConfig(max_iterations=6),
            hitl=self._hitl(True, asked),
            llm_interface=PromptGroundedLLM(facts=facts),
        )
        result = agent.run("What is the capital of France?")

        assert asked == ["danger"]
        assert runs == ["danger:capital of France"]
        assert (result.success, result.stop_reason) == (True, "answered")

    def test_budget_error_cites_the_loop_ceiling(self):
        # LOOP-17: the message named max_iterations, not the ceiling hit.
        from fsm_llm.agents import AgentConfig, ReactAgent
        from fsm_llm.agents.exceptions import BudgetExhaustedError

        agent = ReactAgent(
            tools=_lookup_registry([]),
            config=AgentConfig(max_iterations=2),
            llm_interface=PromptGroundedLLM(),
        )
        with pytest.raises(BudgetExhaustedError) as info:
            agent._check_budgets(time.monotonic(), 7)
        assert info.value.limit == 6
        assert "max_iterations 2 x FSM_BUDGET_MULTIPLIER 3" in str(info.value)

    def test_parallel_react_counts_think_turns_and_narrows_prompts(self):
        # Sibling: ParallelReact shares the limiter and the typed think fields.
        from fsm_llm.agents import AgentConfig
        from fsm_llm.agents.parallel_react import ParallelReactAgent

        runs: list[str] = []
        calls = [{"tool_name": "lookup", "tool_input": {"query": "capital"}}]
        llm = PromptGroundedLLM(facts={"tool_calls": (calls, "capital")})
        agent = ParallelReactAgent(
            tools=_lookup_registry(runs),
            config=AgentConfig(max_iterations=4),
            llm_interface=llm,
        )
        result = agent.run("What is the capital of France?")

        assert len(runs) == 3
        assert result.stop_reason == "max_iterations"
        assert not llm.calls("extract_bulk_data")
        for request in llm.calls("extract_field"):
            assert ContextKeys.AGENT_TRACE not in (request.context or {})

    def test_reflexion_think_prompt_keeps_episodic_memory_without_trace(self):
        from fsm_llm.agents import AgentConfig, ReflexionAgent

        runs: list[str] = []
        llm = PromptGroundedLLM(facts=_REACT_FACTS)
        agent = ReflexionAgent(
            tools=_lookup_registry(runs),
            config=AgentConfig(max_iterations=6),
            llm_interface=llm,
        )
        agent.run("What is the capital of France?")

        think = _field_requests(llm, "tool_name")
        assert runs and think
        for request in think:
            assert ContextKeys.EPISODIC_MEMORY in (request.context or {})
            assert ContextKeys.AGENT_TRACE not in (request.context or {})


def _episodes(context: dict) -> list | None:
    memory = context.get(ContextKeys.EPISODIC_MEMORY)
    return memory if isinstance(memory, list) else None


def _reflect_derived() -> dict[str, object]:
    """``reflection``/``lessons`` grounded on the episodic memory the reflect
    prompt shows, so each episode's value is its own."""

    def reflection(text: str, context: dict) -> object:
        memory = _episodes(context)
        return None if memory is None else f"reflection after {len(memory)} episodes"

    def lessons(text: str, context: dict) -> object:
        memory = _episodes(context)
        return None if memory is None else f"lesson {len(memory) + 1}"

    return {"reflection": reflection, "lessons": lessons}


# The lookup observation carries "is Paris": the self-evaluation fails on it,
# so the run reflects until max_reflections forces the stop.
_REFLEXION_FACTS: dict[str, tuple[object, str]] = {
    "tool_name": ("lookup", "capital"),
    "tool_input": ({"query": "capital of France"}, "capital"),
    "evaluation_passed": (False, "is Paris"),
    "evaluation_score": (0.2, "is Paris"),
    "evaluation_feedback": ("name a second source", "is Paris"),
}


class TestReflexionLoop:
    """Step 17 (REACT-01/02): each episode records its own grounded
    reflection, ``evaluation_fn`` runs even when extraction returns nothing,
    and evaluate/reflect are silent intermediate states."""

    def _agent(self, llm, runs, **kwargs):
        from fsm_llm.agents import AgentConfig, ReflexionAgent

        return ReflexionAgent(
            tools=_lookup_registry(runs),
            config=AgentConfig(max_iterations=10),
            max_reflections=2,
            llm_interface=llm,
            **kwargs,
        )

    def _run(self, **kwargs):
        runs: list[str] = []
        llm = _TurnAwareLLM(_reflect_derived(), facts=_REFLEXION_FACTS)
        result = self._agent(llm, runs, **kwargs).run("What is the capital of France?")
        return llm, runs, result

    def test_each_episode_records_its_own_grounded_reflection(self):
        # REACT-01: the bookkeeping ran on reflect ENTRY, before the reflect
        # extraction: episode 1 stored "" and later episodes lagged one behind
        # (the never-cleared reflection was extracted once).
        _, runs, result = self._run()

        memory = result.final_context[ContextKeys.EPISODIC_MEMORY]
        assert len(runs) == 2
        assert [m["reflection"] for m in memory] == [
            "reflection after 0 episodes",
            "reflection after 1 episodes",
        ]
        assert [m["lessons"] for m in memory] == [["lesson 1"], ["lesson 2"]]
        assert [m["outcome"] for m in memory] == ["name a second source"] * 2

    def test_think_prompt_sees_the_recorded_reflection(self):
        llm, _, _ = self._run()

        seen = [
            r
            for r in _field_requests(llm, "tool_name")
            if "reflection after 0 episodes"
            in json.dumps((r.context or {}).get(ContextKeys.EPISODIC_MEMORY))
        ]
        assert seen, "no think prompt carried episode 1's reflection"

    def test_evaluation_fn_runs_when_extraction_returns_nothing(self):
        # REACT-02: evaluation_fn was a CONTEXT_UPDATE handler, which core runs
        # only when extraction returned data; a null self-evaluation skipped it.
        from fsm_llm.agents.definitions import EvaluationResult

        calls: list[int] = []

        def evaluation_fn(context: dict) -> EvaluationResult:
            calls.append(len(context.get(ContextKeys.OBSERVATIONS) or []))
            passed = len(calls) >= 2
            return EvaluationResult(
                passed=passed,
                score=0.9 if passed else 0.1,
                feedback=f"check {len(calls)}",
            )

        runs: list[str] = []
        facts = {
            k: v for k, v in _REFLEXION_FACTS.items() if not k.startswith("evaluation")
        }
        llm = _TurnAwareLLM(_reflect_derived(), facts=facts)
        result = self._agent(llm, runs, evaluation_fn=evaluation_fn).run(
            "What is the capital of France?"
        )

        assert calls == [1, 2]
        assert len(runs) == 2
        memory = result.final_context[ContextKeys.EPISODIC_MEMORY]
        assert [m["outcome"] for m in memory] == ["check 1"]
        assert result.final_context[ContextKeys.EVALUATION_PASSED] is True
        assert (result.success, result.stop_reason) == (True, "answered")
        # The external verdict replaces the self-evaluation: no model call.
        assert not _field_requests(llm, "evaluation_passed")

    def test_evaluate_and_reflect_are_silent_typed_states(self):
        llm, _, _ = self._run()

        states = [
            m.group(1)
            for r in llm.calls("generate_response")
            if (m := _CURRENT_STATE_TAG.search(r.system_prompt))
        ]
        assert "evaluate" not in states and "reflect" not in states
        assert not llm.calls("extract_bulk_data")
        for request in llm.calls("extract_field"):
            assert ContextKeys.AGENT_TRACE not in (request.context or {})
        evaluate = _field_requests(llm, "evaluation_passed")
        assert evaluate and all(
            ContextKeys.OBSERVATIONS in (r.context or {}) for r in evaluate
        )


# The replan prompt's stash of the plan it revises (a literal, so the tests
# also run on the parent commit, which lacks the constant).
_PREVIOUS_PLAN = "previous_plan_steps"


def _step_description(context: dict) -> str:
    return str(context.get("current_step_description") or "")


def _plan_derived(first_plan: list[str], new_plan: list[str]) -> dict[str, object]:
    """PlanExecute fields grounded on what each prompt shows.

    ``plan_steps`` is ``first_plan`` when the prompt shows the task and no
    earlier plan, ``new_plan`` only when it shows the earlier plan and a failed
    step result. The tool selection follows the current step description.
    """

    def plan_steps(text: str, context: dict) -> object:
        if _PREVIOUS_PLAN in context:
            failed = "[TOOL FAILED]" in json.dumps(
                context.get(ContextKeys.STEP_RESULTS)
            )
            return list(new_plan) if failed else None
        return list(first_plan) if "capital" in json.dumps(context) else None

    def tool_name(text: str, context: dict) -> object:
        return "lookup" if "source" in _step_description(context) else None

    def tool_input(text: str, context: dict) -> object:
        desc = _step_description(context)
        if "source" not in desc:
            return None
        return {"query": "bad" if "bad source" in desc else "good"}

    return {"plan_steps": plan_steps, "tool_name": tool_name, "tool_input": tool_input}


def _source_registry(runs: list[str], *, always_fail: bool = False) -> object:
    from fsm_llm.agents import ToolRegistry

    registry = ToolRegistry()

    def lookup(query: str) -> str:
        runs.append(query)
        if always_fail or query == "bad":
            raise RuntimeError("source offline")
        return "The capital of France is Paris."

    registry.register_function(lookup, name="lookup", description="Look up a fact")
    return registry


def _replan_requests(llm: PromptGroundedLLM) -> list:
    return [
        r
        for r in _field_requests(llm, ContextKeys.PLAN_STEPS)
        if _PREVIOUS_PLAN in (r.context or {})
    ]


class TestPlanExecuteLoop:
    """Step 18 (PAT-01/02): a failed tool step routes to ``replan``, whose new
    typed plan replaces the old one; ``max_replans`` allows exactly N replans;
    a non-list plan never iterates per character; the intermediate states are
    silent and typed."""

    def _agent(self, llm, registry, **kwargs):
        from fsm_llm.agents import AgentConfig, PlanExecuteAgent

        return PlanExecuteAgent(
            tools=registry,
            config=AgentConfig(max_iterations=20),
            llm_interface=llm,
            **kwargs,
        )

    def test_failed_step_replans_and_the_new_plan_replaces_the_old(self):
        # PAT-01: the checker reset step_failed on check_result entry and the
        # seeded False was never re-extracted, so replan was unreachable.
        runs: list[str] = []
        llm = _TurnAwareLLM(_plan_derived(["use bad source"], ["use good source"]))
        result = self._agent(llm, _source_registry(runs)).run(
            "What is the capital of France?"
        )

        assert runs == ["bad", "good"]
        assert len(_replan_requests(llm)) == 1
        final = result.final_context
        assert final[ContextKeys.PLAN_STEPS] == ["use good source"]
        assert _PREVIOUS_PLAN not in final
        entries = final[ContextKeys.STEP_RESULTS]
        # Each entry records the tool observation, not the pre-tool guess.
        assert [e["success"] for e in entries] == [False, True]
        assert "[TOOL FAILED]" in entries[0]["result"]
        assert "is Paris" in entries[1]["result"]
        assert result.success is True

    @pytest.mark.parametrize("max_replans", [1, 2])
    def test_max_replans_allows_exactly_n_replans(self, max_replans):
        # PAT-01: `replan_count >= max_replans` on replan entry gave N - 1.
        runs: list[str] = []
        llm = _TurnAwareLLM(_plan_derived(["use bad source"], ["use bad source again"]))
        result = self._agent(
            llm, _source_registry(runs, always_fail=True), max_replans=max_replans
        ).run("What is the capital of France?")

        assert len(_replan_requests(llm)) == max_replans
        assert len(runs) == max_replans + 1
        assert result.success is False

    def test_string_plan_never_iterates_per_character(self):
        # PAT-02: an `any` plan_steps took a string and check_result counted
        # its characters as steps.
        runs: list[str] = []
        llm = PromptGroundedLLM(
            facts={"plan_steps": ("Look up the capital of France", "capital")}
        )
        result = self._agent(llm, _source_registry(runs)).run(
            "What is the capital of France?"
        )

        assert result.final_context.get(ContextKeys.CURRENT_STEP_INDEX, 0) <= 1
        assert len(result.final_context.get(ContextKeys.STEP_RESULTS) or []) <= 1

    def test_intermediate_states_are_silent_and_typed(self):
        runs: list[str] = []
        llm = _TurnAwareLLM(_plan_derived(["use bad source"], ["use good source"]))
        self._agent(llm, _source_registry(runs)).run("What is the capital of France?")

        states = {
            m.group(1)
            for r in llm.calls("generate_response")
            if (m := _CURRENT_STATE_TAG.search(r.system_prompt))
        }
        assert states <= {"synthesize"}
        for request in llm.calls("extract_field"):
            assert ContextKeys.AGENT_TRACE not in (request.context or {})
        step_fields = _field_requests(llm, "tool_name")
        assert step_fields and all(
            "current_step_description" in (r.context or {}) for r in step_fields
        )
        assert _field_requests(llm, "step_result")
        for request in llm.calls("extract_bulk_data"):
            assert "plan_steps" not in request.system_prompt
            assert "step_failed" not in request.system_prompt


_P1 = "P1: remote work raises output"
_C1 = "C1: it ignores cohesion"
_P2 = "P2: remote work raises output with weekly on-site days"
_C2 = "C2: on-site days cost commute time"
_DEBATE_TASK = "Does remote work raise engineering output?"
_CONCLUDE_TEXT = "Final: yes, with weekly on-site days."


def _debate_derived(consensus: bool = False) -> dict:
    """Round values computed from the prompt: round 2 builds on round 1."""

    def proposition(text, _ctx):
        if _C1 in text:  # round 1's critique, via debate_rounds
            return _P2
        return _P1 if _DEBATE_TASK in text else None

    def critique(_text, ctx):
        return {_P1: _C1, _P2: _C2}.get(ctx.get(ContextKeys.PROPOSITION))

    def counter(_text, ctx):
        crit = ctx.get(ContextKeys.CRITIQUE)
        return f"K answers {crit}" if crit else None

    def verdict(_text, ctx):
        prop = ctx.get(ContextKeys.PROPOSITION)
        return f"V on {prop}" if prop else None

    return {
        ContextKeys.PROPOSITION: proposition,
        ContextKeys.CRITIQUE: critique,
        ContextKeys.COUNTER_ARGUMENT: counter,
        ContextKeys.JUDGE_VERDICT: verdict,
        ContextKeys.CONSENSUS_REACHED: lambda _t, _c: consensus,
    }


class TestDebateLoop:
    """Step 19 (PAT-03): each round extracts fresh grounded values, the answer
    is the conclude reply, and only conclude speaks."""

    def _run(self, num_rounds=2, **derived_kw):
        llm = _TurnAwareLLM(
            _debate_derived(**derived_kw),
            responses={"conclude": _CONCLUDE_TEXT},
        )
        result = DebateAgent(num_rounds=num_rounds, llm_interface=llm).run(_DEBATE_TASK)
        return llm, result

    def test_judge_consensus_prompt_shows_prior_rounds(self):
        # Fix 21.2: the 21.1 narrowing dropped debate_rounds from the judge's
        # consensus prompt; live, the judge then declined consensus every
        # round. Round 2's judge must see round 1, never agent_trace.
        llm, _ = self._run()

        requests = _field_requests(llm, ContextKeys.CONSENSUS_REACHED)
        assert len(requests) == 2
        first, second = requests
        assert not first.context.get(ContextKeys.DEBATE_ROUNDS)
        prior = second.context[ContextKeys.DEBATE_ROUNDS]
        assert [r["proposition"] for r in prior] == [_P1]
        assert _C1 in second.system_prompt
        assert ContextKeys.AGENT_TRACE not in second.context

    def test_round_two_builds_on_round_one(self):
        # Skip-if-set froze every round value after round 1, and the bulk
        # prompt showed none of them.
        _, result = self._run()

        rounds = result.final_context[ContextKeys.DEBATE_ROUNDS]
        assert [r["proposition"] for r in rounds] == [_P1, _P2]
        assert [r["critique"] for r in rounds] == [_C1, _C2]
        assert rounds[1]["counter_argument"] == f"K answers {_C2}"
        assert rounds[1]["judge_verdict"] == f"V on {_P2}"

    def test_critique_prompt_shows_this_rounds_proposition(self):
        llm, _ = self._run()

        critiques = _field_requests(llm, ContextKeys.CRITIQUE)
        assert [r.context.get(ContextKeys.PROPOSITION) for r in critiques] == [
            _P1,
            _P2,
        ]

    def test_answer_is_the_conclude_reply(self):
        # PAT-03: the answer was the judge's frozen round-1 verdict.
        _, result = self._run()

        assert result.answer == _CONCLUDE_TEXT
        # The judge never agreed: num_rounds forced the consensus (D-051).
        assert (result.success, result.stop_reason) == (False, "forced_pass")

    def test_consensus_concludes_after_one_round(self):
        _, result = self._run(num_rounds=3, consensus=True)

        assert len(result.final_context[ContextKeys.DEBATE_ROUNDS]) == 1
        assert result.answer == _CONCLUDE_TEXT

    def test_no_proposition_is_no_result(self):
        llm = PromptGroundedLLM(responses={"conclude": _CONCLUDE_TEXT})
        result = DebateAgent(num_rounds=1, llm_interface=llm).run(_DEBATE_TASK)

        assert result.answer == _CONCLUDE_TEXT
        assert result.success is False
        # Nothing was ever extracted, so the judge handler (CONTEXT_UPDATE)
        # never ran and the limiter forced the consensus: a forced stop
        # outranks no_result (D-051 of plan 06a5ec0a).
        assert result.stop_reason == "max_iterations"

    def test_only_conclude_speaks_and_no_bulk_call_runs(self):
        llm, _ = self._run()

        states = {
            m.group(1)
            for r in llm.calls("generate_response")
            if (m := _CURRENT_STATE_TAG.search(r.system_prompt))
        }
        assert states == {"conclude"}
        assert not llm.calls("extract_bulk_data")
        for name in (
            ContextKeys.PROPOSITION,
            ContextKeys.CRITIQUE,
            ContextKeys.COUNTER_ARGUMENT,
            ContextKeys.JUDGE_VERDICT,
        ):
            for request in _field_requests(llm, name):
                assert ContextKeys.AGENT_TRACE not in (request.context or {})


class _SequenceLLM(PromptGroundedLLM):
    """Replies with ``replies`` in call order (one per sample)."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__()
        self.replies = list(replies)

    def generate_response(self, request: ResponseGenerationRequest):
        super().generate_response(request)
        from fsm_llm.definitions import ResponseGenerationResponse

        return ResponseGenerationResponse(
            message=self.replies.pop(0), message_type="response", reasoning="seq"
        )


class TestSelfConsistencyVote:
    """Step 19 (PAT-05): the vote counts the final ``Answer:`` line, and
    ``confidence`` is the winning share."""

    REPLIES = (
        "Adding them gives 41.\nAnswer: 41",
        "Six times seven is 42.\nAnswer: 42",
        "The product is forty-two.\n**Answer:** 42.",
        "Computing 6*7 we get the value below.\nanswer:  42",
    )

    def _run(self, replies):
        from fsm_llm.agents import AgentConfig, SelfConsistencyAgent

        agent = SelfConsistencyAgent(
            config=AgentConfig(model="mock/model"),
            num_samples=len(replies),
            llm_interface=_SequenceLLM(replies),
        )
        return agent.run("What is 6 times 7?")

    def test_agreeing_answers_in_different_prose_win(self):
        result = self._run(self.REPLIES)

        assert result.answer == self.REPLIES[1]
        assert result.final_context[ContextKeys.CONFIDENCE] == 0.75
        assert result.success is True
        assert result.stop_reason == "answered"

    def test_text_without_answer_line_votes_casefolded(self):
        from fsm_llm.agents.self_consistency import _majority_vote

        assert _majority_vote(["Paris", " paris ", "London"]) == "Paris"

    def test_sample_prompt_asks_for_an_answer_line(self):
        llm = _SequenceLLM(["Answer: 42"])
        from fsm_llm.agents import AgentConfig, SelfConsistencyAgent

        SelfConsistencyAgent(
            config=AgentConfig(model="mock/model"), num_samples=1, llm_interface=llm
        ).run("What is 6 times 7?")

        (request,) = llm.calls("generate_response")
        assert "Answer:" in request.system_prompt
        assert not llm.calls("extract_bulk_data")


# ---------------------------------------------------------------------------
# Step 20: PromptChain, MakerChecker, EvaluatorOptimizer (PAT-06/PAT-11)
# ---------------------------------------------------------------------------

_CHAIN_TASK = "Write a note about tides"


def _spoken_states(llm: PromptGroundedLLM) -> set[str]:
    return {
        m.group(1)
        for r in llm.calls("generate_response")
        if (m := _CURRENT_STATE_TAG.search(r.system_prompt))
    }


def _stage(k: int) -> str:
    return f"Stage {k} output text"


def _chain_derived() -> dict[str, object]:
    """Step ``k`` outputs ``_stage(k)``, grounded on the task and on the ``k``
    earlier outputs its prompt shows in ``chain_results``."""

    def step_result(text: str, ctx: dict) -> object:
        results = ctx.get(ContextKeys.CHAIN_RESULTS)
        if _CHAIN_TASK not in text or not isinstance(results, list):
            return None
        return _stage(len(results))

    return {ContextKeys.CHAIN_STEP_RESULT: step_result}


class TestPromptChainLoop:
    """Step 20 (PAT-06): each step extracts a fresh grounded result that is
    appended to ``chain_results``, and a failed gate stops the chain."""

    def _run(self, gates: dict[int, object] | None = None):
        from fsm_llm.agents import ChainStep, PromptChainAgent

        gates = gates or {}
        chain = [
            ChainStep(
                step_id=f"s{i}",
                name=f"Stage {i}",
                extraction_instructions=f"Extract stage {i} text.",
                response_instructions=f"Present stage {i}.",
                validation_fn=gates.get(i),
            )
            for i in range(3)
        ]
        llm = _TurnAwareLLM(_chain_derived(), responses={"output": "Final note."})
        result = PromptChainAgent(chain=chain, llm_interface=llm).run(_CHAIN_TASK)
        return llm, result

    def test_chain_results_hold_each_steps_grounded_output(self):
        # PAT-06: the step result came only from the user's context-free
        # bulk instructions and froze after step 0, so chain_results was [].
        _, result = self._run()

        assert result.final_context[ContextKeys.CHAIN_RESULTS] == [
            _stage(0),
            _stage(1),
            _stage(2),
        ]
        assert result.answer == _stage(2)
        assert (result.success, result.stop_reason) == (True, "answered")

    def test_step_prompts_show_task_and_earlier_results_not_trace(self):
        llm, _ = self._run()

        requests = _field_requests(llm, ContextKeys.CHAIN_STEP_RESULT)
        assert [len(r.context[ContextKeys.CHAIN_RESULTS]) for r in requests] == [
            0,
            1,
            2,
        ]
        for request in requests:
            assert ContextKeys.AGENT_TRACE not in request.context
        assert not llm.calls("extract_bulk_data")

    def test_failed_gate_stops_the_chain(self):
        # PAT-06: gate_passed/should_terminate were written but every step
        # edge was unconditional, so the chain ran every step anyway.
        llm, result = self._run(gates={0: lambda ctx: False})

        assert result.final_context[ContextKeys.CHAIN_RESULTS] == [_stage(0)]
        assert len(_field_requests(llm, ContextKeys.CHAIN_STEP_RESULT)) == 1
        assert result.success is False
        assert result.stop_reason == "gate_failed"
        assert result.final_context[ContextKeys.GATE_PASSED] is False

    def test_passing_gate_sees_the_step_result(self):
        seen: list[object] = []

        def gate(ctx: dict) -> bool:
            seen.append(ctx.get(ContextKeys.CHAIN_STEP_RESULT))
            return True

        _, result = self._run(gates={1: gate})

        assert seen == [_stage(1)]
        assert result.final_context[ContextKeys.CHAIN_RESULTS] == [
            _stage(0),
            _stage(1),
            _stage(2),
        ]
        assert result.stop_reason == "answered"


_MC_TASK = "Write a haiku about rain"


def _maker_checker_derived() -> dict[str, object]:
    """The maker writes ``DRAFT-1`` from the task and ``DRAFT-2`` only when its
    prompt shows the checker's feedback on ``DRAFT-1``; the checker judges the
    draft its prompt shows."""

    def draft(text: str, ctx: dict) -> object:
        if "FIX-SYLLABLES" in text:
            return "DRAFT-2"
        return "DRAFT-1" if _MC_TASK in text else None

    def feedback(_text: str, ctx: dict) -> object:
        return {"DRAFT-1": "FIX-SYLLABLES", "DRAFT-2": "fine"}.get(
            ctx.get(ContextKeys.DRAFT_OUTPUT)
        )

    def score(_text: str, ctx: dict) -> object:
        return {"DRAFT-1": 0.2, "DRAFT-2": 0.9}.get(ctx.get(ContextKeys.DRAFT_OUTPUT))

    def passed(_text: str, ctx: dict) -> object:
        return {"DRAFT-1": False, "DRAFT-2": True}.get(
            ctx.get(ContextKeys.DRAFT_OUTPUT)
        )

    return {
        ContextKeys.DRAFT_OUTPUT: draft,
        ContextKeys.CHECKER_FEEDBACK: feedback,
        "quality_score": score,
        ContextKeys.CHECKER_PASSED: passed,
    }


class TestMakerCheckerLoop:
    """Step 20 (PAT-11): the maker and reviser write grounded typed drafts, the
    reviser sees the checker's feedback, and only ``output`` speaks."""

    def _run(self):
        from fsm_llm.agents import AgentConfig, MakerCheckerAgent

        llm = _TurnAwareLLM(_maker_checker_derived())
        agent = MakerCheckerAgent(
            maker_instructions="Write a haiku.",
            checker_instructions="Check the syllables.",
            max_revisions=3,
            config=AgentConfig(max_iterations=10),
            llm_interface=llm,
        )
        return llm, agent.run(_MC_TASK)

    def test_reviser_prompt_contains_the_checker_feedback(self):
        # PAT-11: make/revise drafts came from the context-free bulk call.
        llm, result = self._run()

        drafts = _field_requests(llm, ContextKeys.DRAFT_OUTPUT)
        assert len(drafts) == 2
        revise = drafts[1].context
        assert revise[ContextKeys.CHECKER_FEEDBACK] == "FIX-SYLLABLES"
        assert revise[ContextKeys.PREVIOUS_DRAFT] == "DRAFT-1"
        assert result.answer == "DRAFT-2"
        assert (result.success, result.stop_reason) == (True, "answered")

    def test_checker_fields_are_typed_and_judge_each_draft(self):
        llm, result = self._run()

        judged = [
            r.context.get(ContextKeys.DRAFT_OUTPUT)
            for r in _field_requests(llm, "quality_score")
        ]
        assert judged == ["DRAFT-1", "DRAFT-2"]
        assert result.final_context["quality_score"] == 0.9
        assert result.final_context[ContextKeys.REVISION_COUNT] == 2

    def test_only_output_speaks_and_no_bulk_call_runs(self):
        llm, _ = self._run()

        assert _spoken_states(llm) == {"output"}
        assert not llm.calls("extract_bulk_data")
        for request in llm.calls("extract_field"):
            assert ContextKeys.AGENT_TRACE not in (request.context or {})


class TestEvaluatorOptimizerLoop:
    """Step 20 (PAT-11): generate/refine extract a typed ``generated_output``
    whose refine prompt shows the evaluator's feedback, and only ``output``
    speaks."""

    def _run(self):
        from fsm_llm.agents import AgentConfig, EvaluatorOptimizerAgent
        from fsm_llm.agents.definitions import EvaluationResult

        def output(text: str, _ctx: dict) -> object:
            if "ADD-IMAGERY" in text:
                return "HAIKU-2"
            return "HAIKU-1" if "haiku" in text else None

        def evaluate(out: str, _ctx: dict) -> EvaluationResult:
            ok = out == "HAIKU-2"
            return EvaluationResult(
                passed=ok,
                score=1.0 if ok else 0.2,
                feedback="ok" if ok else "ADD-IMAGERY",
            )

        llm = _TurnAwareLLM({ContextKeys.GENERATED_OUTPUT: output})
        agent = EvaluatorOptimizerAgent(
            evaluation_fn=evaluate,
            max_refinements=3,
            config=AgentConfig(max_iterations=10),
            llm_interface=llm,
        )
        return llm, agent.run("Write a haiku about rain")

    def test_refine_sees_feedback_and_ships_the_passing_output(self):
        llm, result = self._run()

        outputs = _field_requests(llm, ContextKeys.GENERATED_OUTPUT)
        assert len(outputs) == 2
        refine = outputs[1].context
        assert refine[ContextKeys.REFINEMENT_FEEDBACK] == "ADD-IMAGERY"
        assert refine[ContextKeys.PREVIOUS_OUTPUT] == "HAIKU-1"
        assert result.answer == "HAIKU-2"
        assert (result.success, result.stop_reason) == (True, "answered")

    def test_only_output_speaks_and_prompts_exclude_trace(self):
        # generate/refine spoke every turn (an unread Pass-2 call each) and
        # their auto-minted `any` configs dumped agent_trace into the prompt.
        llm, _ = self._run()

        assert _spoken_states(llm) == {"output"}
        assert not llm.calls("extract_bulk_data")
        for request in _field_requests(llm, ContextKeys.GENERATED_OUTPUT):
            assert ContextKeys.AGENT_TRACE not in request.context


class TestCutOffArtifactEnvelopeNeverShips:
    """Fix 20.1 (D-050): a real ``LiteLLMInterface`` whose ``generated_output``
    reply is an extraction envelope cut off by ``max_tokens`` must not ship
    the envelope as the EvalOpt answer (live eval_opt_structured did)."""

    _CUT = (
        '{"field_name": "generated_output", "value": "{\\"name\\": '
        '\\"Carbonara\\", \\"steps\\": [\\"Boil the wat'
    )

    def _run(self):
        from fsm_llm.agents import AgentConfig, EvaluatorOptimizerAgent
        from fsm_llm.agents.definitions import EvaluationResult
        from fsm_llm.llm import LiteLLMInterface

        seen: list[str] = []

        def reply(messages, call_type, response_format=None):
            if call_type == "field_extraction":
                content = self._CUT
            elif call_type == "data_extraction":
                content = '{"extracted_data": {}, "confidence": 1.0}'
            else:
                content = '{"message": "Done."}'
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

        llm = LiteLLMInterface(model="test", api_key="test")
        llm._make_llm_call = reply  # type: ignore[method-assign]

        def evaluate(out: str, _ctx: dict) -> EvaluationResult:
            seen.append(out)
            return EvaluationResult(passed=True, score=1.0, feedback="ok")

        agent = EvaluatorOptimizerAgent(
            evaluation_fn=evaluate,
            config=AgentConfig(max_iterations=6),
            llm_interface=llm,
        )
        return seen, agent.run("Write a carbonara recipe as JSON")

    def test_answer_is_the_salvaged_value_not_the_envelope(self):
        seen, result = self._run()

        assert '"field_name"' not in result.answer
        assert result.answer == '{"name": "Carbonara", "steps": ["Boil the wat'
        assert seen and all('"field_name"' not in out for out in seen)

    def test_native_json_artifact_ships_as_parseable_json_text(self):
        # An `any` artifact the model returned as a dict/list reaches the
        # evaluator and the answer as JSON text, so output_schema parses it.
        from pydantic import BaseModel

        from fsm_llm.agents import AgentConfig, EvaluatorOptimizerAgent
        from fsm_llm.agents.base import artifact_text
        from fsm_llm.agents.definitions import EvaluationResult

        class Recipe(BaseModel):
            name: str
            steps: list[str]

        recipe = {"name": "Carbonara", "steps": ["Boil", "Toss"]}
        agent = EvaluatorOptimizerAgent(
            evaluation_fn=lambda out, _ctx: EvaluationResult(passed=True),
            config=AgentConfig(output_schema=Recipe),
            llm_interface=PromptGroundedLLM(),
        )
        answer = agent._extract_answer({ContextKeys.GENERATED_OUTPUT: recipe}, [])

        assert json.loads(answer) == recipe
        assert agent._try_parse_structured_output(answer) == Recipe(**recipe)
        assert artifact_text(["a"]) == '[\n  "a"\n]'
        assert (artifact_text(None), artifact_text("x"), artifact_text(2)) == (
            "",
            "x",
            "2",
        )


_REWOO_TASK = "What is the capital of France?"


def _rewoo_registry(calls: list[str], *, fail: bool = False) -> object:
    from fsm_llm.agents import ToolRegistry

    registry = ToolRegistry()

    def lookup(query: str) -> str:
        calls.append(query)
        if fail:
            raise RuntimeError("source offline")
        return f"found[{query}]"

    registry.register_function(lookup, name="lookup", description="Look up a fact")
    return registry


def _rewoo_run(plan: list, *, fail: bool = False) -> tuple:
    from fsm_llm.agents import REWOOAgent

    calls: list[str] = []
    llm = PromptGroundedLLM(
        facts={
            "plan_blueprint": (plan, "capital of France"),
            "final_answer": ("Paris", "capital of France"),
        }
    )
    agent = REWOOAgent(tools=_rewoo_registry(calls, fail=fail), llm_interface=llm)
    return agent.run(_REWOO_TASK), calls, llm


def _lookup_step(plan_id: object, query: str) -> dict:
    step = {"description": "look up", "tool_name": "lookup"}
    step["tool_input"] = {"query": query}
    if plan_id is not None:
        step["plan_id"] = plan_id
    return step


class TestREWOOOutcome:
    """Step 21 (PAT-09): success needs one successful evidence entry, plan ids
    normalise to ``E<n>``, and only ``solve`` speaks."""

    def test_all_tools_failing_is_no_result(self):
        result, calls, _ = _rewoo_run([_lookup_step(1, "capital")], fail=True)

        assert calls == ["capital"]
        assert result.success is False
        assert result.stop_reason == "no_result"
        assert result.final_context[ContextKeys.EVIDENCE_STATUS] == [
            {"id": "E1", "tool_name": "lookup", "success": False}
        ]

    def test_one_successful_tool_is_evidence(self):
        result, _, _ = _rewoo_run([_lookup_step(1, "capital")])

        assert (result.success, result.stop_reason) == (True, "evidence")

    @pytest.mark.parametrize("plan_id", ["E1", "#E1", "e1", "1", 1.0])
    def test_string_plan_id_resolves_its_reference(self, plan_id):
        plan = [_lookup_step(plan_id, "capital"), _lookup_step("E2", "verify #E1")]
        result, calls, _ = _rewoo_run(plan)

        assert calls == ["capital", "verify found[capital]"]
        assert list(result.final_context[ContextKeys.EVIDENCE]) == ["E1", "E2"]

    def test_missing_plan_id_is_the_step_position(self):
        plan = [_lookup_step(None, "capital"), _lookup_step(None, "verify #E1")]
        result, calls, _ = _rewoo_run(plan)

        assert calls == ["capital", "verify found[capital]"]
        assert list(result.final_context[ContextKeys.EVIDENCE]) == ["E1", "E2"]

    def test_only_solve_speaks(self):
        _, _, llm = _rewoo_run([_lookup_step(1, "capital")])

        assert _spoken_states(llm) <= {"solve"}


class TestOrchestratorWorkers:
    """Step 21 (PAT-10): excess subtasks are recorded as skipped, and a
    worker's budget or timeout error ends the run."""

    def _agent(self, worker, max_workers: int = 5):
        from fsm_llm.agents import AgentConfig, OrchestratorAgent

        llm = PromptGroundedLLM(
            facts={
                "subtasks": (["part one", "part two"], "Plan the trip"),
                "final_answer": ("A plan", "Plan the trip"),
            }
        )
        return OrchestratorAgent(
            worker_factory=worker,
            config=AgentConfig(max_iterations=8),
            max_workers=max_workers,
            llm_interface=llm,
        )

    @pytest.mark.parametrize(
        "error_name", ["BudgetExhaustedError", "AgentTimeoutError"]
    )
    def test_worker_budget_error_propagates(self, error_name):
        from fsm_llm.agents import exceptions

        error_cls = getattr(exceptions, error_name)
        calls: list[str] = []

        def worker(subtask: str):
            calls.append(subtask)
            if error_name == "BudgetExhaustedError":
                raise error_cls(budget_type="iterations", limit=1)
            raise error_cls(timeout_seconds=1.0)

        with pytest.raises(error_cls):
            self._agent(worker).run("Plan the trip")
        assert calls == ["part one"]

    def test_subtasks_over_max_workers_are_skipped_with_a_warning(self):
        from fsm_llm.agents import AgentResult

        calls: list[str] = []

        def worker(subtask: str) -> AgentResult:
            calls.append(subtask)
            return AgentResult(answer=f"done {subtask}", success=True)

        warnings: list[str] = []
        from fsm_llm.logging import logger

        sink = logger.add(lambda m: warnings.append(str(m)), level="WARNING")
        logger.enable("fsm_llm")
        try:
            result = self._agent(worker, max_workers=1).run("Plan the trip")
        finally:
            logger.remove(sink)
            logger.disable("fsm_llm")

        assert calls == ["part one"]
        # D-049: skipped subtasks are reported beside, not inside,
        # worker_results.
        entries = result.final_context[ContextKeys.WORKER_RESULTS]
        assert [e["subtask"] for e in entries] == ["part one"]
        assert result.final_context[ContextKeys.SKIPPED_SUBTASKS] == ["part two"]
        assert any("skipping 1" in w for w in warnings)
        assert result.success is True

    def test_collect_prompt_never_shows_skipped_subtasks(self):
        # D-049: the collect judge read skipped entries in worker_results as
        # unfinished work and re-delegated (live 23 -> 66 calls).
        from fsm_llm.agents import AgentResult

        def worker(subtask: str) -> AgentResult:
            return AgentResult(answer=f"done {subtask}", success=True)

        agent = self._agent(worker, max_workers=1)
        agent.run("Plan the trip")
        llm = agent._api_kwargs["llm_interface"]

        collects = _field_requests(llm, ContextKeys.ALL_COLLECTED)
        assert collects
        for request in collects:
            text = request.system_prompt + json.dumps(request.context, default=str)
            assert "part two" not in text
            assert "skipped" not in text.lower()
            assert request.context[ContextKeys.WORKER_RESULTS] == [
                {"subtask": "part one", "answer": "done part one", "success": True}
            ]


def _no_trace_run_requests(pattern: str) -> tuple[PromptGroundedLLM, dict]:
    """Run ``pattern`` through the real API on the grounded fake with a
    caller hint; returns the fake and the caller context used."""
    from fsm_llm.agents import (
        ADaPTAgent,
        AgentConfig,
        AgentResult,
        DebateAgent,
        OrchestratorAgent,
        REWOOAgent,
    )

    hint = {"audience": "hint-for-kids"}
    if pattern == "rewoo":
        llm = PromptGroundedLLM(
            facts={
                "plan_blueprint": ([_lookup_step(1, "capital")], "capital of France"),
                "final_answer": ("Paris", "capital of France"),
            }
        )
        REWOOAgent(tools=_rewoo_registry([]), llm_interface=llm).run(
            _REWOO_TASK, initial_context=hint
        )
    elif pattern == "orchestrator":
        llm = PromptGroundedLLM(
            facts={
                "subtasks": (["part one"], "Plan the trip"),
                "all_collected": (True, "done part one"),
            }
        )
        OrchestratorAgent(
            worker_factory=lambda t: AgentResult(answer=f"done {t}", success=True),
            config=AgentConfig(max_iterations=8),
            llm_interface=llm,
        ).run("Plan the trip", initial_context=hint)
    elif pattern.startswith("adapt"):
        decompose = pattern == "adapt_decompose"
        llm = PromptGroundedLLM(
            facts={
                "attempt_result": ("A direct answer to the plan", "Plan the trip"),
                "attempt_succeeded": (not decompose, "A direct answer"),
                "subtasks": (["pick dates"], "A direct answer"),
            }
        )
        ADaPTAgent(
            config=AgentConfig(max_iterations=10), max_depth=1, llm_interface=llm
        ).run("Plan the trip", initial_context=hint)
    else:
        llm = _TurnAwareLLM(_debate_derived(), responses={"conclude": _CONCLUDE_TEXT})
        DebateAgent(num_rounds=2, llm_interface=llm).run(_DEBATE_TASK)
    return llm, hint


_NARROWED_FIELDS: dict[str, list[str]] = {
    "orchestrator": [ContextKeys.SUBTASKS, ContextKeys.ALL_COLLECTED],
    "adapt": [ContextKeys.ATTEMPT_RESULT, ContextKeys.ATTEMPT_SUCCEEDED],
    "adapt_decompose": [ContextKeys.SUBTASKS],
    "rewoo": [ContextKeys.PLAN_BLUEPRINT],
    "debate": [ContextKeys.CONSENSUS_REACHED],
}


class TestNarrowedPlannerFields:
    """Fix 21.1 (review loops #7): the fields core used to auto-mint with the
    whole context (``agent_trace`` included) are explicit typed configs whose
    prompts show the task, the values they judge and the caller's keys."""

    @pytest.mark.parametrize("pattern", list(_NARROWED_FIELDS))
    def test_field_prompts_exclude_agent_trace(self, pattern):
        llm, hint = _no_trace_run_requests(pattern)

        for name in _NARROWED_FIELDS[pattern]:
            requests = _field_requests(llm, name)
            assert requests, name
            for request in requests:
                assert ContextKeys.AGENT_TRACE not in (request.context or {}), name
                assert ContextKeys.TASK in request.context, name
                if pattern != "debate":
                    assert request.context.get("audience") == hint["audience"]

    @pytest.mark.parametrize(
        ("builder", "state", "field", "field_type"),
        [
            ("orchestrator", "orchestrate", ContextKeys.SUBTASKS, "any"),
            ("orchestrator", "collect", ContextKeys.ALL_COLLECTED, "bool"),
            ("adapt", "attempt", ContextKeys.ATTEMPT_RESULT, "str"),
            ("adapt", "assess", ContextKeys.ATTEMPT_SUCCEEDED, "bool"),
            ("adapt", "decompose", ContextKeys.SUBTASKS, "list"),
            ("rewoo", "plan_all", ContextKeys.PLAN_BLUEPRINT, "list"),
            ("debate", "judge", ContextKeys.CONSENSUS_REACHED, "bool"),
        ],
    )
    def test_builders_declare_typed_narrowed_configs(
        self, builder, state, field, field_type
    ):
        from fsm_llm.agents import fsm_definitions as defs

        fsm = {
            "orchestrator": lambda: defs.build_orchestrator_fsm("t"),
            "adapt": lambda: defs.build_adapt_fsm(None, "t"),
            "rewoo": lambda: defs.build_rewoo_fsm(_rewoo_registry([]), "t"),
            "debate": lambda: defs.build_debate_fsm("t"),
        }[builder]()
        configs = {
            c["field_name"]: c for c in fsm["states"][state]["field_extractions"]
        }

        assert configs[field]["field_type"] == field_type
        assert ContextKeys.AGENT_TRACE not in configs[field]["context_keys"]
        assert ContextKeys.SKIPPED_SUBTASKS not in configs[field]["context_keys"]


def _field_instructions(fsm: dict, state: str) -> dict[str, str]:
    return {
        f["field_name"]: f["extraction_instructions"]
        for f in fsm["states"][state].get("field_extractions", [])
    }


class TestGeneratedFieldsAreComposed:
    """Step 21 sweep: live qwen3.5:4b returned null for the Reflexion
    reflection/lessons and PlanExecute step_result ("not in the user
    message"); every generated text field opens with the compose sentence,
    the tool selection fields do not."""

    _SENTENCE = "compose it yourself now and never return null"

    def test_reflexion_text_fields(self):
        from fsm_llm.agents.fsm_definitions import build_reflexion_fsm

        fsm = build_reflexion_fsm(_rewoo_registry([]))
        reflect = _field_instructions(fsm, "reflect")
        evaluate = _field_instructions(fsm, "evaluate")

        for text in (
            reflect[ContextKeys.REFLECTION],
            reflect[ContextKeys.LESSONS],
            evaluate[ContextKeys.EVALUATION_FEEDBACK],
        ):
            assert self._SENTENCE in text
        assert self._SENTENCE not in evaluate[ContextKeys.EVALUATION_PASSED]

    @pytest.mark.parametrize("with_tools", [True, False])
    def test_plan_execute_step_result(self, with_tools):
        from fsm_llm.agents.fsm_definitions import build_plan_execute_fsm

        registry = _rewoo_registry([]) if with_tools else None
        fields = _field_instructions(build_plan_execute_fsm(registry), "execute_step")

        assert self._SENTENCE in fields[ContextKeys.STEP_RESULT]
        for name in ("tool_name", "tool_input"):
            if name in fields:
                assert self._SENTENCE not in fields[name]


class _RecordingNode:
    """A graph or swarm member that records each call and echoes its context.

    Real agents return their initial context inside ``final_context``; this
    one does too, plus ``outputs``. ``calls`` holds ``(task, initial_context)``.
    """

    def __init__(self, name: str, outputs: dict | None = None) -> None:
        self.name = name
        self.outputs = dict(outputs or {})
        self.calls: list[tuple[str, dict]] = []

    def run(self, task: str, initial_context: dict | None = None):
        from fsm_llm.agents.definitions import AgentResult

        context = dict(initial_context or {})
        self.calls.append((task, context))
        return AgentResult(
            answer=f"{self.name} answer",
            success=True,
            stop_reason="answered",
            final_context={**context, f"from_{self.name}": True, **self.outputs},
        )


def _graph(nodes: dict, edges: list[tuple[str, str]], entry: str = "a"):
    from fsm_llm.agents.agent_graph import AgentGraphBuilder

    builder = AgentGraphBuilder()
    for name, node in nodes.items():
        builder.add_node(name, node)
    for source, target in edges:
        builder.add_edge(source, target)
    return builder.set_entry(entry).build()


class TestAgentGraphOrder:
    """Step 22 (PAT-07): nodes run in topological order, a convergence node
    runs once after all its predecessors on their merged contexts, and the
    answer comes from the last executed sink."""

    def test_diamond_runs_the_join_once_on_both_contexts(self):
        nodes = {n: _RecordingNode(n) for n in "abcd"}
        graph = _graph(nodes, [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])

        result = graph.run("task")

        assert result.final_context["_graph_execution_order"] == ["a", "b", "c", "d"]
        assert len(nodes["d"].calls) == 1
        _, d_context = nodes["d"].calls[0]
        assert d_context["from_b"] is True
        assert d_context["from_c"] is True
        assert result.answer == "d answer"

    def test_join_waits_for_the_longer_branch(self):
        # BFS ran d straight after a (shortcut edge), before b and c, and
        # answered from c.
        nodes = {n: _RecordingNode(n) for n in "adbc"}
        graph = _graph(nodes, [("a", "d"), ("a", "b"), ("b", "c"), ("c", "d")])

        result = graph.run("task")

        assert result.final_context["_graph_execution_order"] == ["a", "b", "c", "d"]
        assert result.answer == "d answer"
        assert nodes["d"].calls[0][1]["from_c"] is True

    def test_later_predecessor_wins_a_shared_key(self):
        nodes = {
            "a": _RecordingNode("a"),
            "b": _RecordingNode("b", {"shared": "from b"}),
            "c": _RecordingNode("c", {"shared": "from c"}),
            "d": _RecordingNode("d"),
        }
        graph = _graph(nodes, [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])

        graph.run("task")

        assert nodes["d"].calls[0][1]["shared"] == "from c"

    def test_join_runs_on_the_satisfied_edge_only(self):
        from fsm_llm.agents.agent_graph import AgentGraphBuilder

        nodes = {n: _RecordingNode(n) for n in "abcd"}
        graph = (
            AgentGraphBuilder()
            .add_node("a", nodes["a"])
            .add_node("b", nodes["b"])
            .add_node("c", nodes["c"])
            .add_node("d", nodes["d"])
            .add_edge("a", "b")
            .add_edge("a", "c")
            .add_edge("b", "d", condition=lambda ctx: False)
            .add_edge("c", "d")
            .set_entry("a")
            .build()
        )

        result = graph.run("task")

        assert result.final_context["_graph_execution_order"] == ["a", "b", "c", "d"]
        d_context = nodes["d"].calls[0][1]
        assert d_context.get("from_c") is True
        assert "from_b" not in d_context

    def test_unreachable_and_untaken_nodes_do_not_run(self):
        from fsm_llm.agents.agent_graph import AgentGraphBuilder

        nodes = {n: _RecordingNode(n) for n in "xab"}
        graph = (
            AgentGraphBuilder()
            .add_node("x", nodes["x"])  # upstream of the entry: never runs
            .add_node("a", nodes["a"])
            .add_node("b", nodes["b"])
            .add_edge("x", "a")
            .add_edge("a", "b", condition=lambda ctx: False)
            .set_entry("a")
            .build()
        )

        result = graph.run("task")

        assert result.final_context["_graph_execution_order"] == ["a"]
        assert result.answer == "a answer"
        assert nodes["x"].calls == [] and nodes["b"].calls == []

    def test_direct_construction_rejects_a_cycle(self):
        from fsm_llm.agents.agent_graph import AgentGraph

        with pytest.raises(ValueError, match="cycle"):
            AgentGraph(
                nodes={"a": _RecordingNode("a"), "b": _RecordingNode("b")},  # type: ignore[dict-item]
                adjacency={"a": [("b", None)], "b": [("a", None)]},
                entry="a",
            )

    def test_long_chain_builds_and_runs(self):
        # AG-001: the cycle check must stay iterative.
        names = [f"n{i}" for i in range(1500)]
        nodes = {n: _RecordingNode(n) for n in names}
        graph = _graph(nodes, list(itertools.pairwise(names)), entry="n0")

        assert graph.run("task").answer == "n1499 answer"

    def test_downstream_react_does_not_conclude_from_leaked_keys(self):
        # The upstream ReAct concludes with should_terminate, observation_count
        # and final_answer in its final_context; the downstream one must run
        # its own tool before it may conclude.
        from fsm_llm.agents import AgentConfig, ReactAgent

        runs: list[str] = []

        def react() -> ReactAgent:
            return ReactAgent(
                tools=_lookup_registry(runs),
                config=AgentConfig(max_iterations=6),
                llm_interface=PromptGroundedLLM(facts=_REACT_FACTS),
            )

        graph = _graph({"a": react(), "b": react()}, [("a", "b")])
        result = graph.run("What is the capital of France?")

        assert runs == ["capital of France", "capital of France"]
        assert len(result.trace.tool_calls) == 2
        assert result.final_context[ContextKeys.OBSERVATION_COUNT] == 1
        assert (result.success, result.stop_reason) == (True, "answered")


class TestSwarmHandoff:
    """Step 22 (PAT-08): every member works on the original task, the handoff
    message travels as context, and ``max_handoffs=N`` allows N handoffs."""

    def test_task_preserved_and_message_passed_as_context(self):
        from fsm_llm.agents.swarm import SwarmAgent

        a = _RecordingNode(
            "a", {"next_agent": "b", "handoff_message": "check the bill"}
        )
        b = _RecordingNode("b")
        result = SwarmAgent(agents={"a": a, "b": b}, entry_agent="a").run("my task")  # type: ignore[dict-item]

        assert [task for task, _ in a.calls + b.calls] == ["my task", "my task"]
        b_context = b.calls[0][1]
        assert b_context["handoff_message"] == "check the bill"
        assert b_context["previous_agent"] == "a"
        assert result.answer == "b answer"

    def test_missing_message_passes_the_previous_answer(self):
        from fsm_llm.agents.swarm import SwarmAgent

        a = _RecordingNode("a", {"next_agent": "b"})
        b = _RecordingNode("b")
        SwarmAgent(agents={"a": a, "b": b}, entry_agent="a").run("t")  # type: ignore[dict-item]

        assert b.calls[0][1]["handoff_message"] == "a answer"

    def test_echoed_message_is_not_forwarded_again(self):
        # b echoes the message it was handed; c must get b's own answer.
        from fsm_llm.agents.swarm import SwarmAgent

        a = _RecordingNode("a", {"next_agent": "b", "handoff_message": "from a"})
        b = _RecordingNode("b", {"next_agent": "c"})
        c = _RecordingNode("c")
        SwarmAgent(agents={"a": a, "b": b, "c": c}, entry_agent="a").run("t")  # type: ignore[dict-item]

        assert c.calls[0][1]["handoff_message"] == "b answer"

    @pytest.mark.parametrize("max_handoffs", [0, 1, 2, 3])
    def test_max_handoffs_allows_exactly_n(self, max_handoffs):
        from fsm_llm.agents.swarm import SwarmAgent

        a = _RecordingNode("a", {"next_agent": "b"})
        b = _RecordingNode("b", {"next_agent": "a"})
        result = SwarmAgent(
            agents={"a": a, "b": b},  # type: ignore[dict-item]
            entry_agent="a",
            max_handoffs=max_handoffs,
        ).run("t")

        assert result.final_context["_swarm_handoff_count"] == max_handoffs
        assert len(a.calls) + len(b.calls) == max_handoffs + 1
        assert (result.success, result.stop_reason) == (False, "max_iterations")

    def test_budget_not_spent_when_the_chain_ends(self):
        from fsm_llm.agents.swarm import SwarmAgent

        a = _RecordingNode("a", {"next_agent": "b"})
        b = _RecordingNode("b")
        result = SwarmAgent(
            agents={"a": a, "b": b},  # type: ignore[dict-item]
            entry_agent="a",
            max_handoffs=1,
        ).run("t")

        assert (result.success, result.stop_reason) == (True, "answered")
        assert result.final_context["_swarm_handoff_chain"] == ["a", "b"]

    def test_caller_next_agent_is_not_forwarded(self):
        # An echoing member would otherwise re-request the handoff every hop.
        from fsm_llm.agents.swarm import SwarmAgent

        a = _RecordingNode("a")
        result = SwarmAgent(agents={"a": a}, entry_agent="a").run(  # type: ignore[dict-item]
            "t", initial_context={"next_agent": "a", "handoff_context": {"x": 1}}
        )

        assert len(a.calls) == 1
        assert "next_agent" not in a.calls[0][1]
        assert (result.success, result.stop_reason) == (True, "answered")

    @pytest.mark.parametrize("bad", ["a string", ["list"], 7])
    def test_non_mapping_handoff_context_is_ignored_with_a_warning(self, bad):
        from fsm_llm.agents.swarm import SwarmAgent
        from fsm_llm.logging import logger

        a = _RecordingNode("a", {"next_agent": "b", "handoff_context": bad})
        b = _RecordingNode("b")
        warnings: list[str] = []
        sink = logger.add(lambda m: warnings.append(str(m)), level="WARNING")
        logger.enable("fsm_llm")
        try:
            result = SwarmAgent(agents={"a": a, "b": b}, entry_agent="a").run("t")  # type: ignore[dict-item]
        finally:
            logger.remove(sink)
            logger.disable("fsm_llm")

        assert result.answer == "b answer"
        assert any("non-mapping handoff_context" in w for w in warnings)

    def test_mapping_handoff_context_reaches_the_next_agent_filtered(self):
        from fsm_llm.agents.swarm import SwarmAgent

        a = _RecordingNode(
            "a",
            {
                "next_agent": "b",
                "handoff_context": {
                    "account": "42",
                    "final_answer": "PWNED",
                    "next_agent": "a",
                },
            },
        )
        b = _RecordingNode("b")
        result = SwarmAgent(agents={"a": a, "b": b}, entry_agent="a").run("t")  # type: ignore[dict-item]

        b_context = b.calls[0][1]
        assert b_context["account"] == "42"
        assert "final_answer" not in b_context
        assert "next_agent" not in b_context
        assert result.final_context["_swarm_handoff_chain"] == ["a", "b"]


# Step 24 (API-01): the only evidence for the tool call is in the instructions.
_POLICY = "Follow POLICY-7: look every fact up before answering."
_POLICY_FACTS: dict[str, tuple[object, str]] = {
    "tool_name": ("lookup", "POLICY-7"),
    "tool_input": ({"query": "capital of France"}, "POLICY-7"),
    "should_terminate": (True, "is Paris"),
}


def _spoken_replies(llm: PromptGroundedLLM) -> list:
    """Pass-2 requests that reach the model (core skips silent states)."""
    return [r for r in llm.calls("generate_response") if not r.skip_generation]


class TestAgentInstructions:
    """Step 24 (API-01): ``system_prompt`` / ``AgentConfig.instructions``
    reach the prompts the model decides on, and silent states stay silent."""

    def _react(self, *args: object, **kwargs: object):
        from fsm_llm.agents import AgentConfig, create_agent

        runs: list[str] = []
        llm = PromptGroundedLLM(facts=_POLICY_FACTS, default_response="Paris")
        agent = create_agent(
            *args,  # type: ignore[arg-type]
            tools=_lookup_registry(runs),
            config=AgentConfig(max_iterations=6),
            llm_interface=llm,
            **kwargs,
        )
        return runs, llm, agent.run("What is the capital of France?")

    def test_react_system_prompt_steers_the_tool_choice(self):
        runs, llm, result = self._react("react", system_prompt=_POLICY)

        assert runs == ["capital of France"]
        assert result.success
        assert all(_POLICY in r.system_prompt for r in llm.calls("extract_field"))
        replies = _spoken_replies(llm)
        assert replies and all(_POLICY in r.system_prompt for r in replies)

    def test_react_without_instructions_never_sees_the_policy(self):
        runs, llm, _ = self._react("react")

        assert runs == []
        assert not any("POLICY-7" in r.system_prompt for _, r in llm.requests)

    def test_legacy_positional_prompt_reaches_the_prompt(self):
        with pytest.warns(DeprecationWarning, match="first argument"):
            runs, llm, _ = self._react(_POLICY)

        assert runs == ["capital of France"]
        assert any(_POLICY in r.system_prompt for r in llm.calls("extract_field"))

    def test_debate_instructions_reach_every_prompt_and_keep_silence(self):
        from fsm_llm.agents import create_agent

        rule = "House rule DR-9: cite one number per claim."
        llm = _TurnAwareLLM(_debate_derived(), responses={"conclude": _CONCLUDE_TEXT})
        agent = create_agent(
            "debate", system_prompt=rule, num_rounds=1, llm_interface=llm
        )
        result = agent.run(_DEBATE_TASK)

        assert result.answer == _CONCLUDE_TEXT
        fields = llm.calls("extract_field")
        assert fields and all(rule in r.system_prompt for r in fields)
        replies = _spoken_replies(llm)
        assert replies and all(rule in r.system_prompt for r in replies)
        assert _spoken_states(llm) == {"conclude"}


# ---------------------------------------------------------------------------
# Fix 13.1: ``success`` reflects who concluded (D-051 of plan 06a5ec0a)
# ---------------------------------------------------------------------------


def _haiku_output(text: str, _ctx: dict) -> object:
    if "ADD-IMAGERY" in text:
        return "HAIKU-2"
    return "HAIKU-1" if "haiku" in text else None


def _haiku_evaluation(out: str, _ctx: dict) -> object:
    from fsm_llm.agents.definitions import EvaluationResult

    ok = out == "HAIKU-2"
    return EvaluationResult(
        passed=ok, score=1.0 if ok else 0.2, feedback="ok" if ok else "ADD-IMAGERY"
    )


# should_terminate is grounded only once the lookup observation ("is Paris")
# is in the think prompt, so turn 1 selects the tool and turn 2 concludes.
_NO_TERMINATE = {k: v for k, v in _REACT_FACTS.items() if k != "should_terminate"}


class _PlantingLLM(PromptGroundedLLM):
    """Every bulk extraction also returns ``planted`` (a model writing
    framework keys through a state's bulk call)."""

    def __init__(self, planted: dict, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.planted = dict(planted)

    def extract_bulk_data(self, request: BulkExtractionRequest):
        response = super().extract_bulk_data(request)
        response.extracted_data.update(self.planted)
        return response


class _FailingNode(_RecordingNode):
    def run(self, task: str, initial_context: dict | None = None):
        result = super().run(task, initial_context)
        return result.model_copy(
            update={"success": False, "stop_reason": "max_iterations"}
        )


class TestSuccessReflectsWhoConcluded:
    """A forced reason is reported only when the framework, not the model or
    judge, decided the outcome; a genuine pass on the budget's last round is a
    success. Framework keys cannot be planted by extraction."""

    # -- EvalOpt / MakerChecker: forced_pass only on an overridden verdict --

    def _evalopt(self, max_iterations: int):
        from fsm_llm.agents import AgentConfig, EvaluatorOptimizerAgent

        return EvaluatorOptimizerAgent(
            evaluation_fn=_haiku_evaluation,
            max_refinements=3,
            config=AgentConfig(max_iterations=max_iterations),
            llm_interface=_TurnAwareLLM({ContextKeys.GENERATED_OUTPUT: _haiku_output}),
        ).run("Write a haiku about rain")

    @pytest.mark.parametrize("max_iterations", [3, 4, 5])
    def test_evalopt_pass_on_the_limiter_round_is_success(self, max_iterations):
        result = self._evalopt(max_iterations)

        assert result.answer == "HAIKU-2"
        assert result.final_context[ContextKeys.EVALUATION_RESULT]["passed"] is True
        assert (result.success, result.stop_reason) == (True, "answered")

    def test_evalopt_budget_forced_failing_output_is_forced_pass(self):
        result = self._evalopt(2)

        assert result.answer == "HAIKU-1"
        assert (result.success, result.stop_reason) == (False, "forced_pass")

    def _maker_checker(self, max_iterations: int):
        from fsm_llm.agents import AgentConfig, MakerCheckerAgent

        return MakerCheckerAgent(
            maker_instructions="Write a haiku.",
            checker_instructions="Check.",
            max_revisions=3,
            config=AgentConfig(max_iterations=max_iterations),
            llm_interface=_TurnAwareLLM(_maker_checker_derived()),
        ).run(_MC_TASK)

    @pytest.mark.parametrize("max_iterations", [3, 4])
    def test_maker_checker_pass_on_the_limiter_round_is_success(self, max_iterations):
        result = self._maker_checker(max_iterations)

        assert result.answer == "DRAFT-2"
        assert result.final_context["quality_score"] == 0.9
        assert (result.success, result.stop_reason) == (True, "answered")

    def test_maker_checker_budget_forced_failing_draft_is_forced_pass(self):
        result = self._maker_checker(2)

        assert result.answer == "DRAFT-1"
        assert (result.success, result.stop_reason) == (False, "forced_pass")

    # -- Reflexion: the max_reflections cap is a forced stop --

    def test_reflexion_cap_without_a_pass_is_forced_pass(self):
        from fsm_llm.agents import AgentConfig, ReflexionAgent

        runs: list[str] = []
        llm = _TurnAwareLLM(_reflect_derived(), facts=_REFLEXION_FACTS)
        result = ReflexionAgent(
            tools=_lookup_registry(runs),
            config=AgentConfig(max_iterations=10),
            max_reflections=2,
            llm_interface=llm,
        ).run("What is the capital of France?")

        assert len(result.final_context[ContextKeys.EPISODIC_MEMORY]) == 2
        assert result.final_context.get(ContextKeys.EVALUATION_PASSED) is not True
        assert result.answer
        assert (result.success, result.stop_reason) == (False, "forced_pass")

    def test_reflexion_pass_on_the_last_allowed_turn_is_success(self):
        from fsm_llm.agents import AgentConfig, ReflexionAgent

        runs: list[str] = []
        facts = dict(_REFLEXION_FACTS, evaluation_passed=(True, "is Paris"))
        result = ReflexionAgent(
            tools=_lookup_registry(runs),
            config=AgentConfig(max_iterations=2),
            llm_interface=_TurnAwareLLM(_reflect_derived(), facts=facts),
        ).run("What is the capital of France?")

        assert runs == ["capital of France"]
        assert (result.success, result.stop_reason) == (True, "answered")

    # -- ReAct family: a model conclusion on the last think turn succeeds --

    def _react_family(self, pattern: str, facts: dict, max_iterations: int):
        from fsm_llm.agents import AgentConfig, ReactAgent
        from fsm_llm.agents.reasoning_react import ReasoningReactAgent

        cls = {"react": ReactAgent, "reasoning_react": ReasoningReactAgent}[pattern]
        runs: list[str] = []
        result = cls(
            tools=_lookup_registry(runs),
            config=AgentConfig(max_iterations=max_iterations),
            llm_interface=PromptGroundedLLM(facts=facts),
        ).run("What is the capital of France?")
        return runs, result

    @pytest.mark.parametrize("pattern", ["react", "reasoning_react"])
    @pytest.mark.parametrize("max_iterations", [1, 2])
    def test_model_conclusion_on_the_last_think_turn_is_success(
        self, pattern, max_iterations
    ):
        runs, result = self._react_family(pattern, _REACT_FACTS, max_iterations)

        assert runs == ["capital of France"]
        assert result.final_context[ContextKeys.SHOULD_TERMINATE] is True
        assert (result.success, result.stop_reason) == (True, "answered")

    @pytest.mark.parametrize("pattern", ["react", "reasoning_react"])
    @pytest.mark.parametrize("max_iterations", [1, 2, 3])
    def test_limiter_forced_conclusion_is_still_max_iterations(
        self, pattern, max_iterations
    ):
        runs, result = self._react_family(pattern, _NO_TERMINATE, max_iterations)

        assert runs
        assert result.answer
        assert (result.success, result.stop_reason) == (False, "max_iterations")

    @pytest.mark.parametrize(
        ("max_iterations", "tool_turns", "think_turns"),
        [(1, 1, 2), (2, 1, 2), (3, 2, 3)],
    )
    def test_documented_react_budget_counts(
        self, max_iterations, tool_turns, think_turns
    ):
        # D-028 as documented: N >= 2 gives N think turns and N - 1 tool
        # turns; N = 1 behaves like N = 2.
        runs, result = self._react_family("react", _NO_TERMINATE, max_iterations)

        assert len(runs) == tool_turns
        assert result.final_context[ContextKeys.ITERATION_COUNT] == think_turns

    # -- Debate: a forced consensus is forced; the success key survives --

    def _debate(self, derived: dict, num_rounds: int = 2):
        llm = _TurnAwareLLM(derived, responses={"conclude": _CONCLUDE_TEXT})
        return DebateAgent(num_rounds=num_rounds, llm_interface=llm).run(_DEBATE_TASK)

    def test_debate_consensus_forced_by_num_rounds_is_forced_pass(self):
        result = self._debate(_debate_derived(consensus=False))

        assert len(result.final_context[ContextKeys.DEBATE_ROUNDS]) == 2
        assert result.answer == _CONCLUDE_TEXT
        assert (result.success, result.stop_reason) == (False, "forced_pass")

    def test_debate_judge_consensus_on_the_last_round_is_success(self):
        derived = _debate_derived()
        # The judge decides on current_round (its prompt also shows debate_rounds).
        derived[ContextKeys.CONSENSUS_REACHED] = lambda _t, ctx: (
            ctx.get(ContextKeys.CURRENT_ROUND, 1) > 1
        )
        result = self._debate(derived)

        assert len(result.final_context[ContextKeys.DEBATE_ROUNDS]) == 2
        assert (result.success, result.stop_reason) == (True, "answered")

    def test_debate_final_round_without_proposition_keeps_the_success_key(self):
        derived = _debate_derived()
        # Round 2's proposer returns nothing; the judge still agrees.
        derived[ContextKeys.PROPOSITION] = lambda text, ctx: (
            None if ctx.get(ContextKeys.DEBATE_ROUNDS) else _P1
        )
        # The judge decides on current_round (its prompt also shows debate_rounds).
        derived[ContextKeys.CONSENSUS_REACHED] = lambda _t, ctx: (
            ctx.get(ContextKeys.CURRENT_ROUND, 1) > 1
        )
        result = self._debate(derived)

        assert result.final_context[ContextKeys.PROPOSITION] == _P1
        assert (result.success, result.stop_reason) == (True, "answered")

    def test_debate_limiter_forced_consensus_records_max_iterations(self):
        agent = DebateAgent(num_rounds=1, llm_interface=PromptGroundedLLM())
        limiter = agent._make_iteration_limiter()
        spent = {ContextKeys.ITERATION_COUNT: agent._fsm_budget()}

        forced = limiter(dict(spent))
        judged = limiter({**spent, ContextKeys.CONSENSUS_REACHED: True})

        assert forced[ContextKeys.FORCED_STOP_REASON] == "max_iterations"
        assert ContextKeys.FORCED_STOP_REASON not in judged

    # -- AgentGraph / Swarm --

    def test_graph_failed_node_takes_no_outgoing_edge(self):
        nodes = {"a": _FailingNode("a"), "b": _RecordingNode("b")}
        result = _graph(nodes, [("a", "b")]).run("task")

        assert nodes["b"].calls == []
        assert result.final_context["_graph_execution_order"] == ["a"]
        assert (result.success, result.stop_reason) == (False, "max_iterations")

    def test_swarm_handoff_to_unknown_agent_fails(self):
        from fsm_llm.agents.swarm import SwarmAgent

        a = _RecordingNode("a", {"next_agent": "ghost"})
        result = SwarmAgent(agents={"a": a}, entry_agent="a").run("t")  # type: ignore[dict-item]

        assert result.answer == "a answer"
        assert (result.success, result.stop_reason) == (False, "no_result")

    # -- Framework keys are handler-only on every agent FSM --

    def test_every_agent_fsm_declares_the_framework_keys_handler_only(self):
        from fsm_llm.agents import ToolRegistry
        from fsm_llm.agents import fsm_definitions as fd
        from fsm_llm.agents.definitions import ChainStep
        from fsm_llm.agents.parallel_react import build_parallel_react_fsm

        registry = _lookup_registry([])
        defs = [
            fd.build_orchestrator_fsm("t"),
            fd.build_adapt_fsm(task_description="t"),
            fd.build_reflexion_fsm(registry, "t"),
            fd.build_plan_execute_fsm(registry, "t"),
            fd.build_react_fsm(registry, "t"),
            fd.build_self_consistency_fsm("t"),
            fd.build_debate_fsm("t"),
            fd.build_rewoo_fsm(registry, "t"),
            fd.build_evalopt_fsm("t"),
            fd.build_prompt_chain_fsm(
                [
                    ChainStep(
                        step_id="s",
                        name="s",
                        extraction_instructions="x",
                        response_instructions="y",
                    )
                ],
                "t",
            ),
            fd.build_maker_checker_fsm("make", "check", "t"),
            build_parallel_react_fsm(ToolRegistry(), "t"),
        ]
        for fsm in defs:
            listed = set(fsm.get("handler_only_keys") or [])
            assert {
                ContextKeys.MAX_ITERATIONS_REACHED,
                ContextKeys.FORCED_STOP_REASON,
            } <= listed, fsm["name"]

    @pytest.mark.parametrize(
        "planted",
        [
            {ContextKeys.MAX_ITERATIONS_REACHED: True},
            {ContextKeys.FORCED_STOP_REASON: "stalled"},
        ],
    )
    def test_model_cannot_plant_framework_keys_through_bulk_extraction(self, planted):
        # REWOO's plan_all state runs a bulk call; before D-051 its reply
        # could write the forced flag or reason and flip a real run to failed.
        from fsm_llm.agents import REWOOAgent

        calls: list[str] = []
        llm = _PlantingLLM(
            planted,
            facts={
                "plan_blueprint": ([_lookup_step(1, "capital")], "capital of France"),
                "final_answer": ("Paris", "capital of France"),
            },
        )
        result = REWOOAgent(tools=_rewoo_registry(calls), llm_interface=llm).run(
            _REWOO_TASK
        )

        assert llm.calls("extract_bulk_data"), "no bulk channel was exercised"
        assert calls == ["capital"]
        assert result.final_context[ContextKeys.MAX_ITERATIONS_REACHED] is False
        assert ContextKeys.FORCED_STOP_REASON not in result.final_context
        assert (result.success, result.stop_reason) == (True, "evidence")
