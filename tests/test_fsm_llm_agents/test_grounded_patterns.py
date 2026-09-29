"""Agent patterns driven through the real ``API`` by ``PromptGroundedLLM``.

The fake answers a field only when the prompt shows the evidence for it, so
these tests fail when a pattern asks the model for a value without putting
the task or the prior turns in front of it (the RC1 loop class).
"""

from __future__ import annotations

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

    @pytest.mark.parametrize("field_type", ["str", "float", "list", "bool"])
    def test_supported_types(self, field_type):
        field = _typed_field_extraction("item", field_type, "x")
        assert FieldExtractionConfig.model_validate(field).field_type == field_type

    @pytest.mark.parametrize("field_type", ["any", "dict", "int"])
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
        assert result.success is True
        assert result.stop_reason == "answered"

    def test_consensus_concludes_after_one_round(self):
        _, result = self._run(num_rounds=3, consensus=True)

        assert len(result.final_context[ContextKeys.DEBATE_ROUNDS]) == 1
        assert result.answer == _CONCLUDE_TEXT

    def test_no_proposition_is_no_result(self):
        llm = PromptGroundedLLM(responses={"conclude": _CONCLUDE_TEXT})
        result = DebateAgent(num_rounds=1, llm_interface=llm).run(_DEBATE_TASK)

        assert result.answer == _CONCLUDE_TEXT
        assert result.success is False
        assert result.stop_reason == "no_result"

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
