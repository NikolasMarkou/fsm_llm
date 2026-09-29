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
from fsm_llm.agents.fsm_definitions import _typed_field_extraction
from fsm_llm.agents.handlers import make_fresh_keys_handler
from fsm_llm.api import API
from fsm_llm.definitions import (
    BulkExtractionRequest,
    FieldExtractionConfig,
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
        field = _typed_field_extraction("value", field_type, "x")
        assert FieldExtractionConfig.model_validate(field).field_type == field_type

    @pytest.mark.parametrize("field_type", ["any", "dict", "int"])
    def test_other_types_rejected(self, field_type):
        with pytest.raises(ValueError, match="unsupported"):
            _typed_field_extraction("value", field_type, "x")  # type: ignore[arg-type]

    @pytest.mark.parametrize("key", [ContextKeys.AGENT_TRACE, "_max_iterations"])
    def test_trace_and_internal_context_keys_rejected(self, key):
        with pytest.raises(ValueError, match="not allowed"):
            _typed_field_extraction("value", "str", "x", extra_context_keys=[key])

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
