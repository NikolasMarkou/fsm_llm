"""The file-backed ``native_fc`` (plan a555dcfe step 12, D-021).

``NativeFunctionCallingReactAgent`` is thin on ``PatternAgent``: the machine is
``patterns/native_fc.json`` (the full five-state template and two REMOVAL variants), the eight
handlers are the one library GROUP ``agents.native_fc``. Census rows: D1 (a flagged tool is refused
at construction and at every run start, no gate path exists, ``hitl=`` stays a ``TypeError``), D4
(the file is read lazily, the tool schemas are the nested ``json`` option), D5 (the FSM, the
registration order and the request bytes are the builder's: the differential of step 12 is recorded
in progress.md, ``test_native_fc_golden.py`` holds the bytes).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from fsm_llm.agents import NativeFunctionCallingReactAgent, create_agent
from fsm_llm.agents.base import BaseAgent, accepts_tools
from fsm_llm.agents.constants import NativeFCContextKeys, NativeLoopEnd
from fsm_llm.agents.definitions import AgentConfig
from fsm_llm.agents.fsm_definitions import build_native_fc_fsm
from fsm_llm.agents.fsm_definitions.native_fc import native_fc_options
from fsm_llm.agents.handlers import library as lib
from fsm_llm.agents.handlers.context import HandlerContext
from fsm_llm.agents.native_fc import _SYSTEM_PROMPT
from fsm_llm.agents.pattern_agent import PatternAgent
from fsm_llm.agents.pattern_files import load_document, load_envelope
from fsm_llm.core.definitions import AgentError
from fsm_llm.core.definitions.executable_instance import (
    InstantiationError,
    instantiate_body,
)
from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.testing import final_reply
from tests import pattern_runs as runs
from tests.test_fsm_llm.test_completion_state import _ScriptedLLM
from tests.test_fsm_llm_agents import test_native_fc_golden as golden

_K = NativeFCContextKeys


class _Out(BaseModel):
    answer: str


class _FakeApi:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def register_handler(self, handler: Any) -> None:
        self.handlers.append(handler)


def _agent(llm: Any = None, **config: Any) -> NativeFunctionCallingReactAgent:
    return NativeFunctionCallingReactAgent(
        golden._registry(),
        config=AgentConfig(model="mock-model", **config),
        llm_interface=llm or _ScriptedLLM(final_reply("done")),
    )


def _states(fsm: dict[str, Any]) -> list[str]:
    return list(fsm["states"])


def _edges(fsm: dict[str, Any], state: str) -> list[str]:
    return [t["target_state"] for t in fsm["states"][state].get("transitions", [])]


class TestNativeFcIsAPatternAgent:
    def test_the_hierarchy_the_names_and_the_attributes_callers_read(self):
        agent = _agent()
        mro = NativeFunctionCallingReactAgent.__mro__

        assert mro[:3] == (NativeFunctionCallingReactAgent, PatternAgent, BaseAgent)
        assert "run" not in vars(NativeFunctionCallingReactAgent)
        assert agent.system_policy is None and agent.seed is None
        assert accepts_tools(NativeFunctionCallingReactAgent)
        assert type(create_agent("native_fc", golden._registry())) is (
            NativeFunctionCallingReactAgent
        )

    def test_hitl_is_refused_with_the_base_type_error_not_the_pattern_one(self):
        with pytest.raises(TypeError, match="hitl"):
            NativeFunctionCallingReactAgent(golden._registry(), hitl=runs._hitl())

    def test_the_run_block_stays_the_files_and_the_classes_attributes_the_drivers(self):
        run = load_envelope("native_fc").body.run

        assert run.agent_type == "native_fc"
        assert run.success_keys == [_K.ANSWER]
        assert run.run_output_keys == [] and run.seconds_exempt == []
        assert _K.ANSWER in NativeFunctionCallingReactAgent._run_output_keys
        assert "conclude" in NativeFunctionCallingReactAgent._seconds_exempt_states

    def test_the_fsm_of_a_run_is_the_builder_over_what_is_read_now(self):
        agent = _agent(force_final_tool="write_note", output_schema=_Out)
        agent.system_policy = "RULES"

        fsm = agent._fsm_for_run("task", None)

        assert fsm == build_native_fc_fsm(
            golden._registry().get_json_schemas(),
            instructions=f"{_SYSTEM_PROMPT}\n\nRULES",
            force_final_tool="write_note",
            repair=True,
        )
        assert _states(fsm) == [
            "call_model",
            "run_tools",
            "force_final",
            "repair",
            "conclude",
        ]
        assert fsm["states"]["repair"]["completion"]["instructions"].endswith("RULES")

    def test_a_tool_added_after_construction_is_in_the_next_runs_request(self):
        agent = _agent()
        before = agent._fsm_for_run("t", None)["states"]["call_model"]["completion"]
        agent.tools.register_function(lambda params: "x", name="late", description="d")

        after = agent._fsm_for_run("t", None)["states"]["call_model"]["completion"]

        assert len(after["tools"]) == len(before["tools"]) + 1

    def test_the_registration_is_the_eight_handlers_in_order(self):
        agent = _agent()
        api = _FakeApi()

        agent._register_handlers(api, agent._handler_context())

        assert [h.name for h in api.handlers] == [
            "native_seed",
            "native_model_reply",
            "native_run_tools",
            "native_force_entry",
            "native_forced_reply",
            "native_repair_entry",
            "native_repair_reply",
            "native_conclude",
        ]
        assert all(h.critical for h in api.handlers)

    def test_two_runs_share_no_context_and_no_handler(self):
        agent = _agent()
        ctx_one = agent._handler_context()
        ctx_two = agent._handler_context()
        one = lib.native_fc(ctx_one)
        two = lib.native_fc(ctx_two)

        assert ctx_one.scope is not ctx_two.scope
        assert {id(h) for h in one}.isdisjoint({id(h) for h in two})
        assert "_handlers" not in vars(agent)

    def test_the_parser_is_the_classs_internal_callable_not_a_host_object(self):
        envelope = load_envelope("native_fc")
        agent = _agent()

        assert envelope.requires.callables == ["tools"]
        assert set(agent._callables) == {"parse_structured"}


class TestTheFileIsTheFullMachineWithTwoRemovals:
    def test_the_default_instance_has_every_state_and_edge_in_the_builders_order(self):
        envelope = load_envelope("native_fc")

        fsm = instantiate_body(envelope, variables={"system_message": "s"})

        assert _states(fsm) == [
            "call_model",
            "run_tools",
            "force_final",
            "repair",
            "conclude",
        ]
        assert _edges(fsm, "call_model") == [
            "run_tools",
            "force_final",
            "repair",
            "conclude",
        ]
        assert _edges(fsm, "run_tools") == [
            "force_final",
            "repair",
            "conclude",
            "call_model",
        ]
        assert _edges(fsm, "force_final") == ["repair", "conclude"]

    @pytest.mark.parametrize(
        ("forced", "repair", "states"),
        [
            (None, False, ["call_model", "run_tools", "conclude"]),
            (
                "write_note",
                False,
                ["call_model", "run_tools", "force_final", "conclude"],
            ),
            (None, True, ["call_model", "run_tools", "repair", "conclude"]),
            (
                "write_note",
                True,
                ["call_model", "run_tools", "force_final", "repair", "conclude"],
            ),
            ("missing", True, ["call_model", "run_tools", "repair", "conclude"]),
        ],
    )
    def test_each_combination_keeps_the_states_and_the_edge_order(
        self, forced, repair, states
    ):
        fsm = build_native_fc_fsm(
            golden._registry().get_json_schemas(),
            instructions="s",
            force_final_tool=forced,
            repair=repair,
        )

        assert _states(fsm) == states
        every = [t for s in fsm["states"] for t in _edges(fsm, s)]
        assert set(every) <= set(states)  # no edge is left pointing at a removed state
        assert _edges(fsm, "call_model")[0] == "run_tools"
        assert _edges(fsm, "call_model")[-1] == "conclude"
        has_force = "force_final" in states
        has_repair = "repair" in states
        assert _edges(fsm, "run_tools") == [
            *(["force_final"] if has_force else []),
            *(["repair"] if has_repair else []),
            "conclude",
            "call_model",
        ]

    def test_the_variants_are_declared_highest_index_first(self):
        document = load_document("native_fc")

        assert list(document["body"]["variants"]) == [
            "without_repair",
            "without_force_final",
        ]
        assert set(document["body"]["options"]) >= {
            "without_repair",
            "without_force_final",
        }

    def test_an_option_cannot_select_a_variant_the_agent_did_not_derive(self):
        # the agent derives both booleans from the config; a caller option of those names is
        # simply the variant selection of ``pattern_fsm`` (the options are not a caller seam here)
        options = native_fc_options(golden._registry().get_json_schemas(), None, False)

        assert options["without_force_final"] is True
        assert options["without_repair"] is True
        assert options["force_final_tool"] == ""

    def test_the_tool_schemas_are_a_nested_json_option_bounded_by_the_options_subset(
        self,
    ):
        envelope = load_envelope("native_fc")
        spec = envelope.body.options_schema.properties["tool_schemas"]

        assert spec.items is not None and spec.items.type == "json"
        too_many = [
            {"type": "function", "function": {"name": f"t{i}"}} for i in range(257)
        ]
        with pytest.raises(InstantiationError, match="tool_schemas"):
            instantiate_body(
                envelope,
                {"tool_schemas": too_many},
                variables={"system_message": "s"},
            )

    def test_a_system_message_is_inserted_once_and_never_rescanned(self):
        text = "rules {{system_message}} {{x}} é"

        fsm = build_native_fc_fsm(
            golden._registry().get_json_schemas(), instructions=text, repair=True
        )

        for state in ("call_model", "repair"):
            assert fsm["states"][state]["completion"]["instructions"] == text

    def test_a_tool_description_with_braces_is_not_a_template(self):
        schema = {
            "type": "function",
            "function": {"name": "t", "description": "{{system_message}} {x}"},
        }

        fsm = build_native_fc_fsm([schema], instructions="real")

        assert fsm["states"]["call_model"]["completion"]["tools"] == [schema]

    def test_no_tool_schema_is_a_value_error(self):
        with pytest.raises(ValueError, match="at least one tool schema"):
            build_native_fc_fsm([], instructions="s")


def _ctx(**config: Any) -> HandlerContext:
    return HandlerContext(
        tools=golden._registry(), config=AgentConfig(model="m", **config)
    )


class TestTheGroupFactory:
    def test_it_builds_the_eight_critical_handlers_in_order_with_the_registered_tuples(
        self,
    ):
        built = lib.native_fc(_ctx())

        assert [h.name for h in built] == [
            "native_seed",
            "native_model_reply",
            "native_run_tools",
            "native_force_entry",
            "native_forced_reply",
            "native_repair_entry",
            "native_repair_reply",
            "native_conclude",
        ]
        assert [runs.handler_tuple(h)[1:] for h in built] == [
            [["start_conversation"], [], [], [], 100],
            [["context_update"], ["call_model"], [], ["model_reply"], 100],
            [["post_transition"], [], ["run_tools"], [], 100],
            [["post_transition"], [], ["force_final"], [], 100],
            [["context_update"], ["force_final"], [], ["forced_reply"], 100],
            [["post_transition"], [], ["repair"], [], 100],
            [["context_update"], ["repair"], [], ["repair_reply"], 100],
            [["post_transition"], [], ["conclude"], [], 100],
        ]

    def test_a_binding_changes_nothing_in_a_group(self):
        named = SimpleNamespace(
            states=("think",), keys=("_approval_granted",), priority=7, params={"x": 1}
        )

        assert [runs.handler_tuple(h) for h in lib.native_fc(_ctx(), named)] == [
            runs.handler_tuple(h) for h in lib.native_fc(_ctx())
        ]

    def test_a_context_with_no_tools_or_no_config_is_refused(self):
        with pytest.raises(AgentError, match="needs the run's tools and config"):
            lib.native_fc(HandlerContext(config=AgentConfig(model="m")))
        with pytest.raises(AgentError, match="needs the run's tools and config"):
            lib.native_fc(HandlerContext(tools=golden._registry()))

    def _end(self, built: list[Any], text: str) -> dict[str, Any]:
        reply = {"kind": "final", "text": text, "calls": []}
        return built[1].execute({_K.MODEL_REPLY: reply})

    def test_the_turn_cap_the_forced_tool_and_the_parser_are_read_from_the_run(self):
        parse = lambda text: None if text == "bad" else {"ok": text}  # noqa: E731
        ctx = HandlerContext(
            tools=golden._registry(),
            config=AgentConfig(
                model="m",
                max_iterations=3,
                force_final_tool="write_note",
                output_schema=_Out,
            ),
            callables={"parse_structured": parse},
        )
        built = lib.native_fc(ctx)

        flagged = self._end(built, "bad")
        parsed = self._end(built, "good")

        assert flagged[_K.REPAIR_PENDING] is True and flagged[_K.FORCE_PENDING] is True
        assert parsed[_K.REPAIR_PENDING] is False
        assert flagged[_K.LOOP_END] == NativeLoopEnd.FINAL
        exhausted = built[2].execute(
            {
                _K.MODEL_REPLY: {
                    "kind": "calls",
                    "text": None,
                    "calls": [
                        {"id": "1", "name": "weather", "arguments": {"city": "x"}}
                    ],
                },
                _K.TOOL_TURNS: 2,
            }
        )
        assert exhausted[_K.LOOP_END] == NativeLoopEnd.EXHAUSTED

    def test_with_no_output_schema_no_repair_is_ever_flagged_even_with_a_parser(self):
        ctx = HandlerContext(
            tools=golden._registry(),
            config=AgentConfig(model="m"),
            callables={"parse_structured": lambda text: None},
        )

        assert self._end(lib.native_fc(ctx), "anything")[_K.REPAIR_PENDING] is False


class TestFailClosed:
    def test_a_flagged_tool_is_refused_at_construction_and_at_every_run_start(self):
        registry = golden._registry()
        registry.register_function(
            lambda params: "x", name="danger", description="d", requires_approval=True
        )
        with pytest.raises(AgentError, match="requires_approval"):
            NativeFunctionCallingReactAgent(
                registry, config=AgentConfig(model="m"), llm_interface=_ScriptedLLM()
            )
        llm = _ScriptedLLM(final_reply("done"))
        agent = _agent(llm)
        agent.tools.register_function(
            lambda params: "x", name="danger", description="d", requires_approval=True
        )

        with pytest.raises(AgentError, match="danger"):
            agent.run("go")
        assert llm.requests == []

    def test_a_file_that_needs_tools_and_binds_no_gate_runs_the_refusal(self):
        agent = _agent()

        assert agent._executes_tools is True
        assert agent._gate is False and agent._react_loop is False

    def test_an_empty_registry_is_refused(self):
        from fsm_llm.tools import ToolRegistry

        with pytest.raises(AgentError, match="empty tool registry"):
            NativeFunctionCallingReactAgent(ToolRegistry())


class TestTheNativeExecutableIsTheFile:
    def _exe(self):
        registry = ExecutableRegistry()
        registry.load_sources()
        return registry.get("native_fc")

    def test_the_row_keeps_its_id_and_its_objects(self):
        executable = self._exe()

        assert executable.template_identity
        assert executable.optional_callables == ("config",)
        assert list(executable.requires.callables) == ["tools"]
        assert [c.value for c in executable.capabilities] == ["tools"]
        assert executable.requires.tools == "some"

    def test_the_document_is_canonical_text_and_declares_no_gate(self):
        document = load_document("native_fc")

        assert "agents.approval_gate" not in document["requires"]["handlers"]
        assert document["requires"]["handlers"] == ["agents.native_fc"]
        assert document["body"]["handlers"] == [
            {"handler": "agents.native_fc", "on": "state"}
        ]
        assert json.loads(json.dumps(document)) == document


class TestAScriptedRunThroughTheFile:
    def test_a_final_answer_ends_the_run_with_the_answer_and_no_other_call(self):
        llm = _ScriptedLLM(final_reply("Paris is sunny"))
        agent = _agent(llm)

        result = agent.run("weather?")

        assert result.answer == "Paris is sunny"
        assert llm.other_calls == []
        first = llm.requests[0]
        assert first.messages[0]["content"] == _SYSTEM_PROMPT
        assert [t["function"]["name"] for t in first.tools] == [
            t["function"]["name"] for t in golden._registry().get_json_schemas()
        ]
