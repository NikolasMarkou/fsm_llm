"""``PatternAgent`` and the file-backed ``react`` (plan a555dcfe step 9, D-015).

``ReactAgent`` is the first class made thin on ``PatternAgent``: the loop is the shipped file
``patterns/react.json`` and the handlers are the library factories the file binds. Census rows:
D1 (the approval variant is the agent's choice, never an option; a flagged tool with no gate is
refused at construction and at run start; the grant check stays in the executor), D4 (the run is
per-call, nothing is stored on the agent or the registry), D5 (the FSM and the registration are the
ones the builder made: the fixtures, the handler-order golden and the request-byte golden hold the
bytes, and the differential of step 9 is recorded in progress.md).
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from fsm_llm.agents import ReactAgent, create_agent
from fsm_llm.agents import pattern_files as pf
from fsm_llm.agents import pattern_inputs as pi
from fsm_llm.agents.base import BaseAgent, accepts_tools
from fsm_llm.agents.definitions import AgentResult, AgentTrace
from fsm_llm.agents.fsm_definitions import build_react_fsm
from fsm_llm.agents.handlers.context import HandlerContext
from fsm_llm.agents.pattern_agent import PatternAgent
from fsm_llm.core.codes import Status
from fsm_llm.core.definitions import AgentError
from fsm_llm.core.definitions.executable import PatternEnvelope, parse_executable
from fsm_llm.core.definitions.executable_instance import identity_of
from fsm_llm.core.names import NameRegistry
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.executables import ExecutableError, ExecutableRegistry
from tests import pattern_runs as runs

_FIRST_FOUR = [
    "agents.tool_executor",
    "agents.think_turn_limiter",
    "agents.think_fresh_keys",
    "agents.feedback_consumed",
]


class _FakeApi:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def register_handler(self, handler: Any) -> None:
        self.handlers.append(handler)


def _react(**kw: Any) -> ReactAgent:
    return ReactAgent(
        runs._tools(flagged=kw.pop("flagged", False)),
        config=runs._config(),
        llm_interface=kw.pop("llm", runs.ScriptedModel()),
        **kw,
    )


def _user_doc(*, gate: bool, doc_id: str = "user.loop") -> dict[str, Any]:
    """The react document with a new id; without the gate it keeps only the four loop handlers."""
    doc = pf.load_document("react")
    doc["id"] = doc_id
    if not gate:
        doc["capabilities"] = ["tools"]
        doc["requires"]["handlers"] = list(_FIRST_FOUR)
        doc["body"]["handlers"] = doc["body"]["handlers"][:4]
        doc["body"]["variants"].pop("approval")
    return doc


def _envelope(doc: dict[str, Any]) -> PatternEnvelope:
    envelope = parse_executable(doc)
    assert isinstance(envelope, PatternEnvelope)
    return envelope


def _callables(**named: Any) -> NameRegistry[Any]:
    registry: NameRegistry[Any] = NameRegistry("callables")
    for name, value in named.items():
        registry.register(name, value)
    return registry


class TestReactIsAPatternAgent:
    def test_the_hierarchy_and_the_names_callers_and_tests_patch(self):
        assert ReactAgent.__mro__[:3] == (ReactAgent, PatternAgent, BaseAgent)
        for name in ("run", "run_stream", "_stream_via_run", "__init__"):
            assert name in vars(ReactAgent), name
        agent = _react()
        assert isinstance(agent, PatternAgent) and isinstance(agent, BaseAgent)
        assert agent.use_classification is False
        assert accepts_tools(ReactAgent) and accepts_tools(PatternAgent)
        assert type(create_agent("react", runs._tools())) is ReactAgent

    def test_an_empty_registry_is_refused_before_anything_else(self):
        with pytest.raises(AgentError, match="empty tool registry"):
            ReactAgent(runs._tools().__class__())

    def test_the_fsm_of_a_run_is_the_file_instantiated_and_the_builder_is_the_same_call(
        self, monkeypatch
    ):
        agent = _react()
        calls: list[Any] = []
        real = pi.instantiate_body

        def spy(*a: Any, **k: Any) -> Any:
            calls.append(k.get("variants"))
            return real(*a, **k)

        monkeypatch.setattr(pi, "instantiate_body", spy)

        fsm = agent._fsm_for_run("a task", {"hint": "x"})

        assert fsm == build_react_fsm(
            agent.tools, "a task", False, False, None, ("hint",)
        )
        assert calls == [
            [],
            [],
        ]  # both went through the file, neither selected a variant

    def test_the_registration_is_what_the_file_binds_in_order(self):
        agent = _react(hitl=runs._hitl())
        api = _FakeApi()

        agent._register_handlers(api, agent._handler_context())

        assert [h.name for h in api.handlers] == [
            "AgentToolExecutor",
            "AgentIterationLimiter",
            "AgentThinkFreshKeys",
            "AgentFeedbackConsumed",
            "AgentApprovalFreshKeys",
            "AgentHITLGate",
        ]

    def test_register_handlers_needs_the_call_local_context(self):
        with pytest.raises(AgentError, match=r"ReactAgent\._register_handlers called"):
            _react()._register_handlers(_FakeApi(), None)


class TestApprovalIsTheAgentsChoiceNeverAnOption:
    def test_the_state_is_present_exactly_under_the_agents_predicate(self):
        assert "await_approval" not in _react()._fsm_for_run("t", None)["states"]
        gated = _react(hitl=runs._hitl())
        assert "await_approval" in gated._fsm_for_run("t", None)["states"]

    @pytest.mark.parametrize("given", [True, False])
    def test_an_option_named_approval_is_refused_and_never_selects_the_variant(
        self, given
    ):
        envelope = pf.load_envelope("react")

        with pytest.raises(AgentError, match="approval"):
            PatternAgent(
                envelope,
                tools=runs._tools(),
                config=runs._config(),
                llm_interface=runs.ScriptedModel(),
                options={"approval": given},
            )
        with pytest.raises(AgentError, match="approval"):
            pi.pattern_fsm(envelope, tools=runs._tools(), options={"approval": given})

    def test_a_file_that_declares_an_option_named_approval_is_refused(self):
        doc = _user_doc(gate=True)
        doc["body"]["options_schema"]["properties"]["approval"] = {"type": "boolean"}
        doc["body"]["options"]["approval"] = False

        with pytest.raises(AgentError, match="approval"):
            pi.pattern_fsm(_envelope(doc), tools=runs._tools())

    def test_a_boolean_option_named_like_a_variant_selects_that_one(self):
        agent = _react(use_classification=True)

        fsm = agent._fsm_for_run("t", None)
        think = fsm["states"]["think"]

        assert [f["field_name"] for f in think["field_extractions"]] == [
            "tool_input",
            "should_terminate",
        ]
        assert think["classification_extractions"][0]["field_name"] == "tool_name"
        assert "await_approval" not in fsm["states"]

    @pytest.mark.parametrize(
        "given",
        [
            {"context_keys": ["x"]},
            {"schema_field_names": ["x"]},
            {"tool_intents": [{"name": "a", "description": "b"}]},
        ],
    )
    def test_an_input_the_agent_derives_cannot_be_given(self, given):
        with pytest.raises(AgentError, match="filled by the agent"):
            PatternAgent(
                pf.load_envelope("react"),
                tools=runs._tools(),
                config=runs._config(),
                llm_interface=runs.ScriptedModel(),
                options=given,
            )

    def test_an_unknown_option_is_refused_at_the_first_instantiation(self):
        with pytest.raises(Exception, match="unknown option"):
            pi.pattern_fsm(
                pf.load_envelope("react"), tools=runs._tools(), options={"nope": 1}
            )


class TestFailClosed:
    def test_a_flagged_tool_and_a_file_with_no_gate_is_refused_at_creation(self):
        with pytest.raises(AgentError, match="no human-in-the-loop approval"):
            PatternAgent(
                _envelope(_user_doc(gate=False)),
                tools=runs._tools(flagged=True),
                config=runs._config(),
                llm_interface=runs.ScriptedModel(),
            )

    def test_and_at_run_start_when_the_registry_gains_one_later(self):
        model = runs.ScriptedModel()
        tools = runs._tools()
        agent = PatternAgent(
            _envelope(_user_doc(gate=False)),
            tools=tools,
            config=runs._config(),
            llm_interface=model,
        )
        tools.register_function(
            lambda p: "x", name="danger", description="d", requires_approval=True
        )

        with pytest.raises(AgentError, match="no human-in-the-loop approval"):
            agent.run("task")
        assert model.log == []

    def test_a_flagged_tool_with_a_gate_and_nobody_to_approve_is_refused(self):
        with pytest.raises(AgentError, match="nobody can approve"):
            _react(flagged=True)

    def test_and_at_run_start_for_a_tool_added_after_construction(self):
        model = runs.ScriptedModel()
        agent = _react(llm=model)
        agent.tools.register_function(
            lambda p: "x", name="danger", description="d", requires_approval=True
        )

        with pytest.raises(AgentError, match="nobody can approve"):
            agent.run("task")
        assert model.log == []

    def test_a_hitl_given_to_a_file_with_no_gate_is_refused(self):
        with pytest.raises(AgentError, match="could never be consulted"):
            PatternAgent(
                _envelope(_user_doc(gate=False)),
                tools=runs._tools(),
                config=runs._config(),
                hitl=runs._hitl(),
                llm_interface=runs.ScriptedModel(),
            )

    @pytest.mark.parametrize(
        "field, value",
        [
            ("step_budget", 5),
            ("hooks", [{"hook": "agents.x", "at": "after_run"}]),
            ("run_output_keys", ["answer"]),
            ("seconds_exempt", ["conclude"]),
        ],
    )
    def test_a_run_field_with_no_seam_is_refused_not_ignored(self, field, value):
        doc = _user_doc(gate=False)
        doc["body"]["run"][field] = value

        with pytest.raises(AgentError, match=field):
            PatternAgent(
                _envelope(doc),
                tools=runs._tools(),
                config=runs._config(),
                llm_interface=runs.ScriptedModel(),
            )

    def test_a_generator_body_is_refused(self):
        doc = _user_doc(gate=False)
        doc["body"].pop("fsm")
        doc["body"].pop("variants")
        doc["body"]["generator"] = "agents.nothing"

        with pytest.raises(AgentError, match="generator"):
            PatternAgent(_envelope(doc), tools=runs._tools(), config=runs._config())


class TestAUserFileRunsAToolLoopEndToEnd:
    """Roadmap E4.1 done-when: a file that binds only library handlers runs a tool loop."""

    def test_through_the_registry_the_tool_runs_and_the_run_is_ok(self):
        executions: list[str] = []
        registry = ExecutableRegistry()
        executable = registry.add_document(_user_doc(gate=False))
        model = runs.ScriptedModel()
        seen = _counting_registry(executions)

        result = executable.run(
            {"task": "find it"},
            context=RunContext(
                llm_interface=model,
                callables=_callables(tools=seen, config=runs._config()),
            ),
        )

        assert result.status is Status.OK
        assert executions and set(executions) == {"search"}
        assert result.detail["pattern"] == "user.loop"
        assert result.detail["tools_used"] == ["search"]
        assert result.identity.instance_identity is None

    def test_the_executable_is_the_document_not_a_class(self):
        registry = ExecutableRegistry()
        doc = _user_doc(gate=False)

        executable = registry.add_document(doc)

        assert executable.kind == "pattern" and executable.id == "user.loop"
        assert (
            executable.template_identity
            == identity_of(_envelope(doc)).template_identity
        )
        assert list(executable.requires.callables) == ["tools"]
        assert registry.get("user.loop") is executable

    def test_the_tools_callable_is_required_before_any_model_call(self):
        executable = ExecutableRegistry().add_document(_user_doc(gate=False))
        model = runs.ScriptedModel()

        with pytest.raises(ExecutableError, match="tools"):
            executable.run({"task": "t"}, context=RunContext(llm_interface=model))
        assert model.log == []

    def test_a_flagged_tool_through_the_executable_is_refused_with_no_model_call(self):
        executable = ExecutableRegistry().add_document(_user_doc(gate=False))
        model = runs.ScriptedModel()

        with pytest.raises(AgentError, match="no human-in-the-loop approval"):
            executable.run(
                {"task": "t"},
                context=RunContext(
                    llm_interface=model,
                    callables=_callables(tools=runs._tools(flagged=True)),
                ),
            )
        assert model.log == []


def _counting_registry(sink: list[str]) -> Any:
    """A registry like ``runs._tools()`` whose one tool records each execution."""
    from fsm_llm.tools import ToolRegistry

    registry = ToolRegistry()

    def search(params: dict[str, Any]) -> str:
        sink.append("search")
        return "found"

    registry.register_function(
        search,
        name="search",
        description="Search",
        parameter_schema={"properties": {"x": {"type": "string"}}},
    )
    return registry


class TestTheReactExecutableIsTheFile:
    def _exe(self):
        registry = ExecutableRegistry()
        registry.load_sources()
        return registry.get("react")

    def test_the_row_keeps_its_id_and_its_header_and_gains_the_documents_identity(self):
        executable = self._exe()
        envelope = pf.load_envelope("react")

        assert executable.template_identity == identity_of(envelope).template_identity
        assert executable.kind == "pattern" and executable.id == "react"
        assert executable.requires.tools == "some"
        assert list(executable.requires.callables) == ["tools"]
        assert [c.value for c in executable.capabilities] == ["tools", "hitl"]
        assert executable.optional_callables == ("hitl", "config")
        assert executable.lifecycle.value == "builtin"

    def test_the_template_identity_ignores_the_release_and_any_caller(
        self, monkeypatch
    ):
        from fsm_llm.engine.executables import code_backed

        before = self._exe().template_identity
        monkeypatch.setattr(code_backed, "_FSM_LLM_VERSION", "99.0.0")

        assert self._exe().template_identity == before

    def test_it_is_built_from_the_packaged_class_so_a_patched_run_intercepts(
        self, monkeypatch
    ):
        seen: list[str] = []

        def fake_run(self: Any, task: str, initial_context: Any = None) -> Any:
            seen.append(task)
            return AgentResult(
                answer="patched",
                trace=AgentTrace(),
                final_context={},
                stop_reason="answered",
            )

        monkeypatch.setattr(ReactAgent, "run", fake_run)

        result = self._exe().run(
            {"task": "go"},
            context=RunContext(
                llm_interface=runs.ScriptedModel(),
                callables=_callables(tools=runs._tools()),
            ),
        )

        assert seen == ["go"] and result.answer == "patched"


class TestNothingPerRunIsKept:
    def test_two_calls_build_two_contexts_with_two_executors(self):
        agent = _react()
        first = agent._run_inputs("a", None)[2]
        second = agent._run_inputs("a", None)[2]
        for ctx in (first, second):
            agent._register_handlers(_FakeApi(), ctx)

        assert isinstance(first, HandlerContext) and first is not second
        assert first.scope is not second.scope
        assert first.scope["agent_handlers"] is not second.scope["agent_handlers"]

    def test_the_agent_holds_no_handler_object_after_a_run(self):
        agent = _react()
        agent.run("find it")

        held = vars(agent)

        assert "_handlers" not in held
        assert not any(isinstance(v, HandlerContext) for v in held.values())
        assert not any(type(v).__name__ == "AgentHandlers" for v in held.values())

    def test_the_context_is_built_under_the_agents_own_hitl_and_predicate(self):
        hitl = runs._hitl()
        agent = _react(hitl=hitl)

        ctx = agent._handler_context()

        assert ctx.hitl is hitl and ctx.hitl_active is True
        assert ctx.approval_predicate is not None
        assert _react()._handler_context().approval_predicate is None

    def test_the_shared_envelope_is_never_changed_by_a_run(self):
        envelope = pf.load_envelope("react")
        before = copy.deepcopy(envelope.model_dump(mode="json"))

        _react(hitl=runs._hitl(), use_classification=True).run("find it")

        assert envelope.model_dump(mode="json") == before
