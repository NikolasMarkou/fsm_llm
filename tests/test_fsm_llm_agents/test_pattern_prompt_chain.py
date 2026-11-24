"""The file-backed ``prompt_chain``: a generator pattern (plan a555dcfe step 11, D-019).

``PromptChainAgent`` is thin on ``PatternAgent``: ``patterns/prompt_chain.json`` declares the
generator ``agents.prompt_chain``, which writes one state per step and a gate edge after a gated
step; the gates are the library factory ``agents.chain_gate_checker`` and the limiter is
``agents.iteration_limiter`` bound with the chain's budget. Census rows: D5 (the FSM, the
registration order and the request bytes are the builder's: the differential of step 11 is recorded
in progress.md, the goldens hold the bytes), D6 (a generator is operator code: no model-written
document names one), D1 (a chain has no HITL path and no tool).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from fsm_llm.agents import PromptChainAgent, create_agent
from fsm_llm.agents.base import BaseAgent, accepts_tools
from fsm_llm.agents.constants import ContextKeys, StopReason
from fsm_llm.agents.definitions import ChainStep
from fsm_llm.agents.fsm_definitions import build_prompt_chain_fsm
from fsm_llm.agents.fsm_definitions.prompt_chain import chain_options, prompt_chain_fsm
from fsm_llm.agents.handlers import library_react as lib
from fsm_llm.agents.handlers.context import HandlerContext
from fsm_llm.agents.pattern_agent import PatternAgent
from fsm_llm.agents.pattern_files import load_document, load_envelope
from fsm_llm.core.definitions import AgentError
from fsm_llm.core.definitions.executable import InstantiationError
from fsm_llm.core.verify.executable import load_executable
from fsm_llm.engine.executables import ExecutableError, ExecutableRegistry
from fsm_llm.engine.executables.namespaces import GENERATORS, namespace_resolver
from tests import pattern_runs as runs


class _FakeApi:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def register_handler(self, handler: Any) -> None:
        self.handlers.append(handler)


def _step(n: int, gate: Any = None) -> ChainStep:
    return ChainStep(
        step_id=f"s{n}",
        name=f"Step {n}",
        extraction_instructions=f"extract {n} {{x}} é",
        response_instructions=f"respond {n}",
        validation_fn=gate,
    )


def _agent(chain: list[ChainStep], **kw: Any) -> PromptChainAgent:
    return PromptChainAgent(
        chain, config=runs._config(), llm_interface=kw.pop("llm", runs.ScriptedModel())
    )


def _ctx(chain: list[ChainStep], **kw: Any) -> HandlerContext:
    return HandlerContext(
        options=chain_options(chain),
        callables={"validation_fns": [s.validation_fn for s in chain]},
        **kw,
    )


class TestPromptChainIsAPatternAgent:
    def test_the_hierarchy_the_names_and_the_attributes_callers_read(self):
        assert PromptChainAgent.__mro__[:3] == (
            PromptChainAgent,
            PatternAgent,
            BaseAgent,
        )
        assert "run" not in vars(PromptChainAgent)
        chain = [_step(0), _step(1)]
        agent = _agent(chain)
        assert agent.chain == chain and agent.chain is not chain
        assert not accepts_tools(PromptChainAgent)
        assert type(create_agent("prompt_chain", chain=chain)) is PromptChainAgent

    def test_the_fsm_of_a_run_is_the_generator_output_for_the_chain(self):
        chain = [_step(0, lambda c: True), _step(1), _step(2)]
        agent = _agent(chain)

        fsm = agent._fsm_for_run("a task", {"hint": "x"})

        assert (
            fsm
            == build_prompt_chain_fsm(chain)
            == prompt_chain_fsm(chain_options(chain))
        )
        assert list(fsm["states"]) == ["step_0", "step_1", "step_2", "output"]
        edges = [t["priority"] for t in fsm["states"]["step_1"]["transitions"]]
        assert edges == [50, 100]  # the gate edge of step 0, then the way on

    def test_the_task_never_enters_definition_text(self):
        agent = _agent([_step(0)])

        assert agent._fsm_for_run("alpha task", None) == agent._fsm_for_run(
            "a very different task", None
        )

    def test_a_step_prompt_is_text_as_written(self):
        extraction = "extract {{x}} \u00e9 {{task}}"
        response = "respond {{y}} \u4e2d {{ z }}"
        fsm = build_prompt_chain_fsm(
            [
                ChainStep(
                    step_id="s",
                    name="{{n}} name",
                    extraction_instructions=extraction,
                    response_instructions=response,
                )
            ]
        )

        state = fsm["states"]["step_0"]
        assert state["response_instructions"] == response
        assert state["purpose"] == "{{n}} name"
        assert extraction in state["field_extractions"][0]["extraction_instructions"]
        assert response in state["field_extractions"][0]["extraction_instructions"]

    def test_the_key_order_of_the_fsm_is_the_builders(self):
        fsm = build_prompt_chain_fsm([_step(0, lambda c: True), _step(1)])

        assert list(fsm) == [
            "name",
            "description",
            "initial_state",
            "persona",
            "states",
            "handler_only_keys",
        ]
        assert list(fsm["states"]["step_1"]) == [
            "id",
            "description",
            "purpose",
            "extraction_instructions",
            "field_extractions",
            "response_instructions",
            "transitions",
        ]
        assert list(fsm["states"]["step_1"]["transitions"][0]) == [
            "target_state",
            "description",
            "priority",
            "conditions",
        ]

    def test_the_registration_is_the_gates_then_the_limiter(self):
        agent = _agent([_step(0, lambda c: True), _step(1), _step(2)])
        api = _FakeApi()

        agent._register_handlers(api, agent._handler_context())

        assert [h.name for h in api.handlers] == [
            "ChainGateChecker_1",
            "ChainGateChecker_2",
            "ChainGateChecker_final",
            "AgentIterationLimiter",
        ]

    def test_the_step_budget_is_three_per_step_for_run_and_stream(self, monkeypatch):
        seen: list[dict[str, Any]] = []

        def capture(self: Any, *args: Any, **kwargs: Any) -> Any:
            seen.append(kwargs)
            return iter(())

        monkeypatch.setattr(BaseAgent, "_standard_run_stream", capture)
        monkeypatch.setattr(
            BaseAgent, "_standard_run", lambda self, *a, **k: seen.append(k) or "r"
        )
        agent = _agent([_step(0), _step(1), _step(2), _step(3)])

        agent.run("t")
        list(agent.run_stream("t"))

        assert [k["max_iterations"] for k in seen] == [12, 12]
        assert seen[0]["success_keys"] == [ContextKeys.CHAIN_STEP_RESULT]

    def test_the_seed_and_the_run_outputs_are_the_chains(self):
        agent = _agent([_step(0)])

        _, context, handlers = agent._run_inputs("t", {"hint": "x"})

        assert context[ContextKeys.CHAIN_RESULTS] == []
        assert context[ContextKeys.CHAIN_STEP_INDEX] == 0
        assert ContextKeys.CHAIN_STEP_RESULT in PromptChainAgent._run_output_keys
        assert handlers.scope == {} and "_handlers" not in vars(agent)

    def test_two_runs_share_neither_the_context_nor_the_seeded_results(self):
        agent = _agent([_step(0)])
        _, first, ctx_one = agent._run_inputs("a", None)
        first[ContextKeys.CHAIN_RESULTS].append("kept?")
        _, second, ctx_two = agent._run_inputs("a", None)

        assert second[ContextKeys.CHAIN_RESULTS] == []
        assert ctx_one is not ctx_two and ctx_one.scope is not ctx_two.scope

    def test_an_empty_chain_is_refused_before_the_file_is_read(self):
        with pytest.raises(AgentError, match="empty"):
            PromptChainAgent([])

    def test_a_chain_has_no_tool_and_no_hitl_path(self):
        env = load_envelope("prompt_chain")

        assert env.requires.tools == "none"
        assert {b.handler for b in env.body.handlers} == {
            "agents.chain_gate_checker",
            "agents.iteration_limiter",
        }
        for refused in ({"hitl": runs._hitl()}, {"tools": runs._tools()}):
            with pytest.raises(TypeError, match="does not accept"):
                PatternAgent(env, options=chain_options([_step(0)]), **refused)
        with pytest.raises(TypeError, match="does not accept"):
            PromptChainAgent([_step(0)], tools=runs._tools())
        with pytest.raises(TypeError, match="does not take tools"):
            create_agent("prompt_chain", runs._tools(), chain=[_step(0)])


class TestTheGateFactory:
    def test_one_entry_handler_per_state_after_a_step_in_order(self):
        built = lib.chain_gate_checker(_ctx([_step(0), _step(1), _step(2)]))

        assert [h.name for h in built] == [
            "ChainGateChecker_1",
            "ChainGateChecker_2",
            "ChainGateChecker_final",
        ]
        assert [h.target_states for h in built] == [{"step_1"}, {"step_2"}, {"output"}]

    def test_a_gated_step_runs_its_function_on_entry_to_the_next_state(self):
        verdicts = iter([False])
        built = lib.chain_gate_checker(
            _ctx([_step(0, lambda c: next(verdicts)), _step(1)])
        )

        failed = built[0].execute({ContextKeys.CHAIN_STEP_RESULT: "r"})

        assert failed[ContextKeys.GATE_PASSED] is False
        assert failed[ContextKeys.FORCED_STOP_REASON] == StopReason.GATE_FAILED.value
        assert failed[ContextKeys.CHAIN_RESULTS] == ["r"]

    def test_a_gated_step_with_no_function_is_refused_not_skipped(self):
        context = HandlerContext(
            options=chain_options([_step(0, lambda c: True), _step(1)]),
            callables={"validation_fns": [None, None]},
        )

        with pytest.raises(AgentError, match="no validation function"):
            lib.chain_gate_checker(context)
        with pytest.raises(AgentError, match="no validation function"):
            lib.chain_gate_checker(
                HandlerContext(options=chain_options([_step(0, lambda c: True)]))
            )

    def test_the_topology_decides_a_function_of_an_ungated_step_is_ignored(self):
        options = chain_options([_step(0), _step(1)])
        context = HandlerContext(
            options=options,
            callables={"validation_fns": [lambda c: False, None]},
        )

        built = lib.chain_gate_checker(context)

        assert ContextKeys.GATE_PASSED not in built[0].execute({})

    def test_a_binding_moves_the_priority_never_the_states(self):
        named = SimpleNamespace(states=("step_9",), keys=(), priority=42, params={})

        built = lib.chain_gate_checker(_ctx([_step(0), _step(1)]), named)

        assert {h.priority for h in built} == {42}
        assert [h.target_states for h in built] == [{"step_1"}, {"output"}]

    def test_two_calls_build_independent_handlers(self):
        one = lib.chain_gate_checker(_ctx([_step(0), _step(1)]))
        two = lib.chain_gate_checker(_ctx([_step(0), _step(1)]))

        assert {id(h) for h in one}.isdisjoint({id(h) for h in two})


class TestTheGeneratorAndItsCaps:
    def test_the_generator_is_registered_once_under_its_name(self):
        assert GENERATORS.resolve("agents.prompt_chain") is prompt_chain_fsm
        assert GENERATORS.name_of(prompt_chain_fsm) == "agents.prompt_chain"
        assert namespace_resolver()("generators", "agents.prompt_chain")

    def test_the_shipped_file_loads_with_its_default_instance_checked(self):
        envelope, report = load_executable(
            load_document("prompt_chain"),
            resolver=namespace_resolver(),
            generators=GENERATORS.get,
        )

        assert envelope is not None and report.is_valid, report.errors
        assert envelope.body.generator == "agents.prompt_chain"
        assert not any("nothing to check" in line for line in report.info)

    def test_a_generator_pattern_in_the_file_is_never_a_models_document(self):
        envelope, report = load_executable(
            load_document("prompt_chain"),
            origin="model",
            resolver=namespace_resolver(),
            generators=GENERATORS.get,
        )

        assert envelope is None
        assert any("generator" in line for line in report.errors)

    def test_a_step_over_the_text_cap_and_a_chain_over_the_step_cap_are_refused(self):
        env = load_envelope("prompt_chain")
        long_step = ChainStep(
            step_id="s",
            name="n",
            extraction_instructions="e" * 10_001,
            response_instructions="r",
        )

        with pytest.raises(InstantiationError):
            build_prompt_chain_fsm([long_step])
        with pytest.raises(InstantiationError):
            build_prompt_chain_fsm([_step(i) for i in range(257)])
        assert env.body.options["max_iterations"] == 6

    def test_the_registry_runs_the_generator_on_a_user_documents_default_options(self):
        doc = load_document("prompt_chain")
        doc["id"] = "user.chain"
        doc["body"]["options"]["steps"] = []  # the generator refuses an empty chain
        registry = ExecutableRegistry()

        with pytest.raises(ExecutableError, match="default instance cannot be built"):
            registry.add_document(doc)

    def test_a_generator_unknown_to_the_registry_refuses_the_document(self):
        doc = load_document("prompt_chain")
        doc["id"] = "user.chain"
        doc["body"]["generator"] = "user.nothing"
        registry = ExecutableRegistry()

        with pytest.raises(ExecutableError, match="does not resolve"):
            registry.add_document(doc)


class TestTheChainExecutableIsTheFile:
    def _exe(self):
        registry = ExecutableRegistry()
        registry.load_sources()
        return registry.get("prompt_chain")

    def test_the_row_keeps_its_id_its_objects_and_its_class(self):
        executable = self._exe()

        assert executable.template_identity
        assert list(executable.requires.callables) == ["chain"]
        assert executable.optional_callables == ("config",)
        assert executable.impl.endswith("PromptChainAgent")
        assert executable.envelope.body.generator == "agents.prompt_chain"

    def test_the_template_identity_reads_no_chain(self):
        first = self._exe().template_identity
        again = self._exe().template_identity

        assert first == again
        assert chain_options([_step(0)]) != chain_options([_step(0), _step(1)])
