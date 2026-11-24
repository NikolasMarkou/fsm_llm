"""The handler library of the agents package (plan a555dcfe step 7, D-013).

The ``handlers`` and ``generators`` namespaces, ``HandlerContext`` and the react-family factories.
The factories are held against what ``ReactAgent._register_handlers`` registers TODAY (the golden of
step 8 does not exist yet): equal (name, timing, states, target states, watched keys, priority)
tuples in the same order, and equal behavior of the handler functions. These rows guard the census
invariants D1 (the executor is built under the one approval predicate, the gate is not movable by a
file) and D2 (a key a file names is checked), and the per-run state rule (D-014 of plan 089d0ec7).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from fsm_llm.agents.base.registration import RegistrationMixin
from fsm_llm.agents.constants import (
    REACT_THINK_FRESH_KEYS,
    AgentStates,
    ContextKeys,
    HandlerNames,
)
from fsm_llm.agents.definitions import AgentConfig
from fsm_llm.agents.handlers import AgentHandlers, library, make_iteration_limiter
from fsm_llm.agents.handlers import library_react as lib
from fsm_llm.agents.handlers.context import HandlerContext
from fsm_llm.agents.handlers.library import LIBRARY
from fsm_llm.agents.react import ReactAgent
from fsm_llm.core.definitions import AgentError
from fsm_llm.core.names import NameCollisionError, NameRegistry
from fsm_llm.core.verify.executable import load_executable
from fsm_llm.engine.executables.namespaces import (
    GENERATORS,
    HANDLERS,
    namespace_resolver,
)
from fsm_llm.handlers import create_handler
from fsm_llm.policy import HumanInTheLoop
from fsm_llm.tools import ToolRegistry
from tests.test_fsm_llm import kind_corpus as kc

_DRIVER = "_approval_granted"
_INPUT = {"x": "1"}


class _FakeApi:
    """Records what a registration helper registers, in order."""

    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def create_handler(self, name: str):
        return create_handler(name)

    def register_handler(self, handler: Any) -> None:
        self.handlers.append(handler)


def _signature(handler: Any) -> tuple[Any, ...]:
    return (
        handler.name,
        tuple(sorted(t.value for t in (handler.timings or ()))),
        tuple(sorted(handler.states)),
        tuple(sorted(handler.target_states)),
        tuple(sorted(handler.updated_keys)),
        handler.priority,
    )


def _registry(executions: list[str] | None = None, *, flagged: bool = False):
    seen = executions if executions is not None else []
    registry = ToolRegistry()

    def search(params: dict[str, Any]) -> str:
        seen.append("search")
        return "found"

    registry.register_function(
        search,
        name="search",
        description="Search",
        parameter_schema={"properties": {"x": {"type": "string"}}},
        requires_approval=flagged,
    )
    return registry


def _hitl(*, policy: bool = True, callback: bool = True) -> HumanInTheLoop:
    return HumanInTheLoop(
        approval_policy=(lambda call, ctx: True) if policy else None,
        approval_callback=(lambda request: True) if callback else None,
    )


_CASES = {
    "no_hitl": lambda: ReactAgent(_registry(), config=_config()),
    "policy_and_callback": lambda: ReactAgent(
        _registry(), config=_config(), hitl=_hitl()
    ),
    "callback_flagged_tool": lambda: ReactAgent(
        _registry(flagged=True), config=_config(), hitl=_hitl(policy=False)
    ),
    "callback_no_flagged_tool": lambda: ReactAgent(
        _registry(), config=_config(), hitl=_hitl(policy=False)
    ),
    "policy_no_callback": lambda: ReactAgent(
        _registry(), config=_config(), hitl=_hitl(callback=False)
    ),
}


def _config() -> AgentConfig:
    return AgentConfig(max_iterations=5, model="mock/model")


def _context_of(agent: ReactAgent) -> HandlerContext:
    return HandlerContext(
        tools=agent.tools,
        config=agent.config,
        hitl=agent.hitl,
        hitl_active=agent._hitl_active,
        approval_predicate=agent._approval_predicate,
    )


def _live(agent: ReactAgent) -> list[Any]:
    """What ``ReactAgent._register_handlers`` registered before step 9 (plan a555dcfe): the
    registration helpers of ``RegistrationMixin`` and ``HitlGateMixin`` the other agents still use."""
    api = _FakeApi()
    handlers = agent._call_handlers(agent.tools)
    agent._register_tool_executor(api, AgentStates.ACT, handlers.execute_tool)
    agent._register_iteration_limiter(api, handlers.check_iteration_limit)
    agent._register_think_loop_handlers(api, REACT_THINK_FRESH_KEYS)
    agent._register_approval_gate(api)
    return api.handlers


def _library(ctx: HandlerContext) -> list[Any]:
    built: list[Any] = []
    for name in (
        "agents.tool_executor",
        "agents.think_turn_limiter",
        "agents.think_fresh_keys",
        "agents.feedback_consumed",
        "agents.approval_fresh_keys",
        "agents.approval_gate",
    ):
        built.extend(HANDLERS.resolve(name)(ctx, None))
    return built


class TestTheNamespaces:
    def test_the_two_namespaces_exist_with_their_entry_point_groups(self):
        assert HANDLERS.namespace == "handlers"
        assert GENERATORS.namespace == "generators"
        assert HANDLERS._group == "fsm_llm.handlers"
        assert GENERATORS._group == "fsm_llm.generators"

    def test_the_library_is_registered_one_function_per_name(self):
        names = [name for name, _ in LIBRARY]
        assert sorted([*names, *(name for name, _ in library.HOOKS)]) == HANDLERS.list()
        assert names == [
            "agents.tool_executor",
            "agents.think_turn_limiter",
            "agents.iteration_limiter",
            "agents.think_fresh_keys",
            "agents.feedback_consumed",
            "agents.approval_fresh_keys",
            "agents.reflexion_evaluator",
            "agents.reflexion_fresh_keys",
            "agents.reflexion_reflector",
            "agents.chain_gate_checker",
            "agents.native_fc",
            "agents.orchestrator_delegator",
            "agents.orchestrator_decision_reset",
            "agents.rewoo_executor",
            "agents.plan_step_executor",
            "agents.plan_step_fresh_keys",
            "agents.plan_step_checker",
            "agents.plan_replan_counter",
            "agents.revision_count",
            "agents.maker_checker_checker",
            "agents.revise_entry",
            "agents.revise_exit",
            "agents.force_pass",
            "agents.debate_judge",
            "agents.debate_consensus_reset",
            "agents.debate_limiter",
            "agents.evalopt_evaluator",
            "agents.refine_entry",
            "agents.refine_exit",
            "agents.approval_gate",
        ]
        assert [name for name, _ in library.HOOKS] == [
            "agents.raise_run_ending_error",
            "agents.rewoo_cut_off_answer",
            "agents.verify_retry",
            "agents.periodic_reflection",
            "agents.memory_recall",
            "agents.memory_remember",
        ]
        for name, factory in (*LIBRARY, *library.HOOKS):
            assert HANDLERS.resolve(name) is factory
            assert HANDLERS.name_of(factory) == name
        assert len({id(factory) for _, factory in (*LIBRARY, *library.HOOKS)}) == len(
            LIBRARY
        ) + len(library.HOOKS)

    def test_the_gate_name_is_the_one_the_verifier_holds_equal_to_the_hitl_claim(self):
        # D-012 of plan a555dcfe: core names the gate by a constant, the library registers it
        from fsm_llm.agents.constants import AgentStates
        from fsm_llm.core.definitions.executable_pattern import (
            APPROVAL_GATE_HANDLER,
            APPROVAL_STATE,
        )

        assert HANDLERS.resolve(APPROVAL_GATE_HANDLER) is lib.approval_gate
        assert APPROVAL_STATE == AgentStates.AWAIT_APPROVAL

    def test_a_function_under_two_names_is_refused(self):
        with pytest.raises(NameCollisionError, match="already registered as"):
            HANDLERS.register("agents.alias", lib.tool_executor)
        assert "agents.alias" not in HANDLERS
        # the same refusal on a private registry, whatever override says
        private: NameRegistry[Any] = NameRegistry("handlers")
        private.register("agents.a", lib.approval_gate)
        with pytest.raises(NameCollisionError):
            private.register("agents.b", lib.approval_gate, override=True)

    def test_the_resolver_answers_handlers_and_generators_from_the_namespaces(self):
        resolve = namespace_resolver()
        assert resolve("handlers", "agents.tool_executor")
        assert not resolve("handlers", "agents.nothing")
        assert resolve("generators", "agents.prompt_chain")  # step 11 (D-019)
        assert not resolve("generators", "agents.nothing")
        # not judged here: tools, callables and executables answer True
        assert resolve("tools", "anything") and resolve("executables", "anything")

    def test_a_private_namespace_can_stand_in(self):
        private: NameRegistry[Any] = NameRegistry("handlers")
        private.register("mine.only", lib.approval_gate)
        resolve = namespace_resolver(handlers=private)
        assert resolve("handlers", "mine.only")
        assert not resolve("handlers", "agents.tool_executor")


def _file(handlers: list[str]) -> dict[str, Any]:
    fsm = kc._fsm(
        "Lib",
        "start",
        kc._state("start", transitions=[kc._go("done")]),
        kc._state("done"),
    )
    return {
        "schema": "fsm-llm/executable",
        "kind": "pattern",
        "id": "demo/lib",
        "body": {
            "fsm": fsm,
            "handlers": [
                {"handler": name, "on": "entry", "states": ["start"]}
                for name in handlers
            ],
        },
    }


class TestAPatternFileNamingTheLibraryVerifies:
    def test_a_file_binding_library_handlers_loads_through_the_namespace_resolver(self):
        doc = _file(["agents.tool_executor", "agents.think_turn_limiter"])
        envelope, result = load_executable(doc, resolver=namespace_resolver())
        assert envelope is not None and result.is_valid, result.errors

    def test_an_unknown_handler_name_is_refused(self):
        doc = _file(["agents.tool_executor", "agents.not_there"])
        envelope, result = load_executable(doc, resolver=namespace_resolver())
        assert envelope is None
        assert any(
            "agents.not_there" in m and "does not resolve" in m for m in result.errors
        )

    def test_the_registry_resolves_a_pattern_through_the_namespaces_by_default(self):
        from fsm_llm.engine.executables import ExecutableRegistry
        from fsm_llm.engine.executables.protocol import ExecutableError

        # a registry with no adapter for the kind (the agents package registers one, step 9)
        registry = ExecutableRegistry(kinds=NameRegistry("executable_kinds"))
        with pytest.raises(ExecutableError) as unknown:
            registry.add_document(_file(["agents.not_there"]))
        assert "does not resolve" in str(unknown.value)
        # a library name passes the names check; only the missing adapter refuses it
        with pytest.raises(ExecutableError) as known:
            registry.add_document(_file(["agents.tool_executor"]))
        assert "does not resolve" not in str(known.value)
        assert "adapter" in str(known.value)

    def test_an_ordinary_document_naming_handlers_is_not_judged_by_default(self):
        from fsm_llm.engine.executables import ExecutableRegistry
        from tests.test_fsm_llm.test_executable_adapters import _doc

        registry = ExecutableRegistry()
        executable = registry.add_document(
            _doc("fsm", "demo/ordinary", requires={"handlers": ["not.registered"]})
        )
        assert executable.requires.handlers == ["not.registered"]


class TestHandlerContext:
    def test_is_frozen_with_read_only_options(self):
        ctx = HandlerContext(options={"a": 1}, callables={"f": len})
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.tools = None  # type: ignore[misc]
        with pytest.raises(TypeError):
            ctx.options["b"] = 2  # type: ignore[index]
        with pytest.raises(TypeError):
            ctx.callables["g"] = len  # type: ignore[index]

    def test_scope_is_fresh_per_context_and_per_run(self):
        one, two = HandlerContext(), HandlerContext()
        assert one.scope is not two.scope
        one.scope["k"] = 1
        assert two.scope == {}
        again = one.next_run()
        assert again.scope == {} and again.scope is not one.scope
        assert again.options == one.options

    def test_the_predicate_and_the_active_flag_are_set_together(self):
        with pytest.raises(AgentError, match="exactly when hitl_active"):
            HandlerContext(hitl_active=True)
        with pytest.raises(AgentError, match="exactly when hitl_active"):
            HandlerContext(approval_predicate=lambda call, ctx: True)
        HandlerContext(hitl_active=True, approval_predicate=lambda call, ctx: True)
        HandlerContext()


class TestTheReactFamilyEqualsTheLiveRegistration:
    @pytest.mark.parametrize("case", sorted(_CASES))
    def test_same_handlers_in_the_same_order(self, case):
        agent = _CASES[case]()
        live = [_signature(h) for h in _live(agent)]
        library = [_signature(h) for h in _library(_context_of(agent))]
        assert library == live

    @pytest.mark.parametrize("case", sorted(_CASES))
    def test_registered_names_are_the_wire_names(self, case):
        agent = _CASES[case]()
        names = [h.name for h in _library(_context_of(agent))]
        expected = [
            HandlerNames.TOOL_EXECUTOR,
            HandlerNames.ITERATION_LIMITER,
            HandlerNames.THINK_FRESH_KEYS,
            HandlerNames.FEEDBACK_CONSUMED,
        ]
        if agent._hitl_active:
            expected.append(HandlerNames.APPROVAL_FRESH_KEYS)
        if agent._approval_predicate is not None:
            expected.append(HandlerNames.HITL_GATE)
        assert names == expected

    @pytest.mark.parametrize("case", sorted(_CASES))
    def test_the_handler_functions_behave_alike(self, case):
        agent = _CASES[case]()
        live = {h.name: h for h in _live(agent)}
        library = {h.name: h for h in _library(_context_of(agent))}
        contexts = [
            {ContextKeys.REASONING: "r", ContextKeys.SHOULD_TERMINATE: True},
            {
                ContextKeys.SHOULD_TERMINATE: True,
                ContextKeys.MAX_ITERATIONS_REACHED: True,
                ContextKeys.AGENT_FEEDBACK: "warn",
            },
            {ContextKeys.TOOL_NAME: "search", ContextKeys.TOOL_INPUT: dict(_INPUT)},
            {ContextKeys.TOOL_NAME: ContextKeys.NO_TOOL},
        ]
        for name in (
            "AgentThinkFreshKeys",
            "AgentFeedbackConsumed",
            "AgentApprovalFreshKeys",
            "AgentHITLGate",
        ):
            if name not in live:
                assert name not in library
                continue
            for context in contexts:
                assert library[name].execution_lambda(dict(context)) == live[
                    name
                ].execution_lambda(dict(context))

    def test_the_limiters_count_alike_over_a_run(self):
        agent = _CASES["no_hitl"]()
        live = next(h for h in _live(agent) if h.name == HandlerNames.ITERATION_LIMITER)
        built = lib.think_turn_limiter(_context_of(agent))[0]
        script = [
            {ContextKeys.CURRENT_STATE: state, ContextKeys.ITERATION_BUDGET: 3}
            for state in ("think", "act", "think", "act", "think", "act", "think")
        ]
        assert [built.execution_lambda(dict(c)) for c in script] == [
            live.execution_lambda(dict(c)) for c in script
        ]


class TestPerRunState:
    def test_the_executor_and_the_think_turn_limiter_share_one_instance_in_either_order(
        self,
    ):
        for order in (
            (lib.tool_executor, lib.think_turn_limiter),
            (lib.think_turn_limiter, lib.tool_executor),
        ):
            ctx = _context_of(_CASES["no_hitl"]())
            first, second = (factory(ctx)[0] for factory in order)
            assert first.execution_lambda.__self__ is second.execution_lambda.__self__
            assert isinstance(first.execution_lambda.__self__, AgentHandlers)
            assert ctx.scope[lib._SHARED] is first.execution_lambda.__self__

    def test_two_runs_of_one_agent_share_no_handler_state(self):
        agent = _CASES["no_hitl"]()
        one = _context_of(agent)
        two = one.next_run()
        a = lib.tool_executor(one)[0].execution_lambda.__self__
        b = lib.tool_executor(two)[0].execution_lambda.__self__
        assert a is not b and one.scope is not two.scope
        # counting in one run does not move the other (the D-014 race)
        lib.think_turn_limiter(one)[0].execution_lambda(
            {ContextKeys.CURRENT_STATE: "think"}
        )
        assert a._current_iteration == 1 and b._current_iteration == 0

    def test_a_factory_called_twice_builds_independent_handlers(self):
        agent = _CASES["policy_and_callback"]()
        for name, factory in LIBRARY:
            if name == "agents.iteration_limiter":
                continue  # needs a limit; its twin row is in TestTheContextCountingLimiter
            if name == "agents.chain_gate_checker":
                continue  # builds nothing without the chain options; test_pattern_prompt_chain has its row
            if name == "agents.evalopt_evaluator":
                continue  # needs the host's evaluation_fn; test_pattern_evaluator_optimizer has its row
            first = factory(_context_of(agent))
            second = factory(_context_of(agent))
            assert first and second, name
            assert {id(h) for h in first}.isdisjoint({id(h) for h in second}), name
            for a, b in zip(first, second, strict=True):
                owner_a = getattr(a.execution_lambda, "__self__", None)
                owner_b = getattr(b.execution_lambda, "__self__", None)
                assert owner_a is None or owner_a is not owner_b, name

    def test_the_library_module_caches_nothing_between_runs(self):
        agent = _CASES["no_hitl"]()
        before = {k: id(v) for k, v in vars(lib).items() if not k.startswith("__")}
        for _ in range(2):
            lib.tool_executor(_context_of(agent))
            lib.think_turn_limiter(_context_of(agent))
        after = {k: id(v) for k, v in vars(lib).items() if not k.startswith("__")}
        assert after == before
        for _, factory in HANDLERS.items():
            assert getattr(factory, "__closure__", None) is None  # a plain function

    def test_the_limiter_alone_is_a_plain_think_turn_limiter_with_no_registry(self):
        ctx = HandlerContext()  # no tools at all
        binding = _Binding(params={"alone": True})
        built = lib.think_turn_limiter(ctx, binding)[0]
        from fsm_llm.agents.handlers import ThinkTurnLimiter

        owner = built.execution_lambda.__self__
        assert type(owner) is ThinkTurnLimiter
        assert lib._SHARED not in ctx.scope
        # without `alone` the shared AgentHandlers needs the tools and refuses
        with pytest.raises(AgentError, match="needs the run's tools"):
            lib.think_turn_limiter(HandlerContext())

    def test_the_executor_needs_the_tools(self):
        with pytest.raises(AgentError, match="needs the run's tools"):
            lib.tool_executor(HandlerContext())


class _Binding:
    def __init__(self, **fields: Any) -> None:
        self.states = fields.get("states", [])
        self.keys = fields.get("keys")
        self.priority = fields.get("priority")
        self.params = fields.get("params", {})


class TestTheExecutorIsBuiltUnderTheOnePredicate:
    def test_a_gated_call_with_no_grant_is_refused_and_does_not_run(self):
        executions: list[str] = []
        ctx = HandlerContext(
            tools=_registry(executions),
            hitl=_hitl(),
            hitl_active=True,
            approval_predicate=lambda call, context: True,
        )
        handler = lib.tool_executor(ctx)[0]
        delta = handler.execution_lambda(
            {ContextKeys.TOOL_NAME: "search", ContextKeys.TOOL_INPUT: dict(_INPUT)}
        )
        assert executions == []
        assert delta[ContextKeys.TOOL_STATUS] == "awaiting_approval"

    def test_the_executor_holds_the_predicate_of_the_context(self):
        predicate = lambda call, context: True  # noqa: E731
        ctx = HandlerContext(
            tools=_registry(),
            hitl=_hitl(),
            hitl_active=True,
            approval_predicate=predicate,
        )
        owner = lib.tool_executor(ctx)[0].execution_lambda.__self__
        assert owner.requires_approval is predicate

    def test_a_run_with_no_predicate_runs_the_tool(self):
        executions: list[str] = []
        handler = lib.tool_executor(HandlerContext(tools=_registry(executions)))[0]
        handler.execution_lambda(
            {ContextKeys.TOOL_NAME: "search", ContextKeys.TOOL_INPUT: dict(_INPUT)}
        )
        assert executions == ["search"]

    def test_the_gate_and_the_state_handler_exist_only_for_an_active_run(self):
        plain = HandlerContext(tools=_registry())
        assert lib.approval_gate(plain) == [] and lib.approval_fresh_keys(plain) == []
        active = _context_of(_CASES["policy_and_callback"]())
        assert len(lib.approval_gate(active)) == 1
        assert len(lib.approval_fresh_keys(active)) == 1


class TestWhatABindingMayChange:
    def test_states_and_priority_are_overridable_where_safe(self):
        ctx = _context_of(_CASES["no_hitl"]())
        built = lib.tool_executor(ctx, _Binding(states=["do"], priority=7))[0]
        assert built.target_states == {"do"} and built.priority == 7
        fresh = lib.think_fresh_keys(ctx, _Binding(states=["ponder"]))[0]
        assert fresh.target_states == {"ponder"}

    def test_the_gate_cannot_be_moved_by_a_binding(self):
        ctx = _context_of(_CASES["policy_and_callback"]())
        moved = lib.approval_gate(
            ctx, _Binding(keys=["other"], priority=99_999, states=["x"])
        )[0]
        default = lib.approval_gate(ctx)[0]
        assert _signature(moved) == _signature(default)
        assert moved.updated_keys == {ContextKeys.TOOL_NAME}

    @pytest.mark.parametrize("key", [_DRIVER, "_max_iterations", "api_key", "system_x"])
    @pytest.mark.parametrize(
        "factory",
        [
            lib.iteration_limiter,
            lib.think_fresh_keys,
            lib.feedback_consumed,
            lib.approval_fresh_keys,
        ],
    )
    def test_a_file_never_names_an_internal_or_secret_key(self, factory, key):
        ctx = _context_of(_CASES["policy_and_callback"]())
        with pytest.raises(AgentError, match="never named by a pattern file"):
            factory(ctx, _Binding(keys=[key]))

    def test_a_public_key_is_accepted_and_used(self):
        ctx = _context_of(_CASES["no_hitl"]())
        built = lib.feedback_consumed(ctx, _Binding(keys=["note"]))[0]
        assert built.execution_lambda({"note": "x"}) == {"note": None}


class TestTheContextCountingLimiter:
    def test_equals_make_iteration_limiter_registered_by_the_live_helper(self):
        forced = {
            ContextKeys.MAX_ITERATIONS_REACHED: True,
            ContextKeys.EVALUATION_PASSED: True,
        }
        api = _FakeApi()
        helper = RegistrationMixin._register_iteration_limiter
        helper(None, api, make_iteration_limiter(4, forced))  # type: ignore[arg-type]
        live = api.handlers[0]
        ctx = HandlerContext(config=_config())
        built = lib.iteration_limiter(
            ctx,
            _Binding(
                keys=[
                    ContextKeys.MAX_ITERATIONS_REACHED,
                    ContextKeys.EVALUATION_PASSED,
                ],
                params={"limit": 4},
            ),
        )[0]
        assert _signature(built) == _signature(live)
        context: dict[str, Any] = {}
        for _ in range(6):
            expected = live.execution_lambda(dict(context))
            assert built.execution_lambda(dict(context)) == expected
            context[ContextKeys.ITERATION_COUNT] = expected[ContextKeys.ITERATION_COUNT]

    def test_the_limit_comes_from_params_then_options_then_the_config(self):
        config = _config()
        keys = [ContextKeys.SHOULD_TERMINATE]
        by_param = lib.iteration_limiter(
            HandlerContext(config=config, options={"max_iterations": 9}),
            _Binding(keys=keys, params={"limit": 2}),
        )[0]
        by_option = lib.iteration_limiter(
            HandlerContext(config=config, options={"max_iterations": 3}),
            _Binding(keys=keys),
        )[0]
        by_config = lib.iteration_limiter(
            HandlerContext(config=config), _Binding(keys=keys)
        )[0]
        zero, one = {ContextKeys.ITERATION_COUNT: 0}, {ContextKeys.ITERATION_COUNT: 1}
        # early: the stop is forced once the new count reaches limit - 1
        assert ContextKeys.SHOULD_TERMINATE in by_param.execution_lambda(dict(zero))
        assert ContextKeys.SHOULD_TERMINATE not in by_option.execution_lambda(
            dict(zero)
        )
        assert ContextKeys.SHOULD_TERMINATE in by_option.execution_lambda(dict(one))
        assert ContextKeys.SHOULD_TERMINATE not in by_config.execution_lambda(dict(one))

    @pytest.mark.parametrize("limit", [0, -1, True, "3", 2.5, None])
    def test_a_bad_limit_is_refused(self, limit):
        params = {} if limit is None else {"limit": limit}
        with pytest.raises(
            AgentError, match=r"agents\.iteration_limiter limit must be an int"
        ):
            lib.iteration_limiter(HandlerContext(), _Binding(params=params))
