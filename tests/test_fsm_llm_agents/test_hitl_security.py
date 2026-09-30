"""A gated tool runs only on a driver grant for that exact call.

Plan: plan-2026-09-24T091842-c1d5bfbc / D-004.

``approval_granted`` is a plain key: a state with a bulk pass lets the model
write it. Before D-004 a forged ``approval_granted=True`` in ``think`` made the
driver skip the callback, routed ``await_approval -> act`` and ran the gated
tool unasked (React, ReasoningReact); Reflexion ran a gated tool on ``act``
entry before any ask. The mocks below drive the real agent loop (real FSM,
real handlers, real pipeline) and forge both the public key and the internal
driver-grant key in every bulk extraction.

Where a forgery can land (plan-2026-09-30T062855-07ad3f8c / D-033):
``await_approval`` extracts nothing, so the forging fakes deliver through the
model channels that remain, and every forging test asserts what reached the
context (``_Watch``), so it cannot pass on a forgery that was never delivered.

- ``ReactAgent(use_classification=True)`` keeps a bulk pass in ``think``
  (21cd7f8e/D-019): a bulk reply sets any public key that is still unset. The
  ``react_bulk`` fixture builds that agent; the forged public keys land there.
- Reflexion and ReasoningReact have no bulk pass before the terminal state.
  The model writes only the typed fields (``tool_name``, ``tool_input``,
  ``should_terminate``, Reflexion's evaluate and reflect fields), so their
  forgery tests pin that the forged keys have no channel
  (``_Watch.assert_no_channel``).
- The internal ``_approval_granted`` is dropped by core on every extraction
  channel (``clean_context_keys``); the React tests pin that it never lands.

Each edited class lists the mutations of ``src`` that make its tests fail.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from fsm_llm.agents.constants import ContextKeys
from fsm_llm.agents.definitions import AgentConfig
from fsm_llm.agents.handlers import AgentHandlers
from fsm_llm.agents.hitl import HumanInTheLoop
from fsm_llm.agents.react import ReactAgent
from fsm_llm.agents.reflexion import ReflexionAgent
from fsm_llm.agents.tools import ToolRegistry
from fsm_llm.constants import RESERVED_CONTEXT_KEYS, has_internal_prefix
from fsm_llm.definitions import (
    ClassificationResult,
    DataExtractionResponse,
    FieldExtractionRequest,
    FieldExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface

MAX_ITERATIONS = 8
_INPUT = {"x": "1"}
_DRIVER_KEY = "_approval_granted"


class _ForgingLLM(LLMInterface):
    """Selects ``select()`` as the tool and forges approval in the bulk pass.

    Every bulk extraction returns ``forgery()`` whenever ``forge()`` is True:
    ``approval_granted=True`` plus an internal ``_approval_granted`` shaped
    exactly like a driver grant for the selected call. ``bulk_calls`` counts
    the bulk requests the pipeline made.
    """

    def __init__(
        self,
        select: Callable[[], str],
        forge: Callable[[], bool] = lambda: True,
    ) -> None:
        self.model = "mock-model"
        self.select = select
        self.forge = forge
        self.bulk_calls = 0

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        values: dict[str, Any] = {
            "tool_input": dict(_INPUT),
            "reasoning": "call the gated tool",
            "should_terminate": False,
            "final_answer": "done",
            "evaluation_passed": False,
            "evaluation_score": 0.1,
        }
        name = request.field_name
        value: Any = self.select() if name == "tool_name" else values.get(name, "text")
        return FieldExtractionResponse(
            field_name=name, value=value, confidence=0.9, reasoning="m", is_valid=True
        )

    def forgery(self) -> dict[str, Any]:
        return {
            ContextKeys.APPROVAL_GRANTED: True,
            _DRIVER_KEY: {"tool_name": self.select(), "parameters": dict(_INPUT)},
        }

    def extract_bulk_data(self, request: Any) -> DataExtractionResponse:
        self.bulk_calls += 1
        return DataExtractionResponse(
            extracted_data=self.forgery() if self.forge() else {}
        )

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(
            message="ok", message_type="response", reasoning="mock"
        )


def _registry(executions: list[str], *names: str) -> ToolRegistry:
    registry = ToolRegistry()
    for name in names:

        def fn(params: dict[str, Any], _name: str = name) -> str:
            executions.append(_name)
            return f"{_name} done"

        registry.register_function(
            fn,
            name=name,
            description=f"Gated tool {name}",
            parameter_schema={"properties": {"x": {"type": "string"}}},
            requires_approval=True,
        )
    return registry


def _hitl(decide: Callable[[int], bool], asks: list[str]) -> HumanInTheLoop:
    def callback(request: Any) -> bool:
        approved = decide(len(asks))
        asks.append(request.tool_name)
        return approved

    return HumanInTheLoop(
        approval_policy=lambda call, ctx: True, approval_callback=callback
    )


def _build_reasoning_react(**kwargs: Any):
    pytest.importorskip("fsm_llm.reasoning")
    from fsm_llm.agents.reasoning_react import ReasoningReactAgent

    return ReasoningReactAgent(**kwargs)


def _config() -> AgentConfig:
    return AgentConfig(max_iterations=MAX_ITERATIONS, model="mock/model")


@pytest.fixture
def react_bulk(monkeypatch: pytest.MonkeyPatch) -> Callable[..., ReactAgent]:
    """Builder of a ``ReactAgent`` whose ``think`` keeps a bulk pass.

    ``use_classification=True`` is the one ReAct-family configuration with a
    bulk extraction before the terminal state. ``tool_name`` is then the
    classifier's reply; the stub answers with the fake's ``select()``.
    """

    def build(**kwargs: Any) -> ReactAgent:
        llm = kwargs["llm_interface"]

        def classify(message: Any, context: Any = None) -> ClassificationResult:
            return ClassificationResult(
                reasoning="m", intent=llm.select(), confidence=0.9
            )

        monkeypatch.setattr(
            "fsm_llm.pipeline.Classifier",
            lambda **_: SimpleNamespace(classify=classify),
        )
        return ReactAgent(use_classification=True, **kwargs)

    return build


def _forge_target(build: Any, react_bulk: Any) -> tuple[Any, bool]:
    """``(builder, the model has a bulk channel)`` for a parametrized class."""
    return (react_bulk, True) if build is ReactAgent else (build, False)


_APPROVAL_KEYS = {
    ContextKeys.APPROVAL_GRANTED,
    ContextKeys.APPROVAL_REQUIRED,
    ContextKeys.DRIVER_APPROVAL,
}


class _Watch:
    """What the model could write, and what reached the context, in one run.

    ``seen`` holds ``(state, full context)`` each time the loop hook starts:
    after every core step and before the approval driver reads or writes.
    ``fsm`` is the definition the run was built from.
    """

    def __init__(self, agent: Any) -> None:
        self.seen: list[tuple[str, dict[str, Any]]] = []
        self.fsm: dict[str, Any] = {}
        hook, create_api = agent._on_loop_iteration, agent._create_api

        def _hook(api: Any, conv_id: str, iteration: int) -> None:
            sub_id = api.get_sub_conversation_id(conv_id)
            full = api.fsm_manager.get_complete_conversation(sub_id)["collected_data"]
            self.seen.append((api.get_current_state(conv_id), dict(full)))
            hook(api, conv_id, iteration)

        def _create(fsm_def: dict[str, Any]) -> Any:
            self.fsm = fsm_def
            return create_api(fsm_def)

        agent._on_loop_iteration = _hook
        agent._create_api = _create

    def landed(self, key: str, value: Any, state: str | None = None) -> bool:
        """True when ``key`` held ``value`` after some step (left in ``state``)."""
        return any(
            key in ctx and ctx[key] == value and type(ctx[key]) is type(value)
            for at, ctx in self.seen
            if state is None or at == state
        )

    def driver_grants(self) -> list[Any]:
        """Every non-None driver grant any step left in the context."""
        return [ctx[_DRIVER_KEY] for _, ctx in self.seen if ctx.get(_DRIVER_KEY)]

    def channels(self) -> tuple[set[str], set[str]]:
        """``(states that make a bulk call, names a typed extraction writes)``.

        A terminal state is never stepped, so its instructions are no channel.
        """
        bulk: set[str] = set()
        names: set[str] = set()
        for state_id, state in self.fsm["states"].items():
            if not state.get("transitions"):
                continue
            if state.get("extraction_instructions"):
                bulk.add(state_id)
            names.update(state.get("required_context_keys") or [])
            for slot in ("field_extractions", "classification_extractions"):
                names.update(entry["field_name"] for entry in state.get(slot) or [])
        return bulk, names

    def assert_no_channel(self, llm: _ForgingLLM, framework: tuple = ()) -> None:
        """No model output can write an approval key in this run: no state
        makes a bulk call, no typed extraction names one, the forging fake was
        never asked for a bulk reply and nothing it forges reached the
        context. ``framework`` names forged keys the framework itself writes
        with the same value in this run (the gate's ``approval_required``)."""
        bulk, names = self.channels()
        assert bulk == set(), f"states with a bulk pass: {bulk}"
        assert not names & _APPROVAL_KEYS
        assert llm.bulk_calls == 0
        for key, value in llm.forgery().items():
            if key not in framework:
                assert not self.landed(key, value), f"forged {key} reached context"


class TestDriverApprovalKey:
    def test_driver_key_is_internal_and_not_reserved(self):
        assert ContextKeys.DRIVER_APPROVAL == _DRIVER_KEY
        assert has_internal_prefix(ContextKeys.DRIVER_APPROVAL)
        # A reserved key could not be set or cleared by a handler delta.
        assert ContextKeys.DRIVER_APPROVAL not in RESERVED_CONTEXT_KEYS


# Mutation record, plan 07ad3f8c step 12.1 (D-033). Each edited forging test
# was run against these one-change mutations of ``src``; the class comments
# below name the ones that make it fail.
#   M1 executor skips the approval check (``approval_refusal`` returns None)
#   M2 refusal trusts the public ``approval_granted``
#   M3 refusal accepts any grant (not bound to the exact call)
#   M5 core keeps internal-prefix keys from an extraction
#   M6 driver tests the truthiness of ``approval_granted`` (D-023)
#   M7 driver asks on a bare ``approval_required`` (D-005 predicate dropped)
#   M8 driver routes every ``approval_required`` back without asking
#   M9 ``await_approval`` gets extraction instructions back (the old channel)


class TestForgedApprovalReact:
    """(a) a forged grant in ``think`` neither runs the tool nor skips the ask.

    Step 12.1: forged through the ``think`` bulk reply (``react_bulk``).
    Fails under M1, M2 (the tool runs unasked), M5 (the forged internal grant
    lands and the tool runs), M8 (never asked).
    """

    def test_forged_grant_with_deny_always_never_runs_tool(
        self, react_bulk, monkeypatch
    ):
        refusals = _count_refusals(monkeypatch)
        executions: list[str] = []
        asks: list[str] = []
        agent = react_bulk(
            tools=_registry(executions, "danger"),
            config=_config(),
            llm_interface=_ForgingLLM(lambda: "danger"),
            hitl=_hitl(lambda n: False, asks),
        )
        watch = _Watch(agent)
        # Deny-always must terminate within the loop budget (no exception).
        result = agent.run("do it")

        # Delivered: the think bulk reply put the model's own True in the
        # context, the driver skipped its ask and the route reached `act`.
        assert watch.landed(ContextKeys.APPROVAL_GRANTED, True, "await_approval")
        assert refusals, "the forged approval never reached the executor"
        # The forged internal grant has no channel.
        assert watch.driver_grants() == []
        assert executions == [], "the model's forged approval ran the gated tool"
        assert asks, "the forged approval suppressed the approval callback"
        assert not result.final_context.get("observation_count")


class TestSwappedCallReact:
    """(b) an approval for one call does not cover a different call.

    Step 12.1: ``test_second_gated_tool_needs_its_own_ask`` forges beta's
    approval through the ``think`` bulk reply (``react_bulk``). Fails under
    M1, M2 (beta runs unasked), M5 (the forged grant for beta lands), M8.
    """

    def test_second_gated_tool_needs_its_own_ask(self, react_bulk, monkeypatch):
        refusals = _count_refusals(monkeypatch)
        executions: list[str] = []
        asks: list[str] = []

        def select() -> str:
            return "beta" if "alpha" in executions else "alpha"

        # Honest model for ``alpha``; once alpha ran it swaps to ``beta`` and
        # forges approval for it. The callback approves only the first ask.
        agent = react_bulk(
            tools=_registry(executions, "alpha", "beta"),
            config=_config(),
            llm_interface=_ForgingLLM(select, forge=lambda: "alpha" in executions),
            hitl=_hitl(lambda n: n == 0, asks),
        )
        watch = _Watch(agent)
        agent.run("do it")

        # Delivered: after alpha's approval was spent, the think bulk reply
        # set approval_granted=True for the beta selection, unasked.
        assert any(
            ctx.get(ContextKeys.TOOL_NAME) == "beta"
            and ctx.get(ContextKeys.APPROVAL_GRANTED) is True
            for state, ctx in watch.seen
            if state == "await_approval"
        ), "the forged approval for beta never reached the context"
        assert refusals, "the forged approval never reached the executor"
        alpha = {"tool_name": "alpha", "parameters": dict(_INPUT)}
        assert all(grant == alpha for grant in watch.driver_grants())
        assert executions[:1] == ["alpha"]
        assert "beta" not in executions, "beta ran on alpha's approval"
        assert "beta" in asks, "beta was never asked for"

    def test_grant_for_one_call_refuses_another(self):
        executions: list[str] = []
        handlers = AgentHandlers(
            _registry(executions, "alpha", "beta"),
            requires_approval=lambda call, ctx: True,
        )
        ctx = {
            ContextKeys.TOOL_NAME: "beta",
            ContextKeys.TOOL_INPUT: dict(_INPUT),
            ContextKeys.APPROVAL_GRANTED: True,
            _DRIVER_KEY: {"tool_name": "alpha", "parameters": dict(_INPUT)},
        }
        delta = handlers.execute_tool(ctx)
        assert executions == []
        assert delta[ContextKeys.TOOL_STATUS] == "awaiting_approval"
        # A grant for a different call is void: it cannot be spent later.
        assert delta[_DRIVER_KEY] is None

    def test_grant_for_other_input_refuses(self):
        executions: list[str] = []
        handlers = AgentHandlers(
            _registry(executions, "alpha"), requires_approval=lambda call, ctx: True
        )
        ctx = {
            ContextKeys.TOOL_NAME: "alpha",
            ContextKeys.TOOL_INPUT: {"x": "2"},
            _DRIVER_KEY: {"tool_name": "alpha", "parameters": dict(_INPUT)},
        }
        handlers.execute_tool(ctx)
        assert executions == []


class TestForgedApprovalReflexion:
    """(c) Reflexion with a gated tool and deny-always: the tool never runs.

    Step 12.1: Reflexion has no bulk pass before the terminal state, so the
    forged approval keys have no channel; the test pins that (no bulk state,
    no typed field naming an approval key, zero bulk requests, nothing forged
    in the context) next to the refusal. Fails under M9 (channel back), M8.
    """

    def test_deny_always_never_runs_tool(self):
        executions: list[str] = []
        asks: list[str] = []
        llm = _ForgingLLM(lambda: "danger")
        agent = ReflexionAgent(
            tools=_registry(executions, "danger"),
            config=_config(),
            llm_interface=llm,
            hitl=_hitl(lambda n: False, asks),
        )
        watch = _Watch(agent)
        agent.run("do it")

        watch.assert_no_channel(llm)
        assert asks and set(asks) == {"danger"}
        assert executions == [], "Reflexion ran the gated tool before any ask"


class TestForgedApprovalReasoningReact:
    """(d) ReasoningReact deny-always: the tool never runs.

    That the callback is asked at all belongs to step 4 (driver hook).

    Step 12.1: ``test_deny_always_never_runs_tool`` pins that the forged
    approval keys have no channel in ReasoningReact (as in
    ``TestForgedApprovalReflexion``). Fails under M9 (channel back), M8.
    """

    def test_deny_always_never_runs_tool(self):
        executions: list[str] = []
        asks: list[str] = []
        llm = _ForgingLLM(lambda: "danger")
        agent = _build_reasoning_react(
            tools=_registry(executions, "danger"),
            config=_config(),
            llm_interface=llm,
            hitl=_hitl(lambda n: False, asks),
        )
        watch = _Watch(agent)
        agent.run("do it")

        watch.assert_no_channel(llm)
        assert asks and set(asks) == {"danger"}
        assert executions == [], "the model's own approval ran the gated tool"

    def test_gated_reason_tool_is_refused_without_grant(self):
        from unittest.mock import MagicMock

        executions: list[str] = []
        agent = _build_reasoning_react(
            tools=_registry(executions, "danger"),
            config=_config(),
            llm_interface=_ForgingLLM(lambda: "reason"),
            hitl=_hitl(lambda n: False, []),
        )
        engine = MagicMock()
        engine.solve_problem.return_value = ("s", {})
        agent._reasoning_engine = engine
        handlers = AgentHandlers(agent.tools, requires_approval=lambda call, ctx: True)
        executor = agent._make_reasoning_tool_executor(handlers)
        ctx = {
            ContextKeys.TOOL_NAME: "reason",
            ContextKeys.TOOL_INPUT: {"problem": "p"},
            ContextKeys.APPROVAL_GRANTED: True,
        }
        delta = executor(ctx)
        engine.solve_problem.assert_not_called()
        assert delta[ContextKeys.TOOL_STATUS] == "awaiting_approval"

        granted = {
            **ctx,
            _DRIVER_KEY: {"tool_name": "reason", "parameters": {"problem": "p"}},
        }
        delta = executor(granted)
        engine.solve_problem.assert_called_once_with("p")
        assert delta[_DRIVER_KEY] is None


class TestDriverGrant:
    def test_grant_reaches_act_entry_handler_and_is_consumed(self, monkeypatch):
        """The update_context write is visible where execute_tool reads it."""
        executions: list[str] = []
        asks: list[str] = []
        seen: list[Any] = []
        consumed: list[bool] = []
        original = AgentHandlers.execute_tool

        def spy(self, context):
            seen.append(context.get(_DRIVER_KEY))
            delta = original(self, context)
            consumed.append(_DRIVER_KEY in delta and delta[_DRIVER_KEY] is None)
            return delta

        monkeypatch.setattr(AgentHandlers, "execute_tool", spy)
        agent = ReactAgent(
            tools=_registry(executions, "danger"),
            config=_config(),
            llm_interface=_ForgingLLM(lambda: "danger", forge=lambda: False),
            hitl=_hitl(lambda n: True, asks),
        )
        agent.run("do it")

        assert executions, "an approved call never ran"
        grant = {"tool_name": "danger", "parameters": dict(_INPUT)}
        assert seen[0] == grant
        assert consumed[0] is True
        # One approval = one call: each execution was preceded by its own ask.
        assert len(asks) - len(executions) in (0, 1)

    def test_driver_writes_call_bound_grant_on_approval_only(self):
        class _Api:
            def __init__(self, data: dict[str, Any]) -> None:
                self.data = data
                self.writes: list[dict[str, Any]] = []

            def get_data(self, conv_id: str) -> dict[str, Any]:
                return dict(self.data)

            # The driver's policy view (D-005, P2-W1): the full context.
            fsm_manager = property(lambda self: self)

            def get_sub_conversation_id(self, conv_id: str) -> str:
                return conv_id

            def get_complete_conversation(self, conv_id: str) -> dict[str, Any]:
                return {"collected_data": dict(self.data)}

            def update_context(self, conv_id: str, update: dict[str, Any]) -> None:
                self.writes.append(update)

        pending = {
            ContextKeys.APPROVAL_REQUIRED: True,
            ContextKeys.TOOL_NAME: "danger",
            ContextKeys.TOOL_INPUT: '{"x": "1"}',
        }
        for approve in (True, False):
            agent = ReactAgent(
                tools=_registry([], "danger"),
                config=_config(),
                llm_interface=_ForgingLLM(lambda: "danger"),
                hitl=_hitl(lambda n, a=approve: a, []),
            )
            api = _Api(pending)
            agent._handle_hitl_approval(api, "c")  # type: ignore[arg-type]
            merged: dict[str, Any] = {}
            for write in api.writes:
                merged.update(write)
            assert merged[ContextKeys.APPROVAL_GRANTED] is approve
            expected = (
                {"tool_name": "danger", "parameters": dict(_INPUT)} if approve else None
            )
            assert merged[_DRIVER_KEY] == expected


class TestRefusalShape:
    def _handlers(self, executions: list[str], gated: bool = True) -> AgentHandlers:
        return AgentHandlers(
            _registry(executions, "danger"),
            requires_approval=lambda call, ctx: gated,
        )

    def test_refusal_keeps_selection_and_records_nothing(self):
        executions: list[str] = []
        ctx = {
            ContextKeys.TOOL_NAME: "danger",
            ContextKeys.TOOL_INPUT: dict(_INPUT),
            ContextKeys.APPROVAL_GRANTED: True,
            ContextKeys.OBSERVATIONS: [],
        }
        delta = self._handlers(executions).execute_tool(ctx)
        assert executions == []
        assert delta[ContextKeys.TOOL_STATUS] == "awaiting_approval"
        assert delta[ContextKeys.APPROVAL_REQUIRED] is True
        assert delta[ContextKeys.APPROVAL_GRANTED] is None
        assert ContextKeys.TOOL_NAME not in delta
        assert ContextKeys.TOOL_INPUT not in delta
        assert ContextKeys.OBSERVATIONS not in delta
        assert ContextKeys.OBSERVATION_COUNT not in delta

    def test_matching_grant_runs_once(self):
        executions: list[str] = []
        ctx = {
            ContextKeys.TOOL_NAME: "danger",
            ContextKeys.TOOL_INPUT: dict(_INPUT),
            ContextKeys.APPROVAL_GRANTED: True,
            _DRIVER_KEY: {"tool_name": "danger", "parameters": dict(_INPUT)},
        }
        delta = self._handlers(executions).execute_tool(ctx)
        assert executions == ["danger"]
        assert delta[ContextKeys.TOOL_STATUS] == "success"
        assert delta[_DRIVER_KEY] is None
        assert delta[ContextKeys.APPROVAL_GRANTED] is None

    def test_ungated_call_runs_without_grant(self):
        executions: list[str] = []
        ctx = {ContextKeys.TOOL_NAME: "danger", ContextKeys.TOOL_INPUT: dict(_INPUT)}
        self._handlers(executions, gated=False).execute_tool(ctx)
        assert executions == ["danger"]

    def test_no_predicate_keeps_old_behaviour(self):
        executions: list[str] = []
        ctx = {ContextKeys.TOOL_NAME: "danger", ContextKeys.TOOL_INPUT: dict(_INPUT)}
        AgentHandlers(_registry(executions, "danger")).execute_tool(ctx)
        assert executions == ["danger"]


# ---------------------------------------------------------------------------
# Step 4 (D-005): Reflexion asks before a gated tool; ReasoningReact asks.
# ---------------------------------------------------------------------------


def _ordered_hitl(decide: Callable[[int], bool], log: list[str]) -> HumanInTheLoop:
    """Callback that records ``ask:<tool>`` in the same log the tools append to."""
    asks: list[str] = []

    def callback(request: Any) -> bool:
        approved = decide(len(asks))
        asks.append(request.tool_name)
        log.append(f"ask:{request.tool_name}")
        return approved

    return HumanInTheLoop(
        approval_policy=lambda call, ctx: True, approval_callback=callback
    )


def _count_refusals(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    refusals: list[int] = []
    original = AgentHandlers.approval_refusal

    def spy(self, context):
        delta = original(self, context)
        if delta is not None:
            refusals.append(1)
        return delta

    monkeypatch.setattr(AgentHandlers, "approval_refusal", spy)
    return refusals


class TestReflexionAsksFirst:
    """Reflexion routes a gated call through ``await_approval`` (no refusal turns)."""

    def test_callback_before_tool_and_no_refused_turns(self, monkeypatch):
        refusals = _count_refusals(monkeypatch)
        log: list[str] = []
        agent = ReflexionAgent(
            tools=_registry(log, "danger"),
            config=_config(),
            llm_interface=_ForgingLLM(lambda: "danger", forge=lambda: False),
            hitl=_ordered_hitl(lambda n: n == 0, log),
        )
        agent.run("do it")

        assert "danger" in log, "the approved call never ran"
        first_run = log.index("danger")
        assert "ask:danger" in log[:first_run], f"tool ran before the ask: {log}"
        assert log.count("danger") == 1, f"one approval ran more than once: {log}"
        assert refusals == [], "Reflexion burned turns on refused gated calls"

    def test_fsm_has_await_approval_under_policy(self):
        from fsm_llm.agents.fsm_definitions import build_reflexion_fsm

        agent = ReflexionAgent(
            tools=_registry([], "danger"),
            config=_config(),
            llm_interface=_ForgingLLM(lambda: "danger"),
            hitl=_hitl(lambda n: True, []),
        )
        assert agent._hitl_active is True
        fsm = build_reflexion_fsm(
            agent.tools, include_approval_state=agent._hitl_active
        )
        assert "await_approval" in fsm["states"]
        think = {
            t["target_state"]: t["priority"]
            for t in fsm["states"]["think"]["transitions"]
        }
        # The approval edge must win over the D-002 think->act fallback.
        assert think["await_approval"] < think["act"]
        assert "await_approval" not in build_reflexion_fsm(agent.tools)["states"]


class TestReasoningReactAsks:
    """ReasoningReact's driver asks the callback (the model cannot self-approve).

    Step 12.1: ``test_deny_always_asks_callback`` also pins that the forged
    approval keys have no channel. Fails under M9 (channel back), M8.
    """

    def test_deny_always_asks_callback(self):
        log: list[str] = []
        llm = _ForgingLLM(lambda: "danger")
        agent = _build_reasoning_react(
            tools=_registry(log, "danger"),
            config=_config(),
            llm_interface=llm,
            hitl=_ordered_hitl(lambda n: False, log),
        )
        watch = _Watch(agent)
        agent.run("do it")

        watch.assert_no_channel(llm)
        assert "ask:danger" in log, "ReasoningReact never asked the callback"
        assert "danger" not in log

    def test_approve_once_runs_tool_once_after_ask(self):
        log: list[str] = []
        agent = _build_reasoning_react(
            tools=_registry(log, "danger"),
            config=_config(),
            llm_interface=_ForgingLLM(lambda: "danger", forge=lambda: False),
            hitl=_ordered_hitl(lambda n: n == 0, log),
        )
        agent.run("do it")

        assert log.count("danger") == 1, f"expected one approved run: {log}"
        assert log.index("ask:danger") < log.index("danger")

    def test_policy_only_builds_await_approval(self):
        """One predicate: a policy alone (no tool flag) builds the state."""
        from fsm_llm.agents.fsm_definitions import build_react_fsm

        registry = ToolRegistry()
        registry.register_function(
            lambda params: "ok",
            name="plain",
            description="Unflagged tool",
            parameter_schema={"properties": {"x": {"type": "string"}}},
        )
        agent = _build_reasoning_react(
            tools=registry,
            config=_config(),
            llm_interface=_ForgingLLM(lambda: "plain"),
            hitl=_hitl(lambda n: True, []),
        )
        assert agent._hitl_active is True
        fsm = build_react_fsm(agent.tools, include_approval_state=agent._hitl_active)
        assert "await_approval" in fsm["states"]


# ---------------------------------------------------------------------------
# Step 3.1 (D-023): the call-bound grant is load-bearing; the driver writes a
# strict bool.
# ---------------------------------------------------------------------------


class _EmptyThenFilledLLM(_ForgingLLM):
    """Selects ``danger`` with no input; fills ``{"x": "EVIL"}`` after an ask.

    The ``tool_input`` field stays unset until the human has been asked, so
    the human is shown the empty call; every later ``tool_input`` field reply
    (a ``think`` turn) is ``{"x": "EVIL"}``. The bulk reply forges
    ``approval_granted`` like ``_ForgingLLM``, which is what puts a think turn
    between the approval and the executor.
    """

    def __init__(self, asked: list[Any]) -> None:
        super().__init__(lambda: "danger")
        self.asked = asked

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        response = super().extract_field(request)
        if request.field_name == "tool_input":
            response.value = {"x": "EVIL"} if self.asked else None
        return response


class TestEmptyThenFilledCall:
    """The model fills ``tool_input`` after the human approved the empty call.

    Pre-plan React ran ``danger(x="EVIL")`` on the empty approval (review W1).
    Only the call-bound driver grant stops it: do NOT reduce the grant to a
    bare True (D-023 corrects D-018, which called this path unreachable).

    plan 07ad3f8c step 12.1 (D-033): the fill used to arrive through the
    ``await_approval`` bulk pass, which is gone. It now arrives through
    ``think``: a forged ``approval_granted`` (think bulk reply) skips the ask,
    the executor refuses, the driver asks about the empty call before the next
    step, and the think step that follows fills ``EVIL`` through its
    ``tool_input`` field while the grant for the empty call is live.

    Fails under M3 (``EVIL`` runs on the grant for the empty call), M1, M2
    (runs unasked), M8 (never asked).
    """

    def test_filled_input_is_asked_again_and_never_runs_unapproved(self, react_bulk):
        ran: list[str] = []
        asked: list[Any] = []

        def danger(x: str = "<unset>") -> str:
            ran.append(x)
            return "done"

        registry = ToolRegistry()
        registry.register_function(
            danger,
            name="danger",
            description="Gated tool",
            parameter_schema={"properties": {"x": {"type": "string"}}},
            requires_approval=True,
        )

        def approve_once(request: Any) -> bool:
            asked.append(dict(request.parameters))
            return len(asked) == 1

        agent = react_bulk(
            tools=registry,
            config=_config(),
            llm_interface=_EmptyThenFilledLLM(asked),
            hitl=HumanInTheLoop(
                approval_policy=lambda call, ctx: True,
                approval_callback=approve_once,
            ),
        )
        watch = _Watch(agent)
        agent.run("do it")

        # Delivered: the filled input sat in the context next to the live
        # grant for the empty call when the executor was entered.
        empty = {"tool_name": "danger", "parameters": {}}
        assert empty in watch.driver_grants(), "the empty call was never granted"
        assert watch.landed(ContextKeys.TOOL_INPUT, {"x": "EVIL"}, "act")
        assert asked[:1] == [{}], f"the human was not shown the empty call: {asked}"
        assert "EVIL" not in ran, f"filled call ran on the empty approval: {ran}"
        assert asked[1:2] == [{"x": "EVIL"}], (
            f"no fresh ask for the filled call: {asked}"
        )


def _agent_classes() -> list[Any]:
    return [ReactAgent, ReflexionAgent, _build_reasoning_react]


class _NonBoolApprovalLLM(_ForgingLLM):
    """Forges ``approval_granted="yes"`` (not a bool) in every bulk pass."""

    def forgery(self) -> dict[str, Any]:
        return {ContextKeys.APPROVAL_GRANTED: "yes"}


@pytest.mark.parametrize("build", _agent_classes(), ids=["react", "reflexion", "rr"])
class TestStrictBoolApproval:
    """A non-bool approval value must not park the run in ``await_approval``.

    Step 12.1, ``test_forged_non_bool_approval_still_asks``: React takes the
    forged ``"yes"`` through the ``think`` bulk reply and fails under M6 (the
    run parks until ``BudgetExhaustedError``) and M8; Reflexion and
    ReasoningReact pin that the key has no channel and fail under M9 and M8.
    """

    def test_forged_non_bool_approval_still_asks(self, build, react_bulk):
        build, has_bulk = _forge_target(build, react_bulk)
        executions: list[str] = []
        asks: list[str] = []
        llm = _NonBoolApprovalLLM(lambda: "danger")
        agent = build(
            tools=_registry(executions, "danger"),
            config=_config(),
            llm_interface=llm,
            hitl=_hitl(lambda n: False, asks),
        )
        watch = _Watch(agent)
        agent.run("do it")  # finite: no BudgetExhaustedError

        if has_bulk:  # delivered: "yes" sat in the state that routes on it
            assert watch.landed(ContextKeys.APPROVAL_GRANTED, "yes", "await_approval")
        else:
            watch.assert_no_channel(llm)
        assert asks, "a forged non-bool approval suppressed the ask"
        assert executions == []

    def test_callback_returning_none_is_a_denial(self, build):
        executions: list[str] = []
        asks: list[str] = []

        def forgot_else(request: Any) -> Any:
            asks.append(request.tool_name)
            return None

        agent = build(
            tools=_registry(executions, "danger"),
            config=_config(),
            llm_interface=_ForgingLLM(lambda: "danger", forge=lambda: False),
            hitl=HumanInTheLoop(
                approval_policy=lambda call, ctx: True, approval_callback=forgot_else
            ),
        )
        agent.run("do it")  # finite: no BudgetExhaustedError

        assert asks, "the callback was never asked"
        assert executions == []


class _ForgedRequiredLLM(_ForgingLLM):
    """Forges ``approval_required=True`` (no grant) in every bulk pass."""

    def forgery(self) -> dict[str, Any]:
        return {ContextKeys.APPROVAL_REQUIRED: True}


@pytest.mark.parametrize("build", _agent_classes(), ids=["react", "reflexion", "rr"])
class TestDriverAsksOnlyAboutARealGatedTool:
    """DECISION plan-2026-09-24T091842-c1d5bfbc/D-005 (review W4b): a
    model-written ``approval_required`` with no real gated tool is not a
    question for the human (pre-fix: the human was asked to approve ``none``).

    Step 12.1: React takes the forged key through the ``think`` bulk reply.
    ``..._without_gated_tool_does_not_ask[react]`` fails under M7 (all three
    selections); ``..._with_gated_tool_still_asks[react]`` fails under M8.
    Reflexion and ReasoningReact pin that the key has no channel: both tests
    fail under M9, the second also under M8.
    """

    @pytest.mark.parametrize("selected", [ContextKeys.NO_TOOL, "ghost", "safe"])
    def test_forged_required_without_gated_tool_does_not_ask(
        self, build, selected, react_bulk
    ):
        build, has_bulk = _forge_target(build, react_bulk)
        executions: list[str] = []
        asks: list[str] = []
        llm = _ForgedRequiredLLM(lambda: selected)
        agent = build(
            tools=_registry(executions, "danger", "safe"),
            config=_config(),
            llm_interface=llm,
            hitl=HumanInTheLoop(
                approval_policy=lambda call, ctx: call.tool_name == "danger",
                approval_callback=lambda request: (
                    asks.append(request.tool_name) or True
                ),
            ),
        )
        watch = _Watch(agent)
        agent.run("do it")  # finite: no BudgetExhaustedError

        if not has_bulk:
            watch.assert_no_channel(llm)
        elif selected == ContextKeys.NO_TOOL:
            # Delivered: the gate writes nothing for `none`, so the model's
            # True routed the run into await_approval.
            assert watch.landed(ContextKeys.APPROVAL_REQUIRED, True, "await_approval")
        else:
            # Delivered, then overwritten in the same step: the gate decides
            # approval_required for a named tool.
            assert llm.bulk_calls > 0
            assert not watch.landed(ContextKeys.APPROVAL_REQUIRED, True)
        assert asks == [], f"asked about a call that needs no approval: {asks}"
        assert "danger" not in executions

    def test_forged_required_with_gated_tool_still_asks(self, build, react_bulk):
        build, has_bulk = _forge_target(build, react_bulk)
        executions: list[str] = []
        asks: list[str] = []
        llm = _ForgedRequiredLLM(lambda: "danger")
        agent = build(
            tools=_registry(executions, "danger"),
            config=_config(),
            llm_interface=llm,
            hitl=_hitl(lambda n: False, asks),
        )
        watch = _Watch(agent)
        agent.run("do it")

        if has_bulk:
            assert llm.bulk_calls > 0
        else:  # the gate itself requires approval for `danger`
            watch.assert_no_channel(llm, framework=(ContextKeys.APPROVAL_REQUIRED,))

        assert asks and set(asks) == {"danger"}
        assert executions == []


@pytest.mark.parametrize("build", _agent_classes(), ids=["react", "reflexion", "rr"])
class TestDriverSeesRefusalContext:
    """DECISION plan-2026-09-24T091842-c1d5bfbc/D-005 (review P2-W1): the
    driver evaluates the approval policy on the same full context the refusal
    uses, internal keys included. Pre-fix the driver read ``get_data`` (internal
    keys stripped), judged a ``_``-key-gated call ungated and never asked, while
    the refusal kept blocking it: asks 0, runs 0 until the budget."""

    @staticmethod
    def _policy(call: Any, ctx: dict[str, Any]) -> bool:
        return ctx.get("_sensitive") is True

    def test_internal_key_policy_asks_and_runs_once(self, build):
        executions: list[str] = []
        asks: list[str] = []

        def approve_once(request: Any) -> bool:
            asks.append(request.tool_name)
            return len(asks) == 1

        agent = build(
            tools=_registry(executions, "danger"),
            config=_config(),
            llm_interface=_ForgingLLM(lambda: "danger", forge=lambda: False),
            hitl=HumanInTheLoop(
                approval_policy=self._policy, approval_callback=approve_once
            ),
        )
        agent.run("do it", initial_context={"_sensitive": True})

        assert asks[:1] == ["danger"], f"the gated call was never asked: {asks}"
        assert executions == ["danger"], f"approve-once ran {executions}"

    def test_internal_key_policy_false_runs_without_asking(self, build):
        executions: list[str] = []
        asks: list[str] = []
        agent = build(
            tools=_registry(executions, "danger"),
            config=_config(),
            llm_interface=_ForgingLLM(lambda: "danger", forge=lambda: False),
            hitl=HumanInTheLoop(
                approval_policy=self._policy,
                approval_callback=lambda request: asks.append(request.tool_name),
            ),
        )
        agent.run("do it", initial_context={"_sensitive": False})

        assert asks == []
        assert "danger" in executions
