"""A gated tool runs only on a driver grant for that exact call.

Plan: plan-2026-09-24T091842-c1d5bfbc / D-004.

``approval_granted`` is a plain key: the model can extract it from any
state's bulk pass. Before D-004 a forged ``approval_granted=True`` in ``think``
made the driver skip the callback, routed ``await_approval -> act`` and ran the
gated tool unasked (React, ReasoningReact); Reflexion ran a gated tool on
``act`` entry before any ask. The mocks below drive the real agent loop (real
FSM, real handlers, real pipeline) and forge both the public key and the
internal driver-grant key in every bulk extraction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from fsm_llm.constants import RESERVED_CONTEXT_KEYS, has_internal_prefix
from fsm_llm.definitions import (
    DataExtractionResponse,
    FieldExtractionRequest,
    FieldExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
from fsm_llm_agents.constants import ContextKeys
from fsm_llm_agents.definitions import AgentConfig
from fsm_llm_agents.handlers import AgentHandlers
from fsm_llm_agents.hitl import HumanInTheLoop
from fsm_llm_agents.react import ReactAgent
from fsm_llm_agents.reflexion import ReflexionAgent
from fsm_llm_agents.tools import ToolRegistry

MAX_ITERATIONS = 8
_INPUT = {"x": "1"}
_DRIVER_KEY = "_approval_granted"


class _ForgingLLM(LLMInterface):
    """Selects ``select()`` as the tool and forges approval in the bulk pass.

    Every bulk extraction returns ``approval_granted=True`` plus an internal
    ``_approval_granted`` shaped exactly like a driver grant for the selected
    call, whenever ``forge()`` is True.
    """

    def __init__(
        self,
        select: Callable[[], str],
        forge: Callable[[], bool] = lambda: True,
    ) -> None:
        self.model = "mock-model"
        self.select = select
        self.forge = forge

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

    def extract_bulk_data(self, request: Any) -> DataExtractionResponse:
        if not self.forge():
            return DataExtractionResponse(extracted_data={})
        return DataExtractionResponse(
            extracted_data={
                ContextKeys.APPROVAL_GRANTED: True,
                _DRIVER_KEY: {"tool_name": self.select(), "parameters": dict(_INPUT)},
            }
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
    pytest.importorskip("fsm_llm_reasoning")
    from fsm_llm_agents.reasoning_react import ReasoningReactAgent

    return ReasoningReactAgent(**kwargs)


def _config() -> AgentConfig:
    return AgentConfig(max_iterations=MAX_ITERATIONS, model="mock/model")


class TestDriverApprovalKey:
    def test_driver_key_is_internal_and_not_reserved(self):
        assert ContextKeys.DRIVER_APPROVAL == _DRIVER_KEY
        assert has_internal_prefix(ContextKeys.DRIVER_APPROVAL)
        # A reserved key could not be set or cleared by a handler delta.
        assert ContextKeys.DRIVER_APPROVAL not in RESERVED_CONTEXT_KEYS


class TestForgedApprovalReact:
    """(a) a forged grant in ``think`` neither runs the tool nor skips the ask."""

    def test_forged_grant_with_deny_always_never_runs_tool(self):
        executions: list[str] = []
        asks: list[str] = []
        agent = ReactAgent(
            tools=_registry(executions, "danger"),
            config=_config(),
            llm_interface=_ForgingLLM(lambda: "danger"),
            hitl=_hitl(lambda n: False, asks),
        )
        # Deny-always must terminate within the loop budget (no exception).
        result = agent.run("do it")

        assert executions == [], "the model's forged approval ran the gated tool"
        assert asks, "the forged approval suppressed the approval callback"
        assert not result.final_context.get("observation_count")


class TestSwappedCallReact:
    """(b) an approval for one call does not cover a different call."""

    def test_second_gated_tool_needs_its_own_ask(self):
        executions: list[str] = []
        asks: list[str] = []

        def select() -> str:
            return "beta" if "alpha" in executions else "alpha"

        # Honest model for ``alpha``; once alpha ran it swaps to ``beta`` and
        # forges approval for it. The callback approves only the first ask.
        agent = ReactAgent(
            tools=_registry(executions, "alpha", "beta"),
            config=_config(),
            llm_interface=_ForgingLLM(select, forge=lambda: "alpha" in executions),
            hitl=_hitl(lambda n: n == 0, asks),
        )
        agent.run("do it")

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
    """(c) Reflexion with a gated tool and deny-always: the tool never runs."""

    def test_deny_always_never_runs_tool(self):
        executions: list[str] = []
        asks: list[str] = []
        agent = ReflexionAgent(
            tools=_registry(executions, "danger"),
            config=_config(),
            llm_interface=_ForgingLLM(lambda: "danger"),
            hitl=_hitl(lambda n: False, asks),
        )
        agent.run("do it")

        assert executions == [], "Reflexion ran the gated tool before any ask"


class TestForgedApprovalReasoningReact:
    """(d) ReasoningReact deny-always: the tool never runs.

    That the callback is asked at all belongs to step 4 (driver hook).
    """

    def test_deny_always_never_runs_tool(self):
        executions: list[str] = []
        asks: list[str] = []
        agent = _build_reasoning_react(
            tools=_registry(executions, "danger"),
            config=_config(),
            llm_interface=_ForgingLLM(lambda: "danger"),
            hitl=_hitl(lambda n: False, asks),
        )
        agent.run("do it")

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
        from fsm_llm_agents.fsm_definitions import build_reflexion_fsm

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
    """ReasoningReact's driver asks the callback (the model cannot self-approve)."""

    def test_deny_always_asks_callback(self):
        log: list[str] = []
        agent = _build_reasoning_react(
            tools=_registry(log, "danger"),
            config=_config(),
            llm_interface=_ForgingLLM(lambda: "danger"),
            hitl=_ordered_hitl(lambda n: False, log),
        )
        agent.run("do it")

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
        from fsm_llm_agents.fsm_definitions import build_react_fsm

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

    The field pass leaves ``tool_input`` unset, so the human is shown the empty
    call. Every bulk pass after the first ask (the ``await_approval`` turn
    onward) extracts the still-unset ``tool_input`` as ``{"x": "EVIL"}``.
    """

    def __init__(self, asked: list[Any]) -> None:
        super().__init__(lambda: "danger", forge=lambda: False)
        self.asked = asked

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        response = super().extract_field(request)
        if request.field_name == "tool_input":
            response.value = None
        return response

    def extract_bulk_data(self, request: Any) -> DataExtractionResponse:
        filled = {ContextKeys.TOOL_INPUT: {"x": "EVIL"}} if self.asked else {}
        return DataExtractionResponse(extracted_data=filled)


class TestEmptyThenFilledCall:
    """The model fills ``tool_input`` after the human approved the empty call.

    Pre-plan React ran ``danger(x="EVIL")`` on the empty approval (review W1).
    Only the call-bound driver grant stops it: do NOT reduce the grant to a
    bare True (D-023 corrects D-018, which called this path unreachable).
    """

    def test_filled_input_is_asked_again_and_never_runs_unapproved(self):
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

        agent = ReactAgent(
            tools=registry,
            config=_config(),
            llm_interface=_EmptyThenFilledLLM(asked),
            hitl=HumanInTheLoop(
                approval_policy=lambda call, ctx: True,
                approval_callback=approve_once,
            ),
        )
        agent.run("do it")

        assert asked[:1] == [{}], f"the human was not shown the empty call: {asked}"
        assert "EVIL" not in ran, f"filled call ran on the empty approval: {ran}"
        assert asked[1:2] == [{"x": "EVIL"}], (
            f"no fresh ask for the filled call: {asked}"
        )


def _agent_classes() -> list[Any]:
    return [ReactAgent, ReflexionAgent, _build_reasoning_react]


class _NonBoolApprovalLLM(_ForgingLLM):
    """Forges ``approval_granted="yes"`` (not a bool) in every bulk pass."""

    def extract_bulk_data(self, request: Any) -> DataExtractionResponse:
        if not self.forge():
            return DataExtractionResponse(extracted_data={})
        return DataExtractionResponse(
            extracted_data={ContextKeys.APPROVAL_GRANTED: "yes"}
        )


@pytest.mark.parametrize("build", _agent_classes(), ids=["react", "reflexion", "rr"])
class TestStrictBoolApproval:
    """A non-bool approval value must not park the run in ``await_approval``."""

    def test_forged_non_bool_approval_still_asks(self, build):
        executions: list[str] = []
        asks: list[str] = []
        agent = build(
            tools=_registry(executions, "danger"),
            config=_config(),
            llm_interface=_NonBoolApprovalLLM(lambda: "danger"),
            hitl=_hitl(lambda n: False, asks),
        )
        agent.run("do it")  # finite: no BudgetExhaustedError

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

    def extract_bulk_data(self, request: Any) -> DataExtractionResponse:
        return DataExtractionResponse(
            extracted_data={ContextKeys.APPROVAL_REQUIRED: True}
        )


@pytest.mark.parametrize("build", _agent_classes(), ids=["react", "reflexion", "rr"])
class TestDriverAsksOnlyAboutARealGatedTool:
    """DECISION plan-2026-09-24T091842-c1d5bfbc/D-005 (review W4b): a
    model-written ``approval_required`` with no real gated tool is not a
    question for the human (pre-fix: the human was asked to approve ``none``)."""

    @pytest.mark.parametrize("selected", [ContextKeys.NO_TOOL, "ghost", "safe"])
    def test_forged_required_without_gated_tool_does_not_ask(self, build, selected):
        executions: list[str] = []
        asks: list[str] = []
        agent = build(
            tools=_registry(executions, "danger", "safe"),
            config=_config(),
            llm_interface=_ForgedRequiredLLM(lambda: selected),
            hitl=HumanInTheLoop(
                approval_policy=lambda call, ctx: call.tool_name == "danger",
                approval_callback=lambda request: (
                    asks.append(request.tool_name) or True
                ),
            ),
        )
        agent.run("do it")  # finite: no BudgetExhaustedError

        assert asks == [], f"asked about a call that needs no approval: {asks}"
        assert "danger" not in executions

    def test_forged_required_with_gated_tool_still_asks(self, build):
        executions: list[str] = []
        asks: list[str] = []
        agent = build(
            tools=_registry(executions, "danger"),
            config=_config(),
            llm_interface=_ForgedRequiredLLM(lambda: "danger"),
            hitl=_hitl(lambda n: False, asks),
        )
        agent.run("do it")

        assert asks and set(asks) == {"danger"}
        assert executions == []
