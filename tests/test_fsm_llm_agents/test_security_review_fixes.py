"""Security review fixes (plan-2026-09-29T103145-06a5ec0a, completion fix 3.1).

Each class pins one review finding (findings/review-iter-1.md) and fails on
the parent commit ed6f69d:

1. Pattern run outputs (drafts, verdicts, answers) cannot be forged through
   ``initial_context``, ``AgentServer``, AgentGraph edges or Swarm hand-offs.
2. A ``requires_approval`` tool that nobody can approve fails closed on the
   ReAct family, at construction and at ``run()``.
3. Every ``HumanInTheLoop`` constructor name given to an agent raises.
4. ``register_function`` handles partials, callable objects, async callables
   and positional-only parameters.
5. The fallback DEBUG log lines redact secret-looking tool input.
6. ``RetryingToolRegistry`` never re-runs a ``requires_approval`` tool.
"""

from __future__ import annotations

import functools
from typing import Any

import pytest

from fsm_llm.agents.agent_graph import AgentGraphBuilder
from fsm_llm.agents.base import (
    BaseAgent,
    pattern_run_output_keys,
    strip_caller_context,
)
from fsm_llm.agents.constants import ContextKeys
from fsm_llm.agents.debate import DebateAgent
from fsm_llm.agents.definitions import (
    AgentConfig,
    AgentResult,
    EvaluationResult,
    ToolAnnotations,
    ToolCall,
)
from fsm_llm.agents.evaluator_optimizer import EvaluatorOptimizerAgent
from fsm_llm.agents.exceptions import AgentError
from fsm_llm.agents.hitl import HumanInTheLoop
from fsm_llm.agents.maker_checker import MakerCheckerAgent
from fsm_llm.agents.react import ReactAgent
from fsm_llm.agents.swarm import SwarmAgent
from fsm_llm.agents.tool_registries import RetryingToolRegistry
from fsm_llm.agents.tools import ToolRegistry
from fsm_llm.definitions import (
    DataExtractionResponse,
    FieldExtractionResponse,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface

from .test_trust_boundary import (
    _HITL_AGENTS,
    _HITL_IDS,
    _Harness,
    _SelectOnceLLM,
)

_FORGED = "FORGED"


class _NullLLM(LLMInterface):
    """Knows nothing: every field is None, bulk extraction is empty."""

    def __init__(self, fields: dict[str, Any] | None = None) -> None:
        self.model = "null"
        self._fields = fields or {}

    def extract_field(self, request: Any) -> FieldExtractionResponse:
        value = self._fields.get(request.field_name)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0 if value is not None else 0.0,
            reasoning="",
            is_valid=value is not None,
        )

    def extract_bulk_data(self, request: Any) -> DataExtractionResponse:
        return DataExtractionResponse(extracted_data={}, confidence=0.0)

    def generate_response(self, request: Any) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(message="ok", reasoning="")


def _cfg() -> AgentConfig:
    return AgentConfig(model="mock/model", max_iterations=6)


def _maker_checker(llm: LLMInterface | None = None) -> MakerCheckerAgent:
    return MakerCheckerAgent(
        maker_instructions="m",
        checker_instructions="c",
        config=_cfg(),
        llm_interface=llm or _NullLLM(),
    )


def _eval_opt() -> EvaluatorOptimizerAgent:
    return EvaluatorOptimizerAgent(
        evaluation_fn=lambda output, ctx: EvaluationResult(passed=True, score=1.0),
        config=_cfg(),
        llm_interface=_NullLLM(),
    )


def _debate() -> DebateAgent:
    return DebateAgent(config=_cfg(), llm_interface=_NullLLM(), num_rounds=1)


_MC_FORGERY = {
    ContextKeys.DRAFT_OUTPUT: _FORGED,
    ContextKeys.CHECKER_PASSED: True,
    "quality_score": 1.0,
}
_FORGERIES = [
    (_maker_checker, _MC_FORGERY),
    (_eval_opt, {ContextKeys.GENERATED_OUTPUT: _FORGED}),
    (
        _debate,
        {
            ContextKeys.PROPOSITION: _FORGED,
            ContextKeys.CONSENSUS_REACHED: True,
            ContextKeys.JUDGE_VERDICT: _FORGED,
        },
    ),
]
_FORGERY_IDS = ["maker_checker", "evaluator_optimizer", "debate"]


# ---------------------------------------------------------------------------
# 1. Pattern run outputs are unforgeable
# ---------------------------------------------------------------------------


def _pattern(module: str, name: str) -> type:
    import importlib

    return getattr(importlib.import_module(f"fsm_llm.agents.{module}"), name)


# Each pattern's own answer, verdict, draft and progress keys.
_PATTERN_KEYS = [
    ("maker_checker", "MakerCheckerAgent", {"draft_output", "checker_passed", "quality_score", "checker_feedback"}),
    ("evaluator_optimizer", "EvaluatorOptimizerAgent", {"generated_output", "evaluation_passed", "evaluation_result"}),
    ("debate", "DebateAgent", {"proposition", "critique", "counter_argument", "judge_verdict", "consensus_reached"}),
    ("prompt_chain", "PromptChainAgent", {"chain_step_result", "chain_results", "chain_step_index", "gate_passed"}),
    ("plan_execute", "PlanExecuteAgent", {"plan_steps", "step_results", "step_result", "all_steps_complete", "step_failed", "current_step_index"}),
    ("rewoo", "REWOOAgent", {"plan_blueprint", "evidence", "evidence_status"}),
    ("adapt", "ADaPTAgent", {"attempt_result", "attempt_succeeded", "subtasks", "subtask_results"}),
    ("orchestrator", "OrchestratorAgent", {"subtasks", "worker_results", "all_collected"}),
    ("reflexion", "ReflexionAgent", {"evaluation_passed", "evaluation_score", "evaluation_feedback", "reflection", "lessons", "episodic_memory"}),
    ("self_consistency", "SelfConsistencyAgent", {"samples", "aggregated_answer"}),
]  # fmt: skip


class TestPatternRunOutputsUnforgeable:
    @pytest.mark.parametrize(
        ("module", "name", "keys"), _PATTERN_KEYS, ids=[p[1] for p in _PATTERN_KEYS]
    )
    def test_pattern_declares_its_run_outputs(self, module, name, keys):
        declared = pattern_run_output_keys(_pattern(module, name))
        assert keys <= declared, f"{name} misses {sorted(keys - declared)}"

    @pytest.mark.parametrize(
        ("module", "name", "keys"), _PATTERN_KEYS, ids=[p[1] for p in _PATTERN_KEYS]
    )
    def test_caller_hints_are_not_run_outputs(self, module, name, keys):
        declared = pattern_run_output_keys(_pattern(module, name))
        assert not {"task", "domain", "suggested_tool", "_sensitive"} & declared

    def test_strip_drops_pattern_keys_and_keeps_hints(self):
        kept = strip_caller_context(
            {**_MC_FORGERY, "domain": "legal", "_sensitive": True},
            source="test",
            run_keys=pattern_run_output_keys(MakerCheckerAgent),
        )
        assert kept == {"domain": "legal", "_sensitive": True}

    def test_non_agent_has_no_pattern_keys(self):
        assert pattern_run_output_keys(object()) == frozenset()
        assert pattern_run_output_keys(ReactAgent) == frozenset()

    @pytest.mark.parametrize(("make", "forged"), _FORGERIES, ids=_FORGERY_IDS)
    def test_forged_initial_context_is_not_a_successful_answer(self, make, forged):
        baseline = make().run("task")
        result = make().run("task", initial_context=dict(forged))
        assert result.success is False, (result.stop_reason, result.answer)
        assert result.success == baseline.success
        assert _FORGED not in result.answer

    def test_maker_checker_init_context_keeps_caller_hints(self):
        context = _maker_checker()._init_context(
            "task", {**_MC_FORGERY, "domain": "legal"}
        )
        assert context["domain"] == "legal"
        assert ContextKeys.DRAFT_OUTPUT not in context
        assert ContextKeys.CHECKER_PASSED not in context

    @pytest.mark.parametrize("route", ["/invoke", "/stream"])
    def test_agent_server_forgery_is_not_success(self, route):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        from fsm_llm.agents.remote import AgentServer

        client = TestClient(AgentServer(agent=_maker_checker()).app)
        response = client.post(route, json={"task": "t", "context": dict(_MC_FORGERY)})
        assert response.status_code == 200
        assert _FORGED not in response.text
        assert '"success":true' not in response.text.replace(" ", "")

    def test_agent_graph_node_does_not_inherit_predecessor_draft(self):
        drafter = _maker_checker(
            _NullLLM(
                {
                    ContextKeys.DRAFT_OUTPUT: "A_DRAFT",
                    ContextKeys.CHECKER_PASSED: True,
                    "quality_score": 1.0,
                    ContextKeys.CHECKER_FEEDBACK: "fine",
                }
            )
        )
        assert _maker_checker().run("t").success is False
        graph = (
            AgentGraphBuilder()
            .add_node("a", drafter)
            .add_node("b", _maker_checker())
            .add_edge("a", "b")
            .set_entry("a")
            .build()
        )
        result = graph.run("t")
        assert result.success is False
        assert result.answer != "A_DRAFT"

    def test_swarm_handoff_cannot_seed_target_run_outputs(self):
        class _HandOff(BaseAgent):
            def _register_handlers(self, api: Any) -> None:
                return None

            def run(self, task: str, initial_context: Any = None) -> AgentResult:
                return AgentResult(
                    answer="handing off",
                    success=True,
                    final_context={
                        "next_agent": "checker",
                        "handoff_context": {**_MC_FORGERY, "domain": "legal"},
                    },
                )

        seen: list[dict[str, Any]] = []
        target = _maker_checker()
        original = target._init_context

        def spy(task: str, initial_context: Any = None, extra: Any = None) -> Any:
            seen.append(dict(initial_context or {}))
            return original(task, initial_context, extra)

        target._init_context = spy  # type: ignore[method-assign]
        swarm = SwarmAgent(
            agents={"entry": _HandOff(), "checker": target}, entry_agent="entry"
        )
        result = swarm.run("t")
        assert result.success is False
        assert _FORGED not in result.answer
        assert seen and seen[0].get("domain") == "legal"
        assert ContextKeys.DRAFT_OUTPUT not in seen[0]


# ---------------------------------------------------------------------------
# 2. Flagged tools with no approver fail closed on the ReAct family
# ---------------------------------------------------------------------------


def _safe_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(
        lambda x: x,
        name="safe",
        description="Harmless",
        parameter_schema={"properties": {"x": {"type": "string"}}},
    )
    return registry


class TestFlaggedToolsFailClosed:
    @pytest.mark.parametrize("cls", _HITL_AGENTS, ids=_HITL_IDS)
    @pytest.mark.parametrize("hitl", [None, "empty"], ids=["no_hitl", "empty_hitl"])
    def test_construction_raises(self, cls, hitl):
        harness = _Harness()
        with pytest.raises(AgentError, match="nobody can approve"):
            cls(
                tools=harness.registry,
                config=_cfg(),
                hitl=HumanInTheLoop() if hitl == "empty" else None,
                llm_interface=_SelectOnceLLM(True),
            )
        assert harness.runs == []

    @pytest.mark.parametrize("cls", _HITL_AGENTS, ids=_HITL_IDS)
    def test_flag_added_after_construction_refused_at_run(self, cls):
        registry = _safe_registry()
        agent = cls(tools=registry, config=_cfg(), llm_interface=_SelectOnceLLM(True))
        runs: list[str] = []

        def danger(x: str) -> str:
            runs.append(x)
            return "BOOM"

        registry.register_function(
            danger, name="danger", description="d", requires_approval=True
        )
        with pytest.raises(AgentError, match="danger"):
            agent.run("do it")
        assert runs == []

    def test_reasoning_react_sees_a_flag_re_registered_on_the_caller_registry(self):
        pytest.importorskip("fsm_llm.reasoning")
        from fsm_llm.agents.reasoning_react import ReasoningReactAgent

        harness = _Harness()
        registry = _safe_registry()
        agent = ReasoningReactAgent(
            tools=registry,
            config=_cfg(),
            hitl=HumanInTheLoop(approval_callback=harness.deny),
            llm_interface=_SelectOnceLLM(True, tool="safe"),
        )
        runs: list[str] = []

        def safe(x: str) -> str:
            runs.append(x)
            return "BOOM"

        registry.register_function(
            safe, name="safe", description="d", requires_approval=True
        )
        assert agent._hitl_active is True
        agent.run("do it")
        assert harness.asks == ["safe"]
        assert runs == []

    def test_run_stream_refuses_too(self):
        registry = _safe_registry()
        agent = ReactAgent(
            tools=registry, config=_cfg(), llm_interface=_SelectOnceLLM(True)
        )
        registry.register_function(
            lambda x: x, name="danger", description="d", requires_approval=True
        )
        with pytest.raises(AgentError, match="danger"):
            list(agent.run_stream("do it"))

    def test_policy_without_callback_still_constructs(self):
        # An explicit policy owns its decision (D-004); a call it gates raises
        # ApprovalDeniedError instead.
        harness = _Harness()
        ReactAgent(
            tools=harness.registry,
            config=_cfg(),
            hitl=HumanInTheLoop(approval_policy=lambda call, ctx: True),
            llm_interface=_SelectOnceLLM(True),
        )

    def test_callback_only_constructs_and_asks(self):
        harness = _Harness()
        agent = ReactAgent(
            tools=harness.registry,
            config=_cfg(),
            hitl=HumanInTheLoop(approval_callback=harness.deny),
            llm_interface=_SelectOnceLLM(True),
        )
        agent.run("do it")
        assert harness.asks == ["danger"]
        assert harness.runs == []

    def test_verified_react_inherits_the_refusal(self):
        from fsm_llm.agents.verified_react import VerifiedReactAgent

        with pytest.raises(AgentError, match="nobody can approve"):
            VerifiedReactAgent(
                tools=_Harness().registry,
                config=_cfg(),
                llm_interface=_SelectOnceLLM(True),
            )


# ---------------------------------------------------------------------------
# 3. Every HumanInTheLoop constructor name is a misplaced kwarg
# ---------------------------------------------------------------------------


_HITL_NAMES = [
    "approval_policy",
    "approval_callback",
    "on_escalation",
    "confidence_threshold",
    "approval_timeout",
]


class TestHitlKwargsRejected:
    @pytest.mark.parametrize("kwarg", _HITL_NAMES)
    def test_react_rejects(self, kwarg):
        with pytest.raises(TypeError, match=kwarg):
            ReactAgent(_safe_registry(), config=_cfg(), **{kwarg: lambda *a: False})

    @pytest.mark.parametrize("kwarg", _HITL_NAMES)
    def test_pattern_without_hitl_rejects(self, kwarg):
        with pytest.raises(TypeError, match=kwarg):
            MakerCheckerAgent(
                maker_instructions="m",
                checker_instructions="c",
                config=_cfg(),
                **{kwarg: lambda *a: False},
            )

    def test_every_hitl_parameter_is_covered(self):
        import inspect

        params = set(inspect.signature(HumanInTheLoop.__init__).parameters) - {"self"}
        assert params == set(_HITL_NAMES)


# ---------------------------------------------------------------------------
# 4. register_function for every callable shape
# ---------------------------------------------------------------------------


def _search(query: str, limit: int = 5) -> str:
    return f"{query}:{limit}"


async def _asearch(query: str) -> str:
    return f"aio {query}"


class _Obj:
    def __call__(self, query: str) -> str:
        return f"obj {query}"


class _AObj:
    async def __call__(self, query: str) -> str:
        return f"aobj {query}"


class _Meth:
    def go(self, query: str) -> str:
        return f"meth {query}"


def _posonly(query: str, /, limit: int = 5) -> str:
    return f"{query}:{limit}"


_SHAPES = [
    ("partial", functools.partial(_search, limit=2), {"query": "x"}, "x:2"),
    ("callable_object", _Obj(), {"query": "x"}, "obj x"),
    ("async_partial", functools.partial(_asearch), {"query": "x"}, "aio x"),
    ("async_callable_object", _AObj(), {"query": "x"}, "aobj x"),
    ("bound_method", _Meth().go, {"query": "x"}, "meth x"),
    ("positional_only", _posonly, {"query": "x"}, "x:5"),
    ("positional_only_both", _posonly, {"query": "x", "limit": 3}, "x:3"),
]


class TestRegisterFunctionShapes:
    @pytest.mark.parametrize(
        ("label", "fn", "params", "expected"), _SHAPES, ids=[s[0] for s in _SHAPES]
    )
    def test_registers_infers_and_executes(self, label, fn, params, expected):
        registry = ToolRegistry()
        registry.register_function(fn, name=label, description="d")
        schema = registry.get(label).parameter_schema
        assert schema["properties"]["query"] == {"type": "string"}
        result = registry.execute(ToolCall(tool_name=label, parameters=params))
        assert result.success, result.error
        assert result.result == expected

    def test_unnamed_callable_object_gets_its_type_name(self):
        registry = ToolRegistry()
        registry.register_function(_Obj(), description="d")
        assert "_Obj" in registry

    def test_positional_only_missing_value_is_a_failed_call(self):
        runs: list[str] = []

        def two(query: str, /, limit: int = 5) -> str:
            runs.append(query)
            return query

        registry = ToolRegistry()
        registry.register_function(two, name="two", description="d")
        result = registry.execute(ToolCall(tool_name="two", parameters={"limit": 3}))
        assert result.success is False
        assert runs == []


# ---------------------------------------------------------------------------
# 5. Fallback DEBUG log lines are redacted
# ---------------------------------------------------------------------------


_SECRET = "sk-live-ABCDEF1234567890abcdef"


def _debug_lines(action: Any) -> list[str]:
    from fsm_llm.logging import logger

    captured: list[str] = []
    logger.enable("fsm_llm")
    sink_id = logger.add(lambda m: captured.append(str(m)), level="DEBUG")
    try:
        action()
    finally:
        logger.remove(sink_id)
        logger.disable("fsm_llm")
    return captured


class TestFallbackLogsRedacted:
    def test_single_param_fallback(self):
        def login(api_key: str) -> str:
            return "ok"

        registry = ToolRegistry()
        registry.register_function(login, name="login", description="d")
        lines = _debug_lines(
            lambda: registry.execute(
                ToolCall(tool_name="login", parameters={"key": _SECRET})
            )
        )
        assert any("kwarg mismatch" in line for line in lines)
        assert not any(_SECRET in line for line in lines)

    def test_positional_mapping_fallback(self):
        def login(api_key: str, region: str = "eu") -> str:
            return "ok"

        registry = ToolRegistry()
        registry.register_function(login, name="login", description="d")
        lines = _debug_lines(
            lambda: registry.execute(
                ToolCall(tool_name="login", parameters={"key": _SECRET})
            )
        )
        assert any("positional mapping" in line for line in lines)
        assert not any(_SECRET in line for line in lines)


# ---------------------------------------------------------------------------
# 6. RetryingToolRegistry never re-runs a requires_approval tool
# ---------------------------------------------------------------------------


class TestRetryingRegistryRespectsApproval:
    @staticmethod
    def _failing(
        registry: RetryingToolRegistry, runs: list[str], flagged: bool
    ) -> None:
        def pay(amount: str) -> str:
            runs.append(amount)
            raise RuntimeError("gateway down")

        # Idempotent, so only the approval flag decides (step 5 retry rule).
        registry.register_function(
            pay,
            name="pay",
            description="d",
            requires_approval=flagged,
            annotations=ToolAnnotations(idempotent=True),
        )

    def test_flagged_tool_runs_once(self):
        registry = RetryingToolRegistry(max_retries=3)
        runs: list[str] = []
        self._failing(registry, runs, flagged=True)
        result = registry.execute(ToolCall(tool_name="pay", parameters={"amount": "9"}))
        assert result.success is False
        assert runs == ["9"]

    def test_unflagged_idempotent_tool_is_still_retried(self):
        registry = RetryingToolRegistry(max_retries=3)
        runs: list[str] = []
        self._failing(registry, runs, flagged=False)
        registry.execute(ToolCall(tool_name="pay", parameters={"amount": "9"}))
        assert runs == ["9"] * 4
