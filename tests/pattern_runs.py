"""Scripted runs of the built-in agent patterns (plan a555dcfe, step 8).

One harness serves two goldens that guard the conversion of the pattern builders to files
(E4.2, steps 9 to 18): the handlers each pattern registers, read through the run's
``HandlerSystem`` (``handler_order()``), and the bytes of every request each pattern sends to a
scripted model (``request_bytes()``). Both come from the same runs, so one fixture drift shows in
both. Held by ``tests/test_fsm_llm_agents/test_pattern_handler_order.py`` and
``test_pattern_request_bytes.py``; the fixtures are written by ``python -m tests.executables_corpus
--write`` (never to make a failure go away: a difference is a pattern that now registers or sends
something else).

Interface contract:

- ``CASES``: ``{case name: Case}``, built lazily; ``Case(build, task, native)`` where ``build()``
  returns a fresh agent wired to a fresh ``ScriptedModel`` (the second item of the pair), so no
  state is shared between cases or runs.
- ``run_case(name)``: ``{"handlers": [...], "requests": [...], "outcome": {...}}`` for one run.
  ``handlers`` are tuples of the registered handlers in execution order (name, timings, states,
  target states, watched keys, priority); ``requests`` one ``{"call", "label", "sha256"}`` per
  request the model got, hashed over its JSON with the clock frozen and timestamps and uuids
  normalised; ``outcome`` is the agent result's answer, status and tool trace. Failure mode: a
  pattern that raises is recorded as ``{"raised": <type name>}``, never propagated.
- ``handler_order()`` and ``request_bytes()``: the two fixtures as plain dicts.
- ``scripted_model`` is NOT thread safe by design; the patterns that fan out (orchestrator) run
  their workers on a stub that never reaches the model.

The ``native_fc`` case drives the litellm stub of ``test_native_fc_golden.py`` (its request bytes
are pinned there, not here, so ``request_bytes()`` leaves them out).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest import mock

from fsm_llm import agents as A
from fsm_llm.agents.base.agent import BaseAgent
from fsm_llm.agents.definitions import AgentConfig, ChainStep, EvaluationResult
from fsm_llm.core.definitions import (
    DataExtractionResponse,
    FieldExtractionResponse,
    LLMResponseError,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
from fsm_llm.policy import HumanInTheLoop
from fsm_llm.testing import hash_embed
from fsm_llm.tools import ToolRegistry
from tests.test_fsm_llm.kind_corpus import _frozen_today

_TS = re.compile(r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)?")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_BULK_NAME = re.compile(r'- "(\w+)"')
_TASK = "Find the alpha thing, then report it"


def normalise(text: str) -> str:
    """Timestamps and uuids of a rendered request replaced by fixed marks."""
    return _UUID.sub("<uuid>", _TS.sub("<ts>", text))


class ScriptedModel(LLMInterface):
    """A deterministic double: counts per field name, answers by name and counter.

    ``tool_name`` is the first registered tool, ``should_terminate`` is False once and True
    from the second ask, ``evaluation_passed`` is True from the second ask, every other
    field answers by its declared type. ``log`` holds ``(call, request json)`` per request.
    """

    def __init__(self, tool: str = "search") -> None:
        self.model = "mock-model"
        self.tool = tool
        self.log: list[tuple[str, dict[str, Any]]] = []
        self.counts: dict[str, int] = {}

    def _note(self, call: str, request: Any) -> None:
        self.log.append((call, json.loads(request.model_dump_json())))

    def extract_field(self, request: Any) -> FieldExtractionResponse:
        self._note("extract_field", request)
        name = request.field_name
        self.counts[name] = n = self.counts.get(name, 0) + 1
        kind = str(getattr(request, "field_type", "any"))
        if name == "tool_name":
            value: Any = self.tool
        elif name == "tool_input":
            value = {"x": "1"}
        elif name == "plan_blueprint":
            value = [
                {
                    "plan_id": "#E1",
                    "tool_name": self.tool,
                    "tool_input": {"x": "1"},
                    "description": "look it up",
                }
            ]
        elif name == "tool_calls":
            value = [{"tool_name": self.tool, "tool_input": {"x": "1"}}]
        elif name == "should_terminate":
            value = n >= 2
        elif name.endswith("_passed") or name == "approved":
            value = n >= 2
        elif "float" in kind or name.endswith("score"):
            value = 0.9
        elif "bool" in kind:
            value = n % 2 == 0
        elif "list" in kind:
            value = ["step one", "step two"]
        else:
            value = "text"
        return FieldExtractionResponse(
            field_name=name, value=value, confidence=0.9, reasoning="m", is_valid=True
        )

    def extract_bulk_data(self, request: Any) -> DataExtractionResponse:
        self._note("extract_bulk_data", request)
        names = _BULK_NAME.findall(request.system_prompt)
        return DataExtractionResponse(extracted_data={n: "text" for n in names})

    def generate_response(self, request: Any) -> ResponseGenerationResponse:
        self._note("generate_response", request)
        return ResponseGenerationResponse(
            message="Answer: ok", message_type="response", reasoning="m"
        )

    def complete(self, request: Any) -> Any:
        # No classifier reply is scripted: a classification fails soft, as on an outage.
        raise LLMResponseError("ScriptedModel scripts no completion replies")


@dataclass(frozen=True)
class Case:
    """One scripted run: ``build(model)`` makes the agent, ``task`` is what it runs.

    ``stream`` runs ``run_stream`` instead of ``run`` (the outcome is then the normalised chunks);
    ``extra(agent)`` adds fields to the outcome after the run (what a memory backend stored).
    """

    build: Callable[[ScriptedModel], BaseAgent]
    task: str = _TASK
    tool: str = "search"
    native: bool = False
    stream: bool = False
    extra: Callable[[BaseAgent], dict[str, Any]] | None = None


def _tools(*, flagged: bool = False, many: bool = False) -> ToolRegistry:
    registry = ToolRegistry()
    names = ("search", "calc") if many else ("search",)
    for name in names:
        registry.register_function(
            lambda params: "result",
            name=name,
            description=f"Tool {name} {{x}} \u00e9",
            parameter_schema={"properties": {"x": {"type": "string"}}},
            requires_approval=flagged and name == "search",
        )
    return registry


def _semantic() -> ToolRegistry:
    from fsm_llm.agents.semantic_tools import SemanticToolRegistry

    registry = SemanticToolRegistry(embed_fn=hash_embed(8))
    words = ("alpha", "beta", "gamma", "delta")
    for i in range(12):
        registry.register_function(
            lambda params: "result",
            name=f"t{i}",
            description=f"{words[i % 4]} helper {i}",
        )
    return registry


def _hitl() -> HumanInTheLoop:
    return HumanInTheLoop(
        approval_policy=lambda call, ctx: True, approval_callback=lambda request: True
    )


def _config(model: str = "mock-model") -> AgentConfig:
    return AgentConfig(max_iterations=5, model=model)


def _chain() -> list[ChainStep]:
    return [
        ChainStep(
            step_id="a",
            name="a",
            extraction_instructions="extract a",
            response_instructions="respond a",
            validation_fn=lambda ctx: True,
        ),
        ChainStep(
            step_id="b",
            name="b",
            extraction_instructions="extract b",
            response_instructions="respond b",
        ),
    ]


def _passing(output: str, context: dict[str, Any]) -> EvaluationResult:
    return EvaluationResult(passed=True, score=1.0, feedback="fine")


def _kw(model: ScriptedModel) -> dict[str, Any]:
    return {"config": _config(), "llm_interface": model}


def _rejecting(accept_from: int) -> Callable[[str, dict[str, Any]], bool]:
    """A verifier that rejects the first ``accept_from - 1`` answers (a fresh counter per build)."""
    seen: list[str] = []

    def verify(answer: str, context: dict[str, Any]) -> bool:
        seen.append(answer)
        return len(seen) >= accept_from

    return verify


def _memory_store() -> Any:
    from fsm_llm.agents import SemanticMemoryStore

    store = SemanticMemoryStore(embed_fn=hash_embed(8))
    store.add("The alpha thing is kept on the second shelf.")
    store.add("Beta is unrelated to the report.")
    return store


def _stored(agent: BaseAgent) -> dict[str, Any]:
    """What the auto-memory store holds after the run (the remember text)."""
    store = agent.memory  # type: ignore[attr-defined]
    return {
        "memory": sorted(text for text, _score, _meta in store.search("alpha", k=50))
    }


def _verified(m: ScriptedModel, *, accept_from: int, **kw: Any) -> BaseAgent:
    config = AgentConfig(
        max_iterations=5, model="mock-model", verification_fn=_rejecting(accept_from)
    )
    return A.VerifiedReactAgent(
        _tools(), config=config, llm_interface=m, max_verify_retries=1, **kw
    )


def _memory_agent(m: ScriptedModel, floor: float = 0.0) -> BaseAgent:
    return A.AutoMemoryReactAgent(
        _tools(), memory=_memory_store(), recall_min_score=floor, **_kw(m)
    )


def _cases() -> dict[str, Case]:
    return {
        **_base_cases(),
        **_blind_spot_cases(),
    }


def _blind_spot_cases() -> dict[str, Case]:
    """The paths the first golden did not cover (step 28.5 of plan a555dcfe, D-035): the retry note
    of ``verified_react``, the recall and remember texts of ``auto_memory``, the stream of every
    pattern that streams, a semantic registry under ``rewoo`` and ``plan_execute``, custom debate
    personas. Their goldens come from the tree BEFORE the conversion (see the test module)."""
    return {
        "verified_react_retry": Case(lambda m: _verified(m, accept_from=2)),
        "verified_react_rejected": Case(lambda m: _verified(m, accept_from=99)),
        "auto_memory": Case(_memory_agent, extra=_stored),
        "react_stream": Case(
            lambda m: A.ReactAgent(_tools(many=True), **_kw(m)), stream=True
        ),
        "verified_react_stream": Case(
            lambda m: _verified(m, accept_from=2), stream=True
        ),
        # a floor between the two stored scores (0.834 and 0.816 for the task): one memory recalled
        "auto_memory_floor": Case(lambda m: _memory_agent(m, 0.82), extra=_stored),
        "auto_memory_stream": Case(_memory_agent, stream=True, extra=_stored),
        "rewoo_semantic": Case(
            lambda m: A.REWOOAgent(_semantic(), **_kw(m)), tool="t0"
        ),
        "plan_execute_semantic": Case(
            lambda m: A.PlanExecuteAgent(_semantic(), **_kw(m)), tool="t0"
        ),
        "debate_personas": Case(
            lambda m: A.DebateAgent(
                num_rounds=2,
                proposer_persona="A cautious systems engineer",
                critic_persona="A bold product designer",
                judge_persona="A neutral technical editor",
                **_kw(m),
            )
        ),
    }


def _new_stream_cases() -> dict[str, Case]:
    """Streams of the patterns whose OLD classes had no ``run_stream`` (``AttributeError`` before the
    conversion, a stream now): there is no parent golden, so the rule is that a stream sends the very
    requests the same pattern's ``run`` sends (``test_pattern_request_bytes``)."""
    return {
        "reflexion_stream": Case(
            lambda m: A.ReflexionAgent(_tools(many=True), **_kw(m)), stream=True
        ),
        "plan_execute_stream": Case(
            lambda m: A.PlanExecuteAgent(_tools(many=True), **_kw(m)), stream=True
        ),
        "maker_checker_stream": Case(
            lambda m: A.MakerCheckerAgent("make it", "check it", **_kw(m)), stream=True
        ),
        "debate_stream": Case(lambda m: A.DebateAgent(**_kw(m)), stream=True),
        "evaluator_optimizer_stream": Case(
            lambda m: A.EvaluatorOptimizerAgent(_passing, **_kw(m)), stream=True
        ),
        "prompt_chain_stream": Case(
            lambda m: A.PromptChainAgent(_chain(), **_kw(m)), stream=True
        ),
        "orchestrator_stream": Case(
            lambda m: A.OrchestratorAgent(
                worker_factory=lambda sub: A.AgentResult(answer="done"), **_kw(m)
            ),
            stream=True,
        ),
    }


def _base_cases() -> dict[str, Case]:
    return {
        "react": Case(lambda m: A.ReactAgent(_tools(many=True), **_kw(m))),
        "react_approval": Case(
            lambda m: A.ReactAgent(_tools(flagged=True), hitl=_hitl(), **_kw(m))
        ),
        "react_classification": Case(
            lambda m: A.ReactAgent(_tools(), use_classification=True, **_kw(m))
        ),
        "react_semantic": Case(
            lambda m: A.ReactAgent(_semantic(), **_kw(m)), tool="t0"
        ),
        "reflexion": Case(lambda m: A.ReflexionAgent(_tools(many=True), **_kw(m))),
        "reflexion_approval": Case(
            lambda m: A.ReflexionAgent(_tools(flagged=True), hitl=_hitl(), **_kw(m))
        ),
        "rewoo": Case(lambda m: A.REWOOAgent(_tools(many=True), **_kw(m))),
        "debate": Case(lambda m: A.DebateAgent(**_kw(m))),
        "plan_execute": Case(lambda m: A.PlanExecuteAgent(_tools(many=True), **_kw(m))),
        "plan_execute_empty": Case(lambda m: A.PlanExecuteAgent(None, **_kw(m))),
        "prompt_chain": Case(lambda m: A.PromptChainAgent(_chain(), **_kw(m))),
        "evaluator_optimizer": Case(
            lambda m: A.EvaluatorOptimizerAgent(_passing, **_kw(m))
        ),
        "maker_checker": Case(
            lambda m: A.MakerCheckerAgent("make it", "check it", **_kw(m))
        ),
        "orchestrator": Case(
            lambda m: A.OrchestratorAgent(
                worker_factory=lambda sub: A.AgentResult(answer="done"),
                **_kw(m),
            )
        ),
        "parallel_react": Case(
            lambda m: A.ParallelReactAgent(_tools(many=True), **_kw(m))
        ),
        "native_fc": Case(lambda m: _native_agent(), native=True),
    }


def _native_agent() -> BaseAgent:
    from tests.test_fsm_llm_agents import test_native_fc_golden as golden

    return A.NativeFunctionCallingReactAgent(
        golden._registry(), config=AgentConfig(model="gpt-4o-mini", max_iterations=5)
    )


CASES = _cases


def handler_tuple(handler: Any) -> list[Any]:
    """The comparable shape of one registered handler (JSON friendly, sorted sets)."""
    return [
        handler.name,
        sorted(t.value for t in (handler.timings or ())),
        sorted(handler.states),
        sorted(handler.target_states),
        sorted(handler.updated_keys),
        handler.priority,
    ]


def _label(call: str, request: dict[str, Any]) -> str:
    """A short readable name of one request, for a diff of the fixture."""
    if call == "extract_field":
        return f"field:{request.get('field_name')}"
    if call == "extract_bulk_data":
        return "bulk"
    return "reply"


def _request_row(call: str, request: dict[str, Any]) -> dict[str, str]:
    text = normalise(json.dumps(request, sort_keys=True, ensure_ascii=True))
    return {
        "call": call,
        "label": _label(call, request),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def _outcome(result: Any) -> dict[str, Any]:
    trace = result.trace
    return {
        "answer": normalise(str(result.answer)),
        "status": str(getattr(result.status, "value", result.status)),
        "stop_reason": str(getattr(result.stop_reason, "value", result.stop_reason)),
        "tools": [[c.tool_name, c.parameters] for c in trace.tool_calls],
    }


def run_case(name: str) -> dict[str, Any]:
    """Run case *name* once and return its handlers, request hashes and outcome."""
    case = {**_cases(), **_new_stream_cases()}[name]
    seen: list[list[Any]] = []
    original = BaseAgent._prepare_run_api

    def _spy(self: Any, fsm_def: Any, agent_type: Any, handlers: Any) -> Any:
        api = original(self, fsm_def, agent_type, handlers)
        seen.append([handler_tuple(h) for h in api.handler_system.handlers])
        return api

    model = ScriptedModel(case.tool)
    with _frozen_today(), mock.patch.object(BaseAgent, "_prepare_run_api", _spy):
        if case.native:
            result_or_error, requests = _run_native(case)
        else:
            agent = case.build(model)
            try:
                if case.stream:
                    chunks = [normalise(c) for c in agent.run_stream(case.task)]
                    result_or_error = {"chunks": chunks}
                else:
                    result_or_error = agent.run(case.task)
            except Exception as exc:  # recorded, never propagated
                result_or_error = {"raised": type(exc).__name__}
            requests = [_request_row(call, req) for call, req in model.log]
    outcome = (
        result_or_error
        if isinstance(result_or_error, dict)
        else _outcome(result_or_error)
    )
    if case.extra is not None and not case.native:
        outcome = {**outcome, **case.extra(agent)}
    return {
        "handlers": seen[0] if len(seen) == 1 else seen,
        "requests": requests,
        "outcome": outcome,
    }


def _run_native(case: Case) -> tuple[Any, list[Any]]:
    from tests.test_fsm_llm_agents import test_native_fc_golden as golden

    recorder = golden._Recorder(
        [golden._calls(golden._PARIS), golden._final("It is sunny in Paris.")]
    )
    with (
        mock.patch(golden._BINDINGS[0], recorder),
        mock.patch(golden._BINDINGS[1], recorder),
    ):
        result = case.build(ScriptedModel()).run("What is the weather in Paris?")
    return result, []


def handler_order() -> dict[str, Any]:
    """``case -> the handlers of one run in execution order``: the handler-order fixture."""
    return {name: run_case(name)["handlers"] for name in _cases()}


def request_bytes() -> dict[str, Any]:
    """``case -> {requests, outcome}`` of one scripted run: the request-byte fixture.

    ``native_fc`` is left out: ``test_native_fc_golden.py`` pins its bytes.
    """
    out = {}
    for name, case in _cases().items():
        if not case.native:
            run = run_case(name)
            out[name] = {"requests": run["requests"], "outcome": run["outcome"]}
    return out
