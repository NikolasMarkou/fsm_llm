"""
Composition helpers that wire existing agents together.

Two reusable building blocks for the "Claude-like" patterns:

- :func:`react_worker_factory` — a ``worker_factory`` for
  :class:`OrchestratorAgent` that solves each delegated subtask with a fresh
  tool-using :class:`ReactAgent`. This is the lowest-friction decomposition +
  tool-use composition: the orchestrator splits the query, ReAct handles each
  piece with tools.
- :func:`default_llm_judge` — an LLM-as-judge ``evaluation_fn`` compatible with
  :class:`EvaluatorOptimizerAgent` (and usable anywhere an
  ``(output, context) -> EvaluationResult`` callable fits). Closes the gap that
  EvaluatorOptimizer and Reflexion (the patterns that take an ``evaluation_fn``)
  need a hand-written evaluator.

Both are additive and import-light; nothing else depends on them.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from fsm_llm.definitions import CompletionRequest
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.logging import logger
from fsm_llm.utilities import coerce_confidence, extract_json_from_text

from .definitions import (
    AgentConfig,
    AgentResult,
    EvaluationResult,
    resolve_agent_model,
)
from .exceptions import EvaluationError
from .react import ReactAgent
from .tools import ToolRegistry

WorkerFn = Callable[[str], AgentResult]
JudgeFn = Callable[[str, dict[str, Any]], EvaluationResult]


def react_worker_factory(
    tools: ToolRegistry,
    config: AgentConfig | None = None,
    agent_cls: type[ReactAgent] = ReactAgent,
    **agent_kwargs: Any,
) -> WorkerFn:
    """Build an OrchestratorAgent ``worker_factory`` backed by ReactAgent.

    A *fresh* agent is created per subtask (the orchestrator runs workers one
    after the other), so no subtask shares the handler bookkeeping of another.

    Example::

        from fsm_llm.agents import OrchestratorAgent, AgentConfig
        from fsm_llm.agents.composition import react_worker_factory

        worker = react_worker_factory(registry, AgentConfig(model=model))
        agent = OrchestratorAgent(worker_factory=worker)
        result = agent.run("Research X, then summarize its impact on Y.")
    """

    def worker(subtask: str) -> AgentResult:
        agent = agent_cls(tools=tools, config=config, **agent_kwargs)
        return agent.run(subtask)

    return worker


_JUDGE_PROMPT = (
    "You are a strict evaluator. Score the following answer to the task on a "
    "scale from 0.0 (useless) to 1.0 (excellent){criteria_clause}.\n\n"
    "TASK/CONTEXT:\n{task}\n\nANSWER:\n{output}\n\n"
    'Respond with ONLY a JSON object: {{"score": <float 0-1>, '
    '"feedback": "<one or two sentences of actionable feedback>"}}'
)


def _default_complete(model: str, prompt: str) -> str:
    """Send ``prompt`` as one user turn to ``model`` at temperature 0.0.

    The call goes through core's ``LiteLLMInterface.complete``, so the one
    request builder applies (connection kwargs, timeout, and on Ollama models
    thinking off and ``/nothink`` on the user turn). Returns the reply text,
    ``""`` when the model wrote none. A provider failure raises
    ``EvaluationError`` chained from core's ``LLMResponseError``.
    """
    # DECISION plan-2026-10-01T093600-944e2692/D-007: the judge sends through
    # core's LiteLLMInterface.complete. Do NOT import the provider library
    # here or call a provider binding directly, and do NOT add a "skip Ollama
    # preparation" option to keep the old thinking-on judge: that is a second
    # request path and the flag sprawl the user ruled out. Ollama judges now
    # run with thinking off. See decisions.md D-007.
    request = CompletionRequest(messages=[{"role": "user", "content": prompt}])
    # DECISION plan-2026-07-20T040150-876e7164/D-006 [STALE]: wrap the LLM
    # boundary HERE so a provider outage leaves this function as an
    # AgentError subclass, never as core's LLMResponseError or a raw provider
    # error, and never as an empty answer. `EvaluationError` rather than the bare `AgentError` root
    # because this function exists only to back `default_llm_judge`: at this
    # call site an LLM failure IS an evaluation failure. The clause stays broad
    # because this is the provider boundary of the judge. Do NOT drop the
    # `from e`: `judge()` below renders the exception into its user-facing
    # feedback string, and the chain (EvaluationError <- LLMResponseError <-
    # provider error) keeps the provider's own message. See decisions.md D-006.
    try:
        response = LiteLLMInterface(model, temperature=0.0).complete(request)
    except Exception as e:
        raise EvaluationError(f"LLM judge completion failed: {e!s}") from e
    return response.text or ""


def default_llm_judge(
    model: str | None = None,
    criteria: str = "",
    threshold: float = 0.7,
    complete_fn: Callable[[str, str], str] | None = None,
) -> JudgeFn:
    """Build an LLM-as-judge ``evaluation_fn`` returning an EvaluationResult.

    Args:
        model: Provider model id used to grade (when ``complete_fn`` is None).
            ``None`` resolves now: env ``LLM_MODEL``, else ``DEFAULT_LLM_MODEL``.
        criteria: Optional extra grading criteria appended to the prompt.
        threshold: Score at/above which ``passed`` is True.
        complete_fn: Optional ``(model, prompt) -> str`` override (for tests or
            a custom backend). Defaults to one completion through core's
            ``LiteLLMInterface`` at temperature 0.0.

    The returned callable has signature ``(output, context) -> EvaluationResult``
    matching :class:`EvaluatorOptimizerAgent`.
    """
    criteria_clause = f", judged on: {criteria}" if criteria else ""
    complete = complete_fn or _default_complete
    model = resolve_agent_model(model)

    def judge(output: str, context: dict[str, Any] | None = None) -> EvaluationResult:
        task = ""
        if context:
            task = str(context.get("task") or "")
        prompt = _JUDGE_PROMPT.format(
            criteria_clause=criteria_clause, task=task, output=output
        )
        try:
            raw = complete(model, prompt)
            parsed = extract_json_from_text(raw) or {}
        except Exception as e:
            logger.warning(f"LLM judge failed ({e}); defaulting to not-passed")
            return EvaluationResult(
                passed=False, score=0.0, feedback=f"judge error: {e}"
            )

        # coerce_confidence reads NaN and +-inf as 0.0, so a malformed score
        # fails the judge; a bool is no score either (float(True) is 1.0).
        # Do NOT fold this back into a bare max(0.0, min(1.0, float(...))):
        # min(1.0, nan) is 1.0 (see the anchor in utilities.coerce_confidence).
        score = 0.0
        raw_score = parsed.get("score", 0.0)
        if not isinstance(raw_score, bool):
            with contextlib.suppress(TypeError, ValueError):
                score = coerce_confidence(raw_score, 0.0)
        feedback = str(parsed.get("feedback", ""))
        return EvaluationResult(
            passed=score >= threshold, score=score, feedback=feedback
        )

    return judge
