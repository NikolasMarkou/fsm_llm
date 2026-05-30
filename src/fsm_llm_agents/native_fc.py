from __future__ import annotations

"""
NativeFunctionCallingReactAgent — a ReAct loop using provider-native function
calling instead of JSON-in-prompt tool extraction.

The stock agents are provider-agnostic by design: tool selection is extracted
from the LLM's free-form output via the FSM pipeline (``prompts.py`` instructs
the model to emit ``{"tool_name": ...}`` as text). That is portable but loses
the reliability of native ``tools=[...]`` / ``tool_calls`` function calling that
OpenAI/Anthropic/many litellm providers support — the same mechanism Claude
uses for tools.

This agent runs its OWN loop directly against litellm with
``tools=registry.get_json_schemas()`` and parses structured ``tool_calls``. It
does NOT use the FSM 2-pass pipeline, so the core contract is untouched — this
is a fully additive, self-contained alternative for users who want native
function-calling fidelity (and a capable provider).

Example::

    from fsm_llm_agents import AgentConfig, ToolRegistry
    from fsm_llm_agents.native_fc import NativeFunctionCallingReactAgent

    agent = NativeFunctionCallingReactAgent(
        tools=registry, config=AgentConfig(model="gpt-4o-mini"),
    )
    result = agent.run("What is the weather in Paris?")

``complete_fn`` may be injected to test the loop without a live provider; it
takes ``(model, messages, tool_schemas)`` and returns a normalized dict
``{"content": str | None, "tool_calls": [{"id", "name", "arguments": dict}]}``.
"""

import json
import time
from collections.abc import Callable
from typing import Any

from fsm_llm import API
from fsm_llm.logging import logger

from .base import BaseAgent
from .definitions import AgentConfig, AgentResult, AgentTrace, ToolCall
from .exceptions import AgentError
from .tools import ToolRegistry

CompleteFn = Callable[[str, list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]

_SYSTEM_PROMPT = (
    "You are a capable AI agent. Use the provided tools to gather information "
    "and complete the task. Call tools when you need external data; when you "
    "have enough information, reply with the final answer and no further tool "
    "calls."
)


class NativeFunctionCallingReactAgent(BaseAgent):
    """ReAct loop driven by provider-native function calling.

    Args:
        tools: Tool registry (non-empty).
        config: AgentConfig (model, max_iterations, timeout, temperature, ...).
        complete_fn: Optional override ``(model, messages, tool_schemas) -> dict``
            for tests / custom backends. Defaults to a litellm completion.
    """

    def __init__(
        self,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        complete_fn: CompleteFn | None = None,
        **api_kwargs: Any,
    ) -> None:
        if len(tools) == 0:
            raise AgentError("Cannot create agent with empty tool registry")
        super().__init__(config, **api_kwargs)
        self.tools = tools
        self._complete_fn = complete_fn

    # BaseAgent abstract hook — this agent does not use the FSM pipeline.
    def _register_handlers(self, api: API) -> None:  # pragma: no cover - unused
        return None

    # --- LLM completion -------------------------------------------------
    def _complete(
        self, messages: list[dict[str, Any]], schemas: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if self._complete_fn is not None:
            return self._complete_fn(self.config.model, messages, schemas)
        return self._litellm_complete(messages, schemas)

    def _litellm_complete(
        self, messages: list[dict[str, Any]], schemas: list[dict[str, Any]]
    ) -> dict[str, Any]:
        import litellm

        response = litellm.completion(
            model=self.config.model,
            messages=messages,
            tools=schemas or None,
            tool_choice="auto" if schemas else None,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        msg = response.choices[0].message
        tool_calls: list[dict[str, Any]] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            raw_args = tc.function.arguments
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                {"id": tc.id, "name": tc.function.name, "arguments": args or {}}
            )
        return {"content": msg.content, "tool_calls": tool_calls}

    # --- run ------------------------------------------------------------
    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        start_time = time.monotonic()
        schemas = self.tools.get_json_schemas()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        trace_calls: list[ToolCall] = []
        max_iters = self.config.max_iterations
        answer = ""

        try:
            for iteration in range(1, max_iters + 1):
                self._check_budgets(start_time, iteration, max_iters)
                result = self._complete(messages, schemas)
                tool_calls = result.get("tool_calls") or []
                content = result.get("content")

                if not tool_calls:
                    answer = content or ""
                    break

                messages.append(self._assistant_message(content, tool_calls))
                for tc in tool_calls:
                    name = tc.get("name", "")
                    args = tc.get("arguments") or {}
                    if not isinstance(args, dict):
                        args = {}
                    call = ToolCall(tool_name=name, parameters=args)
                    exec_result = self.tools.execute(call)
                    observation = exec_result.summary
                    if not exec_result.success:
                        observation = f"[TOOL FAILED] {observation}"
                    trace_calls.append(call)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": observation,
                        }
                    )
            else:
                # Loop exhausted without a final (tool-call-free) answer.
                logger.warning(
                    "NativeFunctionCallingReactAgent hit max_iterations without "
                    "a final answer."
                )

            trace = AgentTrace(
                tool_calls=trace_calls,
                total_iterations=len(trace_calls) or 1,
            )
            success = bool(answer) or bool(trace_calls)
            structured = self._try_parse_structured_output(answer)
            return AgentResult(
                answer=answer,
                success=success,
                trace=trace,
                final_context={"task": task},
                structured_output=structured,
            )
        except AgentError:
            raise
        except Exception as e:
            from .exceptions import AgentTimeoutError, BudgetExhaustedError

            if isinstance(e, (AgentTimeoutError, BudgetExhaustedError)):
                raise
            raise AgentError(
                f"Native function-calling execution failed: {e}",
                details={"task": task},
            ) from e

    @staticmethod
    def _assistant_message(
        content: Any, tool_calls: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Reconstruct an OpenAI-format assistant message carrying tool_calls."""
        return {
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(tc.get("arguments") or {}),
                    },
                }
                for tc in tool_calls
            ],
        }
