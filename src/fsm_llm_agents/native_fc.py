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
from fsm_llm.ollama import (
    apply_ollama_params,
    is_ollama_model,
    prepare_ollama_messages,
)
from fsm_llm.utilities import _resolve_reasoning_trace

from .base import BaseAgent, _output_response_format
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

#: The user turn appended for the post-loop constrained-decoding repair (D-002).
#:
#: It is appended so the schema echo `prepare_ollama_messages` performs lands on
#: the LAST message rather than on the original task, which by then is buried
#: under the assistant/tool round-trips.
_REPAIR_PROMPT = (
    "Stop calling tools now. Using everything established above, give the "
    "final answer as a single JSON object matching the required schema, and "
    "nothing else."
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
        self,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._complete_fn is not None:
            return self._complete_fn(self.config.model, messages, schemas)
        return self._litellm_complete(messages, schemas, response_format)

    def _litellm_complete(
        self,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import litellm

        call_params: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        # DECISION plan-2026-07-21T191807-bf7ffe24/D-002
        # The two keys are MUTUALLY EXCLUSIVE and the omission is structural,
        # not cosmetic: `tools=` and `response_format=` in one completion was
        # NEVER measured against `qwen3.5:4b` and is the plan's biggest
        # unquantified risk (assumption A1). What WAS measured is one native
        # tool call per turn (5/5 at 1, 4 and 9 tools) OR one
        # response_format-constrained payload (5/5 including an array-of-3).
        # Do NOT "simplify" this back to always setting `tools`/`tool_choice`
        # to `schemas or None` — a present-but-None `tools` key is what a
        # provider sees, and this branch is the assertable proof the repair
        # turn ships no tool surface at all.
        if schemas:
            call_params["tools"] = schemas
            call_params["tool_choice"] = "auto"
        elif response_format is not None:
            call_params["response_format"] = response_format

        # DECISION plan-2026-07-21T191807-bf7ffe24/D-003
        # Ollama prep runs HERE, before the call, in llm.py's own order (params
        # then messages): unprepared, a live `qwen3.5:4b` role dispatch issued
        # 0/3 tool calls; prepared, 3/3. Do NOT route this through
        # `LiteLLMInterface` instead — llm.py has no `tools=`/`tool_calls`
        # machinery at all, so that means building a tool-calling surface into
        # core. Do NOT drop the explicit gate: the helpers self-gate, but the
        # gate is the tested proof off-Ollama calls are untouched.
        # `structured` tracks whether THIS call is schema-enforced: False on a
        # tool turn mirrors llm.py:642 (free-text reply — keep the user's
        # temperature), True on the repair turn pins temperature=0. The
        # `response_format` handed to `prepare_ollama_messages` is what echoes
        # the schema into the prompt text — that echo is what moved an
        # array-of-3 shape from 0/5 to 5/5 in EXPLORE.
        sent_format = call_params.get("response_format")
        if is_ollama_model(self.config.model):
            apply_ollama_params(
                call_params, self.config.model, structured=sent_format is not None
            )
            call_params["messages"] = prepare_ollama_messages(
                messages, self.config.model, sent_format
            )

        # DECISION plan-2026-07-20T040150-876e7164/D-006: wrap the litellm
        # boundary so a provider outage reaches this agent's caller as an
        # AgentError, not as a raw openai.APIError that no `except AgentError`
        # can see. Deliberately the CONCRETE `AgentError` root — do NOT
        # introduce an `LLMCallError` (or similar) subtype for this: one call
        # site does not earn a new exception class, and the plan's Complexity
        # Budget is 0/2 new abstractions (D-001 records the refusal
        # explicitly). If a SECOND provider-call site ever needs the same type,
        # the subtype is earned then, not now. Only the network call belongs
        # inside the try — the tool_call parsing below must keep raising its
        # own programming errors unwrapped. See decisions.md D-006.
        try:
            response = litellm.completion(**call_params)
        except Exception as e:
            raise AgentError(f"Native function-calling LLM call failed: {e!s}") from e
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

        content = msg.content
        # DECISION plan-2026-07-21T191807-bf7ffe24/D-003
        # A reasoning-only reply (no `content`, answer in the reasoning field)
        # is recovered through the SHARED `_resolve_reasoning_trace`. Do NOT
        # hand-roll `getattr(msg, "thinking")`: installed litellm renames that
        # field to `reasoning_content`, so a `.thinking`-only read is dead code
        # for this project's default model. The `not tool_calls` guard is
        # DELIBERATE — with tool calls present, `content=None` is the normal
        # shape (measured 4/4 live), so recovering there is pure noise.
        if not content and not tool_calls:
            content = _resolve_reasoning_trace(msg)
        return {"content": content, "tool_calls": tool_calls}

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
        concluded = False

        try:
            for iteration in range(1, max_iters + 1):
                self._check_budgets(start_time, iteration, max_iters)
                result = self._complete(messages, schemas)
                tool_calls = result.get("tool_calls") or []
                content = result.get("content")

                if not tool_calls:
                    answer = content or ""
                    concluded = True
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
            structured = self._try_parse_structured_output(answer)
            # DECISION plan-2026-07-21T191807-bf7ffe24/D-002
            # Terminal-turn constrained decoding. EXACTLY ONE extra completion,
            # carrying `response_format=` and NO `tools=` — never both in one
            # call (see `_litellm_complete`; that pairing is unmeasured
            # assumption A1). Trigger is deliberately narrow: a schema is
            # configured AND the free-text answer failed to validate. So this
            # costs nothing whenever the model already complied, which is the
            # common case off Ollama — near-zero blast radius on capable
            # providers. Do NOT turn this into a retry loop: one attempt, and
            # a repair whose content does not parse is DISCARDED so a run that
            # already had a usable answer can never be regressed.
            repair_format = _output_response_format(self.config.output_schema)
            if repair_format is not None and structured is None:
                repaired = self._complete(
                    [*messages, {"role": "user", "content": _REPAIR_PROMPT}],
                    [],
                    response_format=repair_format,
                )
                repaired_answer = repaired.get("content") or ""
                repaired_structured = self._try_parse_structured_output(repaired_answer)
                if repaired_structured is not None:
                    answer = repaired_answer
                    structured = repaired_structured

            # DECISION plan-2026-07-21T191807-bf7ffe24/D-005
            # Success = the loop reached a final TOOL-CALL-FREE turn AND that
            # turn carried a non-empty answer. Do NOT restore
            # `bool(answer) or bool(trace_calls)`: it reported success=True on
            # three live runs that wrote zero bytes and answered nothing, so no
            # caller could tell a working role from a doomed one. Computed AFTER
            # the repair turn ON PURPOSE: a repair that fills a previously-empty
            # answer earns success. `concluded` is the term that keeps this
            # honest — a loop that EXHAUSTED max_iterations never signalled it
            # was done, so a payload extracted from it afterwards summarises
            # unfinished work and must NOT be relabelled a success. For "did it
            # do anything at all?", read `trace.tool_calls`, unchanged.
            success = concluded and bool(answer)
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
