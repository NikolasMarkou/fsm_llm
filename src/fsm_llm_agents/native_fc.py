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

``system_policy`` appends caller-owned STANDING instructions (rules, gates,
output shape — anything true for every turn of the dispatch) to the system
message, leaving the user turn to carry only that turn's task. It is read at
``run()`` time, so a caller that obtains the agent from a factory may set the
attribute afterwards. Default ``None`` reproduces the previous system message
byte for byte.
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

#: The user turn appended for the post-loop forced-write finalization (D-003).
#: Appended LAST (like `_REPAIR_PROMPT`) so it is the freshest instruction after
#: the read/tool round-trips: stop reading, commit findings via the forced call.
_FORCE_WRITE_PROMPT = (
    "You have gathered enough context. Stop reading now and record your "
    "findings by calling the required tool with your final content."
)

#: Provider messages that mean "this turn's TOOL CALL did not render", as
#: opposed to "the provider is unreachable".
#:
#: The first two are the measured shape (1 dispatch in 35 on
#: ``ollama_chat/qwen3.5:4b``): Ollama's tool-call template emitted invalid XML
#: and litellm surfaced it as an ``APIConnectionError``, so the class of the
#: exception carries no information and the message is all there is.  Kept
#: deliberately NARROW -- a broad marker such as "tool" would swallow real
#: outages, which is the thing this classification exists to keep separate.
_MALFORMED_TOOL_CALL_MARKERS: tuple[str, ...] = (
    "xml syntax error",
    "element <function>",
    "invalid tool call",
)


def _is_malformed_tool_call(exc: BaseException, tools_declared: bool) -> bool:
    """Whether *exc* is a garbled TOOL CALL rather than a provider failure.

    Interface contract (2 call sites: the ``_litellm_complete`` boundary, which
    labels the error, and :meth:`NativeFunctionCallingReactAgent.run`, which
    reads the label back off ``AgentError.details``):
        - ``tools_declared``: whether THIS completion carried a tool surface at
          all.  A completion with no ``tools=`` cannot garble a tool call, so it
          is never classified as one however its message reads.
        - Returns ``False`` for everything not positively identified.  Fail
          closed: an unrecognised failure stays a failure.
        - Never raises.
    """
    if not tools_declared:
        return False
    text = str(exc).lower()
    return any(marker in text for marker in _MALFORMED_TOOL_CALL_MARKERS)


def _degrades_turn(exc: BaseException) -> bool:
    """Whether *exc* may be absorbed as one failed turn instead of the run."""
    details = getattr(exc, "details", None)
    return bool(isinstance(details, dict) and details.get("malformed_tool_call"))


class NativeFunctionCallingReactAgent(BaseAgent):
    """ReAct loop driven by provider-native function calling.

    Args:
        tools: Tool registry (non-empty).
        config: AgentConfig (model, max_iterations, timeout, temperature, ...).
        complete_fn: Optional override ``(model, messages, tool_schemas) -> dict``
            for tests / custom backends. Defaults to a litellm completion.
        system_policy: Standing instructions appended to the system message.
            ``None`` (the default) leaves the system message exactly as it was.
        seed: Optional sampling seed forwarded to every ``litellm.completion``.
            ``None`` (the default) sends no ``seed`` key at all.
    """

    def __init__(
        self,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        complete_fn: CompleteFn | None = None,
        system_policy: str | None = None,
        *,
        seed: int | None = None,
        **api_kwargs: Any,
    ) -> None:
        if len(tools) == 0:
            raise AgentError("Cannot create agent with empty tool registry")
        super().__init__(config, **api_kwargs)
        self.tools = tools
        self._complete_fn = complete_fn
        self.seed = seed
        #: Public and mutable on purpose -- see :meth:`_system_message`.
        self.system_policy = system_policy

    def _system_message(self) -> str:
        """Return this run's system message: the base prompt plus any policy.

        Interface contract (1 call site, :meth:`run`; the attribute it reads is
        public because callers set it post-construction):
            - Reads ``self.system_policy`` at CALL time, not at construction, so
              a caller that receives the agent from a factory can still supply
              standing instructions.
            - Returns ``_SYSTEM_PROMPT`` unchanged when the policy is unset or
              blank -- the no-policy path is byte-identical to before.
            - Never raises.
        """
        # DECISION plan-2026-07-21T191807-bf7ffe24/D-021
        # Standing instructions belong in the SYSTEM message, and this seam
        # exists because that placement was MEASURED to be the difference
        # between a 4B model doing the work and only talking about it. Live
        # `ollama_chat/qwen3.5:4b`, harness EXECUTE role, n=5 per arm, bytes
        # stat'd on disk: the whole role prompt in the USER turn wrote 0/5;
        # the SAME text, unchanged and complete, with everything standing moved
        # into the system message wrote 4/5. Content ablations in between
        # (fixing the writes line: 0/5; deleting the rules block: 2/5) did not
        # reproduce it, so this is placement, not wording.
        # Do NOT "simplify" this into a `system_prompt` REPLACEMENT parameter:
        # the measured arm kept `_SYSTEM_PROMPT` and appended to it, and the
        # base prompt is what tells the model it may call tools at all.
        # Do NOT make it constructor-only either -- `roles.py` obtains this
        # agent through an injectable builder that cannot see the dispatch, so
        # a construction-time-only parameter would be unreachable exactly where
        # it was measured to matter. See decisions.md D-021.
        if not self.system_policy:
            return _SYSTEM_PROMPT
        return f"{_SYSTEM_PROMPT}\n\n{self.system_policy}"

    # BaseAgent abstract hook — this agent does not use the FSM pipeline.
    def _register_handlers(self, api: API) -> None:  # pragma: no cover - unused
        return None

    # --- LLM completion -------------------------------------------------
    def _complete(
        self,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        tool_choice: Any | None = None,
    ) -> dict[str, Any]:
        if self._complete_fn is not None:
            return self._complete_fn(self.config.model, messages, schemas)
        return self._litellm_complete(messages, schemas, response_format, tool_choice)

    def _litellm_complete(
        self,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        tool_choice: Any | None = None,
    ) -> dict[str, Any]:
        import litellm

        call_params: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        # DECISION plan-2026-07-22T114536-879d04a0/D-008
        # `seed` lands HERE and only here: this agent calls `litellm.completion`
        # DIRECTLY, bypassing both `apply_ollama_params` and core's
        # `LiteLLMInterface`, so plumbing it anywhere else reaches nothing.
        # Live-probed on `ollama_chat/qwen3.5:4b` (digest 2a654d98e6fb) before
        # this line existed: at temperature=0.7 the same seed twice was
        # byte-identical and a different seed diverged -- the server HONORS it.
        # Do NOT emit `seed: None` when unset (the key must be ABSENT, keeping
        # every existing call byte-identical), and do NOT default it to a fixed
        # value -- determinism is opt-in for benches. See decisions.md D-008.
        if self.seed is not None:
            call_params["seed"] = self.seed
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
            # Default `"auto"` (byte-identical to every existing caller); the
            # D-003 forced-write turn overrides it to a named-function choice.
            # Never merges `tools=`/`response_format=` (D-002 branch untouched).
            call_params["tool_choice"] = (
                tool_choice if tool_choice is not None else "auto"
            )
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

        # DECISION plan-2026-07-20T040150-876e7164/D-006 [STALE]: wrap the litellm
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
        # DECISION plan-2026-07-21T191807-bf7ffe24/D-016
        # The `try` still wraps ONLY the network call (D-006's rule is
        # untouched); what is added is a LABEL. Measured 1 dispatch in 35 on
        # `ollama_chat/qwen3.5:4b`: Ollama's tool-call template emitted
        # `element <function> closed by </parameter>`, litellm raised
        # `APIConnectionError`, and `run()`'s `except AgentError: raise` ended a
        # dispatch that had ALREADY written real bytes. Classify HERE, not in
        # `run()`: this is the only frame that knows whether the failed call
        # even carried a tool surface, and that is half the discriminator.
        # Do NOT widen this into "swallow every AgentError from the loop" -- a
        # provider outage is not a model behaviour and must still end the run.
        # See decisions.md D-016.
        try:
            response = litellm.completion(**call_params)
        except Exception as e:
            raise AgentError(
                f"Native function-calling LLM call failed: {e!s}",
                details={
                    "malformed_tool_call": _is_malformed_tool_call(
                        e, bool(call_params.get("tools"))
                    )
                },
            ) from e
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
            {"role": "system", "content": self._system_message()},
            {"role": "user", "content": task},
        ]
        trace_calls: list[ToolCall] = []
        max_iters = self.config.max_iterations
        answer = ""
        concluded = False

        try:
            for iteration in range(1, max_iters + 1):
                self._check_budgets(start_time, iteration, max_iters)
                # DECISION plan-2026-07-21T191807-bf7ffe24/D-016
                # A garbled tool-call turn ends the LOOP, not the dispatch. The
                # `break` is deliberate and is not a silent retry: everything
                # already established -- `trace_calls`, the bytes those calls
                # wrote, any `answer` -- survives into the result, and the
                # post-loop D-002 repair turn then runs with NO tool surface at
                # all, which is precisely the shape that cannot reproduce the
                # failure. `concluded` stays False, so D-005's honest `success`
                # still reports that this run did not finish on its own.
                # Do NOT turn this into `continue`: the next turn would declare
                # the same tools against the same history and, on the one
                # measured occurrence, garble again -- burning the budget to
                # reach the same place with less context.
                # See decisions.md D-016.
                try:
                    result = self._complete(messages, schemas)
                except AgentError as exc:
                    if not _degrades_turn(exc):
                        raise
                    logger.warning(
                        f"Native function-calling tool turn {iteration} was "
                        f"malformed by the provider ({exc}); ending the loop "
                        "with the trace and answer gathered so far."
                    )
                    break
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

            # DECISION plan-2026-07-23T073649-bb230f18/D-003
            # Forced-write finalization: when `force_final_tool` is set and the
            # read loop never called it, force the MODEL (not the driver) to emit
            # ONE real tool call via `self.tools.execute`/`trace_calls`, so
            # `_verified_writes` sees a genuine model write (disk-is-truth intact).
            # Do NOT weaken: (a) `tools=`+`tool_choice` ONLY, NEVER
            # `response_format=` here (the D-002 mutual-exclusion branch, `:229`,
            # is untouched); (b) fires AT MOST ONCE, strictly AFTER the loop (one
            # guarded `if`, no loop -- model reads first, no turn-1 write); (c) a
            # malformed forced turn is ABSORBED like the D-002 repair
            # (`_degrades_turn`) -- no crash, no fabricated write, `answer`
            # untouched so `success` stays honest; (d) placed BEFORE `trace =
            # AgentTrace(...)` so the trace reflects it, and default `None` keeps
            # every other caller byte-identical. Do NOT fold into the D-002
            # repair (that ships structured JSON with NO tools; this ships a tool
            # call with NO `response_format=`). See decisions.md D-003.
            if self.config.force_final_tool and not any(
                tc.tool_name == self.config.force_final_tool for tc in trace_calls
            ):
                forced_schema = [
                    s
                    for s in schemas
                    if s.get("function", {}).get("name")
                    == self.config.force_final_tool
                ]
                if forced_schema:
                    try:
                        forced = self._complete(
                            [
                                *messages,
                                {"role": "user", "content": _FORCE_WRITE_PROMPT},
                            ],
                            forced_schema,
                            tool_choice={
                                "type": "function",
                                "function": {"name": self.config.force_final_tool},
                            },
                        )
                    except AgentError as exc:
                        # Symmetric with the loop and the D-002 repair (D-016): a
                        # degradable forced turn must not delete a run that
                        # already has an answer and a trace.
                        if not _degrades_turn(exc):
                            raise
                        logger.warning(
                            f"Forced-write turn was malformed ({exc}); keeping "
                            "the trace and answer gathered so far."
                        )
                        forced = {"tool_calls": []}
                    for tc in forced.get("tool_calls") or []:
                        name = tc.get("name", "")
                        args = tc.get("arguments") or {}
                        if not isinstance(args, dict):
                            args = {}
                        call = ToolCall(tool_name=name, parameters=args)
                        self.tools.execute(call)
                        trace_calls.append(call)

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
                try:
                    repaired = self._complete(
                        [*messages, {"role": "user", "content": _REPAIR_PROMPT}],
                        [],
                        response_format=repair_format,
                    )
                except AgentError as exc:
                    # Symmetric with the loop above (D-016): a degradable
                    # failure on the OPTIONAL repair must not delete a run that
                    # already has an answer and a trace.
                    if not _degrades_turn(exc):
                        raise
                    logger.warning(f"Repair turn was malformed ({exc}); keeping it.")
                    repaired = {}
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
