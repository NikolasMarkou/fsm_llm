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

    from fsm_llm.agents import AgentConfig, ToolRegistry
    from fsm_llm.agents.native_fc import NativeFunctionCallingReactAgent

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

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming
from fsm_llm.llm import (
    decode_tool_arguments,
    is_malformed_tool_call_error,
    tool_exchange,
)
from fsm_llm.logging import logger
from fsm_llm.ollama import (
    apply_ollama_params,
    is_ollama_model,
    prepare_ollama_messages,
)
from fsm_llm.utilities import _resolve_reasoning_trace

from .base import BaseAgent, _output_response_format
from .constants import (
    ContextKeys,
    HandlerPriorities,
    NativeFCContextKeys,
    NativeFCHandlerNames,
    NativeFCStates,
    NativeLoopEnd,
    StopReason,
)
from .definitions import AgentConfig, AgentResult, AgentStep, AgentTrace, ToolCall
from .exceptions import AgentError
from .handlers import call_label, next_step_number
from .tools import ToolRegistry, redact_secret_entries

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


def _shown_call(call: ToolCall) -> ToolCall:
    """The trace copy of an executed call, secret-looking arguments redacted.

    plan-2026-09-29T103145-06a5ec0a/D-016: the tool ran with ``call``; the
    returned trace (read by the monitor and the harness) gets this copy.
    """
    return call.model_copy(
        update={"parameters": redact_secret_entries(call.parameters)}
    )


def _degrades_turn(exc: BaseException) -> bool:
    """Whether *exc* may be absorbed as one failed turn instead of the run."""
    details = getattr(exc, "details", None)
    return bool(isinstance(details, dict) and details.get("malformed_tool_call"))


# --- FSM handlers (``build_native_fc_fsm``) ---------------------------------

_K = NativeFCContextKeys


def _turn_calls(reply: Any) -> list[dict[str, Any]] | None:
    """The calls of a completion result, or ``None`` when any is unrunnable.

    A runnable call is a dict with a non-empty ``name`` and dict
    ``arguments`` (core's normaliser guarantees it for a ``calls`` reply; a
    result planted or damaged in context is refused here, whole turn).
    """
    calls = reply.get("calls") if isinstance(reply, dict) else None
    if not isinstance(calls, list) or not calls:
        return None
    for call in calls:
        if not (
            isinstance(call, dict)
            and isinstance(call.get("name"), str)
            and call["name"]
            and isinstance(call.get("arguments"), dict)
        ):
            return None
    return calls


def _trace_step(trace: list[Any], call: ToolCall, observation: str) -> dict[str, Any]:
    """The ``agent_trace`` entry of one executed call, parameters redacted.

    Same shape as the ReAct executor's entry (``_build_trace`` reads
    ``action`` and ``tool_input``): the call ran with ``call``, the trace
    holds the ``_shown_call`` copy (D-016 of plan 06a5ec0a).
    """
    step = AgentStep(
        iteration=next_step_number(trace),
        action=call_label(call.tool_name, call.parameters),
        observation=observation,
    ).model_dump(mode="json")
    step["tool_input"] = _shown_call(call).parameters
    return step


def _traced_tool_names(trace: Any) -> set[str]:
    """Names of the tools executed so far, read off ``agent_trace``."""
    if not isinstance(trace, list):
        return set()
    return {
        str(step.get("action", "")).split("(")[0]
        for step in trace
        if isinstance(step, dict)
    }


class NativeFCHandlers:
    """The handlers of one native function-calling run (``build_native_fc_fsm``).

    Interface contract (one caller: ``NativeFunctionCallingReactAgent``, which
    builds one per run and never stores it on the agent):
        - ``tools``: every call runs through ``tools.execute`` (Caching and
          Retrying registries keep working; the harness spies on it).
        - ``max_tool_turns``: the loop ends ``exhausted`` after this many tool
          turns without a final turn (``AgentConfig.max_iterations``), >= 1.
        - ``force_final_tool``: the tool the run must end by calling, or
          ``None``. A forced turn is flagged only when the tool has not run;
          it happens only when the FSM has a ``force_final`` state.
        - ``parse_structured``: answer text -> parsed output, ``None`` when it
          does not parse; ``None`` itself means no output schema (no repair
          is ever flagged). It happens only when the FSM has a ``repair`` state.
        - ``register(api)`` adds every handler (critical: a failure fails the
          turn, never a skipped tool exchange). Each handler takes the context
          and returns a delta; none calls a model.

    Context it owns: ``NativeFCContextKeys`` and ``agent_trace`` (appended),
    plus ``forced_stop_reason`` at ``conclude`` (``max_iterations`` for an
    exhausted loop, ``no_result`` for a malformed turn or an empty answer).
    """

    def __init__(
        self,
        tools: ToolRegistry,
        *,
        max_tool_turns: int,
        force_final_tool: str | None = None,
        parse_structured: Callable[[str], Any] | None = None,
    ) -> None:
        if max_tool_turns < 1:
            raise ValueError("max_tool_turns must be at least 1")
        self.tools = tools
        self.max_tool_turns = max_tool_turns
        self.force_final_tool = force_final_tool
        self.parse_structured = parse_structured

    def register(self, api: API) -> None:
        """Register every native handler on *api*."""
        states = NativeFCStates
        names = NativeFCHandlerNames
        for handler in (
            api.create_handler(names.SEED)
            .at(HandlerTiming.START_CONVERSATION)
            .critical()
            .do(self.seed),
            api.create_handler(names.MODEL_REPLY)
            .on_state(states.CALL_MODEL)
            .on_context_update(_K.MODEL_REPLY)
            .critical()
            .do(self.model_reply),
            api.create_handler(names.RUN_TOOLS)
            .on_state_entry(states.RUN_TOOLS)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .critical()
            .do(self.run_tools),
            api.create_handler(names.FORCE_ENTRY)
            .on_state_entry(states.FORCE_FINAL)
            .critical()
            .do(self.force_entry),
            api.create_handler(names.FORCED_REPLY)
            .on_state(states.FORCE_FINAL)
            .on_context_update(_K.FORCED_REPLY)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .critical()
            .do(self.forced_reply),
            api.create_handler(names.REPAIR_ENTRY)
            .on_state_entry(states.REPAIR)
            .critical()
            .do(self.repair_entry),
            api.create_handler(names.REPAIR_REPLY)
            .on_state(states.REPAIR)
            .on_context_update(_K.REPAIR_REPLY)
            .critical()
            .do(self.repair_reply),
            api.create_handler(names.CONCLUDE)
            .on_state_entry(states.CONCLUDE)
            .critical()
            .do(self.conclude),
        ):
            api.register_handler(handler)

    @staticmethod
    def seed(context: dict[str, Any]) -> dict[str, Any]:
        """START_CONVERSATION: the transcript is the task; every owned key reset.

        Caller context can never pre-set a result key (that would skip a
        model turn), a routing flag or a transcript.
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-026: every key this run
        # owns is written here, at the start, from the task alone. Do NOT keep
        # a caller-supplied transcript, result key or routing flag: a set
        # `model_reply` skips the first model turn (skip-if-set) and a planted
        # transcript speaks for the model. Do NOT seed the task through the
        # completion state's instructions or a neutral turn either: the golden
        # requests send it as the one user turn. See decisions.md D-026.
        task = context.get(ContextKeys.TASK)
        return {
            _K.TRANSCRIPT: [
                {"role": "user", "content": "" if task is None else str(task)}
            ],
            _K.FORCE_MESSAGES: None,
            _K.REPAIR_MESSAGES: None,
            _K.MODEL_REPLY: None,
            _K.FORCED_REPLY: None,
            _K.REPAIR_REPLY: None,
            _K.TOOL_TURNS: 0,
            _K.ANSWER: "",
            _K.LOOP_END: None,
            _K.FORCE_PENDING: None,
            _K.REPAIR_PENDING: None,
        }

    def _execute(self, entry: dict[str, Any], trace: list[Any]) -> str:
        """Run one call through ``tools.execute``; trace it; return its text.

        The text is the result summary, prefixed ``[TOOL FAILED]`` for a
        failed call. Appends the redacted trace entry to ``trace``.
        """
        call = ToolCall(tool_name=entry["name"], parameters=entry["arguments"])
        result = self.tools.execute(call)
        observation = result.summary
        if not result.success:
            observation = f"[TOOL FAILED] {observation}"
        trace.append(_trace_step(trace, call, observation))
        return observation

    def _end_loop(
        self, context: dict[str, Any], how: str, answer: str
    ) -> dict[str, Any]:
        """Delta that ends the model loop and flags the post-loop turns due."""
        force = bool(self.force_final_tool) and (
            self.force_final_tool
            not in _traced_tool_names(context.get(ContextKeys.AGENT_TRACE))
        )
        repair = (
            self.parse_structured is not None and self.parse_structured(answer) is None
        )
        return {
            _K.LOOP_END: how,
            _K.ANSWER: answer,
            _K.FORCE_PENDING: force,
            _K.REPAIR_PENDING: repair,
        }

    def model_reply(self, context: dict[str, Any]) -> dict[str, Any]:
        """``call_model`` reply: a final or malformed turn ends the loop.

        A ``calls`` reply needs nothing here (``run_tools`` runs it).
        """
        reply = context.get(_K.MODEL_REPLY)
        kind = reply.get("kind") if isinstance(reply, dict) else None
        if kind == "calls" or reply is None:
            return {}
        if kind == "final":
            return self._end_loop(context, NativeLoopEnd.FINAL, reply.get("text") or "")
        logger.warning(
            "Native function-calling tool turn was malformed; ending the loop "
            "with the trace and answer gathered so far."
        )
        return self._end_loop(context, NativeLoopEnd.MALFORMED, "")

    def run_tools(self, context: dict[str, Any]) -> dict[str, Any]:
        """``run_tools`` entry: run the turn's calls in order and pair them.

        The whole turn is checked first: if any call is unrunnable, none
        runs and the loop ends malformed. Otherwise each call runs through
        ``tools.execute`` (a failure is shown to the model as ``[TOOL FAILED]
        <summary>``), the redacted trace grows, the paired exchange is
        appended to the transcript, the result key is cleared (so the next
        ``call_model`` asks again) and the tool turn is counted.
        """
        reply = context.get(_K.MODEL_REPLY)
        calls = _turn_calls(reply)
        if calls is None:
            logger.warning(
                "Native function-calling tool turn has a call that cannot run; "
                "running none of its calls and ending the loop."
            )
            return self._end_loop(context, NativeLoopEnd.MALFORMED, "")
        trace = list(context.get(ContextKeys.AGENT_TRACE) or [])
        observations = [self._execute(entry, trace) for entry in calls]
        transcript = list(context.get(_K.TRANSCRIPT) or [])
        text = reply.get("text") if isinstance(reply, dict) else None
        transcript.extend(tool_exchange(text, calls, observations))
        turns = int(context.get(_K.TOOL_TURNS) or 0) + 1
        delta: dict[str, Any] = {
            _K.TRANSCRIPT: transcript,
            ContextKeys.AGENT_TRACE: trace,
            _K.TOOL_TURNS: turns,
            _K.MODEL_REPLY: None,
        }
        if turns >= self.max_tool_turns:
            logger.warning(
                "NativeFunctionCallingReactAgent hit max_iterations without a "
                "final answer."
            )
            delta.update(
                self._end_loop(
                    {**context, ContextKeys.AGENT_TRACE: trace},
                    NativeLoopEnd.EXHAUSTED,
                    "",
                )
            )
        return delta

    @staticmethod
    def force_entry(context: dict[str, Any]) -> dict[str, Any]:
        """``force_final`` entry: its request is the transcript plus the nudge."""
        transcript = list(context.get(_K.TRANSCRIPT) or [])
        return {
            _K.FORCE_MESSAGES: [
                *transcript,
                {"role": "user", "content": _FORCE_WRITE_PROMPT},
            ],
            _K.FORCED_REPLY: None,
        }

    def forced_reply(self, context: dict[str, Any]) -> dict[str, Any]:
        """``force_final`` reply: run its calls through ``tools.execute``.

        A malformed forced turn runs nothing and keeps the trace and answer.
        The results are traced, not fed back: no model turn follows.
        """
        reply = context.get(_K.FORCED_REPLY)
        kind = reply.get("kind") if isinstance(reply, dict) else None
        calls = _turn_calls(reply) if kind == "calls" else None
        if calls is None:
            if kind != "final":
                logger.warning(
                    "Forced-write turn was malformed; keeping the trace and "
                    "answer gathered so far."
                )
            return {_K.FORCE_PENDING: False}
        trace = list(context.get(ContextKeys.AGENT_TRACE) or [])
        for entry in calls:
            self._execute(entry, trace)
        return {ContextKeys.AGENT_TRACE: trace, _K.FORCE_PENDING: False}

    @staticmethod
    def repair_entry(context: dict[str, Any]) -> dict[str, Any]:
        """``repair`` entry: its request is the transcript plus the nudge."""
        transcript = list(context.get(_K.TRANSCRIPT) or [])
        return {
            _K.REPAIR_MESSAGES: [
                *transcript,
                {"role": "user", "content": _REPAIR_PROMPT},
            ],
            _K.REPAIR_REPLY: None,
        }

    def repair_reply(self, context: dict[str, Any]) -> dict[str, Any]:
        """``repair`` reply: it becomes the answer only if it parses.

        A repair that does not parse is discarded, so a run that had a
        usable answer is never regressed.
        """
        reply = context.get(_K.REPAIR_REPLY)
        text = (reply.get("text") if isinstance(reply, dict) else None) or ""
        if (
            self.parse_structured is not None
            and self.parse_structured(text) is not None
        ):
            return {_K.ANSWER: text, _K.REPAIR_PENDING: False}
        logger.info("Repair turn did not parse against the output schema; discarded")
        return {_K.REPAIR_PENDING: False}

    @staticmethod
    def conclude(context: dict[str, Any]) -> dict[str, Any]:
        """``conclude`` entry: record why a run that did not finish stopped.

        Success is a loop that reached a final turn with a non-empty answer
        (after any repair); otherwise ``forced_stop_reason`` is
        ``max_iterations`` (exhausted loop) or ``no_result``.
        """
        how = context.get(_K.LOOP_END)
        if how == NativeLoopEnd.EXHAUSTED:
            return {ContextKeys.FORCED_STOP_REASON: StopReason.MAX_ITERATIONS}
        if how != NativeLoopEnd.FINAL or not context.get(_K.ANSWER):
            return {ContextKeys.FORCED_STOP_REASON: StopReason.NO_RESULT}
        return {}


class NativeFunctionCallingReactAgent(BaseAgent):
    """ReAct loop driven by provider-native function calling.

    Args:
        tools: Tool registry (non-empty).
        config: AgentConfig (model, max_iterations, timeout, temperature, ...).
        complete_fn: Optional override ``(model, messages, tool_schemas) -> dict``
            for tests / custom backends. Defaults to a litellm completion.
        system_policy: Standing instructions appended to the system message.
            ``None`` (the default) falls back to ``config.instructions``; when
            both are unset the system message is exactly as it was.
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
        self._refuse_flagged_tools()
        self._complete_fn = complete_fn
        self.seed = seed
        #: Public and mutable on purpose -- see :meth:`_system_message`.
        self.system_policy = (
            system_policy if system_policy is not None else self.config.instructions
        )

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
                    "malformed_tool_call": is_malformed_tool_call_error(
                        e, tools_sent=bool(call_params.get("tools"))
                    )
                },
            ) from e
        msg = response.choices[0].message
        tool_calls: list[dict[str, Any]] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            # A call whose arguments are not a JSON object keeps its RAW
            # arguments so `run()` can refuse it (`decode_tool_arguments` returns
            # None there; D-025 of plan 06a5ec0a). Do NOT map them to `{}`.
            raw_args = tc.function.arguments
            args = decode_tool_arguments(raw_args)
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": raw_args if args is None else args,
                }
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
        self._refuse_flagged_tools()
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
        # Only a loop that ran out of turns is a forced stop; a malformed turn
        # ends it early with no result.
        exhausted = False

        try:
            for iteration in range(1, max_iters + 1):
                self._check_budgets(start_time)
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

                # DECISION plan-2026-09-29T103145-06a5ec0a/D-025: a tool call
                # whose arguments are not a JSON object is a malformed turn
                # under D-016 above: the loop ENDS (`break`, trace kept) and
                # NO call of that turn runs. Do NOT coerce the arguments to
                # `{}` and run the tool anyway (the old behaviour: a tool ran
                # with parameters the model never sent), and do NOT
                # `continue` (D-016). The whole turn is checked before any
                # call runs so the history never holds an assistant tool-call
                # message without its tool results.
                parsed = [
                    decode_tool_arguments(tc.get("arguments")) for tc in tool_calls
                ]
                if any(args is None for args in parsed):
                    logger.warning(
                        f"Native function-calling tool turn {iteration} sent "
                        "tool-call arguments that are not a JSON object; "
                        "ending the loop without running them, with the "
                        "trace and answer gathered so far."
                    )
                    break

                tool_calls = [
                    {**tc, "arguments": args}
                    for tc, args in zip(tool_calls, parsed, strict=True)
                ]
                messages.append(self._assistant_message(content, tool_calls))
                for tc in tool_calls:
                    call = ToolCall(
                        tool_name=tc.get("name", ""), parameters=tc["arguments"]
                    )
                    exec_result = self.tools.execute(call)
                    observation = exec_result.summary
                    if not exec_result.success:
                        observation = f"[TOOL FAILED] {observation}"
                    trace_calls.append(_shown_call(call))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": observation,
                        }
                    )
            else:
                # Loop exhausted without a final (tool-call-free) answer.
                exhausted = True
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
                    if s.get("function", {}).get("name") == self.config.force_final_tool
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
                        args = decode_tool_arguments(tc.get("arguments"))
                        if args is None:
                            # Malformed like the loop's turn (D-025): never
                            # run a write with arguments the model did not send.
                            logger.warning(
                                f"Forced-write call to {name!r} sent arguments "
                                "that are not a JSON object; not running it."
                            )
                            continue
                        call = ToolCall(tool_name=name, parameters=args)
                        self.tools.execute(call)
                        trace_calls.append(_shown_call(call))

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
            if success:
                stop_reason = StopReason.ANSWERED
            elif exhausted:
                stop_reason = StopReason.MAX_ITERATIONS
            else:
                stop_reason = StopReason.NO_RESULT
            return AgentResult(
                answer=answer,
                success=success,
                stop_reason=stop_reason,
                trace=trace,
                final_context={"task": task},
                structured_output=structured,
            )
        except AgentError:
            raise
        except Exception as e:
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
