"""The handlers of a native function-calling run: ``NativeFCHandlers`` and its helpers.

Moved from ``native_fc`` by plan a555dcfe step 12 (D-021), bodies and anchors unchanged. The library
factory ``agents.native_fc`` (``library.py``) builds the eight handlers from the run's
``HandlerContext``; ``NativeFCHandlers.register(api)`` registers the same eight on an ``API`` (the
tests and any caller that builds the FSM by hand).

Interface contract: see ``NativeFCHandlers``. ``handlers()`` returns the eight built handlers in
registered order (``seed``, ``model_reply``, ``run_tools``, ``force_entry``, ``forced_reply``,
``repair_entry``, ``repair_reply``, ``conclude``), each ``critical``; ``register(api)`` adds them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fsm_llm.agents.constants import (
    ContextKeys,
    HandlerPriorities,
    NativeFCContextKeys,
    NativeFCHandlerNames,
    NativeFCStates,
    NativeLoopEnd,
    StopReason,
)
from fsm_llm.agents.definitions import AgentStep
from fsm_llm.handlers import HandlerTiming, create_handler
from fsm_llm.llm import tool_exchange
from fsm_llm.logging import logger
from fsm_llm.tools import ToolCall, ToolRegistry, ToolResult, redact_secret_entries

from .approval import call_label
from .recording import next_step_number

if TYPE_CHECKING:
    from fsm_llm import API
    from fsm_llm.handlers import BaseHandler

__all__ = ["NativeFCHandlers"]

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

    The tool ran with ``call``; the returned trace (read by the monitor)
    gets this copy.
    """
    # DECISION plan-2026-10-01T093600-944e2692/D-027 (carries
    # 06a5ec0a/D-016): the trace, `agent_trace` and logs hold this redacted
    # copy; the tool gets the real arguments. Do NOT trace `call` itself and
    # do NOT add a second matcher: `redact_secret_entries` is the one rule.
    return call.model_copy(
        update={"parameters": redact_secret_entries(call.parameters)}
    )


# --- FSM handlers (``build_native_fc_fsm``) ---------------------------------

_K = NativeFCContextKeys


def _turn_calls(reply: Any) -> list[dict[str, Any]] | None:
    """The calls of a completion result, or ``None`` when any is unrunnable.

    A runnable call is a dict with a non-empty ``name`` and dict
    ``arguments`` (core's normaliser guarantees it for a ``calls`` reply; a
    result planted or damaged in context is refused here, whole turn).
    """
    # DECISION plan-2026-10-01T093600-944e2692/D-033: a call with an empty
    # tool name is malformed, like one with non-object arguments: the whole
    # turn runs nothing and the loop ends `malformed` (core's normaliser
    # already makes a provider turn with a nameless call `kind="malformed"`,
    # D-027). Do NOT run it as an unknown tool and continue the loop (the
    # e1f63a9 behaviour): a nameless call is a garbled turn, not a choice.
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


def _trace_step(trace: list[Any], call: ToolCall, result: ToolResult) -> dict[str, Any]:
    """The ``agent_trace`` entry of one executed call, parameters redacted.

    Same shape as the ReAct executor's entry (``_build_trace`` reads
    ``action`` and ``tool_input``; ``tool_status`` is ``result.status``): the
    call ran with ``call``, the trace holds the ``_shown_call`` copy (D-016 of
    plan 06a5ec0a).
    """
    step = AgentStep(
        iteration=next_step_number(trace),
        action=call_label(call.tool_name, call.parameters),
        observation=result.observation,
    ).model_dump(mode="json")
    step["tool_input"] = _shown_call(call).parameters
    step[ContextKeys.TOOL_STATUS] = result.status
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
          Retrying registries keep working).
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

    def handlers(self) -> list[BaseHandler]:
        """Every native handler, built, in registered order."""
        states = NativeFCStates
        names = NativeFCHandlerNames
        return [
            create_handler(names.SEED)
            .at(HandlerTiming.START_CONVERSATION)
            .critical()
            .do(self.seed),
            create_handler(names.MODEL_REPLY)
            .on_state(states.CALL_MODEL)
            .on_context_update(_K.MODEL_REPLY)
            .critical()
            .do(self.model_reply),
            create_handler(names.RUN_TOOLS)
            .on_state_entry(states.RUN_TOOLS)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .critical()
            .do(self.run_tools),
            create_handler(names.FORCE_ENTRY)
            .on_state_entry(states.FORCE_FINAL)
            .critical()
            .do(self.force_entry),
            create_handler(names.FORCED_REPLY)
            .on_state(states.FORCE_FINAL)
            .on_context_update(_K.FORCED_REPLY)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .critical()
            .do(self.forced_reply),
            create_handler(names.REPAIR_ENTRY)
            .on_state_entry(states.REPAIR)
            .critical()
            .do(self.repair_entry),
            create_handler(names.REPAIR_REPLY)
            .on_state(states.REPAIR)
            .on_context_update(_K.REPAIR_REPLY)
            .critical()
            .do(self.repair_reply),
            create_handler(names.CONCLUDE)
            .on_state_entry(states.CONCLUDE)
            .critical()
            .do(self.conclude),
        ]

    def register(self, api: API) -> None:
        """Register every native handler on *api*."""
        for handler in self.handlers():
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

        The text is ``ToolResult.observation`` (the summary, prefixed
        ``[TOOL FAILED]`` for a failed call and ``[TOOL OUTCOME UNKNOWN]`` for
        a timed-out one). Appends the redacted trace entry to ``trace``.
        """
        call = ToolCall(tool_name=entry["name"], parameters=entry["arguments"])
        result = self.tools.execute(call)
        trace.append(_trace_step(trace, call, result))
        return result.observation

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
        # DECISION plan-2026-10-01T093600-944e2692/D-027 (carries
        # bf7ffe24/D-016): a malformed tool turn ends the LOOP, not the run:
        # the trace and answer gathered so far survive and the post-loop
        # turns still run. Do NOT route it back to `call_model` (the same
        # tools against the same history garbled again on the one measured
        # occurrence) and do NOT raise. A provider outage is not a malformed
        # turn: core raises it and the run fails. See decisions.md D-027.
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
            # The refused reply's raw arguments never reach final_context.
            return {
                **self._end_loop(context, NativeLoopEnd.MALFORMED, ""),
                _K.MODEL_REPLY: None,
            }
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
        The results are traced, not fed back: no model turn follows. The
        reply is cleared (its raw arguments never reach ``final_context``;
        the trace holds the redacted copy).
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-027 (carries
        # bb230f18/D-003): the forced write is the MODEL's call (named
        # `tool_choice`, tools only, never `response_format`), made at most
        # once after the loop, run through `tools.execute` like every
        # other tool call, never around the registry. Do NOT synthesise the call
        # in the driver, do NOT loop, and do NOT let a malformed forced turn
        # touch the answer or success. See decisions.md D-027.
        reply = context.get(_K.FORCED_REPLY)
        kind = reply.get("kind") if isinstance(reply, dict) else None
        calls = _turn_calls(reply) if kind == "calls" else None
        done: dict[str, Any] = {_K.FORCE_PENDING: False, _K.FORCED_REPLY: None}
        if calls is None:
            if kind != "final":
                logger.warning(
                    "Forced-write turn was malformed; keeping the trace and "
                    "answer gathered so far."
                )
            return done
        trace = list(context.get(ContextKeys.AGENT_TRACE) or [])
        for entry in calls:
            self._execute(entry, trace)
        return {**done, ContextKeys.AGENT_TRACE: trace}

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
        """``conclude`` entry: count the model turns, record why a run stopped.

        Success is a loop that reached a final turn with a non-blank answer
        (after any repair); otherwise ``forced_stop_reason`` is
        ``max_iterations`` (exhausted loop) or ``no_result``, which
        ``BaseAgent._run_outcome`` reads. ``iteration_count`` is the number
        of loop model turns (each tool turn, plus the turn that ended the
        loop unless it was exhausted).
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-027 (carries
        # bf7ffe24/D-005): success = the loop reached a tool-call-free final
        # turn AND the answer (after any repair) is not blank. Do NOT count
        # "a tool ran" as success (three live runs that wrote nothing read as
        # success) and do NOT let a repair relabel an exhausted or malformed
        # loop a success. The verdict reaches `success` only through the one
        # seam (`_run_outcome`, D-028). See decisions.md D-027.
        how = context.get(_K.LOOP_END)
        tool_turns = int(context.get(_K.TOOL_TURNS) or 0)
        delta: dict[str, Any] = {
            ContextKeys.ITERATION_COUNT: tool_turns
            + (0 if how == NativeLoopEnd.EXHAUSTED else 1)
        }
        answer = context.get(_K.ANSWER)
        if how == NativeLoopEnd.EXHAUSTED:
            delta[ContextKeys.FORCED_STOP_REASON] = StopReason.MAX_ITERATIONS.value
        elif how != NativeLoopEnd.FINAL or not (
            isinstance(answer, str) and answer.strip()
        ):
            delta[ContextKeys.FORCED_STOP_REASON] = StopReason.NO_RESULT.value
        return delta
