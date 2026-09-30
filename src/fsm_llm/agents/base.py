"""
BaseAgent — Abstract base class for all fsm_llm agents.

Extracts the common conversation loop, budget enforcement, answer extraction,
trace building, and context filtering from the 12 agent implementations.
"""

from __future__ import annotations

import inspect
import json
import time
from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
    Collection,
    Iterator,
    Mapping,
    Sequence,
    Set,
)
from functools import partial
from typing import Any, ClassVar, cast

from pydantic import BaseModel, ValidationError

from fsm_llm import API
from fsm_llm.constants import CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT, has_internal_prefix
from fsm_llm.context import ContextCompactor
from fsm_llm.definitions import RunBudgetExceededError
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger

from .constants import (
    CONFIG_OWNED_KWARGS,
    MISPLACED_AGENT_KWARGS,
    RESULT_DROPPED_CONTEXT_KEYS,
    RUN_OUTPUT_KEYS,
    AgentStates,
    ContextKeys,
    Defaults,
    ErrorMessages,
    HandlerNames,
    HandlerPriorities,
    LogMessages,
    StopReason,
)
from .definitions import AgentConfig, AgentResult, AgentTrace, ToolCall
from .exceptions import AgentError, AgentTimeoutError, BudgetExhaustedError
from .hitl import ApprovalPolicy, HumanInTheLoop, make_hitl_checker


def with_instructions(
    fsm_def: dict[str, Any], instructions: str | None
) -> dict[str, Any]:
    """Return *fsm_def* with ``AgentConfig.instructions`` in every non-empty
    state and per-field instruction.

    Not reached: ``classification_extractions`` (the ``use_classification``
    think path), core's AMBIGUOUS transition classifier, ReasoningReact's
    reasoning-engine call, and any empty slot (see below).

    Interface contract (2 call sites: :meth:`BaseAgent._create_api` and
    ``SelfConsistencyAgent``, the one pattern that builds its ``API``
    directly):
        - ``None`` or blank *instructions* return *fsm_def* itself, unchanged.
        - Otherwise returns a copy (the caller's dict and its states are not
          mutated) where each state's non-empty ``response_instructions``
          (Pass 2), non-empty ``extraction_instructions`` (bulk Pass 1) and
          every ``field_extractions`` entry's ``extraction_instructions`` start
          with an ``Agent instructions:`` block. Empty instructions stay
          empty, so a silent state stays silent and no bulk call is added.
        - Never raises.

    # DECISION plan-2026-09-29T103145-06a5ec0a/D-047
    Do NOT put the instructions in the FSM ``persona`` (the plan's first
    choice): core shows ``persona`` only to Pass 2 (capped at
    ``MAX_PERSONA_LENGTH``, 4000 chars since D-048), never to the per-field
    calls where agents choose tools and answers. Do NOT fill
    an EMPTY instruction slot either: an empty ``response_instructions`` skips
    Pass 2 and an empty state ``extraction_instructions`` skips the bulk call.
    """
    text = (instructions or "").strip()
    if not text:
        return fsm_def
    block = f"Agent instructions: {text}\n\n"
    states: dict[str, Any] = {}
    for name, state in fsm_def.get("states", {}).items():
        state = dict(state)
        for key in ("response_instructions", "extraction_instructions"):
            if state.get(key):
                state[key] = block + state[key]
        if state.get("field_extractions"):
            state["field_extractions"] = [
                {**fe, "extraction_instructions": block + fe["extraction_instructions"]}
                if fe.get("extraction_instructions")
                else fe
                for fe in state["field_extractions"]
            ]
        states[name] = state
    return {**fsm_def, "states": states}


_INSTRUCTION_SLOTS = frozenset({"extraction_instructions", "response_instructions"})


def prompt_overflow_error(
    exc: ValueError, instructions: str | None, tools: Any
) -> AgentError | None:
    """Translate core's FSM-load length error into an actionable ``AgentError``.

    Interface contract (2 call sites: :meth:`BaseAgent._create_api` and
    ``SelfConsistencyAgent``, the two places that load an agent FSM):
        - *exc* is the ``ValueError`` ``API.from_definition`` raised for a
          dict definition; its ``__cause__`` is core's pydantic
          ``ValidationError``.
        - Returns an ``AgentError`` naming the overflowing slot, core's limit
          (read from the error, never a copy of core's literal), the length of
          *instructions* and the number of *tools* (anything with ``len``, or
          ``None``), when the cause is a ``string_too_long`` error on an
          ``extraction_instructions``/``response_instructions`` slot.
        - Returns ``None`` for any other error (the caller re-raises it).
        - Never raises.
    """
    cause = exc.__cause__
    if not isinstance(cause, ValidationError):
        return None
    for err in cause.errors():
        loc = err.get("loc") or ()
        if (
            err.get("type") == "string_too_long"
            and loc
            and loc[-1] in _INSTRUCTION_SLOTS
        ):
            return AgentError(
                ErrorMessages.PROMPT_SLOT_OVERFLOW.format(
                    slot=".".join(str(part) for part in loc),
                    limit=(err.get("ctx") or {}).get("max_length", "?"),
                    instructions=len((instructions or "").strip()),
                    tools=len(tools) if hasattr(tools, "__len__") else 0,
                )
            )
    return None


def _output_response_format(schema: Any) -> dict[str, Any] | None:
    """Build the ``response_format`` envelope for a Pydantic *schema*.

    Interface contract (two call sites: ``_init_context`` here, and
    ``native_fc``'s post-loop repair turn):

    Args:
        schema: ``AgentConfig.output_schema`` — a Pydantic ``BaseModel``
            subclass, ``None``, or any object (duck-typed on
            ``model_json_schema``).

    Returns:
        The ``{"type": "json_schema", ...}`` dict litellm/OpenAI accept as
        ``response_format``, or ``None`` when *schema* is ``None`` or does not
        expose ``model_json_schema``.  Never raises.
    """
    # DECISION plan-2026-07-21T191807-bf7ffe24/D-002
    # This helper is the ONE new abstraction that plan's Complexity Budget
    # allows inside the five existing packages (1/1), and it is earned by
    # EXACTLY two call sites: `_init_context` below, and `native_fc.run`'s
    # post-loop repair turn. It was extracted rather than copied because the
    # alternative -- native_fc building its own `{"type": "json_schema", ...}`
    # envelope -- is a second builder of the same provider contract, kept in
    # lockstep by hand, which is the drift `hardening.py`'s D-059 block already
    # records this repo paying for once.
    # Do NOT add a third caller by reflex: a call site that can set
    # `AgentConfig.output_schema` and go through `BaseAgent` gets this for free
    # and should. Do NOT make it raise on a bad schema either -- both callers
    # treat `None` as "no constrained decoding available", and an exception here
    # would turn a missing capability into a failed run.
    # See decisions.md D-002.
    if schema is None or not hasattr(schema, "model_json_schema"):
        return None
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
        },
    }


def strip_caller_context(
    context: Mapping[str, Any] | None,
    *,
    source: str,
    drop_internal: bool = False,
    warn: bool = True,
    run_keys: Set[str] = frozenset(),
) -> dict[str, Any]:
    """Copy caller-supplied context without the keys a run owns.

    Interface contract (callers: ``BaseAgent._init_context``,
    ``SelfConsistencyAgent.run``, ``SwarmAgent.run``, ``AgentGraph.run``,
    ``AgentServer`` ``/invoke`` and ``/stream``):

    Args:
        context: The caller's dict, or ``None`` (treated as empty).
        source: Label naming where the dict came from, used in the log line.
        drop_internal: Also drop every ``has_internal_prefix`` key. Only for a
            trust boundary (``AgentServer``); in-process callers may pass
            internal policy inputs such as ``_sensitive``.
        warn: Log dropped keys at WARNING (default) or DEBUG. DEBUG is for
            in-process propagation where dropping is expected (AgentGraph edges).
        run_keys: The receiving pattern's own run outputs (answer, verdict,
            draft and progress keys; ``pattern_run_output_keys``), dropped in
            addition to ``RUN_OUTPUT_KEYS``.

    Returns:
        A new dict (the input is never mutated) without ``RUN_OUTPUT_KEYS``
        (which include ``ContextKeys.DRIVER_APPROVAL``), without *run_keys*,
        and without internal keys when ``drop_internal``. Never raises for a
        mapping input.
    """
    # DECISION plan-2026-09-29T103145-06a5ec0a/D-002: caller context may not
    # carry the driver-only approval grant or any run output. A forged
    # `_approval_granted` plus `tool_name`/`tool_input` ran a gated tool with the
    # callback never asked (SEC-01); forged `observation_count`/`should_terminate`/
    # `final_answer` gave success=True with zero tools (LOOP-12). Do NOT
    # blanket-drop internal-prefix keys here for in-process callers: `_sensitive`
    # reaches the approval policy by design (TestDriverSeesRefusalContext) and the
    # harness passes its roots this way. Only AgentServer sets `drop_internal`.
    # Do NOT answer a forged remote key with 400 either: clients that echo
    # context would break; the drop is logged instead. See decisions.md D-002.
    kept: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in (context or {}).items():
        if (
            key in RUN_OUTPUT_KEYS
            or key in run_keys
            or (drop_internal and isinstance(key, str) and has_internal_prefix(key))
        ):
            dropped.append(str(key))
        else:
            kept[key] = value
    if dropped:
        message = (
            f"{source}: dropped run-owned or internal keys a caller may not "
            f"set: {sorted(dropped)}"
        )
        if warn:
            logger.warning(message)
        else:
            logger.debug(message)
    return kept


def pattern_run_output_keys(agent: Any) -> frozenset[str]:
    """The run outputs *agent*'s pattern writes for itself, beyond ``RUN_OUTPUT_KEYS``.

    Interface contract (callers: ``BaseAgent._init_context``,
    ``SelfConsistencyAgent.run``, ``AgentGraph.run`` and ``SwarmAgent.run``
    for the node or agent they hand context to):

    Args:
        agent: An agent instance or class; anything without a
            ``_run_output_keys`` attribute counts as having none.

    Returns:
        The pattern's ``_run_output_keys`` class attribute as a frozenset
        (empty for the ReAct family and for non-agents). Never raises.
    """
    keys = getattr(agent, "_run_output_keys", None)
    if not isinstance(keys, (set, frozenset)):
        return frozenset()
    return frozenset(k for k in keys if isinstance(k, str))


def caller_prompt_keys(
    context: Mapping[str, Any] | None, exclude: Collection[str] = ()
) -> tuple[str, ...]:
    """Caller context keys a pattern's narrowed field prompts must list.

    Interface contract (callers: ``ReactAgent``, ``ReflexionAgent``,
    ``ReasoningReactAgent``, ``ParallelReactAgent``, ``OrchestratorAgent``,
    ``ADaPTAgent``, ``REWOOAgent`` ``run``): typed field configs name their
    prompt keys (``context_keys``, D-017 of plan 06a5ec0a), so a caller's
    hint such as ``suggested_tool`` would otherwise vanish from the prompt.
    Returns the caller's string keys in order, without internal-prefixed
    keys, ``RUN_OUTPUT_KEYS``, ``agent_trace`` and ``exclude`` (a pattern's
    own ``_run_output_keys``). Core's prompt filter still drops
    secret-looking entries. Never raises.
    """
    return tuple(
        key
        for key in (context or {})
        if isinstance(key, str)
        and not has_internal_prefix(key)
        and key not in RUN_OUTPUT_KEYS
        and key != ContextKeys.AGENT_TRACE
        and key not in exclude
    )


def accepts_tools(agent_cls: type) -> bool:
    """Tell whether *agent_cls*'s constructor takes a ``tools`` argument.

    Interface contract (callers: ``create_agent`` and its tests):

    Args:
        agent_cls: An agent class (any class; non-agents are fine).

    Returns:
        ``True`` when some ``__init__`` on the MRO names a ``tools`` parameter
        and every ``__init__`` before it forwards ``**kwargs`` (VerifiedReact and
        AutoMemory forward to ReactAgent). ``BaseAgent.__init__`` ends the walk:
        its ``**api_kwargs`` is not a tools sink (it rejects ``tools``). Never
        raises; an uninspectable signature counts as ``False``.
    """
    for klass in agent_cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        if klass is BaseAgent or klass is object:
            return False
        try:
            params = inspect.signature(init).parameters
        except (TypeError, ValueError):
            return False
        if "tools" in params and params["tools"].kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return True
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return False
    return False


def flagged_tool_names(tools: Any) -> list[str]:
    """Names of the tools in *tools* registered with ``requires_approval=True``.

    Interface contract (callers: ``BaseAgent._hitl_active``, the construction
    warning, and ``BaseAgent._refuse_flagged_tools``):

    Args:
        tools: a ``ToolRegistry``, or ``None``/any object without
            ``list_tools`` (an agent with no registry).

    Returns:
        The flagged names in registration order (a snapshot; ``[]`` when there
        is no registry or nothing is flagged). Never raises.
    """
    list_tools = getattr(tools, "list_tools", None)
    if list_tools is None:
        return []
    return [t.name for t in list_tools() if getattr(t, "requires_approval", False)]


def _flagged_tool_policy(tools: Any) -> ApprovalPolicy:
    """Default approval policy: the selected call names a flagged registered tool.

    Read at call time, so a tool registered after construction is covered.
    """

    def policy(call: ToolCall, context: dict[str, Any]) -> bool:
        return call.tool_name in flagged_tool_names(tools)

    return policy


# Every HumanInTheLoop constructor name (approval_policy, on_escalation, ...):
# given to an agent they mean "HITL wanted" and must never reach litellm with
# the gate silently absent (D-003, review item 4). Derived from the signature
# so a new HumanInTheLoop parameter is covered without a hand-kept list.
_HITL_CTOR_KWARGS: frozenset[str] = frozenset(
    name
    for name, param in inspect.signature(HumanInTheLoop.__init__).parameters.items()
    if name != "self"
    and param.kind
    not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
)


def _reject_misplaced_kwargs(pattern: str, api_kwargs: Mapping[str, Any]) -> None:
    """Raise ``TypeError`` when *api_kwargs* holds a name that only lands there by mistake.

    Called once from ``BaseAgent.__init__``. *pattern* is the concrete class
    name, used in the message. Every other kwarg is left alone (D-003).
    """
    # DECISION plan-2026-09-29T103145-06a5ec0a/D-003: a denylist, not a
    # whitelist. `hitl=`/`tools=` on a pattern without them used to be forwarded
    # to litellm with HITL silently ignored (SEC-02, PAT-12); `model=` crashed
    # later with "multiple values" (API-02). Do NOT reject unknown kwargs in
    # general: `seed`, `timeout`, `caching`, `handlers`, `llm_interface`, ... are
    # legitimate API/litellm passthrough. See decisions.md D-003.
    misplaced = sorted(
        k for k in api_kwargs if k in MISPLACED_AGENT_KWARGS or k in _HITL_CTOR_KWARGS
    )
    if misplaced:
        raise TypeError(
            f"{pattern} does not accept {misplaced}; this pattern cannot use "
            f"them and they would be forwarded to the LLM provider unchecked"
        )
    config_owned = sorted(k for k in api_kwargs if k in CONFIG_OWNED_KWARGS)
    if config_owned:
        raise TypeError(
            f"{pattern} does not accept {config_owned} as keyword arguments; "
            f"set them on AgentConfig instead, e.g. "
            f"{pattern}(config=AgentConfig({config_owned[0]}=...))"
        )


def artifact_text(value: Any) -> str:
    """The text form of a generated artifact read from context.

    Contract: ``None`` and every falsy non-``str`` value (``False``, ``0``,
    ``0.0``, ``{}``, ``[]``) -> ``""``, i.e. no answer; a ``str`` as is; a
    non-empty ``dict``/``list`` (an ``any`` artifact field the model returned
    as native JSON) -> indented JSON text, so ``AgentResult.answer`` stays a
    string that ``output_schema`` validation can parse; any other value ->
    ``str(value)``. Never raises (unserialisable leaves go through ``str``).
    Callers (answer extraction, the success seam, ``evaluation_fn``) rely on
    the empty string meaning "nothing was produced": an ``any`` artifact the
    model filled with ``[]``/``{}``/``false`` must not count as an answer
    (D-056 of plan 06a5ec0a).
    """
    if value is None or (not isinstance(value, str) and not value):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return str(value)


class BaseAgent(ABC):
    """Abstract base class for FSM-LLM agents.

    Provides the common conversation loop, budget enforcement, answer
    extraction, and trace building. Subclasses implement only the
    pattern-specific logic: FSM building, handler registration, and
    context setup.

    Usage for end-users is unchanged — all existing agent constructors
    and ``run()`` signatures are preserved.  Additionally, agents now
    support ``__call__``::

        result = agent("What is 2+2?")
    """

    # DECISION plan-2026-09-29T103145-06a5ec0a/D-052: each pattern lists the
    # context keys its run writes for itself (answer, draft, verdict, success
    # and progress keys). Caller context and graph/swarm hand-offs never seed
    # them: core skip-if-set never re-extracts a set key, so a forged
    # `draft_output` or `generated_output` shipped as a successful answer with
    # no work done. Do NOT fold these into the global RUN_OUTPUT_KEYS: a
    # pattern's run output can be a legitimate input of another pattern.
    # Do NOT list a key a caller legitimately supplies (task, domain hints).
    _run_output_keys: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        config: AgentConfig | None = None,
        **api_kwargs: Any,
    ) -> None:
        _reject_misplaced_kwargs(type(self).__name__, api_kwargs)
        self.config = config or AgentConfig()
        self._api_kwargs = api_kwargs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @abstractmethod
    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Run the agent on a task. Implemented by each agent pattern."""
        ...

    def __call__(
        self,
        task: str,
        **kwargs: Any,
    ) -> AgentResult:
        """Callable shorthand: ``agent("task")`` → ``agent.run("task")``."""
        return self.run(task, **kwargs)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model})"

    # ------------------------------------------------------------------
    # Common conversation loop
    # ------------------------------------------------------------------

    def _run_conversation_loop(
        self,
        api: API,
        context: dict[str, Any],
        start_time: float,
        agent_type: str,
        max_iterations: int | None = None,
    ) -> tuple[list[str], dict[str, Any], int]:
        """Run the agent's FSM to its terminal state on core's bounded run.

        Returns:
            Tuple of (responses, final_context, step_count). ``responses``
            holds the reply of every state that spoke, in order (a silent
            state contributes nothing).

        Raises:
            AgentTimeoutError, BudgetExhaustedError: a run budget was spent
                (``_run_budgets``, ``_budget_error``).
        """
        # DECISION plan-2026-09-30T062855-07ad3f8c/D-030
        # The run loop, the step cap and the wall-clock budget are core's
        # (`API.run_until_terminal`). Do NOT bring back a `while` loop, a step
        # counter or a synthetic `converse` turn here or in
        # `_standard_run_stream`, and do NOT filter `[state]` markers in this
        # package (supersedes 06a5ec0a/D-030): a step has no user message and
        # a silent state returns no text. If a pattern needs a budget or loop
        # behaviour core cannot express, extend core's loop with a core test.
        # See decisions.md D-030.
        conv_id, greeting = self._start_conversation(api, context)
        log = logger.bind(
            conversation_id=conv_id,
            package="fsm_llm.agents",
            agent_type=agent_type,
        )

        try:
            max_steps, max_seconds = self._run_budgets(start_time, max_iterations)
            try:
                steps = api.run_until_terminal(
                    conv_id,
                    max_steps=max_steps,
                    max_seconds=max_seconds,
                    before_step=partial(self._on_loop_iteration, api, conv_id),
                )
            except RunBudgetExceededError as exc:
                raise self._budget_error(exc, max_iterations) from exc

            replies = [greeting, *(step.response for step in steps)]
            final_context = api.get_data(conv_id)
            log.info(LogMessages.AGENT_COMPLETE.format(iterations=len(steps)))
            return [r for r in replies if r is not None], final_context, len(steps)

        finally:
            api.end_conversation(conv_id)

    @staticmethod
    def _start_conversation(
        api: API, context: dict[str, Any]
    ) -> tuple[str, str | None]:
        """Start the run's conversation; return ``(conv_id, greeting)``.

        Shared by ``_run_conversation_loop`` and ``_standard_run_stream``.
        ``greeting`` is the initial state's reply, or ``None`` when that state
        is silent. Whether it spoke is asked of core (a greeting is recorded
        in history only when the state spoke, 07ad3f8c/D-028), never read off
        the returned text.
        """
        conv_id, greeting = api.start_conversation(context)
        spoke = bool(api.get_conversation_history(conv_id))
        return conv_id, greeting if spoke else None

    def _on_loop_iteration(  # noqa: B027
        self,
        api: API,
        conv_id: str,
        iteration: int,
    ) -> None:
        """Hook called before each step of the run (core's ``before_step``).

        ``iteration`` is the step number, from 1. Override for HITL approval
        gates or other between-step logic. Default is a no-op.
        """

    # ------------------------------------------------------------------
    # Context initialisation
    # ------------------------------------------------------------------

    def _init_context(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the standard initial context for an agent run.

        Sets ``TASK``, ``AGENT_TRACE``, ``ITERATION_COUNT`` and
        ``MAX_ITERATIONS_REACHED`` (False; so a run's ``final_context`` carries it).
        Seeds ``OBSERVATION_COUNT`` to 0. Caller keys in ``RUN_OUTPUT_KEYS``
        (including the driver-only approval grant) and in the pattern's own
        ``_run_output_keys`` are dropped with a WARNING by
        ``strip_caller_context``; other internal-prefix keys pass through.
        Warns if *initial_context* already contains reserved keys.
        """
        context = strip_caller_context(
            initial_context,
            source="initial_context",
            run_keys=pattern_run_output_keys(self),
        )
        reserved = {
            ContextKeys.TASK,
            ContextKeys.AGENT_TRACE,
            ContextKeys.ITERATION_COUNT,
            ContextKeys.MAX_ITERATIONS_REACHED,
        }
        conflicts = reserved & context.keys()
        if conflicts:
            logger.warning(
                f"initial_context contains reserved keys that will be "
                f"overwritten: {conflicts}"
            )
        context[ContextKeys.TASK] = task
        context[ContextKeys.AGENT_TRACE] = []
        context[ContextKeys.ITERATION_COUNT] = 0
        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-007: seed the forced-stop
        # flag False. Bulk extraction fills only UNSET keys, so the model cannot
        # write True and pass every evidence guard (P2-W2). Only the limiter
        # and stall handlers write True. Do NOT drop the seed or set it None.
        context[ContextKeys.MAX_ITERATIONS_REACHED] = False
        # Evidence counter starts at zero for every run; the caller's value was
        # dropped above (D-002 of plan 06a5ec0a, LOOP-12).
        context[ContextKeys.OBSERVATION_COUNT] = 0

        # Schema-enforced output: when output_schema is set, store the
        # JSON schema as response_format so the pipeline can pass it to
        # the LLM for constrained decoding on the conclude state.
        response_format = _output_response_format(self.config.output_schema)
        if response_format is not None:
            context[CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT] = response_format

        if extra:
            context.update(extra)
        return context

    # ------------------------------------------------------------------
    # Handler registration helpers
    # ------------------------------------------------------------------

    def _register_iteration_limiter(
        self,
        api: API,
        handler_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Register the standard iteration-limiter handler."""
        api.register_handler(
            api.create_handler(HandlerNames.ITERATION_LIMITER)
            .with_priority(HandlerPriorities.ITERATION_LIMITER)
            .at(HandlerTiming.PRE_TRANSITION)
            .do(handler_fn)
        )

    def _register_tool_executor(
        self,
        api: API,
        state: str,
        handler_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Register the standard tool-executor handler on a state entry."""
        api.register_handler(
            api.create_handler(HandlerNames.TOOL_EXECUTOR)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_entry(state)
            .do(handler_fn)
        )

    def _register_think_loop_handlers(
        self, api: API, fresh_keys: Sequence[str]
    ) -> None:
        """Register the ReAct-family think bookkeeping (react, reflexion,
        reasoning_react, parallel_react).

        ``fresh_keys`` are the loop values ``think`` produces; they are cleared
        on think entry so think extracts them again (skip-if-set, D-008 of plan
        06a5ec0a). A ``should_terminate`` the limiter forced is cleared too
        (the think conclude edge reads ``max_iterations_reached`` itself), so
        the last think turn gives its own verdict; one a forcing handler
        recorded a reason for is kept (D-051). ``agent_feedback`` is cleared
        on think exit, once the think turn has read it. Under
        :attr:`_hitl_active`, ``should_terminate`` is also cleared on
        ``await_approval`` entry (a forced True is kept).
        """
        from .handlers import make_fresh_keys_handler

        # DECISION plan-2026-09-29T103145-06a5ec0a/D-051: keep_limiter_forced
        # is False only here. Do NOT keep a limiter-forced should_terminate on
        # think entry: the model is then never asked on its last turn and a
        # real conclusion there reads as a forced max_iterations stop.
        api.register_handler(
            api.create_handler(HandlerNames.THINK_FRESH_KEYS)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_entry(AgentStates.THINK)
            .do(make_fresh_keys_handler(fresh_keys, keep_limiter_forced=False))
        )
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-029
        # agent_feedback is consumed by think, so it is cleared on think EXIT.
        # Do NOT clear it on think entry with the fresh keys: act writes it and
        # the act -> think transition enters think before think extracts, so an
        # entry clear erases every warning and denial unread (LOOP-06). Do NOT
        # make it a compactor transient key either. See decisions.md D-029.
        api.register_handler(
            api.create_handler(HandlerNames.FEEDBACK_CONSUMED)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_exit(AgentStates.THINK)
            .do(make_fresh_keys_handler([ContextKeys.AGENT_FEEDBACK]))
        )
        if self._hitl_active:
            # D-029: an early should_terminate from the gated call's think turn
            # must not beat await_approval -> act; a forced True still routes
            # await_approval -> conclude.
            api.register_handler(
                api.create_handler(HandlerNames.APPROVAL_FRESH_KEYS)
                .with_priority(HandlerPriorities.TOOL_EXECUTOR)
                .on_state_entry(AgentStates.AWAIT_APPROVAL)
                .do(make_fresh_keys_handler([ContextKeys.SHOULD_TERMINATE]))
            )

    def _register_hitl_gate(
        self,
        api: API,
        checker_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Register the HITL approval-gate handler."""
        api.register_handler(
            api.create_handler(HandlerNames.HITL_GATE)
            .with_priority(HandlerPriorities.HITL_GATE)
            .at(HandlerTiming.CONTEXT_UPDATE)
            .when_keys_updated(ContextKeys.TOOL_NAME)
            .do(checker_fn)
        )

    # ------------------------------------------------------------------
    # HITL loop-iteration helper
    # ------------------------------------------------------------------

    @property
    def _hitl_active(self) -> bool:
        # Single source of truth for "this run needs HITL approval gating".
        # INVARIANT: the await_approval FSM state (include_approval_state)
        # MUST be built under exactly this predicate, because it is also the
        # predicate that registers the runtime approval gate. If the two
        # diverge, the gate can set approval_required=True with no
        # await_approval state to intercept it, and the tool executes
        # un-gated (THINK -> act runs execute_tool before the loop's
        # _handle_hitl_approval hook).
        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-005
        # Moved here from ReactAgent so React, Reflexion and ReasoningReact
        # share ONE predicate for the FSM state, the gate and the
        # AgentHandlers refusal. Do NOT re-add a per-agent copy, and do NOT
        # AND it with "some tool is flagged": a policy may gate any tool.
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-004
        # With a policy, the policy alone decides (unchanged). With a callback
        # and NO policy, requires_approval=True tools are the default policy:
        # active iff some registered tool is flagged. Do NOT go back to
        # "no policy means no gating" (the flag was a no-op, SEC-03), and do
        # NOT gate a policy-less run with no flagged tool (the harness runs
        # callback-only HITL with no flagged tools). See decisions.md D-004.
        hitl: HumanInTheLoop | None = getattr(self, "hitl", None)
        if hitl is None:
            return False
        if hitl.has_approval_policy:
            return True
        return hitl.has_approval_callback and bool(
            flagged_tool_names(getattr(self, "tools", None))
        )

    @property
    def _approval_predicate(self) -> ApprovalPolicy | None:
        """The approval policy for the refusal, the gate and the driver, or None.

        Set only under :attr:`_hitl_active` (plan-2026-09-24T091842-c1d5bfbc
        D-004/D-005), so the refusal never gates a run without the state. The
        HITL policy when one is set, else the flagged-tool default (D-004).
        """
        hitl: HumanInTheLoop | None = getattr(self, "hitl", None)
        if hitl is None or not self._hitl_active:
            return None
        if hitl.has_approval_policy:
            return hitl.requires_approval
        return _flagged_tool_policy(getattr(self, "tools", None))

    def _register_approval_gate(self, api: API) -> None:
        """Register the HITL gate under :attr:`_hitl_active` with the shared predicate."""
        hitl: HumanInTheLoop | None = getattr(self, "hitl", None)
        predicate = self._approval_predicate
        if hitl is None or predicate is None:
            return
        self._register_hitl_gate(api, make_hitl_checker(hitl, policy=predicate))

    def _refuse_unapprovable_flagged_tools(self) -> None:
        """Raise ``AgentError`` when flagged tools exist and nobody can decide on them.

        For the HITL patterns (ReAct, Reflexion, ReasoningReact). Call it in
        ``__init__`` after ``self.tools`` and ``self.hitl`` are set and at the
        start of ``run()``/``run_stream()`` (a registry can gain a flagged tool
        after construction). Raises when ``self.tools`` holds a
        ``requires_approval`` tool and ``self.hitl`` is ``None`` or has neither
        an approval callback nor a policy. With a policy and no callback it
        only WARNS: the policy owns the decision, and a call it gates raises
        ``ApprovalDeniedError``. No-op without a flagged tool.
        """
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-052: fail closed, as D-005
        # does for the patterns without HITL. With `hitl=None` or an empty
        # HumanInTheLoop() a requires_approval tool ran unasked (a construction
        # WARNING only). Do NOT downgrade this back to a warning, and do NOT
        # raise when a policy is set: an explicit policy owns its decision
        # (D-004 never ANDs it with the flag).
        flagged = flagged_tool_names(getattr(self, "tools", None))
        if not flagged:
            return
        hitl: HumanInTheLoop | None = getattr(self, "hitl", None)
        if hitl is None or not (hitl.has_approval_callback or hitl.has_approval_policy):
            raise AgentError(
                f"{type(self).__name__}: tools {flagged} have requires_approval="
                f"True but nobody can approve them. Pass "
                f"hitl=HumanInTheLoop(approval_callback=...), or register these "
                f"tools without requires_approval."
            )
        if not hitl.has_approval_callback:
            logger.warning(
                f"{type(self).__name__}: tools {flagged} have requires_approval="
                f"True and no HumanInTheLoop approval_callback is configured; "
                f"the approval policy decides, and a call it gates raises "
                f"ApprovalDeniedError"
            )

    def _refuse_flagged_tools(self) -> None:
        """Raise ``AgentError`` when ``self.tools`` holds a ``requires_approval`` tool.

        For patterns with no HITL path (REWOO, PlanExecute, ParallelReact,
        native_fc). Call it in ``__init__`` after ``self.tools`` is set and at
        the start of ``run()``, before any LLM call: a registry can gain a
        flagged tool after construction. No-op without a registry.
        """
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-005: fail closed. Do NOT
        # downgrade this to a warning, and do NOT drop the run() re-check: these
        # patterns execute tools with no approval gate, so a flagged tool would
        # run unapproved. Lift it only when the pattern gains real HITL (Track B).
        flagged = flagged_tool_names(getattr(self, "tools", None))
        if flagged:
            raise AgentError(
                f"{type(self).__name__} has no human-in-the-loop approval, but "
                f"tools {flagged} have requires_approval=True. Use ReactAgent, "
                f"ReflexionAgent or ReasoningReactAgent with a HumanInTheLoop, "
                f"or register these tools without requires_approval."
            )

    def _handle_hitl_approval(self, api: API, conv_id: str) -> None:
        """Check and process HITL approval for the current context.

        Shared logic for agents that use synchronous HITL approval gates
        (ReactAgent, ReflexionAgent, ReasoningReactAgent).  Subclasses must
        set ``self.hitl`` to a :class:`HumanInTheLoop` instance (or ``None``).
        """
        from .handlers import approval_grant
        from .tools import normalize_tool_input, redact_secret_entries

        hitl: HumanInTheLoop | None = getattr(self, "hitl", None)
        if hitl is None:
            return

        current_context = api.get_data(conv_id)
        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-023
        # Ask unless approval_granted is exactly True, and store a strict bool.
        # Do NOT test truthiness: await_approval routes on == True / == False
        # only, so a model-written "yes" or a callback's None parked the run
        # there, unasked, until BudgetExhaustedError.
        if current_context.get(ContextKeys.APPROVAL_GRANTED) is True:
            return

        tool_name = current_context.get(ContextKeys.TOOL_NAME) or ""
        tool_input = normalize_tool_input(current_context.get(ContextKeys.TOOL_INPUT))
        reasoning = current_context.get(ContextKeys.REASONING, "")

        tool_call = ToolCall(
            tool_name=tool_name,
            parameters=tool_input,
            reasoning=reasoning,
        )

        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-005: ask only about a
        # registered tool the shared predicate gates. Do NOT ask on a bare
        # approval_required: the model can write it (tool "none"; approval
        # fatigue), and an extracted True for an ungated tool routes into
        # await_approval after the gate wrote False, where nothing else sets
        # approval_granted (BLOCKED until BudgetExhaustedError). Route it back.
        # The gate reads the FULL context (internal keys included), exactly as
        # AgentHandlers.approval_refusal does. Do NOT pass get_data's stripped
        # view: a policy on a `_` key then never asks while the refusal blocks.
        gate = self._approval_predicate
        tools = getattr(self, "tools", None)
        sub_id = api.get_sub_conversation_id(conv_id)
        full = api.fsm_manager.get_complete_conversation(sub_id)["collected_data"]
        if not (tools and tool_name in tools and gate and gate(tool_call, full)):
            required = current_context.get(ContextKeys.APPROVAL_REQUIRED)
            if required or api.get_current_state(conv_id) == AgentStates.AWAIT_APPROVAL:
                stray = (ContextKeys.APPROVAL_REQUIRED, ContextKeys.APPROVAL_GRANTED)
                api.update_context(conv_id, dict.fromkeys(stray, False))
            return
        if not current_context.get(ContextKeys.APPROVAL_REQUIRED):
            return
        approved = hitl.request_approval(tool_call, current_context) is True
        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-004
        # The public keys only route the FSM (the model can forge them). The
        # driver-only grant, bound to the exact call the human saw, is what
        # AgentHandlers.approval_refusal checks. Do NOT write a bare True here
        # and do NOT write it on denial: a grant must name its one call.
        grant = approval_grant(tool_name, tool_input) if approved else None
        api.update_context(
            conv_id,
            {
                ContextKeys.APPROVAL_GRANTED: approved,
                ContextKeys.APPROVAL_REQUIRED: False,
                ContextKeys.DRIVER_APPROVAL: grant,
            },
        )
        if not approved:
            # LOOP-06 (D-021 of plan 06a5ec0a): the denial reaches the next think
            # turn as feedback, never as an observation: an observation would
            # bump observation_count and satisfy the D-008 conclude guard with
            # no tool result. The input shown is the redacted copy (D-016).
            shown = redact_secret_entries(tool_input)
            # D-034 (plan 07ad3f8c): the feedback is gone after the next think
            # turn, so the refusal is also kept as a final fact for conclude.
            record = (
                f"{tool_name}({shown}): NOT performed. The human approver "
                "refused this action; the refusal is final for this run."
            )
            refused = list(full.get(ContextKeys.REFUSED_ACTIONS) or [])
            if record not in refused:
                refused.append(record)
            api.update_context(
                conv_id,
                {
                    ContextKeys.TOOL_NAME: None,
                    ContextKeys.TOOL_INPUT: None,
                    ContextKeys.AGENT_FEEDBACK: (
                        f"The human reviewer denied the call {tool_name}({shown}). "
                        "Do not repeat it; choose another tool or approach."
                    ),
                    ContextKeys.REFUSED_ACTIONS: refused,
                },
            )

    # ------------------------------------------------------------------
    # Budget enforcement
    # ------------------------------------------------------------------

    def _check_budgets(self, start_time: float) -> None:
        """Raise ``AgentTimeoutError`` when the wall clock of a run is spent.

        For callers that are not inside a core bounded run (SelfConsistency
        samples, ``native_fc``). FSM runs pass their budgets to core instead
        (``_run_budgets``).
        """
        if time.monotonic() - start_time > self.config.timeout_seconds:
            raise AgentTimeoutError(self.config.timeout_seconds)

    def _run_budgets(
        self, start_time: float, max_iterations: int | None = None
    ) -> tuple[int, float]:
        """The ``(max_steps, max_seconds)`` of a core bounded run.

        Interface contract (callers: ``_run_conversation_loop``,
        ``_standard_run_stream``): ``max_steps`` is ``max_iterations`` (default
        ``config.max_iterations``) times ``FSM_BUDGET_MULTIPLIER``;
        ``max_seconds`` is what is left of ``config.timeout_seconds`` since
        ``start_time`` (always above 0). Raises ``AgentTimeoutError`` when
        nothing is left: core refuses a non-positive time budget.
        """
        remaining = self.config.timeout_seconds - (time.monotonic() - start_time)
        if remaining <= 0:
            raise AgentTimeoutError(self.config.timeout_seconds)
        max_iters = max_iterations or self.config.max_iterations
        return max_iters * Defaults.FSM_BUDGET_MULTIPLIER, remaining

    def _budget_error(
        self, exc: RunBudgetExceededError, max_iterations: int | None = None
    ) -> AgentError:
        """The public agent error for a spent core run budget.

        Interface contract (same callers as ``_run_budgets``): the seconds
        budget maps to ``AgentTimeoutError(config.timeout_seconds)``, the
        steps budget to ``BudgetExhaustedError`` citing the step ceiling.
        Returns the error; the caller raises it ``from exc``.
        """
        if exc.budget == "seconds":
            return AgentTimeoutError(self.config.timeout_seconds)
        max_iters = max_iterations or self.config.max_iterations
        ceiling = max_iters * Defaults.FSM_BUDGET_MULTIPLIER
        # LOOP-17: cite the loop-turn ceiling that was hit, not max_iterations.
        return BudgetExhaustedError(
            "iterations",
            ceiling,
            detail=(
                f"{ceiling} loop turns = max_iterations {max_iters} x "
                f"FSM_BUDGET_MULTIPLIER {Defaults.FSM_BUDGET_MULTIPLIER}"
            ),
        )

    # ------------------------------------------------------------------
    # Answer extraction
    # ------------------------------------------------------------------

    def _extract_answer(
        self,
        final_context: dict[str, Any],
        responses: list[str],
        extra_keys: list[str] | None = None,
    ) -> str:
        """Extract answer with a fallback chain.

        1. Try ``ContextKeys.FINAL_ANSWER``
        2. Try each key in *extra_keys* (e.g. ``DRAFT_OUTPUT``)
        3. Try responses in reverse order
        4. Return default message
        """
        # Primary: final_answer
        answer = final_context.get(ContextKeys.FINAL_ANSWER)
        if answer and isinstance(answer, str) and answer.strip():
            return str(answer)

        # Secondary: extra context keys (pattern-specific)
        for key in extra_keys or []:
            val = artifact_text(final_context.get(key)).strip()
            if val:
                return val

        # Tertiary: last non-empty response
        for response in reversed(responses):
            if response and response.strip():
                return response.strip()

        return "Agent could not determine an answer."

    # ------------------------------------------------------------------
    # Trace building
    # ------------------------------------------------------------------

    @staticmethod
    def _has_execution_evidence(
        final_context: dict[str, Any],
        evidence_keys: list[str],
    ) -> bool:
        """True if any ``evidence_keys`` context value proves real execution.

        Generic over the three planner evidence shapes:
        - a non-empty dict (e.g. ReWOO ``evidence`` mapping) → real;
        - a list whose dict entries all carry a ``success`` flag (e.g.
          Orchestrator ``worker_results``) → real only if ≥1 entry succeeded
          (placeholder/failed-worker entries are NOT evidence);
        - any other non-empty list (e.g. plan_execute ``step_results``, which
          have no per-entry success flag) → real.
        """
        for key in evidence_keys:
            value = final_context.get(key)
            if not value:
                continue
            if isinstance(value, dict):
                return True
            if isinstance(value, list):
                dict_entries = [e for e in value if isinstance(e, dict)]
                if dict_entries and all("success" in e for e in dict_entries):
                    if any(e.get("success") for e in dict_entries):
                        return True
                    # all entries failed/placeholder → not evidence; keep looking
                else:
                    return True
        return False

    @staticmethod
    def _completion_is_real(
        final_context: dict[str, Any],
        trace: AgentTrace,
        extra_answer_keys: list[str] | None,
        execution_evidence_keys: list[str] | None = None,
    ) -> bool:
        """True if the run produced a genuine result.

        A real completion has either a designated answer key
        (``FINAL_ANSWER`` or a pattern-specific ``extra_answer_key``) or at
        least one executed tool call. When BOTH are absent the ``answer``
        can only have come from the prose-fallback in ``_extract_answer``
        (a planner state's Pass-2 text leaking as the result) — that is a
        degenerate completion, not a success.

        # DECISION plan_2026-05-31_cb91a9d5/D-001 [STALE]: when ``execution_evidence_keys``
        # is supplied (planner patterns: orchestrator/rewoo/plan_execute), the
        # answer-key/tool-call test above is NOT sufficient. A planner can reach
        # a synthesis state that sets ``final_answer`` (and record a ``delegate``
        # control action that _build_trace turns into a fake ToolCall) while
        # having executed ZERO real work — weak 4b decomposition routes straight
        # to synthesis via the fallback transitions. That filler must report
        # success=False. So in planner mode we require genuine EXECUTION EVIDENCE
        # (a successful worker / non-empty tool evidence / executed steps)
        # instead. This extends D-001 (plan_26c9510a) from "no answer key AND no
        # tool" to also cover "answer key but no real execution". Opt-in per
        # call-site → every non-planner pattern (execution_evidence_keys=None)
        # keeps the original has_answer_key-OR-tools_executed behavior unchanged.
        """
        if execution_evidence_keys:
            return BaseAgent._has_execution_evidence(
                final_context, execution_evidence_keys
            )
        has_answer_key = bool(
            str(final_context.get(ContextKeys.FINAL_ANSWER) or "").strip()
        ) or any(
            artifact_text(final_context.get(k)).strip()
            for k in (extra_answer_keys or [])
        )
        tools_executed = len(trace.tool_calls) > 0
        return has_answer_key or tools_executed

    @staticmethod
    def _forced_stop_reason(
        final_context: dict[str, Any], *, judged: bool = False
    ) -> str | None:
        """The ``StopReason`` of a forced stop, or ``None`` if none was forced.

        Interface contract (callers: ``_run_outcome``, ``ADaPTAgent.run``):
        a run was forced when ``max_iterations_reached is True`` (limiter,
        stall) or a forcing handler recorded a reason under
        ``ContextKeys.FORCED_STOP_REASON`` (forced pass at a revision limit).
        Returns that recorded reason when it is one of ``StopReason.FORCED``,
        else ``StopReason.MAX_ITERATIONS`` for a bare forced flag. With
        ``judged`` (EvalOpt, MakerChecker: a judge handler rules on every
        output that ships) only a recorded reason counts; the bare flag says
        the budget ran out, not that the verdict was overridden. Never raises.
        """
        from .handlers import recorded_forced_reason

        recorded = recorded_forced_reason(final_context)
        if recorded is not None:
            return recorded
        if not judged and final_context.get(ContextKeys.MAX_ITERATIONS_REACHED) is True:
            return StopReason.MAX_ITERATIONS
        return None

    @staticmethod
    def _run_outcome(
        final_context: dict[str, Any],
        trace: AgentTrace,
        extra_answer_keys: list[str] | None,
        execution_evidence_keys: list[str] | None = None,
        judged: bool = False,
    ) -> tuple[bool, str]:
        """Compute ``(success, stop_reason)`` for a finished FSM run.

        Interface contract (callers: ``_standard_run``; ``ADaPTAgent.run``
        for an undecomposed run): the one success rule for every pattern
        whose result comes from an FSM run.

        Returns:
            ``(False, <forced reason>)`` when the run was forced to stop
            (``_forced_stop_reason``, ``judged`` passed through); else, in
            planner mode
            (``execution_evidence_keys``), ``(True, "evidence")`` with real
            execution evidence; otherwise ``(True, "answered")`` when
            ``_completion_is_real`` (an answer key or an executed tool call);
            and ``(False, "no_result")`` in every other case. Never raises.
        """
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-011: a forced stop, forced
        # pass or stall reports success=False with its reason, and the caller
        # still ships the last answer. Do NOT raise here (c1d5bfbc D-022: a
        # forced budget stop ships its output), and do NOT let the answer key
        # or a tool call override the forced flag (a forced EvalOpt/MakerChecker
        # pass always has an answer key, which is how it read as success=True).
        # Do NOT compute success per pattern: this is the one rule.
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-051: `judged` patterns
        # (EvalOpt, MakerChecker) report forced only on the judge handler's
        # recorded override. Do NOT read their bare limiter flag here: a
        # genuine pass on the limiter's round read as forced_pass.
        forced = BaseAgent._forced_stop_reason(final_context, judged=judged)
        if forced is not None:
            return False, forced
        if execution_evidence_keys:
            if BaseAgent._has_execution_evidence(
                final_context, execution_evidence_keys
            ):
                return True, StopReason.EVIDENCE
            return False, StopReason.NO_RESULT
        if BaseAgent._completion_is_real(final_context, trace, extra_answer_keys):
            return True, StopReason.ANSWERED
        return False, StopReason.NO_RESULT

    def _build_trace(
        self,
        final_context: dict[str, Any],
        iteration: int,
    ) -> AgentTrace:
        """Build an AgentTrace from the AGENT_TRACE context entries.

        Agents that don't use tools get an empty trace (just iteration count).
        """
        trace_data = final_context.get(ContextKeys.AGENT_TRACE, [])
        trace = AgentTrace(
            tool_calls=[],
            total_iterations=final_context.get(ContextKeys.ITERATION_COUNT, iteration),
        )

        for step in trace_data:
            if isinstance(step, dict) and "action" in step:
                tool_name = step.get("action", "").split("(")[0]
                if tool_name and tool_name != ContextKeys.NO_TOOL:
                    tool_input = step.get("tool_input", {})
                    if not isinstance(tool_input, dict):
                        tool_input = {"input": tool_input}
                    trace.tool_calls.append(
                        ToolCall(
                            tool_name=tool_name,
                            parameters=tool_input,
                            reasoning=step.get("thought", ""),
                        )
                    )

        return trace

    # ------------------------------------------------------------------
    # Context filtering
    # ------------------------------------------------------------------

    # DECISION plan-2026-07-20T040150-876e7164/D-003 [STALE]
    # This filter's output feeds `AgentResult.final_context`, which is returned
    # straight to the agent's caller. Do NOT re-inline `k.startswith("_")` here:
    # that check is case-SENSITIVE and only sees the literal `_` prefix, so
    # `system_password`, `internal_token` and `__dunder` all leaked through it
    # (F-13, measured). `has_internal_prefix` is the single canonical predicate
    # over INTERNAL_KEY_PREFIXES and case-folds. See decisions.md D-003.
    @staticmethod
    def _filter_context(context: dict[str, Any]) -> dict[str, Any]:
        """Remove internal-prefixed keys from context.

        Top-level only: nested dict values are not recursed into (that is a
        separate contract, see `fsm_llm.context.clean_context_keys`).
        """
        return {k: v for k, v in context.items() if not has_internal_prefix(k)}

    # ------------------------------------------------------------------
    # API factory helper
    # ------------------------------------------------------------------

    def _create_api(self, fsm_def: dict[str, Any]) -> API:
        """Create an API instance from an FSM definition."""
        kwargs = dict(self._api_kwargs)
        if self.config.transition_config is not None:
            kwargs["transition_config"] = self.config.transition_config
        # Additive opt-in config passthroughs (defaults reproduce prior behavior).
        if (
            self.config.max_history_size is not None
            and "max_history_size" not in kwargs
        ):
            kwargs["max_history_size"] = self.config.max_history_size
        if self.config.enable_prompt_cache and "caching" not in kwargs:
            # litellm response-cache flag; no-op where the provider/cache is unset.
            kwargs["caching"] = True
        try:
            return API.from_definition(
                with_instructions(fsm_def, self.config.instructions),
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **kwargs,
            )
        except ValueError as exc:
            error = prompt_overflow_error(
                exc, self.config.instructions, getattr(self, "tools", None)
            )
            if error is None:
                raise
            raise error from exc

    # ------------------------------------------------------------------
    # Lifecycle handler registration (shared by run + run_stream)
    # ------------------------------------------------------------------

    def _register_lifecycle_handlers(self, api: API, agent_type: str) -> None:
        """Register the END_CONVERSATION, ERROR, and context-compactor handlers.

        Extracted from ``_standard_run`` so the streaming path registers the
        exact same lifecycle handlers (no behavior drift between run paths).
        """
        api.register_handler(
            api.create_handler(HandlerNames.END_CONVERSATION)
            .with_priority(HandlerPriorities.END_CONVERSATION)
            .at(HandlerTiming.END_CONVERSATION)
            .do(
                lambda ctx: {
                    "_agent_completed": True,
                    "_agent_type": agent_type,
                }
            )
        )

        def _error_handler(ctx: dict[str, Any]) -> dict[str, Any]:
            logger.warning(
                f"Agent error in {agent_type}: state={ctx.get('_current_state', '?')}"
            )
            return {}

        api.register_handler(
            api.create_handler(HandlerNames.ERROR)
            .with_priority(HandlerPriorities.ERROR)
            .at(HandlerTiming.ERROR)
            .do(_error_handler)
        )

        compactor = ContextCompactor(
            transient_keys={
                ContextKeys.TOOL_RESULT,
                ContextKeys.TOOL_STATUS,
                ContextKeys.TOOL_ERROR,
            },
        )
        api.register_handler(
            api.create_handler("AgentContextCompactor")
            .with_priority(HandlerPriorities.END_CONVERSATION)
            .at(HandlerTiming.PRE_PROCESSING)
            .do(compactor.compact)
        )

        # Opt-in observation summarization (config.auto_summarize_after).
        # No-op for agents that don't accumulate observations.
        if self.config.auto_summarize_after:
            from .summarization import make_observation_summarizer

            summarizer = make_observation_summarizer(self.config.auto_summarize_after)
            api.register_handler(
                api.create_handler("AgentObservationSummarizer")
                .with_priority(HandlerPriorities.END_CONVERSATION)
                .at(HandlerTiming.PRE_PROCESSING)
                .do(summarizer)
            )

    # ------------------------------------------------------------------
    # Streaming run() implementation
    # ------------------------------------------------------------------

    def _standard_run_stream(
        self,
        task: str,
        fsm_def: dict[str, Any],
        context: dict[str, Any],
        agent_type: str,
        max_iterations: int | None = None,
        handlers: Any = None,
    ) -> Iterator[str]:
        """Streaming variant of ``_standard_run``.

        Runs the same FSM on ``API.run_until_terminal_stream`` and streams
        each speaking state's reply token by token. States with empty
        ``response_instructions`` (think/act) yield nothing; the final answer
        state (conclude) streams its output. Yields raw text only — callers needing the structured
        ``AgentResult``/trace should use ``run()``. Errors are wrapped exactly
        like ``_standard_run``: budget and timeout errors propagate, anything
        else is raised as ``AgentError``.

        ``handlers``: see ``_standard_run``'s docstring — an optional,
        call-local handler-state object threaded straight through to
        ``_register_handlers`` (never round-tripped through ``self``).
        """
        start_time = time.monotonic()
        api = self._create_api(fsm_def)
        # DECISION plan-2026-09-12T065608-089d0ec7/D-014
        # Do NOT collapse this to an unconditional `self._register_handlers(api,
        # handlers)` — most subclasses' `_register_handlers(self, api)` override
        # takes only one argument, and only react.py/parallel_react.py's
        # override accepts (and requires) `handlers`. The abstract base
        # signature is deliberately left at `(self, api)` (LSP: subclasses
        # may only WIDEN with optional params, not the base), so this
        # 2-argument call is only type-correct for the subclasses that
        # actually declare it — hence the `type: ignore[call-arg]` below,
        # guarded by the same `handlers is not None` runtime check those
        # subclasses require. See decisions.md D-014.
        if handlers is not None:
            self._register_handlers(api, handlers)  # type: ignore[call-arg]
        else:
            self._register_handlers(api)
        self._register_lifecycle_handlers(api, agent_type)

        try:
            conv_id, greeting = self._start_conversation(api, context)
            try:
                if greeting:
                    yield greeting
                max_steps, max_seconds = self._run_budgets(start_time, max_iterations)
                yield from api.run_until_terminal_stream(
                    conv_id,
                    max_steps=max_steps,
                    max_seconds=max_seconds,
                    before_step=partial(self._on_loop_iteration, api, conv_id),
                )
            finally:
                api.end_conversation(conv_id)
        except RunBudgetExceededError as exc:
            raise self._budget_error(exc, max_iterations) from exc
        except (AgentTimeoutError, BudgetExhaustedError):
            raise
        except Exception as e:
            raise AgentError(
                f"{agent_type.title()} execution failed: {e}",
                details={"task": task},
            ) from e

    # ------------------------------------------------------------------
    # Standard run() implementation
    # ------------------------------------------------------------------

    def _standard_run(
        self,
        task: str,
        fsm_def: dict[str, Any],
        context: dict[str, Any],
        agent_type: str,
        max_iterations: int | None = None,
        extra_answer_keys: list[str] | None = None,
        execution_evidence_keys: list[str] | None = None,
        handlers: Any = None,
        judged: bool = False,
    ) -> AgentResult:
        """Standard run() implementation shared by most agents.

        Handles API creation, handler registration, conversation loop,
        answer extraction, trace building, and error wrapping.

        ``judged``: see ``_forced_stop_reason`` (EvalOpt, MakerChecker).

        ``handlers``: optional, opaque, call-local handler-state object
        (e.g. ``AgentHandlers``) created fresh by the caller's own ``run()``
        and threaded straight through to ``_register_handlers`` as an
        explicit parameter. Interface contract: when a subclass's ``run()``
        passes ``handlers``, that subclass's ``_register_handlers`` override
        MUST accept it as a second positional parameter (see
        ``react.py``/``parallel_react.py``); subclasses that never pass
        ``handlers`` (the default, ``None``) are unaffected and keep their
        original ``_register_handlers(self, api)`` signature. This exists so
        per-call handler state never has to round-trip through a ``self``
        attribute between construction and use — see decisions.md D-014 for
        why that round-trip was itself a data race (D-004's insufficient
        first fix).
        """
        start_time = time.monotonic()
        api = self._create_api(fsm_def)
        # DECISION plan-2026-09-12T065608-089d0ec7/D-014
        # Do NOT collapse this to an unconditional `self._register_handlers(api,
        # handlers)` — most subclasses' `_register_handlers(self, api)` override
        # takes only one argument, and only react.py/parallel_react.py's
        # override accepts (and requires) `handlers`. The abstract base
        # signature is deliberately left at `(self, api)` (LSP: subclasses
        # may only WIDEN with optional params, not the base), so this
        # 2-argument call is only type-correct for the subclasses that
        # actually declare it — hence the `type: ignore[call-arg]` below,
        # guarded by the same `handlers is not None` runtime check those
        # subclasses require. See decisions.md D-014.
        if handlers is not None:
            self._register_handlers(api, handlers)  # type: ignore[call-arg]
        else:
            self._register_handlers(api)
        self._register_lifecycle_handlers(api, agent_type)

        try:
            responses, final_context, iteration = self._run_conversation_loop(
                api, context, start_time, agent_type, max_iterations
            )

            answer = self._extract_answer(final_context, responses, extra_answer_keys)
            trace = self._build_trace(final_context, iteration)

            structured = self._try_parse_structured_output(answer, final_context)

            # DECISION plan_2026-05-30_26c9510a/D-001 [STALE]: a run that produced
            # neither a designated answer key (FINAL_ANSWER or a pattern-specific
            # extra_answer_key) NOR any tool call is degenerate — the `answer`
            # came from the prose-fallback in _extract_answer (a planner state's
            # Pass-2 text leaking as the result). Report success=False rather
            # than passing leaked filler off as a completed task. Mirrors
            # _extract_answer's primary/secondary sources, so any pattern that
            # concludes properly (sets an answer key) or runs a tool is unaffected.
            success, stop_reason = self._run_outcome(
                final_context,
                trace,
                extra_answer_keys,
                execution_evidence_keys,
                judged=judged,
            )
            if stop_reason in StopReason.FORCED:
                logger.warning(
                    f"Agent '{agent_type}' was forced to stop ({stop_reason}); "
                    f"shipping its last answer with success=False."
                )
            elif not success:
                if execution_evidence_keys:
                    logger.warning(
                        f"Agent '{agent_type}' completed with no execution "
                        f"evidence ({execution_evidence_keys}) — planner produced "
                        f"synthesis filler without running planned work; marking "
                        f"success=False."
                    )
                else:
                    logger.warning(
                        f"Agent '{agent_type}' completed with no answer key and no "
                        f"tool calls — answer is prose-fallback only; marking "
                        f"success=False."
                    )

            return AgentResult(
                answer=answer,
                success=success,
                trace=trace,
                # DECISION plan-2026-09-24T091842-c1d5bfbc/D-012: the redo
                # stash leaves the result here, not in _filter_context (shared
                # with adapt) and not via an internal prefix (prompts need it).
                final_context={
                    k: v
                    for k, v in self._filter_context(final_context).items()
                    if k not in RESULT_DROPPED_CONTEXT_KEYS
                },
                structured_output=structured,
                stop_reason=stop_reason,
            )

        except (AgentTimeoutError, BudgetExhaustedError):
            raise
        except Exception as e:
            raise AgentError(
                f"{agent_type.title()} execution failed: {e}",
                details={"task": task},
            ) from e

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    def _try_parse_structured_output(
        self, answer: str, context: dict[str, Any] | None = None
    ) -> Any:
        """Validate *answer* against ``config.output_schema`` if set.

        Returns a Pydantic model instance on success, ``None`` on failure
        or when no schema is configured.
        """
        # `output_schema` is declared `type | None` but `AgentConfig.validate_output_schema`
        # (definitions.py) guarantees any non-None value is a Pydantic BaseModel subclass,
        # so `.model_fields` is always present at runtime. Cast narrows for mypy (683,693).
        # Annotation-only — no runtime change.
        schema = cast("type[BaseModel] | None", self.config.output_schema)
        if schema is None:
            return None

        # 1. Try constructing from context keys (most reliable — uses Pass 1 data)
        # DECISION plan_2026-05-30_26c9510a/D-001 [STALE]: emit a diagnostic instead of
        # silently swallowing the validation error — a structured_output of None
        # otherwise gives no clue which fields were missing/invalid.
        if context:
            try:
                fields = {
                    k: context[k]
                    for k in schema.model_fields
                    if k in context and context[k] is not None
                }
                if fields:
                    return schema(**fields)
            except Exception as e:
                logger.debug(
                    f"Structured output: context-key construction of "
                    f"{schema.__name__} failed ({e}); "
                    f"present keys={sorted(fields)}, "
                    f"schema keys={sorted(schema.model_fields)}. "
                    f"Falling back to JSON parse."
                )

        # 2. Try parsing JSON from the answer string
        try:
            import json as _json

            from fsm_llm.utilities import extract_json_from_text

            data = extract_json_from_text(answer)
            if data is None:
                # Try direct JSON parse
                data = _json.loads(answer)

            if isinstance(data, dict):
                return schema(**data)

            logger.warning(
                f"Structured output: expected dict, got {type(data).__name__}"
            )
        except Exception as e:
            logger.warning(f"Structured output validation failed: {e}")

        # 3. Scan tool observations for JSON matching the schema
        if context:
            import json as _json

            from fsm_llm.utilities import extract_json_from_text

            observations = context.get("observations", [])
            if isinstance(observations, list):
                for obs in reversed(observations):
                    if not isinstance(obs, str):
                        continue
                    # Entries are formatted as "[Step n] Tool: ... | Result: ...";
                    # scan only the trailing result segment so embedded JSON is found.
                    segment = obs.rsplit("Result:", 1)[-1] if "Result:" in obs else obs
                    data = extract_json_from_text(segment)
                    if isinstance(data, dict):
                        try:
                            return schema(**data)
                        except Exception:
                            continue

        return None

    @abstractmethod
    def _register_handlers(self, api: API) -> None:
        """Register pattern-specific handlers. Implemented by each agent.

        Most overrides keep exactly this ``(self, api)`` shape. A handful
        of patterns that need call-local handler state (``react.py``,
        ``parallel_react.py`` — see D-014) widen their OWN override with an
        extra optional ``handlers`` parameter; adding an optional parameter
        to an override is LSP-compatible (every caller of the narrower base
        signature still works unchanged), so this abstract signature is
        deliberately left unwidened — see ``_standard_run``'s docstring for
        the full ``handlers=`` threading contract.
        """
        ...
