"""Module-level helpers of ``BaseAgent`` (the old module ``fsm_llm.agents.base``, plan d454fd62 step 87).

The functions the agent patterns, ``create_agent`` and the meta-builder import by name: the
``AgentConfig.instructions`` prefixing and its overflow error, the one ``response_format`` builder,
the caller-context strip (the driver-only approval grant and the run outputs a caller may not set),
the flagged-tool policy (the reader ``flagged_tool_names`` lives in ``agents/tools.py``, which the handler factories read too) and the misplaced-keyword refusal. Bodies are the old ones. (``artifact_text`` moved
to ``handlers/recording.py``: the debate judge handler reads it and ``handlers`` imports no ``base`` module.)
"""

from __future__ import annotations

import inspect
from collections.abc import Collection, Mapping, Set
from typing import Any

from pydantic import ValidationError

from fsm_llm.agents.constants import (
    CONFIG_OWNED_KWARGS,
    MISPLACED_AGENT_KWARGS,
    RUN_OUTPUT_KEYS,
    ContextKeys,
    ErrorMessages,
)
from fsm_llm.agents.tools import flagged_tool_names
from fsm_llm.constants import has_internal_prefix
from fsm_llm.core.definitions import AgentError
from fsm_llm.logging import logger
from fsm_llm.policy import ApprovalPolicy, HumanInTheLoop
from fsm_llm.tools import ToolCall


def _with_instructions(
    fsm_def: dict[str, Any], instructions: str | None
) -> dict[str, Any]:
    """Return *fsm_def* with ``AgentConfig.instructions`` in every non-empty
    state and per-field instruction.

    Not reached: ``classification_extractions`` (the ``use_classification``
    think path), core's AMBIGUOUS transition classifier, ReasoningReact's
    reasoning-engine call, and any empty slot (see below).

    Interface contract (one call site, :meth:`BaseAgent._create_api`, which
    every pattern's ``API`` goes through):
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


def _prompt_overflow_error(
    exc: ValueError, instructions: str | None, tools: Any
) -> AgentError | None:
    """Translate core's FSM-load length error into an actionable ``AgentError``.

    Interface contract (one call site, :meth:`BaseAgent._create_api`, the only
    place that loads an agent FSM):
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

    Interface contract (one call site, ``_init_context`` here; native_fc's
    repair turn reads the envelope it writes from context):

    Args:
        schema: ``AgentConfig.output_schema`` — a Pydantic ``BaseModel``
            subclass, ``None``, or any object (duck-typed on
            ``model_json_schema``).

    Returns:
        The ``{"type": "json_schema", ...}`` dict litellm/OpenAI accept as
        ``response_format``, or ``None`` when *schema* is ``None`` or does not
        expose ``model_json_schema``.  Never raises.
    """
    # DECISION plan-2026-07-21T191807-bf7ffe24/D-002 (call sites restated by
    # plan-2026-10-01T093600-944e2692/D-033)
    # This helper is the ONE builder of the `{"type": "json_schema", ...}`
    # envelope in this package. It has ONE call site, `_init_context` below,
    # which writes the envelope under CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT; the
    # native_fc repair turn (a core completion state, D-009 of plan 944e2692)
    # reads it from there (`response_format_key`). The alternative -- native_fc
    # building its own envelope -- is a second builder of the same provider
    # contract, kept in lockstep by hand, which is the drift `hardening.py`'s
    # D-059 block already records this repo paying for once.
    # Do NOT add a second caller by reflex: a call site that can set
    # `AgentConfig.output_schema` and go through `BaseAgent` gets this for free
    # and should. Do NOT make it raise on a bad schema either -- the readers
    # treat `None` as "no constrained decoding available", and an exception here
    # would turn a missing capability into a failed run.
    # See decisions.md D-002 (plan bf7ffe24), D-033 (plan 944e2692).
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
    # reaches the approval policy by design (TestDriverSeesRefusalContext).
    # Only AgentServer sets `drop_internal`.
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


def reject_misplaced_kwargs(pattern: str, api_kwargs: Mapping[str, Any]) -> None:
    """Raise ``TypeError`` when *api_kwargs* holds a name that only lands there by mistake.

    Called by ``BaseAgent.__init__`` and by the meta-builder agent. *pattern*
    is the concrete class name, used in the message. Every other kwarg is left
    alone (D-003).
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
