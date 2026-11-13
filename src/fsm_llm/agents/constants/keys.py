"""Key sets and constructor-kwarg sets the run seam, the builders and ``BaseAgent`` read."""

from __future__ import annotations

from .core import ContextKeys

# DECISION plan-2026-09-24T091842-c1d5bfbc/D-012: the redo stash (set by
# ``make_redraft_handlers`` on entry to a redo state) is run-time prompt input,
# not a result. BaseAgent drops these keys from ``AgentResult.final_context``
# only. Do NOT give them an internal prefix instead: internal keys are also
# filtered out of prompts, which hides the previous draft from the checker and
# refiner that need it. Add any new ``previous_*`` stash key here.
RESULT_DROPPED_CONTEXT_KEYS: frozenset[str] = frozenset(
    {
        ContextKeys.PREVIOUS_DRAFT,
        ContextKeys.PREVIOUS_OUTPUT,
        ContextKeys.PREVIOUS_PLAN_STEPS,
    }
)


# Loop values a ReAct think turn produces (react, reflexion, reasoning_react,
# parallel_react): cleared on think entry so each turn extracts them afresh.
# The executors clear the tool selection themselves.
REACT_THINK_FRESH_KEYS: tuple[str, ...] = (
    ContextKeys.REASONING,
    ContextKeys.SHOULD_TERMINATE,
)


# DECISION plan-2026-09-29T103145-06a5ec0a/D-002: keys a run writes for itself
# (tool selection and results, evidence counters, termination, approval state and
# the driver-only grant). Caller context never supplies them; they are removed
# only by ``base.strip_caller_context`` (see the anchor there). Do NOT add the
# whole internal-prefix family here: ``_sensitive``-style policy inputs
# legitimately arrive through ``initial_context``.
RUN_OUTPUT_KEYS: frozenset[str] = frozenset(
    {
        ContextKeys.OBSERVATION_COUNT,
        ContextKeys.OBSERVATIONS,
        ContextKeys.SHOULD_TERMINATE,
        ContextKeys.FINAL_ANSWER,
        ContextKeys.TOOL_NAME,
        ContextKeys.TOOL_INPUT,
        ContextKeys.TOOL_RESULT,
        ContextKeys.TOOL_STATUS,
        ContextKeys.TOOL_ERROR,
        ContextKeys.APPROVAL_REQUIRED,
        ContextKeys.APPROVAL_GRANTED,
        ContextKeys.REASONING,
        ContextKeys.DRIVER_APPROVAL,
        ContextKeys.APPROVALS_SPENT,
        ContextKeys.FORCED_STOP_REASON,
        ContextKeys.AGENT_FEEDBACK,
        ContextKeys.REFUSED_ACTIONS,
    }
)


# DECISION plan-2026-09-29T103145-06a5ec0a/D-051: keys only framework handlers
# write (limiters, stall detector, forcing handlers, tool executors). Every
# agent FSM lists them in core ``handler_only_keys``, so no extraction channel
# (bulk, per-field, post-transition) can plant them: a model writing
# ``max_iterations_reached`` or ``forced_stop_reason`` flipped ``success``,
# and ``observation_count`` is the conclude evidence guard. Do NOT drop a key
# here to let a state extract it; internal-prefixed keys
# (``_approvals_spent``, ``_approval_granted``) need no entry, core never
# extracts them.
FRAMEWORK_ONLY_KEYS: tuple[str, ...] = (
    ContextKeys.MAX_ITERATIONS_REACHED,
    ContextKeys.FORCED_STOP_REASON,
    ContextKeys.ITERATION_COUNT,
    ContextKeys.OBSERVATION_COUNT,
    ContextKeys.REFUSED_ACTIONS,
)


# DECISION plan-2026-09-29T103145-06a5ec0a/D-003: constructor kwargs that reach
# ``BaseAgent.__init__(**api_kwargs)`` only by mistake. The agent-constructor
# names mean the pattern cannot use them (HITL or tools silently ignored and
# forwarded to litellm); the config-owned names collide with the explicit
# ``model``/``temperature``/``max_tokens`` that ``_create_api`` passes. Do NOT
# turn this into a whitelist of allowed kwargs: litellm passthrough (``seed``,
# ``timeout``, ``caching``, ``api_base``, ...) and API kwargs (``handlers``,
# ``llm_interface``, ...) are open-ended and must keep flowing.
MISPLACED_AGENT_KWARGS: frozenset[str] = frozenset(
    {"hitl", "tools", "evaluation_fn", "approval_callback"}
)


CONFIG_OWNED_KWARGS: frozenset[str] = frozenset({"model", "temperature", "max_tokens"})


class ReasoningIntegrationKeys:
    """Context keys for reasoning-agent integration (namespaced to avoid collision)."""

    REASONING_RESULT = "reasoning_integration_result"
    REASONING_TYPE_USED = "reasoning_integration_type_used"
    REASONING_CONFIDENCE = "reasoning_integration_confidence"
    REASONING_TOOL_NAME = "reason"
