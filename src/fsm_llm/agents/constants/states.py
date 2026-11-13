"""State names and native function-calling names of every agent pattern FSM."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Agent FSM state names
# ---------------------------------------------------------------------------
# DECISION plan-2026-09-30T062855-07ad3f8c/D-047: each class names every state
# of its pattern's FSM, and the builders (`fsm_definitions.py`,
# `parallel_react.py`), the per-state prompt maps and the handler
# registrations all read these names. Do NOT write a state id as a string
# literal in a builder or a handler registration, and do NOT drop a member
# because only the builder reads it: the same literal in a builder and in a
# handler registration is a lockstep invariant nothing checks (a rename in one
# place leaves a handler that never fires). See decisions.md D-047.


class AgentStates:
    """States in the ReAct agent FSM (also ParallelReact, ReasoningReact,
    VerifiedReact, AutoMemory) and the shared ``await_approval`` state."""

    THINK = "think"
    ACT = "act"
    AWAIT_APPROVAL = "await_approval"
    CONCLUDE = "conclude"


class ReflexionStates:
    """States in the Reflexion agent FSM."""

    # The shared loop states: the think handlers `_register_think_loop_handlers`
    # registers by `AgentStates` name must land on these, so they alias them.
    THINK = AgentStates.THINK
    ACT = AgentStates.ACT
    EVALUATE = "evaluate"
    REFLECT = "reflect"
    CONCLUDE = AgentStates.CONCLUDE
    # With HITL, the shared approval state (`_await_approval_state`).
    AWAIT_APPROVAL = AgentStates.AWAIT_APPROVAL


class PlanExecuteStates:
    """States in the Plan-and-Execute agent FSM."""

    PLAN = "plan"
    EXECUTE_STEP = "execute_step"
    CHECK_RESULT = "check_result"
    REPLAN = "replan"
    SYNTHESIZE = "synthesize"


class REWOOStates:
    """States in the REWOO agent FSM."""

    PLAN_ALL = "plan_all"
    EXECUTE_PLANS = "execute_plans"
    SOLVE = "solve"


class EvalOptStates:
    """States in the Evaluator-Optimizer agent FSM."""

    GENERATE = "generate"
    EVALUATE = "evaluate"
    REFINE = "refine"
    OUTPUT = "output"


class MakerCheckerStates:
    """States in the Maker-Checker agent FSM."""

    MAKE = "make"
    CHECK = "check"
    REVISE = "revise"
    OUTPUT = "output"


class PromptChainStates:
    """States in the Prompt Chaining agent FSM (dynamic)."""

    OUTPUT = "output"
    STEP_PREFIX = "step_"


class SelfConsistencyStates:
    """States in the Self-Consistency agent FSM."""

    GENERATE = "generate"


class OrchestratorStates:
    """States in the Orchestrator-Workers agent FSM."""

    ORCHESTRATE = "orchestrate"
    DELEGATE = "delegate"
    COLLECT = "collect"
    SYNTHESIZE = "synthesize"


class DebateStates:
    """States in the Debate agent FSM."""

    PROPOSE = "propose"
    CRITIQUE = "critique"
    COUNTER = "counter"
    JUDGE = "judge"
    CONCLUDE = "conclude"


class ADaPTStates:
    """States in the ADaPT agent FSM."""

    ATTEMPT = "attempt"
    ASSESS = "assess"
    DECOMPOSE = "decompose"
    COMBINE = "combine"


class NativeFCStates:
    """States in the native function-calling FSM (``build_native_fc_fsm``)."""

    CALL_MODEL = "call_model"
    RUN_TOOLS = "run_tools"
    FORCE_FINAL = "force_final"
    REPAIR = "repair"
    CONCLUDE = "conclude"
    # States past the model loop's deadline check: a run that reached one
    # finishes even when the wall clock is spent (core's
    # ``seconds_exempt_states``, D-032). Only ``call_model`` starts a loop
    # model turn, so the clock is checked before each one, as before.
    SECONDS_EXEMPT = frozenset({RUN_TOOLS, FORCE_FINAL, REPAIR, CONCLUDE})


class NativeFCContextKeys:
    """Context keys of the native function-calling FSM.

    ``MODEL_REPLY``, ``FORCED_REPLY`` and ``REPAIR_REPLY`` are the result keys
    of the three completion states (core treats them as handler-only). The
    three message keys are the transcripts those states send (internal: no
    prompt, extraction or ``get_data`` reaches them). Every other key is
    written only by the native handlers (``NATIVE_FC_HANDLER_ONLY_KEYS``).
    """

    # The run's transcript: the task as a user turn, then one paired
    # tool exchange per tool turn (``call_model`` sends it).
    TRANSCRIPT = "_native_messages"
    # Copies of the transcript plus one nudge user turn, sent by the
    # post-loop forced-tool and repair turns only.
    FORCE_MESSAGES = "_native_force_messages"
    REPAIR_MESSAGES = "_native_repair_messages"
    # Completion results ({kind, text, calls}).
    MODEL_REPLY = "model_reply"
    FORCED_REPLY = "forced_reply"
    REPAIR_REPLY = "repair_reply"
    # Tool turns run so far (a turn whose calls ran).
    TOOL_TURNS = "_native_tool_turns"
    # The run's answer: the final turn's text, or a repair that parsed.
    ANSWER = "native_answer"
    # How the model loop ended (a ``NativeLoopEnd`` value); unset while it runs.
    LOOP_END = "loop_end"
    # Post-loop routing, written when the loop ends.
    FORCE_PENDING = "force_final_pending"
    REPAIR_PENDING = "repair_pending"


class NativeLoopEnd:
    """Values of ``NativeFCContextKeys.LOOP_END``."""

    # A tool-call-free model turn: the run concluded on its own.
    FINAL = "final"
    # A garbled tool-call turn (provider error or non-object arguments).
    MALFORMED = "malformed"
    # ``max_iterations`` tool turns ran without a final turn.
    EXHAUSTED = "exhausted"

    ALL: tuple[str, ...] = (FINAL, MALFORMED, EXHAUSTED)


class NativeFCHandlerNames:
    """Names of the handlers ``NativeFCHandlers`` registers on its API."""

    SEED = "native_seed"
    MODEL_REPLY = "native_model_reply"
    RUN_TOOLS = "native_run_tools"
    FORCE_ENTRY = "native_force_entry"
    FORCED_REPLY = "native_forced_reply"
    REPAIR_ENTRY = "native_repair_entry"
    REPAIR_REPLY = "native_repair_reply"
    CONCLUDE = "native_conclude"


# Keys of the native FSM that only its handlers write: listed as core
# ``handler_only_keys`` beside ``FRAMEWORK_ONLY_KEYS``, so no extraction
# channel can plant a routing flag, a loop verdict or an answer. The result
# keys are handler-only in core already (D-021) and listed for the record; the
# internal-prefixed keys need no entry.
NATIVE_FC_HANDLER_ONLY_KEYS: tuple[str, ...] = (
    NativeFCContextKeys.MODEL_REPLY,
    NativeFCContextKeys.FORCED_REPLY,
    NativeFCContextKeys.REPAIR_REPLY,
    NativeFCContextKeys.ANSWER,
    NativeFCContextKeys.LOOP_END,
    NativeFCContextKeys.FORCE_PENDING,
    NativeFCContextKeys.REPAIR_PENDING,
)
