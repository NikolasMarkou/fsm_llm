"""The four classes every agent module reads: context keys, defaults, error and log message templates."""

from __future__ import annotations

from fsm_llm.constants import CONTEXT_KEY_AGENT_TRACE, DEFAULT_TEMPERATURE
from fsm_llm.llm import DEFAULT_MAX_TOKENS
from fsm_llm.policy.constants import Defaults as _PolicyDefaults
from fsm_llm.policy.constants import LogMessages as _PolicyLogMessages
from fsm_llm.tools.constants import ContextKeys as _ToolContextKeys
from fsm_llm.tools.constants import Defaults as _ToolDefaults
from fsm_llm.tools.constants import ErrorMessages as _ToolErrorMessages
from fsm_llm.tools.constants import LogMessages as _ToolLogMessages
from fsm_llm.tools.constants import MirroredConstants, mirror


class ContextKeys(metaclass=MirroredConstants):
    """Standard context keys used across the agents package."""

    # Task
    TASK = "task"

    # Tool selection (extracted by LLM in think state); the tool layer owns
    # the three tool-call keys, this class assigns them from there.
    TOOL_NAME = _ToolContextKeys.TOOL_NAME
    TOOL_INPUT = _ToolContextKeys.TOOL_INPUT
    # ParallelReact: the list of calls one think turn selects.
    TOOL_CALLS = "tool_calls"
    REASONING = "reasoning"
    SHOULD_TERMINATE = "should_terminate"

    # Sentinel value for "no tool selected"
    NO_TOOL = _ToolContextKeys.NO_TOOL

    # Tool execution results
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    TOOL_STATUS = "tool_status"

    # Observations accumulated across iterations
    OBSERVATIONS = "observations"
    OBSERVATION_COUNT = "observation_count"

    # No pattern writes or reads it and no state extracts it (D-046): the
    # answer is the speaking state's reply or a pattern answer key. Kept in
    # RUN_OUTPUT_KEYS, so a caller cannot put a ready-made answer into the
    # context the final state's prompt shows.
    FINAL_ANSWER = "final_answer"
    CONFIDENCE = "confidence"

    # Budget tracking
    ITERATION_COUNT = "iteration_count"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    # Why a run was forced to stop (a ``StopReason`` value), written only by
    # the handler that forced it (stall, forced pass); unset on a forced stop
    # means ``StopReason.MAX_ITERATIONS``.
    # DECISION plan-2026-09-29T103145-06a5ec0a/D-027: public, NOT
    # internal-prefixed. Do NOT rename it to `_forced_stop_reason`:
    # `API.get_data` drops internal keys, so the result seam never saw it and a
    # forced pass read as success=True. Only a `FORCED_STOP_REASONS` value counts
    # and it can only lower `success`; caller context never supplies it
    # (`RUN_OUTPUT_KEYS`).
    FORCED_STOP_REASON = "forced_stop_reason"
    # Why the last loop turn did nothing (an executor warning, a HITL denial),
    # for the next think turn's prompt. Not a transient key: the compactor
    # would delete it before think reads it (LOOP-06, D-021 of plan 06a5ec0a).
    AGENT_FEEDBACK = "agent_feedback"
    # Gated calls a human approver refused in this run and that did not run: a
    # list of sentences (tool, redacted parameters, "was refused ... and was
    # not performed", `handlers.refusal_record`), appended only by the HITL
    # driver and removed only when the same call is approved later and runs
    # (D-045). Kept to the end of the run and read by the conclude prompt.
    # DECISION plan-2026-09-30T062855-07ad3f8c/D-034: do NOT report a denial to conclude through
    # `agent_feedback` (cleared on think exit, so the answer then claimed the
    # refused action happened) and do NOT record it as an observation (it
    # would count as tool evidence, LOOP-06). Each entry states a final fact;
    # do NOT word it as a skipped or pending item, the model reads those as
    # work to retry (06a5ec0a/D-049). See decisions.md D-034.
    REFUSED_ACTIONS = "refused_actions"
    # The state a transition leaves, written by core on every transition
    # (absent before the first one). Read by the ReAct-family limiter.
    CURRENT_STATE = "_current_state"

    # Internal run keys: the executor seeds them with its own limits and a
    # handler reads them back (internal prefix, so they never reach a prompt or
    # `final_context`). One spelling here, not a literal per pattern.
    ITERATION_BUDGET = "_max_iterations"
    REPLAN_COUNT = "_replan_count"
    MAX_REFINEMENTS = "_max_refinements"
    MAX_REVISIONS = "_max_revisions"
    QUALITY_THRESHOLD = "_quality_threshold"
    MAX_ROUNDS = "_max_rounds"

    # HITL
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_GRANTED = "approval_granted"
    # Driver-only grant: the exact approved call. Internal prefix, so core drops
    # it from every model extraction; only the approval driver writes it.
    DRIVER_APPROVAL = "_approval_granted"
    # Count of driver grants the executor spent, written by its delta. A value
    # behind the executor's own count means a spending delta was discarded.
    APPROVALS_SPENT = "_approvals_spent"

    # Agent trace
    AGENT_TRACE = CONTEXT_KEY_AGENT_TRACE

    # Reflexion
    EVALUATION_PASSED = "evaluation_passed"
    EVALUATION_SCORE = "evaluation_score"
    EVALUATION_FEEDBACK = "evaluation_feedback"
    EPISODIC_MEMORY = "episodic_memory"
    REFLECTION_COUNT = "reflection_count"
    REFLECTION = "reflection"
    LESSONS = "lessons"

    # Plan-and-Execute
    PLAN_STEPS = "plan_steps"
    CURRENT_STEP_INDEX = "current_step_index"
    STEP_RESULTS = "step_results"
    ALL_STEPS_COMPLETE = "all_steps_complete"
    STEP_FAILED = "step_failed"
    STEP_RESULT = "step_result"
    CURRENT_STEP_DESCRIPTION = "current_step_description"
    # The plan a replan revises (stash, see RESULT_DROPPED_CONTEXT_KEYS).
    PREVIOUS_PLAN_STEPS = "previous_plan_steps"

    # REWOO
    EVIDENCE = "evidence"
    # One entry per executed plan step: {"id", "tool_name", "success"}. The
    # success rule reads it (at least one True), not the evidence mapping.
    EVIDENCE_STATUS = "evidence_status"
    PLAN_BLUEPRINT = "plan_blueprint"
    # True once the plan extraction was cut off at max_tokens, so no tool ran
    # (written only by REWOO's plan executor; a run output).
    PLAN_CUT_OFF = "plan_cut_off"

    # Evaluator-Optimizer
    GENERATED_OUTPUT = "generated_output"
    EVALUATION_RESULT = "evaluation_result"
    REFINEMENT_FEEDBACK = "refinement_feedback"
    REFINEMENT_COUNT = "refinement_count"
    PREVIOUS_OUTPUT = "previous_output"

    # Maker-Checker
    QUALITY_SCORE = "quality_score"
    DRAFT_OUTPUT = "draft_output"
    CHECKER_FEEDBACK = "checker_feedback"
    CHECKER_PASSED = "checker_passed"
    REVISION_COUNT = "revision_count"
    PREVIOUS_DRAFT = "previous_draft"

    # Prompt Chaining
    CHAIN_STEP_INDEX = "chain_step_index"
    CHAIN_STEP_RESULT = "chain_step_result"
    CHAIN_RESULTS = "chain_results"
    GATE_PASSED = "gate_passed"

    # Self-Consistency
    SAMPLES = "samples"
    AGGREGATED_ANSWER = "aggregated_answer"

    # Orchestrator-Workers
    SUBTASKS = "subtasks"
    WORKER_RESULTS = "worker_results"
    # Subtasks over max_workers that never ran; kept out of worker_results
    # (D-049 of plan 06a5ec0a).
    SKIPPED_SUBTASKS = "skipped_subtasks"
    ALL_COLLECTED = "all_collected"

    # Debate
    PROPOSITION = "proposition"
    CRITIQUE = "critique"
    COUNTER_ARGUMENT = "counter_argument"
    JUDGE_VERDICT = "judge_verdict"
    DEBATE_ROUNDS = "debate_rounds"
    CURRENT_ROUND = "current_round"
    CONSENSUS_REACHED = "consensus_reached"

    # ADaPT
    ATTEMPT_RESULT = "attempt_result"
    ATTEMPT_SUCCEEDED = "attempt_succeeded"
    SUBTASK_RESULTS = "subtask_results"
    # How a decomposition's subtasks combine: "AND" (all) or "OR" (any).
    OPERATOR = "operator"
    CURRENT_DEPTH = "current_depth"


class Defaults(metaclass=MirroredConstants):
    """Default configuration values."""

    TEMPERATURE = DEFAULT_TEMPERATURE
    MAX_TOKENS = DEFAULT_MAX_TOKENS
    MAX_ITERATIONS = 10
    # Largest integer an agent limit (``max_*``/``num_*``) or an artifact int
    # config field may take: 63 bits (defined by the tool layer).
    MAX_INT_LIMIT = _ToolDefaults.MAX_INT_LIMIT
    TIMEOUT_SECONDS = 300.0
    MCP_TIMEOUT_SECONDS = 30.0
    REMOTE_TOOL_TIMEOUT_SECONDS = 120.0  # RemoteAgentTool's HTTP call
    MAX_OBSERVATION_LENGTH = _ToolDefaults.MAX_OBSERVATION_LENGTH
    # Largest per-call tool ``timeout_seconds``: the most a thread wait accepts
    # (``threading.TIMEOUT_MAX``; a larger wait raises OverflowError).
    MAX_TOOL_TIMEOUT_S = _ToolDefaults.MAX_TOOL_TIMEOUT_S
    MAX_OBSERVATIONS = 20
    # Escalate below this confidence (defined by the policy layer).
    CONFIDENCE_THRESHOLD = _PolicyDefaults.CONFIDENCE_THRESHOLD
    # Tool-selection classification threshold (``to_classification_schema``).
    TOOL_CONFIDENCE_THRESHOLD = _ToolDefaults.TOOL_CONFIDENCE_THRESHOLD
    MAX_TASK_PREVIEW_LENGTH = 200
    # Cap on AgentConfig.instructions. They are prefixed to prompt slots that
    # core caps at 5000 chars; a ReAct think slot grows with the tool list, so
    # long instructions plus a large registry can still fail the FSM load.
    MAX_INSTRUCTIONS_LENGTH = 2000

    # Multiplier for computing hard iteration ceiling from max_iterations.
    # Each agent cycle uses multiple FSM transitions; this factor provides
    # headroom so the FSM can finish its current cycle before the budget
    # check fires.
    FSM_BUDGET_MULTIPLIER = 3
    # Loop turns a ReAct-family run may still need after a forced stop to
    # reach its conclude state (Reflexion: think -> act -> evaluate ->
    # conclude). ``AgentHandlers.check_iteration_limit`` forces the stop this
    # many transitions before the hard ceiling (capped at ``max_iterations``).
    FORCED_STOP_MARGIN = 3
    # Native function calling (``build_native_fc_fsm``): core steps per model
    # loop turn (``call_model`` then ``run_tools``) and the steps allowed on
    # top of them (the final model turn, the forced-tool turn, the repair
    # turn, one spare). Step ceiling = per turn x max_iterations + post-loop.
    NATIVE_FC_STEPS_PER_TURN = 2
    NATIVE_FC_POST_LOOP_STEPS = 4

    # Reflexion
    MAX_REFLECTIONS = 3

    # Plan-and-Execute
    MAX_PLAN_STEPS = 10
    MAX_REPLANS = 2

    # Evaluator-Optimizer
    MAX_REFINEMENTS = 3

    # Maker-Checker
    MAX_REVISIONS = 3
    QUALITY_THRESHOLD = 0.7

    # Self-Consistency
    NUM_SAMPLES = 5
    SAMPLE_TEMPERATURE_RANGE = (0.5, 1.0)

    # Orchestrator
    MAX_WORKERS = 5

    # Debate
    MAX_DEBATE_ROUNDS = 3
    # Heuristic budget factor for sizing the debate FSM iteration cap
    # (num_rounds * FSM_BUDGET_MULTIPLIER * DEBATE_STATES_PER_ROUND). NOTE: a
    # debate round actually traverses 4 states (propose/critique/counter/judge);
    # this 2 is intentionally an under-count that the FSM_BUDGET_MULTIPLIER (3)
    # headroom compensates for at typical round counts. Do not read it as the
    # literal states-per-round (AP3-004).
    DEBATE_STATES_PER_ROUND = 2

    # ADaPT
    MAX_DECOMPOSITION_DEPTH = 3
    # Subtasks run per decomposition; extra ones are dropped with a WARNING.
    ADAPT_MAX_SUBTASKS = 8

    # AgentServer: agent runs in flight at once; one more request gets 503.
    SERVER_MAX_CONCURRENT = 8


class ErrorMessages(metaclass=MirroredConstants):
    """Standard error messages."""

    TOOL_NOT_FOUND = _ToolErrorMessages.TOOL_NOT_FOUND
    TOOL_EXECUTION_FAILED = _ToolErrorMessages.TOOL_EXECUTION_FAILED
    TOOL_TIMED_OUT = _ToolErrorMessages.TOOL_TIMED_OUT
    TOOL_BAD_TIMEOUT = _ToolErrorMessages.TOOL_BAD_TIMEOUT
    TOOL_NOT_LOGGED = _ToolErrorMessages.TOOL_NOT_LOGGED
    EXECUTE_WITHOUT_GATED = (
        "{registry}.execute does not accept the keyword argument 'gated': a "
        "ToolRegistry subclass must keep the signature "
        "execute(tool_call, *, gated=False) (agents pass gated=True for a call "
        "an approver granted, so it is never re-run)"
    )
    PROMPT_CACHE_WITH_INTERFACE = (
        "AgentConfig.enable_prompt_cache=True cannot apply to an injected "
        "llm_interface: response caching is a setting (litellm caching=True) "
        "of the interface core builds, and an injected interface owns its own "
        "settings. Enable caching on the interface you inject, or drop "
        "llm_interface"
    )
    # The answer when no speaking state replied (base and PromptChain).
    NO_ANSWER = "Agent could not determine an answer."
    # REWOO's answer when its plan was cut off at max_tokens: no tool ran.
    REWOO_PLAN_CUT_OFF = (
        "The plan was cut off at max_tokens before it was complete, so no tool "
        "ran. Raise AgentConfig.max_tokens (or LLM_MAX_TOKENS) or give the "
        "agent a smaller task."
    )
    EMPTY_CHAIN = "Cannot create prompt chain agent with empty chain"
    PROMPT_SLOT_OVERFLOW = (
        "Agent prompt too long: {slot} exceeds core's {limit}-character "
        "instruction limit. It holds AgentConfig.instructions ({instructions} "
        "characters) and, in tool-using patterns, the tool catalogue ({tools} "
        "tools). Shorten AgentConfig.instructions (create_agent system_prompt=), "
        "shorten tool descriptions or register fewer tools."
    )


class LogMessages(metaclass=MirroredConstants):
    """Standard log message templates."""

    AGENT_STARTED = "Agent started with {tool_count} tools, model={model}"
    TOOL_SELECTED = "Selected tool: {name} with input: {input}"
    TOOL_EXECUTED = "Tool '{name}' executed successfully"
    TOOL_FAILED = "Tool '{name}' failed: {error}"
    TOOL_LATE_RESULT = _ToolLogMessages.TOOL_LATE_RESULT
    ITERATION = "Iteration {current}/{max}"
    AGENT_COMPLETE = "Agent completed in {iterations} iterations"
    APPROVAL_REQUESTED = _PolicyLogMessages.APPROVAL_REQUESTED
    APPROVAL_RESULT = _PolicyLogMessages.APPROVAL_RESULT
    ESCALATION = _PolicyLogMessages.ESCALATION
    REFLECTION = "Reflection {current}/{max}: {summary}"
    PLAN_STEP = "Executing plan step {current}/{total}: {description}"
    EVALUATION = "Evaluation {result}: score={score}"
    DEBATE_ROUND = "Debate round {current}/{max}"
    DECOMPOSITION = "Decomposing task at depth {depth}"


# The attributes each class above assigns from a lower layer are linked to it: an
# override of either class reaches every reader of both (D-045; the registry is pinned
# against the assignments above by ``test_constants_mirror.py``).
mirror(_ToolContextKeys, ContextKeys, ("TOOL_NAME", "TOOL_INPUT", "NO_TOOL"))
mirror(
    _ToolDefaults,
    Defaults,
    (
        "MAX_INT_LIMIT",
        "MAX_OBSERVATION_LENGTH",
        "MAX_TOOL_TIMEOUT_S",
        "TOOL_CONFIDENCE_THRESHOLD",
    ),
)
mirror(_PolicyDefaults, Defaults, ("CONFIDENCE_THRESHOLD",))
mirror(
    _ToolErrorMessages,
    ErrorMessages,
    (
        "TOOL_NOT_FOUND",
        "TOOL_EXECUTION_FAILED",
        "TOOL_TIMED_OUT",
        "TOOL_BAD_TIMEOUT",
        "TOOL_NOT_LOGGED",
    ),
)
mirror(_ToolLogMessages, LogMessages, ("TOOL_LATE_RESULT",))
mirror(
    _PolicyLogMessages,
    LogMessages,
    ("APPROVAL_REQUESTED", "APPROVAL_RESULT", "ESCALATION"),
)
