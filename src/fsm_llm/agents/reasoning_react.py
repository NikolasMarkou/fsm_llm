"""
ReasoningReactAgent — ReAct agent with integrated structured reasoning.

Extends the ReAct pattern with a pseudo-tool ``reason`` that invokes
FSM-LLM's reasoning engine via FSM stacking (push_fsm / pop_fsm).
The agent autonomously decides when to use structured reasoning versus
regular tools.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm import API
from fsm_llm.logging import logger

from .base import BaseAgent, caller_prompt_keys
from .constants import (
    REACT_THINK_FRESH_KEYS,
    AgentStates,
    ContextKeys,
    LogMessages,
    ReasoningIntegrationKeys,
)
from .definitions import (
    AgentConfig,
    AgentResult,
    AgentStep,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from .exceptions import AgentError
from .fsm_definitions import build_react_fsm
from .handlers import AgentHandlers, forced_stop_skip, next_step_number
from .hitl import HumanInTheLoop
from .tools import ToolRegistry, redact_secret_entries, refuse_execute_without_gated

# Optional import — reasoning package may not be installed
try:
    from fsm_llm.reasoning import ReasoningEngine

    _HAS_REASONING = True
except ImportError:
    _HAS_REASONING = False


def _problem_text(tool_input: Any, task: Any = "") -> str:
    """The problem text of a ``reason`` call.

    A non-blank string input, else a non-blank ``problem`` entry, else the
    ``str()`` of a dict holding any non-blank value. An input with nothing in
    it (``None``, ``""``, ``{}``, ``{"problem": ""}``) falls back to ``task``,
    so the engine never reasons about the literal ``"{}"`` (REACT-05).
    """
    if isinstance(tool_input, dict):
        problem = tool_input.get("problem")
        if isinstance(problem, str) and problem.strip():
            return problem
        if any(str(v).strip() for v in tool_input.values() if v is not None):
            return str(tool_input)
    elif tool_input is not None and str(tool_input).strip():
        return str(tool_input)
    return str(task or "")


class _ReasonToolRegistry(ToolRegistry):
    """The caller's tools plus the ``reason`` pseudo-tool, without mutating them.

    Interface contract: built from the caller's registry ``base``; a LIVE view
    of ``base`` (a tool registered on ``base`` later is listed, flagged and
    executed with its current definition) plus the tools registered on this
    object (the ``reason`` pseudo-tool). A name ``base`` holds always resolves
    to ``base``. ``execute`` of such a name goes to ``base.execute``, so a
    ``CachingToolRegistry``/``RetryingToolRegistry`` keeps its behaviour
    (REACT-05: a plain copy dropped it); any other name runs here. Semantic
    prompt filtering (``SemanticToolRegistry.retrieve``) is not carried over.
    """

    # DECISION plan-2026-09-29T103145-06a5ec0a/D-052: a live view, not a
    # snapshot. The snapshot kept a tool's construction-time flag while
    # `execute` ran the caller's current definition, so a tool re-registered
    # on the caller's registry with requires_approval=True ran unasked, and
    # the run() refusal could not see a flag added after construction. Do NOT
    # copy `base`'s tools into this registry again.
    def __init__(self, base: ToolRegistry) -> None:
        # `execute` passes `gated` on to *base* (D-033).
        refuse_execute_without_gated(base)
        super().__init__()
        self._base = base

    def _own_tools(self) -> list[ToolDefinition]:
        return [t for t in super().list_tools() if t.name not in self._base]

    def list_tools(self) -> list[ToolDefinition]:
        return self._base.list_tools() + self._own_tools()

    def get(self, name: str) -> ToolDefinition:
        if name in self._base:
            return self._base.get(name)
        return super().get(name)

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.list_tools()]

    def __len__(self) -> int:
        return len(self.list_tools())

    def __contains__(self, name: str) -> bool:
        return name in self._base or super().__contains__(name)

    def execute(self, tool_call: ToolCall, *, gated: bool = False) -> ToolResult:
        """Run a caller tool on the caller's registry, else the ``reason`` tool.

        ``gated`` reaches whichever registry runs the call.
        """
        if tool_call.tool_name in self._base:
            return self._base.execute(tool_call, gated=gated)
        return super().execute(tool_call, gated=gated)


class ReasoningReactAgent(BaseAgent):
    """
    ReAct agent with integrated structured reasoning via FSM stacking.

    Auto-registers a ``reason`` pseudo-tool in the tool registry.
    When the LLM selects ``reason``, the agent pushes a reasoning FSM
    onto the stack (via ``ReasoningEngine``), executes it, and pops
    results back into the agent context under namespaced keys.

    Requires ``fsm_llm.reasoning`` to be installed. Raises
    ``AgentError`` at construction time if the package is missing.

    Usage::

        from fsm_llm.agents import ReasoningReactAgent, ToolRegistry

        registry = ToolRegistry()
        registry.register_function(search, name="search", description="Search")
        agent = ReasoningReactAgent(tools=registry)
        result = agent.run("Analyze: is 97 prime?")
    """

    def __init__(
        self,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        hitl: HumanInTheLoop | None = None,
        reasoning_model: str | None = None,
        **api_kwargs: Any,
    ) -> None:
        """
        Initialize a ReasoningReactAgent.

        :param tools: Tool registry (``reason`` is auto-registered)
        :param config: Agent configuration
        :param hitl: Optional HITL manager
        :param reasoning_model: Model for the reasoning engine (defaults to config.model)
        :param api_kwargs: Additional kwargs passed to fsm_llm.API
        """
        if not _HAS_REASONING:
            raise AgentError(
                "ReasoningReactAgent requires fsm_llm.reasoning. "
                "Reinstall from a clone of https://github.com/NikolasMarkou/fsm_llm: pip install -e ."
            )

        super().__init__(config, **api_kwargs)
        self.hitl = hitl

        reason_name = ReasoningIntegrationKeys.REASONING_TOOL_NAME
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-046: do NOT copy the
        # caller's tools into a plain ToolRegistry (drops a Caching/Retrying
        # subclass's execute) and do NOT intercept a user's own `reason` tool.
        # REACT-05: a user tool named `reason` wins. The agent keeps it, runs
        # it like any tool, and registers no reasoning tool.
        self._reasoning_tool_enabled = reason_name not in tools
        if not self._reasoning_tool_enabled:
            logger.warning(
                f"ReasoningReactAgent: the registry already has a tool named "
                f"'{reason_name}'; keeping it, structured reasoning is off"
            )
            self.tools = tools
        else:
            # A snapshot plus `reason`, so the caller's registry is not mutated.
            self.tools = _ReasonToolRegistry(tools)
            self.tools.register_function(
                self._reason_placeholder,
                name=reason_name,
                description=(
                    "Use structured reasoning to analyze a complex problem. "
                    "Provide the problem statement as 'problem' parameter. "
                    "Use this when the task requires deep analytical, "
                    "deductive, or critical thinking."
                ),
                parameter_schema={
                    "properties": {
                        "problem": {
                            "type": "string",
                            "description": "The problem to reason about",
                        },
                    }
                },
            )

        if len(self.tools) == 0:
            raise AgentError("Cannot create agent with empty tool registry")
        self._refuse_unapprovable_flagged_tools()

        # Create reasoning engine
        reasoning_model_name = reasoning_model or self.config.model
        self._reasoning_engine = ReasoningEngine(
            model=reasoning_model_name, **api_kwargs
        )
        # DECISION plan-2026-09-12T135914-45a654de/D-012
        # No `self._handlers` here (matches react.py's D-014 pattern) — a
        # per-instance AgentHandlers shared across concurrent run() calls is
        # the D-004/D-014 race. Each call builds and uses its own call-LOCAL
        # AgentHandlers (see run() below), threaded explicitly into
        # `_make_reasoning_tool_executor` and `_register_handlers`. Do NOT
        # reintroduce this assignment — see decisions.md D-012.

        logger.info(
            LogMessages.AGENT_STARTED.format(
                tool_count=len(self.tools), model=self.config.model
            )
        )

    @staticmethod
    def _reason_placeholder(params: dict) -> str:
        """Placeholder — actual reasoning is intercepted in the run loop."""
        return "Reasoning executed via FSM stacking."

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Run the agent on a task.

        When the LLM selects the ``reason`` tool, the POST_TRANSITION
        handler intercepts the call and delegates to
        ``ReasoningEngine.solve_problem()`` instead of the placeholder.
        Results are stored under ``ReasoningIntegrationKeys`` in context.

        :param task: The task/question for the agent
        :param initial_context: Optional initial context data
        :return: AgentResult with answer, trace, and metadata
        """
        # DECISION plan-2026-09-12T135914-45a654de/D-012
        # A call-LOCAL AgentHandlers, built from `self.tools` (the copy with
        # the `reason` pseudo-tool auto-registered — NOT the caller's
        # original `tools` param), threaded explicitly into
        # `_register_handlers` via `_standard_run`'s `handlers=` parameter.
        # A fresh instance needs no `.reset()`. Do NOT reintroduce
        # `self._handlers = AgentHandlers(...)` — see decisions.md D-012.
        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-005
        # One predicate (BaseAgent._hitl_active) for the await_approval state,
        # the gate in _register_handlers and the D-004 refusal. Do NOT AND the
        # state with "some tool has requires_approval": the policy may gate an
        # unflagged tool (or `reason`), and a gate without the state makes every
        # gated call a refused act turn instead of an ask.
        # D-052: a registry can gain a flagged tool after construction.
        self._refuse_unapprovable_flagged_tools()
        handlers = AgentHandlers(self.tools, requires_approval=self._approval_predicate)

        fsm_def = build_react_fsm(
            self.tools,
            task_description=task,
            include_approval_state=self._hitl_active,
            context_keys=caller_prompt_keys(initial_context),
        )

        # Build initial context
        context = self._init_context(
            task,
            initial_context,
            extra={
                ContextKeys.OBSERVATIONS: [],
                "_max_iterations": self.config.max_iterations,
            },
        )

        return self._standard_run(
            task, fsm_def, context, "reasoning_react", handlers=handlers
        )

    def _make_reasoning_tool_executor(
        self, handlers: AgentHandlers
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Create tool executor that intercepts 'reason' and invokes the reasoning engine.

        For non-reason tools, delegates to the standard AgentHandlers.execute_tool.
        For the 'reason' tool, runs ReasoningEngine.solve_problem() and returns
        results under namespaced keys — all inside the POST_TRANSITION handler
        so the FSM pipeline processes them at the correct time.
        """
        base_handler = handlers
        reason_name = ReasoningIntegrationKeys.REASONING_TOOL_NAME
        engine = self._reasoning_engine
        intercept = self._reasoning_tool_enabled

        def execute_tool_with_reasoning(context: dict[str, Any]) -> dict[str, Any]:
            tool_name = context.get(ContextKeys.TOOL_NAME)

            # Non-reason tools, and a user's own `reason` tool: standard handler
            if tool_name != reason_name or not intercept:
                return base_handler.execute_tool(context)

            # plan-2026-09-24T091842-c1d5bfbc/D-004: the same refusal and
            # one-call grant consumption as execute_tool; `reason` is a tool
            # the HITL policy may gate too. The grant is spent before the
            # engine runs (plan-2026-09-29T103145-06a5ec0a/D-015).
            forced = forced_stop_skip(context)
            if forced is not None:  # LOOP-05: no tool after the forced stop
                return base_handler.consume_approval(context, forced)
            refusal = base_handler.approval_refusal(context)
            if refusal is not None:
                return refusal
            spent = base_handler.spend_grant(context)
            return base_handler.consume_approval(
                context, {**spent, **run_reason(context)}
            )

        def run_reason(context: dict[str, Any]) -> dict[str, Any]:
            # Extract problem from tool input
            tool_input = context.get(ContextKeys.TOOL_INPUT) or {}
            task = context.get(ContextKeys.TASK, "")
            problem = _problem_text(tool_input, task)
            # plan-2026-09-29T103145-06a5ec0a/D-016: the engine gets `problem`;
            # the log, observation and action get the redacted form.
            shown = _problem_text(redact_secret_entries(tool_input), task)[:100]

            logger.info(f"ReasoningReactAgent: invoking reasoning for: {shown}")

            try:
                solution, trace_info = engine.solve_problem(problem)

                reasoning_type = trace_info.get("reasoning_trace", {}).get(
                    "reasoning_types_used", ["unknown"]
                )
                confidence = trace_info.get("reasoning_trace", {}).get(
                    "final_confidence", 0.0
                )

                observation = (
                    f"Structured reasoning result (type={reasoning_type}): {solution}"
                )

                # Accumulate in observations
                observations = context.get(ContextKeys.OBSERVATIONS, [])
                if not isinstance(observations, list):
                    observations = []
                trace = context.get(ContextKeys.AGENT_TRACE, [])
                if not isinstance(trace, list):
                    trace = []
                step_num = next_step_number(trace)
                observation_entry = (
                    f"[Step {step_num}] Tool: reason | "
                    f"Input: {shown} | "
                    f"Result: {observation}"
                )
                observations.append(observation_entry)

                # Track in agent trace
                trace.append(
                    AgentStep(
                        iteration=step_num,
                        thought=context.get(ContextKeys.REASONING, ""),
                        action=f"reason({shown})",
                        observation=observation,
                    ).model_dump(mode="json")
                )

                return {
                    ContextKeys.TOOL_RESULT: observation,
                    ContextKeys.TOOL_STATUS: "success",
                    ContextKeys.TOOL_ERROR: None,
                    ContextKeys.OBSERVATIONS: observations,
                    ContextKeys.OBSERVATION_COUNT: len(observations),
                    ContextKeys.AGENT_TRACE: trace,
                    ContextKeys.TOOL_NAME: None,
                    ContextKeys.TOOL_INPUT: None,
                    ContextKeys.SHOULD_TERMINATE: None,
                    # Namespaced reasoning results
                    ReasoningIntegrationKeys.REASONING_RESULT: solution,
                    ReasoningIntegrationKeys.REASONING_TYPE_USED: str(reasoning_type),
                    ReasoningIntegrationKeys.REASONING_CONFIDENCE: confidence,
                }

            except Exception as e:
                logger.warning(f"Reasoning failed, recording error: {e}", exc_info=True)
                return {
                    ContextKeys.TOOL_RESULT: f"Reasoning failed: {e}",
                    ContextKeys.TOOL_STATUS: "failed",
                    ContextKeys.TOOL_ERROR: str(e),
                    ContextKeys.TOOL_NAME: None,
                    ContextKeys.TOOL_INPUT: None,
                }

        return execute_tool_with_reasoning

    # DECISION plan-2026-09-12T135914-45a654de/D-012
    # `handlers` is REQUIRED in practice: run() always passes its own
    # call-local `AgentHandlers` explicitly via `_standard_run`'s `handlers=`
    # parameter (mirrors react.py's D-014 guard) — this class's tools are
    # never optional. Do NOT fall back to reading a `self._handlers`
    # attribute here — none exists on this class. See decisions.md D-012.
    def _register_handlers(
        self, api: API, handlers: AgentHandlers | None = None
    ) -> None:
        """Register agent handlers with the API."""
        if handlers is None:
            raise AgentError(
                "ReasoningReactAgent._register_handlers called without a "
                "handlers instance — this is a programming error, not a "
                "runtime condition; run() must always pass one."
            )
        self._register_tool_executor(
            api, AgentStates.ACT, self._make_reasoning_tool_executor(handlers)
        )

        self._register_iteration_limiter(api, handlers.check_iteration_limit)
        self._register_think_loop_handlers(api, REACT_THINK_FRESH_KEYS)

        self._register_approval_gate(api)

    def _on_loop_iteration(self, api: API, conv_id: str, iteration: int) -> None:
        """Ask the HITL callback before each step (same driver as React)."""
        self._handle_hitl_approval(api, conv_id)
