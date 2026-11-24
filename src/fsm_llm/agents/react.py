"""
ReactAgent — ReAct (Reasoning + Acting) agent implementation.

Uses FSM-LLM's 2-pass architecture to implement the ReAct loop:
Think -> Act -> Observe -> Think -> ... -> Conclude

The loop is the pattern file ``patterns/react.json`` run by ``PatternAgent`` (plan a555dcfe step 9):
this class keeps the constructor, the ``run`` and ``run_stream`` names the callers and the tests
patch, and the call of ``build_react_fsm`` that builds the FSM of a run.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from fsm_llm.agents.constants import Defaults
from fsm_llm.agents.fsm_definitions.react import build_react_fsm
from fsm_llm.policy import HumanInTheLoop
from fsm_llm.tools import ToolRegistry

from .base import caller_prompt_keys
from .definitions import AgentConfig, AgentResult
from .handlers.context import HandlerContext
from .pattern_agent import PatternAgent


class ReactAgent(PatternAgent):
    """
    ReAct agent that uses tools to solve tasks step by step.

    Combines FSM-LLM's core (states, transitions, handlers) with
    a tool registry and optional HITL support to implement the
    ReAct pattern.

    Usage::

        registry = ToolRegistry()
        registry.register_function(search, name="search", description="Search the web")
        registry.register_function(calculate, name="calculate", description="Do math")

        agent = ReactAgent(tools=registry)
        result = agent.run("What is the population of France times 2?")
        print(result.answer)
    """

    tools: ToolRegistry  # a ReAct agent always has tools (``PatternAgent.tools`` may be None)

    def __init__(
        self,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        hitl: HumanInTheLoop | None = None,
        use_classification: bool = False,
        **api_kwargs: Any,
    ) -> None:
        from fsm_llm.agents.pattern_files import load_envelope  # lazy: reads the file

        super().__init__(
            load_envelope("react"),
            tools=tools,
            config=config,
            hitl=hitl,
            options={"classification": bool(use_classification)},
            **api_kwargs,
        )
        self.use_classification = use_classification

    def _react_inputs(
        self, task: str, initial_context: dict[str, Any] | None
    ) -> tuple[dict[str, Any], dict[str, Any], HandlerContext]:
        """The FSM, initial context and call-local handlers of one run (the name callers keep)."""
        return self._run_inputs(task, initial_context)

    def _fsm_for_run(
        self, task: str, initial_context: dict[str, Any] | None
    ) -> dict[str, Any]:
        """The FSM of one run, through ``build_react_fsm`` (the builder every caller shares).

        The ``await_approval`` state is built under the SAME predicate that registers the runtime
        approval gate (``BaseAgent._hitl_active``, D-001 of plan c1d5bfbc).
        """
        return build_react_fsm(
            self.tools,
            task_description=task[: Defaults.MAX_TASK_PREVIEW_LENGTH],
            include_approval_state=self._hitl_active,
            use_classification=self.use_classification,
            output_schema=self.config.output_schema,
            context_keys=caller_prompt_keys(initial_context),
        )

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        # DECISION plan-2026-09-12T065608-089d0ec7/D-014 (supersedes D-004)
        # A call-LOCAL handler context (`PatternAgent._run_inputs`), threaded explicitly into
        # `_register_handlers` via `_standard_run`'s `handlers=` parameter. D-004's fix
        # (`self._handlers = AgentHandlers(...)` assigned here, read back later by
        # `_register_handlers`) did NOT close the race: two concurrent run() calls on the SAME
        # ReactAgent instance could each reassign `self._handlers` before either call's
        # `_register_handlers` read it back, so whichever assignment landed LAST would win for
        # BOTH calls. `handlers` is a plain local variable: it is never written to `self`, so no
        # other thread's `run()` call can overwrite the reference this call is about to use. Do
        # NOT reintroduce `self._handlers = ...` in this method; see decisions.md D-014.
        # D-052 refusal and the shared predicate: `_run_inputs`.
        return super().run(task, initial_context)

    def run_stream(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Run the ReAct loop, streaming the final answer token by token.

        Yields response text as it is generated. Intermediate think/act turns
        (empty ``response_instructions``) yield nothing; the ``conclude`` state
        streams the final answer. For the structured ``AgentResult`` (trace,
        tools used, structured_output), use :meth:`run` instead.

        Example::

            for token in agent.run_stream("What is 2+2?"):
                print(token, end="", flush=True)
        """
        # See D-014 note in run() above: call-local handlers, never assigned to `self`.
        yield from super().run_stream(task, initial_context)

    def _stream_via_run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Stream by running :meth:`run` and yielding its final answer once.

        For subclasses whose ``run()`` wraps the loop (verification retries,
        memory recall/persist): streaming the inner loop would skip that
        wrapping, so their ``run_stream`` delegates here. Lazy like
        ``run_stream``: nothing runs until the first ``next()``. Yields exactly
        one chunk (the answer, possibly empty); errors propagate from ``run()``.
        """
        yield self.run(task, initial_context).answer
