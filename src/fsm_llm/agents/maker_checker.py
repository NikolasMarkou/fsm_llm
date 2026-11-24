"""
MakerCheckerAgent: two-persona quality loop agent implementation.

The maker creates content and the checker critiques it. The loop
continues until the checker approves or maximum revisions are reached.

The machine is the pattern file ``patterns/maker_checker.json`` run by ``PatternAgent`` (plan
a555dcfe step 16); its handlers are the library factories ``agents.revision_count``,
``agents.maker_checker_checker``, ``agents.revise_entry``, ``agents.revise_exit``,
``agents.iteration_limiter`` and ``agents.force_pass``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from fsm_llm.agents.constants import ContextKeys, Defaults
from fsm_llm.agents.fsm_definitions.maker_checker import maker_checker_texts
from fsm_llm.tools import require_int_limit

from .definitions import AgentConfig
from .handlers.library import iteration_check
from .handlers.maker_checker import force_pass_at_limit, track_revisions
from .pattern_agent import PatternAgent


class MakerCheckerAgent(PatternAgent):
    """
    Agent that uses a maker-checker pattern for quality assurance.

    The maker persona generates or revises content, and the checker
    persona evaluates it against quality criteria. The loop continues
    until the checker approves or max revisions are reached.

    Trust: ``maker_instructions`` and ``checker_instructions`` are
    developer-authored text interpolated unsanitized into the generated FSM's
    prompts, like an FSM's own ``persona``. Do not pass raw end-user input;
    put user-supplied content in the ``task`` instead.

    Usage::

        agent = MakerCheckerAgent(
            maker_instructions="Write a professional email apologizing for a delay",
            checker_instructions=(
                "Check for: professional tone, clear apology, "
                "concrete next steps, appropriate length"
            ),
            max_revisions=3,
            quality_threshold=0.7,
        )
        result = agent.run("Write an apology email to a client about a project delay")
        print(result.answer)
    """

    # No `tools` argument: `create_agent` raises for tools given to this pattern (`accepts_tools`).
    takes_tools: ClassVar[bool] = False

    # Run outputs caller context may not seed (D-052 of plan 06a5ec0a).
    _run_output_keys = frozenset(
        {
            ContextKeys.DRAFT_OUTPUT,
            ContextKeys.CHECKER_FEEDBACK,
            ContextKeys.CHECKER_PASSED,
            ContextKeys.QUALITY_SCORE,
            ContextKeys.REVISION_COUNT,
            ContextKeys.PREVIOUS_DRAFT,
        }
    )

    def __init__(
        self,
        maker_instructions: str,
        checker_instructions: str,
        config: AgentConfig | None = None,
        max_revisions: int = Defaults.MAX_REVISIONS,
        quality_threshold: float = Defaults.QUALITY_THRESHOLD,
        **api_kwargs: Any,
    ) -> None:
        """
        Initialize a Maker-Checker agent.

        :param maker_instructions: What the maker should produce
        :param checker_instructions: What the checker should evaluate
        :param config: Agent configuration (defaults to AgentConfig())
        :param max_revisions: Maximum number of revision cycles
        :param quality_threshold: Minimum quality score to pass (0.0-1.0)
        :param api_kwargs: Additional kwargs passed to fsm_llm.API
        """
        max_revisions = require_int_limit("max_revisions", max_revisions, minimum=0)
        from .pattern_files import load_envelope  # lazy: reads the file

        super().__init__(
            load_envelope("maker_checker"),
            config=config,
            options={
                "cap": True,
                "max_revisions": max_revisions,
                "quality_threshold": quality_threshold,
            },
            **api_kwargs,
        )
        self.maker_instructions = maker_instructions
        self.checker_instructions = checker_instructions
        self.max_revisions = max_revisions
        self.quality_threshold = quality_threshold

    def _text_variables(self) -> dict[str, str]:
        """The field prompts built around the two instruction texts (read at run time)."""
        return maker_checker_texts(self.maker_instructions, self.checker_instructions)

    def _context_seed(self) -> dict[str, Any]:
        """The file's seed (``revision_count``) and the cap and threshold the handlers read."""
        return {
            **super()._context_seed(),
            ContextKeys.MAX_REVISIONS: self.max_revisions,
            ContextKeys.QUALITY_THRESHOLD: self.quality_threshold,
        }

    def _track_revisions(self, context: dict[str, Any]) -> dict[str, Any]:
        """The ``agents.maker_checker_checker`` body for this agent's cap and threshold."""
        return track_revisions(context, self.max_revisions, self.quality_threshold)

    _force_pass_at_limit = staticmethod(force_pass_at_limit)

    def _make_iteration_limiter(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """The closure ``agents.iteration_limiter`` registers for this agent (the file's binding)."""
        # DECISION plan-2026-09-24T045559-3e4eb3e5/D-014: no `-1` here. The
        # forced pass is read by the check edge, so triggering one iteration
        # early skips a real check round (at max_iterations=2 the checker never
        # judges at all). Do NOT drop `early: false` from the binding of
        # patterns/maker_checker.json without re-measuring.
        binding = next(
            b
            for b in self.envelope.body.handlers
            if b.handler == "agents.iteration_limiter"
        )
        return iteration_check(self._handler_context(), binding)
