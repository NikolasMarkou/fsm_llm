"""
MakerCheckerAgent — Two-persona quality loop agent implementation.

The maker creates content and the checker critiques it. The loop
continues until the checker approves or maximum revisions are reached.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger

from .base import BaseAgent
from .constants import (
    ContextKeys,
    Defaults,
    HandlerNames,
    HandlerPriorities,
    LogMessages,
    MakerCheckerStates,
    StopReason,
)
from .definitions import AgentConfig, AgentResult
from .fsm_definitions import build_maker_checker_fsm
from .handlers import make_iteration_limiter, make_redraft_handlers


class MakerCheckerAgent(BaseAgent):
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
        super().__init__(config, **api_kwargs)
        self.maker_instructions = maker_instructions
        self.checker_instructions = checker_instructions
        self.max_revisions = max_revisions
        self.quality_threshold = quality_threshold

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Run the agent on a task.

        :param task: The task/question for the agent to solve
        :param initial_context: Optional initial context data
        :return: AgentResult with answer, trace, and metadata
        """
        # Build FSM
        fsm_def = build_maker_checker_fsm(
            maker_instructions=self.maker_instructions,
            checker_instructions=self.checker_instructions,
            task_description=task,
            max_revisions=self.max_revisions,
        )

        # Build initial context
        context = self._init_context(
            task,
            initial_context,
            extra={
                ContextKeys.REVISION_COUNT: 0,
                ContextKeys.MAX_REVISIONS: self.max_revisions,
                ContextKeys.QUALITY_THRESHOLD: self.quality_threshold,
            },
        )

        return self._standard_run(
            task,
            fsm_def,
            context,
            "maker_checker",
            extra_answer_keys=[ContextKeys.DRAFT_OUTPUT],
            # Forced only on a recorded override of the checker (D-051).
            judged=True,
        )

    def _register_handlers(self, api: API) -> None:
        """Register agent handlers with the API."""

        # DECISION plan-2026-11-05T134105-5ed71cf2/D-009
        # revision_count advances on check ENTRY, NOT in the CONTEXT_UPDATE
        # tracker below: core skips every CONTEXT_UPDATE handler on a turn
        # that committed no extracted key (a checker that returns null for the
        # score and the verdict), which left the count at 0 and the loop to
        # max_iterations (F-AM-01). The cap itself is the check -> output
        # edge's `revision_count >= max_revisions` disjunct plus the check
        # PRE_TRANSITION handler (_force_pass_at_limit), which records the
        # forced pass; PRE_TRANSITION runs after the gate, so it cannot route.
        # Do NOT move the count back into _track_revisions or key it on
        # extraction. Entry does not clear anything (D-013 below).
        def count_revision(context: dict[str, Any]) -> dict[str, Any]:
            return {
                ContextKeys.REVISION_COUNT: context.get(ContextKeys.REVISION_COUNT, 0)
                + 1
            }

        api.register_handler(
            api.create_handler(HandlerNames.MAKER_CHECKER_REVISION_COUNT)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_entry(MakerCheckerStates.CHECK)
            .do(count_revision)
        )

        # Trace + quality auto-pass: CONTEXT_UPDATE fires after the check
        # state's extraction (current_state still CHECK) and before transition
        # eval, so checker_passed/quality_score are available. Only on a turn
        # that extracted one of them. A pass written here routes this turn:
        # the transition gate reads the committed context, which holds this
        # handler's delta (plan 5ed71cf2/D-005; before it the evaluator
        # overlaid the extracted False and the pass waited one more revise
        # round, D-013).
        api.register_handler(
            api.create_handler(HandlerNames.MAKER_CHECKER_CHECKER)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .at(HandlerTiming.CONTEXT_UPDATE)
            .on_state(MakerCheckerStates.CHECK)
            .when_keys_updated(ContextKeys.CHECKER_PASSED, ContextKeys.QUALITY_SCORE)
            .do(self._track_revisions)
        )

        # DECISION plan-2026-09-24T045559-3e4eb3e5/D-013: re-judge every
        # round. Entering revise moves the draft aside and clears a False
        # verdict; leaving it clears the consumed feedback. Do NOT clear on
        # check entry (it would erase the limiter's forced pass) or clear
        # feedback on entry (revise must still see it). Do NOT clear a True
        # verdict: it is a forced pass (_track_revisions or the limiter); kept,
        # it routes check -> output. (Before plan 5ed71cf2/D-005 the forced
        # pass lost to the extracted False on its own turn, because the
        # evaluator overlaid extracted data on context; the rule stays for a
        # pass written after the gate ran, e.g. the limiter.)
        # With a True verdict do NOT move the draft either: the judged draft
        # must ship, not an unjudged redraft (step 3.2).
        # DECISION plan-2026-09-24T045559-3e4eb3e5/D-017
        redraft_entry, on_exit = make_redraft_handlers(
            ContextKeys.DRAFT_OUTPUT,
            ContextKeys.PREVIOUS_DRAFT,
            clear_on_exit=(ContextKeys.CHECKER_FEEDBACK,),
        )

        def on_entry(context: dict[str, Any]) -> dict[str, Any]:
            if context.get(ContextKeys.CHECKER_PASSED) is True:
                return {}
            delta = redraft_entry(context)
            delta.update(
                dict.fromkeys((ContextKeys.CHECKER_PASSED, ContextKeys.QUALITY_SCORE))
            )
            return delta

        api.register_handler(
            api.create_handler(HandlerNames.MAKER_CHECKER_REVISE_ENTRY)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_entry(MakerCheckerStates.REVISE)
            .do(on_entry)
        )
        api.register_handler(
            api.create_handler(HandlerNames.MAKER_CHECKER_REVISE_EXIT)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_exit(MakerCheckerStates.REVISE)
            .do(on_exit)
        )

        # Iteration limiter: counts every turn, sets only max_iterations_reached.
        self._register_iteration_limiter(api, self._make_iteration_limiter())
        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-007: the budget's forced
        # pass lands only on a check turn. Do NOT put checker_passed back in
        # the global limiter's forced dict: at an odd budget the limit hits on
        # a revise turn, the next check skips the already-set verdict (core
        # extracts a key only while it is unset) and ships the unjudged
        # redraft. Do NOT filter the counter by state either: it must advance
        # every turn. Runs after the limiter (higher priority number) so it
        # reads the flag the limiter set on this same turn.
        api.register_handler(
            api.create_handler(HandlerNames.MAKER_CHECKER_FORCE_PASS)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .at(HandlerTiming.PRE_TRANSITION)
            .on_state(MakerCheckerStates.CHECK)
            .do(self._force_pass_at_limit)
        )

    def _track_revisions(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Record the check in the trace and auto-pass on a high quality score.

        Called as a CONTEXT_UPDATE handler on the check state, after its
        extraction and before transition evaluation (current_state is still
        CHECK), so checker_passed/quality_score reflect the current turn.
        ``revision_count`` is already bumped by the check-entry handler and
        the revision cap is enforced by the check edge and
        :meth:`_force_pass_at_limit`, neither of which depends on this
        handler running (plan 5ed71cf2/D-009).
        """
        revision_count = context.get(ContextKeys.REVISION_COUNT, 0)
        max_revisions = context.get(ContextKeys.MAX_REVISIONS, self.max_revisions)
        # Typed float field (a dict-shaped checker_feedback, which an earlier
        # recovery read the score from, is now rejected by the str field).
        quality_score = context.get(ContextKeys.QUALITY_SCORE, 0.0)

        # Record in trace
        trace = list(context.get(ContextKeys.AGENT_TRACE, []))
        trace.append(
            {
                "type": "check",
                "revision_count": revision_count,
                ContextKeys.QUALITY_SCORE: quality_score,
            }
        )

        logger.info(
            LogMessages.EVALUATION.format(
                result=f"Revision {revision_count}/{max_revisions}",
                score=quality_score,
            )
        )

        result: dict[str, Any] = {ContextKeys.AGENT_TRACE: trace}

        # Auto-pass if quality score meets threshold
        quality_threshold = context.get(
            ContextKeys.QUALITY_THRESHOLD, self.quality_threshold
        )
        if (
            isinstance(quality_score, int | float)
            and quality_score >= quality_threshold
        ):
            logger.info(
                f"Quality score {quality_score:.2f} >= threshold {quality_threshold:.2f}, approving"
            )
            result[ContextKeys.CHECKER_PASSED] = True

        return result

    def _make_iteration_limiter(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Create the iteration limiter handler (shared rule, see handlers)."""
        # DECISION plan-2026-09-24T045559-3e4eb3e5/D-014: no `-1` here. The
        # forced pass is read by the check edge, so triggering one iteration
        # early skips a real check round (at max_iterations=2 the checker never
        # judges at all). Do NOT drop `early=False` without re-measuring.
        return make_iteration_limiter(
            self.config.max_iterations,
            {ContextKeys.MAX_ITERATIONS_REACHED: True},
            early=False,
        )

    @staticmethod
    def _force_pass_at_limit(context: dict[str, Any]) -> dict[str, Any]:
        """Force ``checker_passed`` once the budget or the revision cap is spent.

        Registered on the check state only (D-007); returns ``{}`` otherwise.
        The cap is ``revision_count >= _max_revisions`` (the count is bumped
        on check entry, so a check turn whose extraction was null still
        counts; plan 5ed71cf2/D-009). A pass the checker (or its quality
        score) gave on this turn is not overridden, so it records no forced
        reason (D-051 of plan 06a5ec0a).
        """
        if context.get(ContextKeys.CHECKER_PASSED) is True:
            return {}
        max_revisions = context.get(ContextKeys.MAX_REVISIONS)
        revisions_spent = (
            isinstance(max_revisions, int)
            and context.get(ContextKeys.REVISION_COUNT, 0) >= max_revisions
        )
        if context.get(ContextKeys.MAX_ITERATIONS_REACHED) is not True and not (
            revisions_spent
        ):
            return {}
        return {
            ContextKeys.CHECKER_PASSED: True,
            ContextKeys.FORCED_STOP_REASON: StopReason.FORCED_PASS,
        }
