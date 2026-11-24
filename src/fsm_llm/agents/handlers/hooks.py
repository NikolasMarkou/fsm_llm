"""The run hooks of the handler library: verify-and-retry, periodic reflection, memory recall and
remember (split from ``library.py`` by family, plan a555dcfe step 21, D-032).

A hook is a library entry of the ``HANDLERS`` namespace bound under ``run.hooks`` of a pattern file;
``PatternAgent`` calls it at its position. Interface contract (the shape by ``at``):

- ``before_run``: ``factory(ctx, binding) -> Callable[[str], str]``, built once per ``run`` from a
  new context; the call takes the task and returns the task the attempts run on.
- ``after_run``: ``factory(ctx, binding) -> Callable[[], Retry | None]``, built per attempt from the
  attempt's context (``ctx.scope`` holds ``RUN_TASK``, ``RUN_ATTEMPT`` and, when the loop returned,
  ``RUN_RESULT``, which a hook may replace); ``None`` accepts, a ``Retry`` rejects the attempt.
- ``on_iteration``: ``factory(ctx, binding) -> Callable[[int, Callable[[], dict], Callable[[dict],
  None]], None]``, built at every step from a new context (a hook keeps nothing between steps) and
  called AFTER the HITL driver of that step with the step number, a reader of the conversation data
  and a writer that refuses an internal-prefixed key: a hook has no other way to write, so none
  can write the approval grant.
- Failure: a hook error propagates (the hooks after it do not run). Options are read from
  ``ctx.options`` with the defaults of the constructors the presets replaced.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from fsm_llm.agents.constants import ContextKeys, StopReason
from fsm_llm.core.codes import status_of_stop_reason
from fsm_llm.core.definitions import AgentError
from fsm_llm.logging import logger

from .context import RUN_ATTEMPT, RUN_RESULT, RUN_TASK, HandlerContext, Retry
from .memory import augment_task_with_memories, remember_interaction

if TYPE_CHECKING:
    from fsm_llm.core.definitions.executable_pattern import HookBinding

#: Option defaults of the hooks: the defaults of the constructors the presets replaced.
DEFAULT_MAX_VERIFY_RETRIES = 1
DEFAULT_RECALL_K = 3
DEFAULT_RECALL_MIN_SCORE = 0.25

_REFLECTION_NOTE = (
    "[Reflection] Pause and assess: are recent actions making progress toward "
    "the task? If you are repeating a failed action or looping, try a different "
    "tool or conclude with your best answer."
)


def _verdict(fn: Callable[..., Any], result: Any) -> tuple[bool, str]:
    """``(ok, feedback)`` of the caller's verifier over a result."""
    # DECISION plan-2026-09-29T103145-06a5ec0a/D-011: a raising verifier
    # is a rejection (fail closed). Do NOT treat it as a pass: a verifier
    # that crashes on a malformed answer would ship exactly that answer as
    # verified.
    try:
        verdict = fn(result.answer, result.final_context)
    except Exception as e:
        logger.warning(f"verification_fn raised, treating as rejection: {e}")
        return False, f"the verifier raised an error: {e}"
    if isinstance(verdict, dict):
        ok = bool(verdict.get("ok", verdict.get("passed", False)))
        return ok, str(verdict.get("feedback", ""))
    return bool(verdict), ""


def verify_retry(
    ctx: HandlerContext, binding: HookBinding | None = None
) -> Callable[[], Retry | None]:
    """``agents.verify_retry`` (``at: after_run``): check the answer with ``config.verification_fn``.

    A rejected attempt is retried with the verifier's feedback folded into the task, up to the
    option ``max_verify_retries`` extra attempts; the last answer still rejected ships with
    ``stop_reason="verification_failed"``. With no verifier it accepts.
    """
    fn = getattr(ctx.config, "verification_fn", None)
    limit = ctx.options.get("max_verify_retries", DEFAULT_MAX_VERIFY_RETRIES)

    def check() -> Retry | None:
        result = ctx.scope.get(RUN_RESULT)
        if fn is None or result is None:
            return None
        ok, feedback = _verdict(fn, result)
        if ok:
            return None
        attempt = ctx.scope[RUN_ATTEMPT]
        logger.info(
            f"Verification rejected attempt {attempt + 1}/{limit + 1}: {feedback}"
        )
        if attempt < limit:
            return Retry(
                f"[A previous answer was rejected by verification: {feedback}\n"
                "Produce a corrected answer that addresses this.]"
            )
        # REACT-04: the last answer still ships, but it is not verified.
        # ``model_copy`` does not validate, so the derived pair is set here.
        status, reason = status_of_stop_reason(StopReason.VERIFICATION_FAILED)
        ctx.scope[RUN_RESULT] = result.model_copy(
            update={
                "status": status,
                "reason": reason,
                "stop_reason": StopReason.VERIFICATION_FAILED,
            }
        )
        return None

    return check


def periodic_reflection(
    ctx: HandlerContext, binding: HookBinding | None = None
) -> Callable[
    [int, Callable[[], dict[str, Any]], Callable[[dict[str, Any]], None]], None
]:
    """``agents.periodic_reflection`` (``at: on_iteration``): every ``config.reflect_every_n``
    steps a note goes to ``agent_feedback`` for the next think turn (none when it is unset)."""
    n = getattr(ctx.config, "reflect_every_n", None)

    def reflect(
        iteration: int,
        read: Callable[[], dict[str, Any]],
        write: Callable[[dict[str, Any]], None],
    ) -> None:
        if not (n and iteration > 0 and iteration % n == 0):
            return
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-046: the note goes to
        # `agent_feedback`. Do NOT append it to `observations`: the executor
        # sets `observation_count` from that list, and the conclude edge's
        # evidence guard (c1d5bfbc D-008) must count real tool results only.
        # Any pending executor feedback is kept ahead of the note.
        try:
            pending = read().get(ContextKeys.AGENT_FEEDBACK)
            note = _REFLECTION_NOTE
            if isinstance(pending, str) and pending.strip():
                note = f"{pending}\n{_REFLECTION_NOTE}"
            write({ContextKeys.AGENT_FEEDBACK: note})
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"Reflection injection skipped: {e}")

    return reflect


def _memory(ctx: HandlerContext, name: str) -> Any:
    memory = ctx.callables.get("memory")
    if memory is None:
        raise AgentError(f"{name} needs the host object 'memory' and the run has none")
    return memory


def memory_recall(
    ctx: HandlerContext, binding: HookBinding | None = None
) -> Callable[[str], str]:
    """``agents.memory_recall`` (``at: before_run``): prepend the memories relevant to the task
    (options ``recall_k``, ``recall_min_score``; the task is unchanged when nothing is found)."""
    memory = _memory(ctx, "agents.memory_recall")
    k = ctx.options.get("recall_k", DEFAULT_RECALL_K)
    floor = ctx.options.get("recall_min_score", DEFAULT_RECALL_MIN_SCORE)
    return lambda task: augment_task_with_memories(task, memory, k, min_score=floor)


def memory_remember(
    ctx: HandlerContext, binding: HookBinding | None = None
) -> Callable[[], None]:
    """``agents.memory_remember`` (``at: after_run``): persist the interaction (options
    ``auto_remember``, ``remember_only_on_success``); a run that raised (no result) persists the
    raw task, so a stated fact survives a budget stop."""
    memory = _memory(ctx, "agents.memory_remember")
    options: Mapping[str, Any] = ctx.options

    def remember() -> None:
        if not options.get("auto_remember", True):
            return
        task, result = ctx.scope[RUN_TASK], ctx.scope.get(RUN_RESULT)
        if result is None:
            if task and task.strip():
                try:
                    memory.add(task)
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning(f"Auto-remember (on failure) failed: {e}")
        elif result.success or not options.get("remember_only_on_success", False):
            remember_interaction(memory, task, result.answer)

    return remember
