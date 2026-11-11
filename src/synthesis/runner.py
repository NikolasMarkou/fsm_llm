"""
The runner: obtains a procedure for a ``synthesize`` state, pushes it as a child
conversation on the host's (cheap) model, runs it, pops it and merges only the
return keys back.

Core only loads the ``synthesize`` field; this runner gives it meaning, the way
``reasoning``'s ``_StrategyStack`` gives meaning to its strategy FSMs: a
``before_step`` hook of core ``run_until_terminal`` (``run``), or the same
settling around ``API.converse`` (``converse``). No second engine, no new core
seam.

# DECISION plan-2026-11-10T194139-2cefeb61/D-029
# Return keys go back through ``pop_fsm(context_to_return=...)`` and ONLY when
# every key is present and the child was not aborted. Do NOT pass
# ``shared_context_keys`` at push (D-017): a failed or aborted child would then
# merge partial data into the parent. Read the child's data BEFORE the pop
# (after it the top of the stack is the parent). The child's steps are counted
# by the runner's own ``_Run.child_steps``. Do NOT derive them from core's step
# number: it restarts at 1 on every ``run_until_terminal`` call, so a host that
# resumes after ``RunBudgetExceededError`` would never reach ``max_turns``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from fsm_llm.api import API, ContextMergeStrategy
from fsm_llm.definitions import (
    AdvanceResult,
    FSMContext,
    FSMDefinition,
    FSMError,
    LLMResponseError,
    SynthesizeConfig,
    logic_referenced_keys,
)
from fsm_llm.expressions import is_missing
from fsm_llm.logging import logger
from fsm_llm.transition_evaluator import TransitionEvaluator

from .constants import (
    ABORT_CONTEXT_KEY,
    ABORT_STATE_ID,
    LIFECYCLE_TRUSTED,
    LIFECYCLE_VERIFIED,
    REASON_BUDGET_SPENT,
    REASON_COMPILE_FAILED,
    REASON_COMPLETION_UNSUPPORTED,
    REASON_DEPTH_REFUSED,
    REASON_INPUTS_MISSING,
    REASON_LIBRARY_ERROR,
    REASON_LIFECYCLE_TOO_LOW,
    REASON_PROVIDER_ERROR,
    REASON_PUSH_FAILED,
    REASON_RETURN_KEYS_MISSING,
    REASON_UNEXPECTED_ERROR,
    REASONS,
    STATUS_ABORTED,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_RUNNING,
    TRACE_KIND_OTHER,
    TRACE_KINDS,
)
from .exceptions import LibraryError, SynthesisError, SynthesisRefusedError
from .gates import verify_stored
from .policy import SynthesisPolicy
from .synthesizer import FSMSynthesizer, Procedure

_ALLOWED_LIFECYCLES = {
    LIFECYCLE_VERIFIED: (LIFECYCLE_VERIFIED, LIFECYCLE_TRUSTED),
    LIFECYCLE_TRUSTED: (LIFECYCLE_TRUSTED,),
}


@dataclass
class _Run:
    """One procedure pushed by the runner. ``child_steps`` is the one count of the
    steps it has had, kept by the runner and never read off core: a run-mode hook
    call that finds the child on top and not ended counts the step before it (the
    call that pushed it counts none), ``converse`` counts each user turn.
    ``aborting`` once the abort is set."""

    procedure: Procedure
    config: SynthesizeConfig
    child_steps: int = 0
    aborting: bool = False


def _result(
    status: str, reason: str | None = None, procedure: Procedure | None = None
) -> dict[str, Any]:
    """The value of the result key: codes and ids only, never exception text."""
    record = procedure.record if procedure else None
    return {
        "status": status,
        "procedure_id": procedure.digest if procedure else None,
        "version": record.version if record else None,
        "reason": reason,
    }


def reason_of(error: Exception) -> str:
    """The fixed reason for an ``obtain`` failure; the message is never read."""
    if isinstance(error, SynthesisRefusedError) and error.code in REASONS:
        return error.code
    if isinstance(error, LibraryError):
        return REASON_LIBRARY_ERROR
    if isinstance(error, SynthesisError) and isinstance(
        error.__cause__, LLMResponseError
    ):
        return REASON_PROVIDER_ERROR
    return REASON_COMPILE_FAILED


def _round_codes(error: BaseException | None) -> str | None:
    """The failure codes of each compile round of a ``SynthesisError`` (codes
    comma-joined, rounds semicolon-joined, cut at 200 characters), else ``None``."""
    if not isinstance(error, SynthesisError):
        return None
    return _round_string(error.details.get("rounds") or ())


def _round_string(rounds: Sequence[Sequence[Any]]) -> str:
    return ";".join(",".join(map(str, codes)) for codes in rounds)[:200]


def runnable_reason(
    definition: FSMDefinition,
    lifecycle: str,
    *,
    min_lifecycle: str,
    policy: SynthesisPolicy,
    inputs: Iterable[str],
    return_keys: Iterable[str],
) -> str | None:
    """The reason code a stored procedure may not run, else ``None``: a
    ``completion`` state, a lifecycle below ``min_lifecycle`` (``verified`` or
    ``trusted``) or a failed ``verify_stored``, checked in that order. Shared by
    the runner and the agent tools; never raises."""
    if any(s.completion for s in definition.states.values()):
        return REASON_COMPLETION_UNSUPPORTED
    if lifecycle not in _ALLOWED_LIFECYCLES[min_lifecycle]:
        return REASON_LIFECYCLE_TOO_LOW
    report = verify_stored(definition, policy, inputs=inputs, return_keys=return_keys)
    return None if report.passed else REASON_LIBRARY_ERROR


class ProcedureRunner:
    """Runs the ``synthesize`` states of a host FSM on one ``API``.

    One instance serves one thread at a time per conversation, like
    ``API.converse`` (which raises on a concurrent turn): the runner holds no
    lock. Two threads on one conversation can see the other's ``running`` result
    and spend their step budget on it; a lock around the hook would not make that
    safe, since core's own turn guard still refuses the second driver.

    Interface contract:
        ``ProcedureRunner(api, definition, synthesizer, *, trace_sink=None)``:
        ``definition`` is the host FSM (a dict or ``FSMDefinition``; the root of
        every conversation the runner serves), ``synthesizer`` obtains the
        procedures, ``trace_sink(event, fields)`` an optional plain callable
        (``TypeError`` if not callable). Events: ``obtained`` (digest, version,
        source, rounds: failed compile rounds, seconds, calls and token counts
        spent: D-049), ``pushed`` (digest, step), ``stepped`` (digest, step: the
        child's count, state, keys, kinds: one per counted child step),
        ``aborted`` (digest, step, state, outcome: the runner set the abort;
        ``blocked``, ``ambiguous`` or ``deterministic`` in the child's state),
        ``popped`` (digest, status, reason, step: every pop) and ``failed``
        (reason, step, digest when a procedure was obtained, error: the
        exception class, rounds: compile failure codes; no child was pushed).
        Every event carries ``conv``. Fields are plain strings, ids and ints,
        never a value, user text or reply (D-044): ``keys`` and ``kinds`` are
        comma-joined (declared names present; ``name:type`` for gate-read ones),
        ``rounds`` is codes comma-joined, rounds semicolon-joined. A sink that
        raises is swallowed with a WARNING naming the event.

        ``run(conv_id, *, max_steps, max_seconds=None)`` is
        ``run_until_terminal`` with ``hook(conv_id)``; ``hook(conv_id)`` is the
        ``before_step`` callable for a host that drives core itself.
        ``converse(conv_id, message)`` settles around ``API.converse`` and
        returns the replies of the turn, newline-joined. ``end_conversation``
        ends the conversation and drops the runner's entry for it.

        On a ``synthesize`` state whose result key is unset the runner writes
        ``{status: "running", ...}``, then ``obtain``s and pushes the procedure
        (``inherit_context=False``, only the declared inputs, the host persona).
        When the child ends it is popped and the final
        ``{status, procedure_id, version, reason}`` is written: ``ok`` (every
        return key merged), ``aborted`` (the child spent ``policy.max_turns``
        steps in total, across resumed ``run`` calls; nothing merged) or ``failed`` with a reason from ``constants``
        (nothing merged). The result key is skip-if-set: to run a state again on
        re-entry clear its result key on entry (``clear_keys_on_entry``). What
        ``obtain`` raises other than ``SynthesisError`` and ``ValueError`` (or
        any other ``Exception`` between the ``running`` write and the push) ends
        in ``failed`` / ``unexpected_error`` with one WARNING naming the
        exception type only, and the stack as it was; never ``running``.
        ``KeyboardInterrupt`` and ``SystemExit`` in that window write the same
        final ``failed`` result (a host that catches Ctrl-C and keeps the
        conversation never sees ``running``, and nothing obtains again), unwind
        the stack, and then propagate unchanged; so does anything a core call
        outside that window raises.

    Bookkeeping: ``_runs`` maps a conversation id to the runs it has pushed,
    outermost first, and holds ``len(runs) == stack_depth - 1``; a longer list
    (a child popped from outside) is trimmed on the next call, and a stack that
    holds a frame the runner did not push is left alone. ``end_conversation``
    drops the entry; an end straight on the ``API`` leaves a stale one.
    """

    def __init__(
        self,
        api: API,
        definition: FSMDefinition | dict[str, Any],
        synthesizer: FSMSynthesizer,
        *,
        trace_sink: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        if trace_sink is not None and not callable(trace_sink):
            raise TypeError("trace_sink must be callable")
        self.api = api
        self.synthesizer = synthesizer
        self._definition = api.process_fsm_definition(definition)[0]
        self._sink = trace_sink
        self._runs: dict[str, list[_Run]] = {}

    # --- driving ----------------------------------------------------------

    def hook(self, conv_id: str) -> Callable[[int], None]:
        """The ``before_step`` callable of ``conv_id`` for ``run_until_terminal``."""

        def before_step(step: int) -> None:
            self._count_child_step(conv_id)
            self._tick(conv_id, step)

        return before_step

    def run(
        self, conv_id: str, *, max_steps: int, max_seconds: float | None = None
    ) -> tuple[AdvanceResult, ...]:
        """``run_until_terminal`` with this runner's hook (its budgets and errors)."""
        return self.api.run_until_terminal(
            conv_id,
            max_steps=max_steps,
            max_seconds=max_seconds,
            before_step=self.hook(conv_id),
        )

    def converse(self, conv_id: str, message: str) -> str:
        """One user turn: settle, ``API.converse``, settle again."""
        replies = self._settle(conv_id)
        runs = self._runs_of(conv_id)
        replies.append(self.api.converse(message, conv_id))
        if runs:
            self._count(conv_id, runs[-1])  # a user turn is one step of the child
        replies += self._settle(conv_id)
        return "\n".join(r for r in replies if r)

    def end_conversation(self, conv_id: str) -> None:
        """End ``conv_id`` (every frame) and drop the runner's entry for it."""
        self.api.end_conversation(conv_id)
        self._runs.pop(conv_id, None)

    # --- one tick: the whole of the hook's work ---------------------------

    def _runs_of(self, conv_id: str) -> list[_Run] | None:
        """The runs of ``conv_id`` trimmed to the stack; ``None`` when the stack
        holds a frame this runner did not push."""
        depth = self.api.get_stack_depth(conv_id)
        runs = self._runs.setdefault(conv_id, [])
        del runs[depth - 1 :]
        return runs if len(runs) == depth - 1 else None

    def _count_child_step(self, conv_id: str) -> None:
        """Run mode: a hook call with a live child on top means it just had a step."""
        runs = self._runs_of(conv_id)
        if runs and not self.api.has_conversation_ended(conv_id):
            self._count(conv_id, runs[-1])

    def _count(self, conv_id: str, run: _Run) -> None:
        """Count one child step and trace it."""
        run.child_steps += 1
        self._emit(
            "stepped",
            conv_id,
            digest=run.procedure.digest,
            step=run.child_steps,
            **self._view(conv_id, run),
        )

    def _view(
        self, conv_id: str, run: _Run, *, outcome: bool = False
    ) -> dict[str, Any]:
        """Value-free facts about the child on top for a trace event: its
        ``state``, the declared names present (``keys``), ``name:type`` for those
        an edge gate reads (``kinds``, types from ``TRACE_KINDS``) and, with
        ``outcome``, what the evaluator makes of its edges now. ``{}`` without a
        sink, and on any failure (one WARNING): a trace never changes a run."""
        if self._sink is None:
            return {}
        try:
            definition, config = run.procedure.definition, run.config
            data = self.api.get_data(conv_id)
            gated = {
                key
                for s in definition.states.values()
                for t in s.transitions or []
                for c in t.conditions or []
                for key in (
                    *(c.requires_context_keys or []),
                    *logic_referenced_keys(c.logic),
                )
            }
            seen = sorted(
                k
                for k in gated | {*config.inputs, *config.return_keys}
                if not is_missing(data, k)
            )
            kinds = (
                f"{k}:{TRACE_KINDS.get(type(data[k]), TRACE_KIND_OTHER)}"
                for k in seen
                if k in gated
            )
            state = self.api.get_current_state(conv_id)
            view = {"state": state, "keys": ",".join(seen), "kinds": ",".join(kinds)}
            if outcome:
                found = TransitionEvaluator().evaluate_transitions(
                    definition.states[state], FSMContext(data=data)
                )
                view["outcome"] = found.result_type.value
            return view
        except Exception as e:
            logger.warning(f"could not read the child for a trace: {type(e).__name__}")
            return {}

    def _tick(self, conv_id: str, step: int | None) -> list[str]:
        """Pop an ended child, abort one over budget, else start a pending
        synthesize state. ``step`` is core's next step number (run mode) or
        ``None`` (converse mode); it only labels trace events."""
        runs = self._runs_of(conv_id)
        if runs is None:
            return []
        if runs:
            run = runs[-1]
            if self.api.has_conversation_ended(conv_id):
                self._finish(conv_id, run, step)
                return []
            if run.aborting:
                return []
            if run.child_steps >= self.synthesizer.policy.max_turns:
                run.aborting = True
                # DECISION plan-2026-11-10T194139-2cefeb61/D-044
                # Read the outcome BEFORE the abort key is written: after it the
                # abort edge passes and every outcome reads "deterministic".
                view = self._view(conv_id, run, outcome=True)
                self.api.update_context(conv_id, {ABORT_CONTEXT_KEY: True})
                self._emit(
                    "aborted", conv_id, digest=run.procedure.digest, step=step, **view
                )
                return []
        definition = runs[-1].procedure.definition if runs else self._definition
        state = definition.states.get(self.api.get_current_state(conv_id))
        config = state.synthesize if state else None
        if config is None or config.result_key in self.api.get_data(conv_id):
            return []
        started = self._start(conv_id, config, runs, step)
        # the pushed child may itself begin on a synthesize state (depth above 1)
        return [*started, *self._tick(conv_id, step)] if started else []

    def _settle(self, conv_id: str) -> list[str]:
        """converse mode: tick, take the abort edge if one was set, and step the
        parent once after a pop so it leaves its synthesize state."""
        api = self.api
        depth = api.get_stack_depth(conv_id)
        replies = self._tick(conv_id, None)
        runs = self._runs_of(conv_id)
        if runs and runs[-1].aborting and not api.has_conversation_ended(conv_id):
            replies.append(api.advance(conv_id).response or "")  # the abort edge
            replies += self._tick(conv_id, None)  # pops the aborted child
        if api.get_stack_depth(conv_id) < depth and not api.has_conversation_ended(
            conv_id
        ):
            replies.append(api.advance(conv_id).response or "")
        return replies

    # --- start and finish a run -------------------------------------------

    def _start(
        self,
        conv_id: str,
        config: SynthesizeConfig,
        runs: list[_Run],
        step: int | None,
    ) -> list[str]:
        api, policy = self.api, self.synthesizer.policy
        data = api.get_data(conv_id)
        api.update_context(conv_id, {config.result_key: _result(STATUS_RUNNING)})
        procedure: Procedure | None = None
        error: BaseException | None = None
        reply = ""
        depth = api.get_stack_depth(conv_id)
        try:
            if len(runs) >= policy.max_synthesis_depth:
                reason = REASON_DEPTH_REFUSED
            elif any(is_missing(data, key) for key in config.inputs):
                reason = REASON_INPUTS_MISSING
            else:
                try:
                    procedure = self.synthesizer.obtain(
                        config.goal,
                        inputs=config.inputs,
                        return_keys=config.return_keys,
                        min_lifecycle=config.min_lifecycle,
                    )
                except (SynthesisError, ValueError) as e:
                    reason, error = reason_of(e), e
                else:
                    self._emit(
                        "obtained",
                        conv_id,
                        digest=procedure.digest,
                        version=procedure.record.version if procedure.record else None,
                        source=procedure.source,
                        rounds=_round_string(procedure.rounds),
                        seconds=procedure.seconds,
                        calls=procedure.usage.calls,
                        prompt_tokens=procedure.usage.prompt_tokens,
                        completion_tokens=procedure.usage.completion_tokens,
                    )
                    reason, reply = self._push(
                        conv_id, config, procedure, data, step, runs
                    )
        except Exception as e:  # fail closed: the result must never stay `running`
            logger.warning(f"synthesis ended on an unexpected {type(e).__name__}")
            reason, reply, error = REASON_UNEXPECTED_ERROR, "", e
            self._unwind(conv_id, depth)
        except BaseException as e:  # Ctrl-C, SystemExit: end the result, then let go
            self._unwind(conv_id, depth)
            try:
                self._fail(conv_id, config, REASON_UNEXPECTED_ERROR, procedure, step, e)
            except Exception as e:  # never mask the interrupt being propagated
                logger.warning(f"could not end the result: {type(e).__name__}")
            raise
        if reason:
            self._fail(conv_id, config, reason, procedure, step, error)
            return []
        return [reply]

    def _fail(
        self,
        conv_id: str,
        config: SynthesizeConfig,
        reason: str,
        procedure: Procedure | None,
        step: int | None,
        error: BaseException | None = None,
    ) -> None:
        """Write the final ``failed`` result (nothing obtains again) and trace it
        (``error``: the exception class name; ``rounds``: the compile's codes)."""
        self.api.update_context(
            conv_id, {config.result_key: _result(STATUS_FAILED, reason, procedure)}
        )
        self._emit(
            "failed",
            conv_id,
            reason=reason,
            step=step,
            digest=procedure.digest if procedure else None,
            error=type(error).__name__ if error else None,
            rounds=_round_codes(error),
        )

    def _unwind(self, conv_id: str, depth: int) -> None:
        """Pop frames a failed push left above ``depth`` (the trim heals ``_runs``)."""
        try:
            while self.api.get_stack_depth(conv_id) > depth:
                self.api.pop_fsm(conv_id)
        except Exception as e:
            logger.warning(f"could not unwind a failed push: {type(e).__name__}")

    def _push(
        self,
        conv_id: str,
        config: SynthesizeConfig,
        procedure: Procedure,
        data: dict[str, Any],
        step: int | None,
        runs: list[_Run],
    ) -> tuple[str, str]:
        """Check and push ``procedure``: ``(reason, "")`` on a refusal, else
        ``("", the child's first reply)`` with the run recorded."""
        definition = procedure.definition
        record = procedure.record
        lifecycle = record.lifecycle if record else LIFECYCLE_VERIFIED
        if reason := runnable_reason(
            definition,
            lifecycle,
            min_lifecycle=config.min_lifecycle,
            policy=self.synthesizer.policy,
            inputs=config.inputs,
            return_keys=config.return_keys,
        ):
            return reason, ""
        try:
            reply = self.api.push_fsm(
                conv_id,
                # a compile may not set a persona (G5): the host's is injected here
                definition.model_copy(update={"persona": self._definition.persona}),
                inherit_context=False,
                context_to_pass={key: data[key] for key in config.inputs},
            )
        except (FSMError, ValueError):
            return REASON_PUSH_FAILED, ""
        runs.append(_Run(procedure, config))
        self._emit("pushed", conv_id, digest=procedure.digest, step=step)
        return "", reply

    def _finish(self, conv_id: str, run: _Run, step: int | None) -> None:
        """Pop the ended child on top, merge its return keys on success only,
        write the final result, count the run."""
        api, config, procedure = self.api, run.config, run.procedure
        # read the child's data before the pop: afterwards the top is the parent
        child = api.get_data(conv_id)
        aborted = run.aborting or api.get_current_state(conv_id) == ABORT_STATE_ID
        ok = not aborted and not any(is_missing(child, k) for k in config.return_keys)
        api.pop_fsm(
            conv_id,
            context_to_return={k: child[k] for k in config.return_keys} if ok else None,
            merge_strategy=ContextMergeStrategy.UPDATE,
        )
        self._runs[conv_id].pop()
        status, reason = (
            (STATUS_OK, None)
            if ok
            else (STATUS_ABORTED, REASON_BUDGET_SPENT)
            if aborted
            else (STATUS_FAILED, REASON_RETURN_KEYS_MISSING)
        )
        api.update_context(
            conv_id, {config.result_key: _result(status, reason, procedure)}
        )
        library = self.synthesizer.library
        if library is not None and procedure.record is not None:
            try:
                library.record_run(
                    procedure.digest,
                    ok,
                    trust_after_successes=self.synthesizer.policy.trust_after_successes,
                )
            except SynthesisError:  # the run happened; a library fault must not undo it
                logger.warning(f"could not count the run of {procedure.digest[:12]}")
        self._emit(
            "popped",
            conv_id,
            digest=procedure.digest,
            status=status,
            reason=reason,
            step=step,
        )

    def _emit(self, event: str, conv: str, **fields: Any) -> None:
        if self._sink is None:
            return
        try:
            self._sink(event, {"conv": conv, **fields})
        except Exception as e:
            logger.warning(f"trace_sink raised {type(e).__name__} on {event!r}")
