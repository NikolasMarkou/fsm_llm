"""The trace emitters of ``SynthesizeExecutor``, split out of ``executor.py`` by a pure move.

Layer ``synthesis`` (plan c720601d, step 1; the module was 745 lines and the diagnosis steps grow
its trace plumbing). ``TraceEmitters`` is a mixin in the bases of ``SynthesizeExecutor``: the four
methods are the ones that were defined on it, byte for byte, and read what the executor sets in its
constructor (``_sink``). Nothing here is public: the wire of the events is the one documented in
``executor.py`` (``obtained``, ``pushed``, ``stepped``, ``aborted``, ``popped``, ``failed``), scalars
only, never a value, user text or reply (D-044); a sink that raises, or a child that cannot be
read, never changes a run (one WARNING).

Interface contract (private, one caller): ``_trace_steps(conv, run, child)`` emits ``stepped`` once
per new engine-counted turn of the child; ``_view(run, child, *, outcome=False)`` returns the
value-free facts of the child (``{}`` without a sink or on any failure); ``_emit_obtained(conv,
procedure)`` emits ``obtained``; ``_emit(event, conv, **fields)`` calls the sink and swallows its
failure.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fsm_llm.core.definitions import FSMContext
from fsm_llm.core.expressions import is_missing
from fsm_llm.core.transition_evaluator import TransitionEvaluator
from fsm_llm.engine.kinds.requests import ChildView
from fsm_llm.logging import logger

from .constants import TRACE_KIND_OTHER, TRACE_KINDS
from .procedures import _round_string
from .synthesizer import Procedure
from .traces import gate_reads, signal_fields

if TYPE_CHECKING:
    from .executor import _Run


class TraceEmitters:
    """The trace half of ``SynthesizeExecutor``: ``self._sink`` is set by its constructor."""

    _sink: Callable[[str, dict[str, Any]], None] | None

    def _trace_steps(self, conv: str, run: _Run, child: ChildView) -> None:
        """Trace the turns the engine counted on the child since the last event: one ``stepped``
        event carries the engine's lifetime count (turns run by hand count, and several turns
        between two settle rounds are one event); nothing is counted here."""
        taken = child.steps
        if taken is None or taken <= run.traced:
            return
        run.traced = taken
        self._emit(
            "stepped",
            conv,
            digest=run.procedure.digest,
            step=taken,
            **self._view(run, child),
        )

    def _view(
        self, run: _Run, child: ChildView, *, outcome: bool = False
    ) -> dict[str, Any]:
        """Value-free facts about the child for a trace event: its ``state``, the declared names
        present (``keys``), ``name:type`` for those an edge gate reads (``kinds``, types from
        ``TRACE_KINDS``), with ``outcome`` what the evaluator makes of its edges now, and the
        event-log signals of the child when it has them (names and counts only). ``{}``
        without a sink, and on any failure (one WARNING): a trace never changes a run."""
        if self._sink is None:
            return {}
        try:
            definition, config, data = run.procedure.definition, run.config, child.data
            gated = gate_reads(definition.states.values())
            seen = sorted(
                key
                for key in gated | {*config.inputs, *config.return_keys}
                if not is_missing(data, key)
            )
            kinds = (
                f"{key}:{TRACE_KINDS.get(type(data[key]), TRACE_KIND_OTHER)}"
                for key in seen
                if key in gated
            )
            view = {
                "state": child.state_id,
                "keys": ",".join(seen),
                "kinds": ",".join(kinds),
            }
            if outcome:
                found = TransitionEvaluator().evaluate_transitions(
                    definition.states[child.state_id], FSMContext(data=dict(data))
                )
                view["outcome"] = found.result_type.value
            if child.signals is not None:  # set at the child's end, with the log on
                view.update(signal_fields(child.signals))
            return view
        except Exception as e:
            logger.warning(f"could not read the child for a trace: {type(e).__name__}")
            return {}

    def _emit_obtained(self, conv: str, procedure: Procedure) -> None:
        self._emit(
            "obtained",
            conv,
            digest=procedure.digest,
            version=procedure.record.version if procedure.record else None,
            source=procedure.source,
            rounds=_round_string(procedure.rounds),
            seconds=procedure.seconds,
            calls=procedure.usage.calls,
            prompt_tokens=procedure.usage.prompt_tokens,
            completion_tokens=procedure.usage.completion_tokens,
        )

    def _emit(self, event: str, conv: str, **fields: Any) -> None:
        if self._sink is None:
            return
        try:
            self._sink(event, {"conv": conv, **fields})
        except Exception as e:
            logger.warning(f"trace_sink raised {type(e).__name__} on {event!r}")
