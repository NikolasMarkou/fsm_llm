"""The rules the result-key kinds share, and the home of the ``call`` kind's hooks.

A leaf core module like ``state_kinds`` (standard library only, never imports a
``fsm_llm`` module, never logs): ``state_kinds`` imports it, so it can import
nothing back (it takes what it needs from ``state_kinds`` as arguments).

Interface contract:

- ``check_result_edges(definition, state, evaluator, logic_keys, *, kind,
  producer, outcome, result_key, on_failure, inputs, return_names,
  return_entry, samples, owned)``: the loader's cross-state edge rule of a kind
  whose driver writes ``context[result_key]`` after the state was entered
  (``synthesize`` today). Returns a message (without the ``State '<id>':``
  prefix) or ``None``. *definition* and *state* are read by ``getattr``;
  *evaluator* is the real ``TransitionEvaluator`` and *logic_keys* is
  ``definitions.logic_referenced_keys`` (handed in: this leaf cannot import
  them); *samples* is the tuple of ``(label, result)`` the edges are run on
  (``None`` is "no result yet", each result has a ``status``); *owned* is a
  zero-argument callable returning ``{result key: label}`` of the whole
  definition (``state_kinds.owned_result_keys``), called only when the edge
  rules passed. The four wording parameters name the kind in the messages:
  *kind* (``synthesize``), *producer* (``the runner``), *outcome* (``the
  procedure``) and *return_entry* (``return_keys``, the label of the
  *return_names* list). Never raises on a loaded state.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

__all__ = ["check_result_edges"]

_RUNNING = "running"


# DECISION plan-2026-11-10T194139-2cefeb61/D-034 and
# plan-2026-11-10T194139-2cefeb61/D-035: an edge of a synthesize state may
# read the result key OR a handler-only key (the runner's injected abort
# edge, handler-only `procedure_abort`, sits on every state of a compiled
# procedure, nested synthesize states included), and is then RUN on four
# sample results by the real evaluator. Do NOT
# require the result key alone, do NOT move this check to validator.py (it may
# never be stricter than the loader; the loader calls it, see
# `FSMDefinition._check_synthesize_states`), and do NOT fall back to a
# structural "mentions the key" rule alone: review pass 2 reproduced both holes
# (`status == "failed"` strands the parent after an abort; a `!` edge on
# the result fires only when there is none).
def check_result_edges(
    definition: Any,
    state: Any,
    evaluator: Any,
    logic_keys: Callable[[Any], set[str]],
    *,
    kind: str,
    producer: str,
    outcome: str,
    result_key: str,
    on_failure: str,
    inputs: Iterable[str],
    return_names: Iterable[str],
    return_entry: str,
    samples: Iterable[tuple[str, dict[str, Any] | None]],
    owned: Callable[[], Mapping[str, str]],
) -> str | None:
    """Cross-state rules for a state of a result-key kind (they need the whole FSM).

    1. Every transition must read the state's ``result_key`` (a condition that
       requires it or reads it in its logic) or a ``handler_only_keys`` entry
       (only a handler or the runner writes one; the runner's injected abort
       edge is of this kind). This is necessary, not enough (rule 2 runs the
       edges); it also catches an edge on an unrelated key, which the samples
       of rule 2 cannot see.
    2. The real evaluator (``TransitionEvaluator.transition_passes``, never a
       copy of its logic) runs the state's edges on the four sample results,
       shaped as the runner writes them (``status`` of ``running``, ``failed``
       or ``aborted``, or no result at all). No edge may pass the absent or the
       ``running`` sample (it would fire before the runner has decided), and an
       edge to ``on_failure`` must pass the ``failed`` sample and an edge to it
       the ``aborted`` sample (or the parent is stranded after a failure). An
       edge on a handler-only key alone passes no sample: fine, but it cannot
       be the failure edge.
    3. Host keys a state returns or takes may not be runner-owned: a
       return key merged by ``pop_fsm`` would write a value the child's model
       extracted over a handler-only key, a synthesize result or a completion
       result; an input named like a result would be overwritten by the runner.
       (A completion ``messages_key`` is internal-prefixed and so can never be a
       public return key.)
    """
    handler_only = frozenset(getattr(definition, "handler_only_keys", None) or ())
    reads = {result_key, *handler_only}
    for transition in state.transitions:
        if not any(
            reads.intersection(condition.requires_context_keys or ())
            or reads.intersection(logic_keys(condition.logic))
            for condition in transition.conditions or ()
        ):
            return (
                f"the transition to {transition.target_state!r} must read the "
                f"{kind} result key {result_key!r} (a condition that "
                "requires it or reads it in its logic), or it fires "
                f"before {producer} has produced a result"
            )
    # other keys the edges require are present: an edge also gated on
    # one (an AND) must not escape the samples for lacking it
    others = {
        key: "x"
        for transition in state.transitions
        for condition in transition.conditions or ()
        for key in condition.requires_context_keys or ()
        if key != result_key and key not in handler_only
    }
    for label, result in samples:
        context: dict[str, Any] = dict(others)
        if result is not None:
            context[result_key] = result
        passing = [
            t for t in state.transitions if evaluator.transition_passes(t, context)
        ]
        if result is None or result["status"] == _RUNNING:
            for transition in passing:
                return (
                    f"the transition to {transition.target_state!r} fires before "
                    f"{producer} has produced a result (it passes when "
                    f"{result_key!r} is {label})"
                )
        elif not any(t.target_state == on_failure for t in passing):
            return (
                f"no transition to {on_failure!r} fires when "
                f"{outcome} failed or aborted (it does not pass when "
                f"{result_key!r} is {label}, {result!r})"
            )
    known = owned()
    for entry, names in (
        (return_entry, return_names),
        ("inputs", inputs),
    ):
        for name in names:
            if entry == return_entry and name in handler_only:
                reason = "a handler_only_keys entry"
            elif name in known:
                reason = known[name]
            else:
                continue
            return f"{kind} {entry} entry {name!r} collides with {reason}"
    return None
