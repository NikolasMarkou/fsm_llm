"""
Diagnose run traces into a fixed vocabulary of procedure faults.

``diagnose`` is pure: it reads ``RunTrace`` values, the definition and the
policy, and returns ``Diagnosis`` values of codes, state ids, key names, counts
and shares (never a context value, user text or scenario label). The rule
table ``RULES`` is data, in the order of D-038; first match per trace wins.

# DECISION plan-2026-11-10T194139-2cefeb61/D-050
# A patch needs POSITIVE evidence: only the reasons in
# ``PROCEDURE_FAULT_REASONS`` can be a fault, so an outage, a refusal and an
# unknown reason are ``not-procedure-fault``. Do NOT turn this into a blacklist
# of infrastructure reasons, and do NOT read ``budget_spent`` as an outage (it
# is the procedure's own turn budget).
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from typing import Any

from fsm_llm.definitions import FSMDefinition

from .constants import (
    CODE_ABANDONMENT,
    CODE_CYCLE,
    CODE_EXTRACTION_MISS,
    CODE_NOT_PROCEDURE_FAULT,
    CODE_PATH_TOO_LONG,
    CODE_RETURN_KEYS_MISSING,
    CODE_TIE_LOOP,
    CODE_TYPE_MISMATCH,
    CODE_VALUE_GATE_UNMET,
    PROCEDURE_FAULT_REASONS,
    REASON_RETURN_KEYS_MISSING,
    STATUS_ABORTED,
    STATUS_FAILED,
    STATUS_OK,
)
from .policy import SynthesisPolicy
from .traces import RunTrace, gate_reads

_PERIODS = (2, 3, 4)  # state cycles a run may repeat
_CONCENTRATION = 0.5  # share of unexplained converse failures in one state


@dataclass(frozen=True)
class Diagnosis:
    """One finding: ``code`` (``DIAGNOSIS_CODES``), the ``state`` id ("" when
    none), the sorted key ``keys`` involved, ``runs`` traces it explains and
    ``share`` (``runs`` over all traces given, 3 decimals)."""

    code: str
    state: str
    keys: tuple[str, ...]
    runs: int
    share: float


@dataclass(frozen=True)
class _Ctx:
    definition: FSMDefinition
    policy: SynthesisPolicy
    return_keys: Collection[str]


def _dwell(t: RunTrace) -> int:
    """How many of the last states equal the stuck state."""
    n = 0
    for s in reversed(t.states):
        if s != t.stuck:
            break
        n += 1
    return n


def _blocked(t: RunTrace, ctx: _Ctx) -> tuple[str, set[str]] | None:
    """The stuck state and the keys its gates read, for an aborted run whose
    edges were blocked there."""
    state = ctx.definition.states.get(t.stuck or "")
    if state is None or t.status != STATUS_ABORTED or t.outcome != "blocked":
        return None
    return state.id, gate_reads([state])


def _extraction_miss(t: RunTrace, ctx: _Ctx) -> Any:
    found = _blocked(t, ctx)
    missing = found and sorted(found[1] - set(t.keys))
    return (found[0], tuple(missing)) if found and missing else None


def _type_mismatch(t: RunTrace, ctx: _Ctx) -> Any:
    found = _blocked(t, ctx)
    if not found:
        return None
    declared = {
        f.field_name: f.field_type
        for s in ctx.definition.states.values()
        for f in s.field_extractions or []
    }
    seen = dict(t.kinds)
    bad = sorted(
        k
        for k in found[1]
        if k in seen
        and declared.get(k, "any") not in ("any", seen[k])
        and (declared[k], seen[k]) != ("float", "int")
    )
    return (found[0], tuple(bad)) if bad else None


def _value_gate_unmet(t: RunTrace, ctx: _Ctx) -> Any:
    found = _blocked(t, ctx)
    return (found[0], tuple(sorted(found[1]))) if found else None


def _tie_loop(t: RunTrace, ctx: _Ctx) -> Any:
    ok = t.status == STATUS_ABORTED and t.outcome == "ambiguous" and t.stuck
    return (t.stuck, ()) if ok and _dwell(t) >= ctx.policy.dwell_threshold else None


def _cycle(t: RunTrace, ctx: _Ctx) -> Any:
    if t.status != STATUS_ABORTED:
        return None
    states, repeats = t.states, ctx.policy.dwell_threshold
    for p in _PERIODS:
        tail = states[-p * repeats :]
        if len(tail) == p * repeats and tail == tail[:p] * repeats:
            if len(set(tail[:p])) > 1:  # a stay is no cycle
                return states[-1], ()
    return None


def _path_too_long(t: RunTrace, ctx: _Ctx) -> Any:
    moved = len(t.states) > 1 and all(
        a != b for a, b in zip(t.states, t.states[1:], strict=False)
    )
    ok = t.status == STATUS_ABORTED and t.outcome == "deterministic" and moved
    return (t.stuck or t.states[-1], ()) if ok else None


def _return_keys_missing(t: RunTrace, ctx: _Ctx) -> Any:
    if t.status != STATUS_FAILED or t.reason != REASON_RETURN_KEYS_MISSING:
        return None
    missing = tuple(sorted(set(ctx.return_keys) - set(t.keys)))
    return (t.states[-1] if t.states else "", missing)


#: The rule table, in the fixed order of D-038 (first match per trace wins).
#: ``not-procedure-fault`` is decided before it and the cross-trace
#: abandonment rule after it, both in ``diagnose``.
RULES: tuple[tuple[str, Callable[[RunTrace, _Ctx], Any]], ...] = (
    (CODE_EXTRACTION_MISS, _extraction_miss),
    (CODE_TYPE_MISMATCH, _type_mismatch),
    (CODE_VALUE_GATE_UNMET, _value_gate_unmet),
    (CODE_TIE_LOOP, _tie_loop),
    (CODE_CYCLE, _cycle),
    (CODE_PATH_TOO_LONG, _path_too_long),
    (CODE_RETURN_KEYS_MISSING, _return_keys_missing),
)


def diagnose(
    traces: Iterable[RunTrace],
    definition: FSMDefinition,
    policy: SynthesisPolicy,
    *,
    return_keys: Collection[str] = (),
) -> tuple[Diagnosis, ...]:
    """Classify *traces* of a procedure into ``Diagnosis`` values.

    Interface contract: pure and deterministic (no model, clock or IO); the
    order of *traces* never matters. A trace whose status is ``ok`` or whose
    reason is not in ``PROCEDURE_FAULT_REASONS`` is ``not-procedure-fault``
    (state "", no keys) and never a basis for a patch; a fault trace no rule
    explains yields nothing. ``return_keys`` names the procedure's return keys
    for ``return-keys-missing`` (the traces do not hold them). The result is
    sorted by (runs descending, state, code). It never raises for a trace
    content; a wrong argument type raises ``TypeError`` or ``AttributeError``.
    """
    traces = list(traces)
    ctx = _Ctx(definition, policy, return_keys)
    found: dict[tuple[str, str], tuple[int, set[str]]] = {}
    stray: list[RunTrace] = []  # converse faults no rule explained

    def add(code: str, state: str, keys: Iterable[str], runs: int = 1) -> None:
        n, names = found.get((code, state), (0, set()))
        found[(code, state)] = (n + runs, names | set(keys))

    for t in traces:
        if t.status == STATUS_OK or t.reason not in PROCEDURE_FAULT_REASONS:
            add(CODE_NOT_PROCEDURE_FAULT, "", ())
            continue
        for code, rule in RULES:
            hit = rule(t, ctx)
            if hit:
                add(code, *hit)
                break
        else:
            if t.mode == "converse":
                stray.append(t)
    ends: dict[str, int] = {}
    for t in stray:
        end = t.stuck or (t.states[-1] if t.states else "")
        ends[end] = ends.get(end, 0) + 1
    for end, n in ends.items():
        if n >= policy.repair_min_failures and n / len(stray) >= _CONCENTRATION:
            add(CODE_ABANDONMENT, end, (), n)
    total = len(traces)
    return tuple(
        sorted(
            (
                Diagnosis(code, state, tuple(sorted(names)), n, round(n / total, 3))
                for (code, state), (n, names) in found.items()
            ),
            key=lambda d: (-d.runs, d.state, d.code),
        )
    )
