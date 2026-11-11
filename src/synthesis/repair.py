"""
Repair: ask the strong model for a patch to the diagnosed states, and promote
the patched child only after it is re-verified and replayed.

``propose_patch`` only RETURNS raw ops; it never applies, counts or checks
them (``apply_patch`` and the gates do, D-038). ``repair_procedure`` judges
those ops and writes only when told to.

# DECISION plan-2026-11-10T194139-2cefeb61/D-052
# The request holds codes, state ids, key names and numbers in the diagnosis
# lines and ONLY the diagnosed states' JSON, in the one fence of ``prompts``.
# Nothing model-authored reaches the system message, a state id or key that is
# not an identifier is never sent, and an unreadable reply is an empty patch.
# Do NOT send the whole definition, do NOT parse a truncated or non-final
# reply, do NOT apply or count ops here, and do NOT write a second fence.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

from fsm_llm.definitions import (
    ASCII_IDENTIFIER_PATTERN,
    CompletionRequest,
    FSMDefinition,
    LLMResponseError,
)
from fsm_llm.eval.stats import pass_rate
from fsm_llm.llm import LLMInterface
from fsm_llm.utilities import extract_json_from_text

from .constants import (
    ABORT_STATE_ID,
    CODE_NOT_PROCEDURE_FAULT,
    DIAGNOSIS_CODES,
    GATE_SIMULATION,
    LIFECYCLE_DEPRECATED,
    LIFECYCLE_TRUSTED,
    LIFECYCLE_VERIFIED,
    PATCH_MAX_TOKENS,
)
from .diagnose import Diagnosis, diagnose
from .exceptions import (
    LibraryError,
    PatchError,
    SynthesisError,
    SynthesisRefusedError,
)
from .gates import GATES, run_gates
from .library import FSMLibrary, digest_of
from .patch import apply_patch
from .policy import SynthesisPolicy, strip_abort
from .prompts import patch_messages
from .simulation import Simulator
from .synthesizer import CompileResult, FSMSynthesizer


def _name(value: object) -> bool:
    return isinstance(value, str) and bool(
        re.fullmatch(ASCII_IDENTIFIER_PATTERN, value)
    )


def propose_patch(
    diagnoses: Sequence[Diagnosis],
    definition: dict[str, Any],
    policy: SynthesisPolicy,
    llm: LLMInterface,
) -> list[dict[str, Any]]:
    """Ask ``llm`` for a JSON Patch limited to the diagnosed states.

    Interface contract: ``definition`` is the stripped canonical FSM dict
    (never mutated). A diagnosis qualifies when its code is a fault code (not
    ``not-procedure-fault``) and its state is an identifier, not ``abort`` and
    a state of ``definition``; one that does not is ignored, and key names that
    are not identifiers are left out of its line. No qualifying diagnosis means
    no call and ``[]``. Otherwise exactly one ``llm.complete`` call
    (``PATCH_MAX_TOKENS``) and the reply's ``{"patch": [...]}`` ops, raw. A cut
    reply, ``kind != "final"``, a non-object, a missing or non-list ``patch`` or
    an op that is not an object is ``[]``, never an exception. A provider
    outage raises ``SynthesisError`` chained from the ``LLMResponseError``, with
    no model text and no retry.
    """
    states = definition.get("states")
    states = states if isinstance(states, dict) else {}
    lines: list[str] = []
    shown: dict[str, Any] = {}
    for d in diagnoses:
        if (
            d.code in DIAGNOSIS_CODES
            and d.code != CODE_NOT_PROCEDURE_FAULT
            and _name(d.state)
            and d.state != ABORT_STATE_ID
            and d.state in states
        ):
            shown[d.state] = states[d.state]
            keys = ",".join(k for k in d.keys if _name(k))
            lines.append(f"- {d.code} state={d.state} keys={keys} runs={d.runs}")
    if not shown:
        return []
    request = CompletionRequest(
        messages=patch_messages(lines, shown, policy), max_tokens=PATCH_MAX_TOKENS
    )
    try:
        reply = llm.complete(request)
    except LLMResponseError as e:
        raise SynthesisError("the strong model call failed") from e
    data = None
    if not reply.truncated and reply.kind == "final":
        data = extract_json_from_text(reply.text or "")
    ops = data.get("patch") if data else None
    if isinstance(ops, list) and all(type(op) is dict for op in ops):
        return ops
    return []


@dataclass(frozen=True)
class PromotionResult:
    """How one ``repair_procedure`` call ended: codes, digests and counts only.

    ``code`` is one of ``constants.PROMOTION_CODES`` (``promote`` is the only
    success); ``child`` is the patched definition's digest once it is known;
    ``parent_passes`` and ``child_passes`` are passes per replayed scenario out
    of ``trials`` each; ``written`` and ``deprecated`` say what reached the
    library. No patch value, script turn or model text is ever a field.
    """

    parent: str
    code: str
    child: str | None = None
    parent_passes: tuple[int, ...] = ()
    child_passes: tuple[int, ...] = ()
    trials: int = 0
    written: bool = False
    deprecated: bool = False

    @property
    def promote(self) -> bool:
        return self.code == "promote"

    def rates(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """``(parent, child)`` as ``eval.stats.pass_rate`` dicts (Wilson CIs)."""
        n = self.trials * len(self.parent_passes)
        return pass_rate(sum(self.parent_passes), n), pass_rate(
            sum(self.child_passes), n
        )


def decide(parent: Sequence[int], child: Sequence[int]) -> str:
    """The replay verdict from passes per scenario (D-038 item 6): ``regressed``
    when the child's total is lower or a scenario the parent passed at least
    once never passes for the child, else ``promote`` when some scenario passes
    more often, else ``no-improvement``."""
    pairs = list(zip(parent, child, strict=True))
    if sum(child) < sum(parent) or any(p and not c for p, c in pairs):
        return "regressed"
    return "promote" if any(c > p for p, c in pairs) else "no-improvement"


# DECISION plan-2026-11-10T194139-2cefeb61/D-053
# Nothing is written unless ``write=True`` AND the verdict is ``promote``: the
# child is the parent plus a model-authored patch, re-verified end to end and
# replayed against the parent first. The child is stored ``verified``, never
# ``trusted`` and never with the parent's approver; a ``trusted`` parent is
# never deprecated (checked inside the library lock); the parent is retired
# LAST so a crash leaves two ``verified`` records and ``search`` picks the
# child. Do NOT make ``write=True`` the default, do NOT write before the last
# step, do NOT replay with a second interface, and do NOT put patch values or
# script turns into the result.
def repair_procedure(
    parent_digest: str,
    ops: Sequence[dict[str, Any]],
    *,
    library: FSMLibrary,
    synthesizer: FSMSynthesizer,
    simulator: Simulator,
    replay_scripts: Sequence[dict[str, Any]] | None = None,
    write: bool = False,
) -> PromotionResult:
    """Judge ``ops`` (from ``propose_patch``) as a repair of ``parent_digest``.

    Interface contract: pure until the last step; every refusal is a
    ``PromotionResult`` whose ``code`` is in ``constants.PROMOTION_CODES`` and
    writes nothing. In order: the parent must be ``verified`` or ``trusted`` and
    canonical; the lineage limit (the repairs in the whole tree from the root,
    ``FSMLibrary.count_lineage``) and ``repair_min_failures`` (runs of its stored traces that
    ``diagnose`` calls a procedure fault; a written repair consumes the traces it
    used, so only newer runs count next time) are checked; ``apply_patch`` is scoped to the diagnosed
    states; the child must pass G1 to G6a, differ from the parent and from every
    stored record, be approved when ``needs_approval`` says so, and pass G6b; the
    replay set (the parent's stored scenarios plus ``replay_scripts``, each
    ``{"kind", "turns"}``, a ``kind`` of ``happy`` or ``correction`` must end
    with the return keys) runs ``policy.replay_trials`` times on parent and
    child with ``simulator``'s interface, and ``decide`` judges the counts. Only
    ``write=True`` with ``promote`` stores the child (``verified``, never
    ``trusted``; its record, scenarios and ``patches/<child>.json``), retires the
    parent's traces, then deprecates a non-``trusted`` parent (one already
    deprecated by another writer is a result, not an error). Raises ``LibraryError`` for an unknown
    parent or a corrupt file, ``SynthesisError`` for malformed
    ``replay_scripts``; a provider outage inside G6b is ``simulation-failed``.
    """
    policy = synthesizer.policy
    stop = partial(PromotionResult, parent_digest)
    try:
        scripts = [
            *library.read_scenarios(parent_digest),
            *library.clean_scenarios(replay_scripts or []),
        ]
    except ValueError as exc:
        raise SynthesisError(str(exc)) from exc
    record = library.get_record(parent_digest)
    if record.lifecycle not in (LIFECYCLE_VERIFIED, LIFECYCLE_TRUSTED):
        return stop("parent-not-live")
    parent = library.get_definition(parent_digest)
    stripped = strip_abort(parent.model_dump(exclude_defaults=True))
    if stripped is None:
        return stop("parent-not-canonical")
    inputs, keys = record.signature["inputs_required"], record.signature["return_keys"]
    # DECISION plan-2026-11-10T194139-2cefeb61/D-072
    # The lineage is the whole tree from the root: its repairs (every stored
    # record below the root, deprecated ones too) are bounded together, whichever
    # node is repaired. Do NOT count per record (a trusted root fans out and each
    # child starts its own chain) and do NOT add the minor version (double count).
    if library.count_lineage(parent_digest) >= policy.max_repairs_per_lineage:
        return stop("lineage-limit")
    traces, _ = library.read_traces(parent_digest)
    # the stripped form: the abort edge reads a key no trace holds
    faults = [
        d
        for d in diagnose(traces, FSMDefinition(**stripped), policy, return_keys=keys)
        if d.code != CODE_NOT_PROCEDURE_FAULT
    ]
    failing = sum(d.runs for d in faults)
    if failing < policy.repair_min_failures:
        return stop("too-few-failures")
    try:
        patched = apply_patch(
            stripped, list(ops), scope_states={d.state for d in faults}, policy=policy
        )
    except PatchError:
        return stop("patch-refused")
    report = run_gates(patched, policy, inputs=inputs, return_keys=keys)
    if report.definition is None or report.injected is None:
        return stop("gates-failed")
    child = digest_of(report.definition)
    if child == parent_digest:
        return stop("no-op", child)
    try:
        library.get_record(child)
        return stop("child-exists", child)
    except LibraryError:
        pass  # no record yet (a corrupt one fails loud in `put`)
    goal = str(record.provenance["goal"] or "")
    try:
        approver = synthesizer.approval(
            goal,
            tuple(keys),
            CompileResult(report.definition, report.injected, report, 0, None),
        )
    except SynthesisRefusedError:
        return stop("approval-refused", child)
    try:
        sim = simulator.run(
            report.definition, goal=goal, inputs=inputs, return_keys=keys, policy=policy
        )
    except SynthesisError:  # every trial errored: an outage, not a verdict
        return stop("simulation-failed", child)
    if not sim.passed:
        return stop("simulation-failed", child)
    if not scripts:
        return stop("no-replay-set", child)
    replays = [
        simulator.replay(
            d,
            scripts,
            inputs=inputs,
            return_keys=keys,
            trials=policy.replay_trials,
            policy=policy,
        )
        for d in (parent, report.definition)
    ]
    before, after = replays
    if before is None or after is None:
        return stop("replay-incomplete", child)
    code = decide(before, after)
    result = stop(code, child, tuple(before), tuple(after), policy.replay_trials)
    if code != "promote" or not write:
        return result
    total = policy.replay_trials * len(scripts)
    library.put(
        report.definition,
        goal=goal,
        inputs_required=inputs,
        return_keys=keys,
        synthesizing_model=str(record.provenance["synthesizing_model"] or ""),
        gates_passed=(*GATES, GATE_SIMULATION),
        approved_by=approver,
        lifecycle=LIFECYCLE_VERIFIED,
        parent=parent_digest,
        parent_run=f"{failing} failing runs",
        repair=hashlib.sha256(json.dumps(ops, sort_keys=True).encode()).hexdigest(),
        replay=f"{sum(after)}/{total} vs {sum(before)}/{total}",
    )
    if sim.scenarios:
        library.add_scenarios(child, sim.scenarios)
    library.add_patch(child, ops)
    # DECISION plan-2026-11-10T194139-2cefeb61/D-063
    # The evidence of a written repair is consumed (moved, never deleted), so
    # old runs buy no second repair. Do NOT stamp traces with a clock.
    library.retire_traces(parent_digest, child)
    kept = library.deprecate(parent_digest, keep_trusted=True)
    return replace(
        result, written=True, deprecated=kept.lifecycle == LIFECYCLE_DEPRECATED
    )
