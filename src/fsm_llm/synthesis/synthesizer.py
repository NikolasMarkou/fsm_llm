"""
The synthesizer: a strong model compiles a goal into a gated FSM.

``FSMSynthesizer.compile`` asks the strong model for one FSM JSON object (the
loader's default schema version), runs the gates, and on a failure feeds the
failures back, up to ``policy.max_repair_rounds`` times. ``obtain`` adds the
library (reuse a stored, re-verified procedure, else compile) and the approval
gate G7.

# DECISION plan-2026-11-10T194139-2cefeb61/D-005
# A reply cut off at ``max_tokens`` is a failed round and is NEVER parsed (a cut
# object can still be valid JSON and mean less than the model wrote). A provider
# outage is NOT a failed round: it is raised, chained, with no retry (a retry
# spends the strong model's budget on a dead provider). Do NOT repair a reply
# the gates did not see, and do NOT put model text in an error message.

# DECISION plan-2026-11-10T194139-2cefeb61/D-028
# Approval fails closed: a reviewer result is an approval only when it `is True`
# (a truthy "yes" or 1 is a denial). A stored hit that the CURRENT policy needs
# approved but whose record has no approver is skipped, never reused and never
# run: do NOT re-ask the reviewer for a hit, and do NOT store a refused compile.
"""

from __future__ import annotations

import inspect
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from fsm_llm.agents.semantic_tools import EmbedTextsFn
from fsm_llm.aio import acall, aoffload, run_sync
from fsm_llm.constants import SKILL_DEFANG
from fsm_llm.core.definitions import (
    ASCII_IDENTIFIER_PATTERN,
    CompletionRequest,
    FSMDefinition,
    LLMCallCounts,
    LLMResponseError,
    LLMUsage,
    SynthesizeConfig,
)
from fsm_llm.core.state_kinds import declared_tool_names
from fsm_llm.core.visualize import build_fsm_graph, to_mermaid
from fsm_llm.llm import (
    LiteLLMInterface,
    LLMInterface,
    implements_complete,
    interface_model,
)
from fsm_llm.logging import logger
from fsm_llm.okf import capped_line, strip_unsafe_text
from fsm_llm.policy import HumanInTheLoop
from fsm_llm.tools import ToolCall

from .constants import (
    APPROVAL_UNTRUSTED_LABEL,
    CODE_SIMULATION_INCOMPLETE,
    GATE_SIMULATION,
    LIFECYCLE_DEPRECATED,
    LIFECYCLE_VERIFIED,
    REASON_APPROVAL_DENIED,
    REASON_APPROVAL_REQUIRED,
    REASON_LIFECYCLE_TOO_LOW,
    REASON_REVIEWER_ERROR,
)
from .exceptions import LibraryError, SynthesisError, SynthesisRefusedError
from .gates import (
    GATES,
    GateReport,
    safe_text,
    simulated,
    verify_stored,
)
from .library import FSMLibrary, LibraryRecord, digest_of
from .policy import SynthesisPolicy, needs_approval, strip_abort
from .prompts import compile_messages, repair_message
from .reply import _judge
from .simulation import SimulationReport, Simulator

_NO_REPLY = "(no reply)"  # the assistant turn echoed for a reply with no text
_MERMAID_CHARS = 4000  # the diagram shown to a reviewer
_GOAL_CHARS = 500  # the goal shown to a reviewer
_APPROVER = "reviewer"  # a bool-returning callable carries no identity


@dataclass(frozen=True)
class CompileResult:
    """A compile that passed the gates.

    ``definition`` is the loaded abort-injected procedure and ``data`` the same
    as a plain dict (what the library stores); ``report`` is the gate verdict,
    ``rounds`` the strong-model calls made, ``usage`` a snapshot of the
    interface's call counters (its totals so far, not only this compile) or
    ``None`` when the interface keeps none. ``simulation`` is the G6b report
    when a simulator ran and passed, else ``None`` (never a failed one).
    """

    definition: FSMDefinition
    data: dict[str, Any]
    report: GateReport
    rounds: int
    usage: LLMUsage | None
    simulation: SimulationReport | None = None
    failures: tuple[tuple[str, ...], ...] = ()  # the codes of each failed round


@dataclass(frozen=True)
class ProcedureApprovalRequest:
    """What a reviewer sees before a compiled procedure is stored and run.

    Every field is model-authored or caller text, control-stripped, defanged
    (``[ ] < >``) and capped; ``mermaid`` is the only multi-line field. ``text()``
    renders them with ``APPROVAL_UNTRUSTED_LABEL`` as the first line, which no
    field can spell. ``summary`` counts states and transitions and names the
    return keys and any ``completion`` tools.
    """

    digest: str
    mermaid: str
    report: str
    goal: str
    summary: str

    def text(self) -> str:
        return "\n".join(
            (
                APPROVAL_UNTRUSTED_LABEL,
                f"digest: {self.digest}",
                f"goal: {self.goal}",
                f"summary: {self.summary}",
                f"gates: {self.report}",
                self.mermaid,
            )
        )


@dataclass(frozen=True)
class Procedure:
    """A procedure ``obtain`` hands back: ``source`` is ``"library"`` (a stored,
    re-verified hit) or ``"compiled"``; ``definition`` is the abort-injected
    form, ``data`` the same as the canonical dict the library stores, and
    ``record`` its library record (``None`` without a library).

    Telemetry, codes and numbers only (D-049): ``rounds`` the failure codes of
    each failed compile round (``()`` for a first-pass compile or a hit),
    ``seconds`` the ``obtain`` wall time on the injected clock, ``usage`` a
    frozen snapshot of the strong interface's calls and tokens SPENT by this
    obtain (zeros without a usable ``usage()``)."""

    digest: str
    definition: FSMDefinition
    record: LibraryRecord | None
    source: str
    rounds: tuple[tuple[str, ...], ...] = ()
    seconds: float = 0.0
    usage: LLMUsage = field(default_factory=LLMUsage)

    @property
    def data(self) -> dict[str, Any]:
        return self.definition.model_dump(exclude_defaults=True)


def _meter(llm: Any) -> LLMUsage:
    """The interface's usage totals, or zeros when it keeps none (never raises
    for a missing or foreign meter)."""
    usage = getattr(llm, "usage", None)
    totals = usage() if callable(usage) else None
    return totals if isinstance(totals, LLMUsage) else LLMUsage()


def _spent(after: LLMUsage | None, before: LLMUsage) -> LLMUsage:
    """``after - before`` over the counters, clamped at zero (no ``by_kind``)."""
    if not isinstance(after, LLMUsage):
        return LLMUsage()
    return LLMUsage(
        **{
            name: max(0, getattr(after, name) - getattr(before, name))
            for name in LLMCallCounts.model_fields
        }
    )


def _refuse_deprecated(record: LibraryRecord) -> None:
    """``lifecycle_too_low`` for a stored ``deprecated`` record (nothing leaves it)."""
    if record.lifecycle == LIFECYCLE_DEPRECATED:
        raise SynthesisRefusedError(
            "the compile reproduced a deprecated procedure", REASON_LIFECYCLE_TOO_LOW
        )


def _approval_request(
    goal: str, keys: tuple[str, ...], result: CompileResult, digest: str
) -> ProcedureApprovalRequest:
    states = result.definition.states.values()
    tools = sorted(declared_tool_names(states))
    diagram = strip_unsafe_text(
        to_mermaid(build_fsm_graph(result.definition)), multiline=True
    )
    summary = (
        f"{len(result.definition.states)} states, "
        f"{sum(len(s.transitions) for s in states)} transitions, "
        f"returns: {', '.join(keys)}, tools: {', '.join(tools) or 'none'}"
    )
    verdicts = dict(result.report.verdicts)
    if result.simulation is not None:
        verdicts[GATE_SIMULATION] = "pass"
    return ProcedureApprovalRequest(
        digest,
        diagram.translate(SKILL_DEFANG)[:_MERMAID_CHARS],
        ", ".join(f"{g} {v}" for g, v in verdicts.items()),
        safe_text(goal, _GOAL_CHARS),
        safe_text(summary, 400),
    )


def reviewer_from_hitl(
    hitl: HumanInTheLoop,
) -> Callable[[ProcedureApprovalRequest], bool]:
    """A ``reviewer`` that asks ``hitl``'s approval callback.

    HITL asks about tool calls, so the request goes as a call named
    ``approve_procedure`` with ``digest`` and ``review`` (``request.text()``)
    parameters. Only the callback is used, never ``hitl``'s policy. No callback
    raises ``ApprovalDeniedError``, which ``obtain`` reports as
    ``reviewer_error``.
    """

    def review(request: ProcedureApprovalRequest) -> bool:
        call = ToolCall(
            tool_name="approve_procedure",
            parameters={"digest": request.digest, "review": request.text()},
        )
        return hitl.request_approval(call, {})

    return review


def _checked(
    goal: object, inputs: Any, return_keys: Any
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(inputs, return_keys)`` as checked tuples; ``ValueError`` for a bad goal
    or name.

    The synthesize field's own key rule decides (the config is built only to
    run it); a name must also be an identifier, so it can never spell a marker.
    """
    if not isinstance(goal, str) or not capped_line(goal, 1):
        raise ValueError("goal must be a non-blank string")
    config = SynthesizeConfig(
        goal="x", inputs=inputs, return_keys=return_keys, on_failure="x"
    )
    names = config.inputs + config.return_keys
    if not all(re.fullmatch(ASCII_IDENTIFIER_PATTERN, n) for n in names):
        raise ValueError("key names must be identifiers (letters, digits, _)")
    return config.inputs, config.return_keys


class FSMSynthesizer:
    """Compiles goals into gated procedures with a strong model.

    Interface contract (callers: the runner, the tools, tests):
        Exactly one of ``llm`` (an ``LLMInterface`` that implements
        ``complete``) or ``model`` (a name; a ``LiteLLMInterface`` is built)
        or ``ValueError``: the strong model is never an accidental default.
        ``policy`` (default ``SynthesisPolicy()``) bounds the compile; ``clock``
        is the deadline's time source. ``library`` (optional) stores and serves
        procedures, ``embed`` ranks its hits, ``reviewer(request) -> bool``
        decides approval (G7); ``simulator`` (optional, a ``Simulator``) runs
        G6b on every compile that passed G1 to G6a (a failure is a
        repair round, bounded by ``max_repair_rounds`` and
        ``max_simulation_rounds``; a ``require_simulation`` policy without one
        refuses to compile); a reviewer is called with a
        ``ProcedureApprovalRequest`` and approves only by returning ``True``.
        ``obtain`` returns a ``Procedure`` or raises ``SynthesisRefusedError``
        (``approval_required``, ``approval_denied``, ``reviewer_error``),
        ``SynthesisError`` (a compile that never passed) or ``LibraryError``
        (never swallowed).
        ``compile`` returns a ``CompileResult`` or raises ``SynthesisError``
        whose ``details["rounds"]`` lists the failure codes of every round (no
        model text); a provider outage raises it chained from the
        ``LLMResponseError``. Bad arguments raise ``ValueError`` before any
        call. Caller data is never mutated.
    """

    def __init__(
        self,
        llm: LLMInterface | None = None,
        *,
        model: str | None = None,
        policy: SynthesisPolicy | None = None,
        library: FSMLibrary | None = None,
        reviewer: Callable[[ProcedureApprovalRequest], bool] | None = None,
        embed: EmbedTextsFn | None = None,
        simulator: Simulator | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if llm is not None and model is None:
            if not implements_complete(llm):
                raise ValueError("llm must be an LLMInterface that implements complete")
        elif llm is None and model is not None:
            llm = LiteLLMInterface(model=model)
        else:
            raise ValueError("pass exactly one of llm or model")
        self.llm: LLMInterface = llm
        self.model = model or interface_model(llm)
        self.policy = policy or SynthesisPolicy()
        self.library = library
        self.reviewer = reviewer
        self._embed = embed
        self.simulator = simulator
        self._clock = clock

    def compile(
        self,
        goal: str,
        *,
        inputs: Sequence[str] = (),
        return_keys: Sequence[str],
    ) -> CompileResult:
        """Compile ``goal`` into a gate-passed, abort-injected FSM.

        ``inputs`` are the keys the driver provides, ``return_keys`` (at least
        one) the keys the procedure must hand back; both are identifier-only
        public names. At most ``1 + policy.max_repair_rounds`` calls.
        """
        return run_sync(self.acompile(goal, inputs=inputs, return_keys=return_keys))

    async def acompile(
        self,
        goal: str,
        *,
        inputs: Sequence[str] = (),
        return_keys: Sequence[str],
    ) -> CompileResult:
        """The async body of ``compile`` (same contract; ``compile`` is its ``run_sync`` wrapper).

        The strong model is awaited (``acall``: its ``acomplete`` twin, else a worker thread) and a
        ``simulator`` runs on a worker thread (it has no async twin). A cancel ends the compile at its
        await and stores nothing; a worker already running keeps going and its result is dropped.
        """
        ins, keys = _checked(goal, inputs, return_keys)
        return await self._acompile(goal, ins, keys, None)

    async def _acompile(
        self,
        goal: str,
        ins: tuple[str, ...],
        keys: tuple[str, ...],
        reference: dict[str, Any] | None,
    ) -> CompileResult:
        """``acompile`` after the argument checks; ``reference`` is the stored near
        miss (a definition dict) for ``compile_messages`` (or ``None``)."""
        policy = self.policy
        if policy.require_simulation and self.simulator is None:
            raise SynthesisError(
                "a simulation is required and no simulator is set",
                {"rounds": [(CODE_SIMULATION_INCOMPLETE,)]},
            )
        messages = compile_messages(goal, ins, keys, policy, reference)
        deadline = self._clock() + policy.max_synthesis_seconds
        failed: list[tuple[str, ...]] = []
        runs = 0
        reason = "no round passed the gates"
        for attempt in range(1 + policy.max_repair_rounds):
            if attempt and self._clock() >= deadline:
                reason = "the compile deadline passed"
                break
            request = CompletionRequest(
                messages=messages, max_tokens=policy.max_compile_tokens
            )
            try:
                reply = await acall(self.llm, "complete", request)
            except LLMResponseError as e:
                raise SynthesisError(
                    "the strong model call failed", {"rounds": failed}
                ) from e
            report = _judge(reply, policy, ins, keys)
            simulation = None
            if report.definition is not None and report.injected is not None:
                if self.simulator is not None:
                    runs += 1
                    simulation = await acall(
                        self.simulator,
                        "run",
                        report.definition,
                        goal=goal,
                        inputs=ins,
                        return_keys=keys,
                        policy=policy,
                    )
                if simulation is None or simulation.passed:
                    usage = getattr(self.llm, "usage", None)
                    return CompileResult(
                        report.definition,
                        report.injected,
                        report,
                        attempt + 1,
                        usage() if callable(usage) else None,
                        simulation,
                        tuple(failed),
                    )
                report = GateReport(simulation.findings)  # the G6b failure
            failed.append(tuple(dict.fromkeys(f.code for f in report.findings)))
            # DECISION plan-2026-11-10T194139-2cefeb61/D-048: a spent simulation
            # bound is not repairable and the run count caps the G6b rounds.
            # Do NOT re-ask the strong model about either, and do NOT apply
            # only one of the two round bounds.
            if simulation is not None and (
                runs >= policy.max_simulation_rounds
                or CODE_SIMULATION_INCOMPLETE in failed[-1]
            ):
                reason = "the simulation rounds or bounds are spent"
                break
            messages = [
                *messages,
                {"role": "assistant", "content": reply.text or _NO_REPLY},
                {"role": "user", "content": repair_message(report)},
            ]
        last = ", ".join(failed[-1]) if failed else "none"
        raise SynthesisError(
            f"{reason} after {len(failed)} failed round(s); last codes: {last}",
            {"rounds": failed},
        )

    def obtain(
        self,
        goal: str,
        *,
        inputs: Sequence[str] = (),
        return_keys: Sequence[str],
        min_lifecycle: str = LIFECYCLE_VERIFIED,
    ) -> Procedure:
        """A gate-passed procedure for ``goal``: a library hit, else a compile.

        A hit needs a signature match (the library's exact filter, which checks
        key names only, not what the procedure does), a score and a stored form
        that passes
        ``verify_stored`` (else it is skipped with a WARNING and never run) and,
        where the current policy needs approval, a recorded approver (else it
        is skipped). The score is either an embedder cosine at or above
        ``policy.reuse_min_score``, or, with no embedder, the same goal after
        case and spacing (a symbol such as ``>`` or ``#`` is part of the goal;
        ``policy.reuse_min_score`` is never applied to word overlap). Embedder reuse is a SIMILARITY claim: a
        cosine does not understand negation, so a goal that differs by "not"
        may be served the stored procedure. For exact reuse pass ``embed=None``
        or set ``reuse_min_score=1.0``. A hit makes no strong call. A compile that needs approval
        (policy, or any ``completion`` tool) goes to ``reviewer`` and is
        stored ``verified`` only when approved.
        A compile whose digest is already stored keeps the stored lifecycle: a
        ``draft`` is verified, a ``deprecated`` one raises
        ``SynthesisRefusedError`` (``lifecycle_too_low``).
        """
        return run_sync(
            self.aobtain(
                goal,
                inputs=inputs,
                return_keys=return_keys,
                min_lifecycle=min_lifecycle,
            )
        )

    async def aobtain(
        self,
        goal: str,
        *,
        inputs: Sequence[str] = (),
        return_keys: Sequence[str],
        min_lifecycle: str = LIFECYCLE_VERIFIED,
    ) -> Procedure:
        """The async body of ``obtain`` (same contract; ``obtain`` is its ``run_sync`` wrapper).

        The library ``search`` and ``near_misses`` (a directory scan and the embedder) and the
        reviewer run on worker threads in async mode; the library WRITES stay inline and happen
        only after the compile passed and the approval answered, so a cancel before them stores
        nothing, and a worker a cancel left behind (a reviewer a human still holds) never writes.
        """
        ins, keys = _checked(goal, inputs, return_keys)
        policy, library = self.policy, self.library
        started, before = self._clock(), _meter(self.llm)
        if library is not None:
            for match in await acall(
                library,
                "search",
                goal,
                inputs_available=ins,
                return_keys=keys,
                tools_allowed=policy.allowed_tools,
                min_lifecycle=min_lifecycle,
                embed=self._embed,
                top_k=3,
            ):
                record = match.record
                # DECISION plan-2026-11-10T194139-2cefeb61/D-033: the threshold
                # judges embedder cosines only; a word-overlap score reuses on
                # an exact goal (1.0) alone. Do NOT compare one threshold to
                # both scales: set overlap ignores negation and word order.
                if match.score < (policy.reuse_min_score if match.embedded else 1.0):
                    break
                if policy.require_simulation and not simulated(
                    record.provenance["gates_passed"]
                ):
                    continue  # never served as verified without a G6b record
                definition = library.get_definition(record.digest)
                report = verify_stored(
                    definition,
                    policy,
                    inputs=record.signature["inputs_required"],
                    return_keys=record.signature["return_keys"],
                )
                if not report.passed or report.injected is None:
                    logger.warning(
                        f"library procedure {record.digest[:12]} failed "
                        "re-verification and is skipped"
                    )
                elif record.provenance["approved_by"] or not needs_approval(
                    report.injected, policy
                ):
                    return Procedure(
                        record.digest,
                        definition,
                        record,
                        "library",
                        seconds=self._since(started),
                    )
        reference = await self._areference(goal, ins, keys, min_lifecycle)
        result = await self._acompile(goal, ins, keys, reference)
        digest = digest_of(result.definition)
        if library is not None:
            # a deprecated digest is refused BEFORE the reviewer is asked and
            # before `put` could touch the record
            try:
                _refuse_deprecated(library.get_record(digest))
            except LibraryError:
                pass  # absent (or corrupt: `put` raises it below)
        approver = await acall(self, "approval", goal, keys, result)
        saved: LibraryRecord | None = None
        if library is not None:
            saved = library.put(
                result.definition,
                goal=goal,
                inputs_required=ins,
                return_keys=keys,
                synthesizing_model=self.model or "",
                gates_passed=GATES
                if result.simulation is None
                else (*GATES, GATE_SIMULATION),
                approved_by=approver,
                lifecycle=LIFECYCLE_VERIFIED,
            )
            # `put` is idempotent: a digest stored earlier keeps its lifecycle,
            # so the record may be older than this compile. A draft is verified
            # in one lock hold (the compile just passed every gate), a
            # deprecated one stays deprecated and is refused, never returned as
            # a ready procedure (also when it was deprecated since the check).
            saved = library.ensure_verified(digest)
            _refuse_deprecated(saved)
            if result.simulation is not None and result.simulation.scenarios:
                library.add_scenarios(digest, result.simulation.scenarios)
        return Procedure(
            digest,
            result.definition,
            saved,
            "compiled",
            result.failures,
            self._since(started),
            _spent(result.usage, before),
        )

    async def _areference(
        self, goal: str, ins: Sequence[str], keys: Sequence[str], min_lifecycle: str
    ) -> dict[str, Any] | None:
        """The stored near miss for the compile prompt: the definition of the best
        record ``near_misses`` finds that scores as a reuse would (the same
        threshold per scale, D-033), abort path stripped; else ``None``. A hint
        only: a failure to read it is no failure of the compile."""
        library, policy = self.library, self.policy
        if library is None:
            return None
        try:
            for match in await acall(
                library,
                "near_misses",
                goal,
                inputs_available=ins,
                return_keys=keys,
                tools_allowed=policy.allowed_tools,
                min_lifecycle=min_lifecycle,
                embed=self._embed,
            ):
                if match.score >= (policy.reuse_min_score if match.embedded else 1.0):
                    data = library.get_definition(match.record.digest)
                    stored = strip_abort(data.model_dump(exclude_defaults=True))
                    if stored is not None:
                        return stored
        except LibraryError:
            logger.warning("the library could not give a near miss; compiling without")
        return None

    def approval(
        self, goal: str, return_keys: tuple[str, ...], result: CompileResult
    ) -> str | None:
        """G7 for a gate-passed ``result``: ``None`` when neither the policy nor
        a ``completion`` tool needs approval, the approver label when the
        reviewer said exactly ``True``, else ``SynthesisRefusedError``."""
        return run_sync(self.aapproval(goal, return_keys, result))

    # DECISION plan-2026-11-20T055240-ace7a8c8/D-066
    # The reviewer is a bare sync callable (a human may hold it): async mode runs it with
    # ``aoffload`` on a worker, never inline on the loop. Nothing is stored or approved before it
    # answers ``True`` and a cancel leaves a worker that may say yes late: its verdict is dropped and
    # ``aobtain`` writes inline only AFTER this returns. Do NOT call the reviewer inline, do NOT
    # ``put`` before approval, and do NOT await an awaitable verdict (it is a denial, D-028).
    async def aapproval(
        self, goal: str, return_keys: tuple[str, ...], result: CompileResult
    ) -> str | None:
        """The async body of ``approval``; the reviewer runs on a worker thread (see ``_areview``)."""
        if not needs_approval(result.data, self.policy):
            return None
        request = _approval_request(
            goal, return_keys, result, digest_of(result.definition)
        )
        await self._areview(request)
        return _APPROVER

    def _since(self, started: float) -> float:
        return round(self._clock() - started, 3)

    async def _areview(self, request: ProcedureApprovalRequest) -> None:
        """G7: return only when the reviewer says exactly ``True``."""
        if self.reviewer is None:
            raise SynthesisRefusedError(
                "approval is required and no reviewer is set", REASON_APPROVAL_REQUIRED
            )
        try:
            verdict = await aoffload(self.reviewer, request)
        except Exception as e:
            raise SynthesisRefusedError(
                "the reviewer failed", REASON_REVIEWER_ERROR
            ) from e
        if inspect.iscoroutine(verdict):
            verdict.close()  # an async reviewer is a denial, never awaited
        if verdict is not True:
            raise SynthesisRefusedError(
                "the reviewer did not approve", REASON_APPROVAL_DENIED
            )
