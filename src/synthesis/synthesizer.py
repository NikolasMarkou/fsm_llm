"""
The synthesizer: a strong model compiles a goal into a gated FSM.

``FSMSynthesizer.compile`` asks the strong model for one FSM v4.1 JSON object,
runs the gates, and on a failure feeds the failures back, up to
``policy.max_repair_rounds`` times. ``obtain`` adds the library (reuse a
stored, re-verified procedure, else compile) and the approval gate G7.

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

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from fsm_llm.agents.definitions import ToolCall
from fsm_llm.agents.hitl import HumanInTheLoop
from fsm_llm.agents.semantic_tools import EmbedTextsFn
from fsm_llm.constants import SKILL_DEFANG
from fsm_llm.definitions import (
    ASCII_IDENTIFIER_PATTERN,
    CompletionRequest,
    CompletionResponse,
    FSMDefinition,
    LLMResponseError,
    LLMUsage,
    SynthesizeConfig,
)
from fsm_llm.llm import (
    LiteLLMInterface,
    LLMInterface,
    implements_complete,
    interface_model,
)
from fsm_llm.logging import logger
from fsm_llm.okf import capped_line, strip_unsafe_text
from fsm_llm.utilities import extract_json_from_text
from fsm_llm.visualizer import build_fsm_graph, to_mermaid

from .constants import (
    APPROVAL_UNTRUSTED_LABEL,
    LIFECYCLE_VERIFIED,
    REASON_APPROVAL_DENIED,
    REASON_APPROVAL_REQUIRED,
    REASON_REVIEWER_ERROR,
)
from .exceptions import SynthesisError, SynthesisRefusedError
from .gates import GATES, GateFinding, GateReport, run_gates, verify_stored
from .library import FSMLibrary, LibraryRecord, digest_of
from .policy import SynthesisPolicy, needs_approval
from .prompts import REPLY_FAILURES, compile_messages, repair_message

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
    ``None`` when the interface keeps none.
    """

    definition: FSMDefinition
    data: dict[str, Any]
    report: GateReport
    rounds: int
    usage: LLMUsage | None


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
    ``record`` its library record (``None`` without a library)."""

    digest: str
    definition: FSMDefinition
    record: LibraryRecord | None
    source: str

    @property
    def data(self) -> dict[str, Any]:
        return self.definition.model_dump(exclude_defaults=True)


def _approval_request(
    goal: str, keys: tuple[str, ...], result: CompileResult, digest: str
) -> ProcedureApprovalRequest:
    def safe(text: str, cap: int) -> str:
        return capped_line(text, cap).translate(SKILL_DEFANG)

    states = result.definition.states.values()
    tools = sorted(
        {
            t["function"]["name"]
            for s in states
            if s.completion and s.completion.tools
            for t in s.completion.tools
        }
    )
    diagram = strip_unsafe_text(
        to_mermaid(build_fsm_graph(result.definition)), multiline=True
    )
    summary = (
        f"{len(result.definition.states)} states, "
        f"{sum(len(s.transitions) for s in states)} transitions, "
        f"returns: {', '.join(keys)}, tools: {', '.join(tools) or 'none'}"
    )
    return ProcedureApprovalRequest(
        digest,
        diagram.translate(SKILL_DEFANG)[:_MERMAID_CHARS],
        ", ".join(f"{g} {v}" for g, v in result.report.verdicts.items()),
        safe(goal, _GOAL_CHARS),
        safe(summary, 400),
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
        decides approval (G7); a reviewer is called with a
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
        ins, keys = _checked(goal, inputs, return_keys)
        policy = self.policy
        messages = compile_messages(goal, ins, keys, policy)
        deadline = self._clock() + policy.max_synthesis_seconds
        failed: list[tuple[str, ...]] = []
        reason = "no round passed the gates"
        for attempt in range(1 + policy.max_repair_rounds):
            if attempt and self._clock() >= deadline:
                reason = "the compile deadline passed"
                break
            request = CompletionRequest(
                messages=messages, max_tokens=policy.max_compile_tokens
            )
            try:
                reply = self.llm.complete(request)
            except LLMResponseError as e:
                raise SynthesisError(
                    "the strong model call failed", {"rounds": failed}
                ) from e
            report = _judge(reply, policy, ins, keys)
            if report.definition is not None and report.injected is not None:
                usage = getattr(self.llm, "usage", None)
                return CompileResult(
                    report.definition,
                    report.injected,
                    report,
                    attempt + 1,
                    usage() if callable(usage) else None,
                )
            failed.append(tuple(dict.fromkeys(f.code for f in report.findings)))
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
        case, punctuation and spacing (``policy.reuse_min_score`` is never
        applied to word overlap). Embedder reuse is a SIMILARITY claim: a
        cosine does not understand negation, so a goal that differs by "not"
        may be served the stored procedure. For exact reuse pass ``embed=None``
        or set ``reuse_min_score=1.0``. A hit makes no strong call. A compile that needs approval
        (policy, or any ``completion`` tool) goes to ``reviewer`` and is
        stored ``verified`` only when approved.
        """
        ins, keys = _checked(goal, inputs, return_keys)
        policy, library = self.policy, self.library
        if library is not None:
            for match in library.search(
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
                    return Procedure(record.digest, definition, record, "library")
        result = self.compile(goal, inputs=ins, return_keys=keys)
        digest = digest_of(result.definition)
        approver = None
        if needs_approval(result.data, policy):
            self._review(_approval_request(goal, keys, result, digest))
            approver = _APPROVER
        saved: LibraryRecord | None = None
        if library is not None:
            saved = library.put(
                result.definition,
                goal=goal,
                inputs_required=ins,
                return_keys=keys,
                synthesizing_model=self.model or "",
                gates_passed=GATES,
                approved_by=approver,
                lifecycle=LIFECYCLE_VERIFIED,
            )
        return Procedure(digest, result.definition, saved, "compiled")

    def _review(self, request: ProcedureApprovalRequest) -> None:
        """G7: return only when the reviewer says exactly ``True``."""
        if self.reviewer is None:
            raise SynthesisRefusedError(
                "approval is required and no reviewer is set", REASON_APPROVAL_REQUIRED
            )
        try:
            verdict = self.reviewer(request)
        except Exception as e:
            raise SynthesisRefusedError(
                "the reviewer failed", REASON_REVIEWER_ERROR
            ) from e
        if verdict is not True:
            raise SynthesisRefusedError(
                "the reviewer did not approve", REASON_APPROVAL_DENIED
            )


def _judge(
    reply: CompletionResponse,
    policy: SynthesisPolicy,
    inputs: tuple[str, ...],
    return_keys: tuple[str, ...],
) -> GateReport:
    """One reply to one report: a reply that is not a readable JSON object
    fails as one G1 finding (truncated first, never parsed), else the gates run."""
    code = (
        "reply-truncated"
        if reply.truncated
        else "reply-malformed"
        if reply.kind != "final"
        else ""
    )
    data = None if code else extract_json_from_text(reply.text or "")
    if data is None:
        code = code or "reply-not-json-object"
        return GateReport((GateFinding("G1", code, "", REPLY_FAILURES[code]),))
    return run_gates(data, policy, inputs=inputs, return_keys=return_keys)
