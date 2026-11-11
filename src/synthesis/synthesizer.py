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

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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
    LLMCallCounts,
    LLMResponseError,
    LLMUsage,
    State,
    SynthesizeConfig,
    logic_referenced_keys,
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
    CODE_SIMULATION_INCOMPLETE,
    GATE_SIMULATION,
    LIFECYCLE_VERIFIED,
    REASON_APPROVAL_DENIED,
    REASON_APPROVAL_REQUIRED,
    REASON_REVIEWER_ERROR,
)
from .exceptions import SynthesisError, SynthesisRefusedError
from .gates import (
    GATES,
    GateFinding,
    GateReport,
    run_gates,
    safe_text,
    simulated,
    verify_stored,
)
from .library import FSMLibrary, LibraryRecord, digest_of
from .policy import SynthesisPolicy, needs_approval
from .prompts import KEY_NAMING, REPLY_FAILURES, compile_messages, repair_message
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


def _approval_request(
    goal: str, keys: tuple[str, ...], result: CompileResult, digest: str
) -> ProcedureApprovalRequest:
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
        ins, keys = _checked(goal, inputs, return_keys)
        policy = self.policy
        if policy.require_simulation and self.simulator is None:
            raise SynthesisError(
                "a simulation is required and no simulator is set",
                {"rounds": [(CODE_SIMULATION_INCOMPLETE,)]},
            )
        messages = compile_messages(goal, ins, keys, policy)
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
                reply = self.llm.complete(request)
            except LLMResponseError as e:
                raise SynthesisError(
                    "the strong model call failed", {"rounds": failed}
                ) from e
            report = _judge(reply, policy, ins, keys)
            simulation = None
            if report.definition is not None and report.injected is not None:
                if self.simulator is not None:
                    runs += 1
                    simulation = self.simulator.run(
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
        """
        ins, keys = _checked(goal, inputs, return_keys)
        policy, library = self.policy, self.library
        started, before = self._clock(), _meter(self.llm)
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
        result = self.compile(goal, inputs=ins, return_keys=keys)
        digest = digest_of(result.definition)
        approver = self.approval(goal, keys, result)
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

    def approval(
        self, goal: str, return_keys: tuple[str, ...], result: CompileResult
    ) -> str | None:
        """G7 for a gate-passed ``result``: ``None`` when neither the policy nor
        a ``completion`` tool needs approval, the approver label when the
        reviewer said exactly ``True``, else ``SynthesisRefusedError``."""
        if not needs_approval(result.data, self.policy):
            return None
        request = _approval_request(
            goal, return_keys, result, digest_of(result.definition)
        )
        self._review(request)
        return _APPROVER

    def _since(self, started: float) -> float:
        return round(self._clock() - started, 3)

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


#: The stdlib json messages that carry no input text; any other is dropped.
_PARSER_MESSAGES = frozenset(
    (
        "Expecting value|Expecting ',' delimiter|Expecting ':' delimiter|"
        "Expecting property name enclosed in double quotes|Extra data|"
        "Unterminated string starting at|Invalid control character at|"
        "Invalid \\escape|Invalid \\uXXXX escape"
    ).split("|")
)


def _parser_detail(text: str) -> str:
    """`` (parser: <msg> at line L column C); fix that syntax`` when the reply's
    first-``{``-to-last-``}`` span fails ``json.loads``, else ``""``: a
    ``_PARSER_MESSAGES`` phrase and two integers, never reply text."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return ""
    try:
        json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        if e.msg in _PARSER_MESSAGES:
            return f" (parser: {e.msg} at line {e.lineno} column {e.colno}); fix that syntax"
    except RecursionError:
        pass
    return ""


# DECISION plan-2026-11-10T194139-2cefeb61/D-097
# The strong model's habit (live: 8 first-trial compiles) is to close the
# `states` object with `]`. The slip is named exactly, so it is repaired here,
# once, instead of asked again at about 70 s a round. ONLY the kind of a wrong
# closer changes: a missing closer is NOT appended and a stray one is not
# dropped (no invented content), and the repaired text goes through every gate
# like any reply. Do NOT repair a reply that already parses or one that was cut
# off, do NOT skip G1 for a repaired reply, and do NOT make the scan ignore
# strings (a `]` inside a string would flip the stack).
_CLOSERS = {"{": "}", "[": "]"}
_MAX_NESTING = 100


def _repair_brackets(text: str) -> str | None:
    """The first-``{``-to-last-``}`` span of ``text`` with every closer of the
    wrong kind replaced, or ``None``.

    Interface contract (caller: ``_judge``, after ``extract_json_from_text``
    found nothing): ``text`` is a reply, any shape. Returns the span only when
    it parses with ``json.loads`` after the swaps; ``None`` for a span that is
    missing, has a stray closer, an unclosed container or string, nests deeper
    than ``_MAX_NESTING``, or still is not JSON. Pure; one linear scan, never
    raises.
    """
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    chars = list(text[start : end + 1])
    expected: list[str] = []
    quoted = escaped = False
    for i, ch in enumerate(chars):
        if quoted:
            quoted = ch != '"' or escaped
            escaped = ch == "\\" and not escaped
        elif ch == '"':
            quoted = True
        elif ch in _CLOSERS:
            if len(expected) >= _MAX_NESTING:
                return None
            expected.append(_CLOSERS[ch])
        elif ch in "}]":
            if not expected:
                return None
            chars[i] = expected.pop()
    if expected or quoted:
        return None
    fixed = "".join(chars)
    try:
        json.loads(fixed)
    except (json.JSONDecodeError, RecursionError):
        return None
    return fixed


# DECISION plan-2026-11-10T194139-2cefeb61/D-096
# The bulk pass is the only channel for a later-turn correction (per-field
# extraction is skip-if-set) and answers under the name its instructions give:
# a compile that gates on a key it extracts, without naming that key at state
# level, can never correct a refused value (live: `group_size` for `party_size`).
# So the sentence is added HERE, on the parsed reply before `run_gates`, and the
# stored definition is the normalised one. Do NOT change `logic`, do NOT add a
# field extraction, do NOT run this after the gates, do NOT replace the model's
# text, do NOT test "mentions" as a substring, and do NOT exceed the policy's text
# cap (G5 `text-too-long` would reject a compile that passed before).
_INSTRUCTION_CAP = next(
    m.max_length
    for m in State.model_fields["extraction_instructions"].metadata
    if hasattr(m, "max_length")
)


def _naming_text(state: Any, limit: int) -> str | None:
    """The state-level extraction text ``state`` should have, ``None`` to leave it."""
    if (
        not isinstance(state, dict)
        or state.get("completion")
        or state.get("synthesize")
    ):
        return None
    text = state.get("extraction_instructions") or ""
    fields, edges = state.get("field_extractions"), state.get("transitions")
    if not (
        isinstance(text, str) and isinstance(fields, list) and isinstance(edges, list)
    ):
        return None
    read: set[str] = set()
    for edge in edges:
        conds = edge.get("conditions") if isinstance(edge, dict) else None
        for cond in conds if isinstance(conds, list) else ():
            try:
                read |= logic_referenced_keys(
                    cond.get("logic") if isinstance(cond, dict) else None
                )
            except ValueError:
                pass  # malformed logic: G1 reports it
    names = dict.fromkeys(
        f["field_name"]
        for f in fields
        if isinstance(f, dict) and f.get("field_name") in read
    )
    missing = [
        k
        for k in names
        if not re.search(rf"(?<!\w){re.escape(k)}(?!\w)", text, re.IGNORECASE)
    ]
    if not missing:
        return None
    added = " ".join(KEY_NAMING.format(key=k) for k in missing)
    out = f"{text} {added}" if text.strip() else added
    return out if len(out) <= limit else None


def normalise_extraction(data: Any, max_chars: int) -> Any:
    """Name, in the state-level ``extraction_instructions``, every key a state
    extracts and gates on (D-096).

    Interface contract (caller: ``_judge``, once per compile reply):
        data: the parsed reply, any shape (never mutated). ``max_chars`` is the
        policy's text cap. A state with ``field_extractions`` and an edge whose
        ``logic`` reads such a key gets one ``KEY_NAMING`` sentence per key its
        text does not already name (whole identifier, any case), appended after
        the model's own text; a state with no such gate, a terminal,
        ``completion`` or ``synthesize`` state, a classification key, a result
        over the cap or any malformed part is left as it is. Returns ``data``
        itself when nothing changes (so its digest is the same), else a shallow
        copy with only the changed states replaced. Pure, idempotent, never raises.
    """
    states = data.get("states") if isinstance(data, dict) else None
    if not isinstance(states, dict):
        return data
    limit = min(max_chars, _INSTRUCTION_CAP)
    fixed = {
        sid: {**state, "extraction_instructions": text}
        for sid, state in states.items()
        if (text := _naming_text(state, limit)) is not None
    }
    return {**data, "states": {**states, **fixed}} if fixed else data


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
    if data is None and not code:
        fixed = _repair_brackets(reply.text or "")
        data = json.loads(fixed) if fixed else None
    if data is None:
        detail = "" if code else _parser_detail(reply.text or "")
        code = code or "reply-not-json-object"
        return GateReport((GateFinding("G1", code, "", REPLY_FAILURES[code] + detail),))
    data = normalise_extraction(data, policy.max_text_chars)
    return run_gates(data, policy, inputs=inputs, return_keys=return_keys)
