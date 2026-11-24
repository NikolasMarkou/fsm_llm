"""The shared vocabulary of run statuses, reason codes and completion kinds.

Eight closed vocabularies, each a ``str`` enum whose values are the strings
already on the wire (or, for ``Lifecycle`` and ``MessageSource``, the strings of
the envelope and the event log), so a member and its string are interchangeable
for ``==``, hashing, ``in``, ``json.dumps`` and pydantic:

- ``Status``: how a run or a step ended (``ok``, ``failed``, ``aborted``,
  ``running``, ``abstained``).
- ``Reason``: why, when a status needs one. The 15 reason codes of
  ``fsm_llm.synthesis`` in their pinned order, plus ``needs_user_input`` and
  ``cancelled`` (a run stopped from outside).
- ``CompletionKind``: the ``kind`` of a ``CompletionResponse`` (``calls``,
  ``final``, ``malformed``).
- ``Lifecycle``: the lifecycle claim of an executable or a library procedure
  (``draft``, ``verified``, ``trusted``, ``deprecated``, ``builtin``). A claim,
  never authorization.
- ``MessageSource``: who sent a message or an input (``user``, ``run``,
  ``remote_client``, ``simulator``). ``run/start`` and ``turn/start`` write it;
  ``message_source`` is the scope a caller sets around its ``converse`` call.
- ``ToolErrorCode``, ``Purpose`` and ``LlmFailureCode``: why a tool call or a model
  call failed, and what a model call was for (event log rows 1.15 and 1.16).

Three small pure mappers translate the agents' own vocabularies into it.
They take plain strings or a bool and import nothing, so this module has no
dependency on agents or synthesis. The modules that own those vocabularies
keep their strings; a test (``tests/test_fsm_llm/test_codes.py``) reads them
and fails when one drifts from the vocabulary here.

Do not format a member with an f-string or ``str()``: that differs between
Python versions for a ``str`` mixin enum. Use ``.value``.

Imported by name (``from fsm_llm.core.codes import Status``); ``fsm_llm/__init__.py``
does not import it. Standard library only.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import TypeVar

__all__ = [
    "CompletionKind",
    "Lifecycle",
    "LlmFailureCode",
    "MessageSource",
    "Purpose",
    "Reason",
    "Status",
    "ToolErrorCode",
    "approval_code",
    "current_message_source",
    "message_source",
    "status_of_stop_reason",
    "status_of_tool_status",
]

_E = TypeVar("_E", bound=Enum)


def _parse(cls: type[_E], value: object, name: str) -> _E:
    """Return the member of ``cls`` for a member or an exact value string.

    Raises ``ValueError`` naming ``name`` and the valid values for anything
    else (another string, ``None``, a number).
    """
    try:
        return cls(value)
    except ValueError:
        valid_values = [member.value for member in cls]
        raise ValueError(
            f"Invalid {name}: {value!r}. Must be one of {valid_values}"
        ) from None


class Status(str, Enum):
    """How a run, a step or a tool call ended.

    ``ABSTAINED`` is a deliberate non-action (a skipped tool call); it is not a
    failure. ``RUNNING`` means no outcome yet.
    """

    OK = "ok"
    FAILED = "failed"
    ABORTED = "aborted"
    RUNNING = "running"
    ABSTAINED = "abstained"

    @classmethod
    def from_string(cls, value: Status | str) -> Status:
        """The member for a member or an exact value; ``ValueError`` otherwise."""
        return _parse(cls, value, "status")


class Reason(str, Enum):
    """Why a status is not ``ok``: the fixed reason codes.

    The first 15 members equal ``fsm_llm.synthesis.constants.REASONS`` in
    order. ``NEEDS_USER_INPUT`` is the 16th: ``synthesis.tools`` writes it but
    it is not in the pinned ``REASONS`` tuple (a bench manifest pins that
    file), so a test asserts the difference rather than hiding it.
    ``CANCELLED`` is the 17th: a run ended from outside (``aborted`` with this
    reason); no synthesis string names it. ``UNKNOWN_TARGET`` is the 18th: a ``call``
    state named a target that is not resolvable or not in its closed list (``failed``
    with this reason); no synthesis string names it either. ``RUN_LOST`` is the 19th: a
    delegated ``call`` whose run the host never saw (a crash, a snapshot restore, a fork
    inside the running window) ends ``failed`` with it, and nothing is re-run.
    """

    INPUTS_MISSING = "inputs_missing"
    RETURN_KEYS_MISSING = "return_keys_missing"
    PROVIDER_ERROR = "provider_error"
    LIBRARY_ERROR = "library_error"
    COMPILE_FAILED = "compile_failed"
    PUSH_FAILED = "push_failed"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_DENIED = "approval_denied"
    REVIEWER_ERROR = "reviewer_error"
    DEPTH_REFUSED = "depth_refused"
    COMPLETION_UNSUPPORTED = "completion_unsupported"
    LIFECYCLE_TOO_LOW = "lifecycle_too_low"
    NOT_SIMULATED = "not_simulated"
    BUDGET_SPENT = "budget_spent"
    UNEXPECTED_ERROR = "unexpected_error"
    NEEDS_USER_INPUT = "needs_user_input"
    CANCELLED = "cancelled"
    UNKNOWN_TARGET = "unknown_target"
    RUN_LOST = "run_lost"

    @classmethod
    def from_string(cls, value: Reason | str) -> Reason:
        """The member for a member or an exact value; ``ValueError`` otherwise."""
        return _parse(cls, value, "reason")


class CompletionKind(str, Enum):
    """The ``kind`` of a ``CompletionResponse``, and of a completion-state result."""

    CALLS = "calls"
    FINAL = "final"
    MALFORMED = "malformed"

    @classmethod
    def from_string(cls, value: CompletionKind | str) -> CompletionKind:
        """The member for a member or an exact value; ``ValueError`` otherwise."""
        return _parse(cls, value, "completion kind")


class Lifecycle(str, Enum):
    """The lifecycle claim of an executable or a library procedure.

    The first four members equal ``fsm_llm.synthesis.constants.LIFECYCLES`` in
    order (``draft`` to ``deprecated``); ``BUILTIN`` is the fifth: a built-in
    executable that ships in code. It is a claim of its source, never an
    authorization.
    """

    DRAFT = "draft"
    VERIFIED = "verified"
    TRUSTED = "trusted"
    DEPRECATED = "deprecated"
    BUILTIN = "builtin"

    @classmethod
    def from_string(cls, value: Lifecycle | str) -> Lifecycle:
        """The member for a member or an exact value; ``ValueError`` otherwise."""
        return _parse(cls, value, "lifecycle")


class MessageSource(str, Enum):
    """Who sent a message or an input to a run (roadmap C4 and 1.17).

    ``run/start`` records it (``run``) and so does ``turn/start`` of a turn with a user
    message (``message_source``; ``user`` when no scope is set). ``RUN`` is a parent run
    or a workflow step, ``REMOTE_CLIENT`` a caller over a protocol, ``SIMULATOR`` the
    user simulator. A record only: nothing decides on it.
    """

    USER = "user"
    RUN = "run"
    REMOTE_CLIENT = "remote_client"
    SIMULATOR = "simulator"

    @classmethod
    def from_string(cls, value: MessageSource | str) -> MessageSource:
        """The member for a member or an exact value; ``ValueError`` otherwise."""
        return _parse(cls, value, "message source")


# DECISION plan-2026-11-24T193347-3f4530d3/D-012
# The source of a message is a record on `turn/start`, never an input of a decision. Its one
# reader is `engine/events.begin_turn`; do NOT read it in `policy/`, `handlers/`, the HITL gate or
# a prompt (an approval is never selected by source, and no request byte may carry it), do NOT
# hold the scope across a `yield`, and do NOT move it to `runtime` (`testing/` may import only
# core). See decisions.md D-012.
_MESSAGE_SOURCE: ContextVar[MessageSource] = ContextVar(
    "fsm_llm_message_source", default=MessageSource.USER
)


@contextmanager
def message_source(source: MessageSource | str) -> Iterator[MessageSource]:
    """Label the user messages of the ``with`` body (wrap one ``converse`` call, never a ``yield``).

    Interface contract: *source* is a ``MessageSource`` or its exact value (``ValueError``
    otherwise); the scope resets on exit, an error included. Callers: the simulators, the
    monitor send route, the workflow conversation step.
    """
    token = _MESSAGE_SOURCE.set(MessageSource.from_string(source))
    try:
        yield _MESSAGE_SOURCE.get()
    finally:
        _MESSAGE_SOURCE.reset(token)


def current_message_source() -> MessageSource:
    """The source in this context, ``MessageSource.USER`` when no scope is set."""
    return _MESSAGE_SOURCE.get()


class ToolErrorCode(str, Enum):
    """Why a tool call did not return a value (roadmap row 1.16).

    Set where the cause is known, never by matching message text:
    ``UNKNOWN_TOOL`` (the name is not registered), ``INVALID_ARGS`` (the arguments
    did not bind to the function), ``EXECUTION_FAILED`` (the body raised, or the
    call could not be logged), ``TIMEOUT`` (the call stopped waiting; the outcome
    is unknown, never failed). ``DENIED`` and ``SANDBOX_VIOLATION`` are members
    with no producer yet.
    """

    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGS = "invalid_args"
    EXECUTION_FAILED = "execution_failed"
    TIMEOUT = "timeout"
    DENIED = "denied"
    SANDBOX_VIOLATION = "sandbox_violation"

    @classmethod
    def from_string(cls, value: ToolErrorCode | str) -> ToolErrorCode:
        """The member for a member or an exact value; ``ValueError`` otherwise."""
        return _parse(cls, value, "tool error code")


class Purpose(str, Enum):
    """Why a model call was made (roadmap row 1.15): the label of an LLM event.

    Written on ``llm/request``, ``llm/response`` and ``provider/error`` from a ContextVar
    scope (``fsm_llm.llm.purpose``), never from a request field: a purpose moves no request
    byte. ``TIE_BREAK`` is the classifier call that settles an AMBIGUOUS transition;
    ``CLASSIFY`` the one of ``classification_extractions``. Embedding is not logged.
    """

    EXTRACT = "extract"
    CLASSIFY = "classify"
    TIE_BREAK = "tie_break"
    REPLY = "reply"
    COMPLETE = "complete"
    COMPILE = "compile"
    SIMULATE = "simulate"
    JUDGE = "judge"
    SUMMARIZE = "summarize"

    @classmethod
    def from_string(cls, value: Purpose | str) -> Purpose:
        """The member for a member or an exact value; ``ValueError`` otherwise."""
        return _parse(cls, value, "purpose")


class LlmFailureCode(str, Enum):
    """Why a model call failed (roadmap row 1.15): the ``failure_code`` of ``provider/error``.

    Set from the exception class, never from message text, except the one existing
    garbled-tool-call marker (``MALFORMED_RESPONSE``). ``HTTP_ERROR`` is any other HTTP
    status of 400 or more (the number is the event's ``status_code``), ``CANCELLED`` a call
    cancelled by its caller, ``UNKNOWN`` the fail-closed fallback.
    """

    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    AUTH = "auth"
    CONTEXT_OVERFLOW = "context_overflow"
    CONTENT_REFUSED = "content_refused"
    MALFORMED_RESPONSE = "malformed_response"
    TRANSPORT = "transport"
    HTTP_ERROR = "http_error"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: LlmFailureCode | str) -> LlmFailureCode:
        """The member for a member or an exact value; ``ValueError`` otherwise."""
        return _parse(cls, value, "llm failure code")


_STOP_REASON_CODES: dict[str, tuple[Status, Reason | None]] = {
    "answered": (Status.OK, None),
    "evidence": (Status.OK, None),
    "max_iterations": (Status.ABORTED, Reason.BUDGET_SPENT),
    "ended": (Status.ABORTED, None),
    "forced_pass": (Status.FAILED, None),
    "stalled": (Status.FAILED, None),
    "verification_failed": (Status.FAILED, None),
    "no_result": (Status.FAILED, None),
    "gate_failed": (Status.FAILED, None),
    "truncated": (Status.FAILED, None),
}

_TOOL_STATUS_CODES: dict[str, tuple[Status, Reason | None]] = {
    "success": (Status.OK, None),
    "failed": (Status.FAILED, None),
    "skipped": (Status.ABSTAINED, None),
    "rejected": (Status.FAILED, Reason.APPROVAL_DENIED),
    "awaiting_approval": (Status.RUNNING, Reason.APPROVAL_REQUIRED),
}


def status_of_stop_reason(stop_reason: str) -> tuple[Status, Reason | None]:
    """Map an agent ``AgentResult.stop_reason`` to a status and a reason.

    Lossy: only a spent ``max_iterations`` budget gets a reason
    (``budget_spent``), and the kind of budget stays in ``stop_reason``. The
    other forced stops (``forced_pass``, ``stalled``, ``verification_failed``,
    ``no_result``, ``gate_failed``, ``truncated``) map to ``failed`` with no
    reason, and ``ended`` (closed from outside) to ``aborted`` with none;
    ``stop_reason`` stays the wire field and carries the detail. The
    ``aborted`` and ``failed`` choices are a judgement a later plan may
    revise. Raises ``ValueError`` for a value it does not know.
    """
    try:
        return _STOP_REASON_CODES[stop_reason]
    except (KeyError, TypeError):
        raise ValueError(
            f"Invalid stop_reason: {stop_reason!r}. "
            f"Must be one of {list(_STOP_REASON_CODES)}"
        ) from None


def status_of_tool_status(tool_status: str) -> tuple[Status, Reason | None]:
    """Map a ``ToolResult.status`` to a status and a reason.

    ``rejected`` is ``failed`` with ``approval_denied``; ``awaiting_approval``
    is ``running`` with ``approval_required``; ``skipped`` is ``abstained``.

    ``unknown`` (a timed-out call) is deliberately NOT mapped: it raises
    ``ValueError``. The call may still be running and may still take effect,
    so an unknown outcome must not be reported as failed (``failed`` reads as
    safe to run again; the D-043 rule at ``agents/handlers/executor.py``). Any other
    unrecognised value raises too.
    """
    if tool_status == "unknown":
        raise ValueError(
            "tool_status 'unknown' has no status: an unknown outcome must not "
            "be reported as failed (the call may still be running or may "
            "still take effect)"
        )
    try:
        return _TOOL_STATUS_CODES[tool_status]
    except (KeyError, TypeError):
        raise ValueError(
            f"Invalid tool_status: {tool_status!r}. "
            f"Must be one of {[*_TOOL_STATUS_CODES, 'unknown']}"
        ) from None


def approval_code(
    approved: bool | None, *, errored: bool = False
) -> tuple[Status, Reason | None]:
    """Map an approval outcome to a status and a reason.

    ``True`` is ``ok``; ``False`` is ``failed`` with ``approval_denied``;
    ``None`` (no approver to ask) is ``failed`` with ``approval_required``;
    ``errored=True`` (the reviewer raised) is ``failed`` with
    ``reviewer_error`` whatever ``approved`` says. Lossy: an approval timeout
    is folded into denied, as ``policy/hitl.py`` does today, and no
    ``approval_timeout`` reason exists.
    """
    if errored:
        return Status.FAILED, Reason.REVIEWER_ERROR
    if approved is None:
        return Status.FAILED, Reason.APPROVAL_REQUIRED
    if approved:
        return Status.OK, None
    return Status.FAILED, Reason.APPROVAL_DENIED
