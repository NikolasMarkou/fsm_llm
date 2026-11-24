"""Privacy of the event log: two levels, one redaction hook, one allow-list (plan 5a6a1f80,
roadmap row 1.11a).

Interface contract (callers: ``EventLog``, the two folds, ``runtime.scope`` and the engine
producers through ``EventLog.stores_values``):
    - ``PrivacyLevel`` (``str, Enum``): ``VALUES`` (the default of ``EventLog(root)``: prompts,
      replies, user messages and states are blobs, as slice 1 wrote them) and ``SHAPE`` (a log
      with no value at all). ``from_string`` refuses an unknown name (``ValueError``).
    - ``SHAPE_FIELDS``: per event type, the field names a ``shape`` event may carry. Everything
      else is dropped by ``EventLog._envelope``, so a field added to a producer later fails
      closed. No name in it ends in ``_ref``, and a type with no entry carries no field.
    - ``RedactionKind`` (``str, Enum``): what a hook is handed, a closed vocabulary: ``PROMPT``,
      ``RESPONSE``, ``USER_MESSAGE``, ``STATE``, ``DELTA`` (also a pushed frame's return
      context), ``ARGUMENTS``, ``DEFINITION``.
    - ``run_redaction(hook, kind, value) -> value``: calls ``hook(kind, copy_of_value)`` and
      returns what it gave back. The hook must be PURE and DETERMINISTIC (the state digest of
      a turn compares two blobs, and a blob is deduplicated by its bytes); it gets a deep copy
      (writing into it changes nothing the engine uses) and must return a JSON value no deeper
      than ``MAX_PAYLOAD_DEPTH``. Any failure is ``EventPayloadError`` and nothing is written:
      an exception (the type name only is kept), an input nested too deep to hand over, and,
      at the one canonical encoding both consumers (``put_json`` and the inline arguments of a
      tool call) apply right after, a result that is not JSON or nested too deep. The caller
      turns that into its own refusal, so a checkpoint that cannot be redacted stops the model
      call it guards.
    - ``refuse_shape(event)``: ``LogPrivacyError`` when *event* carries the ``lvl`` marker. The
      folds call it on EVERY event they read, so a log written at ``shape``, and a ``values``
      log a ``shape`` writer later continued, are refused loudly; never ``None``.

The hook runs INSIDE the log, on the JSON-ready value, AFTER the built-in filters
(``filter_context`` and ``redact_context``: a secret-looking context entry reaches the hook
already replaced) and before canonical encoding, so what the hook returns is what is stored.
It sees blobs and the inline arguments of a tool call; ``put_blob`` (raw bytes) is refused
while a hook is set, because bytes cannot be redacted.

Honest limits. (1) At level ``values`` the log holds the text the model saw. The built-in
filters judge context ENTRIES (names and shapes), not free text: a card number typed into a
message reaches disk, and so does a secret inside a reply or a prompt, unless the hook handles
it. A free-text secret is the one thing only the user's hook can recognise: this module ships
no scanner, because a half filter gives false safety. (2) A hook that rewrites a ``state`` or
a ``definition`` blob (an id, the history) can make a later fork, resume or restore fail or
differ. (3) A ``shape`` log carries ids, names, counts, outcomes and codes. Those are not
secrets, but they say which states a conversation visited and when. Keep the directory as
private as a session directory at either level.

Imports ``runtime.constants`` and ``runtime.events`` only.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from enum import Enum
from types import MappingProxyType
from typing import Any

from fsm_llm.runtime.constants import LEVEL_KEY, MAX_PAYLOAD_DEPTH
from fsm_llm.runtime.events import (
    EventPayloadError,
    EventType,
    LogPrivacyError,
    check_depth,
)


class PrivacyLevel(str, Enum):
    """What an event log may hold (see the module docstring)."""

    SHAPE = "shape"
    VALUES = "values"

    @classmethod
    def from_string(cls, value: PrivacyLevel | str) -> PrivacyLevel:
        """Return the member for a member or an exact value string (``ValueError`` else)."""
        try:
            return cls(value)
        except ValueError:
            valid_values = [member.value for member in cls]
            raise ValueError(
                f"Invalid privacy level: {value!r}. Must be one of {valid_values}"
            ) from None


class RedactionKind(str, Enum):
    """The first argument of a redaction hook: what kind of value it is handed."""

    PROMPT = "prompt"
    RESPONSE = "response"
    USER_MESSAGE = "user_message"
    STATE = "state"
    DELTA = "delta"
    ARGUMENTS = "arguments"
    DEFINITION = "definition"

    @classmethod
    def from_string(cls, value: RedactionKind | str) -> RedactionKind:
        """Return the member for a member or an exact value string (``ValueError`` else)."""
        try:
            return cls(value)
        except ValueError:
            valid_values = [member.value for member in cls]
            raise ValueError(
                f"Invalid redaction kind: {value!r}. Must be one of {valid_values}"
            ) from None


#: The hook of ``EventLog(redact=)``: ``(kind, value) -> value``.
RedactionHook = Callable[[RedactionKind, Any], Any]

_PROVIDER_CALL = ("source", "call_type", "stream")

#: The fields a ``shape`` event may carry (see the module docstring). Names, ids, counts,
#: codes and flags: no reference, no text, no value. The produced types are listed; a type
#: absent here carries no field at all.
SHAPE_FIELDS: Mapping[EventType, frozenset[str]] = MappingProxyType(
    {
        EventType.TURN_START: frozenset({"state", "has_user_message"}),
        EventType.TURN_END: frozenset({"outcome"}),
        EventType.LLM_REQUEST: frozenset({*_PROVIDER_CALL, "model"}),
        EventType.LLM_RESPONSE: frozenset(
            {
                *_PROVIDER_CALL,
                "request_seq",
                "finish_reason",
                "abandoned",
                "model",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            }
        ),
        EventType.PROVIDER_ERROR: frozenset(
            {*_PROVIDER_CALL, "request_seq", "error_type", "status_code"}
        ),
        EventType.TOOL_CALL: frozenset(
            {"name", "call_id", "side_effecting", "gated", "argument_keys"}
        ),
        EventType.TOOL_RESULT: frozenset(
            {
                "name",
                "call_id",
                "status",
                "error_type",
                "error_code",
                "truncated",
                "elapsed_ms",
                "call_frame",
            }
        ),
        EventType.APPROVAL_REQUEST: frozenset(
            {"tool_name", "has_callback", "argument_keys"}
        ),
        EventType.APPROVAL_OUTCOME: frozenset(
            {"request_seq", "approved", "reason", "error_type"}
        ),
        EventType.CONTEXT_RECORD: frozenset(
            {
                "source",
                "state",
                "keys",
                "restored_from",
                "forked_from",
                "resumed_from",
            }
        ),
        EventType.STACK_PUSH: frozenset(
            {
                "child_conversation_id",
                "fsm_id",
                "depth",
                "shared_context_keys",
                "preserve_history",
                "restored",
                "max_steps",
                "max_seconds",
            }
        ),
        EventType.STACK_POP: frozenset(
            {
                "child_conversation_id",
                "fsm_id",
                "merge_strategy",
                "merged_key_count",
                "depth",
            }
        ),
        EventType.EXECUTOR_START: frozenset({"kind", "state"}),
        EventType.EXECUTOR_RESULT: frozenset(
            {"kind", "state", "extracted_keys", "extraction_failed"}
        ),
        EventType.TRANSITION: frozenset(
            {
                "from_state",
                "to_state",
                "outcome",
                "executed",
                "priority",
                "candidates",
            }
        ),
        EventType.DECISION: frozenset({"kind", "budget", "limit", "steps_done"}),
        EventType.RUN_START: frozenset(
            {
                "handle",
                "target",
                "template_identity",
                "instance_identity",
                "source",
                "parent",
            }
        ),
        EventType.RUN_END: frozenset(
            {
                "handle",
                "target",
                "status",
                "reason",
                "template_identity",
                "instance_identity",
            }
        ),
    }
)


def shape_only(event_type: EventType, fields: Mapping[str, Any]) -> dict[str, Any]:
    """Return *fields* cut to the allow-list of *event_type* (empty for an unlisted type)."""
    allowed = SHAPE_FIELDS.get(event_type, frozenset())
    return {name: value for name, value in fields.items() if name in allowed}


def refuse_shape(event: Mapping[str, Any]) -> None:
    """Raise ``LogPrivacyError`` when *event* was written at a level without values."""
    if LEVEL_KEY in event:
        raise LogPrivacyError(
            f"event {event.get('seq')} of this log was written at privacy level "
            f"{event[LEVEL_KEY]!r:.20}: a log without values holds no state, so it cannot "
            "be restored, forked or resumed"
        )


def run_redaction(hook: RedactionHook, kind: RedactionKind | str, value: Any) -> Any:
    """Return what *hook* makes of *value* (see the module docstring); ``EventPayloadError`` else."""
    try:
        member = RedactionKind.from_string(kind)
    except ValueError as exc:
        raise EventPayloadError(f"a redaction hook is set: {exc}") from None
    if not check_depth(value, MAX_PAYLOAD_DEPTH):
        raise EventPayloadError(
            f"a {member.value} value nests too deep for the redaction hook (or holds a "
            "cycle); nothing was written"
        )
    try:
        result = hook(member, copy.deepcopy(value))
    except Exception as exc:
        raise EventPayloadError(
            f"the redaction hook failed on a {member.value} value "
            f"({type(exc).__name__}); nothing was written"
        ) from None
    return result
