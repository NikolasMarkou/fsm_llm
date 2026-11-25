"""Constants of the tool layer: the subset of the agents context keys, defaults and
message templates that the registry and the tool models read, the tool statuses and the
text of one stored tool observation (``OBSERVATION_ENTRY``).

The agents classes of the same names (``fsm_llm.agents.constants``) assign every
attribute below from these classes, in their own position, so ``vars()`` of the agents
classes lists the same names as before and each value has one source at import time.
At run time the classes that share a name are linked (``MirroredConstants``, ``mirror``
below): setting or deleting a shared name on ANY of them sets or deletes it on all of
them, so a reader of any class sees an override made on any other, as when one class
held the value.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

# DECISION plan-2026-11-12T073150-d454fd62/D-045: a shared constant is mirrored on the
# classes that hold it and a change is written through to every one of them.
# Do NOT make the agents classes subclasses of these (a subclass hides the names
# from `vars()`, and three tests enumerate `vars(ContextKeys)` and `vars(Defaults)`);
# do NOT replace the copies by properties or a read-through descriptor (a class
# attribute that is not a plain value changes `vars()` and `repr` and breaks the
# "assigned from the lower class" identity); do NOT document the lower classes as the
# place to configure (BASE had one class: an override of the agents class reached every
# reader, and `monkeypatch.setattr(agents Defaults, ...)` in tests relies on that).
# The registry lives HERE, in a module dict, never as a class attribute (it would show
# in `vars()`); a class registers its copy with `mirror()` in its own constants module.
# Concurrent overrides are serialised by ONE module lock (D-053): the write to the class
# and the loop over its peers are two steps, and two threads writing two classes of a
# group could otherwise leave the copies permanently different. Do NOT drop the lock,
# and do NOT replace it by a lock per group (groups join in `mirror()`, so two locks
# could be taken in two orders).
_LOCK = threading.RLock()
_MIRRORS: dict[tuple[type, str], tuple[type, ...]] = {}


class MirroredConstants(type):
    """Metaclass of a constants class that shares attributes with another class.

    Interface contract: a class made with it behaves as a plain class (its values
    stay in ``vars()`` in definition order, ``repr`` and ``__bases__`` unchanged);
    only assignment and deletion of a name registered with :func:`mirror` also
    assign or delete that name on the other classes of the group. The write to a
    peer goes through ``type.__setattr__``, which bypasses this hook, so no write
    loops. A name that is not registered stays local. Never raises beyond what a
    plain class would (``AttributeError`` when deleting an absent name).

    Thread safety: the write to the class and the writes to its peers happen under
    one module lock, so concurrent overrides of a shared name are serialised and the
    last writer wins on every class of the group (a reader takes no lock).
    """

    def __setattr__(cls, name: str, value: Any) -> None:
        with _LOCK:
            super().__setattr__(name, value)
            for peer in _MIRRORS.get((cls, name), ()):
                type.__setattr__(peer, name, value)

    def __delattr__(cls, name: str) -> None:
        with _LOCK:
            super().__delattr__(name)
            for peer in _MIRRORS.get((cls, name), ()):
                if name in vars(peer):
                    type.__delattr__(peer, name)


def mirror(owner: type, copy: type, names: Iterable[str]) -> None:
    """Link attribute *names* of *copy* to the same names of *owner*.

    Interface contract: both classes must use :class:`MirroredConstants` (``TypeError``
    otherwise) and hold the very same object under each name when called
    (``ValueError`` otherwise: a copy that already drifted is a defect, not a link).
    The link is symmetric and joins groups: a name that *owner* already shares with a
    third class is shared by all three. Call it once per pair, at import, in the module
    that defines *copy*; ``mirror_links`` returns what is registered.
    """
    for cls in (owner, copy):
        if not isinstance(cls, MirroredConstants):
            raise TypeError(f"{cls.__qualname__} must use MirroredConstants")
    pending = tuple(names)
    with _LOCK:
        for name in pending:
            if name not in vars(owner) or vars(owner)[name] is not vars(copy).get(name):
                raise ValueError(
                    f"{copy.__qualname__}.{name} is not the object "
                    f"{owner.__qualname__}.{name}"
                )
        for name in pending:
            group = dict.fromkeys(
                [
                    owner,
                    copy,
                    *_MIRRORS.get((owner, name), ()),
                    *_MIRRORS.get((copy, name), ()),
                ]
            )
            for member in group:
                _MIRRORS[(member, name)] = tuple(c for c in group if c is not member)


def mirror_links() -> dict[tuple[type, str], tuple[type, ...]]:
    """A copy of the registry: ``(class, name)`` to the other classes sharing the name."""
    with _LOCK:
        return dict(_MIRRORS)


class ContextKeys(metaclass=MirroredConstants):
    """The context keys a tool call is read from and the sentinel for no tool."""

    # Tool selection (extracted by LLM in think state)
    TOOL_NAME = "tool_name"
    TOOL_INPUT = "tool_input"

    # Sentinel value for "no tool selected"
    NO_TOOL = "none"


class Defaults(metaclass=MirroredConstants):
    """Default configuration values of the tool layer."""

    # Largest integer an agent limit (``max_*``/``num_*``) or an artifact int
    # config field may take: 63 bits. A larger int is no real limit, and past
    # 4300 digits it breaks every f-string, log line and ``json.dumps`` of it.
    MAX_INT_LIMIT = 2**63 - 1
    MAX_OBSERVATION_LENGTH = 2000
    # Largest per-call tool ``timeout_seconds``: the most a thread wait accepts
    # (``threading.TIMEOUT_MAX``; a larger wait raises OverflowError).
    MAX_TOOL_TIMEOUT_S = threading.TIMEOUT_MAX
    # Tool-selection classification threshold (``to_classification_schema``).
    TOOL_CONFIDENCE_THRESHOLD = 0.4


class ErrorMessages(metaclass=MirroredConstants):
    """Error messages of the tool layer."""

    TOOL_NOT_FOUND = "Tool '{name}' not found in registry"
    TOOL_EXECUTION_FAILED = "Tool '{name}' execution failed: {error}"
    TOOL_NOT_LOGGED = (
        "Tool '{name}' was not run: the event log could not record the call"
    )
    TOOL_TIMED_OUT = (
        "Tool '{name}' timed out after {timeout} s; its outcome is unknown: it "
        "may still be running and its side effects may still happen. Do not "
        "assume it did not run"
    )
    TOOL_BAD_TIMEOUT = (
        "Tool '{name}' has an invalid timeout_seconds {timeout}: it must be a "
        "finite number above 0 and at most {limit} s; the tool was not run"
    )


class LogMessages(metaclass=MirroredConstants):
    """Log message templates of the tool layer."""

    TOOL_LATE_RESULT = (
        "Tool '{name}' finished after its {timeout} s timeout; "
        "its {outcome} was discarded"
    )


# One tool observation as the agent loop stores it (``handlers.record_tool_step``).
# Read back by ``BaseAgent._try_parse_structured_output`` (the text after the last
# "Result:") and sized by ``okf._budget`` and the synthesis tools' ``_room``, which
# measure the wrapper with this. Moved down from ``agents/constants/outcomes.py`` (plan
# d454fd62, step 95) so the synthesis tools read it without importing ``agents``.
OBSERVATION_ENTRY = "[Step {step}] Tool: {tool} | Input: {input} | Result: {result}"


class ToolObservationPrefix:
    """Prefixes that tell the model how a tool call ended (``ToolResult.observation``)."""

    # The call ran and failed (or never ran: unknown tool, bad arguments).
    FAILED = "[TOOL FAILED]"
    # The call timed out: it may still be running, its effects may happen.
    OUTCOME_UNKNOWN = "[TOOL OUTCOME UNKNOWN]"


class ToolRunStatus:
    """How a call that was sent to a tool ended (``ToolResult.status``).

    Written to ``ContextKeys.TOOL_STATUS`` by the executors that run a tool;
    the other statuses there (``skipped``, ``rejected``,
    ``awaiting_approval``) mean no tool ran.
    """

    SUCCESS = "success"
    FAILED = "failed"
    # The call timed out: it may still be running and may still take effect.
    # Distinct from FAILED so no reader treats it as "safe to run again".
    UNKNOWN = "unknown"
    # The three values below mean NO tool ran for the turn.
    SKIPPED = "skipped"
    REJECTED = "rejected"
    AWAITING_APPROVAL = "awaiting_approval"
    # Statuses that mean a tool really ran.
    RAN = frozenset({SUCCESS, FAILED, UNKNOWN})
