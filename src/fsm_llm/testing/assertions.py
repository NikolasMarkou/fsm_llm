"""
Assertions for conversations and run results, and an opt-in path recorder.

Four plain functions say what a test expects of a driven conversation in one
line and fail with the first divergence in the message:

- ``assert_state(source, state_id)``: the conversation is in that state now.
- ``assert_context(source, **expected)``: those context keys hold those values.
- ``assert_path(source, expected, *, exact=True)``: the states visited.
- ``assert_result(result, *, kind=None, status=None, reason=...)``: a completion
  state's ``{kind, text, calls}`` or a synthesis ``{status, procedure_id,
  version, reason}`` result.

``source`` is a ``ConversationRun`` (from ``ScriptedUser.run`` or
``advance_until_terminal``) or an ``(api, conversation_id)`` pair. The functions
raise ``AssertionError`` explicitly, never through an ``assert`` statement, so
``python -O`` does not turn them into no-ops, and set ``__tracebackhide__`` so
pytest shows the failing line of the test. They do not import pytest and need no
``pytest.register_assert_rewrite``: every message is built by hand.

``PathRecorder`` is a handler that records the states a conversation visits, for
conversations driven by code under test (an agent, a runner) where no
``ConversationRun`` exists. It is opt-in because registering any handler changes
how the pipeline copies the context (see its docstring).

Expected codes pass through ``fsm_llm.core.codes`` (``CompletionKind``, ``Status``,
``Reason`` ``from_string``), so a typo raises ``ValueError`` listing the
vocabulary instead of failing a comparison that could never pass.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any

from fsm_llm.core.codes import CompletionKind, Reason, Status
from fsm_llm.core.definitions import AdvanceResult, CompletionResponse, FSMError
from fsm_llm.engine.api.conversation import API
from fsm_llm.handlers import BaseHandler, HandlerTiming

from .user import ConversationRun

_MAX_SHOWN = 200  # characters of one value or path in a failure message


class _Unset:
    """The type of ``_UNSET``: "this expectation was not given" (``None`` is a value)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<unset>"


#: Default of ``assert_result(reason=)``: not checked. ``reason=None`` means
#: "expects no reason", which is a different claim, so ``None`` cannot be the default.
_UNSET = _Unset()


def _short(text: str) -> str:
    """``text`` cut to ``_MAX_SHOWN`` characters, so one huge value cannot flood a report."""
    if len(text) <= _MAX_SHOWN:
        return text
    return text[: _MAX_SHOWN - 3] + "..."


def _shown(value: object) -> str:
    return _short(repr(value))


def _conversation_of(source: object, who: str) -> tuple[API, str]:
    """The ``(api, conversation_id)`` a ``source`` names; ``TypeError`` for anything else."""
    if isinstance(source, ConversationRun):
        return source.api, source.conversation_id
    if (
        isinstance(source, tuple)
        and len(source) == 2
        and isinstance(source[0], API)
        and isinstance(source[1], str)
    ):
        return source[0], source[1]
    raise TypeError(
        f"{who} reads a ConversationRun or an (api, conversation_id) pair, "
        f"got {type(source).__name__}"
    )


def _read_live(who: str, source: object, what: str) -> Any:
    """Read ``what`` ("state" or "data") of the conversation ``source`` names.

    A conversation the API cannot read (never started on this API, or dropped
    from its ended-conversation cache) fails the assertion with one clear line.
    """
    __tracebackhide__ = True
    api, conversation_id = _conversation_of(source, who)
    try:
        if what == "state":
            return api.get_current_state(conversation_id)
        return api.get_data(conversation_id)
    except (ValueError, FSMError) as error:
        raise AssertionError(
            f"{who}: cannot read the {what} of conversation {conversation_id!r}: "
            f"{error}. The API answers a conversation closed with end_conversation "
            f"from the snapshot taken at the close, and has none for a conversation "
            f"it never started or whose snapshot it dropped; "
            f"ConversationRun.states and .final_state keep the recorded path"
        ) from error


def assert_state(source: ConversationRun | tuple[API, str], /, state_id: str) -> None:
    """Assert the conversation is in state ``state_id`` right now.

    Interface contract:
        - ``source``: a ``ConversationRun`` or an ``(api, conversation_id)``
          pair. The state is read live from the API (``api.get_current_state``),
          so it is the state at the moment of the call, not the end of the run:
          a test that kept talking after ``ScriptedUser.run`` asserts the new
          state. With a pushed FSM it is the state of the top of the stack.
        - A conversation closed with ``end_conversation`` (``close=True``) is
          read from the API's snapshot taken at the close; for one the API never
          started, or whose snapshot it dropped (its cache keeps 10,000), the
          assertion fails with a line saying so.
        - Raises ``AssertionError`` (``state: expected 'x', got 'y'``, with the
          recorded path when ``source`` is a ``ConversationRun``); ``TypeError``
          for a bad ``source`` or a non-string ``state_id``.
    """
    __tracebackhide__ = True
    if not isinstance(state_id, str):
        raise TypeError(f"state_id is a string, got {type(state_id).__name__}")
    actual = _read_live("assert_state", source, "state")
    if actual != state_id:
        message = f"state: expected {state_id!r}, got {actual!r}"
        if isinstance(source, ConversationRun):
            message += f" (recorded path: {_short(' -> '.join(source.states))})"
        raise AssertionError(message)


def assert_context(
    source: ConversationRun | tuple[API, str], /, **expected: Any
) -> None:
    """Assert named context keys hold the given values; other keys are ignored.

    Interface contract:
        - ``source``: as for ``assert_state``; the data is ``api.get_data(...)``
          read live, so internal keys (a leading ``_`` and the other reserved
          prefixes) are not visible and cannot be asserted.
        - ``expected``: key names and values; a key is compared with ``==``.
          This is a subset check. To demand the exact key set, compare
          ``api.get_data(conversation_id)`` with a dict yourself.
        - Checks the keys in the order given and reports the first divergence:
          ``context: key 'x' is missing (keys: [...])`` or
          ``context['x']: expected 1, got 2`` (long values are cut).
        - ``ValueError`` when no key is named (it would assert nothing);
          ``TypeError`` for a bad ``source``; the conversation-unreadable
          failure is the one of ``assert_state``.
    """
    __tracebackhide__ = True
    if not expected:
        raise ValueError("assert_context needs at least one key=value to check")
    data = _read_live("assert_context", source, "data")
    for key, wanted in expected.items():
        if key not in data:
            raise AssertionError(
                f"context: key {key!r} is missing (keys: {sorted(data)})"
            )
        actual = data[key]
        if actual != wanted:
            raise AssertionError(
                f"context[{key!r}]: expected {_shown(wanted)}, got {_shown(actual)}"
            )


def _path_of(source: object) -> tuple[str, ...]:
    """The visited path a ``source`` holds: a run, state ids, or ``AdvanceResult`` steps."""
    if isinstance(source, ConversationRun):
        return source.states
    if isinstance(source, (str, bytes)) or not isinstance(source, Sequence):
        raise TypeError(
            "assert_path reads a ConversationRun, a sequence of state ids or a "
            f"sequence of AdvanceResult, got {type(source).__name__}"
        )
    items = tuple(source)
    if all(isinstance(item, str) for item in items):
        return items  # also the empty path
    if all(isinstance(item, AdvanceResult) for item in items):
        steps: tuple[AdvanceResult, ...] = items
        return (steps[0].state_before, *(step.state_after for step in steps))
    raise TypeError(
        "assert_path: a sequence holds state ids (str) or AdvanceResult, not a mix "
        "or anything else"
    )


def _arrow(path: Sequence[str]) -> str:
    return _short(" -> ".join(path)) if path else "(empty)"


def assert_path(
    source: ConversationRun | Sequence[str] | Sequence[AdvanceResult],
    expected: Sequence[str],
    /,
    *,
    exact: bool = True,
) -> None:
    """Assert the states a conversation visited, in order.

    Interface contract:
        - ``source``: a ``ConversationRun`` (its ``states``: the state after the
          start, then after each reply; a stay repeats the id), a sequence of
          state ids (``PathRecorder.path(conversation_id)``, a plain list), or a
          non-empty sequence of ``AdvanceResult`` (the path is the first
          ``state_before`` then every ``state_after``). A path recorded by a
          ``PathRecorder`` has no repeat for a stay, a run's has.
        - ``expected``: a sequence of state ids (not one string).
        - ``exact=True`` (default): the path must equal ``expected``. The
          failure names the first index where they differ and prints both paths
          (``path: first difference at index 2: expected 'c', got 'b'``); a path
          that stops early or goes on says which state it missed or added.
        - ``exact=False``: ``expected`` must be an ordered subsequence of the
          path (each state visited, in that order, other states and repeats may
          lie between); the failure names the first expected state not found.
        - ``TypeError`` for a bad ``source`` or ``expected``; ``AssertionError``
          otherwise.
    """
    __tracebackhide__ = True
    if isinstance(expected, (str, bytes)) or not isinstance(expected, Sequence):
        raise TypeError(
            f"expected is a sequence of state ids, got {type(expected).__name__}"
        )
    wanted = tuple(expected)
    for item in wanted:
        if not isinstance(item, str):
            raise TypeError(f"a state id is a string, got {type(item).__name__}")
    actual = _path_of(source)
    both = f"\n  expected: {_arrow(wanted)}\n  actual:   {_arrow(actual)}"
    if not exact:
        position = 0
        for number, state in enumerate(wanted):
            try:
                position = actual.index(state, position) + 1
            except ValueError:
                after = (
                    f"after {actual[position - 1]!r} (index {position - 1})"
                    if position
                    else "in the path"
                )
                raise AssertionError(
                    f"path: expected state {state!r} (item {number} of the expected "
                    f"path) was not visited {after}{both}"
                ) from None
        return
    if actual == wanted:
        return
    for index, (want, got) in enumerate(zip(wanted, actual, strict=False)):
        if want != got:
            raise AssertionError(
                f"path: first difference at index {index}: expected {want!r}, "
                f"got {got!r}{both}"
            )
    if len(actual) < len(wanted):
        stopped = f"after {actual[-1]!r}" if actual else "before the first state"
        raise AssertionError(
            f"path: stopped {stopped} (index {len(actual) - 1}), expected "
            f"{wanted[len(actual)]!r} next{both}"
        )
    if not wanted:
        raise AssertionError(
            f"path: expected an empty path, but it started at {actual[0]!r}{both}"
        )
    raise AssertionError(
        f"path: expected it to end at {wanted[-1]!r} (index {len(wanted) - 1}), "
        f"but it went on to {actual[len(wanted)]!r}{both}"
    )


def _normalised(
    kind: CompletionKind | str | None,
    status: Status | str | None,
    reason: Reason | str | None | _Unset,
) -> tuple[str | None, str | None, str | None | _Unset]:
    """Expected values as wire strings; a typo raises ``ValueError`` listing the vocabulary."""
    return (
        None if kind is None else CompletionKind.from_string(kind).value,
        None if status is None else Status.from_string(status).value,
        reason
        if reason is _UNSET or reason is None
        else Reason.from_string(reason).value,  # type: ignore[arg-type]
    )


def _field_of(result: object, name: str) -> tuple[bool, object]:
    """``(present, value)`` of ``name`` in a result mapping or ``CompletionResponse``."""
    if isinstance(result, Mapping):
        return (name in result, result.get(name))
    if isinstance(result, CompletionResponse):
        if name == "kind":
            return True, result.kind
        return False, None
    raise TypeError(
        "assert_result reads a result mapping ({kind, text, calls} of a completion "
        "state or {status, procedure_id, version, reason} of a synthesis state) or a "
        f"CompletionResponse, got {type(result).__name__}"
    )


def assert_result(
    result: Mapping[str, Any] | CompletionResponse | None,
    *,
    kind: CompletionKind | str | None = None,
    status: Status | str | None = None,
    reason: Reason | str | None | _Unset = _UNSET,
) -> None:
    """Assert the ``kind``, ``status`` and ``reason`` of a run result.

    Interface contract:
        - ``result``: the value a state wrote to its result key
          (``api.get_data(cid)["completion_result"]``, a completion state's
          ``{kind, text, calls}``, or a synthesis state's ``{status,
          procedure_id, version, reason}``), or a ``CompletionResponse``
          (``kind`` only).
        - ``kind``, ``status``: a member of ``CompletionKind`` or ``Status`` or its
          string; ``None`` (default) does not check it.
        - ``reason``: a ``Reason`` member or string; ``None`` asserts the result
          carries NO reason (the key is ``None`` or absent); left out, it is
          not checked (the default is a private ``<unset>`` marker, not ``None``).
        - Every expected value is validated first, whatever the result holds: a
          typo raises ``ValueError`` listing the vocabulary. Naming nothing to
          check is a ``ValueError`` too.
        - Checks ``kind``, then ``status``, then ``reason``, and reports the first
          divergence (``kind: expected 'final', got 'calls'``). An actual value
          outside the vocabulary fails with a message saying so; a result that is
          ``None`` (the key was never written) or lacks the checked field fails
          with its own message; a ``CompletionResponse`` has no status or reason.
        - ``TypeError`` for a result that is neither a mapping nor a
          ``CompletionResponse``.
    """
    __tracebackhide__ = True
    want_kind, want_status, want_reason = _normalised(kind, status, reason)
    if want_kind is None and want_status is None and want_reason is _UNSET:
        raise ValueError("assert_result needs at least one of kind, status or reason")
    if result is None:
        raise AssertionError(
            "result: expected a result mapping, got None (was the result key written?)"
        )
    checks: tuple[tuple[str, Any, tuple[str, ...]], ...] = (
        ("kind", want_kind, tuple(member.value for member in CompletionKind)),
        ("status", want_status, tuple(member.value for member in Status)),
        ("reason", want_reason, tuple(member.value for member in Reason)),
    )
    for name, wanted, vocabulary in checks:
        if wanted is _UNSET or (wanted is None and name != "reason"):
            continue
        present, actual = _field_of(result, name)
        if not present and wanted is None:
            continue  # "no reason" holds when the result carries none
        if not present:
            holder = (
                "a CompletionResponse has no " + name
                if isinstance(result, CompletionResponse)
                else f"the result has no {name!r} (keys: {sorted(result)})"
            )
            raise AssertionError(f"{name}: expected {wanted!r}, but {holder}")
        shown = getattr(actual, "value", actual)  # a member shows as its value
        if shown == wanted:
            continue
        note = (
            ""
            if shown is None or shown in vocabulary
            else f" (not in the {name} vocabulary {list(vocabulary)})"
        )
        raise AssertionError(f"{name}: expected {wanted!r}, got {shown!r}{note}")


class PathRecorder(BaseHandler):
    """A read-only handler that records the states each conversation visits.

    Register it on the ``API`` that code under test drives (an agent, a runner)
    when no ``ConversationRun`` exists to ask for the path::

        recorder = PathRecorder()
        api = API.from_definition(fsm, llm_interface=llm, handlers=[recorder])
        ...                                   # code under test talks to api
        assert_path(recorder.path(conversation_id), ["start", "done"])

    Interface contract:
        - ``PathRecorder(*, name="PathRecorder", priority=10_000)``: a
          ``BaseHandler``; the high priority number makes it run after the
          handlers a test registers beside it. ``API(handlers=[recorder])`` or
          ``api.register_handler(recorder)``.
        - Records the initial state (at ``START_CONVERSATION``) and the target
          of every transition (at ``POST_TRANSITION``), per frame, keyed by the
          ``_conversation_id`` of the context: a pushed FSM is its own frame
          with its own id. A stay is not a transition, so it adds nothing
          (``ConversationRun.states`` repeats the id, the recorder does not). A
          conversation restored from a session records no initial state (the
          start handlers are suppressed). A ``POST_TRANSITION`` handler that
          fails later in the same chain rolls the turn back after the recorder
          saw the hop; the hop stays recorded.
        - ``path(conversation_id) -> tuple[str, ...]``; ``ValueError`` listing
          the recorded ids for one it never saw. ``frames`` is the tuple of
          conversation ids seen, in order. ``clear()`` forgets everything.
        - Never changes a conversation: ``should_execute`` returns ``False`` and
          ``execute`` returns ``{}``. It records on the thread that fired the
          timing and is safe to read from another.
        - SIDE EFFECT of registering ANY handler, this one included: the
          pipeline deep-copies the context at every handler timing that has a
          handler (with zero handlers it skips the copy, D-022), so each turn
          makes more copies, and a context value that cannot be deep-copied
          raises on a turn that ran without a handler. Use the recorder only
          where that is acceptable, and prefer ``ConversationRun.states`` for a
          conversation a ``ScriptedUser`` drives.
    """

    def __init__(self, *, name: str = "PathRecorder", priority: int = 10_000) -> None:
        super().__init__(name=name, priority=priority)
        self._lock = threading.Lock()
        self._paths: dict[str, list[str]] = {}

    def should_execute(
        self,
        timing: HandlerTiming,
        current_state: str,
        target_state: str | None,
        context: dict[str, Any],
        updated_keys: set[str] | None = None,
    ) -> bool:
        """Record the visit for the two timings that carry one; never run ``execute``."""
        if timing is HandlerTiming.START_CONVERSATION:
            self._record(context, current_state)
        elif timing is HandlerTiming.POST_TRANSITION:
            self._record(context, target_state or current_state)
        return False

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Never called (``should_execute`` is always ``False``); returns no update."""
        return {}

    def _record(self, context: dict[str, Any], state: str) -> None:
        frame = context.get("_conversation_id")
        key = frame if isinstance(frame, str) else ""
        with self._lock:
            self._paths.setdefault(key, []).append(state)

    def path(self, conversation_id: str) -> tuple[str, ...]:
        """The states recorded for ``conversation_id``; ``ValueError`` if it was never seen."""
        with self._lock:
            if conversation_id not in self._paths:
                raise ValueError(
                    f"no path recorded for conversation {conversation_id!r}; "
                    f"recorded: {list(self._paths)}"
                )
            return tuple(self._paths[conversation_id])

    @property
    def frames(self) -> tuple[str, ...]:
        """The conversation ids recorded so far, in the order first seen."""
        with self._lock:
            return tuple(self._paths)

    def clear(self) -> None:
        """Forget every recorded path."""
        with self._lock:
            self._paths.clear()
