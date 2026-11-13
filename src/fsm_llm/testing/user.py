"""
A scripted user: drive a conversation without a person, record what happened.

``ScriptedUser`` sends messages to an ``API`` conversation, one at a time: first
the messages it was given, then an answer chosen by a rule that reads the last
reply, then a fallback, then it stops. ``advance_until_terminal`` does the same
for an FSM that needs no user message (``API.run_until_terminal``). Both return
a ``ConversationRun``, the record a test (and ``assert_path``) reads.

The path of a run is sampled the way ``fsm_llm.eval`` samples it: one
``get_current_state`` call after the start and after each reply. One ``converse``
or ``advance`` makes at most one transition, so the sampled states are the exact
path. The core keeps no state history of its own.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from fsm_llm.api import API

#: A rule's trigger: a regex (``str`` or compiled, searched in the last reply) or
#: a predicate that takes the last reply and returns whether the rule fires.
RuleMatcher = str | re.Pattern[str] | Callable[[str], bool]

#: A rule: ``(matcher, utterance)`` or ``(matcher, (utterance, ...))``.
ScriptedRule = tuple[RuleMatcher, str | Sequence[str]]


@dataclass(frozen=True)
class ConversationRun:
    """The record of one driven conversation.

    Interface contract (returned by ``ScriptedUser.run`` and
    ``advance_until_terminal``; read by the assertions):
        - ``api`` / ``conversation_id``: the API and the conversation that was
          driven, so a test can read it further (``api.get_data(...)``) unless
          the run closed it.
        - ``greeting``: the reply of ``start_conversation`` (``""`` when the
          initial state is silent).
        - ``messages``: the user messages sent, in order (empty for a
          message-free run).
        - ``replies``: one reply per ``messages`` item, or one per step of an
          ``advance_until_terminal`` run (``""`` for a silent state, the same
          value ``converse`` returns there).
        - ``states``: the path, ``(state after start, state after each reply)``;
          always ``len(replies) + 1`` long. A stay shows as a repeated id.
          After a pushed FSM the ids are those of the stack top at each sample.
        - ``ended``: whether the conversation had ended when the run stopped,
          read before ``close`` ran.
        - Frozen; ``messages``, ``replies`` and ``states`` are tuples. The
          ``api`` field is left out of ``repr`` and ``==``.
    """

    api: API = field(repr=False, compare=False)
    conversation_id: str
    greeting: str
    messages: tuple[str, ...]
    replies: tuple[str, ...]
    states: tuple[str, ...]
    ended: bool

    @property
    def final_state(self) -> str:
        """The last state of the path (the state the conversation stopped in)."""
        return self.states[-1]


def _copied(initial_context: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A copy of a mapping; anything else goes to the API unchanged, which refuses it."""
    if isinstance(initial_context, Mapping):
        return dict(initial_context)
    return initial_context


class _Rule:
    """One compiled rule: a matcher and its utterances, with a per-run cursor."""

    def __init__(self, matcher: RuleMatcher, utterance: str | Sequence[str]) -> None:
        if isinstance(matcher, str):
            pattern = re.compile(matcher)
            self.matches: Callable[[str], bool] = lambda reply: bool(
                pattern.search(reply)
            )
        elif isinstance(matcher, re.Pattern):
            self.matches = lambda reply: bool(matcher.search(reply))
        elif callable(matcher):
            self.matches = lambda reply: bool(matcher(reply))
        else:
            raise TypeError(
                f"a rule matcher is a regex string, a compiled pattern or a "
                f"callable, got {type(matcher).__name__}"
            )
        # A string repeats on every match; a sequence is used up in order.
        self.repeat = isinstance(utterance, str)
        self.utterances: tuple[str, ...] = (
            (utterance,) if isinstance(utterance, str) else tuple(utterance)
        )
        for item in self.utterances:
            if not isinstance(item, str):
                raise TypeError(
                    f"a rule utterance is a string, got {type(item).__name__}"
                )


class ScriptedUser:
    """A user that follows a script: fixed messages, then rules, then a fallback.

    Interface contract (build once, ``run`` against any ``API``; a user holds no
    state between runs):
        - ``messages``: sent first, in order. An iterable of strings, not one
          string (``TypeError``: ``"hi"`` would be sent letter by letter).
        - ``rules``: ``(matcher, utterance)`` pairs tried in order against the
          last reply (the greeting before the first message) once the messages
          are spent; the first that fires supplies the next message. A
          ``matcher`` is a regex (``str`` or compiled, ``re.search``) or a
          callable ``reply -> bool``. An ``utterance`` that is a string is sent
          on every match; a sequence of strings is sent in order, one per
          match, and the rule stops firing when it is used up.
        - ``fallback``: the message sent when no rule fires; ``None`` (default)
          stops the run there. A string fallback repeats, so ``max_turns``
          bounds it.
        - ``max_turns``: the most messages one run sends (non-negative int,
          default 25), a stop and no error. A run also stops when the
          conversation has ended or nothing is left to send.
        - ``run(api, *, initial_context=None, close=False)``: starts a
          conversation on ``api`` (``initial_context`` is copied), drives it and
          returns a ``ConversationRun``. ``close=True`` ends the conversation
          afterwards, also when the run raised.
        - Failure modes: exceptions of the API, the LLM or a matcher
          PROPAGATE (``fsm_llm.eval`` records them as data; a test wants the
          traceback). A bad argument raises ``TypeError`` or ``ValueError`` at
          construction, a bad regex ``re.error``.
    """

    def __init__(
        self,
        messages: Sequence[str] = (),
        *,
        rules: Sequence[ScriptedRule] = (),
        fallback: str | None = None,
        max_turns: int = 25,
    ) -> None:
        if isinstance(messages, str):
            raise TypeError(
                "messages must be a sequence of strings, not one string: "
                "pass [text] for a single message"
            )
        self.messages: tuple[str, ...] = tuple(messages)
        for item in self.messages:
            if not isinstance(item, str):
                raise TypeError(f"a message is a string, got {type(item).__name__}")
        if fallback is not None and not isinstance(fallback, str):
            raise TypeError(f"fallback is a string or None, got {fallback!r}")
        if isinstance(max_turns, bool) or not isinstance(max_turns, int):
            raise TypeError(f"max_turns must be an int, got {max_turns!r}")
        if max_turns < 0:
            raise ValueError(f"max_turns must be at least 0, got {max_turns}")
        self.fallback = fallback
        self.max_turns = max_turns
        self._rules: tuple[_Rule, ...] = tuple(
            _Rule(matcher, utterance) for matcher, utterance in rules
        )

    def run(
        self,
        api: API,
        *,
        initial_context: Mapping[str, Any] | None = None,
        close: bool = False,
    ) -> ConversationRun:
        """Start a conversation on ``api``, drive it to a stop, return the record."""
        conversation_id, greeting = api.start_conversation(_copied(initial_context))
        try:
            sent: list[str] = []
            replies: list[str] = []
            states = [api.get_current_state(conversation_id)]
            used = [0] * len(self._rules)  # per-run cursors: a user is reusable
            last = greeting
            while len(sent) < self.max_turns and not api.has_conversation_ended(
                conversation_id
            ):
                message = self._next_message(len(sent), last, used)
                if message is None:
                    break
                last = api.converse(message, conversation_id)
                sent.append(message)
                replies.append(last)
                states.append(api.get_current_state(conversation_id))
            return ConversationRun(
                api=api,
                conversation_id=conversation_id,
                greeting=greeting,
                messages=tuple(sent),
                replies=tuple(replies),
                states=tuple(states),
                ended=api.has_conversation_ended(conversation_id),
            )
        finally:
            if close:
                api.end_conversation(conversation_id)

    def _next_message(self, index: int, last_reply: str, used: list[int]) -> str | None:
        """The message to send as number ``index``, or ``None`` to stop."""
        if index < len(self.messages):
            return self.messages[index]
        for position, rule in enumerate(self._rules):
            if not rule.repeat and used[position] >= len(rule.utterances):
                continue
            if not rule.matches(last_reply):
                continue
            if rule.repeat:
                return rule.utterances[0]
            used[position] += 1
            return rule.utterances[used[position] - 1]
        return self.fallback


def advance_until_terminal(
    api: API,
    *,
    max_steps: int,
    initial_context: Mapping[str, Any] | None = None,
    max_seconds: float | None = None,
    close: bool = False,
) -> ConversationRun:
    """Run a message-free FSM to its end and return the record of the run.

    Interface contract (for an FSM whose states need no user message):
        - Starts a conversation on ``api`` (``initial_context`` is copied), runs
          ``api.run_until_terminal(conversation_id, max_steps=..., max_seconds=...)``
          and returns a ``ConversationRun`` with ``messages == ()``, one reply per
          step (``""`` for a silent state) and ``states`` built from the
          ``AdvanceResult`` of each step: the state after the start, then each
          ``state_after``.
        - ``max_steps`` is required, as for ``run_until_terminal``.
        - ``close=True`` ends the conversation afterwards, also on an error.
        - Failure modes: ``RunBudgetExceededError`` when a budget is spent, and
          any ``FSMError`` of a failed step, propagate unchanged.
    """
    conversation_id, greeting = api.start_conversation(_copied(initial_context))
    try:
        initial_state = api.get_current_state(conversation_id)
        steps = api.run_until_terminal(
            conversation_id, max_steps=max_steps, max_seconds=max_seconds
        )
        return ConversationRun(
            api=api,
            conversation_id=conversation_id,
            greeting=greeting,
            messages=(),
            replies=tuple(step.response or "" for step in steps),
            states=(initial_state, *(step.state_after for step in steps)),
            ended=api.has_conversation_ended(conversation_id),
        )
    finally:
        if close:
            api.end_conversation(conversation_id)
