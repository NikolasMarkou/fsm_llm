"""The generic turn loop: one conversation, scripted turns, the result as data."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from fsm_llm.core.codes import MessageSource, message_source
from fsm_llm.engine.api.conversation import API


@dataclass
class Trial:
    """What one played conversation showed; no expectation, no verdict.

    ``error`` is the turn failure (``"Type: text"``); ``close_error`` is
    ``"close failed: ..."`` when closing the API raised. Both ``None`` is a clean run.
    """

    responses: list[str] = field(default_factory=list)
    visited_states: list[str] = field(default_factory=list)
    turns_sent: int = 0
    final_state: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    ended: bool = False
    error: str | None = None
    close_error: str | None = None
    duration: float = 0.0


def play(
    make_api: Callable[[], API],
    initial_context: Mapping[str, Any],
    turns: Sequence[str],
    *,
    stop: Callable[[], bool] | None = None,
) -> Trial:
    """Play ``turns`` on a fresh ``make_api()`` and return the :class:`Trial`. Never raises.

    Before each turn the loop asks whether the conversation ended, then ``stop()``
    (a true result ends the trial there). Each turn runs under
    ``message_source(SIMULATOR)``. Any exception becomes ``error`` and the API is
    closed in every case.
    """
    trial = Trial()
    start = time.monotonic()
    api: API | None = None
    try:
        api = make_api()
        conv_id, greeting = api.start_conversation(dict(initial_context))
        trial.responses.append(greeting)
        trial.visited_states.append(api.get_current_state(conv_id))
        for turn in turns:
            if api.has_conversation_ended(conv_id) or (stop is not None and stop()):
                break
            with message_source(MessageSource.SIMULATOR):
                trial.responses.append(api.converse(turn, conv_id))
            trial.turns_sent += 1
            trial.visited_states.append(api.get_current_state(conv_id))
        trial.final_state = trial.visited_states[-1]
        trial.context = api.get_data(conv_id)
        trial.ended = api.has_conversation_ended(conv_id)
    except Exception as exc:  # one trial's failure is data, never a crashed run
        trial.error = f"{type(exc).__name__}: {exc}"
    finally:
        if api is not None:
            try:
                api.close()  # ends every active conversation of this API
            except Exception as exc:
                trial.close_error = f"close failed: {exc}"
        trial.duration = time.monotonic() - start
    return trial
