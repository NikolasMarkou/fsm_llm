"""The values ``settle`` and a state executor exchange about a child frame.

Layer ``engine`` (plan baa37f40, step 20.2, D-038). ``engine/api/calls.py`` applies the request
a kind's executor wrote in Pass 1: it pushes the child and, when the child ends or its budget is
spent, pops it with ONE merge. The ``call`` kind is judged inside ``settle`` (a registry
resolves the target); any other settled kind (``synthesize``) asks its executor, through the
hooks of ``StateExecutor`` (``aplan_push``, ``arefusal``, ``aplan_pop``, ``on_child_turn``).
``engine/kinds`` imports no api, turn or manager, so everything an executor may know of the
stack crosses here as a plain value, never as an object of the API.

Interface contract: every class is a frozen dataclass of copies (no field aliases a live
conversation). ``PushContext``: the root id, the parent frame's conversation id, its state, its
definition and a copy of its raw context, and the ``running`` record the state wrote.
``PushPlan``: the definition to push, ONLY the inputs the child may see, and the frame budget
(``None`` for none). ``ChildView``: a child frame as it is now: its conversation id, ``fsm_id``,
current state, a copy of its raw context, whether it ended (its state has no transition), the
name of its spent lifetime budget (``"steps"`` or ``"seconds"``) or ``None``, the engine's
count of its turns (``None`` when unknown), the provider errors made since the push (``0``
unless the executor sets ``watches_outage``) and its ``signals`` (a frozen ``FrameSignals`` read
once, at the child's end, from the event log; ``None`` with the log off, after a restart or a
fork that lost the reader, and on every turn before the end). ``PopContext``: the root and parent ids, the
parent's state and definition, the ``running`` record and the ``ChildView``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fsm_llm.core.definitions import FSMDefinition, State
from fsm_llm.runtime.signals import FrameSignals


@dataclass(frozen=True)
class PushContext:
    """What an executor sees of the parent frame when settle is about to push its child."""

    root_id: str
    parent_id: str
    state: State
    definition: FSMDefinition
    data: dict[str, Any]
    record: Mapping[str, Any]


@dataclass(frozen=True)
class PushPlan:
    """What an executor asks settle to push: the definition, the child's only keys, the budget."""

    definition: FSMDefinition
    inputs: Mapping[str, Any]
    max_steps: int | None = None
    max_seconds: float | None = None


@dataclass(frozen=True)
class ChildView:
    """A child frame as settle read it (copies; see the module docstring)."""

    conversation_id: str
    fsm_id: str | None
    state_id: str
    data: dict[str, Any]
    ended: bool
    spent: str | None
    steps: int | None
    errors: int = 0
    signals: FrameSignals | None = None


@dataclass(frozen=True)
class PopContext:
    """What an executor sees when settle is about to pop a child, or sees it take a turn."""

    root_id: str
    parent_id: str
    state: State
    definition: FSMDefinition
    record: Mapping[str, Any]
    child: ChildView
