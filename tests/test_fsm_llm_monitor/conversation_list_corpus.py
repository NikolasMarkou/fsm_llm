"""The fixture and the readings of the conversation list rows (plan 9283c1fd step 3). Not collected.

``build()`` makes an ``InstanceManager`` with 20 managed FSMs of 5 conversations each, one attached
API with two live conversations, and the shapes the list has to keep:

- every seventh conversation is ended: gone from its API, kept in ``conversation_ids`` and in the
  ended cache (the cache is filled in a different order from the creation order);
- ``fsm02-c1`` is live AND cached (the window of an end): the live snapshot must win;
- ``fsm04`` is a ``completed`` instance with live conversations (never listed as active, still read
  through its ``conversation_ids``);
- ``fsm09`` raises when asked for its active conversations;
- ``fsm13`` has one live conversation that is not in its ``conversation_ids``;
- an agent instance sits between two FSMs, two cached conversations have no owner left.

The APIs are doubles: ``get_complete_conversation`` answers a live id and raises ``KeyError`` for
any other, and every call is written to ``Probe`` with the state of ``manager._lock`` at that moment.

``rows(snapshots)`` is what the golden holds: ``[conversation_id, instance_id, current_state,
is_terminal]`` per snapshot, in list order. ``tests/fixtures/conversation_list_golden.json`` was
written from a ``git archive`` of BASE ``c8c96d5b`` (the old per-id probe loop) by
``python -m tests.test_fsm_llm_monitor.conversation_list_corpus --write PATH``; the tree never
rewrites it.
"""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

from fsm_llm.monitor.definitions import ConversationSnapshot, MonitorConfig
from fsm_llm.monitor.instance_manager import (
    InstanceManager,
    ManagedAgent,
    ManagedFSM,
)

FSM_COUNT = 20
PER_FSM = 5
ATTACHED_IDS = ["att-0", "att-1"]


class RecordingLock:
    """An ``RLock`` that counts its acquisitions and knows whether it is held (one thread)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.depth = 0
        self.acquisitions = 0

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        taken = self._lock.acquire(blocking, timeout)
        if taken:
            self.acquisitions += 1
            self.depth += 1
        return taken

    def release(self) -> None:
        self.depth -= 1
        self._lock.release()

    def __enter__(self) -> RecordingLock:
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


@dataclass
class Probe:
    """Every call into an API double: ``(api name, method, conversation id, lock held)``."""

    lock: RecordingLock = field(default_factory=RecordingLock)
    calls: list[tuple[str, str, str, bool]] = field(default_factory=list)

    def record(self, api: str, method: str, conversation_id: str = "") -> None:
        self.calls.append((api, method, conversation_id, self.lock.depth > 0))

    def reads(self) -> list[str]:
        return [c[2] for c in self.calls if c[1] == "get_complete_conversation"]

    def lists(self) -> list[str]:
        return [c[0] for c in self.calls if c[1] == "list_active_conversations"]

    def calls_under_lock(self) -> list[tuple[str, str, str, bool]]:
        return [c for c in self.calls if c[3]]


def _api(
    name: str, live: list[str], states: dict[str, str], probe: Probe, *, raises=False
) -> MagicMock:
    api = MagicMock(name=name)
    api.get_stack_depth.return_value = 1

    def list_active() -> list[str]:
        probe.record(name, "list_active_conversations")
        if raises:
            raise RuntimeError("cannot list")
        return list(live)

    def read(conversation_id: str) -> dict[str, Any]:
        probe.record(name, "get_complete_conversation", conversation_id)
        if conversation_id not in live:
            raise KeyError(conversation_id)
        return {
            "current_state": {
                "id": states.get(conversation_id, "s0"),
                "description": "d",
                "is_terminal": False,
            },
            "collected_data": {},
            "conversation_history": [],
        }

    api.list_active_conversations.side_effect = list_active
    api.fsm_manager.get_complete_conversation.side_effect = read
    return api


def _ended(conversation_id: str, instance_id: str) -> ConversationSnapshot:
    return ConversationSnapshot(
        conversation_id=conversation_id,
        instance_id=instance_id,
        current_state="end",
        is_terminal=True,
    )


@dataclass
class Fixture:
    manager: InstanceManager
    probe: Probe
    #: every id the list can reach through an API: each one is read once and no more
    reachable: set[str]


def build() -> Fixture:
    """The manager of the rows above, with its lock replaced by a ``RecordingLock``."""
    probe = Probe()
    manager = InstanceManager(config=MonitorConfig(max_instances=100))
    manager.global_collector.cleanup()
    manager.attach_api(
        _api(
            "attached", list(ATTACHED_IDS), dict.fromkeys(ATTACHED_IDS, "s_att"), probe
        )
    )
    reachable = set(ATTACHED_IDS)
    cache_order: list[ConversationSnapshot] = []
    for i in range(FSM_COUNT):
        instance_id = f"fsm{i:02d}"
        ids = [f"{instance_id}-c{j}" for j in range(PER_FSM)]
        ended = {c for j, c in enumerate(ids) if (i * PER_FSM + j) % 7 == 3}
        live = [c for c in ids if c not in ended]
        states = {c: f"s{(i + j) % 3}" for j, c in enumerate(ids)}
        if i == 2:  # live and cached at once: the live snapshot wins
            cache_order.append(_ended("fsm02-c1", instance_id))
        if i == 13:
            live.append("fsm13-extra")
            states["fsm13-extra"] = "s2"
            reachable.add("fsm13-extra")
        api = _api(instance_id, live, states, probe, raises=(i == 9))
        inst = ManagedFSM(instance_id=instance_id, api=api, label=f"F{i}")
        inst.conversation_ids = list(ids)
        if i == 4:
            inst.status = "completed"
        manager._instances[instance_id] = inst
        reachable.update(ids)
        cache_order.extend(_ended(c, instance_id) for c in sorted(ended, reverse=True))
        if i == 6:
            manager._instances["agent-1"] = ManagedAgent(
                instance_id="agent-1", agent_type="react", task="t"
            )
    cache_order.append(_ended("old-0", "gone-fsm"))
    cache_order.append(_ended("old-1", ""))
    for snap in cache_order:
        manager._ended_conversations[snap.conversation_id] = snap
    probe.calls.clear()
    manager._lock = probe.lock
    return Fixture(manager=manager, probe=probe, reachable=reachable)


def rows(snapshots: list[ConversationSnapshot]) -> list[list[Any]]:
    return [
        [s.conversation_id, s.instance_id, s.current_state, s.is_terminal]
        for s in snapshots
    ]


def golden() -> dict[str, list[list[Any]]]:
    """The two lists of the golden, read from the checkout that is imported."""
    out: dict[str, list[list[Any]]] = {}
    for include_ended in (True, False):
        fixture = build()
        try:
            snaps = fixture.manager.get_all_conversation_snapshots(
                include_ended=include_ended
            )
        finally:
            fixture.manager._lock = threading.RLock()
            fixture.manager.close()
        out[f"include_ended_{str(include_ended).lower()}"] = rows(snaps)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--write":
        raise SystemExit("usage: python -m ...conversation_list_corpus --write PATH")
    import fsm_llm

    sys.stderr.write(f"fsm_llm from {fsm_llm.__file__}\n")
    lists = golden()
    lines = ['{"base": "c8c96d5b"']
    for key in sorted(lists):
        body = ",\n".join("  " + json.dumps(row) for row in lists[key])
        lines.append(f' "{key}": [\n{body}\n ]')
    with open(sys.argv[2], "w", encoding="utf-8") as handle:
        handle.write(",\n".join(lines) + "\n}\n")
