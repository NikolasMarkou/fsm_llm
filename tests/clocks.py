"""A stand-in clock for tests whose subject reads the wall clock (plan d7719b03 step 11.4).

Lives beside ``tests/conftest.py``, which re-exports ``EveryReadLater`` (import it as
``from tests.conftest import EveryReadLater``): ``conftest.py`` defines no class of its
own (``test_doubles.py`` pins that).
"""

from __future__ import annotations

import time
from typing import Any

__all__ = ["EveryReadLater"]


class EveryReadLater:
    """Stand-in for the ``time`` module of a module under test: every read is later.

    Interface contract (call sites: a spent load budget in the skills catalog, an idle
    sweep in ``fsm_llm.engine.api``, the transition timestamp in
    ``fsm_llm.engine.turn.transitions``, the run budget of ``fsm_llm.agents.base.runs``):
        - Bind it with ``monkeypatch.setattr(module, "time", EveryReadLater())`` (or
          ``bind_api_clock``); the module keeps calling ``time.monotonic()`` or
          ``time.time()`` unchanged.
        - ``monotonic()`` and ``time()`` return one shared counter that is *step*
          seconds (default one second) later on every call, starting above *start*.
          Two reads are never equal, however fast the code between them is, so a
          budget of ``1e-9`` s is spent at the second read, an idle sweep with
          ``max_idle_seconds=0`` finds every conversation idle, and a timestamp
          written on two transitions differs.
        - Any other attribute (``sleep``, ``perf_counter``, ``monotonic_ns``) is the real
          ``time``'s, so code that sleeps or measures with them still works.

    Why not the real clock: on Windows before Python 3.13 ``time.time()``,
    ``time.monotonic()`` and ``time.process_time()`` tick every 15.6 ms (the system
    clock, ``GetTickCount64`` and ``GetProcessTimes``), so a fast sequence reads one
    value twice (CI runs 44028115098 and 44038473796, Windows 3.11). A test that needs a
    clock that has moved must not depend on the platform's tick.
    """

    def __init__(self, start: float = 1000.0, step: float = 1.0) -> None:
        self.now = start
        self.step = step

    def monotonic(self) -> float:
        self.now += self.step
        return self.now

    def time(self) -> float:
        self.now += self.step
        return self.now

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)
