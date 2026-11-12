"""Helpers shared by several test files of this suite. Not collected (no test_ prefix)."""

from __future__ import annotations

from fsm_llm.monitor.definitions import MonitorConfig
from fsm_llm.monitor.instance_manager import InstanceManager


def _manager() -> InstanceManager:
    mgr = InstanceManager(config=MonitorConfig())
    mgr.global_collector.cleanup()
    return mgr
