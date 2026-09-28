"""
Exception hierarchy for fsm_llm_monitor package.
"""

from __future__ import annotations


class MonitorError(Exception):
    """Base exception for all monitor errors."""

    pass


class MonitorInitializationError(MonitorError):
    """Raised when monitor initialization fails."""

    pass


class MetricCollectionError(MonitorError):
    """Raised when metric collection fails."""

    pass


class MonitorConnectionError(MonitorError):
    """Raised when monitor cannot connect to the API instance."""

    pass


class MonitorCapacityError(MonitorError):
    """Raised when a launch would exceed a configured capacity limit
    (``max_instances``, ``max_running_agents``, builder sessions)."""

    pass
