from __future__ import annotations

"""
Exception hierarchy for fsm_llm_monitor package.
"""


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
