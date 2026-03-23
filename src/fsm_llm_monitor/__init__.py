from __future__ import annotations

"""
FSM-LLM Monitor Extension
==========================

Web-based monitoring dashboard for FSM-LLM with a Grafana-inspired dark theme.

Provides real-time observability for FSM conversations, agent executions,
workflow instances, and system logs via a browser-based dashboard.

Quick Start::

    # Install with monitor extra
    pip install fsm-llm[monitor]

    # Launch the monitor (opens browser at http://127.0.0.1:8420)
    python -m fsm_llm_monitor

    # Or programmatically
    from fsm_llm_monitor import MonitorBridge, configure, app
    bridge = MonitorBridge(api=my_api)
    configure(bridge)
    # Run with uvicorn: uvicorn fsm_llm_monitor.server:app
"""

from .__version__ import __version__

# Core classes
from .bridge import MonitorBridge
from .collector import EventCollector

# Constants
from .constants import (
    COLOR_PRIMARY,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_EVENTS,
    DEFAULT_MAX_LOG_LINES,
    DEFAULT_REFRESH_INTERVAL,
    EVENT_CONTEXT_UPDATE,
    EVENT_CONVERSATION_END,
    EVENT_CONVERSATION_START,
    EVENT_ERROR,
    EVENT_LOG,
    EVENT_POST_PROCESSING,
    EVENT_PRE_PROCESSING,
    EVENT_STATE_TRANSITION,
    MONITOR_HANDLER_NAME,
    MONITOR_HANDLER_PRIORITY,
    THEME_NAME,
)

# Definitions
from .definitions import (
    ConversationSnapshot,
    FSMSnapshot,
    LogRecord,
    MetricSnapshot,
    MonitorConfig,
    MonitorEvent,
    StateInfo,
    TransitionInfo,
)

# Exceptions
from .exceptions import (
    MetricCollectionError,
    MonitorConnectionError,
    MonitorError,
    MonitorInitializationError,
)

__all__ = [
    # Version
    "__version__",
    # Core classes
    "MonitorBridge",
    "EventCollector",
    # Definitions
    "MonitorEvent",
    "LogRecord",
    "MetricSnapshot",
    "ConversationSnapshot",
    "FSMSnapshot",
    "StateInfo",
    "TransitionInfo",
    "MonitorConfig",
    # Constants
    "THEME_NAME",
    "COLOR_PRIMARY",
    "DEFAULT_REFRESH_INTERVAL",
    "DEFAULT_MAX_EVENTS",
    "DEFAULT_MAX_LOG_LINES",
    "DEFAULT_LOG_LEVEL",
    "EVENT_CONVERSATION_START",
    "EVENT_CONVERSATION_END",
    "EVENT_STATE_TRANSITION",
    "EVENT_PRE_PROCESSING",
    "EVENT_POST_PROCESSING",
    "EVENT_CONTEXT_UPDATE",
    "EVENT_ERROR",
    "EVENT_LOG",
    "MONITOR_HANDLER_NAME",
    "MONITOR_HANDLER_PRIORITY",
    # Exceptions
    "MonitorError",
    "MonitorInitializationError",
    "MetricCollectionError",
    "MonitorConnectionError",
]
