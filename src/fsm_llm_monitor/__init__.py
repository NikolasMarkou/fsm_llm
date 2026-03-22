from __future__ import annotations

"""
FSM-LLM Monitor Extension
==========================

Terminal-based monitoring dashboard for FSM-LLM with a retro 90s aesthetic.

Provides real-time observability for FSM conversations, agent executions,
workflow instances, and system logs via a Textual TUI.

Quick Start::

    # Install with monitor extra
    pip install fsm-llm[monitor]

    # Launch the monitor
    python -m fsm_llm_monitor

    # Or programmatically
    from fsm_llm_monitor import MonitorApp
    app = MonitorApp()
    app.run()
"""

from .__version__ import __version__

# Core classes
from .app import MonitorApp
from .bridge import MonitorBridge
from .collector import EventCollector

# Constants
from .constants import (
    COLOR_ACCENT,
    COLOR_BACKGROUND,
    COLOR_BORDER,
    COLOR_ERROR,
    COLOR_FOREGROUND,
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SUCCESS,
    COLOR_SURFACE,
    COLOR_WARNING,
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
    KEY_AGENTS,
    KEY_CONVERSATION,
    KEY_DASHBOARD,
    KEY_FSM_VIEWER,
    KEY_LOGS,
    KEY_QUIT,
    KEY_SETTINGS,
    KEY_WORKFLOWS,
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
    "MonitorApp",
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
    # Constants — Theme
    "THEME_NAME",
    "COLOR_PRIMARY",
    "COLOR_SECONDARY",
    "COLOR_BACKGROUND",
    "COLOR_SURFACE",
    "COLOR_FOREGROUND",
    "COLOR_ACCENT",
    "COLOR_WARNING",
    "COLOR_ERROR",
    "COLOR_SUCCESS",
    "COLOR_MUTED",
    "COLOR_BORDER",
    # Constants — Defaults
    "DEFAULT_REFRESH_INTERVAL",
    "DEFAULT_MAX_EVENTS",
    "DEFAULT_MAX_LOG_LINES",
    "DEFAULT_LOG_LEVEL",
    # Constants — Event Types
    "EVENT_CONVERSATION_START",
    "EVENT_CONVERSATION_END",
    "EVENT_STATE_TRANSITION",
    "EVENT_PRE_PROCESSING",
    "EVENT_POST_PROCESSING",
    "EVENT_CONTEXT_UPDATE",
    "EVENT_ERROR",
    "EVENT_LOG",
    # Constants — Keybindings
    "KEY_DASHBOARD",
    "KEY_FSM_VIEWER",
    "KEY_CONVERSATION",
    "KEY_AGENTS",
    "KEY_WORKFLOWS",
    "KEY_LOGS",
    "KEY_SETTINGS",
    "KEY_QUIT",
    # Constants — Handler
    "MONITOR_HANDLER_NAME",
    "MONITOR_HANDLER_PRIORITY",
    # Exceptions
    "MonitorError",
    "MonitorInitializationError",
    "MetricCollectionError",
    "MonitorConnectionError",
]
