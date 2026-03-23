from __future__ import annotations

"""
Constants for fsm_llm_monitor package.
"""

# --- Theme Colors (Grafana dark) ---
THEME_NAME = "grafana_dark"
COLOR_PRIMARY = "#3274d9"
COLOR_SECONDARY = "#1f60c4"
COLOR_BACKGROUND = "#111217"
COLOR_SURFACE = "#1e2028"
COLOR_FOREGROUND = "#d8d9da"
COLOR_ACCENT = "#5794f2"
COLOR_WARNING = "#ff9830"
COLOR_ERROR = "#f2495c"
COLOR_SUCCESS = "#73bf69"
COLOR_MUTED = "#8e8e8e"
COLOR_BORDER = "#2c3235"

# --- Defaults ---
DEFAULT_REFRESH_INTERVAL = 1.0  # seconds
DEFAULT_MAX_EVENTS = 1000
DEFAULT_MAX_LOG_LINES = 5000
DEFAULT_LOG_LEVEL = "INFO"

# --- Event Types ---
EVENT_CONVERSATION_START = "conversation_start"
EVENT_CONVERSATION_END = "conversation_end"
EVENT_STATE_TRANSITION = "state_transition"
EVENT_PRE_PROCESSING = "pre_processing"
EVENT_POST_PROCESSING = "post_processing"
EVENT_CONTEXT_UPDATE = "context_update"
EVENT_ERROR = "error"
EVENT_LOG = "log"

# --- Handler ---
MONITOR_HANDLER_NAME = "fsm_llm_monitor"
MONITOR_HANDLER_PRIORITY = 9999  # Lowest priority — observe only
