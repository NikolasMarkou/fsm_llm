from __future__ import annotations

"""
Constants for fsm_llm_monitor package.
"""

# --- Theme Colors (90s retro) ---
THEME_NAME = "retro_green"
COLOR_PRIMARY = "#00ff00"
COLOR_SECONDARY = "#00cc00"
COLOR_BACKGROUND = "#000000"
COLOR_SURFACE = "#0a0a0a"
COLOR_FOREGROUND = "#00ff00"
COLOR_ACCENT = "#33ff33"
COLOR_WARNING = "#ffff00"
COLOR_ERROR = "#ff0000"
COLOR_SUCCESS = "#00ff00"
COLOR_MUTED = "#006600"
COLOR_BORDER = "#004400"

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

# --- Keybindings ---
KEY_DASHBOARD = "d"
KEY_FSM_VIEWER = "f"
KEY_CONVERSATION = "c"
KEY_AGENTS = "a"
KEY_WORKFLOWS = "w"
KEY_LOGS = "l"
KEY_SETTINGS = "s"
KEY_QUIT = "q"

# --- Handler ---
MONITOR_HANDLER_NAME = "fsm_llm_monitor"
MONITOR_HANDLER_PRIORITY = 9999  # Lowest priority — observe only
