"""
Constants and configuration values for the LLM-FSM framework.

This module centralizes all constants used throughout the framework
to improve maintainability and make configuration changes easier.
"""

# FSM Definition defaults
DEFAULT_FSM_VERSION = "3.0"
DEFAULT_TRANSITION_PRIORITY = 100

# Conversation history defaults
DEFAULT_MAX_HISTORY_SIZE = 5
DEFAULT_MAX_MESSAGE_LENGTH = 1000

# Message truncation defaults
DEFAULT_MESSAGE_TRUNCATE_LENGTH = 50

# Logging configuration
LOG_ROTATION_SIZE = "10 MB"
LOG_RETENTION_PERIOD = "1 month"

# LLM configuration defaults
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 1000

# Environment variable keys
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_LLM_MODEL = "LLM_MODEL"
ENV_LLM_TEMPERATURE = "LLM_TEMPERATURE"
ENV_LLM_MAX_TOKENS = "LLM_MAX_TOKENS"
ENV_FSM_PATH = "FSM_PATH"

# Prompt XML tags
XML_TAGS = {
    "fsm_open": "<fsm>",
    "fsm_close": "</fsm>",
    "metadata_open": "<metadata>",
    "metadata_close": "</metadata>",
    "state_info_open": "<stateInfo>",
    "state_info_close": "</stateInfo>",
    "transitions_open": "<transitions>",
    "transitions_close": "</transitions>",
    "context_open": "<context>",
    "context_close": "</context>",
    "conversation_history_open": "<conversationHistory>",
    "conversation_history_close": "</conversationHistory>",
}

# ASCII visualization constants
BOX_STYLES = {
    "default": {
        "topleft": "┌", "topright": "┐", "bottomleft": "└", "bottomright": "┘",
        "horizontal": "─", "vertical": "│", "title_sep": "─"
    },
    "initial": {
        "topleft": "╔", "topright": "╗", "bottomleft": "╚", "bottomright": "╝",
        "horizontal": "═", "vertical": "║", "title_sep": "═"
    },
    "terminal": {
        "topleft": "┏", "topright": "┓", "bottomleft": "┗", "bottomright": "┛",
        "horizontal": "━", "vertical": "┃", "title_sep": "━"
    },
    "both": {  # For states that are both initial and terminal
        "topleft": "╔", "topright": "╗", "bottomleft": "┗", "bottomright": "┛",
        "horizontal": "═", "vertical": "║", "title_sep": "═"
    },
    "section": {
        "topleft": "╭", "topright": "╮", "bottomleft": "╰", "bottomright": "╯",
        "horizontal": "─", "vertical": "│", "title_sep": "─"
    }
}

ARROW_STYLES = {
    "forward": "↓",
    "backward": "↑",
    "self": "⟲",
    "connector": "→",
    "bidirectional": "↔",
    "down_arrow": "▼",
    "right_arrow": "▶",
    "diamond": "◆"
}

ICONS = {
    "input": "✎",      # States that require user input
    "branching": "⎇",   # States with multiple outbound transitions
    "merge": "⊕",       # States with multiple inbound transitions
    "key": "🔑",        # Used for required keys
    "note": "📝"       # For notes and observations
}

# Validator configuration
COMPLEX_STATE_THRESHOLD = 3  # States with more than this many transitions are considered complex