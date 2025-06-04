# /constants.py

"""
Enhanced constants and configuration values for 2-Pass LLM-FSM Architecture.

This module centralizes all constants used throughout the enhanced framework
including new constants for the 2-pass architecture, transition evaluation,
and improved prompt management.

Key Additions:
- Constants for 2-pass architecture components
- Transition evaluation configuration
- Enhanced prompt building parameters
- Performance optimization settings
"""

# --------------------------------------------------------------
# FSM Definition Constants
# --------------------------------------------------------------

# Default FSM version for new architecture
DEFAULT_FSM_VERSION = "4.0"

# Transition evaluation constants
DEFAULT_TRANSITION_PRIORITY = 100
TRANSITION_PRIORITY_MIN = 0
TRANSITION_PRIORITY_MAX = 1000

# Condition evaluation constants
DEFAULT_CONDITION_PRIORITY = 100
CONDITION_EVALUATION_TIMEOUT = 5.0  # seconds


# --------------------------------------------------------------
# Conversation Management Constants
# --------------------------------------------------------------

# History management defaults
DEFAULT_MAX_HISTORY_SIZE = 5
DEFAULT_MAX_MESSAGE_LENGTH = 1000

# Message processing constants
DEFAULT_MESSAGE_TRUNCATE_LENGTH = 50
MESSAGE_TRUNCATION_SUFFIX = "... [truncated]"

# Context filtering constants
INTERNAL_KEY_PREFIXES = ['_', 'system_', 'internal_', '__']
BLOCKED_CONTEXT_KEYS = {'fsm_stack', 'handler_data', 'system_metadata'}


# --------------------------------------------------------------
# LLM Interface Constants
# --------------------------------------------------------------

# Default LLM configuration
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 1000
DEFAULT_LLM_MODEL = "gpt-4o-mini"

# Request timeout and retry settings
LLM_REQUEST_TIMEOUT = 30.0  # seconds
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 1.0  # seconds

# Response processing constants
JSON_EXTRACTION_MAX_ATTEMPTS = 3
CONTENT_RESPONSE_MAX_LENGTH = 5000
TRANSITION_RESPONSE_MAX_LENGTH = 200


# --------------------------------------------------------------
# Prompt Building Constants
# --------------------------------------------------------------

# Content generation prompt limits
CONTENT_PROMPT_MAX_LENGTH = 20000
CONTENT_HISTORY_TOKEN_BUDGET = 3000
CONTENT_CONTEXT_MAX_KEYS = 50

# Transition decision prompt limits
TRANSITION_PROMPT_MAX_LENGTH = 10000
TRANSITION_OPTIONS_MAX_COUNT = 10
TRANSITION_DESCRIPTION_MAX_LENGTH = 300

# Token estimation constants (conservative estimates)
CHARS_PER_TOKEN = 2.5
TOKEN_ESTIMATION_SAFETY_FACTOR = 1.3
JSON_OVERHEAD_FACTOR = 1.3
UTF8_EXPANSION_FACTOR = 1.5
CDATA_OVERHEAD_TOKENS = 50

# Prompt template constants
XML_TAG_ESCAPE_PATTERN = r'</?(?:{})[^>]*/?>'
CRITICAL_PROMPT_TAGS = [
    "task", "content_generation", "persona", "current_objective",
    "conversation_history", "current_context", "response_format",
    "instructions", "examples", "transition_decision", "available_options"
]


# --------------------------------------------------------------
# Transition Evaluation Constants
# --------------------------------------------------------------

# Evaluation thresholds
DEFAULT_AMBIGUITY_THRESHOLD = 0.1
DEFAULT_MINIMUM_CONFIDENCE = 0.5
DEFAULT_AUTO_TRANSITION_THRESHOLD = 0.8

# Evaluation modes
TRANSITION_MODE_DETERMINISTIC = "deterministic"
TRANSITION_MODE_LLM = "llm"
TRANSITION_MODE_HYBRID = "hybrid"

# Priority ranges for different transition types
PRIORITY_HIGH_MIN = 1
PRIORITY_HIGH_MAX = 50
PRIORITY_NORMAL_MIN = 51
PRIORITY_NORMAL_MAX = 100
PRIORITY_LOW_MIN = 101
PRIORITY_LOW_MAX = 1000

# Condition evaluation constants
CONDITION_EVALUATION_MAX_DEPTH = 10
JSONLOGIC_MAX_OPERATIONS = 100
JSONLOGIC_TIMEOUT = 2.0  # seconds


# --------------------------------------------------------------
# Handler System Constants
# --------------------------------------------------------------

# Handler execution priorities
HANDLER_PRIORITY_CRITICAL = 1
HANDLER_PRIORITY_HIGH = 10
HANDLER_PRIORITY_NORMAL = 50
HANDLER_PRIORITY_LOW = 100
HANDLER_PRIORITY_BACKGROUND = 200

# Handler error modes
HANDLER_ERROR_CONTINUE = "continue"
HANDLER_ERROR_RAISE = "raise"
HANDLER_ERROR_SKIP = "skip"

# System handler keys (for filtering)
SYSTEM_HANDLER_KEYS = {
    'system', 'handlers', '_handler_execution_log',
    '_handler_errors', '_last_handler_run'
}


# --------------------------------------------------------------
# Environment Variable Keys
# --------------------------------------------------------------

ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
ENV_LLM_MODEL = "LLM_MODEL"
ENV_LLM_TEMPERATURE = "LLM_TEMPERATURE"
ENV_LLM_MAX_TOKENS = "LLM_MAX_TOKENS"
ENV_FSM_PATH = "FSM_PATH"

# New environment variables for 2-pass architecture
ENV_TRANSITION_MODE = "FSM_TRANSITION_MODE"
ENV_AMBIGUITY_THRESHOLD = "FSM_AMBIGUITY_THRESHOLD"
ENV_ENABLE_DETERMINISTIC_TRANSITIONS = "FSM_ENABLE_DETERMINISTIC"
ENV_DEBUG_TRANSITIONS = "FSM_DEBUG_TRANSITIONS"


# --------------------------------------------------------------
# Logging Configuration
# --------------------------------------------------------------

LOG_ROTATION_SIZE = "10 MB"
LOG_RETENTION_PERIOD = "1 month"
LOG_COMPRESSION = "zip"

# Log levels for different components
LOG_LEVEL_FSM_MANAGER = "INFO"
LOG_LEVEL_TRANSITION_EVALUATOR = "INFO"
LOG_LEVEL_PROMPT_BUILDER = "INFO"
LOG_LEVEL_LLM_INTERFACE = "INFO"
LOG_LEVEL_HANDLER_SYSTEM = "DEBUG"

# Conversation logging constants
CONVERSATION_LOG_MAX_MESSAGE_LENGTH = 200
CONVERSATION_LOG_MAX_CONTEXT_KEYS = 20


# --------------------------------------------------------------
# Performance and Optimization Constants
# --------------------------------------------------------------

# Caching settings
FSM_DEFINITION_CACHE_SIZE = 100
PROMPT_CACHE_SIZE = 50
TRANSITION_EVALUATION_CACHE_SIZE = 200

# Concurrency settings
MAX_CONCURRENT_CONVERSATIONS = 1000
CONVERSATION_CLEANUP_INTERVAL = 300  # seconds
HANDLER_EXECUTION_TIMEOUT = 10.0  # seconds

# Memory management
CONTEXT_DATA_MAX_SIZE = 1000000  # bytes
CONVERSATION_HISTORY_MAX_SIZE = 100  # exchanges
AUTO_CLEANUP_THRESHOLD = 0.8  # cleanup when 80% of limits reached


# --------------------------------------------------------------
# Validation and Safety Constants
# --------------------------------------------------------------

# FSM validation limits
MAX_FSM_STATES = 1000
MAX_TRANSITIONS_PER_STATE = 50
MAX_CONDITIONS_PER_TRANSITION = 20
MAX_FSM_DEFINITION_SIZE = 10000000  # bytes

# Input validation limits
MAX_USER_MESSAGE_LENGTH = 10000
MAX_CONTEXT_KEY_LENGTH = 100
MAX_CONTEXT_VALUE_LENGTH = 10000
MAX_STATE_NAME_LENGTH = 100

# Security constants
ALLOWED_JSONLOGIC_OPERATIONS = {
    '==', '!=', '===', '!==', '>', '>=', '<', '<=',
    'and', 'or', '!', 'if', 'var', 'missing', 'missing_some',
    '+', '-', '*', '/', '%', 'min', 'max', 'cat',
    'in', 'contains', 'context_has', 'context_count'
}

FORBIDDEN_CONTEXT_PATTERNS = [
    r'__.*__',  # Double underscore patterns
    r'.*password.*',  # Password-related keys
    r'.*secret.*',  # Secret-related keys
    r'.*token.*',  # Token-related keys
    r'.*key.*api.*'  # API key patterns
]


# --------------------------------------------------------------
# Visualization Constants (for ASCII diagrams)
# --------------------------------------------------------------

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
    "both": {
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
    "input": "✎",
    "branching": "⎇",
    "merge": "⊕",
    "key": "🔑",
    "note": "📝",
    "content": "💬",
    "transition": "🔄",
    "deterministic": "⚡",
    "llm_assisted": "🤖"
}

# Validator configuration
COMPLEX_STATE_THRESHOLD = 3


# --------------------------------------------------------------
# Feature Flags and Experimental Settings
# --------------------------------------------------------------

# Feature flags for new capabilities
ENABLE_TRANSITION_CACHING = True
ENABLE_PROMPT_OPTIMIZATION = True
ENABLE_CONTEXT_COMPRESSION = False  # Experimental
ENABLE_PARALLEL_EVALUATION = False  # Experimental
ENABLE_SMART_FALLBACKS = True

# Debug and development flags
DEBUG_TRANSITION_EVALUATION = False
DEBUG_PROMPT_GENERATION = False
DEBUG_CONTEXT_FILTERING = False
DEBUG_HANDLER_EXECUTION = False
VERBOSE_LOGGING = False

# A/B testing flags
USE_ENHANCED_JSONLOGIC = True
USE_SMART_CONTEXT_FILTERING = True
USE_ADAPTIVE_THRESHOLDS = False  # Experimental


# --------------------------------------------------------------
# Version and Compatibility Constants
# --------------------------------------------------------------

# Framework version
FRAMEWORK_VERSION = "4.0.0"
API_VERSION = "2.0"
MINIMUM_PYTHON_VERSION = "3.8"

# Compatibility settings
BACKWARD_COMPATIBILITY_MODE = True
LEGACY_RESPONSE_FORMAT_SUPPORT = True
MIGRATION_WARNINGS_ENABLED = True

# Version-specific constants
V3_COMPATIBILITY_SHIM = True
V4_ENHANCED_FEATURES = True