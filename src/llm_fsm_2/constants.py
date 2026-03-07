"""
Enhanced constants and configuration values
"""

# --------------------------------------------------------------
# FSM Definition Constants
# --------------------------------------------------------------

DEFAULT_FSM_VERSION = "4.0"

# --------------------------------------------------------------
# Conversation Management Constants
# --------------------------------------------------------------

DEFAULT_MAX_HISTORY_SIZE = 5
DEFAULT_MAX_MESSAGE_LENGTH = 1000

# --------------------------------------------------------------
# Environment Variable Keys
# --------------------------------------------------------------

ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_LLM_MODEL = "LLM_MODEL"
ENV_LLM_TEMPERATURE = "LLM_TEMPERATURE"
ENV_LLM_MAX_TOKENS = "LLM_MAX_TOKENS"
ENV_FSM_PATH = "FSM_PATH"

# --------------------------------------------------------------
# Feature Flags
# --------------------------------------------------------------

ENABLE_TRANSITION_CACHING = True
ENABLE_PROMPT_OPTIMIZATION = True
ENABLE_CONTEXT_COMPRESSION = False
ENABLE_PARALLEL_EVALUATION = False
ENABLE_SMART_FALLBACKS = True

USE_ENHANCED_JSONLOGIC = True
USE_SMART_CONTEXT_FILTERING = True
USE_ADAPTIVE_THRESHOLDS = False

# --------------------------------------------------------------
# Version and Compatibility Constants
# --------------------------------------------------------------

FRAMEWORK_VERSION = "4.0.0"
API_VERSION = "2.0"
MIGRATION_WARNINGS_ENABLED = True
