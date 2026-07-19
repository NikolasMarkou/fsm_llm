from __future__ import annotations

"""
Constants and configuration values for the FSM-LLM framework.
"""

import re
from collections.abc import Iterable

# --------------------------------------------------------------
# Internal Context Key Prefixes
# --------------------------------------------------------------

INTERNAL_KEY_PREFIXES = ["_", "system_", "internal_", "__"]


# DECISION plan-2026-07-19-4b664252/D-009
# This helper exists because the identical `any(key.startswith(p) ...)`
# expression was copy-pasted to FIVE call sites (context.py, fsm.py,
# definitions.py x2, prompts.py) and NONE of them case-folded -- so
# `SYSTEM_foo` bypassed the filter at every site at once (F-13).
# Do NOT re-inline `key.startswith(...)` at a call site, and do NOT hardcode
# the prefix list: five hand-maintained copies is what caused this defect.
# Import this helper instead. See decisions.md D-009.
def has_internal_prefix(key: str, prefixes: Iterable[str] | None = None) -> bool:
    """Return True if ``key`` carries an internal context-key prefix.

    Matching is CASE-INSENSITIVE: ``SYSTEM_foo`` and ``Internal_x`` are
    internal keys just as ``system_foo`` and ``internal_x`` are.

    Args:
        key: The context key to test.
        prefixes: Prefixes to test against. Defaults to
            :data:`INTERNAL_KEY_PREFIXES`. Callers with a configurable
            prefix list (e.g. ``PromptConfig.internal_key_prefixes``) pass
            their own; the case-folding rule is applied either way.

    Returns:
        True if the key should be treated as internal (and therefore hidden
        from user-visible data and from LLM prompts). Never raises.
    """
    if prefixes is None:
        prefixes = INTERNAL_KEY_PREFIXES
    lowered = key.lower()
    return any(lowered.startswith(prefix.lower()) for prefix in prefixes)


# --------------------------------------------------------------
# LLM Configuration Defaults
# --------------------------------------------------------------

DEFAULT_LLM_MODEL = "ollama_chat/qwen3.5:4b"
DEFAULT_TEMPERATURE = 0.5

# --------------------------------------------------------------
# Conversation Management Constants
# --------------------------------------------------------------

# History management defaults
DEFAULT_MAX_HISTORY_SIZE = 5
DEFAULT_MAX_MESSAGE_LENGTH = 1000

# Message processing constants
MESSAGE_TRUNCATION_SUFFIX = "... [truncated]"

# FSM stacking depth limit
DEFAULT_MAX_STACK_DEPTH = 10

# --------------------------------------------------------------
# LLM Interface Constants
# --------------------------------------------------------------

# Log preview lengths
LOG_MESSAGE_PREVIEW_LENGTH = 100
LOG_RESPONSE_PREVIEW_LENGTH = 200

# FSM ID generation
FSM_ID_HASH_LENGTH = 8

# --------------------------------------------------------------
# Transition Evaluation Constants
# --------------------------------------------------------------

# Transition evaluation internals
PRIORITY_SCALING_DIVISOR = 1000.0
MIN_BASE_CONFIDENCE = 0.1
CONDITION_SUCCESS_RATE_BOOST = 0.5
FLOAT_EQUALITY_EPSILON = 1e-9

# Classification-aware transition defaults
DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE = 0.6
TRANSITION_CLASSIFICATION_FALLBACK_INTENT = "_stay_in_state"
CONTEXT_KEY_CLASSIFICATION_RESULT = "_transition_classification_result"

# Classification extraction defaults
CLASSIFICATION_EXTRACTION_RESULT_SUFFIX = "_classification"

# --------------------------------------------------------------
# Environment Variable Keys
# --------------------------------------------------------------

ENV_LLM_MODEL = "LLM_MODEL"
ENV_LLM_TEMPERATURE = "LLM_TEMPERATURE"
ENV_LLM_MAX_TOKENS = "LLM_MAX_TOKENS"
ENV_FSM_PATH = "FSM_PATH"

# --------------------------------------------------------------
# Validation and Safety Constants
# --------------------------------------------------------------

# Security constants
ALLOWED_JSONLOGIC_OPERATIONS = {
    "==",
    "!=",
    "===",
    "!==",
    ">",
    ">=",
    "<",
    "<=",
    "and",
    "or",
    "!",
    "!!",
    "if",
    "var",
    "missing",
    "missing_some",
    "+",
    "-",
    "*",
    "/",
    "%",
    "min",
    "max",
    "cat",
    "in",
    "contains",
    "has_context",
    "context_length",
}

# DECISION plan-2026-07-19-4b664252/D-009
# These patterns fail in BOTH directions, and both are real harm:
#   - Over-match STRIPS legitimate user data out of context and out of the
#     LLM prompt, silently degrading replies. The old password pattern's
#     trailing `(?:.*|$)` was VACUOUS (`.*` always matches), so every
#     "password"-containing flag -- `passwordless_login`,
#     `forgot_password_supported` -- was destroyed as collateral.
#   - Under-match LEAKS secrets into prompts. The plural forms `secrets`,
#     `access_tokens`, `private_keys`, `oauth_tokens` all passed through
#     untouched because only `credential(?:s)?` had been given the optional
#     plural.
# Every term therefore ends in an explicit boundary -- `[\W_]`, `$`, or an
# optional `(?:s)?`/digit-suffix -- and NEVER a bare `.*`. Do NOT "simplify"
# a trailing `(?:[\W_].*|$)` to `.*`: that re-creates the over-match, and
# `secretary`/`secretariat`/`access_tokenizer`/`private_keystone` are the
# pinned near-miss negatives that catch it.
# Any change here must be re-checked against the ADVERSARIAL negative set in
# tests/test_fsm_llm/test_context_unit.py, where each negative is maximally
# similar to a positive. A negative set of obviously-safe keys ("username",
# "email") validates whatever the implementation happens to do.
# Accepted gap: `password_hash` is now KEPT. No regex can separate it from
# `password_reset_flow_enabled` (both are `password_<suffix>`); SC-9 requires
# the latter be kept. See decisions.md D-009.
FORBIDDEN_CONTEXT_PATTERNS = [
    r"(?:^|.*[\W_])password(?:s)?(?:[-_.]?\d+)?$",  # Password keys, incl. password123/passwords (not "passwordless_login")
    r"(?:^|.*[\W_])secret(?:s)?(?:[\W_].*|$)",  # Secret-related keys (not "secretary")
    r"(?:^|.*[\W_])(?:api[-_.]?token|auth[-_.]?token|access[-_.]?token|refresh[-_.]?token|bearer[-_.]?token)(?:s)?(?:[\W_].*|$)",  # Auth token keys (not "tokenizer")
    r".*(?:api[-_.]?key|key[-_.]?api).*",  # API key patterns (both orderings, with dash/underscore/dot)
    r"(?:^|.*[\W_])credential(?:s)?(?:[\W_].*|$)",  # Credential-related keys
    r"(?:^|.*[\W_])private[-_.]?key(?:s)?(?:[\W_].*|$)",  # Private key patterns
    r"(?:^|.*[\W_])oauth[-_.]?token(?:s)?(?:[\W_].*|$)",  # OAuth token patterns
]

# Pre-compiled versions for performance (avoid recompiling in loops)
COMPILED_FORBIDDEN_CONTEXT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in FORBIDDEN_CONTEXT_PATTERNS
]

# --------------------------------------------------------------
# Logging Configuration
# --------------------------------------------------------------

ENV_LOG_LEVEL = "FSM_LLM_LOG_LEVEL"
ENV_LOG_FORMAT = "FSM_LLM_LOG_FORMAT"

LOG_FORMAT_HUMAN = "human"
LOG_FORMAT_JSON = "json"

LOG_SINK_STDERR = "stderr"
LOG_SINK_STDOUT = "stdout"
LOG_SINK_FILE = "file"

LOG_DEFAULT_LEVEL = "DEBUG"
LOG_DEFAULT_ROTATION = "10 MB"
LOG_DEFAULT_RETENTION = "1 month"
LOG_DEFAULT_COMPRESSION = "zip"
LOG_DEFAULT_FILE_PATTERN = "fsm-llm_{time}.log"

LOG_HUMAN_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}:{function}:{line}</cyan> | "
    "{message}"
)

LOG_HUMAN_FORMAT_WITH_CONTEXT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "conv:<yellow>{extra[conversation_id]:<12}</yellow> | "
    "<cyan>{name}:{function}:{line}</cyan> | "
    "{message}"
)

LOG_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | "
    "{level: <8} | "
    "conv_id: {extra[conversation_id]:<12} | "
    "{name}:{function}:{line} | "
    "{message}"
)

# Standard context field names for structured logging
LOG_FIELD_TIMESTAMP = "timestamp"
LOG_FIELD_LEVEL = "level"
LOG_FIELD_MESSAGE = "message"
LOG_FIELD_MODULE = "module"
LOG_FIELD_FUNCTION = "function"
LOG_FIELD_LINE = "line"
LOG_FIELD_CONVERSATION_ID = "conversation_id"
LOG_FIELD_PACKAGE = "package"

# Default value for unbound context fields
LOG_DEFAULT_CONVERSATION_ID = "GENERAL"

# --------------------------------------------------------------
# Timeout Defaults
# --------------------------------------------------------------

# Handler execution timeout (seconds). None = no timeout.
# Recommended: set handler_timeout=DEFAULT_HANDLER_TIMEOUT when creating
# HandlerSystem for safety against handlers that block indefinitely.
DEFAULT_HANDLER_TIMEOUT = 30.0

# Workflow step execution timeout (seconds). None = no timeout.
DEFAULT_STEP_TIMEOUT = 120.0
