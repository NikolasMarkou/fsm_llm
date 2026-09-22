"""
Constants and configuration values for the FSM-LLM framework.
"""

from __future__ import annotations

# The context-key security filters live in `security.py`; this block keeps
# every existing `fsm_llm.constants` import of them working.
from .security import (  # noqa: F401
    _AUTH_SCHEME_WORDS,
    _BEARER_TOKEN_QUALIFIERS,
    _CREDENTIAL_NAME_RE,
    _CREDENTIAL_VALUE_CHARSET_RE,
    _CREDENTIAL_VALUE_PREFIXES,
    _CRYPTO_GAP,
    _CRYPTO_KEY_QUALIFIERS,
    _IDENTIFIER_NOUN_VOCABULARY,
    _KEY_MATERIAL_HEADS,
    _MIN_CREDENTIAL_VALUE_LENGTH,
    _PASSWORD_POLICY_SUFFIXES,
    _PATH_VALUE_RE,
    _SAFE_TOKEN_QUALIFIERS,
    _SEP_RUN,
    _TOKEN_MATERIAL_HEADS,
    _TOKEN_VALUE_SCAN_NAME_RE,
    _VALUE_SCAN_LIMIT,
    COMPILED_FORBIDDEN_CONTEXT_PATTERNS,
    FORBIDDEN_CONTEXT_PATTERNS,
    INTERNAL_KEY_PREFIXES,
    _colon_composite_tail,
    _generic_shape_is_credential,
    _looks_like_credential_value,
    _normalise_credential_value,
    _shannon_entropy,
    _token_value_is_credential,
    has_internal_prefix,
    is_forbidden_context_entry,
)

# DECISION plan-2026-07-19T191147-4b664252/D-011 [STALE]
# One bound for BOTH recursive context filters (`context.clean_context_keys`
# and `prompts.BasePromptBuilder._filter_context_for_security`). Do NOT
# re-declare a local depth limit in either module: two hand-maintained copies
# of a security bound is the same duplication that produced F-13, and a filter
# that stops one level shallower than its sibling is a bypass. The behaviour AT
# the bound is fail-CLOSED in both: a container nested deeper is DROPPED, never
# passed through unfiltered (D-010). See decisions.md D-010, D-011.
MAX_CONTEXT_FILTER_DEPTH = 16

# DECISION plan-2026-09-21T203800-8a03483a/D-011
# DECISION plan-2026-09-21T203800-8a03483a/D-045
# Work bound for the PROMPT walker (`prompts.BasePromptBuilder`): every value
# visited costs one node, and past the budget the rest is DROPPED from the
# prompt (fail-closed, like the depth bound). The depth bound does not bound
# work on aliased input (3-way aliasing at 14 levels never finished). Do NOT
# apply this truncating budget to `utilities.filter_context_tree` (`get_data`,
# `save_session`, the extracted-data commit): truncating DATA silently cut a
# 150,000-item list and dropped later keys. That walker memoises every
# container that cannot reach a cycle (D-052) and raises
# `ContextFilterWorkError` when cycle-reaching containers unfold past
# `MAX_CONTEXT_FILTER_NODES + CONTEXT_FILTER_CYCLIC_WORK_FACTOR * distinct
# items`. See decisions.md D-011, D-045.
MAX_CONTEXT_FILTER_NODES = 100_000
CONTEXT_FILTER_CYCLIC_WORK_FACTOR = 16


# --------------------------------------------------------------
# LLM Configuration Defaults
# --------------------------------------------------------------

DEFAULT_LLM_MODEL = "ollama_chat/qwen3.5:4b"
DEFAULT_TEMPERATURE = 0.5

# litellm.completion() kwargs the framework owns on every call. A constructor's
# pass-through ``**kwargs`` may not set them: ``stream`` would hand a
# non-streaming parser a stream object, and ``response_format`` would force (or
# silently lose) structured output the parser does not expect (audit D12).
# Shared by ``LiteLLMInterface`` and ``Classifier``.
RESERVED_LLM_CALL_KWARGS = frozenset(
    {"model", "messages", "temperature", "max_tokens", "stream", "response_format"}
)

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

# FSM ID generation
FSM_ID_HASH_LENGTH = 8

# --------------------------------------------------------------
# Transition Evaluation Constants
# --------------------------------------------------------------

# Classification-aware transition defaults
DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE = 0.6
# Reserved key of `State.transition_classification`: its value is the threshold,
# every other key names a target state (validated in definitions.State).
TRANSITION_CLASSIFICATION_THRESHOLD_KEY = "confidence_threshold"
TRANSITION_CLASSIFICATION_FALLBACK_INTENT = "_stay_in_state"
CONTEXT_KEY_CLASSIFICATION_RESULT = "_transition_classification_result"
CONTEXT_KEY_AGENT_TRACE = "agent_trace"

# Framework-seeded context keys a handler delta may neither set nor delete
# (MessagePipeline.execute_handlers merge_delta, C3). Closed on purpose: other
# internal-prefixed keys a handler owns (agents' `_replan_count`) still merge.
RESERVED_CONTEXT_KEYS: frozenset[str] = frozenset(
    {
        "_conversation_id",
        "_conversation_start",
        "_timestamp",
        "_fsm_id",
        "_previous_state",
        "_current_state",
        "_transition_timestamp",
        "_error",
        "_traceback",
        "_inherited_history",
        "_sub_conversation_summary",
        CONTEXT_KEY_CLASSIFICATION_RESULT,
    }
)

# `context.metadata` keys for full classification records (A4, D-005): the
# latest result per classification-extraction field, and the current turn's
# transition-classification record (mirror of CONTEXT_KEY_CLASSIFICATION_RESULT).
METADATA_KEY_CLASSIFICATION_RESULTS = "classification_results"
METADATA_KEY_TRANSITION_CLASSIFICATION = "transition_classification"

# `context.metadata` key holding {context key: value digest} for every value the
# pipeline itself extracted (never the values). D-015 of the pipeline.
PROVENANCE_METADATA_KEY = "_pipeline_extracted"

# Context key an agent (`fsm_llm_agents.BaseAgent._init_context`) sets to the
# structured-output schema; Pass 2 enforces it on terminal states only.
CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT = "_output_response_format"

# Recent exchanges (user+assistant pairs) the pipeline gives each classifier
# call as history (MessagePipeline._build_classifier_context, D-004).
CLASSIFIER_HISTORY_EXCHANGES = 3

# Upper bound on intents in one multi-intent classification. Single source for
# the `MultiClassificationResult.intents` pydantic cap (definitions.py), the
# `ClassificationPromptConfig.max_intents` bound (prompts.py) and the
# truncation in `Classifier.classify_multi` (classification.py).
MAX_MULTI_INTENTS = 5

# Upper bound on cached `Classifier` instances per `MessagePipeline`
# (`MessagePipeline._get_classifier`), keyed on a content hash of schema +
# model + prompt config + connection kwargs; the oldest entry is evicted at
# the bound. Matches `DEFAULT_MAX_FSM_CACHE_SIZE`.
MAX_CLASSIFIER_CACHE_SIZE = 64

# Default bound of `FSMManager`'s FSM definition LRU cache (`max_fsm_cache_size`,
# also settable through `API(max_fsm_cache_size=...)`).
DEFAULT_MAX_FSM_CACHE_SIZE = 64

# How long FSMManager.end_conversation waits for a running turn to release the
# conversation lock before refusing with FSMError (see the anchor in
# FSMManager.end_conversation).
END_CONVERSATION_LOCK_TIMEOUT_SECONDS = 30.0

# --------------------------------------------------------------
# Environment Variable Keys
# --------------------------------------------------------------

ENV_LLM_MODEL = "LLM_MODEL"
ENV_LLM_TEMPERATURE = "LLM_TEMPERATURE"
ENV_LLM_MAX_TOKENS = "LLM_MAX_TOKENS"
ENV_FSM_PATH = "FSM_PATH"

# --------------------------------------------------------------
# CLI Exit Codes (fsm-llm, fsm-llm-validate, fsm-llm-visualize)
# --------------------------------------------------------------

CLI_EXIT_OK = 0
CLI_EXIT_FAILURE = 1
# 128 + SIGINT, the shell convention for a Ctrl-C'd process.
CLI_EXIT_INTERRUPTED = 130

# --------------------------------------------------------------
# Validation and Safety Constants
# --------------------------------------------------------------

#: Deepest operator nesting `evaluate_logic` accepts (DoS guard). It lives here,
#: not in expressions.py, so `definitions.TransitionCondition` can enforce the
#: same bound at load time without importing expressions (which imports
#: definitions). Both import it from this one place.
MAX_JSONLOGIC_DEPTH = 50

#: JsonLogic operators whose arguments `evaluate_logic` reads as raw DATA (a
#: var name and default, key names) and never evaluates as logic. The load-time
#: walk in definitions.py skips their arguments for the same reason.
JSONLOGIC_RAW_ARGUMENT_OPERATIONS: frozenset[str] = frozenset(
    {"var", "missing", "missing_some"}
)

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

# Default value for unbound context fields
LOG_DEFAULT_CONVERSATION_ID = "GENERAL"

# --------------------------------------------------------------
# Timeout Defaults
# --------------------------------------------------------------

# Per-HandlerSystem cap on timed-handler threads still running after their
# timeout (stragglers). At the cap a new timed call fails like a timeout
# instead of starting another thread (the pre-C2 pool also had 4 workers).
# One HandlerSystem serves every conversation of an `API`, so the cap is
# shared API-wide, not per conversation.
MAX_TIMED_HANDLER_STRAGGLERS = 4
