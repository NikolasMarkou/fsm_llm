"""
Constants for fsm_llm.monitor package.
"""

from __future__ import annotations

# --- Defaults ---
DEFAULT_REFRESH_INTERVAL = 1.0  # seconds
DEFAULT_MAX_EVENTS = 1000
DEFAULT_MAX_LOG_LINES = 5000
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_MAX_INSTANCES = 200
DEFAULT_MAX_RUNNING_AGENTS = 8
DEFAULT_AGENT_MAX_ITERATIONS = 10
DEFAULT_AGENT_TIMEOUT_SECONDS = 120.0
DEFAULT_FSM_TEMPERATURE = 0.5

# --- Config bounds (enforced by MonitorConfig) ---
MIN_REFRESH_INTERVAL = 0.5  # seconds
MAX_REFRESH_INTERVAL = 60.0
MIN_BUFFER_SIZE = 10
MAX_BUFFER_SIZE = 100_000
MAX_INSTANCES_LIMIT = 10_000
MAX_RUNNING_AGENTS_LIMIT = 256
# Minimum-level ranking for log filters (loguru's numeric levels); the one level list.
LOG_LEVEL_RANK = {
    "TRACE": 5,
    "DEBUG": 10,
    "INFO": 20,
    "SUCCESS": 25,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}
LOG_LEVELS = tuple(LOG_LEVEL_RANK)

# --- Request bounds ---
MAX_MESSAGE_LENGTH = 10_000  # matches core ResponseGenerationRequest.user_message
MAX_TASK_LENGTH = 20_000
MAX_AGENT_ITERATIONS = 100
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MAX_BUILDER_TOKENS = 100_000
MAX_AGENT_TIMEOUT_SECONDS = 3600.0
MAX_STUB_TOOLS = 50
MAX_REQUEST_BODY_BYTES = 1_048_576  # 1 MiB
MAX_BUILDER_SESSIONS = 50

# Host header values the server answers to by default (DNS-rebinding defence);
# these are also the loopback names the CLI recognises.
DEFAULT_TRUSTED_HOSTS = ("localhost", "127.0.0.1", "::1")

# --- OTEL export ---
MAX_OPEN_CONVERSATION_SPANS = (
    10_000  # cap on open conversation spans; oldest ended first
)
MAX_ERROR_TEXT = 200  # error text exported as a span attribute is cut to this

# --- Instance statuses ---
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
ACTIVE_AGENT_STATUSES = frozenset({"running", "cancelling"})

# --- Visualizer page ---
VIZ_EDGE_LABEL_CHARS = 30  # edge label shown on the graph and in its table

# --- Event Types ---
EVENT_CONVERSATION_START = "conversation_start"
EVENT_CONVERSATION_END = "conversation_end"
EVENT_STATE_TRANSITION = "state_transition"
EVENT_PRE_PROCESSING = "pre_processing"
EVENT_POST_PROCESSING = "post_processing"
EVENT_CONTEXT_UPDATE = "context_update"
EVENT_ERROR = "error"

# --- Instance Lifecycle Event Types ---
EVENT_INSTANCE_LAUNCHED = "instance_launched"
EVENT_INSTANCE_DESTROYED = "instance_destroyed"

# --- Workflow Event Types ---
EVENT_WORKFLOW_STARTED = "workflow_started"
EVENT_WORKFLOW_ADVANCED = "workflow_advanced"
EVENT_WORKFLOW_COMPLETED = "workflow_completed"
EVENT_WORKFLOW_CANCELLED = "workflow_cancelled"
EVENT_WORKFLOW_FAILED = "workflow_failed"
EVENT_WORKFLOW_EVENT_DELIVERED = "workflow_event_delivered"

# --- Agent Event Types ---
EVENT_AGENT_STARTED = "agent_started"
EVENT_AGENT_COMPLETED = "agent_completed"
EVENT_AGENT_FAILED = "agent_failed"
EVENT_AGENT_ITERATION = "agent_iteration"
EVENT_AGENT_CANCELLED = "agent_cancelled"
EVENT_AGENT_TOOL_CALL = "agent_tool_call"

# --- Handler ---
MONITOR_HANDLER_NAME = "fsm_llm.monitor"
MONITOR_HANDLER_PRIORITY = 9999  # Lowest priority — observe only
