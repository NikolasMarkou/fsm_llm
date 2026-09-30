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

# --- Config bounds (enforced by MonitorConfig) ---
MIN_REFRESH_INTERVAL = 0.5  # seconds
MAX_REFRESH_INTERVAL = 60.0
MIN_BUFFER_SIZE = 10
MAX_BUFFER_SIZE = 100_000
LOG_LEVELS = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")

# --- Request bounds ---
MAX_MESSAGE_LENGTH = 10_000  # matches core ResponseGenerationRequest.user_message
MAX_TASK_LENGTH = 20_000
MAX_AGENT_ITERATIONS = 100
MAX_AGENT_TIMEOUT_SECONDS = 3600.0
MAX_STUB_TOOLS = 50
MAX_REQUEST_BODY_BYTES = 1_048_576  # 1 MiB
MAX_BUILDER_SESSIONS = 50

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
# Reserved / public export. Log records flow through EventCollector.record_log
# (a separate channel), NOT record_event, so this type is never emitted as a
# MonitorEvent. Kept as a stable public symbol; do not rely on it for routing.
EVENT_LOG = "log"

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
