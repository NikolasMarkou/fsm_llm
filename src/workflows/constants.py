"""
Constants for the FSM-LLM Workflow System.

Engine-owned context keys, engine limits, and the step types that pause a
workflow (and therefore break a synchronous step chain).
"""

from __future__ import annotations

# --------------------------------------------------------------
# Engine-owned context keys
# --------------------------------------------------------------

#: Set by ``WaitEventStep``; tells the engine to park the instance and
#: register an event listener. Cleared on every transition.
CONTEXT_KEY_WAITING_INFO = "_waiting_info"
#: Set by ``TimerStep``; tells the engine to park the instance and schedule a
#: timer. Cleared on every transition.
CONTEXT_KEY_TIMER_INFO = "_timer_info"
#: ``{"workflow_id", "instance_id"}`` of the running instance.
CONTEXT_KEY_WORKFLOW_INFO = "_workflow_info"
#: Written when an event wait times out: ``{"event_type", "timeout_at"}``.
CONTEXT_KEY_EVENT_TIMEOUT = "_timeout"
#: Written when a timer fires: ``{"expired_at"}``.
CONTEXT_KEY_TIMER_EXPIRED = "_timer_expired"
#: The last event delivered to the instance (``WorkflowEvent.model_dump()``).
CONTEXT_KEY_LAST_EVENT = "_last_event"
#: ``user_input`` of the latest ``advance`` call. Present only while
#: the step that call re-runs is executing; removed afterwards.
CONTEXT_KEY_USER_INPUT = "_user_input"
#: Reason passed to ``cancel``.
CONTEXT_KEY_CANCELLATION_REASON = "_cancellation_reason"

#: Internal keys a step result may set (they bypass the internal-prefix
#: filter because the engine reads them to detect waiting/timer steps).
STEP_INTERNAL_WHITELIST = frozenset({CONTEXT_KEY_WAITING_INFO, CONTEXT_KEY_TIMER_INFO})

# --------------------------------------------------------------
# Engine limits and defaults
# --------------------------------------------------------------

#: Maximum number of steps one engine call (start, advance, event, timer)
#: may execute before the instance is failed. Synchronous cycles are rejected
#: at registration, so this only bounds custom steps with dynamic routing and
#: very long chains. Configurable per engine (``max_steps``).
DEFAULT_MAX_STEPS = 1000

#: Default cap on terminal (completed/failed/cancelled) instances kept in
#: memory; the oldest are purged first. ``None`` disables the cap.
DEFAULT_MAX_COMPLETED_INSTANCES = 1000

#: Default cap on history entries kept per instance (oldest dropped first).
DEFAULT_MAX_HISTORY_ENTRIES = 1000

#: Targeted events (``WorkflowEvent.instance_id`` set) that no listener of the
#: target instance matched are buffered for that instance, up to this many.
MAX_BUFFERED_EVENTS_PER_INSTANCE = 100

#: ParallelStep logs a memory warning above this many children.
PARALLEL_DEEPCOPY_WARNING_THRESHOLD = 10
