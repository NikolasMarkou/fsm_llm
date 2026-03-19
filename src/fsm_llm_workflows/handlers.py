from __future__ import annotations

"""
Handlers for integrating workflows with FSM-LLM.
"""

from typing import TYPE_CHECKING

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from fsm_llm.logging import logger
from fsm_llm.handlers import BaseHandler, HandlerTiming

# --------------------------------------------------------------

if TYPE_CHECKING:
    from .engine import WorkflowEngine
    from fsm_llm.fsm import FSMManager

# --------------------------------------------------------------


class AutoTransitionHandler(BaseHandler):
    """Handler that detects and executes automatic transitions in FSM-LLM."""

    def __init__(self, workflow_engine: 'WorkflowEngine', fsm_manager: 'FSMManager'):
        """Initialize the auto transition handler."""
        super().__init__(name="AutoTransitionHandler", priority=10)
        self.workflow_engine = workflow_engine
        self.fsm_manager = fsm_manager

    def should_execute(self, timing, current_state, target_state, context, updated_keys=None):
        """Determine if this handler should execute."""
        if timing == HandlerTiming.POST_TRANSITION:
            workflow_info = context.get("_workflow_info", {})
            is_auto_transition = workflow_info.get("auto_transition_states", {}).get(target_state, False)
            return is_auto_transition
        return False

    def execute(self, context):
        """Execute the handler logic — schedule auto-advance synchronously."""
        conversation_id = context.get("_conversation_id")
        if not conversation_id:
            logger.error("No conversation ID found in context for auto transition")
            return {"_auto_transition_error": "No conversation ID"}

        workflow_info = context.get("_workflow_info", {})
        workflow_instance_id = workflow_info.get("instance_id")

        if not workflow_instance_id:
            logger.error("No workflow instance ID found in context for auto transition")
            return {"_auto_transition_error": "No workflow instance ID"}

        # Mark for deferred auto-advance instead of creating a fire-and-forget task.
        # The WorkflowEngine should poll this flag after handler dispatch.
        return {
            "_auto_transition_scheduled": True,
            "_auto_transition_instance_id": workflow_instance_id,
        }


class EventHandler(BaseHandler):
    """Handler that processes external events in FSM-LLM."""

    def __init__(self, workflow_engine: 'WorkflowEngine', fsm_manager: 'FSMManager'):
        """Initialize the event handler."""
        super().__init__(name="EventHandler", priority=20)
        self.workflow_engine = workflow_engine
        self.fsm_manager = fsm_manager

    def should_execute(self, timing, current_state, target_state, context, updated_keys=None):
        """Determine if this handler should execute."""
        if timing == HandlerTiming.POST_TRANSITION:
            waiting_info = context.get("_waiting_info", {})
            return waiting_info.get("waiting_for_event", False)
        return False

    def execute(self, context):
        """Execute the handler logic — register event listener synchronously."""
        workflow_info = context.get("_workflow_info", {})
        workflow_instance_id = workflow_info.get("instance_id")

        if not workflow_instance_id:
            logger.error("No workflow instance ID found in context for event handler")
            return {"_event_listener_error": "No workflow instance ID"}

        waiting_info = context.get("_waiting_info", {})
        event_type = waiting_info.get("event_type")

        if not event_type:
            logger.error("No event type specified in waiting info")
            return {"_event_listener_error": "No event type specified"}

        # Mark for deferred event registration
        return {
            "_event_listener_pending": True,
            "_event_listener_instance_id": workflow_instance_id,
            "_event_listener_type": event_type,
            "_event_listener_success_state": waiting_info.get("success_state"),
            "_event_listener_timeout_seconds": waiting_info.get("timeout_seconds"),
            "_event_listener_timeout_state": waiting_info.get("timeout_state"),
            "_event_listener_event_mapping": waiting_info.get("event_mapping", {}),
        }

# --------------------------------------------------------------


class TimerHandler(BaseHandler):
    """Handler that processes timers in FSM-LLM."""

    def __init__(self, workflow_engine: 'WorkflowEngine', fsm_manager: 'FSMManager'):
        """Initialize the timer handler."""
        super().__init__(name="TimerHandler", priority=30)
        self.workflow_engine = workflow_engine
        self.fsm_manager = fsm_manager

    def should_execute(self, timing, current_state, target_state, context, updated_keys=None):
        """Determine if this handler should execute."""
        if timing == HandlerTiming.POST_TRANSITION:
            timer_info = context.get("_timer_info", {})
            return timer_info.get("waiting_for_timer", False)
        return False

    def execute(self, context):
        """Execute the handler logic — schedule timer synchronously."""
        workflow_info = context.get("_workflow_info", {})
        workflow_instance_id = workflow_info.get("instance_id")

        if not workflow_instance_id:
            logger.error("No workflow instance ID found in context for timer handler")
            return {"_timer_error": "No workflow instance ID"}

        timer_info = context.get("_timer_info", {})
        delay_seconds = timer_info.get("delay_seconds")
        next_state = timer_info.get("next_state")

        if not delay_seconds or not next_state:
            logger.error("Invalid timer information")
            return {"_timer_error": "Invalid timer configuration"}

        # Mark for deferred timer scheduling
        return {
            "_timer_pending": True,
            "_timer_instance_id": workflow_instance_id,
            "_timer_delay_seconds": delay_seconds,
            "_timer_next_state": next_state,
        }

# --------------------------------------------------------------
