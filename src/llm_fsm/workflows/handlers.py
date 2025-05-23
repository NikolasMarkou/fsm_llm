"""
Handlers for integrating workflows with LLM-FSM.
"""

import asyncio
from typing import TYPE_CHECKING

from ..handler_system import BaseHandler, HandlerTiming
from ..logging import logger

if TYPE_CHECKING:
    from .engine import WorkflowEngine
    from ..fsm import FSMManager


class AutoTransitionHandler(BaseHandler):
    """Handler that detects and executes automatic transitions in LLM-FSM."""

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
        """Execute the handler logic."""
        conversation_id = context.get("_conversation_id")
        if not conversation_id:
            logger.error("No conversation ID found in context for auto transition")
            return {"_auto_transition_error": "No conversation ID"}

        workflow_info = context.get("_workflow_info", {})
        workflow_instance_id = workflow_info.get("instance_id")

        if not workflow_instance_id:
            logger.error("No workflow instance ID found in context for auto transition")
            return {"_auto_transition_error": "No workflow instance ID"}

        # Schedule automatic advancement
        try:
            # Create task to advance workflow
            task = asyncio.create_task(
                self.workflow_engine.advance_workflow(workflow_instance_id, "")
            )

            # Store task reference to prevent garbage collection
            context["_auto_transition_task"] = task

            return {"_auto_transition_scheduled": True}
        except Exception as e:
            logger.error(f"Error scheduling auto transition: {str(e)}")
            return {"_auto_transition_error": str(e)}


class EventHandler(BaseHandler):
    """Handler that processes external events in LLM-FSM."""

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
        """Execute the handler logic."""
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

        try:
            # Register the event listener
            asyncio.create_task(
                self.workflow_engine.register_event_listener(
                    workflow_instance_id,
                    event_type,
                    waiting_info.get("success_state"),
                    waiting_info.get("timeout_seconds"),
                    waiting_info.get("timeout_state"),
                    waiting_info.get("event_mapping", {})
                )
            )
            return {"_event_listener_registered": True}
        except Exception as e:
            logger.error(f"Error registering event listener: {str(e)}")
            return {"_event_listener_error": str(e)}


class TimerHandler(BaseHandler):
    """Handler that processes timers in LLM-FSM."""

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
        """Execute the handler logic."""
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

        try:
            # Schedule the timer
            asyncio.create_task(
                self.workflow_engine.schedule_timer(
                    workflow_instance_id,
                    delay_seconds,
                    next_state
                )
            )
            return {"_timer_scheduled": True}
        except Exception as e:
            logger.error(f"Error scheduling timer: {str(e)}")
            return {"_timer_error": str(e)}