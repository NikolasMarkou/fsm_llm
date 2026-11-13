"""
FSM Handler System: A Comprehensive Framework for Self-Determining Function Handlers in FSM-LLM.

This module provides a sophisticated and flexible architecture for executing custom functions
during Finite State Machine (FSM) execution. The key innovation is that each handler contains
its own logic for determining when it should run, making the system highly extensible and
maintainable.

Overview
--------
The handler system operates on the principle of self-determination: rather than having a
central dispatcher decide which handlers to run, each handler implements its own
``should_execute()`` method that evaluates the current FSM state and context to determine
if it should be activated.

Core Components
---------------
1. **HandlerTiming**: Enum defining execution hook points throughout the FSM lifecycle
2. **FSMHandler**: Protocol defining the interface for all handlers
3. **HandlerSystem**: Central orchestrator that manages and executes handlers
4. **BaseHandler**: Base implementation class for creating custom handlers
5. **HandlerBuilder**: Fluent API for creating handlers using lambda functions
6. **LambdaHandler**: Internal implementation for lambda-based handlers

Architecture
------------
The system follows these key architectural principles:

- **Self-Determination**: Each handler decides when it should execute
- **Priority-Based Execution**: Handlers execute in priority order (lower numbers first)
- **Error Isolation**: Handler failures don't break the entire system
- **Context Awareness**: Handlers have full access to FSM state and context
- **Flexible Conditions**: Support for complex execution conditions via lambdas

Handler Execution Flow
----------------------
1. **Registration**: Handlers are registered with the HandlerSystem
2. **Filtering**: At each timing point, potentially applicable handlers are filtered
3. **Condition Evaluation**: Each handler's ``should_execute()`` method is called
4. **Execution**: Qualifying handlers execute in priority order
5. **Context Update**: Handler results are merged into the FSM context
6. **Error Handling**: Failures are handled according to the configured error mode

Usage Patterns
--------------

Basic Handler Creation::

    class MyHandler(BaseHandler):
        def should_execute(self, timing, current_state, target_state, context, updated_keys):
            return timing == HandlerTiming.PRE_PROCESSING and current_state == "collecting_info"

        def execute(self, context):
            return {"processed": True}

Lambda-Based Handler Creation::

    handler = (create_handler("my_handler")
               .at(HandlerTiming.POST_TRANSITION)
               .on_state("completed")
               .do(lambda ctx: {"completion_time": datetime.now().isoformat()}))

Advanced Conditional Logic::

    handler = (create_handler("conditional_handler")
               .when(lambda timing, state, target, ctx, keys:
                     ctx.get("user_score", 0) > 80 and "premium" in ctx.get("features", []))
               .do(lambda ctx: enable_premium_features(ctx)))

Package layout
--------------
``fsm_llm.handlers`` is a package (plan d454fd62, step 77: split of the old single module,
the dotted path is unchanged and every public name is re-exported here):

- ``protocol``: ``HandlerTiming``, ``HandlerErrorMode``, ``FSMHandler``, the lambda aliases
- ``errors``: ``HandlerSystemError``, ``HandlerExecutionError``
- ``base``: ``BaseHandler``
- ``system``: ``HandlerSystem``
- ``lambda_handler``: ``LambdaHandler``
- ``builder``: ``HandlerBuilder``, ``create_handler``
- ``helpers``: ``clear_keys_on_entry``

``clear_keys_delta`` is defined in ``fsm_llm.context.commit`` (step 74) and re-bound here.
"""

from fsm_llm.context.commit import clear_keys_delta

from .base import BaseHandler
from .builder import HandlerBuilder, create_handler
from .errors import HandlerExecutionError, HandlerSystemError
from .helpers import clear_keys_on_entry
from .lambda_handler import LambdaHandler
from .protocol import (
    ConditionLambda,
    ExecutionLambda,
    FSMHandler,
    HandlerErrorMode,
    HandlerTiming,
)
from .system import HandlerSystem

__all__ = [
    "BaseHandler",
    "ConditionLambda",
    "ExecutionLambda",
    "FSMHandler",
    "HandlerBuilder",
    "HandlerErrorMode",
    "HandlerExecutionError",
    "HandlerSystem",
    "HandlerSystemError",
    "HandlerTiming",
    "LambdaHandler",
    "clear_keys_delta",
    "clear_keys_on_entry",
    "create_handler",
]
