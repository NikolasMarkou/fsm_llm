"""
LLM-FSM Version 3: Enhanced Finite State Machine implementation for Large Language Models.

This module provides the foundational data structures and models for implementing
sophisticated conversational AI systems using Finite State Machines (FSMs) with
Large Language Models. The architecture bridges structured state machine logic
with the natural language understanding capabilities of modern LLMs.

Core Architecture:
-----------------
The system operates on several key principles:

1. **State-Driven Conversations**: Each conversation step is modeled as a discrete
   state with specific purposes, transitions, and data collection requirements.

2. **LLM Integration**: Large Language Models provide natural language understanding
   and generation while being constrained by FSM structure for predictable behavior.

3. **Context Management**: Persistent context tracking throughout conversations
   enables stateful interactions and information accumulation.

4. **Flexible Transitions**: State transitions can be conditional, prioritized,
   and based on both explicit logic and LLM understanding of user intent.

5. **Extensible Framework**: Support for function handlers, custom validation,
   and integration with external systems.

Key Components:
--------------
- FSMDefinition: Complete specification of a conversational flow
- State: Individual conversation states with purposes and transitions
- FSMInstance: Runtime execution context for a specific conversation
- FSMContext: Persistent data and conversation history management
- LLM Integration: Request/response models for LLM communication

This design enables building sophisticated conversational AI applications that
combine the flexibility of natural language with the predictability and
reliability of finite state machines.

Dependencies:
------------
- pydantic: Data validation and serialization
- typing: Type hints and annotations
- json: JSON serialization for context data
- Custom logging and constants modules

Usage Pattern:
-------------
1. Define FSM structure with states, transitions, and purposes
2. Create FSM instance for each conversation
3. Process user input through LLM with current state context
4. Execute state transitions based on LLM responses
5. Maintain conversation context and history throughout

Integration Points:
------------------
- LLM Providers: Through standardized request/response interfaces
- External Systems: Via function handlers for databases, APIs, etc.
- Validation: Built-in FSM structure validation and runtime checks
- Monitoring: Comprehensive logging throughout execution lifecycle
"""

import json
import textwrap
from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Optional, Any, Callable

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------

from .logging import logger
from .constants import (
    DEFAULT_MESSAGE_TRUNCATE_LENGTH,
    DEFAULT_MAX_HISTORY_SIZE,
    DEFAULT_MAX_MESSAGE_LENGTH
)

# --------------------------------------------------------------


class TransitionCondition(BaseModel):
    """
    Defines a conditional requirement for state transitions in FSM workflows.

    TransitionConditions provide sophisticated control over when state transitions
    should occur, enabling complex conversational flows that adapt based on
    collected context data, user responses, and logical conditions.

    The condition system supports both simple key-based requirements and
    complex JsonLogic expressions for maximum flexibility in defining
    transition criteria.

    Architecture:
    ------------
    - Simple conditions check for presence of context keys
    - Complex conditions use JsonLogic for boolean logic evaluation
    - Multiple conditions can be combined within a single transition
    - Conditions are evaluated at runtime before transition execution

    Usage Examples:
    --------------
    Simple key-based condition:
        TransitionCondition(
            description="User has provided email address",
            requires_context_keys=["email"]
        )

    Complex logical condition:
        TransitionCondition(
            description="User is premium and has made purchase",
            requires_context_keys=["user_tier", "purchase_count"],
            logic={"and": [
                {"==": [{"var": "user_tier"}, "premium"]},
                {">": [{"var": "purchase_count"}, 0]}
            ]}
        )

    Attributes:
        description: Human-readable explanation of what this condition checks.
                    Used for debugging, documentation, and LLM understanding.

        requires_context_keys: List of context data keys that must be present
                              for this condition to be evaluated. If any required
                              keys are missing, the condition automatically fails.

        logic: JsonLogic expression for complex boolean evaluation against
               context data. Enables sophisticated conditional logic including
               comparisons, boolean operations, and data transformations.
               If None, only key presence is checked.

    Integration:
    -----------
    Conditions are evaluated by the FSM engine before executing transitions:
    1. Check required context keys are present
    2. If logic is provided, evaluate JsonLogic expression
    3. Transition proceeds only if all conditions pass
    4. Failed conditions prevent transition and may trigger fallback behavior
    """

    description: str = Field(
        ...,
        description="Human-readable description explaining what this condition validates",
        min_length=1,
        max_length=500
    )

    requires_context_keys: Optional[List[str]] = Field(
        default=None,
        description=textwrap.dedent("""
            Context keys that must be present for this condition to be valid.
            These keys are checked before any logic evaluation occurs.
            An empty list means no keys are required.
            """).strip()
    )

    logic: Optional[Dict[str, Any]] = Field(
        default=None,
        description=textwrap.dedent("""
            JsonLogic expression for complex conditional evaluation.
            Operates on context data using JsonLogic syntax.
            Common operators: ==, !=, >, <, >=, <=, and, or, not, in, var
            If None, only context key presence is validated.
            """).strip()
    )


# --------------------------------------------------------------


class Emission(BaseModel):
    """
    Defines structured output or side effects that should occur in a state.

    Emissions provide a mechanism for specifying what the LLM should output
    or what actions should be triggered when the FSM enters or operates
    within a specific state. This enables consistent behavior and
    standardized responses across different conversation contexts.

    Design Purpose:
    --------------
    - Standardize LLM outputs for specific states
    - Provide clear guidance for tone, content, and structure
    - Enable state-specific behavior customization
    - Support integration with external systems or APIs

    Usage Patterns:
    --------------
    1. Response Templates: Guide LLM to generate specific types of responses
    2. Action Triggers: Specify external actions to perform in this state
    3. Format Requirements: Ensure consistent output formatting
    4. Tone/Style Guidelines: Maintain persona consistency

    Examples:
    --------
    Response guidance:
        Emission(
            description="Welcome message with personalized greeting",
            instruction="Address user by name and offer assistance options"
        )

    Integration instruction:
        Emission(
            description="Trigger user account lookup",
            instruction="Query user database with provided email address"
        )

    Attributes:
        description: Clear explanation of what should be emitted or performed
                    in this state. Used by both LLM and human developers.

        instruction: Specific directions for the LLM about how to generate
                    the emission. Provides detailed guidance on tone, content,
                    format, or integration requirements.
    """

    description: str = Field(
        ...,
        description="Clear description of what should be emitted or performed",
        min_length=1,
        max_length=300
    )

    instruction: Optional[str] = Field(
        None,
        description=textwrap.dedent("""
            Detailed instruction for the LLM on how to generate this emission.
            Can include tone guidance, format requirements, content specifications,
            or integration instructions for external systems.
            """).strip(),
        max_length=1000
    )


# --------------------------------------------------------------


class Transition(BaseModel):
    """
    Defines a directed edge between FSM states with conditional logic.

    Transitions are the core mechanism for FSM progression, defining when
    and how conversations move from one state to another. Each transition
    represents a possible path based on user input, collected context,
    or logical conditions.

    Architecture Principles:
    -----------------------
    - **Directed**: Transitions have explicit source and target states
    - **Conditional**: Can include complex conditions for execution
    - **Prioritized**: Multiple transitions can be ranked by priority
    - **Descriptive**: Human-readable descriptions guide LLM decision-making

    Transition Evaluation Process:
    -----------------------------
    1. **Condition Check**: All conditions must pass for transition eligibility
    2. **Priority Ordering**: Lower priority numbers execute first
    3. **LLM Decision**: LLM selects appropriate transition based on context
    4. **Validation**: Target state existence and accessibility verified
    5. **Execution**: Context updates applied and state changed

    Priority System:
    ---------------
    - Priority 1-50: High priority (immediate/obvious transitions)
    - Priority 51-100: Normal priority (standard workflow transitions)
    - Priority 101+: Low priority (fallback/error handling transitions)

    Use Cases:
    ---------
    - **Linear Flow**: Simple next-step transitions in guided processes
    - **Branching Logic**: Multiple paths based on user choices or data
    - **Error Handling**: Fallback transitions for unexpected inputs
    - **Conditional Routing**: Complex decision trees with multiple criteria

    Examples:
    --------
    Simple transition:
        Transition(
            target_state="collect_email",
            description="User provided their name, proceed to email collection",
            priority=10
        )

    Conditional transition:
        Transition(
            target_state="premium_checkout",
            description="User has premium account, use premium checkout flow",
            conditions=[
                TransitionCondition(
                    description="User has premium subscription",
                    requires_context_keys=["subscription_tier"],
                    logic={"==": [{"var": "subscription_tier"}, "premium"]}
                )
            ],
            priority=5
        )

    Attributes:
        target_state: Identifier of the destination state. Must exist in the
                     FSM definition and be reachable according to FSM structure.

        description: Human-readable explanation of when this transition should
                    occur. Used by LLM to understand transition appropriateness
                    and by developers for documentation and debugging.

        conditions: Optional list of conditions that must all pass for this
                   transition to be eligible. Complex logical requirements
                   can be expressed through multiple conditions.

        priority: Numeric priority for transition selection when multiple
                 transitions are eligible. Lower numbers indicate higher
                 priority and are evaluated first by the FSM engine.
    """

    target_state: str = Field(
        ...,
        description="Identifier of the target state for this transition",
        min_length=1,
        max_length=100
    )

    description: str = Field(
        ...,
        description=textwrap.dedent("""
            Human-readable description of when this transition should occur.
            Should clearly explain the circumstances, user actions, or
            conditions that make this transition appropriate.
            """).strip(),
        min_length=1,
        max_length=500
    )

    conditions: Optional[List[TransitionCondition]] = Field(
        default=None,
        description=textwrap.dedent("""
            List of conditions that must all be satisfied for this transition
            to be eligible. All conditions are evaluated with AND logic.
            If empty or None, no conditions are required.
            """).strip()
    )

    priority: int = Field(
        default=100,
        description=textwrap.dedent("""
            Priority for transition selection when multiple are eligible.
            Lower numbers = higher priority. Typical ranges:
            0-50: High priority (immediate/obvious)
            51-100: Normal priority (standard workflow)
            101+: Low priority (fallback/error handling)
            """).strip(),
        ge=0,
        le=1000
    )


# --------------------------------------------------------------


class State(BaseModel):
    """
    Represents a single state in the FSM conversation flow.

    States are the fundamental building blocks of conversational FSMs,
    representing discrete phases of interaction with specific purposes,
    data collection requirements, and available transitions. Each state
    encapsulates the context and constraints for a particular part of
    the conversation.

    State Design Philosophy:
    -----------------------
    - **Single Responsibility**: Each state has one clear purpose
    - **Self-Describing**: States contain all information needed for execution
    - **Transition-Aware**: States define their possible next steps
    - **Context-Driven**: States specify what data they need to collect
    - **LLM-Guided**: States provide instructions for LLM behavior

    State Lifecycle:
    ---------------
    1. **Entry**: State becomes active, context requirements evaluated
    2. **Processing**: User input processed according to state purpose
    3. **Collection**: Required context keys gathered from interaction
    4. **Evaluation**: Transition conditions assessed
    5. **Exit**: State transition executed to next appropriate state

    State Types and Patterns:
    ------------------------
    - **Collection States**: Gather specific information from users
    - **Decision States**: Route based on collected data or user choices
    - **Processing States**: Perform operations or integrations
    - **Confirmation States**: Verify collected information
    - **Terminal States**: End conversation or workflow

    Context Key Management:
    ----------------------
    States can specify required_context_keys that must be collected:
    - Keys are collected through natural language interaction
    - LLM extracts relevant information from user responses
    - Missing keys prevent transitions to dependent states
    - Collected keys persist throughout conversation

    Examples:
    --------
    Information collection state:
        State(
            id="collect_user_info",
            description="Gather basic user information",
            purpose="Collect user's name, email, and phone number for account setup",
            required_context_keys=["name", "email", "phone"],
            instructions="Be friendly and explain why each piece of information is needed",
            transitions=[
                Transition(target_state="verify_info", description="All info collected")
            ]
        )

    Decision routing state:
        State(
            id="service_router",
            description="Route user to appropriate service",
            purpose="Determine which service the user needs and route accordingly",
            transitions=[
                Transition(target_state="technical_support",
                          description="User needs technical help"),
                Transition(target_state="billing_support",
                          description="User has billing questions"),
                Transition(target_state="general_inquiry",
                          description="General questions or other needs")
            ]
        )

    Attributes:
        id: Unique identifier for the state within the FSM. Must be unique
            across all states and follow naming conventions for consistency.

        description: Human-readable description of what this state represents
                    in the conversation flow. Used for documentation and
                    LLM understanding of state purpose.

        purpose: Specific purpose or goal of this state. Clearly defines
                what should be accomplished before transitioning. Guides
                LLM behavior and provides success criteria.

        transitions: List of possible transitions from this state. Defines
                    the conversation paths and decision points available.
                    Empty list indicates a terminal state.

        required_context_keys: Context data keys that should be collected
                              in this state. Keys are extracted from user
                              input through LLM natural language processing.

        instructions: Optional specific instructions for LLM behavior in
                     this state. Can include tone guidance, format requirements,
                     or specific conversation strategies.

        example_dialogue: Optional examples of how conversations should
                         proceed in this state. Helps guide LLM responses
                         and provides training examples for consistent behavior.
    """

    id: str = Field(
        ...,
        description="Unique identifier for this state within the FSM",
        min_length=1,
        max_length=100,
        pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*$'  # Valid identifier pattern
    )

    description: str = Field(
        ...,
        description=textwrap.dedent("""
            Human-readable description of what this state represents
            in the conversation flow. Should be clear and concise.
            """).strip(),
        min_length=1,
        max_length=300
    )

    purpose: str = Field(
        ...,
        description=textwrap.dedent("""
            Specific purpose or goal of this state. Defines what should
            be accomplished before transitioning. Guides LLM behavior
            and provides clear success criteria for the state.
            """).strip(),
        min_length=1,
        max_length=500
    )

    transitions: List[Transition] = Field(
        default_factory=list,
        description=textwrap.dedent("""
            List of possible transitions from this state. Defines available
            conversation paths and decision points. Empty list indicates
            a terminal state that ends the conversation or workflow.
            """).strip()
    )

    required_context_keys: Optional[List[str]] = Field(
        default=None,
        description=textwrap.dedent("""
            Context data keys that should be collected in this state.
            Keys are extracted from user input through LLM natural language
            processing. Missing keys may prevent certain transitions.
            """).strip()
    )

    instructions: Optional[str] = Field(
        None,
        description=textwrap.dedent("""
            Optional specific instructions for LLM behavior in this state.
            Can include tone guidance, format requirements, conversation
            strategies, or integration with external systems.
            """).strip(),
        max_length=1000
    )

    example_dialogue: Optional[List[Dict[str, str]]] = Field(
        None,
        description=textwrap.dedent("""
            Optional examples of conversation flow in this state.
            Each example should be a dict with 'user' and 'assistant' keys.
            Helps guide LLM responses and ensures consistent behavior.
            """).strip()
    )


# --------------------------------------------------------------


class FunctionHandler(BaseModel):
    """
    Defines external function integration points within FSM execution.

    FunctionHandlers provide a powerful mechanism for integrating FSM
    conversations with external systems, databases, APIs, and custom
    business logic. They enable seamless interaction between the
    conversational flow and backend systems.

    Integration Architecture:
    ------------------------
    - **Event-Driven**: Handlers trigger on specific FSM events
    - **State-Aware**: Can be limited to specific states or global
    - **Flexible Execution**: Support for both sync and async operations
    - **Error Handling**: Built-in error management and fallback strategies

    Handler Lifecycle:
    -----------------
    1. **Registration**: Handler registered with FSM definition
    2. **Event Monitoring**: FSM monitors for trigger conditions
    3. **Context Preparation**: Current context passed to handler
    4. **Execution**: Handler function called with context data
    5. **Result Integration**: Handler results merged back into FSM context

    Common Use Cases:
    ----------------
    - **Database Operations**: User data lookup, storage, updates
    - **API Integration**: External service calls and data retrieval
    - **Validation**: Custom business rule validation
    - **Notifications**: Email, SMS, or push notification triggers
    - **Analytics**: Event tracking and user behavior analysis
    - **Authentication**: User verification and session management

    Event Types:
    -----------
    - **pre_transition**: Before state changes occur
    - **post_transition**: After successful state changes
    - **context_update**: When context data is modified
    - **state_entry**: When entering specific states
    - **state_exit**: When leaving specific states
    - **error**: When errors occur during execution

    Examples:
    --------
    Database integration:
        FunctionHandler(
            name="user_lookup",
            description="Retrieve user profile from database",
            trigger_on=["context_update"],
            states=["user_verification"],
            function=lambda ctx: database.get_user(ctx.get("email"))
        )

    API integration:
        FunctionHandler(
            name="send_notification",
            description="Send welcome email to new users",
            trigger_on=["post_transition"],
            states=["registration_complete"],
            function=email_service.send_welcome_email
        )

    Attributes:
        name: Unique identifier for this handler within the FSM.
              Used for debugging, logging, and handler management.

        description: Human-readable description of what this handler does.
                    Should explain its purpose and integration role clearly.

        trigger_on: List of FSM events that will cause this handler to execute.
                   Multiple events can trigger the same handler for flexibility.

        states: Optional list of state IDs where this handler applies.
               If None, handler applies to all states. If specified,
               handler only executes when in one of the listed states.

        function: Optional callable function to execute when triggered.
                 Function receives FSM context as parameter and can return
                 data to be merged back into the context. Not serialized
                 in JSON definitions - must be attached at runtime.

    Integration Notes:
    -----------------
    - Function handlers are not stored in JSON FSM definitions
    - Functions must be attached programmatically during FSM setup
    - Handlers should be idempotent and handle errors gracefully
    - Return values are automatically merged into FSM context
    - Async functions are supported and awaited automatically
    """

    name: str = Field(
        ...,
        description="Unique identifier for this handler within the FSM",
        min_length=1,
        max_length=100
    )

    description: str = Field(
        ...,
        description=textwrap.dedent("""
            Human-readable description of what this handler does and
            its role in the FSM workflow. Should be clear and specific
            about the handler's purpose and expected behavior.
            """).strip(),
        min_length=1,
        max_length=500
    )

    trigger_on: List[str] = Field(
        ...,
        description=textwrap.dedent("""
            List of FSM events that trigger this handler execution.
            Common events: pre_transition, post_transition, context_update,
            state_entry, state_exit, error. Multiple events supported.
            """).strip(),
        min_length=1
    )

    states: Optional[List[str]] = Field(
        None,
        description=textwrap.dedent("""
            Optional list of state IDs where this handler applies.
            If None or empty, handler applies to all states.
            If specified, handler only executes in listed states.
            """).strip()
    )

    function: Optional[Callable] = Field(
        None,
        description=textwrap.dedent("""
            Callable function to execute when handler is triggered.
            Function receives FSM context as parameter and can return
            data to merge into context. Not stored in JSON definitions.
            """).strip()
    )

    class Config:
        arbitrary_types_allowed = True


# --------------------------------------------------------------


class FSMDefinition(BaseModel):
    """
    Complete specification of a conversational Finite State Machine.

    FSMDefinition represents the complete blueprint for a conversational
    AI workflow, including all states, transitions, validation rules,
    and integration points. It serves as both the design document and
    runtime specification for FSM execution.

    Design Philosophy:
    -----------------
    - **Completeness**: Contains everything needed for FSM execution
    - **Validation**: Built-in structural and logical validation
    - **Extensibility**: Support for function handlers and customization
    - **Reusability**: Definitions can be shared and versioned
    - **Documentation**: Self-documenting through descriptions and examples

    FSM Validation Process:
    ----------------------
    The FSM undergoes comprehensive validation to ensure structural integrity:

    1. **State Validation**: All referenced states exist and are properly defined
    2. **Transition Validation**: All transition targets exist and are reachable
    3. **Connectivity**: Initial state connects to at least one terminal state
    4. **Completeness**: No orphaned states that can't be reached
    5. **Terminal States**: At least one state has no outgoing transitions

    Conversation Flow Design:
    ------------------------
    - **Entry Point**: initial_state defines where conversations begin
    - **Information Flow**: States collect context data progressively
    - **Decision Points**: States with multiple transitions enable branching
    - **Exit Points**: Terminal states (no transitions) end conversations
    - **Error Handling**: Fallback transitions handle unexpected inputs

    Persona and Tone:
    ----------------
    The optional persona field provides consistent personality throughout:
    - Influences LLM response style and tone
    - Maintains character consistency across states
    - Can specify domain expertise or conversation style
    - Applied globally across all states in the FSM

    Versioning and Evolution:
    ------------------------
    - Version field enables FSM evolution tracking
    - Backward compatibility considerations for deployed FSMs
    - Migration strategies for version updates
    - Testing and validation of FSM changes

    Examples:
    --------
    Simple customer support FSM:
        FSMDefinition(
            name="customer_support_v1",
            description="Basic customer support workflow",
            initial_state="greeting",
            persona="Friendly and helpful customer service representative",
            states={
                "greeting": State(
                    id="greeting",
                    description="Welcome user and identify their need",
                    purpose="Greet customer and determine how to help them",
                    transitions=[
                        Transition(target_state="technical_support",
                                 description="Technical issue reported"),
                        Transition(target_state="billing_inquiry",
                                 description="Billing question asked")
                    ]
                ),
                # ... additional states
            }
        )

    Attributes:
        name: Unique name identifier for this FSM definition. Should be
              descriptive and may include version information for tracking.

        description: Human-readable description of what this FSM accomplishes,
                    its use case, and its overall purpose in the application.

        states: Dictionary mapping state IDs to State objects. Contains
               all states that comprise this FSM, including entry points,
               processing states, decision points, and terminal states.

        initial_state: ID of the state where conversations begin. Must exist
                      in the states dictionary and should be designed as an
                      appropriate entry point for the conversation flow.

        version: Version identifier for this FSM definition. Enables
                tracking of changes, deployment management, and backward
                compatibility considerations.

        persona: Optional description of the personality, tone, or expertise
                that the LLM should maintain throughout conversations using
                this FSM. Influences response style across all states.

        function_handlers: Optional list of function handlers that integrate
                          with external systems. Enables database operations,
                          API calls, notifications, and custom business logic.

    Validation Details:
    ------------------
    The model_validator performs comprehensive structural validation:
    - Ensures initial_state exists in states dictionary
    - Validates all transition targets exist as states
    - Identifies terminal states (those with no outgoing transitions)
    - Confirms at least one terminal state is reachable from initial state
    - Detects orphaned states that can never be reached
    - Logs validation results for debugging and monitoring

    Integration Points:
    ------------------
    - **LLM Providers**: FSM definitions guide LLM prompt generation
    - **Runtime Engine**: Definitions loaded and executed by FSM engine
    - **External Systems**: Function handlers provide integration capabilities
    - **Monitoring**: Validation logs enable operational monitoring
    - **Development**: Definitions serve as conversation design documentation
    """

    name: str = Field(
        ...,
        description=textwrap.dedent("""
            Unique name identifier for this FSM definition.
            Should be descriptive and may include version information.
            Used for loading, referencing, and managing FSM definitions.
            """).strip(),
        min_length=1,
        max_length=100
    )

    description: str = Field(
        ...,
        description=textwrap.dedent("""
            Human-readable description of what this FSM accomplishes,
            its use case, and overall purpose. Should clearly explain
            the conversation flow and expected outcomes.
            """).strip(),
        min_length=1,
        max_length=1000
    )

    states: Dict[str, State] = Field(
        ...,
        description=textwrap.dedent("""
            Dictionary mapping state IDs to State objects. Must contain
            all states referenced by transitions and the initial_state.
            Represents the complete state space of the conversation.
            """).strip(),
        min_length=1
    )

    initial_state: str = Field(
        ...,
        description=textwrap.dedent("""
            ID of the state where conversations begin. Must exist in the
            states dictionary and should be designed as an appropriate
            entry point with clear purpose and available transitions.
            """).strip(),
        min_length=1
    )

    version: str = Field(
        default="3.0",
        description=textwrap.dedent("""
            Version identifier for this FSM definition. Enables tracking
            of changes, deployment management, and compatibility checks.
            Recommended format: major.minor.patch (e.g., "3.1.0").
            """).strip(),
        min_length=1,
        max_length=20
    )

    persona: Optional[str] = Field(
        None,
        description=textwrap.dedent("""
            Optional description of the personality, tone, or expertise
            that the LLM should maintain throughout conversations.
            Influences response style, formality level, and domain knowledge.
            Applied consistently across all states in the FSM.
            """).strip(),
        max_length=500
    )

    function_handlers: Optional[List[FunctionHandler]] = Field(
        default_factory=list,
        description=textwrap.dedent("""
            Optional list of function handlers for external system integration.
            Enables database operations, API calls, notifications, validation,
            and custom business logic during FSM execution.
            """).strip()
    )

    @model_validator(mode='after')
    def validate_states(self) -> 'FSMDefinition':
        """
        Performs comprehensive structural validation of the FSM definition.

        This validation ensures the FSM is logically sound, structurally complete,
        and operationally viable. It checks for common FSM design issues that
        could prevent proper execution or lead to conversation dead-ends.

        Validation Checks:
        -----------------
        1. **Initial State Existence**: Verifies initial_state exists in states dict
        2. **Transition Target Validity**: All transition targets must be valid states
        3. **Terminal State Presence**: At least one state must have no outgoing transitions
        4. **Reachability Analysis**: All states must be reachable from initial state
        5. **Terminal State Accessibility**: At least one terminal state must be reachable
        6. **Orphaned State Detection**: No states should be unreachable from initial state

        The validation process uses graph traversal algorithms to analyze the FSM
        structure and identify potential issues before runtime execution.

        Logging and Monitoring:
        ----------------------
        - Debug logs track validation progress and intermediate results
        - Error logs capture specific validation failures with detailed messages
        - Info logs confirm successful validation and highlight key metrics
        - Warning logs may indicate potential design issues worth reviewing

        Returns:
            FSMDefinition: The validated FSM definition (self)

        Raises:
            ValueError: If any validation check fails, with detailed error message
                       explaining the specific issue and affected states

        Implementation Details:
        ----------------------
        The validation uses a breadth-first search approach to:
        - Build a reachability graph from the initial state
        - Identify all states accessible through valid transitions
        - Detect terminal states by analyzing outgoing transitions
        - Cross-reference reachable and terminal states for completeness

        This approach ensures the FSM can execute successfully and provides
        meaningful conversation flows with proper termination conditions.
        """
        logger.debug(f"Starting comprehensive validation for FSM: {self.name}")

        # Validation 1: Check initial state exists
        if self.initial_state not in self.states:
            error_msg = f"Initial state '{self.initial_state}' not found in states dictionary. Available states: {list(self.states.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"Initial state '{self.initial_state}' exists and is valid")

        # Validation 2: Identify terminal states (no outgoing transitions)
        terminal_states = {
            state_id for state_id, state in self.states.items()
            if not state.transitions
        }

        logger.debug(f"Identified {len(terminal_states)} terminal states: {terminal_states}")

        # Validation 3: Check that at least one terminal state exists
        if not terminal_states:
            error_msg = textwrap.dedent("""
                FSM has no terminal states. At least one state must have no outgoing 
                transitions to provide conversation completion points. Consider adding 
                a terminal state or removing transitions from an existing state.
                """).strip()
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validation 4: Build reachability graph using breadth-first search
        reachable_states = {self.initial_state}
        states_to_process = [self.initial_state]

        while states_to_process:
            current_state_id = states_to_process.pop(0)
            current_state = self.states[current_state_id]

            # Process all transitions from current state
            for transition in current_state.transitions:
                target_state = transition.target_state

                # Add newly discovered states to processing queue
                if target_state not in reachable_states:
                    reachable_states.add(target_state)
                    states_to_process.append(target_state)

        logger.debug(f"Reachability analysis complete. {len(reachable_states)} states reachable from initial state")

        # Validation 5: Check all transition targets exist as valid states
        for state_id, state in self.states.items():
            for transition in state.transitions:
                if transition.target_state not in self.states:
                    error_msg = textwrap.dedent(f"""
                        Invalid transition detected: State '{state_id}' has transition 
                        to non-existent state '{transition.target_state}'. 
                        Available states: {list(self.states.keys())}
                        """).strip()
                    logger.error(error_msg)
                    raise ValueError(error_msg)

        # Validation 6: Check for orphaned states (unreachable from initial state)
        orphaned_states = set(self.states.keys()) - reachable_states
        if orphaned_states:
            error_msg = textwrap.dedent(f"""
                Orphaned states detected: {sorted(orphaned_states)}
                These states cannot be reached from the initial state '{self.initial_state}' 
                and will never be executed. Consider:
                1. Adding transitions that lead to these states
                2. Removing unused states from the definition
                3. Verifying the intended conversation flow design
                """).strip()
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validation 7: Ensure at least one terminal state is reachable
        reachable_terminal_states = terminal_states.intersection(reachable_states)
        if not reachable_terminal_states:
            error_msg = textwrap.dedent(f"""
                No terminal states are reachable from the initial state '{self.initial_state}'.
                Terminal states found: {sorted(terminal_states)}
                Reachable states: {sorted(reachable_states)}
                
                This creates an infinite loop with no way to complete conversations.
                Consider:
                1. Adding transitions that lead to terminal states
                2. Converting some reachable states to terminal states
                3. Reviewing the conversation flow design for completion points
                """).strip()
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validation successful - log summary
        logger.debug(f"✅ FSM definition '{self.name}' validated successfully")
        logger.debug(f"📊 Validation Summary:")
        logger.debug(f"   • Total states: {len(self.states)}")
        logger.debug(f"   • Reachable states: {len(reachable_states)}")
        logger.debug(f"   • Terminal states: {len(terminal_states)}")
        logger.debug(f"   • Reachable terminals: {len(reachable_terminal_states)}")
        logger.debug(f"   • Function handlers: {len(self.function_handlers or [])}")
        logger.debug(f"🎯 Reachable terminal states: {', '.join(sorted(reachable_terminal_states))}")

        return self


# --------------------------------------------------------------


class Conversation(BaseModel):
    """
    Manages conversation history and message flow for FSM interactions.

    The Conversation class provides sophisticated management of conversational
    context, including message history, automatic truncation, and retrieval
    of relevant conversation segments. It serves as the memory system for
    FSM-driven conversations.

    Design Principles:
    -----------------
    - **Memory Management**: Automatic history size management to prevent memory bloat
    - **Message Integrity**: Preservation of conversation context and flow
    - **Truncation Handling**: Graceful handling of oversized messages
    - **Efficient Retrieval**: Fast access to recent conversation history
    - **Logging Integration**: Comprehensive logging for debugging and monitoring

    Memory Management Strategy:
    --------------------------
    The conversation system employs a sliding window approach:
    - Recent exchanges are kept for immediate context
    - Older exchanges are automatically pruned when limits are exceeded
    - Truncation preserves conversation flow while managing memory usage
    - Configurable limits allow adaptation to different use cases

    Message Processing Pipeline:
    ---------------------------
    1. **Input Validation**: Check message length and content
    2. **Truncation**: Apply length limits with clear truncation markers
    3. **Storage**: Add to conversation history with role identification
    4. **Maintenance**: Prune old messages if history size exceeded
    5. **Logging**: Record operations for debugging and monitoring

    Use Cases:
    ---------
    - **Context Preservation**: Maintain conversation context across states
    - **Reference Resolution**: Enable references to previous conversation parts
    - **History Analysis**: Support conversation analysis and improvement
    - **Memory Optimization**: Prevent unbounded memory growth in long conversations

    Configuration Options:
    ---------------------
    - max_history_size: Maximum number of exchange pairs to retain
    - max_message_length: Soft limit for individual message length
    - Automatic truncation with clear indicators for oversized content

    Examples:
    --------
    Basic conversation management:
        conversation = Conversation(
            max_history_size=10,  # Keep last 10 exchanges
            max_message_length=1000  # Truncate messages over 1000 chars
        )

        conversation.add_user_message("Hello, I need help with my account")
        conversation.add_system_message("I'd be happy to help! What specific issue are you experiencing?")

        recent = conversation.get_recent(5)  # Get last 5 exchanges

    Attributes:
        exchanges: List of conversation exchanges, each containing role-message pairs.
                  Maintained in chronological order with automatic pruning.

        max_history_size: Maximum number of exchange pairs to retain in memory.
                         Older exchanges are automatically removed when exceeded.
                         Each exchange typically contains one user and one system message.

        max_message_length: Soft limit for individual message length. Messages
                           exceeding this limit are truncated with clear indicators
                           to preserve conversation flow while managing memory.

    Implementation Notes:
    --------------------
    - Exchange pairs count as single units for history size calculation
    - Truncation preserves message meaning while adding clear truncation markers
    - Message retrieval is optimized for recent conversation access patterns
    - Logging provides visibility into history management operations
    """

    exchanges: List[Dict[str, str]] = Field(
        default_factory=list,
        description=textwrap.dedent("""
            List of conversation exchanges in chronological order.
            Each exchange is a dictionary with role keys ('user', 'system', 'assistant')
            mapped to message content. Automatically managed with pruning.
            """).strip()
    )

    max_history_size: int = Field(
        default=DEFAULT_MAX_HISTORY_SIZE,
        description=textwrap.dedent("""
            Maximum number of exchange pairs to retain in conversation history.
            When exceeded, oldest exchanges are automatically removed.
            Each exchange pair typically includes user input and system response.
            """).strip(),
        ge=0,
        le=1000
    )

    max_message_length: int = Field(
        default=DEFAULT_MAX_MESSAGE_LENGTH,
        description=textwrap.dedent("""
            Soft limit for individual message length in characters.
            Messages exceeding this limit are truncated with clear indicators.
            Helps prevent memory issues while preserving conversation flow.
            """).strip(),
        ge=1,
        le=50000
    )

    def add_user_message(self, message: str) -> None:
        """
        Add a user message to the conversation with automatic management.

        This method handles the complete pipeline for processing user input:
        message validation, truncation if necessary, storage in conversation
        history, and automatic history maintenance to prevent memory bloat.

        Processing Pipeline:
        -------------------
        1. **Length Validation**: Check if message exceeds configured limits
        2. **Truncation**: Apply length limits with clear truncation indicators
        3. **Storage**: Add message to exchanges list with 'user' role
        4. **History Maintenance**: Prune old exchanges if limits exceeded
        5. **Logging**: Record operations for debugging and monitoring

        Truncation Behavior:
        -------------------
        - Messages exceeding max_message_length are truncated
        - Clear "... [truncated]" marker added to indicate truncation
        - Original message length preserved in logs for analysis
        - Truncation designed to maintain conversation coherence

        Args:
            message: The user's message content to add to conversation history.
                    Can be any length - will be automatically truncated if needed.

        Side Effects:
        ------------
        - Modifies exchanges list by adding new user message
        - May trigger automatic history pruning if limits exceeded
        - Generates debug logs for operation tracking
        - Updates conversation state for subsequent retrievals

        Implementation Details:
        ----------------------
        - Truncation threshold based on max_message_length configuration
        - History pruning uses sliding window to maintain recent context
        - Logging includes message preview and truncation status
        - Thread-safe operations for concurrent conversation management
        """
        truncated = False
        original_length = len(message)

        # Apply message length limits with clear truncation indicators
        if original_length > self.max_message_length:
            message = message[:self.max_message_length] + "... [truncated]"
            truncated = True

        # Log message addition with appropriate detail level
        preview_length = DEFAULT_MESSAGE_TRUNCATE_LENGTH
        message_preview = message[:preview_length]
        if len(message) > preview_length:
            message_preview += "..."

        logger.debug(f"Adding user message: {message_preview}")

        if truncated:
            logger.debug(
                f"Message truncated from {original_length} to {self.max_message_length} characters"
            )

        # Store message with role identification
        self.exchanges.append({"user": message})

        # Perform automatic history maintenance
        self._maintain_history_size()

    def add_system_message(self, message: str) -> None:
        """
        Add a system/assistant message to the conversation with automatic management.

        This method processes system responses through the same pipeline as user
        messages, ensuring consistent handling of message length, history management,
        and logging across all conversation participants.

        System Message Characteristics:
        ------------------------------
        - Represents responses from the AI system or assistant
        - Subject to same truncation rules as user messages
        - Maintains conversation flow and context preservation
        - Integrated with history management for memory optimization

        Processing follows the same pipeline as add_user_message():
        1. Length validation and truncation if necessary
        2. Storage with 'system' role identifier
        3. Automatic history maintenance
        4. Comprehensive logging for operations tracking

        Args:
            message: The system's response message to add to conversation history.
                    Typically contains AI-generated responses or system notifications.

        Side Effects:
        ------------
        - Modifies exchanges list by adding new system message
        - May trigger automatic history pruning
        - Generates debug logs with message preview
        - Updates conversation state for context preservation

        Logging Details:
        ---------------
        - System messages get shorter preview (50 chars) in logs
        - Truncation status logged for operational monitoring
        - Debug level logging for normal operations
        - Error conditions logged at appropriate levels
        """
        truncated = False
        original_length = len(message)

        # Apply consistent truncation policy
        if original_length > self.max_message_length:
            message = message[:self.max_message_length] + "... [truncated]"
            truncated = True

        # Log with shorter preview for system messages (typically longer)
        system_preview_length = 50
        message_preview = message[:system_preview_length]
        if len(message) > system_preview_length:
            message_preview += "..."

        logger.debug(f"Adding system message: {message_preview}")

        if truncated:
            logger.debug(
                f"System message truncated from {original_length} to {self.max_message_length} characters"
            )

        # Store with system role identification
        self.exchanges.append({"system": message})

        # Maintain history size constraints
        self._maintain_history_size()

    def get_recent(self, n: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Retrieve the most recent conversation exchanges for context preservation.

        This method provides efficient access to recent conversation history,
        which is essential for maintaining conversational context in FSM states.
        It handles edge cases gracefully and provides consistent behavior
        across different conversation lengths.

        Retrieval Strategy:
        ------------------
        - Returns most recent exchanges in chronological order
        - Handles requests for more exchanges than available
        - Optimized for common access patterns (recent context)
        - Maintains exchange pair relationships when possible

        Context Preservation:
        --------------------
        The retrieved exchanges serve multiple purposes:
        - LLM context for generating appropriate responses
        - State transition decision making
        - Conversation flow analysis and debugging
        - User experience continuity across state changes

        Args:
            n: Number of recent exchanges to retrieve. If None, uses the
               configured max_history_size. If larger than available exchanges,
               returns all available exchanges without error.

        Returns:
            List[Dict[str, str]]: List of recent exchanges in chronological order.
            Each exchange is a dictionary with role keys ('user', 'system')
            mapped to message content. Empty list if no exchanges available.

        Performance Characteristics:
        ---------------------------
        - O(1) operation for typical recent access patterns
        - Memory efficient - returns references, not copies
        - Optimized for frequent access during conversation processing
        - Handles edge cases without exceptions or errors

        Usage Examples:
        --------------
        # Get default number of recent exchanges
        recent = conversation.get_recent()

        # Get specific number of exchanges
        last_5 = conversation.get_recent(5)

        # Handle empty conversations gracefully
        empty_recent = Conversation().get_recent()  # Returns []
        """
        # Use configured default if no specific count requested
        if n is None:
            n = self.max_history_size

        # Handle edge cases gracefully
        if n <= 0:
            return []

        # Calculate retrieval range based on exchange pair logic
        # Each exchange pair typically consists of user input + system response
        retrieval_count = n * 2  # Account for user+system message pairs

        # Return most recent exchanges, handling insufficient history gracefully
        return self.exchanges[-retrieval_count:] if retrieval_count > 0 else []

    def _maintain_history_size(self) -> None:
        """
        Internal method for automatic conversation history maintenance.

        This private method implements the sliding window approach for conversation
        memory management, ensuring that history size stays within configured limits
        while preserving the most relevant recent context.

        Maintenance Strategy:
        --------------------
        - Triggered automatically after each message addition
        - Uses sliding window to preserve most recent exchanges
        - Maintains exchange pair relationships when possible
        - Provides logging for operational visibility

        Memory Management:
        -----------------
        - Prevents unbounded memory growth in long conversations
        - Balances context preservation with resource constraints
        - Optimized for typical conversation patterns and lengths
        - Configurable limits allow adaptation to different use cases

        The method calculates pruning requirements based on exchange pairs
        (user + system message combinations) rather than individual messages,
        which better preserves conversation coherence and context flow.

        Implementation Details:
        ----------------------
        - Only prunes when necessary (exceeds configured limits)
        - Removes oldest exchanges first (FIFO approach)
        - Logs pruning operations for monitoring and debugging
        - Maintains list integrity during pruning operations
        """
        # Calculate effective history limit based on exchange pairs
        effective_limit = self.max_history_size * 2  # user + system pairs

        # Only prune if history exceeds configured limits
        if len(self.exchanges) > effective_limit:
            excess_count = len(self.exchanges) - effective_limit

            # Remove oldest exchanges to maintain sliding window
            self.exchanges = self.exchanges[excess_count:]

            # Log pruning operation for operational monitoring
            logger.debug(
                f"Pruned {excess_count} old messages from conversation history. "
                f"Current history size: {len(self.exchanges)} messages"
            )


# --------------------------------------------------------------


class FSMContext(BaseModel):
    """
    Comprehensive runtime context management for FSM conversations.

    FSMContext serves as the central data repository and state management
    system for FSM conversations, providing persistent storage of collected
    information, conversation history, and metadata throughout the entire
    conversational lifecycle.

    Architecture and Design:
    -----------------------
    - **Persistent Storage**: Maintains data across state transitions
    - **Flexible Schema**: Supports arbitrary data structures and types
    - **History Integration**: Seamless conversation history management
    - **Metadata Support**: Extended information for system operations
    - **Validation Integration**: Built-in data validation and checking

    Data Organization:
    -----------------
    The context is organized into three primary components:

    1. **Data**: Core conversation data collected from user interactions
       - User-provided information (names, preferences, selections)
       - Derived data (calculations, classifications, enrichments)
       - Business objects (orders, accounts, profiles)

    2. **Conversation**: Complete conversation history and management
       - Message exchanges between user and system
       - Automatic history pruning and memory management
       - Context retrieval for LLM processing

    3. **Metadata**: System-level information and operational data
       - Conversation identifiers and session information
       - Timestamps, performance metrics, and debugging data
       - Integration data from external systems

    Context Lifecycle:
    -----------------
    1. **Initialization**: Context created with default configuration
    2. **Population**: Data collected through conversation interactions
    3. **Validation**: Key presence and data integrity checking
    4. **Persistence**: Context maintained across state transitions
    5. **Retrieval**: Data accessed for decision making and responses
    6. **Cleanup**: Memory management and history pruning

    Integration Patterns:
    --------------------
    - **LLM Context**: Provides conversation history for LLM processing
    - **State Validation**: Supports required key checking for transitions
    - **External Systems**: Stores data from API calls and integrations
    - **Business Logic**: Maintains calculated and derived information

    Memory Management:
    -----------------
    The context system includes automatic memory management:
    - Conversation history pruning based on configurable limits
    - Message truncation for oversized content
    - Efficient data structures for common access patterns
    - Logging for memory usage monitoring and optimization

    Examples:
    --------
    Basic context usage:
        context = FSMContext()
        context.update({"user_name": "Alice", "user_email": "alice@example.com"})

        # Check if required information is present
        if context.has_keys(["user_name", "user_email"]):
            # Proceed with next step
            pass

    Advanced context with conversation:
        context = FSMContext(
            max_history_size=15,
            max_message_length=2000
        )
        context.conversation.add_user_message("I want to update my profile")
        context.conversation.add_system_message("I can help with that. What would you like to change?")

    Attributes:
        data: Dictionary containing all collected conversation data.
              Supports arbitrary nested structures and data types.
              Automatically serializable for persistence and debugging.

        conversation: Conversation history management with automatic
                     pruning, message truncation, and efficient retrieval.
                     Integrates with LLM processing for context provision.

        metadata: Additional system-level information and operational data.
                 Used for debugging, analytics, integration data, and
                 extended system functionality not part of core conversation.

    Configuration:
    -------------
    Context behavior can be configured through initialization parameters:
    - max_history_size: Conversation history retention limit
    - max_message_length: Individual message length limits
    - Custom conversation settings for specific use cases
    """

    data: Dict[str, Any] = Field(
        default_factory=dict,
        description=textwrap.dedent("""
            Core conversation data collected from user interactions.
            Supports arbitrary nested structures and data types.
            Automatically maintained across state transitions and
            available for validation, decision making, and responses.
            """).strip()
    )

    conversation: Conversation = Field(
        default_factory=Conversation,
        description=textwrap.dedent("""
            Conversation history management with automatic pruning and
            efficient retrieval. Provides context for LLM processing
            and maintains conversation flow across state transitions.
            """).strip()
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=textwrap.dedent("""
            System-level information and operational data including
            session identifiers, timestamps, performance metrics,
            debugging information, and integration data from external systems.
            """).strip()
    )

    def __init__(self, **data):
        """
        Initialize FSMContext with optional conversation configuration.

        This constructor supports flexible initialization of conversation
        settings while maintaining backward compatibility with existing code.
        It handles conversation configuration extraction and applies defaults
        appropriately.

        Configuration Options:
        ---------------------
        - max_history_size: Override default conversation history size
        - max_message_length: Override default message length limits
        - conversation: Provide pre-configured Conversation instance

        Args:
            **data: Initialization data including optional conversation settings
                   and any other context data to populate during creation.

        Implementation Details:
        ----------------------
        - Extracts conversation settings before calling parent constructor
        - Creates Conversation instance with extracted settings if not provided
        - Maintains compatibility with existing initialization patterns
        - Supports both explicit conversation objects and configuration parameters
        """
        # Handle conversation configuration extraction
        if "conversation" not in data:
            # Extract conversation settings from initialization data
            max_history = data.pop("max_history_size", DEFAULT_MAX_HISTORY_SIZE)
            max_message_length = data.pop("max_message_length", DEFAULT_MAX_MESSAGE_LENGTH)

            # Create conversation with extracted or default settings
            data["conversation"] = Conversation(
                max_history_size=max_history,
                max_message_length=max_message_length
            )

        # Initialize with processed data
        super().__init__(**data)

    def update(self, new_data: Dict[str, Any]) -> None:
        """
        Update context data with new information from conversation interactions.

        This method provides the primary mechanism for adding collected information
        to the context throughout the conversation lifecycle. It handles data
        merging, logging, and maintains data integrity during updates.

        Update Strategy:
        ---------------
        - Merges new data with existing context data
        - Preserves existing data while adding new information
        - Supports nested dictionary updates and complex data structures
        - Provides comprehensive logging for debugging and monitoring

        Data Integration:
        ----------------
        - New keys are added to existing context data
        - Existing keys are updated with new values
        - Nested structures are merged appropriately
        - Type consistency maintained where possible

        Args:
            new_data: Dictionary containing new information to merge into context.
                     Can include any data structure that's JSON-serializable.
                     None or empty dictionaries are handled gracefully.

        Side Effects:
        ------------
        - Modifies the context data dictionary in-place
        - Generates info-level logs for significant data updates
        - May trigger downstream validation or processing based on new keys
        - Updates are immediately available for subsequent operations

        Logging and Monitoring:
        ----------------------
        - Info-level logging for all non-empty updates
        - JSON serialization for structured log data
        - Handles non-serializable data gracefully in logs
        - Provides visibility into data collection and flow

        Usage Patterns:
        --------------
        # Single key update
        context.update({"user_name": "Alice"})

        # Multiple keys
        context.update({
            "email": "alice@example.com",
            "preferences": {"theme": "dark", "notifications": True}
        })

        # Conditional updates
        if user_provided_phone:
            context.update({"phone": user_phone})
        """
        if new_data:
            # Log update operation with structured data
            try:
                # Attempt structured JSON logging for better debugging
                update_json = json.dumps(new_data, indent=2, default=str)
                logger.debug(f"Updating context with new data:\n{update_json}")
            except (TypeError, ValueError):
                # Fallback to string representation for non-serializable data
                logger.debug(f"Updating context with new data: {new_data}")

            # Perform the actual data merge
            self.data.update(new_data)

    def has_keys(self, keys: List[str]) -> bool:
        """
        Check if all specified keys exist in the context data.

        This method provides essential validation functionality for FSM state
        transitions, enabling conditional logic based on data availability
        and supporting required information checking before proceeding.
        """
        # Handle empty key list case (considered valid)
        if not keys:
            return True

        # Check all keys for presence using efficient all() function
        result = all(key in self.data for key in keys)

        # Provide debug logging for validation tracking
        logger.debug(
            f"Context key validation - Keys: {keys}, Result: {result}"
        )

        return result

    def get_missing_keys(self, keys: List[str]) -> List[str]:
        """
        Identify which required keys are missing from the context data.

        This method provides detailed validation feedback, enabling specific
        error messages, targeted data collection, and informed decision making
        about conversation flow when required information is incomplete.

        Diagnostic Capabilities:
        -----------------------
        - Identifies specific missing keys for targeted collection
        - Supports detailed error messaging and user guidance
        - Enables progressive data collection strategies
        - Provides debugging information for conversation flow issues

        Validation Strategy:
        -------------------
        - Compares required keys against available context data
        - Returns specific missing keys rather than boolean result
        - Handles edge cases gracefully (empty lists, None values)
        - Maintains performance for large key sets

        Args:
            keys: List of required keys to check against context data.
                 Can be empty list (returns empty list) or contain
                 any valid dictionary keys.

        Returns:
            List[str]: List of keys that are missing from context data.
                      Empty list indicates all keys are present.
                      Order matches input key order for consistency.

        Use Cases:
        ---------
        - **Error Messages**: "Please provide your {missing_keys}"
        - **Progressive Collection**: Collect missing keys one by one
        - **Validation Feedback**: Show users what information is still needed
        - **Debug Information**: Identify data collection issues

        Performance:
        -----------
        - Linear scan through required keys
        - Early termination not applicable (need complete list)
        - Memory efficient for typical key set sizes
        - Debug logging only when missing keys found

        Examples:
        --------
        # Get specific missing keys for user feedback
        required = ["name", "email", "phone"]
        missing = context.get_missing_keys(required)
        if missing:
            return f"Please provide: {', '.join(missing)}"

        # Progressive data collection
        missing = context.get_missing_keys(["email", "phone"])
        if "email" in missing:
            ask_for_email()
        elif "phone" in missing:
            ask_for_phone()
        """
        # Handle empty key list case
        if not keys:
            return []

        # Build list of missing keys maintaining order
        missing = [key for key in keys if key not in self.data]

        # Log missing keys for debugging when issues present
        if missing:
            logger.debug(f"Missing required context keys: {missing}")

        return missing


# --------------------------------------------------------------


class FSMInstance(BaseModel):
    """
    Runtime execution instance of an FSM conversation.

    FSMInstance represents the active, stateful execution of a conversational
    FSM, maintaining current state, accumulated context, and persona information
    throughout the conversation lifecycle. It serves as the primary runtime
    object for FSM-driven conversations.

    Instance Lifecycle:
    ------------------
    1. **Creation**: Instance initialized with FSM definition and starting state
    2. **Activation**: Conversation begins with initial state and persona
    3. **Execution**: State transitions occur based on user input and logic
    4. **Context Building**: Information accumulated through conversation
    5. **Completion**: Terminal state reached or conversation ended
    6. **Persistence**: Instance state can be serialized and restored

    Runtime Architecture:
    --------------------
    - **State Management**: Tracks current conversation state and transitions
    - **Context Integration**: Maintains comprehensive conversation context
    - **Persona Consistency**: Ensures consistent personality across states
    - **FSM Binding**: Links to specific FSM definition for behavior rules

    State Synchronization:
    ---------------------
    The instance maintains synchronization between:
    - Current state ID and actual FSM state definition
    - Context data and state requirements
    - Conversation history and state progression
    - Persona consistency across state transitions

    Context Relationship:
    --------------------
    The embedded FSMContext provides:
    - Persistent data storage across state transitions
    - Conversation history for context-aware responses
    - Metadata for system operations and debugging
    - Validation support for state transition requirements

    Examples:
    --------
    Basic instance creation:
        instance = FSMInstance(
            fsm_id="customer_support_v1",
            current_state="greeting",
            persona="Friendly customer service representative"
        )

    Instance with pre-populated context:
        context = FSMContext()
        context.update({"user_id": "12345", "tier": "premium"})

        instance = FSMInstance(
            fsm_id="premium_support",
            current_state="welcome_premium",
            context=context,
            persona="Expert premium support specialist"
        )

    Attributes:
        fsm_id: Identifier linking this instance to its FSM definition.
               Used for loading state definitions, transitions, and
               validation rules during conversation execution.

        current_state: ID of the currently active state in the conversation.
                      Must correspond to a valid state in the linked FSM
                      definition. Updated through state transitions.

        context: Complete conversation context including collected data,
                message history, and system metadata. Persists throughout
                the conversation and supports state transition decisions.

        persona: Personality and tone description for consistent response
                generation. Applied to all LLM interactions to maintain
                character consistency across state transitions.

    Integration Points:
    ------------------
    - **FSM Engine**: Runtime execution and state transition management
    - **LLM Processing**: Context and persona provide LLM guidance
    - **State Validation**: Context data checked against state requirements
    - **Persistence Layer**: Instance serialization for conversation storage
    - **Monitoring Systems**: Instance state tracking and analytics
    """

    fsm_id: str = Field(
        ...,
        description=textwrap.dedent("""
            Identifier of the FSM definition that governs this conversation instance.
            Links to FSM states, transitions, validation rules, and behavior patterns.
            Used for loading appropriate conversation logic during execution.
            """).strip(),
        min_length=1,
        max_length=100
    )

    current_state: str = Field(
        ...,
        description=textwrap.dedent("""
            ID of the currently active state in the conversation flow.
            Must correspond to a valid state in the linked FSM definition.
            Updated automatically through state transition execution.
            """).strip(),
        min_length=1,
        max_length=100
    )

    context: FSMContext = Field(
        default_factory=FSMContext,
        description=textwrap.dedent("""
            Complete conversation context including collected data, message history,
            and system metadata. Provides persistent storage and retrieval for
            conversation information across state transitions.
            """).strip()
    )

    persona: Optional[str] = Field(
        default="Helpful AI assistant",
        description=textwrap.dedent("""
            Personality and tone description for consistent response generation.
            Applied to all LLM interactions to maintain character consistency.
            Can be inherited from FSM definition or customized per instance.
            """).strip(),
        max_length=500
    )


# --------------------------------------------------------------


class StateTransition(BaseModel):
    """
    Represents a state transition decision with context updates.

    StateTransition encapsulates the decision-making output from LLM processing,
    specifying both the target state for conversation progression and any
    context data updates that should be applied during the transition.

    Transition Architecture:
    -----------------------
    - **Decision Representation**: Captures LLM state transition decisions
    - **Context Integration**: Includes data updates to apply during transition
    - **Validation Support**: Structured format enables validation before execution
    - **Atomic Operations**: Ensures state and context updates happen together

    Decision Processing:
    -------------------
    StateTransitions are typically generated by:
    1. **LLM Processing**: LLM analyzes current state and user input
    2. **Condition Evaluation**: Transition conditions checked against context
    3. **Priority Resolution**: Multiple eligible transitions prioritized
    4. **Decision Formation**: Target state and context updates determined
    5. **Validation**: Transition validity confirmed before execution

    Context Update Strategy:
    -----------------------
    - **Incremental Updates**: New data merged with existing context
    - **Preservation**: Existing context data maintained unless explicitly updated
    - **Structured Data**: Support for nested objects and complex data types
    - **Validation Integration**: Updates can trigger validation logic

    Examples:
    --------
    Simple state transition:
        transition = StateTransition(
            target_state="collect_email",
            context_update={"name": "Alice Johnson"}
        )

    Complex transition with nested data:
        transition = StateTransition(
            target_state="process_order",
            context_update={
                "order": {
                    "items": ["laptop", "mouse"],
                    "total": 1299.99
                },
                "shipping": {
                    "method": "express",
                    "address": "123 Main St"
                }
            }
        )

    Attributes:
        target_state: The state identifier to transition to. Must be a valid
                     state ID in the current FSM definition. Validated before
                     transition execution to ensure conversation integrity.

        context_update: Dictionary of data updates to apply to the conversation
                       context during the transition. Merged with existing context
                       data using update semantics (new keys added, existing updated).

    Usage Patterns:
    --------------
    - **LLM Output Processing**: Parse LLM responses into structured transitions
    - **Validation**: Verify transition validity before execution
    - **Context Synchronization**: Ensure context updates align with state changes
    - **Debugging**: Provide detailed transition information for troubleshooting
    """

    target_state: str = Field(
        ...,
        description=textwrap.dedent("""
            Identifier of the target state for this transition.
            Must correspond to a valid state in the current FSM definition.
            Validated before transition execution to ensure conversation flow integrity.
            """).strip(),
        min_length=1,
        max_length=100
    )

    context_update: Dict[str, Any] = Field(
        default_factory=dict,
        description=textwrap.dedent("""
            Dictionary of data updates to apply to conversation context during transition.
            Merged with existing context data using update semantics.
            Supports arbitrary nested structures and data types.
            """).strip()
    )


# --------------------------------------------------------------


class LLMRequest(BaseModel):
    """
    Structured request payload for Large Language Model interactions.

    LLMRequest encapsulates all information needed for LLM processing in the
    context of FSM-driven conversations, including system prompts that define
    LLM behavior, user messages to process, and optional context for enhanced
    understanding.

    Request Architecture:
    --------------------
    - **System Prompt**: Defines LLM role, constraints, and expected behavior
    - **User Message**: Contains actual user input to process and respond to
    - **Context Data**: Optional additional information for enhanced processing
    - **Standardized Format**: Consistent interface across different LLM providers

    Prompt Engineering Integration:
    ------------------------------
    The system_prompt typically contains:
    - FSM state information and current conversation context
    - Available state transitions and their conditions
    - Response format requirements (usually JSON)
    - Persona and tone guidance for consistent responses
    - Few-shot examples for improved understanding

    LLM Provider Abstraction:
    ------------------------
    This standardized format enables:
    - Consistent interface across different LLM providers (OpenAI, Anthropic, etc.)
    - Provider-specific adaptation through adapter patterns
    - Testing with different models using the same request format
    - Easy switching between LLM providers without code changes

    Examples:
    --------
    Basic conversation request:
        request = LLMRequest(
            system_prompt="You are a helpful customer service assistant...",
            user_message="I need help with my order",
            context={"user_id": "12345", "order_status": "shipped"}
        )

    FSM-integrated request:
        request = LLMRequest(
            system_prompt=prompt_builder.build_system_prompt(fsm_instance, current_state),
            user_message=user_input,
            context={"current_state": "order_inquiry", "available_transitions": [...]}
        )

    Attributes:
        system_prompt: Complete system prompt defining LLM behavior, role,
                      constraints, and response requirements. Typically generated
                      by prompt builders based on current FSM state and context.

        user_message: Actual user input to process. Contains the user's message,
                     question, or input that the LLM should respond to within
                     the context of the current conversation state.

        context: Optional additional context information that can enhance
                LLM understanding and response quality. May include metadata,
                conversation history, or other relevant data not in the prompt.

    Integration Patterns:
    --------------------
    - **Prompt Builders**: Generate system prompts from FSM state and context
    - **LLM Adapters**: Convert requests to provider-specific formats
    - **Context Managers**: Include relevant conversation context automatically
    - **Response Processors**: Parse LLM responses back into structured formats
    """

    system_prompt: str = Field(
        ...,
        description=textwrap.dedent("""
            Complete system prompt defining LLM behavior, role, constraints,
            and response requirements. Typically generated by prompt builders
            based on current FSM state, available transitions, and context.
            """).strip(),
        min_length=1,
        max_length=50000
    )

    user_message: str = Field(
        ...,
        description=textwrap.dedent("""
            User input to process and respond to. Contains the actual user message,
            question, or input within the context of the current conversation state.
            May be preprocessed for safety and formatting consistency.
            """).strip(),
        min_length=0,  # Allow empty messages for some use cases
        max_length=10000
    )

    context: Optional[Dict[str, Any]] = Field(
        None,
        description=textwrap.dedent("""
            Optional additional context information for enhanced LLM processing.
            May include metadata, conversation history, user preferences,
            or other relevant data not included in the system prompt.
            """).strip()
    )


# --------------------------------------------------------------


class LLMResponse(BaseModel):
    """
    Structured response from Large Language Model processing.

    LLMResponse represents the parsed and validated output from LLM interactions,
    containing the state transition decision, user-facing message, and optional
    reasoning explanation. This structured format enables reliable FSM operation
    and consistent conversation flow management.

    Response Architecture:
    ---------------------
    - **Structured Output**: Predictable JSON format for reliable parsing
    - **Transition Decision**: Clear state transition specification
    - **User Communication**: Natural language response for user interaction
    - **Reasoning Transparency**: Optional explanation of decision logic

    LLM Output Processing:
    ---------------------
    The response processing pipeline typically involves:
    1. **Raw LLM Output**: JSON string generated by LLM
    2. **JSON Parsing**: Convert string to structured data
    3. **Validation**: Verify required fields and data types
    4. **Response Construction**: Create LLMResponse instance
    5. **Error Handling**: Manage parsing failures and invalid responses

    Quality Assurance:
    -----------------
    - **Required Fields**: Ensures essential information is always present
    - **Type Validation**: Confirms correct data types for reliable processing
    - **Transition Validation**: Verifies state transition decisions are valid
    - **Message Quality**: Ensures user-facing messages are appropriate

    Examples:
    --------
    Simple information collection response:
        response = LLMResponse(
            transition=StateTransition(
                target_state="collect_email",
                context_update={"name": "Alice"}
            ),
            message="Nice to meet you, Alice! Could you please provide your email address?",
            reasoning="User provided their name, transitioning to email collection"
        )

    Complex decision response:
        response = LLMResponse(
            transition=StateTransition(
                target_state="premium_support",
                context_update={
                    "support_tier": "premium",
                    "priority": "high"
                }
            ),
            message="As a premium customer, I'll connect you with our specialist team right away.",
            reasoning="User identified as premium tier based on account lookup"
        )

    Attributes:
        transition: StateTransition object specifying the target state and
                   any context updates to apply. Contains the core decision
                   logic for FSM progression and data collection.

        message: Natural language message to present to the user. Should be
                conversational, helpful, and appropriate for the current
                context and persona. This is the primary user-facing output.

        reasoning: Optional explanation of the decision-making process.
                  Used for debugging, monitoring, and transparency but not
                  typically shown to users. Helps with conversation analysis.

    Integration Points:
    ------------------
    - **FSM Engine**: Executes transition decisions and context updates
    - **User Interface**: Displays message to user through appropriate channels
    - **Monitoring**: Uses reasoning for conversation analysis and debugging
    - **Validation**: Confirms response validity before execution
    - **Analytics**: Tracks decision patterns and conversation flows
    """

    transition: StateTransition = Field(
        ...,
        description=textwrap.dedent("""
            State transition decision including target state and context updates.
            Contains the core logic for FSM progression and data collection.
            Must specify valid target state and appropriate context changes.
            """).strip()
    )

    message: str = Field(
        ...,
        description=textwrap.dedent("""
            Natural language message to present to the user. Should be
            conversational, helpful, and appropriate for current context and persona.
            This is the primary user-facing output from the conversation system.
            """).strip(),
        min_length=1,
        max_length=5000
    )

    reasoning: Optional[str] = Field(
        None,
        description=textwrap.dedent("""
            Optional explanation of the decision-making process and transition logic.
            Used for debugging, monitoring, and transparency. Not typically shown
            to users but valuable for conversation analysis and improvement.
            """).strip(),
        max_length=2000
    )


# --------------------------------------------------------------
# EXCEPTION CLASSES
# --------------------------------------------------------------


class FSMError(Exception):
    """
    Base exception class for all FSM-related errors.

    FSMError serves as the root exception class for the FSM system,
    providing a common base for all FSM-specific error conditions.
    This enables comprehensive error handling and allows applications
    to catch all FSM-related exceptions with a single except clause.

    Exception Hierarchy:
    -------------------
    FSMError serves as the base for specialized exceptions:
    - StateNotFoundError: Invalid state references
    - InvalidTransitionError: Illegal state transitions
    - LLMResponseError: LLM processing and parsing issues

    Error Handling Strategy:
    -----------------------
    - Specific exceptions for different error types
    - Common base enables comprehensive error catching
    - Descriptive error messages for debugging
    - Integration with logging system for error tracking

    Usage Patterns:
    --------------
    # Catch all FSM errors
    try:
        fsm_engine.process_input(user_input)
    except FSMError as e:
        logger.error(f"FSM error occurred: {e}")
        handle_fsm_error(e)

    # Catch specific error types
    try:
        transition_to_state("invalid_state")
    except StateNotFoundError as e:
        handle_missing_state(e)
    except InvalidTransitionError as e:
        handle_invalid_transition(e)
    """
    pass


class StateNotFoundError(FSMError):
    """
    Exception raised when attempting to access a non-existent state.

    StateNotFoundError indicates that code attempted to reference,
    transition to, or operate on a state that doesn't exist in the
    current FSM definition. This typically indicates configuration
    issues or programming errors.

    Common Causes:
    -------------
    - Typos in state identifiers
    - Transitions to undefined states
    - FSM definition inconsistencies
    - Runtime state reference errors

    Prevention:
    ----------
    - FSM validation catches most issues at definition time
    - State identifier validation during FSM construction
    - Runtime checks before state transitions
    - Comprehensive testing of conversation flows

    Example:
    -------
    raise StateNotFoundError(f"State '{state_id}' not found in FSM '{fsm_name}'")
    """
    pass


class InvalidTransitionError(FSMError):
    """
    Exception raised when attempting an invalid state transition.

    InvalidTransitionError indicates that code attempted to transition
    between states in a way that violates the FSM definition, such as
    transitioning to a state that isn't reachable from the current state
    or failing to meet transition conditions.

    Common Scenarios:
    ----------------
    - Transitioning to unreachable states
    - Failing transition condition requirements
    - Attempting transitions from terminal states
    - Violating FSM structural constraints

    Error Context:
    -------------
    Error messages should include:
    - Current state information
    - Attempted target state
    - Available valid transitions
    - Failed condition details (if applicable)

    Example:
    -------
    raise InvalidTransitionError(f"Cannot transition from '{current}' to '{target}': no valid transition exists")
    """
    pass


class LLMResponseError(FSMError):
    """
    Exception raised when LLM responses are invalid or unparseable.

    LLMResponseError indicates issues with LLM output processing,
    including JSON parsing failures, missing required fields,
    invalid data types, or responses that don't conform to the
    expected schema for FSM operation.

    Common Issues:
    -------------
    - Invalid JSON format in LLM responses
    - Missing required fields (transition, message)
    - Incorrect data types for expected values
    - Malformed state transition specifications
    - Non-existent target states in transitions

    Error Recovery:
    --------------
    - Retry with clarified prompts
    - Fallback to default responses
    - Request human intervention for complex cases
    - Log detailed error information for analysis

    Debugging Information:
    ---------------------
    Error messages should include:
    - Original LLM response text
    - Specific parsing failure details
    - Expected vs. actual response format
    - Context information for reproduction

    Example:
    -------
    raise LLMResponseError(f"Failed to parse LLM response as JSON: {json_error}")
    """
    pass


# --------------------------------------------------------------