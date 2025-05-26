# LLM Guide: Understanding LLM-FSM Architecture and Your Role

## Purpose of This Document

This document is designed to help Large Language Models understand the LLM-FSM framework, their role within it, and how to properly respond to requests when operating as the natural language processing component of a Finite State Machine system.

## Table of Contents

1. [System Overview for LLMs](#system-overview-for-llms)
2. [Your Role as the LLM Component](#your-role-as-the-llm-component)
3. [Understanding State Machines](#understanding-state-machines)
4. [Prompt Structure You Will Receive](#prompt-structure-you-will-receive)
5. [Response Format Requirements](#response-format-requirements)
6. [Context Management](#context-management)
7. [State Transition Logic](#state-transition-logic)
8. [Information Extraction](#information-extraction)
9. [Error Handling](#error-handling)
10. [Advanced Scenarios](#advanced-scenarios)
11. [Common Patterns and Examples](#common-patterns-and-examples)
12. [Implementation Details](#implementation-details)

---

## System Overview for LLMs

### What is LLM-FSM?

LLM-FSM is a framework that combines:
- **Finite State Machines (FSMs)**: Providing structured conversation flow
- **Large Language Models (You)**: Providing natural language understanding and generation
- **Python Orchestration**: Managing state, context, and transitions

### Your Position in the Architecture

```
User Input → Python Framework → YOU (LLM) → Python Framework → User Output
                ↓                              ↑
            State & Context ←──────────────────┘
```

You are the intelligence layer that:
1. Understands natural language input
2. Extracts relevant information
3. Determines appropriate state transitions
4. Generates natural responses

---

## Your Role as the LLM Component

### Primary Responsibilities

1. **Natural Language Understanding**
   - Parse user messages regardless of phrasing
   - Identify user intent within the current state context
   - Handle variations, typos, and colloquialisms

2. **Information Extraction**
   - Extract data specified in `required_context_keys`
   - Identify additional relevant information
   - Structure extracted data appropriately

3. **State Transition Decision**
   - Evaluate which transition best matches user input
   - Consider transition descriptions and priorities
   - Make logical decisions based on conversation flow

4. **Response Generation**
   - Create natural, contextually appropriate responses
   - Maintain specified persona if provided
   - Guide users toward state objectives

### What You DON'T Handle

- Actual state management (Python handles this)
- Persistent storage (Python handles this)
- External API calls (Handler system handles this)
- Conversation history management (Python handles this)

---

## Understanding State Machines

### Core Concepts

**State**: A specific point in the conversation with a defined purpose
```
Example: "collecting_email" state
- Purpose: Obtain user's email address
- Will transition to: "collecting_phone" after email received
```

**Transition**: A path from one state to another
```
Example: From "greeting" to "collect_info"
- Triggered when: User indicates readiness to proceed
- Description: "User wants to continue with registration"
```

**Context**: Information collected during conversation
```
Example: {"name": "John", "email": "john@example.com", "age": 25}
```

### State Types

1. **Initial State**: Where conversations begin
2. **Intermediate States**: Information gathering or processing
3. **Terminal States**: No outgoing transitions (conversation ends)

---

## Prompt Structure You Will Receive

You will receive structured prompts in XML-like format:

```xml
<task>
You are the Natural Language Understanding component in a Finite State Machine (FSM) based conversational system.
Your responsibilities:
- Process user input based on current state
- Collect required information
- Select appropriate transitions
- Generate natural responses
</task>

<fsm>
<persona>
[Optional: Personality/tone instructions]
Example: "You are a friendly, professional customer service representative"
</persona>

<current_state>
<id>state_identifier</id>
<description>Human-readable state description</description>
<purpose>What this state should accomplish</purpose>
<state_instructions>
[Optional: Specific instructions for this state]
</state_instructions>
<information_to_collect>
[Optional: List of required context keys]
Example: name, email, phone_number
</information_to_collect>
</current_state>

<current_context><![CDATA[
{
    "existing_key": "existing_value",
    "user_name": "Alice",
    "_conversation_id": "uuid-here"
}
]]></current_context>

<conversation_history><![CDATA[
[
    {"user": "Previous user message"},
    {"system": "Previous system response"},
    {"user": "Another user message"},
    {"system": "Another system response"}
]
]]></conversation_history>

<valid_states>
state1, state2, current_state
</valid_states>

<transitions><![CDATA[
[
    {
        "to": "next_state_id",
        "desc": "When this transition should occur",
        "priority": 100,
        "conditions": [
            {
                "desc": "User has provided email",
                "keys": ["email"]
            }
        ]
    },
    {
        "to": "alternate_state",
        "desc": "Alternative transition condition",
        "priority": 200
    }
]
]]></transitions>

<response>
Your response must be valid JSON with this structure:
{
    "transition": {
        "target_state": "state_id",
        "context_update": {
            "key": "value"
        }
    },
    "message": "Your natural language response",
    "reasoning": "Optional explanation"
}
</response>

<guidelines>
- Extract all required information from user input
- Store relevant information even if unexpected (using `_extra`)
- Reference current context for continuity
- Only transition when conditions are met
- Maintain persona consistently
</guidelines>

<format_rules>
Return ONLY valid JSON - no markdown, no explanations outside JSON.
</format_rules>
</fsm>
```

---

## Response Format Requirements

### Exact JSON Structure Required

```json
{
    "transition": {
        "target_state": "exact_state_id_from_valid_states",
        "context_update": {
            "extracted_key": "extracted_value",
            "another_key": "another_value",
            "_extra": {
                "unexpected_info": "value"
            }
        }
    },
    "message": "Natural language response to the user",
    "reasoning": "Optional: Why you made this decision"
}
```

### Field Specifications

1. **transition.target_state** (REQUIRED)
   - Must be EXACTLY one of the state IDs from `<valid_states>`
   - Case-sensitive
   - No variations or modifications

2. **transition.context_update** (REQUIRED)
   - Dictionary of extracted information
   - Keys should match `required_context_keys` when applicable
   - Use `_extra` sub-object for unexpected but relevant information

3. **message** (REQUIRED)
   - Natural language response to show the user
   - Should align with state purpose and persona
   - Never mention technical details about states or transitions

4. **reasoning** (OPTIONAL)
   - Internal explanation of your decision
   - Not shown to user
   - Useful for debugging

### Example Responses

#### Successful Information Collection
```json
{
    "transition": {
        "target_state": "phone_collection",
        "context_update": {
            "email": "user@example.com",
            "email_domain": "example.com",
            "_extra": {
                "mentioned_urgency": true
            }
        }
    },
    "message": "Thank you! I've noted your email as user@example.com. Could you also provide a phone number where we can reach you?",
    "reasoning": "User provided valid email, extracted domain, noted they mentioned urgency"
}
```

#### Staying in Current State
```json
{
    "transition": {
        "target_state": "email_collection",
        "context_update": {
            "_extra": {
                "invalid_email_attempt": "userexample"
            }
        }
    },
    "message": "I noticed that doesn't look like a complete email address. Could you please provide your full email address including the @ symbol?",
    "reasoning": "User provided malformed email, staying in same state to retry"
}
```

---

## Context Management

### Understanding Context Keys

1. **User-Defined Keys**: Keys specified in the FSM definition
   ```
   "name", "email", "order_id", "feedback"
   ```

2. **System Keys** (prefixed with `_`): Managed by the framework
   ```
   "_conversation_id": Unique conversation identifier
   "_current_state": Current state (don't modify)
   "_user_input": Recent user message
   "_timestamp": Last update time
   ```

3. **Handler-Added Keys**: May be added by custom handlers
   ```
   "validated_email": true
   "customer_tier": "premium"
   ```

### Context Update Rules

1. **Always preserve existing context** unless explicitly updating
2. **Extract information even if not required** by current state
3. **Use descriptive keys** that match the information type
4. **Structure nested data** appropriately

#### Good Context Update
```json
{
    "context_update": {
        "shipping_address": {
            "street": "123 Main St",
            "city": "Boston",
            "state": "MA",
            "zip": "02101"
        },
        "shipping_preference": "express"
    }
}
```

#### Poor Context Update
```json
{
    "context_update": {
        "address": "123 Main St Boston",  // Lost structure
        "info": "express shipping"        // Vague key
    }
}
```

---

## State Transition Logic

### Decision Process

1. **Check Required Context Keys**
   - If state requires keys not yet collected, generally stay in current state
   - Exception: User explicitly wants to skip or go back

2. **Evaluate Transition Descriptions**
   - Match user intent to transition descriptions
   - Consider priority (lower numbers = higher priority)

3. **Apply Logical Reasoning**
   - Consider conversation flow
   - Respect user's explicit requests
   - Handle edge cases gracefully

### Transition Examples

#### Scenario 1: All Information Collected
```
Current State: "collecting_email"
Required Keys: ["email"]
User Input: "My email is john@example.com"

Decision: Transition to next state
Reasoning: Email provided and valid
```

#### Scenario 2: Partial Information
```
Current State: "collecting_contact"
Required Keys: ["email", "phone"]
User Input: "My email is john@example.com"

Decision: Stay in current state
Response: "Thanks! I have your email. Could you also provide your phone number?"
```

#### Scenario 3: User Wants to Skip
```
Current State: "collecting_phone"
Required Keys: ["phone"]
User Input: "I don't want to provide my phone number"

Decision: Transition to next state (if allowed)
Context Update: {"phone_declined": true}
```

---

## Information Extraction

### Extraction Strategies

1. **Direct Extraction**
   ```
   User: "My email is alice@example.com"
   Extract: {"email": "alice@example.com"}
   ```

2. **Inference from Context**
   ```
   User: "Yes, I'm over 18"
   Extract: {"age_confirmed": true, "is_adult": true}
   ```

3. **Multiple Information**
   ```
   User: "I'm John Smith and my number is 555-1234"
   Extract: {"name": "John Smith", "phone": "555-1234"}
   ```

4. **Structured Data**
   ```
   User: "Ship it to 123 Main St, Boston, MA 02101"
   Extract: {
       "shipping_address": {
           "street": "123 Main St",
           "city": "Boston",
           "state": "MA",
           "zip": "02101"
       }
   }
   ```

### Handling Ambiguity

When information is ambiguous:
1. Ask for clarification in your message
2. Store partial information in `_extra`
3. Stay in current state if critical information is unclear

```json
{
    "transition": {
        "target_state": "collecting_date",
        "context_update": {
            "_extra": {
                "ambiguous_date": "next Friday",
                "clarification_needed": true
            }
        }
    },
    "message": "When you say 'next Friday', do you mean November 8th or November 15th?"
}
```

---

## Error Handling

### Invalid State Transitions

If you accidentally specify an invalid state:
- The framework will catch this and keep the current state
- Your message will still be shown to the user
- The conversation continues normally

### Malformed Responses

Always return valid JSON. If you return malformed JSON:
- The framework will attempt to extract JSON from your response
- If that fails, an error will occur
- Always validate your JSON structure

### Recovery Strategies

1. **When confused about state**: Stay in current state
2. **When information is partial**: Request clarification
3. **When user is frustrated**: Acknowledge and offer alternatives

---

## Advanced Scenarios

### 1. Multi-State Information Collection

Sometimes information collected spans multiple states:

```
State 1: Collect name
State 2: Collect email  
State 3: Collect phone
State 4: Confirm all information
```

In State 4, reference previously collected context:
```json
{
    "message": "Let me confirm: Your name is {name}, email is {email}, and phone is {phone}. Is this correct?",
    "context_update": {
        "confirmed": true
    }
}
```

### 2. Conditional Transitions

When multiple transitions are possible:

```json
transitions: [
    {"to": "premium_flow", "desc": "User is premium member"},
    {"to": "standard_flow", "desc": "User is standard member"},
    {"to": "guest_flow", "desc": "User is not a member"}
]
```

Check context and user input to determine the appropriate path.

### 3. Handling User Corrections

User: "Actually, my email is jane@example.com, not john@example.com"

```json
{
    "context_update": {
        "email": "jane@example.com",
        "email_previous": "john@example.com",
        "_correction_made": true
    },
    "message": "I've updated your email to jane@example.com. Thank you for the correction!"
}
```

### 4. Complex State Purposes

Some states have multiple objectives:

```xml
<purpose>
1. Verify user's identity with security question
2. If verification fails, offer password reset
3. If user requests help, provide support options
</purpose>
```

Handle each objective appropriately based on user input.

---

## Common Patterns and Examples

### Pattern 1: Welcome and Route

**State Purpose**: Greet user and understand their need
**Transitions**: Different paths based on user intent

```json
// User: "I need help with my order"
{
    "transition": {
        "target_state": "order_support",
        "context_update": {
            "intent": "order_help"
        }
    },
    "message": "I'd be happy to help you with your order. Could you please provide your order number?"
}
```

### Pattern 2: Form Filling

**State Purpose**: Collect required information
**Key Behavior**: Stay until all required keys collected

```json
// State requires: ["first_name", "last_name", "email"]
// User: "I'm John Smith"
{
    "transition": {
        "target_state": "personal_info",  // Stay in same state
        "context_update": {
            "first_name": "John",
            "last_name": "Smith"
        }
    },
    "message": "Nice to meet you, John Smith! Could you also provide your email address?"
}
```

### Pattern 3: Confirmation

**State Purpose**: Verify collected information
**Transitions**: Proceed if confirmed, go back if corrections needed

```json
// User: "Yes, that's all correct"
{
    "transition": {
        "target_state": "processing",
        "context_update": {
            "info_confirmed": true,
            "confirmation_timestamp": "2024-01-20T10:30:00Z"
        }
    },
    "message": "Perfect! I'm now processing your request. This will just take a moment."
}
```

### Pattern 4: Error Recovery

**State Purpose**: Handle problems gracefully
**Key Behavior**: Provide alternatives

```json
// User: "This isn't working, I keep getting errors"
{
    "transition": {
        "target_state": "support_escalation",
        "context_update": {
            "error_reported": true,
            "user_frustration_level": "high",
            "_extra": {
                "user_comment": "keeps getting errors"
            }
        }
    },
    "message": "I apologize for the difficulties you're experiencing. Let me connect you with our support team who can help resolve this issue immediately."
}
```

---

## Implementation Details

### How Your Responses Are Processed

1. **JSON Parsing**
   ```python
   response_data = json.loads(llm_response)
   transition = StateTransition(
       target_state=response_data["transition"]["target_state"],
       context_update=response_data["transition"]["context_update"]
   )
   ```

2. **Validation**
   ```python
   # Verify target state exists
   # Check transition is allowed
   # Validate conditions are met
   ```

3. **Context Update**
   ```python
   instance.context.update(transition.context_update)
   ```

4. **State Change**
   ```python
   instance.current_state = transition.target_state
   ```

5. **Response Delivery**
   ```python
   return response_data["message"]  # To user
   ```

### Token Considerations

- Conversation history is automatically trimmed to stay within token limits
- Long messages are truncated with "... [truncated]"
- Focus on recent context rather than full history

### Security Boundaries

- User input is sanitized before reaching you
- Your responses are also sanitized
- Don't attempt to break out of the JSON structure
- Don't reference system internals in user-facing messages

---

## Best Practices for LLMs

1. **Always Return Valid JSON**
   - No markdown code blocks
   - No explanatory text outside JSON
   - Validate structure before responding

2. **Stay in Character**
   - Maintain the specified persona
   - Don't mention states, transitions, or technical details to users
   - Be helpful and natural

3. **Extract Comprehensively**
   - Capture all relevant information
   - Use appropriate data types (boolean for yes/no, etc.)
   - Structure complex data properly

4. **Handle Edge Cases**
   - User refuses to provide information
   - User provides partial information  
   - User wants to go back or start over
   - User is confused or frustrated

5. **Be Predictable**
   - Follow the FSM structure
   - Make logical transitions
   - Provide clear, helpful responses

---

## Summary

As an LLM in the LLM-FSM system, you are the intelligence that makes structured conversations feel natural. Your role is to:

1. Understand user intent within the current state context
2. Extract required and additional relevant information
3. Make appropriate state transition decisions
4. Generate helpful, natural responses
5. Always return properly formatted JSON

The framework handles all state management, persistence, and execution flow. Your focus is purely on natural language understanding and generation within the structured context provided.

Remember: You make conversations feel human while the FSM ensures they follow a reliable, testable path.

---

## Quick Reference Card

**Required JSON Structure**:
```json
{
    "transition": {
        "target_state": "valid_state_id",
        "context_update": {}
    },
    "message": "Response to user",
    "reasoning": "Optional internal note"
}
```

**Key Rules**:
- ✅ Always return valid JSON
- ✅ Use exact state IDs from valid_states
- ✅ Extract information to context_update
- ✅ Generate natural messages
- ❌ Don't mention technical details to users
- ❌ Don't modify system keys (those with _)
- ❌ Don't break JSON structure
- ❌ Don't include markdown formatting