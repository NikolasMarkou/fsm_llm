# LLM Guide: Understanding LLM-FSM Architecture and Your Roles

## Purpose of This Document

This document is designed to help Large Language Models understand the **LLM-FSM framework**, their specialized roles within its **2-pass architecture**, and how to properly respond to requests for each distinct task.

## Table of Contents

1.  [System Overview: The 2-Pass Architecture](#system-overview-the-2-pass-architecture)
2.  [Your Specialized Roles as the LLM Component](#your-specialized-roles-as-the-llm-component)
3.  [Task 1: Data Extraction](#task-1-data-extraction)
    *   Prompt Structure for Data Extraction
    *   Response Format for Data Extraction
4.  [Task 2: Transition Decision (When Ambiguous)](#task-2-transition-decision-when-ambiguous)
    *   Prompt Structure for Transition Decision
    *   Response Format for Transition Decision
5.  [Task 3: Response Generation](#task-3-response-generation)
    *   Prompt Structure for Response Generation
    *   Response Format for Response Generation
6.  [Context and History Management](#context-and-history-management)
7.  [Common Patterns and Examples](#common-patterns-and-examples)
8.  [Best Practices Summary](#best-practices-summary)

---

## System Overview: The 2-Pass Architecture

LLM-FSM uses a sophisticated **2-pass architecture** to build high-quality, stateful conversations. This separates the process of understanding user input from generating a response. You, the LLM, are a critical component in this flow, but you will be called for different, highly-focused tasks.

### The Flow of a User Message

```
User Input
    │
    ▼
┌───────────────────────────────────────────────┐
│              Python Framework                 │
│         (State & Context Management)          │
└───────────────────────────────────────────────┘
    │
    ├─► **Pass 1: Analysis & Transition**
    │   ├─► Call YOU for [Data Extraction]
    │   └─◄ Return Extracted Data
    │   ├─── System Evaluates Transitions (Rule-based)
    │   └─── (Optional) Call YOU for [Transition Decision] if ambiguous
    │
    ├─► **Pass 2: Response Generation**
    │   ├─► Call YOU for [Response Generation] based on the *new* state
    │   └─◄ Return User-Facing Message
    │
    ▼
┌───────────────────────────────────────────────┐
│              Python Framework                 │
│ (Sends final message to user, updates history)│
└───────────────────────────────────────────────┘
    │
    ▼
User Output
```

This ensures that the final response you generate is always based on the correct state after a transition has occurred, leading to more consistent and logical conversations.

---

## Your Specialized Roles as the LLM Component

Your responsibilities are divided into three distinct roles. Each prompt you receive will clearly state which role you need to perform. **It is critical that you only perform the requested task.**

### Role 1: The Data Extractor

*   **Goal:** Understand the user's message and extract structured information.
*   **DO:** Analyze text, identify entities, and structure them as JSON. Provide a confidence score.
*   **DON'T:** Generate a user-facing message or decide on the next state.

### Role 2: The Transition Decider

*   **Goal:** When the system identifies multiple valid transitions, you choose the most logical one.
*   **DO:** Select one `target_state` from a provided list of options based on the conversation context.
*   **DON'T:** Extract new data or generate a user-facing message.

### Role 3: The Response Generator

*   **Goal:** Create a natural, context-aware, user-facing message.
*   **DO:** Use the persona, conversation history, and final state context to craft a helpful response.
*   **DON'T:** Extract data or make transition decisions. The state transition has *already happened*.

---

## Task 1: Data Extraction

This is the first pass. Your job is to understand and structure information from the user's message.

### Prompt Structure for Data Extraction

```xml
<task>
You are the data extraction component.
- Analyze and understand user input thoroughly.
- Extract relevant information and data from the user input.
</task>

<data_extraction>
    <extraction_focus>
        <purpose>The objective of the current state.</purpose>
        <extraction_instructions>
            Specific instructions for what to extract.
        </extraction_instructions>
        <information_to_extract>
            <collect>key1</collect>
            <collect>key2</collect>
        </information_to_extract>
    </extraction_focus>

    <conversation_history><![CDATA[[{"user": "..."}, {"assistant": "..."}]]]></conversation_history>
    <current_context><![CDATA[{"existing_key": "value"}]]></current_context>

    <response_format>
    Your response must be valid JSON with the following structure:
    {
        "extracted_data": {
            "key1": "value1",
            "_extra": {}
        },
        "confidence": 0.95,
        "reasoning": "Brief explanation of extraction decisions"
    }
    </response_format>

    <format_rules>
    - Return ONLY valid JSON - no markdown, no explanations outside the JSON.
    - `confidence` is REQUIRED (0.0 to 1.0).
    </format_rules>
</data_extraction>
```

### Response Format for Data Extraction

**Your entire output MUST be this JSON structure and nothing else.**

```json
{
    "extracted_data": {
        "key_from_instructions": "value_extracted_from_user_message",
        "another_key": 123,
        "_extra": {
            "unexpected_but_relevant_info": "value"
        }
    },
    "confidence": 0.95,
    "reasoning": "User clearly stated their name and mentioned they were in a hurry, which I've noted in _extra."
}
```

---

## Task 2: Transition Decision (When Ambiguous)

You will only receive this task if the system's rule-based evaluator finds multiple valid next steps. Your job is to break the tie.

### Prompt Structure for Transition Decision

```xml
<task>
You are the decision-making component.
- Analyze the situation and select the most appropriate next step.
</task>

<transition_decision>
    <current_situation>
        <current_step>current_state_id</current_step>
        <user_message>The user's latest message.</user_message>
        <extracted_information><![CDATA[{"info_from_pass_1": "value"}]]></extracted_information>
    </current_situation>

    <available_options>
        <option id="1">
            <target>next_state_1</target>
            <when>Description of when to choose this transition.</when>
            <priority>100</priority>
        </option>
        <option id="2">
            <target>next_state_2</target>
            <when>Description of another condition.</when>
            <priority>200</priority>
        </option>
    </available_options>

    <context_summary><![CDATA[{"relevant_context": "for_decision"}]]></context_summary>

    <response_format>
    Your response must be valid JSON with the following structure:
    {
        "selected_transition": "target_state_name",
        "reasoning": "Brief explanation for your choice"
    }
    </response_format>

    <format_rules>
    - Return ONLY valid JSON.
    - `selected_transition` MUST EXACTLY match one of the <target> values.
    </format_rules>
</transition_decision>
```

### Response Format for Transition Decision

**Your entire output MUST be this JSON structure and nothing else.**

```json
{
    "selected_transition": "next_state_1",
    "reasoning": "The user explicitly asked for help with their order, which matches the description for the 'order_support' transition."
}
```

---

## Task 3: Response Generation

This is the final pass. The state transition is complete. Your job is to craft the user-facing message.

### Prompt Structure for Response Generation

```xml
<task>
You are the Response Generation component.
- Generate an appropriate user-facing response.
- Acknowledge new information and guide the conversation.
</task>

<response_generation>
    <persona>Your persona, e.g., "A friendly and professional assistant."</persona>

    <final_state_context>
        <current_state>new_state_id</current_state>
        <purpose>The goal of this new state.</purpose>
        <response_instructions>
            Specific instructions for how to respond in this new state.
        </response_instructions>
        <information_still_needed>
            - key1
            - key2
        </information_still_needed>
    </final_state_context>

    <user_message><original_input>The user's original message.</original_input></user_message>
    <extracted_data><![CDATA[{"info_from_pass_1": "value"}]]></extracted_data>
    <conversation_history><![CDATA[[{"user": "..."}, {"assistant": "..."}]]]></conversation_history>
    <current_context><![CDATA[{"user_name": "Alice", "key1": "value1"}]]></current_context>

    <response_format>
    Your response must be valid JSON with the following structure:
    {
        "message": "Your natural response to the user",
        "reasoning": "Brief internal reasoning (optional)"
    }
    </response_format>
</response_generation>
```

### Response Format for Response Generation

**Your entire output MUST be this JSON structure and nothing else.**

```json
{
    "message": "Thanks for confirming your email, Alice! Now, could you please provide a phone number where we can reach you?",
    "reasoning": "Acknowledged the extracted email from the previous turn and am now prompting for the phone number, which is the purpose of the current 'collect_phone' state."
}
```

---

## Context and History Management

*   **`<current_context>`**: Contains all information collected so far. You should reference this for continuity.
*   **`<conversation_history>`**: Contains recent turns of the conversation. This is automatically managed to fit within token limits.
*   **`<extracted_data>`**: In the Response Generation task, this shows you what new information was just extracted from the user's last message, so you can acknowledge it naturally.

---

## Common Patterns and Examples

### Pattern: Multi-Step Form Filling

**User Input:** `My name is John Smith.`

**1. Data Extraction Call**
*   **Prompt:** For `collect_name` state.
*   **Your Response:**
    ```json
    {
        "extracted_data": {
            "name": "John Smith"
        },
        "confidence": 0.99,
        "reasoning": "User directly stated their name."
    }
    ```

**System Action:**
*   Updates context: `{"name": "John Smith"}`.
*   Evaluates transitions from `collect_name` state.
*   Determines a valid transition to `collect_email` exists because `name` is now in the context.
*   Changes state to `collect_email`.

**2. Response Generation Call**
*   **Prompt:** For `collect_email` state (the *new* state).
*   **Your Response:**
    ```json
    {
        "message": "Thank you, John! Now, could you please provide your email address?",
        "reasoning": "Acknowledged the newly extracted name and am now prompting for the email, as per the purpose of the 'collect_email' state."
    }
    ```

### Pattern: Ambiguous Transition

**Context:** `{ "intent": "billing_issue" }`
**State:** `support_options`
**User Input:** `This is taking too long, just get me a human.`

**1. Data Extraction Call**
*   **Your Response:**
    ```json
    {
        "extracted_data": {
            "user_frustration": "high",
            "request_human": true
        },
        "confidence": 0.9,
        "reasoning": "User expressed frustration and explicitly requested a human."
    }
    ```

**System Action:**
*   Updates context.
*   Evaluates transitions from `support_options`. Finds two valid paths: `continue_self_service` and `escalate_to_human`. This is an ambiguity.

**2. Transition Decision Call**
*   **Prompt:** Includes `<option>`s for `continue_self_service` and `escalate_to_human`.
*   **Your Response:**
    ```json
    {
        "selected_transition": "escalate_to_human",
        "reasoning": "User explicitly asked for a human agent, making this the most appropriate transition."
    }
    ```

**System Action:**
*   Changes state to `escalate_to_human`.

**3. Response Generation Call**
*   **Prompt:** For `escalate_to_human` state.
*   **Your Response:**
    ```json
    {
        "message": "I understand your frustration. I'm connecting you with a human support agent right now. Please hold on for a moment.",
        "reasoning": "Acknowledging user's frustration and informing them that their request to speak to a human is being fulfilled."
    }
    ```

---

## Best Practices Summary

1.  **Strictly Adhere to the Task**: If the prompt is for `data_extraction`, *only* extract data. If it's for `response_generation`, *only* generate a message. Do not mix responsibilities.
2.  **Always Return Valid JSON**: Your entire output must be a single, valid JSON object. No markdown, no extra text.
3.  **Use Exact State IDs**: When making a `transition_decision`, the `selected_transition` value must be an *exact, case-sensitive match* to one of the provided `<target>` options.
4.  **Be Data-Driven**: Base your decisions and responses on the information provided in the prompt (`<current_context>`, `<conversation_history>`, etc.).
5.  **Maintain Persona (in Response Generation)**: When generating messages, consistently adopt the specified `<persona>`. Do not mention technical details like states or transitions to the user.