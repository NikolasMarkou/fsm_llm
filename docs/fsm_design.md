# FSM Design Guide

Best practices for designing effective Finite State Machines for conversational AI.

## Core Principles

### 1. Single Responsibility States

Each state should have one clear purpose:

```json
{
  "collect_email": {
    "id": "collect_email",
    "purpose": "Collect and validate user's email address",
    "extraction_instructions": "Extract the user's email address",
    "response_instructions": "Ask for email, or confirm the one provided",
    "required_context_keys": ["email"],
    "transitions": [{"target_state": "collect_phone", "description": "Valid email provided"}]
  }
}
```

**Avoid** states that try to collect everything at once.

### 2. Clear Transition Logic

Make conditions explicit with JsonLogic:

```json
{
  "transitions": [
    {
      "target_state": "vip_service",
      "description": "High-value customer",
      "conditions": [{"logic": {">": [{"var": "lifetime_value"}, 1000]}}]
    },
    {"target_state": "standard_service", "description": "Default path"}
  ]
}
```

### 3. Graceful Error Handling

Always provide paths for error cases -- avoid dead-end states with no transitions.

### 4. Natural Language Cues

Write purposes that guide natural conversation:

```json
{
  "purpose": "Warmly greet the returning customer by name and ask how we can help",
  "response_instructions": "Use the customer's name and offer a personalized welcome",
  "required_context_keys": ["customer_name"]
}
```

## Design Patterns

### The Gatekeeper -- Authentication/Validation

```json
{
  "verify_identity": {
    "purpose": "Verify user identity with security questions",
    "required_context_keys": ["account_number", "security_answer"],
    "transitions": [
      {"target_state": "authenticated", "description": "Identity verified"},
      {"target_state": "auth_failed", "description": "Failed verification"}
    ]
  }
}
```

### The Collector -- Gathering Information

```json
{
  "shipping_address": {
    "purpose": "Collect complete shipping address",
    "extraction_instructions": "Extract street, city, state, and zip code",
    "required_context_keys": ["street", "city", "state", "zip"],
    "transitions": [{"target_state": "confirm_address", "description": "All fields collected"}]
  }
}
```

### The Router -- Directing to Specialized Flows

```json
{
  "issue_classifier": {
    "purpose": "Understand the issue type and route appropriately",
    "transitions": [
      {"target_state": "technical_flow", "description": "Technical issue"},
      {"target_state": "billing_flow", "description": "Billing issue"},
      {"target_state": "general_inquiry", "description": "General question"}
    ]
  }
}
```

### The Confirmer -- Validating Understanding

```json
{
  "confirm_order": {
    "purpose": "Summarize order details and get final confirmation",
    "transitions": [
      {"target_state": "process_order", "description": "Confirmed"},
      {"target_state": "modify_order", "description": "Wants changes"}
    ]
  }
}
```

## Transition Field Reference

**`evaluation_priority`** on `TransitionCondition`: Controls evaluation order (default 100, range 0-1000). Lower values evaluated first -- useful for short-circuiting:

```json
{"conditions": [
  {"description": "Account suspended", "logic": {"==": [{"var": "account_status"}, "suspended"]}, "evaluation_priority": 10},
  {"description": "Account overdue", "logic": {"==": [{"var": "payment_status"}, "overdue"]}, "evaluation_priority": 50}
]}
```

**`llm_description`** on `Transition`: Optional string (max 300 chars) that customizes how a transition is presented to the LLM during ambiguous decisions. When omitted, `description` is used.

## Context Management

**Required vs optional**: Use `required_context_keys` for mandatory data -- the state won't transition until all keys are present.

**Progressive building**: Collect information gradually across states rather than all at once.

**Context scope**: Use `context_scope` to restrict which keys a state can read/write.

## FSM Stacking Patterns

FSM stacking (`push_fsm`/`pop_fsm`) enables modular, composable design.

### When to Use

- **Shared sub-flows** -- A "collect address" FSM reused by checkout, returns, account setup
- **Progressive detail** -- High-level routing FSM pushes specialized FSMs per topic
- **Reasoning integration** -- The reasoning engine uses stacking internally

### Guidelines

1. Keep child FSMs self-contained -- they should work independently
2. Define clear context contracts via `context_to_pass` and `shared_context_keys`
3. Design terminal states (no outgoing transitions) to trigger `pop_fsm`
4. Choose merge strategies: `"update"` (overwrite parent) or `"preserve"` (only add new keys)

```python
api.push_fsm(conv_id, "address_form.json",
    context_to_pass={"flow": "checkout"},
    shared_context_keys=["user_id"])

api.pop_fsm(conv_id,
    context_to_return={"address_complete": True},
    merge_strategy="update")
```

## Classification-Based Routing

Classification is built into the core (`fsm_llm.Classifier`). Use it with `classification_extractions` on states for automatic intent-based routing, or use a lightweight classifier layer that pushes the appropriate FSM via stacking.

## Designing for Agents

Agent patterns (`fsm_llm_agents`) auto-generate FSMs from tool registries. The core ReAct loop is a 3-4 state FSM: **Think -> Act -> Observe -> Conclude**. Tool execution happens via handlers, not state instructions.

## Anti-Patterns to Avoid

| Anti-Pattern | Problem | Fix |
|-------------|---------|-----|
| Megazord State | One state handles everything | Split into single-responsibility states |
| Dead End | Error state with no transitions out | Add recovery/retry transitions |
| Infinite Loop | State transitions back to itself | Add max retry logic or a fallback transition |
| Context Blackhole | Too many `required_context_keys` at once | Collect progressively across states |

## Testing FSMs

```python
# Validate structure
from fsm_llm import validate_fsm_from_file
result = validate_fsm_from_file("my_fsm.json")  # checks reachability, missing transitions

# Or from CLI
fsm-llm validate --fsm my_fsm.json   # or legacy alias: fsm-llm-validate
```

Test all paths, context flow, and edge cases (empty input, very long input, off-topic).

---

**Next:** [Handler Development](./handlers.md)
