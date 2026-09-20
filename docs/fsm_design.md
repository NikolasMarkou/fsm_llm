# FSM Design Guide

> Covers FSM-LLM v0.5.0

Best practices for designing effective Finite State Machines for conversational AI.

## Core Principles

### 1. Single Responsibility States

Each state should have one clear purpose:

```json
{
  "collect_email": {
    "id": "collect_email",
    "description": "Collect the user's email address",
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
      "conditions": [{
        "description": "Lifetime value above 1000",
        "requires_context_keys": ["lifetime_value"],
        "logic": {">": [{"var": "lifetime_value"}, 1000]}
      }]
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

With no `conditions`, both transitions are ambiguous and the LLM picks between them from the descriptions. To make "Identity verified" deterministic, give it a condition with `requires_context_keys` and a `logic`.

### The Collector -- Gathering Information

```json
{
  "shipping_address": {
    "purpose": "Collect complete shipping address",
    "extraction_instructions": "Extract street, city, state, and zip code",
    "required_context_keys": ["street", "city", "state", "zip"],
    "transitions": [{
      "target_state": "confirm_address",
      "description": "All fields collected",
      "conditions": [{
        "description": "Every address field is present",
        "requires_context_keys": ["street", "city", "state", "zip"],
        "logic": {"and": [
          {"has_context": "street"}, {"has_context": "city"},
          {"has_context": "state"}, {"has_context": "zip"}
        ]}
      }]
    }]
  }
}
```

The condition is what holds the state until the address is complete; `required_context_keys` alone only asks the pipeline to extract those keys.

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

**Required vs optional**: `required_context_keys` lists the data a state should extract; it does **not** block a transition (an unconditional transition fires even when a listed key is missing). To hold a state until data exists, give the transition a condition with `requires_context_keys` and a `logic` such as `{"has_context": "email"}`; `fsm-llm-validate` warns about a required key no condition gates on.

**Progressive building**: Collect information gradually across states rather than all at once.

**Context scope**: `context_scope.read_keys` restricts which context keys a state's prompts see. `write_keys` is advisory: it is not enforced and not validated.

**Gate keys the user must not set**: a key a transition condition reads (`is_admin`, `is_verified`) can be written by a steered extraction unless you list it in the FSM-level `handler_only_keys`. Listed keys are dropped from every LLM extraction channel and stay writable by handlers, `update_context` and `initial_context`. It is opt-in and covers only the listed keys (an unlisted gate key, or one owned by a `classification_extractions` entry, is not protected).

**Corrections**: a later-turn value the LLM returns for an already-set key replaces it only when the pipeline extracted that key itself (handler-set and `update_context` values are never overwritten). A refused correction leaves the stored value in place, and the Pass-2 prompt then carries a `<rejected_corrections>` block so the reply does not claim the change was made. A handler that normalises an extracted value on `CONTEXT_UPDATE` (`blue` -> `BLUE`) counts as a handler edit and turns later corrections of that key off (the stored value stands and the reply is told so).

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
4. Choose merge strategies: `"update"` (overwrite parent) or `"preserve"` (only add new keys). Only keys named in `shared_context_keys` (or given as `return_context` / `context_to_return`) come back, whatever the strategy; a child's other extracted data is dropped on pop.

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

For agent-style flows that accumulate intermediate results across turns, `WorkingMemory` (`fsm_llm.WorkingMemory`) provides named buffers (`core`, `scratch`, `environment`, `reasoning`) so tool outputs and intermediate reasoning are organized rather than flattened into one context dict. Intermediate states can also set an empty `response_instructions` to skip the Pass-2 response LLM call entirely (see the 2-pass notes), which is the common pattern for tool-dispatch states inside a ReAct loop.

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
fsm-llm-validate --fsm my_fsm.json
```

Test all paths, context flow, and edge cases (empty input, very long input, off-topic).

---

**Next:** [Handler Development](./handlers.md)
