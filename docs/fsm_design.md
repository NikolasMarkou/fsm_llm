# FSM Design Guide - Best Practices for Conversation Design

This guide covers best practices for designing effective Finite State Machines for conversational AI applications.

## Table of Contents

1. [Core Design Principles](#core-design-principles)
2. [State Design Patterns](#state-design-patterns)
3. [Transition Strategies](#transition-strategies)
4. [Context Management](#context-management)
5. [Common Patterns](#common-patterns)
6. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
7. [Testing Your FSMs](#testing-your-fsms)
8. [Real-World Examples](#real-world-examples)

## Core Design Principles

### 1. Single Responsibility States

Each state should have one clear purpose:

```json
{
  "states": {
    "collect_email": {
      "id": "collect_email",
      "description": "Email collection state",
      "purpose": "Collect and validate user's email address",
      "extraction_instructions": "Extract the user's email address from their message",
      "response_instructions": "Ask the user for their email address, or confirm the one they provided",
      "required_context_keys": ["email"],
      "transitions": [{
        "target_state": "collect_phone",
        "description": "Valid email provided"
      }]
    }
  }
}
```

**Avoid:** States that try to do too much
```json
{
  "collect_all_info": {
    "description": "Collects all user information at once",
    "purpose": "Get email, phone, address, and preferences"
  }
}
```

### 2. Clear Transition Logic

Make transition conditions explicit and unambiguous:

```json
{
  "transitions": [
    {
      "target_state": "premium_support",
      "description": "User is a premium customer",
      "conditions": [{
        "description": "Customer tier is premium",
        "requires_context_keys": ["customer_tier"],
        "logic": {"==": [{"var": "customer_tier"}, "premium"]}
      }]
    },
    {
      "target_state": "standard_support",
      "description": "User is a standard customer"
    }
  ]
}
```

### 3. Graceful Error Handling

Always provide paths for error cases:

```json
{
  "states": {
    "process_payment": {
      "description": "Handles payment processing",
      "purpose": "Process the user's payment and handle success or failure",
      "transitions": [
        {
          "target_state": "payment_success",
          "description": "Payment processed successfully"
        },
        {
          "target_state": "payment_retry",
          "description": "Payment failed but can retry"
        },
        {
          "target_state": "payment_help",
          "description": "Payment failed, needs assistance"
        }
      ]
    }
  }
}
```

### 4. Natural Language Cues

Write purposes that guide natural conversation:

```json
{
  "description": "Returning customer greeting",
  "purpose": "Warmly greet the returning customer by name and ask how we can help them today",
  "response_instructions": "Use the customer's name and offer a warm, personalized welcome",
  "required_context_keys": ["customer_name"]
}
```

## State Design Patterns

### Pattern 1: The Gatekeeper

Use for authentication or validation:

```json
{
  "states": {
    "verify_identity": {
      "description": "Identity verification gate",
      "purpose": "Verify user identity with security questions",
      "required_context_keys": ["account_number", "security_answer"],
      "transitions": [
        {
          "target_state": "authenticated",
          "description": "Identity verified"
        },
        {
          "target_state": "authentication_failed",
          "description": "Failed verification"
        }
      ]
    }
  }
}
```

### Pattern 2: The Collector

For gathering multiple pieces of information:

```json
{
  "states": {
    "shipping_address": {
      "description": "Shipping address collection",
      "purpose": "Collect complete shipping address",
      "extraction_instructions": "Extract street, city, state, and zip code from the user's message",
      "response_instructions": "Ask for missing address fields or confirm the complete address",
      "required_context_keys": ["street", "city", "state", "zip"],
      "transitions": [{
        "target_state": "confirm_address",
        "description": "All address fields collected"
      }]
    }
  }
}
```

### Pattern 3: The Router

For directing to specialized flows:

```json
{
  "states": {
    "issue_classifier": {
      "description": "Routes user issues to the appropriate support flow",
      "purpose": "Understand the type of issue and route appropriately",
      "transitions": [
        {
          "target_state": "technical_flow",
          "description": "Technical issue identified"
        },
        {
          "target_state": "billing_flow",
          "description": "Billing issue identified"
        },
        {
          "target_state": "general_inquiry",
          "description": "General question"
        }
      ]
    }
  }
}
```

### Pattern 4: The Confirmer

For validating understanding:

```json
{
  "states": {
    "confirm_order": {
      "description": "Order confirmation step",
      "purpose": "Summarize order details and get final confirmation",
      "transitions": [
        {
          "target_state": "process_order",
          "description": "Order confirmed"
        },
        {
          "target_state": "modify_order",
          "description": "User wants to make changes"
        }
      ]
    }
  }
}
```

## Transition Strategies

### 1. Explicit Transitions

Best for predictable flows:

```json
{
  "transitions": [
    {
      "target_state": "next_step",
      "description": "User completed current task"
    }
  ]
}
```

### 2. Conditional Transitions

For dynamic routing based on context:

```json
{
  "transitions": [
    {
      "target_state": "vip_service",
      "description": "High-value customer gets VIP service",
      "conditions": [{
        "description": "Customer lifetime value exceeds 1000",
        "logic": {">": [{"var": "lifetime_value"}, 1000]}
      }]
    },
    {
      "target_state": "standard_service",
      "description": "Default path"
    }
  ]
}
```

### 3. Multi-Path Transitions

For user choice:

```json
{
  "description": "Options or checkout decision point",
  "purpose": "Ask if they want to see more options or proceed to checkout",
  "transitions": [
    {
      "target_state": "show_more",
      "description": "User wants more options"
    },
    {
      "target_state": "checkout",
      "description": "User ready to purchase"
    }
  ]
}
```

### Transition Field Reference

**`evaluation_priority`** on `TransitionCondition`: Controls the order in which conditions are evaluated. Integer value (default `100`, range 0--1000). Lower values are evaluated earlier. Useful when one condition should short-circuit others:

```json
{
  "conditions": [
    {
      "description": "Account is suspended",
      "logic": {"==": [{"var": "account_status"}, "suspended"]},
      "evaluation_priority": 10
    },
    {
      "description": "Account is overdue",
      "logic": {"==": [{"var": "payment_status"}, "overdue"]},
      "evaluation_priority": 50
    }
  ]
}
```

**`llm_description`** on `Transition`: Optional string (max 300 characters) that customizes how a transition is presented to the LLM during ambiguous transition decisions. When omitted, the regular `description` field is used instead. Useful when you want the LLM to see more detailed guidance than the human-readable description:

```json
{
  "target_state": "escalate",
  "description": "Escalate to specialist",
  "llm_description": "Choose this transition when the user's issue requires domain-specific expertise that a general agent cannot provide, such as legal, medical, or financial questions"
}
```

## Context Management

### 1. Required vs Optional Data

Be explicit about what's required:

```json
{
  "states": {
    "user_profile": {
      "description": "User profile data collection",
      "purpose": "Collect basic user profile information",
      "required_context_keys": ["name", "email"],
      "extraction_instructions": "Extract name, email, and phone number if offered. Store phone in 'phone'",
      "response_instructions": "Ask for missing profile information. If phone is offered, acknowledge it"
    }
  }
}
```

### 2. Context Namespacing

Organize complex context:

```json
{
  "required_context_keys": [
    "user.name",
    "user.email",
    "order.items",
    "order.total"
  ]
}
```

### 3. Progressive Context Building

Collect information gradually:

```json
{
  "states": {
    "basic_info": {
      "description": "Collects the user's basic information",
      "purpose": "Get the user's name",
      "required_context_keys": ["name"],
      "transitions": [{"target_state": "preferences"}]
    },
    "preferences": {
      "description": "Collects user preferences",
      "purpose": "Discover the user's favorite category",
      "required_context_keys": ["favorite_category"],
      "transitions": [{"target_state": "recommendations"}]
    }
  }
}
```

## Common Patterns

### 1. The Onboarding Flow

```json
{
  "name": "user_onboarding",
  "initial_state": "welcome",
  "states": {
    "welcome": {
      "description": "Initial welcome state for new users",
      "purpose": "Welcome new user and explain the process",
      "transitions": [{"target_state": "account_type"}]
    },
    "account_type": {
      "description": "Account type selection",
      "purpose": "Help user choose between personal and business account",
      "transitions": [
        {"target_state": "personal_setup", "description": "Personal account"},
        {"target_state": "business_setup", "description": "Business account"}
      ]
    },
    "personal_setup": {
      "description": "Personal account setup",
      "purpose": "Collect personal account details",
      "required_context_keys": ["name", "email"],
      "transitions": [{"target_state": "preferences"}]
    },
    "business_setup": {
      "description": "Business account setup",
      "purpose": "Collect business account details",
      "required_context_keys": ["company_name", "email", "role"],
      "transitions": [{"target_state": "preferences"}]
    },
    "preferences": {
      "description": "User preference collection",
      "purpose": "Collect user preferences for personalization",
      "transitions": [{"target_state": "complete"}]
    },
    "complete": {
      "description": "Onboarding completion",
      "purpose": "Confirm account creation and provide next steps",
      "transitions": []
    }
  }
}
```

### 2. The Support Escalation

```json
{
  "states": {
    "self_service": {
      "description": "Self-service support resources",
      "purpose": "Provide self-help resources",
      "transitions": [
        {"target_state": "resolved", "description": "Issue resolved"},
        {"target_state": "human_agent", "description": "Needs human help"}
      ]
    },
    "human_agent": {
      "description": "Human agent handoff preparation",
      "purpose": "Collect info for agent handoff",
      "required_context_keys": ["contact_method", "availability"],
      "transitions": [{"target_state": "queued"}]
    }
  }
}
```

### 3. The Feedback Loop

```json
{
  "states": {
    "service_complete": {
      "description": "Service interaction completed",
      "purpose": "Wrap up the service interaction",
      "transitions": [{"target_state": "satisfaction_check"}]
    },
    "satisfaction_check": {
      "description": "Customer satisfaction survey",
      "purpose": "Ask if the service met their needs",
      "transitions": [
        {"target_state": "thank_you", "description": "Satisfied"},
        {"target_state": "improvement", "description": "Not satisfied"}
      ]
    },
    "improvement": {
      "description": "Collects improvement feedback",
      "purpose": "Understand how we can do better",
      "required_context_keys": ["feedback"],
      "transitions": [{"target_state": "thank_you"}]
    }
  }
}
```

## Anti-Patterns to Avoid

### Avoid: The Megazord State
```json
{
  "do_everything": {
    "description": "Handles all tasks in a single state",
    "purpose": "Handle login, get info, process payment, ship order"
  }
}
```

### Avoid: The Dead End
```json
{
  "error_state": {
    "description": "Generic error state",
    "purpose": "Something went wrong",
    "transitions": []  // No way out!
  }
}
```

### Avoid: The Infinite Loop
```json
{
  "ask_question": {
    "description": "Asks the user a question",
    "purpose": "Get an answer from the user",
    "transitions": [{
      "target_state": "ask_question",  // Loops forever
      "description": "Invalid answer"
    }]
  }
}
```

### Avoid: The Context Blackhole
```json
{
  "collect_everything": {
    "description": "Collects all user data at once",
    "purpose": "Gather every piece of user information in one step",
    "required_context_keys": [
      "name", "email", "phone", "address", "ssn", 
      "mothers_maiden_name", "first_pet", "favorite_color"
      // Too much at once!
    ]
  }
}
```

## Testing Your FSMs

### 1. Path Coverage

Ensure all paths are reachable:

```python
from fsm_llm import validate_fsm_from_file

result = validate_fsm_from_file("my_fsm.json")
print(result)  # Shows unreachable states, missing transitions
```

### 2. Context Flow Testing

```python
# Test that required data is collected
test_cases = [
    {"input": "John Doe", "expected_context": {"name": "John Doe"}},
    {"input": "john@example.com", "expected_context": {"email": "john@example.com"}}
]

for test in test_cases:
    response = api.converse(test["input"], conv_id)
    assert api.get_data(conv_id) == test["expected_context"]
```

### 3. Edge Case Handling

Test unusual inputs:
- Empty responses
- Very long responses
- Off-topic responses
- Malformed data

## Real-World Examples

### E-Commerce Checkout

```json
{
  "name": "checkout_flow",
  "initial_state": "cart_review",
  "persona": "You are a helpful shopping assistant guiding through checkout",
  "states": {
    "cart_review": {
      "description": "Cart review before checkout",
      "purpose": "Show cart contents and confirm ready to checkout",
      "transitions": [
        {"target_state": "shipping", "description": "Proceed to checkout"},
        {"target_state": "cart_modify", "description": "Want to change cart"}
      ]
    },
    "shipping": {
      "description": "Shipping address collection",
      "purpose": "Collect shipping information",
      "required_context_keys": ["shipping_address"],
      "transitions": [{"target_state": "shipping_method"}]
    },
    "shipping_method": {
      "description": "Shipping method selection",
      "purpose": "Choose shipping speed and see costs",
      "required_context_keys": ["shipping_choice"],
      "transitions": [{"target_state": "payment"}]
    },
    "payment": {
      "description": "Payment information collection",
      "purpose": "Collect payment information securely",
      "required_context_keys": ["payment_method"],
      "transitions": [{"target_state": "review_order"}]
    },
    "review_order": {
      "description": "Final order review before submission",
      "purpose": "Show complete order for final review",
      "transitions": [
        {"target_state": "process_order", "description": "Confirm order"},
        {"target_state": "shipping", "description": "Change shipping"},
        {"target_state": "payment", "description": "Change payment"}
      ]
    },
    "process_order": {
      "description": "Order processing and payment execution",
      "purpose": "Process payment and create order",
      "transitions": [
        {"target_state": "order_success", "description": "Payment successful"},
        {"target_state": "payment_failed", "description": "Payment failed"}
      ]
    }
  }
}
```

### Healthcare Triage

```json
{
  "name": "symptom_checker",
  "initial_state": "welcome",
  "persona": "You are a caring medical assistant. Be empathetic but clear you're not replacing professional medical advice",
  "states": {
    "welcome": {
      "description": "Symptom checker introduction and consent",
      "purpose": "Explain the symptom checker and get consent to proceed",
      "transitions": [
        {"target_state": "primary_symptom", "description": "User consents"},
        {"target_state": "declined", "description": "User declines"}
      ]
    },
    "primary_symptom": {
      "description": "Primary symptom identification",
      "purpose": "Understand their main symptom or concern",
      "required_context_keys": ["main_symptom"],
      "transitions": [
        {"target_state": "emergency_check", "description": "Symptom identified"}
      ]
    },
    "emergency_check": {
      "description": "Emergency symptom screening",
      "purpose": "Check for emergency symptoms requiring immediate care",
      "transitions": [
        {"target_state": "emergency_referral", "description": "Emergency symptoms present"},
        {"target_state": "symptom_details", "description": "No emergency symptoms"}
      ]
    },
    "emergency_referral": {
      "description": "Emergency medical referral",
      "purpose": "Strongly advise seeking immediate medical attention",
      "transitions": []
    },
    "symptom_details": {
      "description": "Detailed symptom information gathering",
      "purpose": "Gather details about duration, severity, and related symptoms",
      "required_context_keys": ["duration", "severity", "other_symptoms"],
      "transitions": [{"target_state": "recommendation"}]
    },
    "recommendation": {
      "description": "Care recommendation based on symptom analysis",
      "purpose": "Provide appropriate care recommendations based on symptoms",
      "transitions": []
    }
  }
}
```

## Summary

Good FSM design is about:
1. **Clarity** - Each state has one job
2. **Flow** - Natural conversation progression
3. **Flexibility** - Handle edge cases gracefully
4. **Testability** - Predictable, verifiable behavior

Remember: The FSM provides structure, the LLM provides intelligence. Design your FSMs to leverage both strengths.

---

**Next:** Learn about [Handler Development](./handlers.md) to add custom logic to your FSMs.