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
      "purpose": "Collect and validate user's email address",
      "required_context_keys": ["email"],
      "transitions": [{
        "target_state": "collect_phone",
        "description": "Valid email provided"
      }]
    }
  }
}
```

**❌ Avoid:** States that try to do too much
```json
{
  "collect_all_info": {
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
  "purpose": "Warmly greet the returning customer by name and ask how we can help them today",
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
      "purpose": "Collect complete shipping address",
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
      "conditions": [{
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

## Context Management

### 1. Required vs Optional Data

Be explicit about what's required:

```json
{
  "states": {
    "user_profile": {
      "required_context_keys": ["name", "email"],
      "instructions": "Also try to collect phone number if offered, store in 'phone'"
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
      "required_context_keys": ["name"],
      "transitions": [{"target_state": "preferences"}]
    },
    "preferences": {
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
      "purpose": "Welcome new user and explain the process",
      "transitions": [{"target_state": "account_type"}]
    },
    "account_type": {
      "purpose": "Help user choose between personal and business account",
      "transitions": [
        {"target_state": "personal_setup", "description": "Personal account"},
        {"target_state": "business_setup", "description": "Business account"}
      ]
    },
    "personal_setup": {
      "required_context_keys": ["name", "email"],
      "transitions": [{"target_state": "preferences"}]
    },
    "business_setup": {
      "required_context_keys": ["company_name", "email", "role"],
      "transitions": [{"target_state": "preferences"}]
    },
    "preferences": {
      "purpose": "Collect user preferences for personalization",
      "transitions": [{"target_state": "complete"}]
    },
    "complete": {
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
      "purpose": "Provide self-help resources",
      "transitions": [
        {"target_state": "resolved", "description": "Issue resolved"},
        {"target_state": "human_agent", "description": "Needs human help"}
      ]
    },
    "human_agent": {
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
      "transitions": [{"target_state": "satisfaction_check"}]
    },
    "satisfaction_check": {
      "purpose": "Ask if the service met their needs",
      "transitions": [
        {"target_state": "thank_you", "description": "Satisfied"},
        {"target_state": "improvement", "description": "Not satisfied"}
      ]
    },
    "improvement": {
      "purpose": "Understand how we can do better",
      "required_context_keys": ["feedback"],
      "transitions": [{"target_state": "thank_you"}]
    }
  }
}
```

## Anti-Patterns to Avoid

### ❌ The Megazord State
```json
{
  "do_everything": {
    "purpose": "Handle login, get info, process payment, ship order"
  }
}
```

### ❌ The Dead End
```json
{
  "error_state": {
    "purpose": "Something went wrong",
    "transitions": []  // No way out!
  }
}
```

### ❌ The Infinite Loop
```json
{
  "ask_question": {
    "transitions": [{
      "target_state": "ask_question",  // Loops forever
      "description": "Invalid answer"
    }]
  }
}
```

### ❌ The Context Blackhole
```json
{
  "collect_everything": {
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
from llm_fsm import validate_fsm_from_file

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
      "purpose": "Show cart contents and confirm ready to checkout",
      "transitions": [
        {"target_state": "shipping", "description": "Proceed to checkout"},
        {"target_state": "cart_modify", "description": "Want to change cart"}
      ]
    },
    "shipping": {
      "purpose": "Collect shipping information",
      "required_context_keys": ["shipping_address"],
      "transitions": [{"target_state": "shipping_method"}]
    },
    "shipping_method": {
      "purpose": "Choose shipping speed and see costs",
      "required_context_keys": ["shipping_choice"],
      "transitions": [{"target_state": "payment"}]
    },
    "payment": {
      "purpose": "Collect payment information securely",
      "required_context_keys": ["payment_method"],
      "transitions": [{"target_state": "review_order"}]
    },
    "review_order": {
      "purpose": "Show complete order for final review",
      "transitions": [
        {"target_state": "process_order", "description": "Confirm order"},
        {"target_state": "shipping", "description": "Change shipping"},
        {"target_state": "payment", "description": "Change payment"}
      ]
    },
    "process_order": {
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
      "purpose": "Explain the symptom checker and get consent to proceed",
      "transitions": [
        {"target_state": "primary_symptom", "description": "User consents"},
        {"target_state": "declined", "description": "User declines"}
      ]
    },
    "primary_symptom": {
      "purpose": "Understand their main symptom or concern",
      "required_context_keys": ["main_symptom"],
      "transitions": [
        {"target_state": "emergency_check", "description": "Symptom identified"}
      ]
    },
    "emergency_check": {
      "purpose": "Check for emergency symptoms requiring immediate care",
      "transitions": [
        {"target_state": "emergency_referral", "description": "Emergency symptoms present"},
        {"target_state": "symptom_details", "description": "No emergency symptoms"}
      ]
    },
    "emergency_referral": {
      "purpose": "Strongly advise seeking immediate medical attention",
      "transitions": []
    },
    "symptom_details": {
      "purpose": "Gather details about duration, severity, and related symptoms",
      "required_context_keys": ["duration", "severity", "other_symptoms"],
      "transitions": [{"target_state": "recommendation"}]
    },
    "recommendation": {
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

**Next:** Learn about [Handler Development](./handler_development.md) to add custom logic to your FSMs.