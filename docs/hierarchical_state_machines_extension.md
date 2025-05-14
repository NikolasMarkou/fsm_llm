# Hierarchical State Machines for LLM-FSM
## Technical Specification

**Document Version:** 1.0  
**Date:** May 14, 2025  
**Status:** Proposed

## 1. Introduction

This document outlines the theoretical foundation and implementation strategy for extending the LLM-FSM framework with Hierarchical State Machine (HSM) capabilities. The proposal aims to address limitations in the current flat state structure while maintaining backward compatibility.

### 1.1 Purpose

The purpose of this extension is to:
1. Introduce modularity and reusability to conversation flows
2. Enable more complex conversation patterns without increased complexity
3. Improve state organization and maintenance
4. Allow for inherited behavior across related states

### 1.2 Scope

This specification covers:
- Theoretical foundation of HSMs
- Extended data models
- State management algorithms
- Prompt engineering considerations
- Migration strategy for existing FSMs

## 2. Theoretical Foundation

### 2.1 From FSMs to HSMs

Finite State Machines (FSMs) are formally defined as a 5-tuple:
```
M = (Q, Σ, δ, q₀, F)
```
Where:
- Q is a finite set of states
- Σ is a finite set of inputs (symbols)
- δ is the transition function (Q × Σ → Q)
- q₀ is the initial state
- F is the set of final states

Traditional FSMs create a "flat" state space where each state exists at the same level. This structure struggles with:
1. State explosion for complex systems
2. Difficulty representing common behaviors
3. Limited reusability of state patterns
4. Maintenance challenges as systems grow

Hierarchical State Machines (HSMs) extend this model by:
1. Organizing states in a tree-like structure
2. Allowing parent states to contain child states
3. Enabling inheritance of transitions from parents to children
4. Supporting entry and exit actions at multiple levels

### 2.2 Statecharts and UML Influence

The HSM concept draws heavily from David Harel's Statecharts and UML State Machines, which introduced:

1. **State Hierarchy**: States containing other states
2. **History States**: Remembering previous active substates
3. **Orthogonal Regions**: Parallel state machines
4. **Entry/Exit Actions**: Behavior triggered on state entry/exit
5. **Event Broadcasting**: Events processed at multiple levels

For LLM-FSM, we focus primarily on state hierarchy and transition inheritance, with potential for future extensions.

### 2.3 Hierarchical Structure for Conversations

In conversation design, hierarchy naturally emerges in several forms:

1. **Topical Organization**: Major conversation topics contain subtopics
2. **Information Collection**: Multi-stage collection processes with related fields
3. **Error Handling**: Specialized error states within broader conversation flows
4. **Progressive Disclosure**: Revealing conversational options as the user progresses

HSMs map directly to these patterns, making them ideal for structuring complex conversational agents.

## 3. Extended Data Model

### 3.1 Current Schema

The current LLM-FSM framework uses a flat structure:

```python
class FSMDefinition(BaseModel):
    name: str
    description: str
    states: Dict[str, State]
    initial_state: str
    version: str = "3.0"

class State(BaseModel):
    id: str
    description: str
    purpose: str
    transitions: List[Transition]
    required_context_keys: Optional[List[str]]
    instructions: Optional[str]
    example_dialogue: Optional[List[Dict[str, str]]]

class Transition(BaseModel):
    target_state: str
    description: str
    conditions: Optional[List[TransitionCondition]]
    priority: int = 100
```

### 3.2 Extended Schema for HSMs

The extended schema adds hierarchical capabilities:

```python
class FSMDefinition(BaseModel):
    name: str
    description: str
    states: Dict[str, State]
    initial_state: str
    version: str = "4.0"  # Version bump

class State(BaseModel):
    id: str
    description: str
    purpose: str
    transitions: List[Transition]
    required_context_keys: Optional[List[str]]
    instructions: Optional[str]
    example_dialogue: Optional[List[Dict[str, str]]]
    
    # New HSM fields
    sub_states: Optional[Dict[str, "State"]]
    initial_sub_state: Optional[str]
    entry_actions: Optional[List[Action]]
    exit_actions: Optional[List[Action]]
    inherit_transitions: bool = True

class Action(BaseModel):
    """Defines an action to be performed on state entry or exit."""
    type: str  # "context_update", "notification", etc.
    params: Dict[str, Any]

class StateReference(BaseModel):
    """Enables path-based state references."""
    path: str  # E.g., "support/technical/diagnostics"
```

### 3.3 Path-Based State Identification

HSMs require a path-based addressing system for states:

1. **Full Paths**: Represent the complete hierarchy (e.g., "billing/subscription/cancel")
2. **Relative Paths**: Reference states relative to current position (e.g., "../payment")
3. **Root Paths**: Reference from the top level with leading slash (e.g., "/feedback")

Path expressions will be used for:
- Target state references in transitions
- Current state tracking
- Context scoping
- History state references

## 4. Core Algorithms and Logic

### 4.1 State Resolution

Finding a state by path:

```python
def get_state_by_path(fsm_definition: FSMDefinition, path: str) -> State:
    """Resolve a state by its path."""
    parts = path.split('/')
    current = fsm_definition.states.get(parts[0])
    
    if not current:
        raise StateNotFoundError(f"State not found: {parts[0]}")
    
    # Navigate down the hierarchy
    for i in range(1, len(parts)):
        if not current.sub_states:
            raise StateNotFoundError(f"No sub-states in {'/'.join(parts[:i])}")
        
        current = current.sub_states.get(parts[i])
        if not current:
            raise StateNotFoundError(f"Sub-state not found: {parts[i]} in {'/'.join(parts[:i])}")
    
    return current
```

### 4.2 State Entry

Entering a state with hierarchy support:

```python
def enter_state(instance: FSMInstance, target_path: str) -> List[str]:
    """Enter a state and any initial sub-states, returning the complete path."""
    # Split the path
    parts = target_path.split('/')
    
    # Track the complete path including auto-entered sub-states
    complete_path = []
    current_path = ""
    
    # Enter each state in the hierarchy
    for part in parts:
        # Update current path
        if current_path:
            current_path += f"/{part}"
        else:
            current_path = part
        
        complete_path.append(current_path)
        
        # Execute entry actions for this state
        state = get_state_by_path(instance.fsm_definition, current_path)
        if state.entry_actions:
            execute_actions(instance, state.entry_actions)
    
    # Check for initial sub-state at the final level
    final_state = get_state_by_path(instance.fsm_definition, current_path)
    if final_state.sub_states and final_state.initial_sub_state:
        # Auto-enter the initial sub-state
        sub_path = f"{current_path}/{final_state.initial_sub_state}"
        complete_path.append(sub_path)
        
        # Execute entry actions for initial sub-state
        sub_state = get_state_by_path(instance.fsm_definition, sub_path)
        if sub_state.entry_actions:
            execute_actions(instance, sub_state.entry_actions)
    
    # Update instance state
    instance.current_state = complete_path[-1]
    
    return complete_path
```

### 4.3 Transition Inheritance

Computing available transitions with inheritance:

```python
def get_available_transitions(instance: FSMInstance) -> List[Transition]:
    """Get all available transitions including inherited ones."""
    path = instance.current_state
    parts = path.split('/')
    transitions = []
    
    # Start from the root and collect transitions
    current_path = ""
    for i, part in enumerate(parts):
        # Update current path
        if current_path:
            current_path += f"/{part}"
        else:
            current_path = part
            
        # Get the state at this level
        state = get_state_by_path(instance.fsm_definition, current_path)
        
        # Skip if transitions aren't inherited
        if i < len(parts) - 1 and not state.inherit_transitions:
            continue
            
        # Add transitions from this level
        transitions.extend(state.transitions)
    
    return transitions
```

### 4.4 Context Scoping

Managing context at different hierarchy levels:

```python
def get_scoped_context(instance: FSMInstance, state_path: str) -> Dict[str, Any]:
    """Get context data scoped to the given state path."""
    # Split the path
    parts = state_path.split('/')
    
    # Start with global context
    context = instance.context.data.get("global", {}).copy()
    
    # Add context from each level
    current_path = ""
    for part in parts:
        # Update current path
        if current_path:
            current_path += f"/{part}"
        else:
            current_path = part
            
        # Merge with context at this level if it exists
        level_context = instance.context.data.get(current_path, {})
        context.update(level_context)
    
    return context
```

## 5. Prompt Engineering for HSMs

### 5.1 Communicating Hierarchy to LLMs

The system prompt needs to communicate the hierarchical structure:

```
<fsm>
<current_state>billing/subscription/cancel</current_state>
<current_state_description>Process subscription cancellation</current_state_description>
<current_purpose>Handle the cancellation process for user subscriptions</current_purpose>

<hierarchy>
- <parent path="billing">Billing department handling all payment related inquiries</parent>
- <parent path="billing/subscription">Subscription management for recurring payments</parent>
- <current path="billing/subscription/cancel">Current state handling cancellation flow</current>
</hierarchy>

<inherited_transitions>
- From 'billing': transition to 'support' if technical issues mentioned
- From 'billing/subscription': transition to 'billing/payment' if payment issues mentioned
</inherited_transitions>

<available_state_transitions>
[
  {
    "target_state": "billing/subscription/cancel/confirm",
    "description": "Confirm cancellation intent",
    "priority": 1
  },
  {
    "target_state": "billing/subscription/cancel/reason",
    "description": "Collect cancellation reason",
    "priority": 2
  },
  {
    "target_state": "billing/subscription/retention",
    "description": "Attempt to retain customer",
    "priority": 3
  }
]
</available_state_transitions>
</fsm>
```

### 5.2 Enhanced PromptBuilder

The PromptBuilder class needs enhancement to handle hierarchy:

```python
def build_system_prompt(self, instance: FSMInstance) -> str:
    """Build a system prompt that includes hierarchical information."""
    state_path = instance.current_state
    current_state = get_state_by_path(instance.fsm_definition, state_path)
    
    parts = []
    # ... existing preamble ...
    
    # Add current state basics
    parts.append(f"<current_state>{state_path}</current_state>")
    parts.append(f"<current_state_description>{current_state.description}</current_state_description>")
    parts.append(f"<current_purpose>{current_state.purpose}</current_purpose>")
    
    # Add hierarchy information
    parts.append("<hierarchy>")
    path_parts = state_path.split('/')
    for i in range(1, len(path_parts)):
        parent_path = '/'.join(path_parts[:i])
        parent_state = get_state_by_path(instance.fsm_definition, parent_path)
        parts.append(f"- <parent path=\"{parent_path}\">{parent_state.description}</parent>")
    parts.append(f"- <current path=\"{state_path}\">{current_state.description}</current>")
    parts.append("</hierarchy>")
    
    # Add inherited transitions
    transitions = get_available_transitions(instance)
    
    # ... continue with existing prompt logic ...
    
    return "\n".join(parts)
```

### 5.3 Transition Handling

The LLM needs to understand how to reference states in the hierarchy:

```
<transition_instructions>
When specifying target states, you can use:
1. Absolute paths starting from the root (e.g., "/feedback")
2. Relative paths from current state:
   - Child state: "confirm" (becomes "billing/subscription/cancel/confirm")
   - Sibling state: "../retention" (becomes "billing/subscription/retention")
   - Parent state: "../../payment" (becomes "billing/payment")
3. Full paths: "billing/subscription/cancel/confirm"

Only use transitions that are valid based on the <available_state_transitions> section.
</transition_instructions>
```

## 6. Implementation Strategy

### 6.1 Phase 1: Core Data Model Extensions

1. Extend `State` class with hierarchical fields
2. Implement path-based state reference
3. Update serialization/deserialization with hierarchy support
4. Maintain backward compatibility with flat FSMs

### 6.2 Phase 2: Core Algorithms

1. Implement state resolution by path
2. Develop transition inheritance mechanism
3. Add context scoping support
4. Create entry/exit action execution

### 6.3 Phase 3: Enhanced Prompt Engineering

1. Update PromptBuilder to communicate hierarchy
2. Modify system prompts to explain path references
3. Enhance LLM response parsing for path-based transitions

### 6.4 Phase 4: Migration Tools

1. Create tooling to convert flat FSMs to hierarchical structure
2. Develop visualization for hierarchical state machines
3. Build validation tools for HSM integrity checks

### 6.5 Backward Compatibility

All existing FSMs should continue to work without modification:

```python
def is_hierarchical(fsm_definition: FSMDefinition) -> bool:
    """Determine if an FSM uses hierarchical features."""
    if fsm_definition.version < "4.0":
        return False
        
    # Check for hierarchical features in states
    for state in fsm_definition.states.values():
        if state.sub_states or state.initial_sub_state or state.entry_actions or state.exit_actions:
            return True
            
    return False

def process_message(self, instance_id: str, message: str) -> str:
    """Process a message with HSM or flat FSM handling."""
    instance = self.get_instance(instance_id)
    
    # Choose appropriate handling
    if is_hierarchical(instance.fsm_definition):
        return self._process_message_hsm(instance_id, message)
    else:
        return self._process_message_flat(instance_id, message)
```

## 7. Migration Path

### 7.1 Converting Flat FSMs to HSMs

A step-by-step process for migrating:

1. **State Analysis**: Identify logical groupings in existing states
2. **Hierarchy Design**: Create parent-child relationships
3. **Common Transition Extraction**: Move shared transitions to parent states
4. **Path Conversion**: Update state references to use paths
5. **Testing**: Validate equivalent behavior between flat and hierarchical versions

### 7.2 Migration Tool

```python
def convert_to_hsm(flat_fsm: FSMDefinition) -> FSMDefinition:
    """Convert a flat FSM to hierarchical form using pattern recognition."""
    
    # Identify state name patterns (e.g., "billing_payment", "billing_invoice")
    patterns = identify_state_groups(flat_fsm)
    
    # Create hierarchical structure
    hsm = FSMDefinition(
        name=flat_fsm.name,
        description=flat_fsm.description,
        initial_state=flat_fsm.initial_state,
        version="4.0",
        states={}
    )
    
    # Build new states with hierarchy
    for group, states in patterns.items():
        # Create parent state
        parent = State(
            id=group,
            description=f"{group.capitalize()} module",
            purpose=f"Handle {group} related operations",
            transitions=[],
            sub_states={}
        )
        
        # Find common transitions
        common_transitions = find_common_transitions(flat_fsm, states)
        parent.transitions = common_transitions
        
        # Add child states
        for state_id in states:
            original = flat_fsm.states[state_id]
            # Create child without common transitions
            child = create_child_state(original, common_transitions)
            parent.sub_states[state_id.replace(f"{group}_", "")] = child
        
        hsm.states[group] = parent
    
    # Update transition references to use paths
    update_transition_paths(hsm)
    
    return hsm
```

### 7.3 Incremental Adoption

Organizations can adopt HSMs incrementally:

1. Start with new conversation flows using HSMs
2. Convert simple existing FSMs as needed
3. Tackle complex FSMs after gaining experience
4. Develop reusable modules for common patterns

## 8. Real-World Applications

### 8.1 Customer Support System

```json
{
  "name": "Customer Support HSM",
  "description": "Hierarchical support ticket handling system",
  "initial_state": "greeting",
  "version": "4.0",
  "states": {
    "greeting": {
      "id": "greeting",
      "description": "Initial customer greeting",
      "purpose": "Welcome the customer and determine issue type",
      "transitions": [
        {
          "target_state": "technical",
          "description": "Technical issue identified",
          "conditions": [
            {
              "description": "Issue is technical in nature",
              "logic": {
                "or": [
                  {"in": ["error", {"var": "message"}]},
                  {"in": ["bug", {"var": "message"}]},
                  {"in": ["doesn't work", {"var": "message"}]}
                ]
              }
            }
          ]
        },
        {
          "target_state": "billing",
          "description": "Billing issue identified",
          "conditions": [
            {
              "description": "Issue is billing related",
              "logic": {
                "or": [
                  {"in": ["payment", {"var": "message"}]},
                  {"in": ["charge", {"var": "message"}]},
                  {"in": ["bill", {"var": "message"}]}
                ]
              }
            }
          ]
        }
      ]
    },
    "technical": {
      "id": "technical",
      "description": "Technical support module",
      "purpose": "Handle technical issues and troubleshooting",
      "transitions": [
        {
          "target_state": "escalation",
          "description": "Escalate for urgent issues",
          "conditions": [
            {
              "description": "Issue is urgent",
              "logic": {
                "or": [
                  {"in": ["urgent", {"var": "message"}]},
                  {"in": ["emergency", {"var": "message"}]}
                ]
              }
            }
          ]
        }
      ],
      "sub_states": {
        "diagnostics": {
          "id": "diagnostics",
          "description": "Technical diagnostics",
          "purpose": "Gather information about technical issues",
          "transitions": [
            {
              "target_state": "technical/troubleshooting",
              "description": "Move to troubleshooting",
              "conditions": [
                {
                  "description": "Enough diagnostic info collected",
                  "requires_context_keys": ["error_code", "device_type"]
                }
              ]
            }
          ]
        },
        "troubleshooting": {
          "id": "troubleshooting",
          "description": "Technical troubleshooting",
          "purpose": "Provide solutions to technical issues",
          "transitions": [
            {
              "target_state": "technical/resolution",
              "description": "Issue resolved"
            }
          ]
        },
        "resolution": {
          "id": "resolution",
          "description": "Technical resolution",
          "purpose": "Confirm issue resolution",
          "transitions": [
            {
              "target_state": "/feedback",
              "description": "Proceed to feedback"
            }
          ]
        }
      },
      "initial_sub_state": "diagnostics"
    },
    "billing": {
      "id": "billing",
      "description": "Billing support module",
      "purpose": "Handle billing and payment issues",
      "sub_states": {
        "verification": {
          "id": "verification",
          "description": "Account verification",
          "purpose": "Verify customer billing details",
          "transitions": [
            {
              "target_state": "billing/issue_identification",
              "description": "Move to issue identification",
              "conditions": [
                {
                  "description": "Account verified",
                  "requires_context_keys": ["account_id"]
                }
              ]
            }
          ]
        },
        "issue_identification": {
          "id": "issue_identification",
          "description": "Billing issue identification",
          "purpose": "Identify specific billing issue",
          "transitions": [
            {
              "target_state": "billing/refund",
              "description": "Refund request",
              "conditions": [
                {
                  "description": "Issue requires refund",
                  "logic": {
                    "or": [
                      {"in": ["refund", {"var": "message"}]},
                      {"in": ["money back", {"var": "message"}]}
                    ]
                  }
                }
              ]
            },
            {
              "target_state": "billing/payment_issue",
              "description": "Payment issue"
            }
          ]
        },
        "refund": {
          "id": "refund",
          "description": "Refund processing",
          "purpose": "Process customer refund request",
          "transitions": [
            {
              "target_state": "/feedback",
              "description": "Proceed to feedback"
            }
          ]
        },
        "payment_issue": {
          "id": "payment_issue",
          "description": "Payment issue resolution",
          "purpose": "Resolve payment problems",
          "transitions": [
            {
              "target_state": "/feedback",
              "description": "Proceed to feedback"
            }
          ]
        }
      },
      "initial_sub_state": "verification"
    },
    "feedback": {
      "id": "feedback",
      "description": "Customer feedback",
      "purpose": "Collect feedback about support experience",
      "transitions": []
    },
    "escalation": {
      "id": "escalation",
      "description": "Urgent escalation",
      "purpose": "Handle urgent customer issues",
      "transitions": []
    }
  }
}
```

### 8.2 E-commerce Order Flow

```json
{
  "name": "E-commerce Order HSM",
  "description": "Hierarchical order processing system",
  "initial_state": "welcome",
  // ...structure omitted for brevity...
  "states": {
    "welcome": {},
    "product_selection": {
      "sub_states": {
        "browse": {},
        "search": {},
        "details": {},
        "recommendations": {}
      }
    },
    "checkout": {
      "sub_states": {
        "cart_review": {},
        "shipping": {
          "sub_states": {
            "address": {},
            "method": {}
          }
        },
        "payment": {
          "sub_states": {
            "method": {},
            "details": {},
            "confirmation": {}
          }
        },
        "review": {}
      }
    },
    "confirmation": {},
    "support": {}
  }
}
```

## 9. Challenges and Considerations

### 9.1 Prompt Size Management

Hierarchical information increases prompt size. Mitigation strategies:

1. **Selective Hierarchy Inclusion**: Only include relevant parts of hierarchy
2. **Compression Techniques**: Use abbreviated paths where appropriate
3. **Progressive Disclosure**: Add detail only when needed for transitions
4. **Caching**: Cache prompt components to optimize generation

### 9.2 LLM Comprehension

Ensuring LLMs understand hierarchical structures:

1. **Clear Visualization**: Use indentation and symbols to show relationships
2. **Explicit Examples**: Include examples of path-based references
3. **Test Suite**: Develop comprehensive tests for hierarchy comprehension
4. **Fallback Mechanisms**: Implement recovery for path parsing errors

### 9.3 Debugging Complexity

Debugging hierarchical systems is more complex:

1. **Path Visualization**: Tools to show current position in hierarchy
2. **Transition Tracing**: Log inherited vs. direct transitions
3. **Context Visibility**: Show which context values come from which levels
4. **Step-by-Step Execution**: Tools to walk through hierarchy traversal

### 9.4 Performance Considerations

Efficiency concerns with hierarchical processing:

1. **Caching**: Cache resolved paths and available transitions
2. **Lazy Loading**: Load sub-states only when needed
3. **Tree Traversal Optimization**: Minimize redundant hierarchy walks
4. **Context Scoping Efficiency**: Optimize context merging operations

## 10. Roadmap and Future Directions

### 10.1 Implementation Phases

1. **Foundation (Q2 2025)**
   - Core data model extensions
   - Basic path resolution
   - Simple transition inheritance

2. **Core Features (Q3 2025)**
   - Complete transition inheritance
   - Context scoping
   - Entry/exit actions
   - Updated prompt engineering

3. **Advanced Features (Q4 2025)**
   - History states
   - Transition guards
   - Parallel states
   - Dynamic state loading

### 10.2 Future Extensions

1. **Orthogonal Regions**: Supporting parallel conversation flows
2. **Event Broadcasting**: Sending events to multiple levels of hierarchy  
3. **State Templates**: Parameterized state patterns for reuse
4. **Dynamic FSM Loading**: Loading FSM definitions at runtime
5. **Conversation Mining**: Generating HSMs from conversation logs

## 11. Conclusion

Hierarchical State Machines represent a significant advancement for the LLM-FSM framework, enabling more complex conversation designs with improved maintainability and reusability. While implementation requires careful consideration of prompt engineering, context management, and backward compatibility, the benefits in terms of conversation design flexibility and code organization are substantial.

This specification provides a comprehensive foundation for implementing HSMs in the LLM-FSM framework, with a clear migration path from the current flat structure and detailed technical approaches for addressing the unique challenges of hierarchical state management with LLMs.

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| State | A distinct condition in which the system can exist |
| Sub-state | A state contained within another (parent) state |
| Transition | Rule governing movement between states |
| Path | Hierarchical address of a state (e.g., "billing/refund") |
| Inheritance | Child states receiving transitions from parent states |
| Entry Action | Behavior executed when entering a state |
| Exit Action | Behavior executed when exiting a state |
| Context Scope | Visibility of context data at different hierarchy levels |

## Appendix B: Sample Migration

### Flat FSM:
```json
{
  "states": {
    "greeting": {},
    "account_verify": {},
    "account_problem": {},
    "account_update": {},
    "payment_method": {},
    "payment_issue": {},
    "payment_confirm": {}
  }
}
```

### Hierarchical FSM:
```json
{
  "states": {
    "greeting": {},
    "account": {
      "sub_states": {
        "verify": {},
        "problem": {},
        "update": {}
      }
    },
    "payment": {
      "sub_states": {
        "method": {},
        "issue": {},
        "confirm": {}
      }
    }
  }
}
```

## Appendix C: References

1. Harel, D. (1987). Statecharts: A visual formalism for complex systems. Science of computer programming, 8(3), 231-274.
2. UML 2.5.1 Specification: Finite State Machines
3. Samek, M. (2002). Practical statecharts in C/C++: Quantum programming for embedded systems.
4. Wagner, F., Schmuki, R., Wagner, T., & Wolstenholme, P. (2006). Modeling Software with Finite State Machines: A Practical Approach.