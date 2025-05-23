# LLM-FSM: Adding State to the Stateless

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

![logo](./images/fsm-llm-logo-1.png)

## Table of Contents
- [The Problem: Stateless LLMs in Structured Conversations](#the-problem-stateless-llms-in-structured-conversations)
- [The Solution: Finite State Machines + LLMs](#the-solution-finite-state-machines--llms)
  - [Key Features](#key-features)
- [Theoretical Foundation](#theoretical-foundation)
  - [The Nature of Finite State Machines](#the-nature-of-finite-state-machines)
  - [The Theoretical Synthesis](#the-theoretical-synthesis)
- [Example Conversation](#example-conversation)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Python API](#python-api)
  - [Simplified API](#simplified-api)
  - [Starting Conversations with Initial Context](#starting-conversations-with-initial-context)
  - [Command Line Interface](#command-line-interface)
- [Core Architecture](#core-architecture)
  - [FSM Definition](#fsm-definition)
  - [JsonLogic Expressions for Powerful Transition Conditions](#jsonlogic-expressions-for-powerful-transition-conditions)
  - [The Execution Flow](#the-execution-flow)
  - [LLM Response Format](#llm-response-format)
- [Conversation Patterns](#conversation-patterns)
  - [1. Linear Flows](#1-linear-flows)
  - [2. Conversational Loops](#2-conversational-loops)
  - [3. Decision Trees](#3-decision-trees)
  - [4. Hybrid Patterns](#4-hybrid-patterns)
- [Key Design Decisions](#key-design-decisions)
  - [1. LLM as the NLU Engine](#1-llm-as-the-nlu-engine)
  - [2. Context Outside States](#2-context-outside-states)
  - [3. Expressive Transition Conditions with JsonLogic](#3-expressive-transition-conditions-with-jsonlogic)
- [Examples](#examples)
  - [Simple Linear Flow: Form Filling](#simple-linear-flow-form-filling)
  - [Decision Tree: Product Recommendation](#decision-tree-product-recommendation)
  - [Interactive Loop: Book Recommendation](#interactive-loop-book-recommendation)
  - [Advanced Examples](#advanced-examples)
- [Persona Support](#persona-support)
- [License](#license)

## The Problem: Stateless LLMs in Structured Conversations

Large Language Models have revolutionized natural language processing with their remarkable generation capabilities. 
However, they have a fundamental limitation: **they are inherently stateless**. 
Each interaction is processed independently with only the context provided in the prompt.

This statelessness creates significant challenges for building robust conversational applications:

- **State Fragility**: Without explicit tracking, conversations easily lose their place
- **Context Limitations**: As conversations grow, context windows fill up quickly
- **Transition Ambiguity**: Determining when to move to different conversation stages is difficult
- **Information Extraction Inconsistency**: Extracting structured data from free-form text is unreliable
- **Validation Challenges**: Ensuring required information is collected before proceeding is complex

Consider a flight booking scenario:

```
User: I'd like to book a flight
System: Where would you like to fly to?
User: I'm thinking maybe Hawaii
System: Great choice! And where will you be departing from?
User: Actually, I'd prefer Bali instead of Hawaii
```

Without explicit state tracking, the system might miss the change in destination or maintain inconsistent information.

## The Solution: Finite State Machines + LLMs

LLM-FSM elegantly combines classical Finite State Machines with modern Large Language Models:

> "We keep the state as a JSON structure inside the system prompt of an LLM, describing transition nodes and conditions for that specific state, along with any emittance of symbols that the LLM might do."

This hybrid approach gives you the best of both worlds:
- ✅ **Predictable conversation flows** with clear rules and transitions
- ✅ **Natural language understanding** powered by state-of-the-art LLMs
- ✅ **Persistent context** across the entire conversation
- ✅ **Dynamic adaptation** to user inputs
- ✅ **Expressive Logic** for complex transitional decision-making

### Key Features

- 🚦 **Structured Conversation Flows**: Define states, transitions, and conditions
- 🧠 **LLM-Powered Entity Extraction**: Let the LLM do what it does best
- 👤 **Persona Support**: Define a consistent tone and style for responses
- 📝 **Persistent Context Management**: Maintain information throughout the conversation
- 🔄 **Provider-Agnostic**: Works with OpenAI, Anthropic, and other LLM providers via LiteLLM
- 📊 **Visualization**: See your FSM structure with a built-in ASCII visualizer
- 📝 **Validation & Error Handling**: Catch and handle issues gracefully
- 🪵 **Comprehensive Logging**: Detailed logs for debugging and monitoring
- 🧪 **Test-Friendly**: Easy to unit test and verify behavior
- 🧮 **JsonLogic Expressions**: Powerful conditional logic for transitions

## Theoretical Foundation

### The Nature of Finite State Machines

Finite State Machines (FSMs) represent one of computer science's most elegant and powerful abstractions. Dating back to the theoretical work of mathematicians like Alan Turing and John von Neumann, FSMs provide a formal model for systems that can exist in exactly one of a finite number of states at any given time.

The core components of an FSM are remarkably simple:

- **States**: The distinct conditions in which a system can exist
- **Transitions**: Rules governing movement between states based on inputs
- **Initial State**: Where the system begins execution
- **Terminal States**: Where the system concludes execution (optional)

### The Theoretical Synthesis

The LLM-FSM framework represents a theoretical bridge between two fundamentally different computational paradigms:

1. **Deterministic Computation** (FSMs): Rule-based, predictable, and structured
2. **Probabilistic Computation** (LLMs): Adaptable, nuanced, and natural

This synthesis creates a hybrid system where:
- The FSM provides the "skeleton" that ensures logical conversation flow
- The LLM provides the "muscles and skin" that handle natural language understanding and production

## Example Conversation

Here's a sample conversation using a personal information collection FSM:

```
System: Hello! I'm an assistant that will be collecting some information. 
Could you please tell me your name?

User: I'm Alex Thompson

System: Nice to meet you, Alex Thompson! Could you please provide your email address?

User: alex.thompson@example.com

System: Thank you! Could you please tell me your date of birth?

User: March 15, 1992

System: Great! I've collected:
- Name: Alex Thompson
- Email: alex.thompson@example.com
- Date of Birth: March 15, 1992
Is this information correct?

User: Actually, my email is alex.j.thompson@example.com

System: I've updated your email to alex.j.thompson@example.com. Is the information correct now?

User: Yes, that's correct

System: Thank you for confirming your information! Your profile has been saved.
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/llm-fsm.git
cd llm-fsm

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

Or install via pip:

```bash
pip install llm-fsm
```

## Quick Start

### Python API

```python
from llm_fsm.llm import LiteLLMInterface
from llm_fsm.fsm import FSMManager
from llm_fsm.utilities import load_fsm_definition

# Initialize the LLM interface
llm_interface = LiteLLMInterface(
   model="gpt-4o",  # Use any model supported by LiteLLM
   api_key="your-api-key",
   temperature=0.5
)

# Create an FSM manager
fsm_manager = FSMManager(
   fsm_loader=load_fsm_definition,
   llm_interface=llm_interface
)

# Start a conversation with your FSM
conversation_id, response = fsm_manager.start_conversation("examples/personal_information_collection.json")
print(f"System: {response}")

# Process user input
user_input = "My name is Alex Thompson"
response = fsm_manager.process_message(conversation_id, user_input)
print(f"System: {response}")

# Continue the conversation until completion
while not fsm_manager.is_conversation_ended(conversation_id):
   user_input = input("You: ")
   response = fsm_manager.process_message(conversation_id, user_input)
   print(f"System: {response}")

# Get the collected data when done
user_data = fsm_manager.get_conversation_data(conversation_id)
print(f"Collected data: {user_data}")
```

### Simplified API

For an even easier approach, you can use the streamlined `LLM_FSM` class:

```python
from llm_fsm import LLM_FSM

# Create the LLM-FSM instance
fsm = LLM_FSM.from_file(
    path="examples/personal_information_collection.json",
    model="gpt-4o",
    api_key="your-api-key"
)

# Start a conversation
conversation_id, response = fsm.converse("")
print(f"System: {response}")

# Continue conversation
while not fsm.is_conversation_ended(conversation_id):
    user_input = input("You: ")
    _, response = fsm.converse(user_input, conversation_id)
    print(f"System: {response}")
```

### Starting Conversations with Initial Context

You can pre-populate context data when starting a conversation, which is useful for personalization, session continuation, or skipping unnecessary states:

```python
# Define initial context with user information
initial_context = {
    "name": "Alex Thompson",
    "email": "alex.thompson@example.com",
    "preferred_genres": ["science fiction", "mystery", "fantasy"],
    "membership_level": "premium"
}

# Start a conversation with initial context
conversation_id, response = fsm_manager.start_conversation(
    "examples/personal_information_collection.json",
    initial_context=initial_context
)
```

### Command Line Interface

You can also run conversations directly from the command line:

```bash
# Run a conversation with a specific FSM
llm-fsm --fsm examples/personal_information_collection.json

# Visualize an FSM using ASCII art
llm-fsm-visualize --fsm examples/personal_information_collection.json

# Validate an FSM definition
llm-fsm-validate --fsm examples/personal_information_collection.json
```

## Core Architecture

### FSM Definition

At its core, LLM-FSM uses a JSON structure to define states, transitions, and conditions:

```json
{
  "name": "Personal Information Collection",
  "description": "A conversation flow to collect user information",
  "initial_state": "welcome",
  "persona": "A helpful and friendly assistant who speaks in a warm, conversational tone",
  "states": {
    "welcome": {
      "id": "welcome",
      "description": "Initial welcome state",
      "purpose": "Welcome the user and explain purpose",
      "transitions": [
        {
          "target_state": "collect_name",
          "description": "Always transition to collecting name",
          "priority": 0
        }
      ],
      "instructions": "Warmly welcome the user and explain that you'll be collecting information."
    },
    "collect_name": {
      "id": "collect_name",
      "description": "Collect user's name",
      "purpose": "Ask for and record the user's full name",
      "required_context_keys": ["name"],
      "transitions": [
        {
          "target_state": "collect_email",
          "description": "Transition once name is obtained",
          "conditions": [
            {
              "description": "Name has been provided",
              "requires_context_keys": ["name"]
            }
          ],
          "priority": 0
        }
      ],
      "instructions": "Ask the user for their full name. Extract and store it in the 'name' context variable."
    }
  }
}
```

### JsonLogic Expressions for Powerful Transition Conditions

LLM-FSM includes a powerful JsonLogic implementation that allows for complex conditional logic in state transitions. This enables sophisticated decision-making based on the conversation context:

```json
{
  "transitions": [
    {
      "target_state": "premium_support",
      "description": "Route to premium support",
      "conditions": [
        {
          "description": "Customer is premium member",
          "logic": {
            "or": [
              {"==": [{"var": "customer.tier"}, "premium"]},
              {">": [{"var": "customer.lifetime_value"}, 5000]}
            ]
          }
        }
      ]
    }
  ]
}
```

The expression system supports:

- **Comparison Operators**: `==`, `===`, `!=`, `!==`, `>`, `>=`, `<`, `<=`
- **Logical Operators**: `!`, `!!`, `and`, `or`
- **Context Access**: `var` for retrieving values from context
- **Validation Operators**: `missing`, `missing_some` for checking required fields
- **Conditional Logic**: `if` for if/else decision branches
- **Arithmetic Operations**: `+`, `-`, `*`, `/`, `%`
- **String Operations**: `cat` for concatenation
- **Membership Checks**: `in`, `contains` 

### The Execution Flow

When a user sends a message:

1. **Prompt Construction**: We build a system prompt containing:
   - Current state information
   - Available transitions
   - Extraction instructions
   - Current context values
   - Recent conversation history
   - Response format instructions

2. **LLM Processing**: The LLM:
   - Extracts relevant information from the user's message
   - Determines which state to transition to
   - Generates a natural language response
   - Returns a structured JSON response

3. **State Management**: The FSM Manager:
   - Updates the context with extracted information
   - Validates the proposed state transition (using JsonLogic expressions if present)
   - Updates the current state
   - Stores the conversation history

### LLM Response Format

The LLM returns a structured JSON response:

```json
{
  "transition": {
    "target_state": "collect_email",
    "context_update": {"name": "Alex Thompson"}
  },
  "message": "Nice to meet you, Alex Thompson! Could you please provide your email address?",
  "reasoning": "The user provided their full name 'Alex Thompson', so I'm transitioning to collect their email."
}
```

This format cleanly separates:
- User-facing content (the message)
- System-facing content (transition decision and context updates)
- Debugging information (reasoning)

## Conversation Patterns

LLM-FSM supports various conversation patterns:

### 1. Linear Flows
Step-by-step information collection:
- Personal information forms
- Survey administration
- Onboarding processes

### 2. Conversational Loops
Maintain ongoing engagement:
- Recommendation systems
- Coaching conversations
- Learning assistants

### 3. Decision Trees
Guide users through branching options:
- Product recommendations
- Troubleshooting flows
- Decision support

### 4. Hybrid Patterns
Combine multiple patterns:
- Customer support (identification → troubleshooting → resolution)
- Medical triage (symptoms → assessment → recommendations)
- Educational systems (assessment → instruction → testing)

## Key Design Decisions

### 1. LLM as the NLU Engine

Instead of brittle extraction rules in code:

```python
# Old approach
def extract_name(message):
    patterns = [r"My name is ([\w\s]+)", r"I am ([\w\s]+)"]
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            return match.group(1)
    return None
```

We leverage the LLM's understanding:

```
When the user provides their name, extract it and store it in the 'name' context variable.
Consider explicit mentions ('My name is John') and implicit mentions ('Call me John').
```

### 2. Context Outside States

We maintain context at the conversation level rather than within individual states:

```python
# Context naturally persists across state transitions
{
  "current_state": "collect_email",
  "context": {"name": "John Smith", "email": "john@example.com"}
}
```

### 3. Expressive Transition Conditions with JsonLogic

Using JsonLogic expressions for transition conditions:

```python
# Complex routing logic based on multiple factors
{
  "and": [
    {"==": [{"var": "customer.status"}, "vip"]},
    {"or": [
      {"==": [{"var": "issue.category"}, "billing"]},
      {">": [{"var": "issue.priority"}, 3]}
    ]}
  ]
}
```

## Examples

The repository includes several examples organized by complexity level:

### Basic Examples

#### Simple Greeting Conversation

A minimal FSM with just three states for basic greeting and farewell interactions.

```python
from llm_fsm import LLM_FSM

# Initialize from JSON file
fsm = LLM_FSM.from_file(
    path="examples/basic/simple_greeting/fsm.json",
    model="gpt-4o-mini",
    api_key="your-api-key"
)

# Start conversation
conversation_id, response = fsm.converse("")
print(f"System: {response}")

# Continue conversation until it ends
while not fsm.is_conversation_ended(conversation_id):
    user_input = input("You: ")
    _, response = fsm.converse(user_input, conversation_id)
    print(f"System: {response}")
```

#### Form Filling Example

A step-by-step form filling conversation that collects and validates user information.

```python
from llm_fsm import LLM_FSM

# Initialize the form filling FSM
fsm = LLM_FSM.from_file(
    path="examples/basic/form_filling/fsm.json",
    model="gpt-4o",
    api_key="your-api-key"
)

# Start conversation
conversation_id, response = fsm.converse("")
print(f"System: {response}")

# Process user input
while not fsm.is_conversation_ended(conversation_id):
    user_input = input("You: ")
    _, response = fsm.converse(user_input, conversation_id)
    print(f"System: {response}")

# Get collected data
data = fsm.get_data(conversation_id)
print("Collected information:", data)
```

**Sample Conversation:**
```
System: Hello! I'd like to collect some information from you. Could you please tell me your name?
You: John Smith
System: Thank you, John Smith! Now, could you please provide your email address?
You: john.smith@example.com
System: Thanks for providing your email. Finally, could you tell me your age?
You: 35
System: Great! I've collected the following information:
- Name: John Smith
- Email: john.smith@example.com
- Age: 35
Is this information correct?
You: Yes, that's correct
System: Thank you for confirming your information! Your profile has been created.
```

#### Book Recommendation System

A conversational loop that recommends books based on user preferences and tracks engagement.

```python
from llm_fsm import LLM_FSM

# Create book recommendation system
fsm = LLM_FSM.from_file(
    path="examples/basic/book_recommendation/fsm.json",
    model="gpt-4o",
    api_key="your-api-key"
)

# Start with user preferences
initial_context = {
    "user_name": "Alex",
    "preferred_genres": ["science fiction", "mystery"]
}

# Start conversation with initial context
conversation_id, response = fsm.converse("", initial_context=initial_context)
print(f"System: {response}")

# Continue conversation
while not fsm.is_conversation_ended(conversation_id):
    user_input = input("You: ")
    _, response = fsm.converse(user_input, conversation_id)
    print(f"System: {response}")
```

#### Interactive Storytelling

An interactive storytelling experience based on the classic tale of the Three Little Pigs.

```python
from llm_fsm import LLM_FSM

# Create interactive storytelling experience
fsm = LLM_FSM.from_file(
    path="examples/basic/story_time/fsm.json",
    model="gpt-4o",
    api_key="your-api-key",
    temperature=0.8  # Higher creativity for storytelling
)

print("\n📚 INTERACTIVE STORY EXPERIENCE 📚")
print("=" * 60)

# Start story
conversation_id, response = fsm.converse("")
print(f"\nStoryteller: {response}\n")

# Continue through story
while not fsm.is_conversation_ended(conversation_id):
    user_input = input("You: ")
    _, response = fsm.converse(user_input, conversation_id)
    print(f"\nStoryteller: {response}\n")
```

### Intermediate Examples

#### Product Recommendation System

A decision tree conversation that recommends technology products based on user preferences.

```python
from llm_fsm import LLM_FSM

# Create tech product recommendation system
fsm = LLM_FSM.from_file(
    path="examples/intermediate/product_recommendation_system/fsm.json",
    model="gpt-4o",
    api_key="your-api-key"
)

# Start the conversation
conversation_id, response = fsm.converse("")
print(f"Advisor: {response}")

# Main conversation loop
while not fsm.is_conversation_ended(conversation_id):
    user_input = input("You: ")
    _, response = fsm.converse(user_input, conversation_id)
    print(f"Advisor: {response}")
```

**Sample Conversation:**
```
Advisor: Hello and welcome to our tech product recommendation service! I'm here to help you find the perfect tech product. What are you looking for today?
You: I need a new laptop
Advisor: Great! I'd be happy to help you find a laptop. Could you tell me about your budget range? Are you looking for something more budget-friendly (under $800) or would you prefer a premium option (over $800)?
You: I'd like something premium, around $1500
Advisor: Excellent choice! With a budget of $1500, here are some premium laptops I recommend:

1. MacBook Air M2 ($1199-1499) - Excellent performance with great battery life, perfect for professionals.
2. Dell XPS 13 ($1299-1499) - Beautiful display and build quality, ideal for Windows users.
3. ThinkPad X1 Carbon ($1349-1599) - Known for durability and business features.

Which of these interests you most?
```

### Advanced Examples

#### Adaptive Yoga Instruction

A sophisticated FSM that monitors user engagement and adapts the conversation flow accordingly.

```python
from llm_fsm import LLM_FSM

# Create adaptive yoga instruction system
fsm = LLM_FSM.from_file(
    path="examples/advanced/yoga_instructions/fsm.json",
    model="gpt-4o",
    api_key="your-api-key",
    temperature=0.7
)

# Initialize with engagement tracking
initial_context = {
    "engagement_level": "medium",
    "completed_poses": []
}

# Start conversation
conversation_id, response = fsm.converse("", initial_context=initial_context)
print(f"Instructor: {response}")

# Continue conversation
while not fsm.is_conversation_ended(conversation_id):
    user_input = input("You: ")
    _, response = fsm.converse(user_input, conversation_id)
    print(f"Instructor: {response}")
```

### Feature Examples

#### Context Persistence & Correction Flows

The LLM-FSM framework maintains context throughout the conversation, allowing users to correct information:

```
System: Here's what I've collected:
- Name: John Smith
- Email: john.smith@example.com
- Age: 35
Is this information correct?
You: Actually, my email is john.s.smith@example.com
System: I've updated your email to john.s.smith@example.com. Is the information correct now?
You: Yes
System: Thank you for confirming your information!
```

#### Using JsonLogic Expressions for Powerful Transitions

```json
{
  "transitions": [
    {
      "target_state": "premium_support",
      "description": "Route premium customers to special support",
      "conditions": [
        {
          "description": "Customer is premium tier",
          "logic": {
            "or": [
              {"==": [{"var": "customer.tier"}, "premium"]},
              {">": [{"var": "customer.spending"}, 5000]}
            ]
          }
        }
      ]
    }
  ]
}
```

## Persona Support

One powerful feature of LLM-FSM is the ability to define a consistent persona for the entire conversation flow:

```json
{
  "name": "Three Little Pigs Interactive Story",
  "description": "An interactive storytelling experience based on the classic tale",
  "initial_state": "introduction",
  "persona": "You are J.R.R Tolkien master epic story teller.",
  "states": {
    // State definitions
  }
}
```

The persona:
- Defines a consistent tone, style, and voice for all responses
- Is maintained across all states in the conversation
- Is incorporated into the system prompt for each LLM request
- Enables creating specialized conversational experiences (storytellers, educators, customer service agents, etc.)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.