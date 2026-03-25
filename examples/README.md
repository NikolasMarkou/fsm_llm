# FSM-LLM Examples

Practical examples demonstrating FSM-LLM capabilities, organized by complexity.

## Getting Started

```bash
pip install fsm-llm
export OPENAI_API_KEY="your-key-here"

# Or use a local Ollama model (no API key needed):
export LLM_MODEL="ollama_chat/qwen3.5:4b"
```

Each example contains a `run.py` entry point. Most examples also include `*.json` FSM definition files, though some (e.g., `e_commerce`, `support_pipeline`, `intent_routing`) define FSMs inline in Python.

Basic and intermediate examples work well with `qwen3.5:4b`. Classification, reasoning, and workflows examples may need `qwen3.5:9b` for reliable structured output.

All interactive examples support typing `exit` or `quit` to end the conversation.

## Examples by Complexity

### Basic

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [simple_greeting](basic/simple_greeting/) | 3-state greeting bot | States, transitions, initial/terminal states |
| [form_filling](basic/form_filling/) | Data collection form | `required_context_keys`, data extraction |
| [story_time](basic/story_time/) | Interactive story narrator | Multi-turn conversation, context flow |

### Intermediate

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [book_recommendation](intermediate/book_recommendation/) | Book recommender | Conditional transitions, context-based routing |
| [product_recommendation](intermediate/product_recommendation/) | Product discovery assistant | Multi-step discovery, preference tracking |
| [adaptive_quiz](intermediate/adaptive_quiz/) | Adaptive trivia quiz with handler-driven scoring | `HandlerBuilder` fluent API, `HandlerTiming`, runtime context mutation |

### Advanced

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [yoga_instructions](advanced/yoga_instructions/) | Adaptive yoga instructor | User engagement tracking, non-linear FSM design |
| [e_commerce](advanced/e_commerce/) | E-commerce checkout flow | FSM stacking (`push_fsm`/`pop_fsm`), context merging |
| [support_pipeline](advanced/support_pipeline/) | Full support pipeline with classification, stacking, and handlers | `Classifier` + `push_fsm`/`pop_fsm` + `HandlerBuilder`, multi-package integration |

### Classification

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [intent_routing](classification/intent_routing/) | Customer support intent classifier with handler routing | `Classifier`, `IntentRouter`, schema design |
| [smart_helpdesk](classification/smart_helpdesk/) | Classify intent then route to specialized FSM conversations | `Classifier` + `API.from_file()`, classification-driven FSM selection |
| [classified_transitions](classification/classified_transitions/) | Classification-aware transition routing | `classified_transitions` mode, multi-path FSM routing |

### Reasoning

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [math_tutor](reasoning/math_tutor/) | Math tutor combining conversational FSM with structured reasoning | `ReasoningEngine.solve_problem()`, injecting reasoning traces into FSM context |

### Workflows

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [order_processing](workflows/order_processing/) | Order pipeline with FSM conversation step inside a workflow | `WorkflowEngine`, `ConversationStep`, `api_step`, `condition_step`, async execution |

### Agents

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [react_search](agents/react_search/) | ReAct agent with a search tool | `ReactAgent`, `ToolRegistry`, `@tool` decorator |
| [hitl_approval](agents/hitl_approval/) | Human-in-the-loop approval gates | `HumanInTheLoop`, `approval_policy`, `approval_callback` |
| [react_hitl_combined](agents/react_hitl_combined/) | ReAct agent with HITL approval on sensitive tools | `ReactAgent` + `HumanInTheLoop` combined |
| [plan_execute](agents/plan_execute/) | Plan decomposition and sequential execution | `PlanExecuteAgent`, upfront planning, step-by-step execution |
| [reflexion](agents/reflexion/) | Self-reflection with episodic memory | `ReflexionAgent`, `evaluation_fn`, verbal self-critique |
| [debate](agents/debate/) | Multi-perspective debate with judge | `DebateAgent`, proposer/critic/judge personas |
| [self_consistency](agents/self_consistency/) | Multiple samples with majority voting | `SelfConsistencyAgent`, `num_samples`, `aggregation_fn` |
| [rewoo](agents/rewoo/) | Planning-first tool execution | `REWOOAgent`, evidence references (`#E1`, `#E2`) |
| [prompt_chain](agents/prompt_chain/) | Sequential prompt pipeline with gates | `PromptChainAgent`, `ChainStep` pipeline |
| [evaluator_optimizer](agents/evaluator_optimizer/) | Iterative evaluation and optimization | `EvaluatorOptimizerAgent`, `evaluation_fn`, refinement loop |
| [maker_checker](agents/maker_checker/) | Draft-review verification loop | `MakerCheckerAgent`, two-persona quality loop |
| [classified_dispatch](agents/classified_dispatch/) | Classification-driven agent selection | `Classifier` + agent dispatch based on intent |
| [classified_tools](agents/classified_tools/) | Classification for tool selection within an agent | `ToolRegistry.to_classification_schema()` |
| [full_pipeline](agents/full_pipeline/) | End-to-end pipeline: classify → agent → tools | Classification + ReactAgent + multiple tools |
| [hierarchical_tools](agents/hierarchical_tools/) | Hierarchical tool composition | Nested tool registries, tool grouping |
| [reasoning_stacking](agents/reasoning_stacking/) | Agent with FSM stacking for reasoning | `ReasoningReactAgent`, push/pop reasoning FSMs |
| [reasoning_tool](agents/reasoning_tool/) | Reasoning engine exposed as a tool | `ReasoningEngine` wrapped in `@tool` decorator |
| [workflow_agent](agents/workflow_agent/) | Agent integrated with workflow orchestration | Agent + `WorkflowEngine`, FSM stacking |

### Meta

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [build_fsm](meta/build_fsm/) | Interactive FSM builder using the meta-agent | `MetaAgent`, `FSMBuilder`, turn-by-turn conversation, artifact validation |

## Sub-Package Usage Matrix

| Example | Core FSM | Classification | Reasoning | Workflows | Agents | Handlers | Meta |
|---------|:--------:|:--------------:|:---------:|:---------:|:------:|:--------:|:----:|
| simple_greeting | x | | | | | | |
| form_filling | x | | | | | | |
| story_time | x | | | | | | |
| book_recommendation | x | | | | | | |
| product_recommendation | x | | | | | | |
| **adaptive_quiz** | x | | | | | x | |
| yoga_instructions | x | | | | | | |
| e_commerce | x | | | | | | |
| intent_routing | | x | | | | | |
| **smart_helpdesk** | x | x | | | | | |
| **classified_transitions** | x | x | | | | | |
| **math_tutor** | x | | x | | | | |
| **order_processing** | x | | | x | | | |
| **support_pipeline** | x | x | | | | x | |
| react_search | x | | | | x | | |
| hitl_approval | x | | | | x | | |
| react_hitl_combined | x | | | | x | | |
| plan_execute | x | | | | x | | |
| reflexion | x | | | | x | | |
| debate | x | | | | x | | |
| self_consistency | x | | | | x | | |
| rewoo | x | | | | x | | |
| prompt_chain | x | | | | x | | |
| evaluator_optimizer | x | | | | x | | |
| maker_checker | x | | | | x | | |
| **classified_dispatch** | x | x | | | x | | |
| **classified_tools** | x | x | | | x | | |
| **full_pipeline** | x | x | | | x | | |
| hierarchical_tools | x | | | | x | | |
| **reasoning_stacking** | x | | x | | x | | |
| **reasoning_tool** | x | | x | | x | | |
| **workflow_agent** | x | | | x | x | | |
| **build_fsm** | | | | | | | x |

## Suggested Learning Path

1. Start with **simple_greeting** to understand the basic FSM structure
2. Move to **form_filling** to learn about data extraction
3. Try **adaptive_quiz** to learn the handler system
4. Try **book_recommendation** for conditional transitions
5. Explore **smart_helpdesk** to combine classification with FSM conversations
6. Try **react_search** to learn the agentic ReAct pattern with tools
7. Try **hitl_approval** to understand human-in-the-loop approval gates
8. Explore **plan_execute** or **reflexion** for more advanced agent patterns
9. Explore **math_tutor** to see reasoning engine integration
10. Explore **order_processing** to see workflow orchestration
11. Study **full_pipeline** for the complete classify → agent → tools pipeline
12. Study **support_pipeline** for the full multi-package integration
13. Try **build_fsm** to interactively construct FSM definitions with the meta-agent
