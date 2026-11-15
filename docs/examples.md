# Examples

> Covers FSM-LLM v0.12.0

The repository holds 100 runnable examples in 8 categories. Each is a folder with a `run.py` and
its FSM JSON. The examples are not part of this site: the links below open the folders on
GitHub.

The examples are not in the installed package (`pip install llm-fsm`). Clone the repository to get them
(`git clone https://github.com/NikolasMarkou/fsm_llm.git`) and run one from the clone:

```bash
python examples/basic/simple_greeting/run.py     # needs a model: an OpenAI key, or a local Ollama
```

Set the model first (`LLM_MODEL`, see the [Quick Start](quickstart.md)); an example makes real
model calls.

| Category | Count | Highlights |
|----------|-------|------------|
| [Basic](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/basic) | 14 | [simple_greeting](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/basic/simple_greeting), [form_filling](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/basic/form_filling), [multi_turn_extraction](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/basic/multi_turn_extraction) |
| [Intermediate](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/intermediate) | 3 | [book_recommendation](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/intermediate/book_recommendation), [product_recommendation](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/intermediate/product_recommendation), [adaptive_quiz](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/intermediate/adaptive_quiz) |
| [Advanced](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/advanced) | 17 | [e_commerce](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/advanced/e_commerce) (FSM stacking), [support_pipeline](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/advanced/support_pipeline), [concurrent_conversations](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/advanced/concurrent_conversations) |
| [Classification](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/classification) | 4 | [intent_routing](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/classification/intent_routing), [smart_helpdesk](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/classification/smart_helpdesk), [multi_intent](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/classification/multi_intent) |
| [Reasoning](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/reasoning) | 1 | [math_tutor](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/reasoning/math_tutor) |
| [Workflows](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/workflows) | 8 | [order_processing](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/workflows/order_processing), [parallel_steps](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/workflows/parallel_steps), [conditional_branching](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/workflows/conditional_branching) |
| [Agents](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/agents) | 48 | [react_search](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/agents/react_search), [plan_execute](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/agents/plan_execute), [reflexion](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/agents/reflexion), [debate](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/agents/debate), [orchestrator](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/agents/orchestrator) |
| [Meta](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/meta) | 5 | [build_fsm](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/meta/build_fsm), [build_workflow](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/meta/build_workflow), [build_agent](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/meta/build_agent) |

All of them are on GitHub under [`examples/`](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples).

## They are evaluation baselines

The examples are the baselines for `fsm-llm-eval examples`, which runs them in parallel and
scores each from 0 to 4. The score is a heuristic that overstates quality; see
[Evaluate](evaluate.md) for the method and the recorded runs, and the
[command line reference](reference/cli.md) for the flags. Do not edit the examples unless a
maintainer asks: changing one moves the baseline.
