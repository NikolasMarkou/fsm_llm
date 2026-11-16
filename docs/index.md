# FSM-LLM

> Covers FSM-LLM v0.13.0

Stateful conversational AI: you describe a conversation as a finite state machine in JSON, and a language model does the language work inside each state.

A language model call starts from nothing. A multi-turn conversation that must collect data, follow a set flow and decide the same way every time needs structure around the model. FSM-LLM splits the work three ways: the model understands the user, pulls out facts and writes replies; the state machine decides where the conversation goes next by rules, not by mood; the framework keeps the state, the history, your hooks and the saved sessions.

It is one package, `fsm_llm`, with a small core and seven subpackages on top: reasoning strategies, a workflow engine, 18 agent patterns, a monitoring dashboard, an evaluation harness, runtime FSM synthesis and public test doubles. Every subpackage is built from the same parts as the core (stacked FSMs, message-free steps, one LLM layer), so there is one way to run a model.

## Install

```bash
pip install llm-fsm
```

The distribution is `llm-fsm` and the import package is `fsm_llm`. Python 3.11 or newer. `fsm-llm` on PyPI is a different, unrelated project: if you installed it by mistake, run `pip uninstall fsm-llm` first. The [Quick Start](quickstart.md) covers the extras and the model setting.

## The shortest working example

This runs with no model and no network: a scripted model double from `fsm_llm.testing` stands in for the LLM. With a real model you drop the double and pass `model=` (or set `LLM_MODEL`).

```python
from fsm_llm import API
from fsm_llm.testing import ScriptedLLM

greeter = {
    "name": "Greeter",
    "description": "Learns the user's name, then says goodbye",
    "initial_state": "ask_name",
    "states": {
        "ask_name": {
            "id": "ask_name",
            "description": "Find out the user's name",
            "purpose": "Learn the user's name",
            "extraction_instructions": "Extract the user's name if they gave it",
            "response_instructions": "Ask for the user's name",
            "required_context_keys": ["name"],
            "transitions": [
                {
                    "target_state": "goodbye",
                    "description": "The name is known",
                    "conditions": [
                        {
                            "description": "A name was collected",
                            "requires_context_keys": ["name"],
                            "logic": {"has_context": "name"},
                        }
                    ],
                }
            ],
        },
        "goodbye": {
            "id": "goodbye",
            "description": "Say goodbye",
            "purpose": "Wrap up the conversation",
            "response_instructions": "Say goodbye by name",
        },
    },
}

llm = ScriptedLLM(
    extraction_data={"name": "Alice"},
    responses={"ask_name": "What is your name?", "goodbye": "Goodbye, Alice!"},
)
api = API.from_definition(greeter, llm_interface=llm)
conversation_id, opening = api.start_conversation()
print(opening)                                      # What is your name?
print(api.converse("I am Alice", conversation_id))  # Goodbye, Alice!
print(api.get_data(conversation_id), api.get_current_state(conversation_id))
```

The conversation starts in `ask_name`, the extraction fills `name`, the transition condition passes, and the reply is written from the state the conversation ended in, `goodbye`. The [Quick Start](quickstart.md) runs the same shape against a real model.

## Where to go next

| You want to | Read |
|---|---|
| Run a first conversation against a model | [Quick Start](quickstart.md) |
| Understand the idea, the two passes and what each subpackage adds | [Concepts](concepts.md) |
| Design states, transitions, priorities, context and classification | [FSM Design Guide](fsm_design.md) |
| Run your own code at the eight points of a turn | [Handler Development](handlers.md) |
| Let a model write a sub-conversation at run time | [Runtime FSM synthesis](synthesis.md) |
| Test a bot, an agent or a workflow without a model | [Testing](testing.md) |
| Look up a class, a function or an exception | [API Reference](api_reference.md) |
| Look up a command-line tool | [Command line](cli.md) |
| See how the layers fit together | [Architecture](architecture.md) |
| Run an example from the repository | [Examples](examples.md) |
| See what changed and what moves next | [Changelog](changelog.md), [2026 roadmap](2026_roadmap.md), [MCP server roadmap](20261116_mcp_roadmap.md) |

## The subpackages

Each subpackage is imported by name (`from fsm_llm.agents import create_agent`); `import fsm_llm` loads only the core. Their usage is in the API Reference, and each has a README with its own guide:

| Subpackage | What it adds | README |
|---|---|---|
| `fsm_llm.reasoning` | 9 reasoning strategies as stacked FSMs | [reasoning](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/reasoning/README.md) |
| `fsm_llm.workflows` | An async in-memory workflow engine with 11 step types | [workflows](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/workflows/README.md) |
| `fsm_llm.agents` | 18 agent patterns, tools, human approval, memory, MCP, A2A | [agents](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/agents/README.md) |
| `fsm_llm.monitor` | A FastAPI dashboard and OpenTelemetry export | [monitor](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/monitor/README.md) |
| `fsm_llm.eval` | An examples scorer and scripted conversation evals | [eval](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/eval/README.md) |
| `fsm_llm.synthesis` | A model compiles a goal into an FSM that gates judge | [synthesis](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/synthesis/README.md) |
| `fsm_llm.testing` | Test doubles, a scripted user and assertions | [testing](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/testing/README.md) |

## Project

[Contributing](contributing.md) (setup, checks and the rules for a change), [Security](security.md) (reporting a vulnerability and the dependency checks) and [Evaluate](evaluate.md) (the method for scoring the examples against a model, and the run log). The code and the issue tracker are on [GitHub](https://github.com/NikolasMarkou/fsm_llm). The library is released under the Apache License 2.0.
