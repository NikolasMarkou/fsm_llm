<p align="center">
  <a href="https://www.apache.org/licenses/LICENSE-2.0">
    <img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg">
  </a>
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14%20%7C%203.15-blue">
  <img alt="Version" src="https://img.shields.io/badge/version-0.12.0-green">
  <a href="https://github.com/NikolasMarkou/fsm_llm/actions/workflows/python-package.yml">
    <img alt="Tests" src="https://github.com/NikolasMarkou/fsm_llm/actions/workflows/python-package.yml/badge.svg">
  </a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/NikolasMarkou/fsm_llm/main/images/fsm-llm-logo-1.png" alt="FSM-LLM Logo" width="500"/>
</p>

A language model call starts from nothing. A multi-turn conversation that must collect data, follow a set flow and decide the same way every time needs structure around the model. **FSM-LLM** is a Python framework for chatbots and agents that follow a fixed structure: you describe the conversation as a finite state machine in JSON, and the language model does the language work inside each state.

It is one package, `fsm_llm`, with a small core and seven subpackages on top: reasoning strategies, a workflow engine, 18 agent patterns, a monitoring dashboard, an evaluation harness, runtime FSM synthesis and public test doubles. Every subpackage is built from the same parts as the core (stacked FSMs, message-free steps, one LLM layer), so there is one way to run a model, not five.

## Documentation

-   **Documentation site:** [https://nikolasmarkou.github.io/fsm_llm/](https://nikolasmarkou.github.io/fsm_llm/) (guides, concepts, the API reference, the command-line tools and the decision index).
-   **Guides on GitHub:** [Quick Start](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/quickstart.md) (also the builders, an agent, a workflow and the reasoning engine), [Concepts](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/concepts.md) (what is in the box, where it fits, how a turn works), [FSM Design](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/fsm_design.md), [Handlers](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/handlers.md), [Architecture](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/architecture.md), [Testing](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/testing.md) and the [API Reference](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/api_reference.md).
-   **Examples and command-line tools:** [runnable examples](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/examples.md) (each is a folder with `run.py` and its FSM JSON) and the [command-line tools](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/reference/cli.md).
-   **Per-package READMEs:** [core](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/README.md), [reasoning](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/reasoning/README.md), [workflows](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/workflows/README.md), [agents](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/agents/README.md), [monitor](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/monitor/README.md), [eval](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/eval/README.md), [synthesis](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/synthesis/README.md) and [testing](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/testing/README.md).
-   **Project:** [source on GitHub](https://github.com/NikolasMarkou/fsm_llm), [CHANGELOG](https://github.com/NikolasMarkou/fsm_llm/blob/main/CHANGELOG.md), [CONTRIBUTING](https://github.com/NikolasMarkou/fsm_llm/blob/main/CONTRIBUTING.md) (setup, tests, repository layout), [SECURITY](https://github.com/NikolasMarkou/fsm_llm/blob/main/SECURITY.md) and [EVALUATE](https://github.com/NikolasMarkou/fsm_llm/blob/main/EVALUATE.md) (evaluation method and run log).

## Installation

> **Note:** This library requires Python 3.11 or newer (tested on 3.11 to 3.15). It is published on PyPI as `llm-fsm`; the import package is `fsm_llm`. `fsm-llm` on PyPI is a different, unrelated project: if you installed it by mistake, run `pip uninstall fsm-llm` first.

1.  **Create an environment and install:**
    ```bash
    python -m venv .venv && source .venv/bin/activate
    pip install llm-fsm
    ```

2.  **Optional extras:**
    `import fsm_llm` loads only the core, and every subpackage ships in every install. An extra only adds third-party dependencies.

    | Extra | Adds |
    |-------|------|
    | `reasoning`, `agents`, `workflows`, `eval`, `synthesis`, `testing` | No third-party packages (the code ships in every install) |
    | `monitor` | fastapi, uvicorn, jinja2 |
    | `mcp` | mcp |
    | `otel` | opentelemetry-api, opentelemetry-sdk |
    | `a2a` | httpx |
    | `all` | Every extra above |
    | `dev` | The test, lint and type tools (for contributors, see [CONTRIBUTING](https://github.com/NikolasMarkou/fsm_llm/blob/main/CONTRIBUTING.md)) |

    ```bash
    pip install "llm-fsm[monitor]"
    pip install "llm-fsm[all]"
    ```

    The core install needs loguru, litellm, pydantic, python-dotenv and pyyaml (>=6.0, which parses OKF and Agent Skills frontmatter; declared explicitly because it used to arrive only through another package).

3.  **Pick a model:**
    ```bash
    export LLM_MODEL=ollama_chat/qwen3.5:4b      # or openai/gpt-4o-mini with OPENAI_API_KEY set
    ```
    The model is the `model=` argument, then `LLM_MODEL`; there is no built-in default, and with neither set the call raises `ModelNotConfiguredError` (a CLI prints its message and exits 1). No `.env` file is read unless you ask: call `fsm_llm.load_env()` (or `fsm_llm.load_env("path/to/.env")`) in Python, or pass `--env-file PATH` to a CLI that needs a model (`fsm-llm`, `fsm-llm-eval`, `fsm-llm-meta`, `python -m fsm_llm.reasoning`, `fsm-llm-monitor`); a variable the process already has is not replaced unless `override=True`, except an empty or whitespace-only one, which counts as unset and is filled when the file gives it a value (to keep one empty on purpose, do not name it in the file; see `.env.example`).

4.  **Verify the installation:**
    ```bash
    python -c "import fsm_llm; print('Installation successful!')"
    ```

## Quick Start

### 1. Define and Run a Conversation

Define the conversation (`greeter.json`). It asks for a name, then moves to a final state:

```json
{
  "name": "Greeter",
  "description": "Learns the user's name, then says goodbye",
  "initial_state": "ask_name",
  "persona": "A friendly assistant",
  "states": {
    "ask_name": {
      "id": "ask_name",
      "description": "Find out the user's name",
      "purpose": "Learn the user's name",
      "extraction_instructions": "Extract the user's name if they gave it",
      "response_instructions": "Greet the user and ask for their name if you do not know it yet",
      "required_context_keys": ["name"],
      "transitions": [
        {
          "target_state": "goodbye",
          "description": "The name is known",
          "conditions": [
            {
              "description": "A name was collected",
              "requires_context_keys": ["name"],
              "logic": {"has_context": "name"}
            }
          ]
        }
      ]
    },
    "goodbye": {
      "id": "goodbye",
      "description": "Say goodbye",
      "purpose": "Wrap up the conversation",
      "response_instructions": "Say a short goodbye to the user by name. Do not ask a question"
    }
  }
}
```

Run it:

```python
from fsm_llm import API

api = API.from_file("greeter.json")   # no model given: env LLM_MODEL, else ModelNotConfiguredError
conv_id, opening = api.start_conversation()
print("bot:", opening)
print("bot:", api.converse("Hi, I'm Alice.", conv_id))
print(api.get_data(conv_id))
print(api.get_current_state(conv_id), api.has_conversation_ended(conv_id))
```

A real run against a local Ollama `qwen3.5:4b` (the wording differs on every run):

```text
bot: Hello! How do you do? I'm here to get to know you better. What's your name?
bot: Nice meeting you, Alice! It was a pleasure getting to know you.
{'name': 'Alice'}
goodbye True
```

To use a hosted model, pass `model=` (for example `API.from_file("greeter.json", model="openai/gpt-4o-mini")`) and set the provider key (`OPENAI_API_KEY`).

A state with no transitions (`goodbye`) is terminal. `required_context_keys` only tells Pass 1 what to extract; it never blocks a transition, so the transition above carries its own condition (`requires_context_keys` plus `logic`).

### 2. Try It Without a Model

Pass your own `llm_interface` (a subclass of `LLMInterface`) and nothing touches the network. This is how the repository's tests run:

```python
from fsm_llm import API, LLMInterface
from fsm_llm.core.definitions import FieldExtractionResponse, ResponseGenerationResponse


class FakeLLM(LLMInterface):
    """Fixed reply and a fixed name; makes no network call."""

    def generate_response(self, request):
        return ResponseGenerationResponse(
            message="Nice to meet you!", message_type="response", reasoning="fake"
        )

    def extract_field(self, request):
        return FieldExtractionResponse(
            field_name=request.field_name, value="Alice", confidence=1.0,
            reasoning="fake", is_valid=True,
        )


api = API.from_file("greeter.json", llm_interface=FakeLLM())
conv_id, _ = api.start_conversation()
print(api.converse("anything", conv_id))                    # Nice to meet you!
print(api.get_data(conv_id), api.get_current_state(conv_id))  # {'name': 'Alice'} goodbye
```

`API` refuses `model=`, `temperature=` and similar settings next to `llm_interface=`, because the interface owns them.

Builders, agents, workflows and the reasoning engine are in the [Quick Start guide](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/quickstart.md).

## License

This project is licensed under the **Apache License 2.0**.

- **Permissive**: you may use, modify and distribute the library, including in commercial and closed-source products.
- **Attribution**: redistributions must retain the copyright notice and a copy of the license, and modified files must carry a notice stating that they were changed.
- **Patent grant**: contributors grant an express patent license covering their contributions.

See the [LICENSE](https://github.com/NikolasMarkou/fsm_llm/blob/main/LICENSE) file for complete details.
