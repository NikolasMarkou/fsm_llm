# Default model and dotenv files

Covers FSM-LLM 0.12.0 to 0.13.0. The package no longer picks a model for you and no longer reads a `.env` file behind your back. In 0.12.0 a call with no model used `ollama_chat/bonsai-2:27b` silently, and importing `fsm_llm` could load a stale `.env` found by litellm. In 0.13.0 a model is always named by the caller, by `LLM_MODEL` in the process environment, or by an injected `llm_interface`; a dotenv file is read only when you ask for it.

The precedence everywhere is: the `model=` argument (or `--model`), then the `LLM_MODEL` environment variable, then, for the command line tools only, the file given with `--env-file`, then `ModelNotConfiguredError`. A blank value counts as unset.

The sections follow the order of the CHANGELOG entries.

## resolve_model raises when no model is configured

`fsm_llm.llm.resolve_model` is the one resolver behind `API`, `AgentConfig()` and the eval runners. With no `model=` and no `LLM_MODEL` it raises the new `ModelNotConfiguredError` (exported from `fsm_llm`; it is an `FSMError` and a `ValueError`, so an existing `except ValueError` still catches it). Its one-line message names `model=`, `LLM_MODEL` and `fsm_llm.load_env()`.

An injected `llm_interface` needs no model: the interface owns it.

**Before**

```python
from fsm_llm import API

api = API.from_file("bot.json")  # silently used ollama_chat/bonsai-2:27b
```

**After**

```python
import os

from fsm_llm import API, ModelNotConfiguredError

definition = {
    "name": "Bot",
    "description": "A one-state bot",
    "initial_state": "hello",
    "states": {
        "hello": {
            "id": "hello",
            "description": "Greet",
            "purpose": "Say hello",
            "response_instructions": "Say hello",
        }
    },
}

os.environ.pop("LLM_MODEL", None)
try:
    API.from_definition(definition)
except ModelNotConfiguredError as error:
    assert "LLM_MODEL" in str(error)

api = API.from_definition(definition, model="ollama_chat/example")  # the argument
os.environ["LLM_MODEL"] = "ollama_chat/example"
api = API.from_definition(definition)  # or the environment
```

## Classifier and agents need a model too

`Classifier(schema)` and `HierarchicalClassifier(schema)` with neither `model=` nor `llm_interface=` no longer default to the removed `DEFAULT_LLM_MODEL`, and they no longer ignore the environment: they resolve the model like everything else and raise `ModelNotConfiguredError` when none is set. Inside a conversation nothing changes: the classifier uses the conversation's own interface.

`AgentConfig()` and `default_llm_judge()` raise the same error when no model is set. `default_llm_judge(complete_fn=...)` does not, because a custom backend owns the model. An agent, `create_agent` or `ConfiguredAgentBuilder` that holds an injected `llm_interface` needs no model: `default_agent_config` takes the interface's own `model`, else `LLM_MODEL`, else a placeholder that is never sent. `AgentArtifactBuilder._config_problems` reports a missing model as a problem instead of raising.

**Before**

```python
from fsm_llm import Classifier
from fsm_llm.agents import AgentConfig

classifier = Classifier(schema)  # used the default model
config = AgentConfig()
```

**After**

```python
import os

from fsm_llm import (
    ClassificationSchema,
    Classifier,
    IntentDefinition,
    ModelNotConfiguredError,
)
from fsm_llm.agents import AgentConfig

schema = ClassificationSchema(
    intents=[
        IntentDefinition(name="buy", description="wants to purchase"),
        IntentDefinition(name="browse", description="just looking"),
    ],
    fallback_intent="browse",
)

os.environ.pop("LLM_MODEL", None)
for build in (lambda: Classifier(schema), lambda: AgentConfig()):
    try:
        build()
    except ModelNotConfiguredError:
        pass
    else:
        raise AssertionError("a model was invented")

classifier = Classifier(schema, model="ollama_chat/example")
config = AgentConfig(model="ollama_chat/example")
assert config.model == "ollama_chat/example"
```

## The default model constants are removed

`fsm_llm.constants.DEFAULT_LLM_MODEL`, `fsm_llm.agents.constants.Defaults.MODEL` and `fsm_llm.reasoning.constants.Defaults.MODEL` are removed, with no alias: the package has no default model. Anyone who imported `DEFAULT_LLM_MODEL` for its value should name the model they want.

The signatures change with them. `ReasoningEngine(model=None)` (was `model=Defaults.MODEL`; an `llm_interface` in its keyword arguments needs no model), `InstanceManager.launch_fsm(model=None)` and `launch_agent(model=None)`, and the `model` field of the monitor requests `LaunchFSMRequest`, `LaunchAgentRequest` and `BuilderStartRequest` is `None` by default (a blank value counts as unset). Every one resolves the model through `resolve_model`.

- The monitor routes `POST /api/fsm/launch`, `/api/agent/launch` and `/api/builder/start` answer HTTP 400 with the error message when neither the request nor `LLM_MODEL` names a model.
- `python -m fsm_llm.reasoning` has `--model` defaulting to `None`, so the engine now reads `LLM_MODEL`, which it ignored before.
- `fsm-llm-eval examples` and `run`, `fsm-llm-meta` and `fsm-llm` print the message as one stderr line (`fsm-llm`: `Error: ...`) and exit 1 with no traceback. `fsm-llm-eval --help` and `fsm-llm-meta --help` no longer name a default.

**Before**

```python
from fsm_llm.constants import DEFAULT_LLM_MODEL
from fsm_llm.reasoning import ReasoningEngine

engine = ReasoningEngine()
print(DEFAULT_LLM_MODEL)
```

**After**

```python
import fsm_llm.constants
import fsm_llm.reasoning.constants
from fsm_llm.monitor import LaunchFSMRequest
from fsm_llm.reasoning import ReasoningEngine

assert not hasattr(fsm_llm.constants, "DEFAULT_LLM_MODEL")
assert not hasattr(fsm_llm.reasoning.constants.Defaults, "MODEL")
assert LaunchFSMRequest.model_fields["model"].default is None

engine = ReasoningEngine(model="ollama_chat/example")
```

## Importing fsm_llm no longer loads a dotenv file

litellm reads a `.env` file when it is imported (python-dotenv searches upward from litellm's own directory, or from the working directory in `python -c`, a REPL or a debugger), so a stale file could fill `LLM_MODEL` or a provider key behind your back. `fsm_llm.llm` now sets `LITELLM_MODE=PRODUCTION` around its one `import litellm` when you have not set the variable, and restores the environment right after, so `LITELLM_MODE` is never left in `os.environ` for child processes.

A caller that set `LITELLM_MODE` itself is respected (`DEV` opts back in). A host that imports litellm before `fsm_llm` has already loaded the file, and that cannot be undone.

To read a file on purpose, call `fsm_llm.load_env()`: with no argument it searches from the working directory upward (never from the package directory); an explicit path that does not exist raises `FileNotFoundError`. A variable the process already has is not replaced unless you pass `override=True`; an empty or whitespace-only one counts as unset and is filled.

**Before**

```python
import fsm_llm  # litellm had already read a .env found on its own search path

api = fsm_llm.API.from_file("bot.json")
```

**After**

```python
import os
from pathlib import Path

import fsm_llm

Path("settings.env").write_text("LLM_MODEL=ollama_chat/example\n")
os.environ.pop("LLM_MODEL", None)
found = fsm_llm.load_env("settings.env")  # read on purpose, never at import
assert found is not None
assert os.environ["LLM_MODEL"] == "ollama_chat/example"

try:
    fsm_llm.load_env("missing.env")
except FileNotFoundError:
    pass
```

## The fsm-llm command reads a dotenv file only with --env-file

The `fsm-llm` command no longer reads a `.env` file by itself. It called `dotenv.load_dotenv()` on every run, which searched upward from the package directory (a checkout's own `.env`, never the file in the directory you ran it from) and let a stale file supply `LLM_MODEL` and provider keys unasked. It now reads a file only through `--env-file PATH`, with the same rule as `fsm_llm.load_env`. The other model command line tools take the same flag: `fsm-llm-eval`, `fsm-llm-meta`, `python -m fsm_llm.reasoning` and `fsm-llm-monitor`.

A missing `--env-file` path, or a missing `--fsm` file, now prints one `Error:` line on stderr and exits 1 instead of a traceback. To keep a variable deliberately empty, for example `HTTP_PROXY=`, do not name it in the file.

`fsm_llm.runner.dotenv` is gone (the `runner` module no longer imports python-dotenv; `src/fsm_llm/env.py` is the one module that does), so a test or a host that patched `fsm_llm.runner.dotenv.load_dotenv` must patch `fsm_llm.runner.load_env` instead.

**Before**

```bash
cd ~/work && fsm-llm --fsm bot.json   # picked up a .env from the checkout
```

**Before**

```python
from unittest.mock import patch

with patch("fsm_llm.runner.dotenv.load_dotenv"):
    ...
```

**After**

```bash
fsm-llm --fsm bot.json --env-file ~/work/.env
```

**After**

```python
import fsm_llm.runner

assert not hasattr(fsm_llm.runner, "dotenv")
assert callable(fsm_llm.runner.load_env)  # the name to patch
```

## The default embedder changed

The default embedder is now `ollama/embeddinggemma-2:740m` (`DEFAULT_EMBEDDING_MODEL`, was `ollama/qwen3-embedding:0.6b`). It returns 768-dimensional vectors, so vectors embedded with the old default are not comparable with it. A `SemanticMemoryStore` file records its own `embedding_model` and keeps using it, but a legacy file with no stored `embedding_model` now falls to the new default and must be re-embedded: name the old model explicitly to keep reading it, or re-embed.

The eval baselines (Run 006 and Run 007 in `EVALUATE.md`, measured on the 4b model) are not comparable with a run on another model. The registration of the `skills-disclosure` bench is voided because its manifests pin the sha256 of `constants.py` (it was registered, never run, and is not re-registered).

**Before**

```python
from fsm_llm.agents import SemanticMemoryStore

store = SemanticMemoryStore()  # embedded with ollama/qwen3-embedding:0.6b
```

**After**

```python
from fsm_llm.agents import SemanticMemoryStore
from fsm_llm.llm import DEFAULT_EMBEDDING_MODEL

assert DEFAULT_EMBEDDING_MODEL == "ollama/embeddinggemma-2:740m"
store = SemanticMemoryStore()  # the new default
assert store.to_dict()["embedding_model"] == DEFAULT_EMBEDDING_MODEL
legacy = SemanticMemoryStore(embedding_model="ollama/qwen3-embedding:0.6b")
assert legacy.to_dict()["embedding_model"] == "ollama/qwen3-embedding:0.6b"
```
