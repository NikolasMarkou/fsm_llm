# Core renames

Covers FSM-LLM 0.12.0 to 0.13.0. Five public names of the core package changed so that one concept has one spelling everywhere. None has an alias, so a caller that used the old name fails at once with an `ImportError`, a `TypeError` or an `AttributeError`; nothing is silently ignored. Each section below shows the code before and after.

The old module paths in the **Before** blocks (`fsm_llm.utilities`, `fsm_llm.visualizer`) moved again in this release. The **After** blocks use the new locations.

## Classifier takes llm_interface instead of llm

`Classifier(llm=)` and `HierarchicalClassifier(llm=)` are renamed `llm_interface=`, the spelling of every other core entry point (`API`, `MessagePipeline`, the agents). Behaviour is unchanged. Both constructors also take `**llm_kwargs` (connection settings for a private interface), so a leftover `llm=x` would have been swallowed and a private provider interface built that ignores `x`. The old keyword is therefore refused with a `TypeError` that names `llm_interface`.

**Before**

```python
from fsm_llm import Classifier

classifier = Classifier(schema, llm=my_interface)
```

**After**

```python
from fsm_llm import (
    ClassificationSchema,
    Classifier,
    IntentDefinition,
    LiteLLMInterface,
)

schema = ClassificationSchema(
    intents=[
        IntentDefinition(name="buy", description="wants to purchase"),
        IntentDefinition(name="browse", description="just looking"),
    ],
    fallback_intent="browse",
)
my_interface = LiteLLMInterface(model="ollama_chat/example")
classifier = Classifier(schema, llm_interface=my_interface)
```

`FSMSynthesizer(llm=)` and `Simulator(llm)` in `fsm_llm.synthesis` are unchanged.

## The path parameter is now path

The path parameter of `load_fsm_from_file` (was `file_path`), of `read_json_object` (was `file_path`) and of the package-level `quick_start` (was `fsm_file`) is now `path`, the spelling of `API.from_file`. Behaviour is unchanged. A caller that passed the old keyword gets a `TypeError`; a positional call is not affected. `load_fsm_definition(fsm_id_or_path)` keeps its name, and so does the `fsm_file` field of the workflows `ConversationStep`.

**Before**

```python
from fsm_llm.utilities import load_fsm_from_file

definition = load_fsm_from_file(file_path="bot.json")
```

**After**

```python
import json
from pathlib import Path

from fsm_llm import load_fsm_from_file, quick_start

Path("bot.json").write_text(
    json.dumps(
        {
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
    )
)
definition = load_fsm_from_file(path="bot.json")
api = quick_start(path="bot.json", model="ollama_chat/example")
```

`read_json_object` now lives in `fsm_llm.core.loaders`, next to `load_fsm_from_file`.

## The visualizer text helpers are build functions

The seven text helpers of the old `fsm_llm.visualizer` are renamed from `create_*` to `build_*`: `create_fancy_header`, `create_metadata_section`, `create_persona_section`, `create_states_section`, `create_transitions_section`, `create_legend` and `create_state_boxes` are now `build_fancy_header`, `build_metadata_section`, `build_persona_section`, `build_states_section`, `build_transitions_section`, `build_legend` and `build_state_boxes`. They build text and return it, like `build_graph_representation`; `create_*` stays for the functions that make an object. None is in `__all__`, but each was importable from the old module. They now live in `fsm_llm.core.visualize.ascii_render` (and `build_state_boxes` in `fsm_llm.core.visualize.ascii_layout`).

**Before**

```python
from fsm_llm.visualizer import create_fancy_header, create_legend

header = create_fancy_header("My Bot")
legend = create_legend(False, True)
```

**After**

```python
from fsm_llm.core.visualize.ascii_render import build_fancy_header, build_legend

header = build_fancy_header("My Bot")
legend = build_legend(False, True)
```

## ContextFilterWorkError is now ContextFilterError

`ContextFilterWorkError` is renamed `ContextFilterError` and now subclasses `FSMError` as well as `ValueError`, like `BuildError`. It is the refusal `filter_context_tree` raises for a cyclic context whose aliasing would unfold past its work ceiling (`get_data`, `save_session`, the extracted-data commit, the monitor display filter). `except ValueError` keeps catching it, and `except FSMError` now does too. The class was never in `fsm_llm.__all__`, so only a caller that imported it from the old `fsm_llm.utilities` is affected. `API` behaviour is unchanged.

**Before**

```python
from fsm_llm.utilities import ContextFilterWorkError

try:
    data = api.get_data(conversation_id)
except ContextFilterWorkError:
    data = {}
```

**After**

```python
from fsm_llm import FSMError
from fsm_llm.context.filtering import ContextFilterError

assert issubclass(ContextFilterError, FSMError)
assert issubclass(ContextFilterError, ValueError)


def read_data(api, conversation_id):
    try:
        return api.get_data(conversation_id)
    except ContextFilterError:
        return {}
```

## HandlerSystem takes handler_error_mode

`HandlerSystem(error_mode=)` is renamed `handler_error_mode=`, and its attribute `HandlerSystem.error_mode` is now `HandlerSystem.handler_error_mode`. The value is a `HandlerErrorMode`, which still compares equal to `"continue"` and `"raise"`. `API`, `FSMManager` and the builders already used that name, so every entry point now spells it one way. A `HandlerSystem(error_mode=...)` call raises `TypeError`, and the `ValueError` for an unknown value now reads `Invalid handler_error_mode: ...`.

**Before**

```python
from fsm_llm.handlers import HandlerSystem

handlers = HandlerSystem(error_mode="raise")
print(handlers.error_mode)
```

**After**

```python
from fsm_llm import HandlerErrorMode
from fsm_llm.handlers import HandlerSystem

handlers = HandlerSystem(handler_error_mode=HandlerErrorMode.RAISE)
assert handlers.handler_error_mode == "raise"
```
