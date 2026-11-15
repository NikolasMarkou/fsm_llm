# Definition version

Covers FSM-LLM 0.12.0 to 0.13.0. The default `version` of an FSM definition is now `"4.2"`. A definition that omits `version` is not affected. A definition that declares `"version": "4.1"` in its JSON now gets a different `fsm_id`, so this guide matters to anyone who keeps FSM files, saved sessions or stored procedures that carry an explicit `"4.1"`.

## Default version change and fsm_id

The default definition `version` is now `"4.2"` (it was `"4.1"`), the version that introduced the `synthesize` state field. `FSMDefinition.version` and `FSMArtifactBuilder.to_dict()` follow the new default. The `fsm_id` is a hash of `model_dump(exclude_defaults=True)`, so a version equal to the default is not part of it:

- a definition that omits `version`, or declares `"4.2"`, keeps the `fsm_id` and the library digest that 0.12.0 computed for it;
- a definition that declares `"version": "4.1"` explicitly now hashes its version, so it gets a new `fsm_id` and a new library digest (in 0.12.0 `"4.1"` was the default and was not hashed);
- definitions that declare `"3.0"` or `"4.0"` already hashed their version and do not move.

What a new `fsm_id` does:

- a session saved under the old id only logs the existing id-mismatch WARNING when it is restored; the restore still happens;
- a stored procedure whose definition declares `"4.1"` is stored again under its new digest (a duplicate, not a break);
- anything of yours that keys data on `API.fsm_id` sees a new key.

The `examples/` folder is not edited: eight of its files declare `"4.1"` and so get a new id.

To keep the 0.12.0 id, drop the version from the file or change it to `"4.2"`. Both give the id 0.12.0 computed for the same content. `fsm-llm-validate --upgrade` does the second for a file that declares `4.0` or `4.1`, after validating it. Note that a file that declared `"4.0"` hashed its version in 0.12.0, so for that file the id moves either way.

**Before**

```python
from fsm_llm import API

definition = {
    "name": "Bot",
    "description": "A one-state bot",
    "version": "4.1",
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
api = API.from_definition(definition, model="gpt-4o-mini")
print(api.fsm_id)  # the same id as a definition without a version
```

**After**

```python
from fsm_llm import API

model = "ollama_chat/example"
omitted = {
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
explicit_41 = {**omitted, "version": "4.1"}
explicit_42 = {**omitted, "version": "4.2"}

omitted_id = API.from_definition(omitted, model=model).fsm_id
assert API.from_definition(explicit_42, model=model).fsm_id == omitted_id
assert API.from_definition(explicit_41, model=model).fsm_id != omitted_id
```

To rewrite a file in place from the command line:

```bash
fsm-llm-validate --fsm bot.json --upgrade
```
