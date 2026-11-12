# Testing Guide

> Covers FSM-LLM v0.12.0

`fsm_llm.testing` is the public home of the fakes and assertions the FSM-LLM repository uses for its own tests: fake LLM interfaces that need no network, a scripted user that drives a conversation, and four assertions that name a state, a context value, a visited path or a run result in one line. Use it to test a bot, an agent or a workflow without a model: a model call is slow, costs money and answers differently on each run.

Every Python block in this guide runs offline, top to bottom in one namespace, as part of the test suite (`tests/test_fsm_llm_testing/test_guide.py`). A block fenced `python notest` is shown only.

## Install and import

The subpackage ships in every install of the package; the `testing` extra adds no third-party dependency (it exists so the extra lists and `all` name every subpackage). Import it by name. `import fsm_llm` never loads it, and no library module imports it.

```python
from fsm_llm import API
from fsm_llm.testing import (
    PathRecorder,
    PromptGroundedLLM,
    ScriptedLLM,
    ScriptedUser,
    advance_until_terminal,
    assert_context,
    assert_path,
    assert_result,
    assert_state,
    final_reply,
    tool_call_reply,
)
```

It imports the core and `fsm_llm.codes` only, never `pytest`, `litellm` or another subpackage, so it works under pytest, unittest or a plain script. Importing it has no side effect: no clock, no randomness, no socket.

## A first test

A bot that asks a name, then an order, then ends. The model is a `ScriptedLLM`: the fields it extracts come from a dict and the reply of each state from a second dict, so the whole conversation is decided by the test.

```python
ORDER_BOT = {
    "name": "OrderBot",
    "description": "Take a name, then an order",
    "initial_state": "ask_name",
    "persona": "A polite waiter",
    "states": {
        "ask_name": {
            "id": "ask_name",
            "description": "Learn who is ordering",
            "purpose": "Get the customer's name",
            "extraction_instructions": "Extract the customer's name",
            "response_instructions": "Ask for the name",
            "required_context_keys": ["name"],
            "transitions": [
                {
                    "target_state": "ask_item",
                    "description": "The name is known",
                    "priority": 100,
                    "conditions": [
                        {
                            "description": "name is set",
                            "requires_context_keys": ["name"],
                            "logic": {"has_context": "name"},
                        }
                    ],
                }
            ],
        },
        "ask_item": {
            "id": "ask_item",
            "description": "Learn what to order",
            "purpose": "Get the item",
            "extraction_instructions": "Extract the item",
            "response_instructions": "Ask what to order",
            "required_context_keys": ["item"],
            "transitions": [
                {
                    "target_state": "done",
                    "description": "The item is known",
                    "priority": 100,
                    "conditions": [
                        {
                            "description": "item is set",
                            "requires_context_keys": ["item"],
                            "logic": {"has_context": "item"},
                        }
                    ],
                }
            ],
        },
        "done": {
            "id": "done",
            "description": "Terminal state",
            "purpose": "Confirm the order",
            "response_instructions": "Confirm the order",
        },
    },
}

REPLIES = {
    "ask_name": "Welcome! Who is ordering?",
    "ask_item": "What would you like?",
    "done": "Coming right up.",
}


def make_bot():
    llm = ScriptedLLM(
        extraction_data={"name": "Ada", "item": "tea"}, responses=REPLIES
    )
    return llm, API.from_definition(ORDER_BOT, llm_interface=llm)


llm, api = make_bot()
run = ScriptedUser(["I am Ada", "a tea please"]).run(api)

assert_path(run, ["ask_name", "ask_item", "done"])
assert_state(run, "done")
assert_context(run, name="Ada", item="tea")
llm.assert_exhausted()
```

`ScriptedUser.run` starts a conversation, sends the scripted messages and returns a frozen `ConversationRun`: `greeting`, `messages` (what was sent), `replies`, `states` (the path), `ended`, `final_state`, plus `api` and `conversation_id` for further queries.

Inside pytest the same lines go into a test function; a fixture builds the bot.

```python
import pytest


@pytest.fixture
def bot():
    return make_bot()


def test_the_order_is_taken(bot):
    llm, api = bot
    run = ScriptedUser(["I am Ada", "a tea please"]).run(api)
    assert_path(run, ["ask_name", "ask_item", "done"])
    llm.assert_exhausted()


test_the_order_is_taken(make_bot())
```

## The doubles

| Double | Use it when | What it does |
| --- | --- | --- |
| `ScriptedLLM` | You want to decide every answer: fields, the reply of each state, tool calls, errors | Fields from a dict, replies per state, a queue for `complete`; metered; an unscripted call is recorded and raised |
| `PromptGroundedLLM` | You want proof that the data reached the model | A fact `{field: (value, evidence)}` comes back only when the evidence text is in the request |
| `MockLLM2Interface` | You need a fixed answer and nothing else (the simplest, oldest double) | Every field from a dict, every reply one text; implements neither bulk extraction nor `complete` |

All three are plain classes, not `unittest.mock` objects: their attributes are public, may change between turns, and a test may subclass them. Pass one as `llm_interface=`.

`ScriptedLLM` keeps `requests` (the `complete` calls) and `call_history` (every call as `(method, request)`), so a test can read what the pipeline sent. The pipeline swallows an error raised during a field extraction, so a call nothing scripted would otherwise look like a model that found nothing. `ScriptedLLM` records it in `unexpected`, raises `UnscriptedCallError`, and `assert_exhausted()` fails on it and on a scripted reply that was never used. End a test with `llm.assert_exhausted()`.

```python
llm = ScriptedLLM([tool_call_reply(("lookup", {"city": "Rome"})), "It is sunny."])
api = API.from_definition(ORDER_BOT, llm_interface=llm)
# ... run the code under test with `api`, then:
try:
    llm.assert_exhausted()
except AssertionError as error:
    print("replies never used:", error)
```

`final_reply(text)`, `tool_call_reply(*(name, arguments))` and `malformed_reply()` build the three kinds of `CompletionResponse`; put one in the `replies` queue, or use a plain string for a final text and an exception instance to make that call fail. `state_in_prompt(prompt)` returns the state id a response prompt was written for, for a double of your own that answers per state. `hash_embed(dim=8)` is a pure embedder (`Callable[[str], list[float]]`): the same text gives the same vector and shared words give a closer one.

`PromptGroundedLLM` catches the bug a fixed dict hides, an extraction prompt that never shows the model what it must read:

```python
grounded = PromptGroundedLLM(
    facts={"name": ("Ada", "Ada"), "item": ("tea", "tea")},
    responses=REPLIES,
)
api = API.from_definition(ORDER_BOT, llm_interface=grounded)

lost = ScriptedUser(["hello there"]).run(api)
assert_path(lost, ["ask_name", "ask_name"])  # no evidence, no fact, no transition

api = API.from_definition(ORDER_BOT, llm_interface=grounded)
found = ScriptedUser(["I am Ada", "tea please"]).run(api)
assert_path(found, ["ask_name", "ask_item", "done"])
```

## The scripted user

`ScriptedUser(messages=(), *, rules=(), fallback=None, max_turns=25)`. The next message is the next scripted one; when they are spent, the answer of the first rule whose matcher fits the last reply (a regex or a callable, with one utterance or a sequence of them); then the `fallback`; then the run stops. It also stops when the conversation has ended or after `max_turns` messages (a stop, not an error). Exceptions of the API or the model propagate, so a test shows the traceback.

```python
llm = PromptGroundedLLM(
    facts={"name": ("Ada", "Ada"), "item": ("tea", "tea")}, responses=REPLIES
)
api = API.from_definition(ORDER_BOT, llm_interface=llm)
user = ScriptedUser(
    ["hi"],
    rules=[(r"Who is ordering", "I am Ada"), (r"What would you like", "tea")],
)
run = user.run(api)
assert_path(run, ["ask_name", "ask_name", "ask_item", "done"])
assert run.messages == ("hi", "I am Ada", "tea")
```

For an FSM that needs no user message (silent states, completion states, reasoning loops), drive it with `advance_until_terminal(api, *, max_steps)`; it returns the same record and raises `RunBudgetExceededError` when the budget is spent.

```python
CHAIN = {
    "name": "Chain",
    "description": "Two steps with no user",
    "initial_state": "first",
    "persona": "A test persona",
    "states": {
        "first": {
            "id": "first",
            "description": "Start",
            "purpose": "Begin",
            "response_instructions": "Say first",
            "transitions": [
                {"target_state": "last", "description": "Always", "priority": 100}
            ],
        },
        "last": {
            "id": "last",
            "description": "End",
            "purpose": "Finish",
            "response_instructions": "Say last",
        },
    },
}

llm = ScriptedLLM(responses={"first": "One.", "last": "Two."})
chain = advance_until_terminal(
    API.from_definition(CHAIN, llm_interface=llm), max_steps=5
)
assert_path(chain, ["first", "last"])
assert chain.ended
```

## The assertions

Four plain functions. Each raises `AssertionError` explicitly (never an `assert` statement, so `python -O` changes nothing), hides its own frames so pytest shows the failing line of your test, and names the first difference. `source` is a `ConversationRun` or an `(api, conversation_id)` pair.

| Assertion | Right when | Not for |
| --- | --- | --- |
| `assert_state(source, state_id)` | You care where the conversation is now (read live, so it follows later turns) | How it got there: use `assert_path` |
| `assert_context(source, **expected)` | Named keys must hold values; other keys are ignored (a subset check) | Internal `_` keys (not visible), or proving a key is absent |
| `assert_path(source, expected, *, exact=True)` | The route matters: a `ConversationRun`, a list of state ids or a sequence of `AdvanceResult`. `exact=False` accepts an ordered subsequence | A path of a conversation nobody recorded: add a `PathRecorder` |
| `assert_result(result, *, kind=None, status=None, reason=...)` | A completion state's `{kind, text, calls}`, a synthesis state's `{status, procedure_id, version, reason}` or a `CompletionResponse` | Plain FSM states |

Expected values may be `fsm_llm.codes` members or their strings; a typo raises `ValueError` listing the vocabulary instead of passing as a wrong guess. `reason=None` asserts the result carries no reason; leaving `reason` out checks nothing.

```python
llm, api = make_bot()
run = ScriptedUser(["I am Ada", "a tea please"]).run(api)

assert_path(run, ["ask_name", "done"], exact=False)  # a subsequence

try:
    assert_path(run, ["ask_name", "done"])
except AssertionError as error:
    print(error)  # names the first difference, with its index

assert_result(final_reply("ok"), kind="final")
assert_result({"status": "ok", "reason": None}, status="ok", reason=None)
try:
    assert_result({"status": "ok"}, status="okay")
except ValueError as error:
    print(error)  # lists the vocabulary
```

## PathRecorder

When the code under test drives the conversation (an agent, a runner), no `ConversationRun` exists. Register a `PathRecorder` as a handler and read the route back by conversation id.

```python
recorder = PathRecorder()
llm = ScriptedLLM(extraction_data={"name": "Ada", "item": "tea"}, responses=REPLIES)
api = API.from_definition(ORDER_BOT, llm_interface=llm, handlers=[recorder])

run = ScriptedUser(["I am Ada", "a tea please"]).run(api)
assert_path(recorder.path(run.conversation_id), ["ask_name", "ask_item", "done"])
```

It records the initial state and each transition target per conversation, and never changes a conversation. A stay is not a transition, so a recorder does not repeat the state; a `ConversationRun` does.

Caveat: registering ANY handler makes the pipeline deep-copy the context at every handler timing (with no handler it skips the copy). A context value that cannot be deep-copied, such as a lock, a socket or a client object, can therefore fail a turn that passed without the recorder. Test such code with `ConversationRun` and no handler.

## With pytest

The assertions need nothing from pytest. They build every message by hand, so `pytest.register_assert_rewrite("fsm_llm.testing")` is not needed. Add it, before the import, to a `conftest.py` only if you write your own helpers in a module that use plain `assert` and want pytest's introspection there:

```python notest
# conftest.py
import pytest

pytest.register_assert_rewrite("fsm_llm.testing")
```

The doubles never open a socket, so a test that uses them passes in a suite that blocks the network.

## Migrating from `tests.conftest`

Projects that copied `MockLLM2Interface` or `PromptGroundedLLM` out of this repository's `tests/conftest.py` can delete the copy. The classes are the same (the dead `transition_target` argument of `MockLLM2Interface`, stored and never read, is gone); the import changes.

```diff notest
-from tests.conftest import MockLLM2Interface, PromptGroundedLLM
+from fsm_llm.testing import MockLLM2Interface, PromptGroundedLLM
```

Inside this repository `tests/conftest.py` no longer exports the doubles at all (a guard test fails when a test imports them from there). Other copies, such as a private `_final(...)` reply helper or a queue-of-replies class, usually become `final_reply(...)` and `ScriptedLLM(replies)`; keep a private double when it encodes behaviour of its own, such as a provider-level fake.

## Limits

- The fakes stand in for a model; they say nothing about how a real model behaves. A passing scripted test proves your FSM, handlers and wiring, not the quality of your prompts.
- `ScriptedLLM` answers every field extraction from one dict, whatever the state asks for; use `PromptGroundedLLM` when the evidence matters.
- `PathRecorder` and the assertions read the conversation through the public `API`; a conversation closed while still open is read from the snapshot the API kept at the close.
