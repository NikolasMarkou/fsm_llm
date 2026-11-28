# FSM Design Guide

> Covers FSM-LLM v0.13.0

Best practices for designing effective Finite State Machines for conversational AI.

## Core Principles

### 1. Single Responsibility States

Each state should have one clear purpose:

```json
{
  "collect_email": {
    "id": "collect_email",
    "description": "Collect the user's email address",
    "purpose": "Collect and validate user's email address",
    "extraction_instructions": "Extract the user's email address",
    "response_instructions": "Ask for email, or confirm the one provided",
    "required_context_keys": ["email"],
    "transitions": [{"target_state": "collect_phone", "description": "Valid email provided"}]
  }
}
```

**Avoid** states that try to collect everything at once.

### 2. Clear Transition Logic

Make conditions explicit with JsonLogic:

```json
{
  "transitions": [
    {
      "target_state": "vip_service",
      "description": "High-value customer",
      "conditions": [{
        "description": "Lifetime value above 1000",
        "requires_context_keys": ["lifetime_value"],
        "logic": {">": [{"var": "lifetime_value"}, 1000]}
      }]
    },
    {"target_state": "standard_service", "description": "Default path"}
  ]
}
```

### 3. Graceful Error Handling

Always provide paths for error cases -- avoid dead-end states with no transitions.

### 4. Natural Language Cues

Write purposes that guide natural conversation:

```json
{
  "purpose": "Warmly greet the returning customer by name and ask how we can help",
  "response_instructions": "Use the customer's name and offer a personalized welcome",
  "required_context_keys": ["customer_name"]
}
```

## Design Patterns

### The Gatekeeper -- Authentication/Validation

```json
{
  "verify_identity": {
    "purpose": "Verify user identity with security questions",
    "required_context_keys": ["account_number", "security_answer"],
    "transitions": [
      {"target_state": "authenticated", "description": "Identity verified"},
      {"target_state": "auth_failed", "description": "Failed verification"}
    ]
  }
}
```

With no `conditions` and equal (default) priorities, both transitions pass and tie, so they are ambiguous and the LLM picks between them from the descriptions. (Two tied transitions to the same target are one candidate and need no choice. A tie of 15 or more distinct targets cannot be classified: the conversation stays in the state and `fsm-llm-validate` warns.) Among passing transitions the unique lowest `priority` value wins outright, so to make "Identity verified" deterministic, give each transition a condition with `requires_context_keys` and a `logic` so only one passes, or give it a lower `priority` than the fallback edge.

### The Collector -- Gathering Information

```json
{
  "shipping_address": {
    "purpose": "Collect complete shipping address",
    "extraction_instructions": "Extract street, city, state, and zip code",
    "required_context_keys": ["street", "city", "state", "zip"],
    "transitions": [{
      "target_state": "confirm_address",
      "description": "All fields collected",
      "conditions": [{
        "description": "Every address field is present",
        "requires_context_keys": ["street", "city", "state", "zip"],
        "logic": {"and": [
          {"has_context": "street"}, {"has_context": "city"},
          {"has_context": "state"}, {"has_context": "zip"}
        ]}
      }]
    }]
  }
}
```

The condition is what holds the state until the address is complete; `required_context_keys` alone only asks the pipeline to extract those keys.

### The Router -- Directing to Specialized Flows

```json
{
  "issue_classifier": {
    "purpose": "Understand the issue type and route appropriately",
    "transitions": [
      {"target_state": "technical_flow", "description": "Technical issue"},
      {"target_state": "billing_flow", "description": "Billing issue"},
      {"target_state": "general_inquiry", "description": "General question"}
    ]
  }
}
```

### The Confirmer -- Validating Understanding

```json
{
  "confirm_order": {
    "purpose": "Summarize order details and get final confirmation",
    "transitions": [
      {"target_state": "process_order", "description": "Confirmed"},
      {"target_state": "modify_order", "description": "Wants changes"}
    ]
  }
}
```

## Transition Field Reference

**`evaluation_priority`** on `TransitionCondition`: Controls evaluation order (default 100, range 0-1000). Lower values evaluated first -- useful for short-circuiting:

```json
{"conditions": [
  {"description": "Account suspended", "logic": {"==": [{"var": "account_status"}, "suspended"]}, "evaluation_priority": 10},
  {"description": "Account overdue", "logic": {"==": [{"var": "payment_status"}, "overdue"]}, "evaluation_priority": 50}
]}
```

**`llm_description`** on `Transition`: Optional string (max 300 chars) that customizes how a transition is presented to the LLM during ambiguous decisions. When omitted, `description` is used.

## Context Management

**Required vs optional**: `required_context_keys` lists the data a state should extract; it does **not** block a transition (an unconditional transition fires even when a listed key is missing). To hold a state until data exists, give the transition a condition with `requires_context_keys` and a `logic` such as `{"has_context": "email"}`; `fsm-llm-validate` warns about a required key no condition gates on.

**Progressive building**: Collect information gradually across states rather than all at once.

**Context scope**: `context_scope.read_keys` restricts which context keys a state's prompts see. `write_keys` is advisory: it is not enforced. Both must be lists of strings, and any other key in `context_scope` (for example a misspelt `read_key`) is a load error.

**Gate keys the user must not set**: a key a transition condition reads (`is_admin`, `is_verified`) can be written by a steered extraction unless you list it in the FSM-level `handler_only_keys`. Listed keys are dropped from the bulk and per-field extraction, in the state's own pass and in the pass after a transition, and stay writable by handlers, `update_context` and `initial_context`. It is opt-in and covers only the listed keys: an unlisted gate key is not protected, and a listed key that a `classification_extractions` entry owns stays classifiable (in Pass 1 and after a transition, since the classifier is the author's own channel for it). The `result_key` of a `completion`, `synthesize` or `call` state needs no listing: it is closed to every channel, classification included. `fsm-llm-validate` warns about a listed key no state references (usually a typo) and about a listed key that a `classification_extractions` entry writes.

**Corrections**: a later-turn value the LLM returns for an already-set key replaces it only when the pipeline extracted that key itself (handler-set and `update_context` values are never overwritten). A refused correction leaves the stored value in place, and the Pass-2 prompt then carries a `<rejected_corrections>` block so the reply does not claim the change was made. The block appears only when the user's message states the value as a whole word of at least 3 characters (`bored` does not ground `red`; a 1 or 2 character value such as `US` never does), and never for a value that ended up stored. A handler that normalises an extracted value on `CONTEXT_UPDATE` (`blue` -> `BLUE`) counts as a handler edit and turns later corrections of that key off (the stored value stands and the reply is told so).

## FSM Stacking Patterns

FSM stacking (`push_fsm`/`pop_fsm`) enables modular, composable design. A hand-written push is for a host that decides at run time; when the FSM itself should say "run that target now", use a `call` state (below), which pushes and pops the frame for you. In async code `await api.apush_fsm(...)` is the one push body and `push_fsm` its sync wrapper.

### When to Use

- **Shared sub-flows** -- A "collect address" FSM reused by checkout, returns, account setup
- **Progressive detail** -- High-level routing FSM pushes specialized FSMs per topic
- **Reasoning integration** -- The reasoning engine's orchestrator pushes one strategy FSM per solve through a `call` state (see below)

### Guidelines

1. Keep child FSMs self-contained -- they should work independently
2. Define clear context contracts via `context_to_pass` and `shared_context_keys`
3. Design terminal states (no outgoing transitions) to trigger `pop_fsm`
4. Choose merge strategies: `"update"` (overwrite parent) or `"preserve"` (only add new keys). Only keys named in `shared_context_keys` (or given as `return_context` / `context_to_return`) come back, whatever the strategy; a child's other extracted data is dropped on pop.

```python
api.push_fsm(conv_id, "address_form.json",
    context_to_pass={"flow": "checkout"},
    shared_context_keys=["user_id"])

api.pop_fsm(conv_id,
    context_to_return={"address_complete": True},
    merge_strategy="update")
```

## Classification-Based Routing

Classification is built into the core (`fsm_llm.Classifier`). Use it with `classification_extractions` on states for automatic intent-based routing, or use a lightweight classifier layer that pushes the appropriate FSM via stacking.

Inside a conversation the classifier uses the conversation's own LLM interface, so a custom `llm_interface` given to `API` also answers the classification calls. An entry that sets its own `model` gets a separate interface for that model (no connection settings are copied to it).

## Tool Calling and Structured Turns (`completion`)

A state may declare the optional `completion` field. Its Pass 1 is then one native tool-calling call (`tools`) or one JSON-schema call (`response_format_key`, an internal context key holding the response format), never both, over `[system(instructions)] + context[messages_key]`. The result `{kind, text, calls}` lands in `result_key` (default `completion_result`), and transitions route on `<result_key>.kind`:

```json
"transitions": [
  {"target_state": "run_tools", "description": "The model called tools", "priority": 10,
   "conditions": [{"description": "Calls", "logic": {"==": [{"var": "completion_result.kind"}, "calls"]}}]},
  {"target_state": "conclude", "description": "Final answer or malformed turn", "priority": 900}
]
```

Design rules:

- Give every result kind (`calls`, `final`, `malformed`; no `calls` without `tools`) a route, or end on an unconditional fallback edge; `fsm-llm-validate` warns otherwise.
- The transcript (`messages_key`, default `_completion_messages`) belongs to you: seed it with the task as a `user` message, and after running the calls append `fsm_llm.tool_exchange(text, calls, results)`. Core refuses an unpaired transcript and a tool-calling turn with no user message.
- The call runs only while `result_key` is unset: the handler that acts on a result clears it, otherwise the state makes no new call.
- A completion state has no other extraction (`field_extractions`, `classification_extractions`, `required_context_keys`, `extraction_instructions`) and cannot be terminal. Leave its `response_instructions` empty unless it should also speak.
- Core runs no tool. Run tools in a handler on the next state (the `native_fc` agent does exactly this).

## Procedures Written on Demand (`synthesize`)

A state may declare the optional `synthesize` field when a sub-conversation cannot be written ahead of time. The state carries a goal; a `fsm_llm.synthesis.SynthesizeExecutor` registered with `API(executors=...)` has a model write a child FSM for it, checks it, runs it on the host's model as a pushed child and merges back only the declared `return_keys`. The state itself is silent and routes on the result the executor writes to `synthesis_result`:

```json
"gather": {
  "id": "gather", "description": "Run a procedure written on demand", "purpose": "Collect the booking",
  "response_instructions": "",
  "synthesize": {"goal": "Book a table: ask the party size and the date.",
                 "inputs": ["venue"], "return_keys": ["party_size", "visit_date"], "on_failure": "sorry"},
  "transitions": [
    {"target_state": "done", "description": "It finished", "priority": 1,
     "conditions": [{"description": "ok", "requires_context_keys": ["synthesis_result"],
                     "logic": {"==": [{"var": "synthesis_result.status"}, "ok"]}}]},
    {"target_state": "sorry", "description": "It failed or was stopped", "priority": 2,
     "conditions": [{"description": "not ok", "requires_context_keys": ["synthesis_result"],
                     "logic": {"in": [{"var": "synthesis_result.status"}, ["failed", "aborted"]]}}]}
  ]
}
```

Design rules:

- Route on `synthesis_result.status` (`ok`, `failed`, `aborted`; `running` while it works). Every edge must read the result key, or it would fire before a result was written.
- Write the failure edge as `status in [failed, aborted]`. A procedure that spends its turn budget ends `aborted`, so a bare `== "failed"` would strand the host: the loader runs the edges on sample results and refuses it.
- The failure edge must also WIN. At the `failed` and the `aborted` sample the unique winner of the real ranking (the lowest `priority` number among the passing edges) must be the edge to `on_failure`: an edge with a lower number that also passes, or a second edge to another state at the same number (a tie goes to the classifier model), is refused. Let the failure edge read the result key only, and put any other gate (an attempt counter, a retry condition) on a retry edge with a LOWER priority number than the failure edge: the retry edge wins only when its gate passes, and the failure edge wins otherwise. A failure edge gated on another key is refused, because the loader fills every other key with a placeholder string.
- A retry that depends on a counter looks like this: the gate is on the retry edge, with a lower number than the failure edge, and the failure edge reads the result key only.

```json
"transitions": [
  {"target_state": "done", "description": "It finished", "priority": 1,
   "conditions": [{"description": "ok", "requires_context_keys": ["synthesis_result"],
                   "logic": {"==": [{"var": "synthesis_result.status"}, "ok"]}}]},
  {"target_state": "ask_again", "description": "It failed and fewer than 3 attempts were made", "priority": 2,
   "conditions": [{"description": "retry", "requires_context_keys": ["synthesis_result", "attempts"],
                   "logic": {"and": [{"==": [{"var": "synthesis_result.status"}, "failed"]},
                                     {"<": [{"var": "attempts"}, 3]}]}}]},
  {"target_state": "sorry", "description": "It failed or was stopped", "priority": 3,
   "conditions": [{"description": "not ok", "requires_context_keys": ["synthesis_result"],
                   "logic": {"in": [{"var": "synthesis_result.status"}, ["failed", "aborted"]]}}]}
]
```

- To treat `aborted` (a spent budget or a cancel) differently from `failed`, do not give it an edge of its own beside the failure edge: an edge `status == aborted` to another state with a lower priority number than the failure edge wins at the `aborted` sample, and the loader refuses the state. Make `on_failure` a router state instead, and know what it costs. The synthesize state keeps the one failure edge, `status in [failed, aborted]` to the router, and the router state declares its own edges on the same result key, one for `aborted` and one for `failed`, each to its own reply state. A transition runs on the turn after the state is entered, and the router is silent (empty `response_instructions`), so after a failure the conversation stops in the router and the next turn, a user message or an `advance`, gets an empty reply before it moves on to the reply state. The single failure edge `status in [failed, aborted]` straight to a reply state has no such turn, and it sends both outcomes to the same state. Choose the router only when `aborted` needs a route of its own:

```json
"sorry": {
  "id": "sorry", "description": "Tell a stop from a failure", "purpose": "Route the failure",
  "response_instructions": "",
  "transitions": [
    {"target_state": "timed_out", "description": "It was stopped", "priority": 1,
     "conditions": [{"description": "aborted", "requires_context_keys": ["synthesis_result"],
                     "logic": {"==": [{"var": "synthesis_result.status"}, "aborted"]}}]},
    {"target_state": "failed_reply", "description": "It failed", "priority": 2,
     "conditions": [{"description": "failed", "requires_context_keys": ["synthesis_result"],
                     "logic": {"==": [{"var": "synthesis_result.status"}, "failed"]}}]}
  ]
}
```

- `on_failure` names the state the failure edge reaches, one of the state's own transition targets. Give that state a reply that apologises or offers an alternative, because a failure is a normal outcome, not an exception.
- Name what the procedure may use (`inputs`) and what it must return (`return_keys`) with plain key names. Do not use a handler-only key as a return key.
- The state declares no extraction and no `completion`. Core only loads the field; an engine that would run it needs a `SynthesizeExecutor`: an `API`, `FSMManager` or `MessagePipeline` with none refuses the definition with `MissingExecutorError` (at construction, `push_fsm`, `fork` and `resume` for an `API`, at the first turn for a bare manager), so register the executor on the `API` and drive the host with the core loops. `fsm-llm-validate` and `load_fsm_definition` accept the definition without one.
- The result is skip-if-set. A state that runs again on re-entry must clear `synthesis_result` first (`clear_keys_on_entry`).

The guide is [synthesis.md](./synthesis.md); the reference is the Synthesis section of [api_reference.md](./api_reference.md).

## Running a Named Target From a State (`call`)

A state may declare the optional `call` field (definition version 4.3, the default) when what it needs already exists as an [executable](./executables.md): a document that wraps an FSM, or a built-in strategy. The state names the target by id (`target`), or picks one from a closed list by a context value (`choose`), says which context keys the target receives (`inputs`), which of its outputs come back (`returns`, `{parent_key: child_output}`) and which state takes over when it fails (`on_failure`). The engine pushes the target as a child frame, pops it when it ends and merges only the declared `returns`, and the state routes on the result it writes to `call_result`. The state itself makes no model call and is silent.

```json
"work": {
  "id": "work", "description": "Run the greeter", "purpose": "Greet the person",
  "response_instructions": "",
  "call": {"target": "support/greeter", "inputs": ["name"],
           "returns": {"greeting": "greeting"}, "on_failure": "sorry", "max_steps": 5},
  "transitions": [
    {"target_state": "done", "description": "It returned", "priority": 1,
     "conditions": [{"description": "ok", "requires_context_keys": ["call_result"],
                     "logic": {"==": [{"var": "call_result.status"}, "ok"]}}]},
    {"target_state": "sorry", "description": "It failed or was stopped", "priority": 2,
     "conditions": [{"description": "not ok", "requires_context_keys": ["call_result"],
                     "logic": {"in": [{"var": "call_result.status"}, ["failed", "aborted"]]}}]}
  ]
}
```

The `API` needs a target resolver, because the definition only holds ids. `ExecutableRegistry` is one:

```python
from fsm_llm import API
from fsm_llm.engine.executables import ExecutableRegistry

registry = ExecutableRegistry.from_paths(["executables/"])      # holds support/greeter
api = API.from_file("host.json", llm_interface=llm, targets=registry)
conversation_id, _ = api.start_conversation({"name": "Ada"})
api.run_until_terminal(conversation_id, max_steps=10)
api.get_data(conversation_id)["call_result"]["status"]          # "ok"
```

Design rules:

- Precondition: `API(targets=...)`, keyword-only and internal like `executors=` (the builders have no setter). A bare `API` refuses a definition that holds a `call` state with `MissingExecutorError` naming the resolver; the loader and `fsm-llm-validate` accept it. Frame mode runs targets of kind `fsm`, `strategy` and `procedure`; any other id, a pattern or a workflow resolves to nothing and the call ends `failed/unknown_target`.
- Route on `call_result.status` (`ok`, `failed`, `aborted`; `running` while it works) and write the failure edge as `status in [failed, aborted]`, as for `synthesize`: every edge must read the result key and the loader refuses an edge that could fire before a result exists. The failure edge must also win outright at the `failed` and the `aborted` sample (no earlier-priority edge passes there and no tie); put a retry gate on an edge with a lower priority number. An edge of its own for `aborted` (`status == aborted` to another state, ahead of the failure edge) is refused for the same reason: make `on_failure` a router state whose own edges branch on `call_result.status` for `aborted` and `failed`, as in the synthesize section above. The router costs one turn: it is silent and a transition runs on the turn after the state is entered, so after a failure the conversation stops in the router and the next turn gets an empty reply before it moves on; the single failure edge `status in [failed, aborted]` straight to a reply state has no such turn.
- The child receives only the declared `inputs` (copied, never the parent's whole context, an internal key or an approval grant) and the parent receives only the declared `returns`. Both lists hold public key names; the loader refuses an internal-prefixed or secret-looking name. A declared return the child did not write ends `failed/return_keys_missing` and merges nothing. A `strategy` target that declares no `returns` hands back every key of its own `results` map that the child wrote.
- `max_steps` and `max_seconds` are the child frame's lifetime budget. A child that spends it ends `aborted/budget_spent` with no merge, and the parent takes the failure edge. The other reasons are `unknown_target`, `inputs_missing`, `depth_refused`, `push_failed` and `provider_error` (the child's greeting could not be generated).
- `choose` takes `key` (the context key that holds a target id), a closed non-empty `targets` list and an optional `fallback` from that list. A value outside the list takes the fallback, else the call ends `failed/unknown_target`. Write the key from a handler or an earlier state. If a user message must never pick the target, list the key in `handler_only_keys` (the reasoning orchestrator does so for `classified_problem_type`). `targets_from {capability}` records where the list came from and is never resolved by core.
- The state runs when a transition enters it, in that turn; the engine pushes the child at the end of that turn, so the next message (or the next message-free step) goes to the child. Core pops the child at its end or budget stop, the parent stays in the call state, and the next parent turn takes the exit edge. `run_until_terminal` does all of this for you; a stream does it at its start and at its normal end.
- The result key is skip-if-set per call state. A call state revisited after a final `call_result` of its own state does not call again unless the state clears the key on entry (`clear_keys_on_entry`).
- Known limits: a target of kind `procedure` runs with no lifecycle check, so do not list a model-written procedure in a `choose` list an untrusted value can reach; a child ended from outside (`end_conversation` on the child) leaves the parent stuck until the host ends the root; delegated mode (a built-in pattern as a polled child run on a daemon thread, with `API(targets=ExecutableTargets(registry, context))`) is built for patterns only, with its limits stated in [Executables](executables.md): a handle the host never saw (after a fork, a restore or a restart) ends `failed/run_lost` and nothing is re-run; a deadline or a cancel only detaches a pattern that may keep running; a delegated parent spends one step per poll and `aadvance` waits at most 0.5 seconds once; `max_seconds` defaults to 600 and is capped at 3600; each merged return is capped at 8000 characters and the pseudo output `answer` reads the pattern's answer; a `workflow` target runs delegated like a pattern, only when the host's `ExecutableTargets` context binds its `requires.callables` (otherwise the call ends `failed/unknown_target`), and has no persisted instance (roadmap E5.3 carried); the mark that owes `run/end` is in memory.

The reference is the Call state section of [api_reference.md](./api_reference.md).

## Designing for Agents

Agent patterns (`fsm_llm.agents`) are FSMs too, and most are JSON files (`fsm_llm/agents/patterns/`, kind `pattern`, see [Executables](executables.md)): a static template whose tool section is filled from the run's tool registry, and a list of handlers the agent binds to its states. The core ReAct loop is a 3-4 state FSM: **Think -> Act -> Observe -> Conclude**. Tool execution happens via handlers, not state instructions.

Design rules follow from the pattern files. The `description` of an agent FSM is static text, never the caller's task: the task is an input of the run (`task`), so an agent definition keeps one `fsm_id` however many tasks it runs, and any text a caller supplies reaches the model as data, not as definition text. (These still depend on the caller: the key names of its `initial_context`, the tool section when the registry is a `SemanticToolRegistry`, and the think prompt of `parallel_react`; see the known limits in [Executables](executables.md#known-limits).) An FSM that must change shape with its input, such as a chain with one state per step, is a generator: a named function from typed options (`agents.prompt_chain`), not caller text spliced into a template.

For agent-style flows that accumulate intermediate results across turns, `WorkingMemory` (`fsm_llm.WorkingMemory`) provides named buffers (`core`, `scratch`, `environment`, `reasoning`) so tool outputs and intermediate reasoning are organized rather than flattened into one context dict. Note the reach: buffer data is merged into the Pass-1 per-field extraction prompt only (default `context_keys`), never into the Pass-2 response prompt, which reads `context.data` alone; put anything the response model must see in `context.data` (via `update_context` or a handler). Intermediate states can also set an empty `response_instructions` to skip the Pass-2 response LLM call entirely (see the 2-pass notes), which is the common pattern for tool-dispatch states inside a ReAct loop.

## Anti-Patterns to Avoid

| Anti-Pattern | Problem | Fix |
|-------------|---------|-----|
| Megazord State | One state handles everything | Split into single-responsibility states |
| Dead End | Error state with no transitions out | Add recovery/retry transitions |
| Infinite Loop | State transitions back to itself | Add max retry logic or a fallback transition |
| Context Blackhole | Too many `required_context_keys` at once | Collect progressively across states |

## Testing FSMs

```python
# Validate structure
from fsm_llm import validate_fsm_from_file
result = validate_fsm_from_file("my_fsm.json")  # checks reachability, missing transitions
```

Or from the command line:

```bash
fsm-llm-validate --fsm my_fsm.json
```

Test all paths, context flow, and edge cases (empty input, very long input, off-topic).

---

**Next:** [Handler Development](./handlers.md)
