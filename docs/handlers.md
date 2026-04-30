# Handlers

Handlers extend FSM and term programs with custom logic at 8 lifecycle points. They validate data, call external APIs, log interactions, implement business rules, and trigger side effects. This page is the reference and a cookbook.

## At a glance

| Concept | Lives in | Role |
|---------|----------|------|
| `HandlerTiming` | `fsm_llm.handlers` | Enum of 8 timing points |
| `FSMHandler` (alias `Handler`) | same | Protocol every handler implements |
| `BaseHandler` | same | Canonical base class |
| `LambdaHandler` | same | Concrete impl built by `HandlerBuilder` |
| `HandlerBuilder` | same | Fluent builder; entry via `create_handler(name)` |
| `HandlerSystem` | same | Orchestrator: priority sort, error mode, timeout |
| `compose(term, handlers) -> Term` | same | Pure AST→AST splice for AST-side timings |

All names are **L2 COMPOSE** — importable directly from `fsm_llm`:

```python
from fsm_llm import (
    HandlerTiming, Handler, FSMHandler, BaseHandler,
    HandlerBuilder, HandlerSystem, create_handler, compose,
)
```

## The 8 timings

```python
class HandlerTiming(Enum):
    START_CONVERSATION = "start_conversation"   # host-side
    PRE_PROCESSING     = "pre_processing"       # AST-side (spliced via compose)
    POST_PROCESSING    = "post_processing"      # AST-side (spliced via compose)
    PRE_TRANSITION     = "pre_transition"       # host-side
    POST_TRANSITION    = "post_transition"      # host-side
    CONTEXT_UPDATE     = "context_update"       # host-side
    END_CONVERSATION   = "end_conversation"     # host-side
    ERROR              = "error"                # host-side
```

### AST-side vs host-side

Two timings (`PRE_PROCESSING`, `POST_PROCESSING`) are spliced into the compiled λ-term by `compose(term, handlers)`. Their handlers run via a real AST node (`Combinator(op=HOST_CALL, ...)`) — every Leaf they affect is visible to the planner and the executor's `oracle_calls` count.

The other six (`START_CONVERSATION`, `END_CONVERSATION`, `PRE_TRANSITION`, `POST_TRANSITION`, `CONTEXT_UPDATE`, `ERROR`) **dispatch host-side** through `HandlerSystem.execute_handlers` from `MessagePipeline` and `FSMManager`. They are intentionally invisible to `predicted_calls` because their semantics — handler-invocation cardinality, conditional firing, exception escapes, post-transition rollbacks — are out of scope for the planner. See `D-STEP-04-RESOLUTION` in `plans/plan_2026-04-27_1b5c3b2f/decisions.md` for the falsification record.

All eight still route through the same `make_handler_runner` callable, so execution semantics (priority, error mode, timeout, `should_execute`) are uniform.

## Building a handler — the fluent path

`HandlerBuilder` is the recommended path. Build one and pass it at construction time:

```python
from fsm_llm import HandlerBuilder, HandlerTiming, Program

audit = (
    HandlerBuilder("audit")
    .at(HandlerTiming.PRE_PROCESSING)
    .do(lambda **kw: log_event(kw))
    .build()
)

prog = Program.from_fsm("bot.json", handlers=[audit])
```

### Cookbook

```python
from fsm_llm import HandlerBuilder, HandlerTiming, create_handler

# Fire on entry to a specific state.
greet = (
    create_handler("greet")
    .at(HandlerTiming.START_CONVERSATION)
    .on_state("hello")
    .do(lambda **kw: {"greeting_at": time.time()})
    .build()
)

# Conditional firing (priority + when).
audit_high_priority = (
    create_handler("audit_priority")
    .at(HandlerTiming.PRE_TRANSITION)
    .when(lambda current_state, target_state, **_: target_state == "checkout")
    .priority(900)               # higher fires first; range [0, 1000]
    .do(lambda **kw: send_metric("checkout_attempt", kw["context"]))
    .build()
)

# Async execution.
import asyncio

async def fetch_recommendations(**kw):
    return {"recs": await my_rec_service(kw["context"]["user_id"])}

recs = (
    create_handler("recs")
    .at(HandlerTiming.CONTEXT_UPDATE)
    .do(fetch_recommendations)   # async lambda detected at build-time
    .build()
)
```

`HandlerBuilder.do(execution)` returns a `LambdaHandler` (the concrete subclass of `BaseHandler` produced from the lambda). `.build()` is invoked by `.do()` for ergonomics; you don't need to call it explicitly.

## Building a handler — the protocol path

For richer handlers, implement the protocol directly:

```python
from fsm_llm import BaseHandler, HandlerTiming

class RateLimitHandler(BaseHandler):
    def __init__(self, name: str, *, max_per_minute: int):
        super().__init__(name=name)
        self._max = max_per_minute
        self._timestamps: list[float] = []

    @property
    def priority(self) -> int:
        return 800

    def should_execute(self, timing, current_state, target_state, context, updated_keys):
        return timing == HandlerTiming.PRE_PROCESSING

    def execute(self, context):
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < 60]
        if len(self._timestamps) >= self._max:
            raise RuntimeError("rate limit exceeded")
        self._timestamps.append(now)
        return {}                # no context updates

prog = Program.from_fsm("bot.json", handlers=[RateLimitHandler("rate", max_per_minute=20)])
```

`should_execute(...)` is the gate — cheap predicate that decides whether `execute(...)` runs. `execute(...)` is the side-effecting body and returns a dict of context updates (or `{}`).

## Registering handlers

The canonical path is the `handlers=` constructor kwarg:

```python
prog = Program.from_fsm("bot.json", handlers=[h1, h2, h3])
prog = Program.from_term(t,         handlers=[h1])
prog = Program.from_factory(f, ...,  handlers=[h1])
```

Handlers passed at construction are sorted by priority (descending) and applied uniformly: `PRE_PROCESSING` and `POST_PROCESSING` are spliced into the compiled term via `compose(term, handlers)`; the other six are routed through `HandlerSystem.execute_handlers` from the FSM dispatch sites.

`Program.register_handler(h)` was removed at 0.7.0 (the I5 epoch closure — see [`migration_0.6_to_0.7.md`](migration_0.6_to_0.7.md)). The constructor `handlers=[...]` kwarg is the only supported path on `Program`. The lower-level `API.register_handler(h)` is still available on the dialog-side class for callers who reach for it directly.

## Direct `HandlerSystem` usage

If you need fine-grained control without `Program`:

```python
from fsm_llm import HandlerSystem, HandlerTiming, create_handler

system = HandlerSystem(error_mode="continue", handler_timeout=2.0)
system.register_handler(my_handler)

updates = system.execute_handlers(
    timing=HandlerTiming.PRE_PROCESSING,
    current_state="greet",
    target_state="ask",
    context={"user_id": "u-42"},
    updated_keys=set(),
)
# updates is the merged dict of all execute(...) returns
```

`error_mode` is one of `"continue"` (skip failed handlers) or `"raise"` (propagate). `handler_timeout` is per-handler in seconds; `None` means no timeout.

## Composing manually

For term programs, `compose(term, handlers)` is the public splice function:

```python
from fsm_llm import compose

new_term = compose(my_term, my_handlers)   # identity if handlers in (None, [])
```

`compose` applies eight splice functions in canonical order. Three are identity stubs (`START_CONVERSATION`, `END_CONVERSATION`, `ERROR` — host-side only). Five are structural rewrites for the AST-side timings.

## Built-in handler factories

`fsm_llm.handlers` exposes a few convenience constructors:

```python
from fsm_llm.handlers import create_handler

builder = create_handler("name")        # returns HandlerBuilder
```

`HandlerBuilder` shorthands:

| Method | Behavior |
|--------|----------|
| `.on_state(*ids)` | filter to `current_state in ids` |
| `.on_state_entry(state)` | sugar for `.at(START_CONVERSATION).on_state(state)` |
| `.on_state_exit(state)` | sugar for `.at(END_CONVERSATION).on_state(state)` |
| `.on_context_update(*keys)` | sugar for `.at(CONTEXT_UPDATE).when(any-of-keys-changed)` |
| `.when(predicate)` | arbitrary predicate over kwargs |
| `.priority(n)` | integer priority in `[0, 1000]` |

## Error semantics

A handler raising any exception:

- **`error_mode="continue"`** (default): the exception is logged via loguru, the handler is skipped, the system proceeds. Other handlers at the same timing still fire.
- **`error_mode="raise"`**: re-raised as `HandlerExecutionError(handler_name, original_error)` (a subclass of `HandlerSystemError(FSMError)`).

The `ERROR` timing fires whenever an exception escapes the dispatch site — gives you a hook for logging or alerting without affecting normal flow.

## Threading

`HandlerSystem` keeps a sorted list of handlers behind an `RLock`. `register_handler` is concurrency-safe. Per-conversation `RLock`s in `FSMManager` ensure that handlers attached to the same FSM run serially within a conversation.

## Best practices

1. **Keep `should_execute` cheap.** It runs on every dispatch; expensive checks belong in `execute`.
2. **Return only what you mean to update.** The dict returned by `execute(...)` is merged into the context; returning everything you read defeats the merge.
3. **Make handlers idempotent.** A retry on `error_mode="continue"` may re-fire some timings.
4. **Prefer pure functions for AST-side timings.** PRE/POST_PROCESSING handlers run inside the executor's reduction loop; side effects there are visible to the planner indirectly via Leaf re-counts. Keep them deterministic.
5. **Use `priority` deliberately.** The default is 500. Bump higher for handlers that gate (rate-limit, auth); drop lower for opportunistic ones (telemetry, fanout).
6. **Pass `handlers=[...]` at construction.** It's the only supported path on `Program` since 0.7.0. `Program.register_handler` was removed at that gate; the dialog-side `API.register_handler` is still available if you reach for the class directly.

## See also

- [`api_reference.md`](api_reference.md) §L2 COMPOSE — full type signatures.
- `src/fsm_llm/handlers.py` — single-file implementation; the section header `# R5 handler splicing` is the entry point for AST-side composition.
- `tests/test_fsm_llm/test_handlers_ast.py` and `tests/test_fsm_llm/test_handlers.py` — extensive coverage by example.
