# Handler Development Guide

Handlers extend FSM behavior with custom logic at 8 lifecycle points. They can validate data, call external APIs, log interactions, implement business rules, and trigger side effects.

## Basic Structure

```python
def my_handler(context: dict) -> dict:
    """Handlers receive context, return updates to merge back."""
    return {"new_key": "new_value"}
```

## Handler Timing Points

### START_CONVERSATION -- Conversation begins
```python
def welcome(context):
    user_id = context.get("user_id")
    return {"preferences": load_prefs(user_id)} if user_id else {}

api.register_handler(
    api.create_handler("Welcome").at(HandlerTiming.START_CONVERSATION).do(welcome))
```

### PRE_PROCESSING -- Before LLM processes input
Good for input sanitization, PII detection, pre-validation.

### POST_PROCESSING -- After Pass 1 (extraction + transition), before Pass 2 (response)
Good for entity extraction, context enrichment, analytics.

### CONTEXT_UPDATE -- When specific context keys change
```python
api.register_handler(
    api.create_handler("EmailValidator")
        .on_context_update("email")  # shorthand for .at(CONTEXT_UPDATE).when_keys_updated("email")
        .do(validate_email))
```

### PRE_TRANSITION / POST_TRANSITION -- Before/after state change
```python
api.register_handler(
    api.create_handler("OnCheckout")
        .on_state_entry("checkout")  # shorthand for .at(POST_TRANSITION).on_target_state("checkout")
        .do(check_inventory))
```

### END_CONVERSATION -- Conversation ends
Good for resource cleanup, saving summaries, releasing locks.

### ERROR -- On any exception
Good for graceful recovery, fallback responses, error logging.

## HandlerBuilder Reference

The fluent API returned by `api.create_handler(name)`:

| Method | Description |
|--------|-------------|
| `.at(*timings)` | Specify one or more HandlerTiming values |
| `.on_state(*states)` | Execute only in these states |
| `.not_on_state(*states)` | Exclude these states |
| `.on_target_state(*states)` | Execute only when transitioning TO these states |
| `.not_on_target_state(*states)` | Exclude transitions TO these states |
| `.when_context_has(*keys)` | Require these context keys to exist |
| `.when_keys_updated(*keys)` | Execute when these keys change |
| `.on_state_entry(*states)` | Shorthand: `.at(POST_TRANSITION).on_target_state()` |
| `.on_state_exit(*states)` | Shorthand: `.at(PRE_TRANSITION).on_state()` |
| `.on_context_update(*keys)` | Shorthand: `.at(CONTEXT_UPDATE).when_keys_updated()` |
| `.when(condition)` | Custom lambda: `(timing, state, target, ctx, keys) -> bool` |
| `.with_priority(n)` | Execution priority (lower runs first, default 100) |
| `.do(fn)` | Set handler function and build |

## Error Handling

The `HandlerSystem` has an `error_mode`:
- **`"continue"`** (default) -- Log error, skip failed handler, continue with others
- **`"raise"`** -- Stop execution, propagate exception

Handlers marked `critical=True` (via `BaseHandler` subclass) always raise regardless of error mode:

```python
class PaymentHandler(BaseHandler):
    def __init__(self):
        super().__init__(name="PaymentHandler", critical=True)

    def should_execute(self, timing, current_state, target_state, context, updated_keys):
        return timing == HandlerTiming.POST_TRANSITION and current_state == "process_payment"

    def execute(self, context):
        return process_payment(context)  # failure always propagates
```

## Patterns

### External API Calls

Handlers execute synchronously. Use `requests` or `concurrent.futures` for I/O:

```python
def check_inventory(context):
    product_id = context.get("selected_product_id")
    if not product_id:
        return {}
    try:
        resp = requests.get(f"https://api.store.com/inventory/{product_id}", timeout=10)
        if resp.ok:
            data = resp.json()
            return {"in_stock": data["quantity"] > 0, "quantity": data["quantity"]}
    except requests.RequestException:
        pass
    return {"inventory_check_failed": True}

api.register_handler(
    api.create_handler("Inventory").on_state_entry("product_selection").do(check_inventory))
```

### Conditional Execution

```python
api.register_handler(
    api.create_handler("PremiumBenefits")
        .at(HandlerTiming.POST_TRANSITION)
        .when(lambda t, s, tgt, ctx, keys: ctx.get("user_tier") == "premium")
        .do(lambda ctx: {"discount_rate": 0.20, "free_shipping": True}))
```

### Validation Chain

```python
api.register_handler(
    api.create_handler("AgeValidator")
        .on_context_update("age")
        .with_priority(10)
        .do(validate_age))

api.register_handler(
    api.create_handler("ConsentValidator")
        .on_context_update("age_group", "parental_consent")
        .with_priority(20)
        .do(validate_consent))
```

### Exclusion Filters

```python
# Run on every state EXCEPT "error" and "done"
api.register_handler(
    api.create_handler("GlobalLogger")
        .at(HandlerTiming.POST_PROCESSING)
        .not_on_state("error", "done")
        .do(lambda ctx: {"logged": True}))
```

## Best Practices

1. **Keep handlers focused** -- One handler, one responsibility
2. **Handle errors gracefully** -- Always expect failures in external calls
3. **Choose the right timing** -- Use the most specific timing point
4. **Use priority for ordering** -- Lower numbers run first
5. **Avoid side effects in validators** -- Pure validation, no database writes
6. **Test handlers in isolation** -- They're just functions that take and return dicts

## Testing

```python
def test_email_validator():
    assert validate_email({"email": "user@example.com"})["email_valid"] is True
    assert validate_email({"email": "invalid"})["email_valid"] is False
    assert validate_email({})["email_valid"] is False

def test_handler_execution_order():
    order = []
    api.register_handler(api.create_handler("H1").at(HandlerTiming.PRE_PROCESSING)
        .with_priority(10).do(lambda ctx: order.append("H1") or {}))
    api.register_handler(api.create_handler("H2").at(HandlerTiming.PRE_PROCESSING)
        .with_priority(20).do(lambda ctx: order.append("H2") or {}))
    conv_id, _ = api.start_conversation()
    api.converse("test", conv_id)
    assert order == ["H1", "H2"]
```

## Handlers in Extension Packages

- **Agents**: POST_TRANSITION executes tools, PRE_TRANSITION enforces budgets, CONTEXT_UPDATE checks HITL approval
- **Reasoning**: Validation, tracing, context pruning, retry limiting handlers
- **Monitor**: Registers at all 8 timing points (priority 9999) as pure observers -- returns `{}`, never modifies state
- **Workflows**: Manages step execution directly via async pipeline; ConversationStep has its own handler system
