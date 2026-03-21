# Handler Development - Build Custom Handlers

Handlers are the bridge between your FSM's conversation flow and external systems, business logic, and data processing. This guide shows you how to build powerful handlers for any use case.

## Table of Contents

1. [Understanding Handlers](#understanding-handlers)
2. [Handler Timing Points](#handler-timing-points)
3. [Building Your First Handler](#building-your-first-handler)
4. [Advanced Handler Patterns](#advanced-handler-patterns)
5. [Handler Best Practices](#handler-best-practices)
6. [Real-World Handler Examples](#real-world-handler-examples)
7. [Testing Handlers](#testing-handlers)
8. [Performance Optimization](#performance-optimization)

## Understanding Handlers

Handlers are functions that execute at specific points during FSM execution. They can:
- Validate and transform data
- Call external APIs
- Log interactions
- Implement business rules
- Trigger side effects

### Basic Handler Structure

```python
def my_handler(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Args:
        context: Current conversation context

    Returns:
        Dictionary of updates to merge into context
    """
    # Your logic here
    return {"new_key": "new_value"}
```

## Handler Timing Points

Handlers can execute at different stages of the conversation lifecycle:

### 1. **START_CONVERSATION**
Executes when a new conversation begins.

```python
def welcome_handler(context):
    """Initialize conversation with user preferences"""
    user_id = context.get("user_id")
    if user_id:
        # Load user preferences from database
        preferences = load_user_preferences(user_id)
        return {"preferences": preferences, "returning_user": True}
    return {"returning_user": False}

api.register_handler(
    api.create_handler("WelcomeHandler")
        .at(HandlerTiming.START_CONVERSATION)
        .do(welcome_handler)
)
```

### 2. **PRE_PROCESSING**
Executes before the LLM processes user input.

```python
def sanitize_input(context):
    """Clean and validate user input"""
    user_input = context.get("_user_input", "")

    # Remove sensitive data
    sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', user_input)

    # Add input metadata
    return {
        "_sanitized_input": sanitized,
        "_input_length": len(user_input),
        "_contains_pii": user_input != sanitized
    }

api.register_handler(
    api.create_handler("InputSanitizer")
        .at(HandlerTiming.PRE_PROCESSING)
        .do(sanitize_input)
)
```

### 3. **POST_PROCESSING**
Executes after LLM response but before state transition.

```python
def extract_entities(context):
    """Extract entities from conversation"""
    user_input = context.get("_user_input", "")

    # Extract email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_input)
    if email_match:
        return {"detected_email": email_match.group(0)}

    # Extract phone
    phone_match = re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', user_input)
    if phone_match:
        return {"detected_phone": phone_match.group(0)}

    return {}

api.register_handler(
    api.create_handler("EntityExtractor")
        .at(HandlerTiming.POST_PROCESSING)
        .do(extract_entities)
)
```

### 4. **CONTEXT_UPDATE**
Executes when specific context keys are updated.

```python
def validate_email(context):
    """Validate email format when collected"""
    email = context.get("email", "")

    if re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
        domain = email.split('@')[1]
        return {
            "email_valid": True,
            "email_domain": domain,
            "email_corporate": domain not in ['gmail.com', 'yahoo.com', 'hotmail.com']
        }
    else:
        return {"email_valid": False}

api.register_handler(
    api.create_handler("EmailValidator")
        .at(HandlerTiming.CONTEXT_UPDATE)
        .when_keys_updated("email")
        .do(validate_email)
)
```

### 5. **PRE_TRANSITION**
Executes before state transition.

```python
def log_transition(context):
    """Log state transitions for analytics"""
    current_state = context.get("_current_state")
    target_state = context.get("_target_state")

    logger.info(f"Transition: {current_state} -> {target_state}")

    # Track in analytics
    track_event("state_transition", {
        "from": current_state,
        "to": target_state,
        "user_id": context.get("user_id")
    })

    return {"last_transition": f"{current_state}->{target_state}"}
```

### 6. **POST_TRANSITION**
Executes after successful state transition.

```python
def check_completion(context):
    """Check if key milestones are reached"""
    current_state = context.get("_current_state")

    if current_state == "order_complete":
        # Send confirmation email
        send_order_confirmation(
            email=context.get("email"),
            order_id=context.get("order_id")
        )
        return {"confirmation_sent": True}

    return {}
```

### 7. **ERROR**
Executes when an error occurs.

```python
def error_recovery(context):
    """Handle errors gracefully"""
    error = context.get("error", {})
    error_type = error.get("type")

    if error_type == "APIError":
        return {
            "_fallback_response": "I'm having trouble connecting to our systems. Let me try another way...",
            "_retry_count": context.get("_retry_count", 0) + 1
        }

    return {"_error_logged": True}
```

### 8. **END_CONVERSATION**
Executes when a conversation ends (either by reaching a terminal state or explicit cleanup).

```python
def cleanup_session(context):
    """Clean up resources when conversation ends"""
    session_id = context.get("_conversation_id")
    user_id = context.get("user_id")

    # Save conversation summary
    save_conversation_summary(
        session_id=session_id,
        user_id=user_id,
        final_state=context.get("_current_state")
    )

    # Release any held resources
    release_user_lock(user_id)

    return {"session_cleaned_up": True}

api.register_handler(
    api.create_handler("SessionCleanup")
        .at(HandlerTiming.END_CONVERSATION)
        .do(cleanup_session)
)
```

## Building Your First Handler

Let's build a complete handler for credit card validation:

```python
from fsm_llm import API
from fsm_llm.handlers import HandlerTiming
import re

def validate_credit_card(context):
    """
    Validate credit card number using Luhn algorithm
    """
    cc_number = context.get("credit_card_number", "").replace(" ", "").replace("-", "")

    if not cc_number.isdigit() or len(cc_number) < 13 or len(cc_number) > 19:
        return {"cc_valid": False, "cc_error": "Invalid format"}

    # Luhn algorithm
    total = 0
    reverse_digits = cc_number[::-1]

    for i, digit in enumerate(reverse_digits):
        n = int(digit)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n

    is_valid = total % 10 == 0

    # Detect card type
    card_type = "unknown"
    if cc_number.startswith("4"):
        card_type = "visa"
    elif cc_number.startswith(("51", "52", "53", "54", "55")):
        card_type = "mastercard"
    elif cc_number.startswith(("34", "37")):
        card_type = "amex"

    return {
        "cc_valid": is_valid,
        "cc_type": card_type,
        "cc_last_four": cc_number[-4:] if is_valid else None
    }

# Create API and register handler
api = API.from_file("payment_flow.json", model="gpt-4o-mini")

api.register_handler(
    api.create_handler("CreditCardValidator")
        .at(HandlerTiming.CONTEXT_UPDATE)
        .when_keys_updated("credit_card_number")
        .with_priority(10)  # Higher priority
        .do(validate_credit_card)
)
```

### HandlerBuilder Methods Reference

The `HandlerBuilder` (returned by `api.create_handler()`) supports these chaining methods:

| Method | Description |
|--------|-------------|
| `.at(*timings)` | Specify one or more `HandlerTiming` values when the handler should execute |
| `.on_state(*states)` | Execute only when the FSM is in one of the specified states |
| `.not_on_state(*states)` | Exclude handler when the FSM is in any of the specified states |
| `.on_target_state(*states)` | Execute only when transitioning TO one of the specified states |
| `.not_on_target_state(*states)` | Exclude handler when transitioning TO any of the specified states |
| `.when_context_has(*keys)` | Execute only when the context contains all specified keys |
| `.when_keys_updated(*keys)` | Execute only when one or more specified context keys are updated |
| `.on_state_entry(*states)` | Shorthand for `.at(HandlerTiming.POST_TRANSITION).on_target_state(*states)` |
| `.on_state_exit(*states)` | Shorthand for `.at(HandlerTiming.PRE_TRANSITION).on_state(*states)` |
| `.on_context_update(*keys)` | Shorthand for `.at(HandlerTiming.CONTEXT_UPDATE).when_keys_updated(*keys)` |
| `.when(condition)` | Add a custom condition lambda `(timing, state, target, ctx, keys) -> bool` |
| `.with_priority(n)` | Set execution priority (lower numbers run first, default 100) |
| `.do(fn)` | Set the execution function and build the handler |

**Exclusion filter example:**

```python
# Run on every state EXCEPT "error" and "done"
api.register_handler(
    api.create_handler("GlobalLogger")
        .at(HandlerTiming.POST_PROCESSING)
        .not_on_state("error", "done")
        .do(lambda ctx: {"logged": True})
)

# Run on context update, using the shorthand
api.register_handler(
    api.create_handler("PhoneValidator")
        .on_context_update("phone_number")
        .do(validate_phone)
)
```

## Advanced Handler Patterns

### 1. **External API Calls**

> **Note:** Handlers execute synchronously. The framework does not have async handler
> execution. If you need to call async APIs, use `asyncio.run()` or a synchronous
> HTTP client like `requests` inside your handler.

```python
import requests

def check_inventory(context):
    """Check inventory from external API"""
    product_id = context.get("selected_product_id")

    if not product_id:
        return {}

    try:
        response = requests.get(
            f"https://api.store.com/inventory/{product_id}",
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return {
                "in_stock": data["quantity"] > 0,
                "quantity_available": data["quantity"],
                "warehouse_location": data["location"]
            }
    except requests.RequestException:
        pass

    return {"inventory_check_failed": True}

# Register handler
api.register_handler(
    api.create_handler("InventoryChecker")
        .on_state_entry("product_selection")
        .do(check_inventory)
)
```

### 2. **Conditional Handlers**

```python
def premium_benefits(context):
    """Apply premium user benefits"""
    return {
        "discount_rate": 0.20,
        "free_shipping": True,
        "priority_support": True
    }

# Only execute for premium users
api.register_handler(
    api.create_handler("PremiumBenefits")
        .at(HandlerTiming.POST_TRANSITION)
        .when(lambda timing, state, target, ctx, keys:
              ctx.get("user_tier") == "premium")
        .do(premium_benefits)
)
```

### 3. **Chain of Responsibility**

```python
class ValidationChain:
    """Chain multiple validators"""

    @staticmethod
    def validate_age(context):
        age = context.get("age")
        if age and isinstance(age, (int, str)):
            age_int = int(age)
            if age_int < 13:
                return {"age_valid": False, "age_error": "Must be 13 or older"}
            elif age_int > 120:
                return {"age_valid": False, "age_error": "Invalid age"}
            return {"age_valid": True, "age_group": "adult" if age_int >= 18 else "minor"}
        return {"age_valid": False}

    @staticmethod
    def validate_consent(context):
        if context.get("age_group") == "minor":
            if not context.get("parental_consent"):
                return {"consent_valid": False, "consent_error": "Parental consent required"}
        return {"consent_valid": True}

# Register validation chain
api.register_handler(
    api.create_handler("AgeValidator")
        .when_keys_updated("age")
        .do(ValidationChain.validate_age)
)

api.register_handler(
    api.create_handler("ConsentValidator")
        .when_keys_updated("age_group", "parental_consent")
        .do(ValidationChain.validate_consent)
)
```

### 4. **State Machine within Handler**

```python
class OrderStateMachine:
    """Complex order processing logic"""

    STATES = {
        "pending": ["processing", "cancelled"],
        "processing": ["shipped", "failed", "cancelled"],
        "shipped": ["delivered", "returned"],
        "delivered": ["completed", "returned"],
        "failed": ["pending", "cancelled"],
        "cancelled": [],
        "returned": ["refunded"],
        "refunded": ["completed"],
        "completed": []
    }

    @classmethod
    def process_order_update(cls, context):
        current_status = context.get("order_status", "pending")
        action = context.get("order_action")

        if not action:
            return {}

        # Map actions to new states
        action_to_state = {
            "process": "processing",
            "ship": "shipped",
            "deliver": "delivered",
            "cancel": "cancelled",
            "return": "returned",
            "refund": "refunded",
            "complete": "completed",
            "fail": "failed"
        }

        new_state = action_to_state.get(action)

        # Validate transition
        if new_state and new_state in cls.STATES.get(current_status, []):
            return {
                "order_status": new_state,
                "order_status_updated": True,
                "status_history": context.get("status_history", []) + [{
                    "from": current_status,
                    "to": new_state,
                    "timestamp": datetime.now().isoformat(),
                    "action": action
                }]
            }

        return {
            "order_status_error": f"Invalid transition from {current_status} with action {action}"
        }
```

## Handler Best Practices

### 1. **Keep Handlers Focused**

```python
# Good: Single responsibility
def validate_phone(context):
    phone = context.get("phone", "")
    # Just validate phone format
    is_valid = re.match(r'^\+?1?\d{10,14}$', phone.replace("-", ""))
    return {"phone_valid": bool(is_valid)}

# Bad: Doing too much
def do_everything(context):
    # Validate phone
    # Send SMS
    # Update database
    # Calculate shipping
    # etc...
```

### 2. **Handle Errors Gracefully**

```python
def safe_api_call(context):
    """Always handle potential failures"""
    try:
        result = external_api_call(context.get("data"))
        return {"api_result": result, "api_success": True}
    except ConnectionError:
        return {"api_success": False, "api_error": "Connection failed"}
    except Exception as e:
        logger.error(f"Unexpected error in API call: {e}")
        return {"api_success": False, "api_error": "Unknown error"}
```

**Error modes and critical handlers:**

The `HandlerSystem` has an `error_mode` setting (`"continue"` or `"raise"`) that controls whether handler failures propagate or are silently logged. When subclassing `BaseHandler`, you can set `critical=True` to force errors to always raise regardless of the error mode:

```python
class PaymentHandler(BaseHandler):
    def __init__(self):
        super().__init__(name="PaymentHandler", critical=True)

    def should_execute(self, timing, current_state, target_state, context, updated_keys):
        return timing == HandlerTiming.POST_TRANSITION and current_state == "process_payment"

    def execute(self, context):
        # If this handler fails, the error will always propagate,
        # even when handler_error_mode="continue"
        return process_payment(context)
```

This is useful for handlers where failure must halt execution (e.g., payment processing, security checks) rather than being silently swallowed.

### 3. **Use Type Hints and Documentation**

```python
from typing import Dict, Any, Optional

def calculate_discount(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate customer discount based on history and tier.

    Expected context keys:
        - order_total: float - Current order amount
        - customer_tier: str - Customer tier (bronze/silver/gold)
        - purchase_history: list - Previous purchases

    Returns:
        - discount_amount: float - Calculated discount
        - discount_reason: str - Explanation of discount
    """
    # Implementation...
```

### 4. **Avoid Side Effects in Validators**

```python
# Good: Pure validation
def validate_data(context):
    return {"valid": context.get("value", 0) > 0}

# Bad: Side effects in validation
def validate_and_save(context):
    database.save(context.get("data"))  # Don't do this!
    return {"valid": True}
```

### 5. **Use Priority for Execution Order**

```python
# Execute in specific order
api.register_handler(
    api.create_handler("Sanitizer")
        .with_priority(10)  # Runs first
        .do(sanitize_input)
)

api.register_handler(
    api.create_handler("Validator")
        .with_priority(20)  # Runs second
        .do(validate_input)
)

api.register_handler(
    api.create_handler("Enricher")
        .with_priority(30)  # Runs third
        .do(enrich_input)
)
```

## Real-World Handler Examples

### 1. **Payment Processing Integration**

```python
import stripe

class PaymentHandler:
    def __init__(self, stripe_key):
        stripe.api_key = stripe_key

    def process_payment(self, context):
        """Process payment through Stripe"""
        amount = context.get("order_total", 0)
        currency = context.get("currency", "usd")
        payment_method = context.get("payment_method_id")

        if not all([amount, payment_method]):
            return {"payment_error": "Missing payment information"}

        try:
            # Create payment intent
            intent = stripe.PaymentIntent.create(
                amount=int(amount * 100),  # Convert to cents
                currency=currency,
                payment_method=payment_method,
                confirm=True,
                metadata={
                    "order_id": context.get("order_id"),
                    "customer_id": context.get("customer_id")
                }
            )

            if intent.status == "succeeded":
                return {
                    "payment_status": "completed",
                    "payment_id": intent.id,
                    "payment_amount": amount,
                    "payment_timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "payment_status": "failed",
                    "payment_error": intent.status
                }

        except stripe.error.CardError as e:
            return {
                "payment_status": "failed",
                "payment_error": str(e),
                "payment_decline_code": e.decline_code
            }

# Register payment handler
payment_handler = PaymentHandler(os.getenv("STRIPE_KEY"))
api.register_handler(
    api.create_handler("PaymentProcessor")
        .on_state_entry("process_payment")
        .do(payment_handler.process_payment)
)
```

### 2. **Multi-Language Support**

```python
from googletrans import Translator

class LanguageHandler:
    def __init__(self):
        self.translator = Translator()
        self.supported_languages = ['en', 'es', 'fr', 'de', 'ja', 'zh-cn']

    def detect_language(self, context):
        """Detect user's language from input"""
        user_input = context.get("_user_input", "")

        if len(user_input) < 3:
            return {}

        try:
            detection = self.translator.detect(user_input)
            if detection.confidence > 0.8:
                return {
                    "detected_language": detection.lang,
                    "language_confidence": detection.confidence,
                    "needs_translation": detection.lang != 'en'
                }
        except Exception as e:
            logger.error(f"Language detection failed: {e}")

        return {}

    def translate_response(self, context):
        """Translate bot response to user's language"""
        if not context.get("needs_translation"):
            return {}

        target_lang = context.get("detected_language", "en")
        bot_response = context.get("_last_response", "")

        try:
            translated = self.translator.translate(
                bot_response,
                src='en',
                dest=target_lang
            )
            return {
                "translated_response": translated.text,
                "original_response": bot_response
            }
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {}

# Register language handlers
lang_handler = LanguageHandler()
api.register_handlers([
    api.create_handler("LanguageDetector")
        .at(HandlerTiming.PRE_PROCESSING)
        .do(lang_handler.detect_language),

    api.create_handler("ResponseTranslator")
        .at(HandlerTiming.POST_PROCESSING)
        .when(lambda t, s, tgt, ctx, k: ctx.get("needs_translation"))
        .do(lang_handler.translate_response)
])
```

### 3. **Analytics and Metrics**

```python
class AnalyticsHandler:
    def __init__(self, analytics_client):
        self.client = analytics_client

    def track_conversation_start(self, context):
        """Track conversation initiation"""
        self.client.track({
            "event": "conversation_started",
            "properties": {
                "fsm_name": context.get("_fsm_name"),
                "initial_state": context.get("_initial_state"),
                "timestamp": datetime.now().isoformat(),
                "user_id": context.get("user_id"),
                "session_id": context.get("_conversation_id")
            }
        })
        return {"analytics_session_started": True}

    def track_state_transition(self, context):
        """Track state transitions for funnel analysis"""
        self.client.track({
            "event": "state_transition",
            "properties": {
                "from_state": context.get("_previous_state"),
                "to_state": context.get("_current_state"),
                "transition_time": datetime.now().isoformat(),
                "session_id": context.get("_conversation_id"),
                "context_keys": list(context.keys())
            }
        })

        # Calculate time in state
        if "_state_entered_at" in context:
            time_in_state = (datetime.now() - context["_state_entered_at"]).total_seconds()
            return {
                "time_in_previous_state": time_in_state,
                "_state_entered_at": datetime.now()
            }

        return {"_state_entered_at": datetime.now()}

    def track_conversation_end(self, context):
        """Track conversation completion"""
        outcome = "completed" if context.get("_ended_gracefully") else "abandoned"

        self.client.track({
            "event": "conversation_ended",
            "properties": {
                "outcome": outcome,
                "final_state": context.get("_current_state"),
                "duration": context.get("_conversation_duration"),
                "total_messages": context.get("_message_count"),
                "collected_fields": [k for k in context.keys() if not k.startswith("_")]
            }
        })

        return {"analytics_session_ended": True}
```

## Testing Handlers

### Unit Testing

```python
import pytest
from unittest.mock import Mock, patch

def test_email_validator():
    """Test email validation handler"""
    # Test valid email
    context = {"email": "user@example.com"}
    result = validate_email(context)
    assert result["email_valid"] == True
    assert result["email_domain"] == "example.com"

    # Test invalid email
    context = {"email": "invalid-email"}
    result = validate_email(context)
    assert result["email_valid"] == False

    # Test missing email
    context = {}
    result = validate_email(context)
    assert result["email_valid"] == False

@patch('requests.post')
def test_api_handler(mock_post):
    """Test external API handler"""
    # Mock API response
    mock_post.return_value.json.return_value = {"status": "success"}
    mock_post.return_value.status_code = 200

    context = {"api_data": "test"}
    result = api_call_handler(context)

    assert result["api_success"] == True
    assert mock_post.called

def test_handler_execution():
    """Test handler executes and updates context"""
    api = API.from_file("test_fsm.json", model="gpt-4o-mini")

    handler = api.create_handler("TestHandler") \
        .at(HandlerTiming.POST_PROCESSING) \
        .do(lambda ctx: {"test": True})

    api.register_handler(handler)

    # Verify handler works by starting a conversation and checking context
    conv_id, _ = api.start_conversation()
    api.converse("hello", conv_id)
    data = api.get_data(conv_id)
    assert data.get("test") == True
```

### Integration Testing

```python
def test_handler_execution_flow():
    """Test handlers execute in correct order"""
    execution_order = []

    def handler1(ctx):
        execution_order.append("handler1")
        return {"handler1": True}

    def handler2(ctx):
        execution_order.append("handler2")
        return {"handler2": True}

    api = API.from_file("test_fsm.json", model="gpt-4o-mini")

    api.register_handler(
        api.create_handler("Handler1")
            .at(HandlerTiming.PRE_PROCESSING)
            .with_priority(10)
            .do(handler1)
    )

    api.register_handler(
        api.create_handler("Handler2")
            .at(HandlerTiming.PRE_PROCESSING)
            .with_priority(20)
            .do(handler2)
    )

    conv_id, _ = api.start_conversation()
    api.converse("test", conv_id)

    assert execution_order == ["handler1", "handler2"]
```

## Performance Optimization

### 1. **Batch Operations**

```python
class BatchProcessor:
    def __init__(self, batch_size=10):
        self.batch = []
        self.batch_size = batch_size

    def add_to_batch(self, context):
        """Accumulate items for batch processing"""
        item = context.get("item")
        if item:
            self.batch.append(item)

            if len(self.batch) >= self.batch_size:
                # Process batch
                results = self.process_batch(self.batch)
                self.batch = []
                return {"batch_processed": True, "batch_results": results}

        return {"batch_size": len(self.batch)}

    def process_batch(self, items):
        """Process accumulated items efficiently"""
        # Bulk API call or database operation
        return bulk_api_call(items)
```

### 2. **Caching**

```python
from functools import lru_cache
import hashlib

class CachedHandler:
    def __init__(self, cache_ttl=3600):
        self.cache = {}
        self.cache_ttl = cache_ttl

    def get_cache_key(self, context):
        """Generate cache key from context"""
        relevant_data = {
            k: v for k, v in context.items()
            if k in ["user_id", "product_id", "query"]
        }
        return hashlib.md5(
            json.dumps(relevant_data, sort_keys=True).encode()
        ).hexdigest()

    def cached_api_call(self, context):
        """Cache expensive API calls"""
        cache_key = self.get_cache_key(context)

        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return {"data": cached_data, "from_cache": True}

        # Make API call
        result = expensive_api_call(context)

        # Cache result
        self.cache[cache_key] = (result, time.time())

        return {"data": result, "from_cache": False}
```

### 3. **Parallel I/O in Handlers**

> **Note:** Handlers execute synchronously. The framework does not have async handler
> execution. To perform parallel I/O within a handler, use `concurrent.futures`
> as shown below.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

class ParallelFetchHandler:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=5)

    def fetch_user_data(self, context):
        """Make multiple API calls in parallel using threads"""
        user_id = context.get("user_id")

        if not user_id:
            return {}

        # Submit tasks to thread pool
        futures = {
            self.executor.submit(self.get_user_profile, user_id): "user_profile",
            self.executor.submit(self.get_user_orders, user_id): "user_orders",
            self.executor.submit(self.get_user_preferences, user_id): "user_preferences",
        }

        results = {}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result(timeout=10)
            except Exception:
                results[key] = None

        return results
```

## Summary

Handlers are powerful tools that extend FSM capabilities:

1. **Keep them focused** - One handler, one responsibility
2. **Handle errors gracefully** - Always expect failures
3. **Use appropriate timing** - Choose the right execution point
4. **Test thoroughly** - Unit and integration tests
5. **Optimize when needed** - Cache, batch, and parallelize

With handlers, you can integrate any external system, implement complex business logic, and create rich conversational experiences.
