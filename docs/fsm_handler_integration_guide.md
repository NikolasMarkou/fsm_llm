# Quick Integration Guide: FSM Handlers

This guide provides an overview of the key changes and how to use the new handler system with the FSM Manager.

## Key Changes in the Updated FSM Manager

1. **Fully Asynchronous Architecture**
   - All methods now use `async/await` for consistent execution
   - Handler integration at every lifecycle point

2. **Enhanced Error Handling**
   - Added a dedicated exception hierarchy
   - Support for error recovery via ERROR handlers
   - Proper error propagation to maintain context

3. **Complete Handler Integration Points**
   - `START_CONVERSATION`: When a conversation begins
   - `PRE_PROCESSING`: Before processing user input
   - `POST_PROCESSING`: After processing but before transition
   - `CONTEXT_UPDATE`: When context data is updated
   - `PRE_TRANSITION`: Before a state transition
   - `POST_TRANSITION`: After a state transition
   - `END_CONVERSATION`: When a conversation ends
   - `ERROR`: When an error occurs

4. **Context Management**
   - Context updates from handlers are properly integrated
   - Added special keys like `_previous_state` for tracking transitions
   - Fallback messaging support via error handlers

## Basic Usage Example

```python
from llm_fsm.fsm_manager import FSMManager
from llm_fsm.handler_system import create_handler, HandlerTiming
from llm_fsm.llm import LiteLLMInterface

# Initialize components
llm_interface = LiteLLMInterface(
    model="gpt-4o",
    api_key="your-api-key"
)

# Create FSM Manager
manager = FSMManager(
    llm_interface=llm_interface
)

# Create and register a simple handler
email_validator = create_handler("EmailValidator") \
    .on_context_update("email") \
    .do(lambda context: {
        "email_validation": {
            "is_valid": "@" in context.get("email", ""),
            "timestamp": datetime.now().isoformat()
        }
    })

# Register the handler
manager.handler_system.register_handler(email_validator)

# Start a conversation
conversation_id, initial_response = await manager.start_conversation("personal_info.json")
print(f"Initial response: {initial_response}")

# Process a message
response = await manager.process_message(conversation_id, "My email is user@example.com")
print(f"Response: {response}")

# End the conversation
await manager.end_conversation(conversation_id)
```

## Creating Handlers

There are two ways to create handlers:

### 1. Using the Builder Pattern (for simple handlers)

```python
# Create a phone formatter
phone_formatter = create_handler("PhoneFormatter") \
    .when_context_has("phone") \
    .on_context_update("phone") \
    .with_priority(50) \
    .do(format_phone)

def format_phone(context):
    phone = context.get("phone", "")
    # Strip all non-numeric characters
    digits = ''.join(c for c in phone if c.isdigit())
    # Format as (123) 456-7890
    if len(digits) == 10:
        formatted = f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"
        return {"phone_formatted": formatted}
    return {}
```

### 2. Creating a Custom Handler Class (for complex handlers)

```python
from llm_fsm.handler_system import BaseHandler, HandlerTiming

class ExternalAPIHandler(BaseHandler):
    def __init__(self, api_client, api_key):
        super().__init__(name="ExternalAPIHandler", priority=10)
        self.api_client = api_client
        self.api_key = api_key
        
    def should_execute(self, timing, current_state, target_state, context, updated_keys=None):
        # Execute when transitioning to confirmation state
        return (
            timing == HandlerTiming.POST_TRANSITION and
            target_state == "confirmation" and
            "customer_id" in context
        )
        
    async def execute(self, context):
        try:
            # Call external API
            result = await self.api_client.update_customer(
                customer_id=context["customer_id"],
                data=context,
                api_key=self.api_key
            )
            
            return {
                "api_response": {
                    "success": True,
                    "transaction_id": result.get("transaction_id"),
                    "timestamp": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {
                "api_response": {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }
```

## Common Use Cases

### 1. Validation Handlers

```python
# Create validation handlers for different fields
email_validator = create_handler("EmailValidator") \
    .on_context_update("email") \
    .do(validate_email)
    
phone_validator = create_handler("PhoneValidator") \
    .on_context_update("phone") \
    .do(validate_phone)
    
zipcode_validator = create_handler("ZipcodeValidator") \
    .on_context_update("zipcode") \
    .do(validate_zipcode)
```

### 2. Data Transformation

```python
# Format and normalize data
address_normalizer = create_handler("AddressNormalizer") \
    .when_context_has("address", "city", "state", "zip") \
    .on_context_update("address", "city", "state", "zip") \
    .do(normalize_address)
```

### 3. External API Integration

```python
# Create handler for external API calls
crm_sync = create_handler("CRMSync") \
    .on_state_entry("confirmation") \
    .when_context_has("email", "name") \
    .do(sync_with_crm)
```

### 4. Analytics and Logging

```python
# Create analytics handler that runs on every transition
analytics_logger = create_handler("AnalyticsLogger") \
    .at(HandlerTiming.POST_TRANSITION) \
    .with_priority(1000)  # Low priority (runs last)
    .do(log_analytics_event)
```

### 5. Error Recovery

```python
# Create error handler for graceful recovery
error_handler = create_handler("ErrorRecoveryHandler") \
    .at(HandlerTiming.ERROR) \
    .with_priority(1)  # High priority (runs first)
    .do(handle_error)
    
def handle_error(context):
    error = context.get("error", {})
    error_type = error.get("type", "Unknown")
    
    # Provide fallback message based on error type
    if "database" in error.get("message", "").lower():
        return {
            "error_recovery": {
                "fallback_message": "Our system is experiencing temporary issues. Please try again later."
            }
        }
    elif "validation" in error.get("message", "").lower():
        return {
            "error_recovery": {
                "fallback_message": "There seems to be an issue with the information provided. Could you try again?"
            }
        }
    else:
        return {
            "error_recovery": {
                "fallback_message": "I'm having trouble processing that. Let's try something else."
            }
        }
```

## Best Practices

1. **Set Appropriate Priorities**
   - Critical handlers should have higher priority (lower numbers)
   - Analytics/logging handlers should have lower priority (higher numbers)

2. **Use Handler Timing Appropriately**
   - `CONTEXT_UPDATE`: For validation and transformation
   - `POST_TRANSITION`: For tracking state changes
   - `PRE_PROCESSING`/`POST_PROCESSING`: For message augmentation

3. **Handle Errors Gracefully**
   - Always implement at least one ERROR handler
   - Provide user-friendly fallback messages

4. **Keep Handlers Focused**
   - Each handler should do one thing well
   - Use proper conditions to limit when handlers execute

5. **Monitor Performance**
   - Long-running handlers might delay responses
   - Consider making API calls asynchronous when possible

## Testing Handlers

Create unit tests for your handlers to ensure they work correctly:

```python
import pytest
from llm_fsm.handler_system import create_handler, HandlerTiming

@pytest.mark.asyncio
async def test_email_validator():
    # Create the handler
    validator = create_handler("EmailValidator") \
        .on_context_update("email") \
        .do(lambda ctx: {"email_valid": "@" in ctx.get("email", "")})
    
    # Test valid condition
    should_execute = validator.should_execute(
        HandlerTiming.CONTEXT_UPDATE,
        "collect_email", 
        None,
        {"email": "test@example.com"},
        {"email"}
    )
    assert should_execute is True
    
    # Test execution with valid email
    result = await validator.execute({"email": "test@example.com"})
    assert result["email_valid"] is True
    
    # Test execution with invalid email
    result = await validator.execute({"email": "invalid-email"})
    assert result["email_valid"] is False
```

## Conclusion

The handler system provides a powerful way to extend the FSM framework with custom business logic, validations, integrations, and more. By using handlers, you can keep your FSM definitions clean and focused on conversation flow while implementing complex functionality in a modular way.