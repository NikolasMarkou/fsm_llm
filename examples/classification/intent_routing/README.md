# Intent Classification and Routing

Classifies customer support messages into predefined intents and routes them to the appropriate handler function.

## Key Concepts

- **ClassificationSchema**: Define intents with descriptions and a fallback class
- **Classifier**: LLM-backed classifier that returns structured JSON (intent, confidence, entities)
- **IntentRouter**: Maps intents to handler functions with automatic low-confidence clarification

## Running

```bash
export OPENAI_API_KEY="your-key-here"
python run.py
```

## Example Session

```
Customer Support Classifier (type 'quit' to exit)
--------------------------------------------------

You: Where is my order #12345?
  Intent:     order_status
  Confidence: 0.95
  Entities:   {'order_id': '12345'}

Bot: [Order Status] Looking up order 12345. Your package is on its way!

You: I want to return the headphones I bought
  Intent:     return_request
  Confidence: 0.90
  Entities:   {'product_name': 'headphones'}

Bot: [Return Request] Starting a return for your order. You'll receive a shipping label shortly.

You: hmm
  Intent:     general_support
  Confidence: 0.40

Bot: [Clarification] I'm not quite sure what you need. Could you rephrase your question?
```

## Schema

Five intent classes: `order_status`, `product_info`, `payment_issue`, `return_request`, `general_support` (fallback). Confidence threshold set to 0.6.
