# Product Recommendation System

This example demonstrates using LLM-FSM to create a decision-tree conversation flow that recommends technology products based on user preferences.

## Purpose

This example shows how to:
- Implement a branching decision tree for product recommendations
- Use conditional transitions to navigate different recommendation paths
- Extract and categorize user preferences
- Tailor recommendations based on collected preferences
- Create a conversational product advisor experience

## FSM Structure

This example implements a recommendation system with the following states:
1. **welcome**: Introduces the recommendation service
2. **device_type_selection**: Determines whether the user wants a smartphone or laptop
3. **smartphone_budget**: Determines budget range for smartphones
4. **laptop_budget**: Determines budget range for laptops
5. **recommend_budget_smartphone**: Provides budget smartphone recommendations
6. **recommend_premium_smartphone**: Provides premium smartphone recommendations
7. **recommend_budget_laptop**: Provides budget laptop recommendations
8. **recommend_premium_laptop**: Provides premium laptop recommendations
9. **end**: Concludes the conversation gracefully

## How to Run

1. Make sure you have LLM-FSM installed:
```bash
pip install llm-fsm
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your-api-key-here
```

3. Run the example:
```bash
python product_recommendation.py
```

4. To visualize the FSM structure (optional):
```bash
python fsm_visualizer.py
```

## Expected Output

```
🛒 TECH PRODUCT RECOMMENDATION SYSTEM 🛒
============================================================

Advisor: Hello and welcome to our tech product recommendation service! I'm here to help you find the perfect technology product that matches your needs and preferences. I'll need to ask you a few questions to better understand what you're looking for. Let's get started with finding the right tech product for you!

You: I'm looking for a new device

Advisor: Great! I'd be happy to help you find a new device. To start narrowing down the options, could you tell me whether you're interested in a smartphone or a laptop?

You: I need a laptop for work

Advisor: Excellent choice! Laptops are versatile tools for productivity. To help me recommend the right laptop for your work needs, could you tell me about your budget range? Are you looking for something more budget-friendly (under $800) or would you prefer a premium option (over $800)?

You: I'd like something high-end, around $1200-1500

Advisor: Perfect! With a budget of $1200-1500, I can recommend several excellent premium laptops that will serve your work needs well:

1. **MacBook Air M2** ($1199-1499) - Offers exceptional performance with the M2 chip, 8-10 core GPU, up to 24GB unified memory, and all-day battery life. Great for professionals who value portability and power.

2. **Dell XPS 13** ($1299-1499) - Features a beautiful InfinityEdge display, 12th Gen Intel processors, up to 32GB RAM, and a premium build quality. Excellent for business professionals who need Windows.

3. **ThinkPad X1 Carbon** ($1349-1599) - Known for its durability, excellent keyboard, and business features. Comes with Intel Core i5/i7 processors, up to 32GB RAM, and numerous security features.

Would you like more details about any of these models? Feel free to ask if you have any specific questions!

You: The XPS sounds good. Thanks!

Advisor: You're welcome! The Dell XPS 13 is indeed an excellent choice. It's known for its premium build quality with a CNC machined aluminum chassis, carbon fiber palm rest, and edge-to-edge Corning Gorilla Glass. The InfinityEdge display provides a stunning visual experience with minimal bezels.

For your budget range, you'll be able to get a configuration with a 12th Gen Intel Core i7 processor, 16GB RAM, and a 512GB SSD, which offers plenty of power for multitasking, productivity work, and even some creative tasks.

Thank you for using our product recommendation service! If you have any more questions about the Dell XPS 13 or want to explore other options in the future, feel free to reach out. Happy computing!

============================================================
🏁 RECOMMENDATION COMPLETE 🏁

Your Technology Preferences:
• Device Type: Laptop
• Budget Range: Premium

Thank you for using our Product Recommendation System!
```

## Learning Points

This example demonstrates:
- How to implement a decision tree conversation structure
- Using conditional transitions to create different recommendation paths
- How to extract and categorize user preferences (device type and budget tier)
- Tailoring recommendations based on collected user preferences
- How to structure an FSM for product recommendations with branching paths
- Building a system that can be easily extended with additional product categories