# Form Filling Example

This example demonstrates using LLM-FSM to create a structured form-filling conversation flow to collect user information.

## Purpose

This example shows how to:
- Build a multi-step form-filling conversation
- Extract and validate specific information from user input
- Use conditional transitions based on collected context data
- Implement a confirmation step with correction capabilities
- Leverage the simplified LLM_FSM API

## FSM Structure

This example implements a form with the following states:
1. **welcome**: Introduces the form and explains the purpose
2. **collect_name**: Collects the user's full name
3. **collect_email**: Collects the user's email address
4. **collect_age**: Collects the user's age
5. **collect_occupation**: Collects the user's occupation
6. **confirm**: Shows collected information and asks for confirmation
7. **thank_you**: Thanks the user and completes the process

The confirmation state has multiple transitions to allow users to correct specific fields.

## How to Run

1. Make sure you have LLM-FSM installed:
```bash
pip install llm-fsm
```

2. Set your OpenAI API key (or use the environment variable):
```bash
export OPENAI_API_KEY=your-api-key-here
```

3. Run the example:
```bash
python run.py
```

## Expected Output

```
System: Hello! I'd like to collect some information from you to complete your profile. Let's get started!

First, could you please tell me your full name?
You: John Smith
System: Thank you, John Smith! Now, could you please provide your email address?
You: john.smith@example.com
System: Great! Now, could you tell me your age?
You: 35
System: Thanks! And what is your current occupation?
You: Software Engineer
System: Thank you for all that information! Here's what I've collected:

- Name: John Smith
- Email: john.smith@example.com
- Age: 35
- Occupation: Software Engineer

Is this information correct? If not, you can tell me which field needs correction.
You: Yes, that's correct
System: Thank you for providing your information, John Smith! Your profile has been successfully created. Have a great day!
Conversation has ended.
```

## Learning Points

This example demonstrates:
- How to structure a sequential form-filling conversation
- How to use `required_context_keys` to ensure information is collected
- How to implement validation and correction flows
- How to use transitions with conditions to enable branching based on user input
- How to create a confirmation step with the ability to correct specific fields
- How LLM-FSM maintains context across the entire conversation