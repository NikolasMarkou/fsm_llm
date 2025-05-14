# Adaptive Yoga Instruction Example

This example demonstrates using LLM-FSM to create an adaptive yoga instruction flow that monitors user engagement and adjusts accordingly.

## Purpose

This example shows how to:
- Create a responsive conversational flow that tracks user engagement
- Periodically gauge user interest and adapt the conversation path
- Implement sentiment analysis within state transitions
- Gracefully exit when user interest wanes
- Design a non-linear FSM with multiple potential paths

## FSM Structure

This example implements a yoga instruction flow with the following states:
1. **welcome**: Introduces the yoga session and gauges initial interest
2. **gauge_interest**: Periodically checks user's engagement level
3. **beginner_pose**: Teaches mountain pose (Tadasana)
4. **intermediate_pose**: Teaches tree pose (Vrikshasana)
5. **advanced_pose**: Teaches warrior pose (Virabhadrasana)
6. **cool_down**: Provides cool down instructions
7. **farewell**: Concludes the session

The FSM periodically transitions to the `gauge_interest` state, which evaluates the user's responses to determine whether to continue with more poses or gracefully exit.

## Interest Tracking

The system tracks engagement through:
- Direct questions about interest level
- Sentiment analysis of user responses
- Detection of short or minimal responses
- Explicit requests to stop

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

## Example Conversations

### Engaged User Flow:
```
System: Welcome to your personalized yoga session! I'll guide you through some simple poses to help you relax and improve flexibility. How experienced are you with yoga?
You: I'm a beginner, but I've done a few classes before
System: Great! Let's start with a simple Mountain Pose (Tadasana) which is perfect for beginners.

Stand with your feet hip-width apart, distribute your weight evenly, and let your arms rest at your sides. Take a deep breath, roll your shoulders back, and stand tall.

How does that feel? Are you ready to try it?
You: Yes, I'm standing tall now. It feels good.
System: Excellent! I'm glad you're enjoying it. Before we move to the next pose, I'm curious - how are you finding this session so far? Are you enjoying the pace?
You: Yes, it's good! I'd like to try something a bit more challenging.
System: That's great to hear you're enjoying it and ready for more of a challenge! Let's try the Tree Pose (Vrikshasana).

Start in Mountain Pose, then shift your weight to your left foot. Place your right foot on your left inner thigh or calf (avoid the knee). Bring your palms together at heart center or raise them above your head. Focus on a point in front of you for balance.

Take your time and breathe deeply. How are you doing with this pose?
```

### Disengaged User Flow:
```
System: Welcome to your personalized yoga session! I'll guide you through some simple poses to help you relax and improve flexibility. How experienced are you with yoga?
You: Not very interested, just checking this out
System: Thanks for being honest about your interest level. Let's start with something very basic and see if it catches your interest.

The Mountain Pose (Tadasana) is the foundation of all standing poses. Simply stand with your feet hip-width apart, distribute your weight evenly, and let your arms rest at your sides. Take a deep breath.

How does that feel? Would you like to continue with yoga?
You: It's ok I guess. Not really feeling it.
System: I understand yoga isn't for everyone. It sounds like this might not be what you're looking for today, and that's perfectly okay.

Thank you for giving it a try! If you're ever interested in exploring yoga in the future, feel free to come back. Is there something else you'd be more interested in trying instead?
```

## Learning Points

This example demonstrates:
- How to implement a conversational flow that adapts to user engagement
- Techniques for gauging interest through conversational cues
- Creating graceful exit paths when user interest decreases
- Designing flexible FSMs that can take different paths based on user state
- Using context variables to track sentiment across states