# Book Recommendation System Example

This example demonstrates using LLM-FSM to create an interactive book recommendation system that suggests books based on user genre preferences and engagement level.

## Purpose

This example shows how to:
- Create a conversational flow that adapts to user engagement
- Extract and utilize user preferences for personalized recommendations
- Implement a cyclical conversation pattern with decision points
- Track user engagement level to determine conversation direction
- Use the LLM-FSM framework to maintain context between conversation turns

## FSM Structure

This example implements a book recommendation system with the following states:

1. **welcome**: Introduces the service and explains its purpose
2. **genre_selection**: Collects the user's favorite book genres
3. **recommend_book**: Provides a personalized book recommendation based on collected genres
4. **check_engagement**: Determines if the user wants more recommendations or is losing interest
5. **end**: Concludes the conversation with a summary of recommendations

The system loops between recommendation and engagement checking states until the user shows low engagement or explicitly ends the conversation.

## How to Run

1. Make sure you have LLM-FSM installed:
```bash
pip install llm-fsm
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your-api-key-here
```

3. Run the basic implementation:
```bash
python book_recommendation_system.py
```

Or run the enhanced version with advanced features:
```bash
python enhanced_book_recommendation.py
```

4. Visualize the FSM structure (optional):
```bash
python fsm_visualizer.py
```

## Expected Output

```
System: Welcome to the Book Recommendation System! I'm here to help you discover new books that you might enjoy. To get started, I'd like to know what genres of books you typically enjoy reading. Could you share some of your favorite genres with me?

You: I like science fiction, fantasy, and historical fiction

System: Thank you for sharing your preferences! Based on your interest in science fiction, I'd recommend "Project Hail Mary" by Andy Weir. It's a thrilling sci-fi novel about a lone astronaut who must save humanity from extinction. The story combines compelling character development with fascinating scientific concepts, much like Weir's previous hit "The Martian." Would you like another recommendation?

You: Yes, please recommend something in fantasy

System: Great! For fantasy, I'd recommend "The Fifth Season" by N.K. Jemisin. It's the first book in the Broken Earth trilogy which won the Hugo Award for Best Novel three years in a row. The novel takes place on a supercontinent called the Stillness which experiences catastrophic climate change seasons. It features complex characters, innovative magic systems based on earth control, and tackles themes of oppression and survival. Would you like another recommendation?

You: No, I think that's enough for now

System: Thank you for using our book recommendation service today! I've recommended some books from your preferred genres of science fiction and fantasy, including "Project Hail Mary" by Andy Weir and "The Fifth Season" by N.K. Jemisin. I hope you enjoy these recommendations. Happy reading!
Conversation has ended.

--- Session Summary ---
Duration: 1 minutes, 45 seconds
Your genre preferences: science fiction, fantasy, historical fiction

Books recommended (2):
1. Project Hail Mary by Andy Weir
2. The Fifth Season by N.K. Jemisin

Thank you for using the Book Recommendation System!
```

## Learning Points

This example demonstrates:

- How to implement a stateful conversation flow with LLM-FSM
- Using `required_context_keys` to ensure necessary information is collected
- Creating a cyclical conversation pattern with state loops
- Determining user engagement level to guide conversation flow
- Managing context across the entire conversation
- Extracting and utilizing user preferences for personalized responses
- Implementing graceful conversation termination with summary
- How to design an FSM with both linear and cyclical elements

## Extension Ideas

Some ways you could extend this example:
- Add a state for collecting user feedback on recommendations
- Implement a "narrow preferences" state for when user genres are too broad
- Add integration with a book database/API for real-time recommendations
- Implement a "save preferences" feature for returning users
- Create a "related books" state that recommends similar titles
- Add book category filtering (e.g., no mature content, specific time periods)