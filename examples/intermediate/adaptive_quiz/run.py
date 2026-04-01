"""
Adaptive Quiz Example — Core FSM + Handlers
=============================================

Demonstrates using custom handlers to track quiz scores, adjust difficulty
dynamically, and log state transitions. Shows the HandlerBuilder fluent API
with multiple handler timing points.

Key Concepts:
    - HandlerBuilder fluent API (.at(), .on_state(), .do())
    - HandlerTiming: POST_PROCESSING, POST_TRANSITION, END_CONVERSATION
    - Runtime context mutation from handlers
    - Adaptive behavior driven by handler logic

Usage:
    export OPENAI_API_KEY="your-key-here"
    python run.py

    # Or with a local Ollama model:
    export LLM_MODEL="ollama_chat/qwen3.5:9b"
    python run.py
"""

import os

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming


def main():
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Error: OPENAI_API_KEY not set. Export it or use Ollama.")
        print("       export LLM_MODEL='ollama_chat/qwen3.5:9b'")
        return

    current_dir = os.path.dirname(os.path.abspath(__file__))
    fsm_path = os.path.join(current_dir, "fsm.json")

    fsm = API.from_file(path=fsm_path, model=model, api_key=api_key, temperature=0.7)

    # ------------------------------------------------------------------
    # Handler 1: Score Tracker — runs after each answer evaluation
    # Updates score, question count, and adjusts difficulty.
    # ------------------------------------------------------------------
    score_tracker = (
        fsm.create_handler("ScoreTracker")
        .at(HandlerTiming.POST_PROCESSING)
        .on_state("evaluate_answer")
        .do(lambda ctx: _update_score(ctx))
    )
    fsm.register_handler(score_tracker)

    # ------------------------------------------------------------------
    # Handler 2: Difficulty Adjuster — runs after transitioning from evaluate_answer
    # Adjusts difficulty based on recent performance (last 3 answers).
    # ------------------------------------------------------------------
    difficulty_adjuster = (
        fsm.create_handler("DifficultyAdjuster")
        .at(HandlerTiming.POST_TRANSITION)
        .on_state("ask_question")
        .do(lambda ctx: _adjust_difficulty(ctx))
    )
    fsm.register_handler(difficulty_adjuster)

    # ------------------------------------------------------------------
    # Handler 3: Session Logger — runs when conversation ends
    # Prints a performance breakdown.
    # ------------------------------------------------------------------
    session_logger = (
        fsm.create_handler("SessionLogger")
        .at(HandlerTiming.END_CONVERSATION)
        .do(lambda ctx: _log_session(ctx))
    )
    fsm.register_handler(session_logger)

    # ------------------------------------------------------------------
    # Start the quiz
    # ------------------------------------------------------------------
    initial_context = {
        "score": 0,
        "questions_asked": 0,
        "difficulty": "medium",
        "answer_history": [],  # list of booleans
        "difficulty_history": [],  # tracks difficulty changes
    }

    print("Adaptive Trivia Quiz")
    print("=" * 50)

    conversation_id, response = fsm.start_conversation(initial_context=initial_context)
    print(f"\nQuiz Master: {response}")

    while not fsm.has_conversation_ended(conversation_id):
        user_input = input("\nYou: ").strip()
        if not user_input or user_input.lower() in ("exit", "quit"):
            break

        response = fsm.converse(user_input, conversation_id)
        print(f"  State: {fsm.get_current_state(conversation_id)}")

        # Show current stats inline
        ctx = fsm.get_data(conversation_id)
        state = fsm.get_current_state(conversation_id)
        if state == "ask_question":
            print(
                f"  [Score: {ctx.get('score', 0)}/{ctx.get('questions_asked', 0)} | "
                f"Difficulty: {ctx.get('difficulty', 'medium')}]"
            )

        print(f"\nQuiz Master: {response}")

    # Final summary
    ctx = fsm.get_data(conversation_id)
    asked = ctx.get("questions_asked", 0)
    score = ctx.get("score", 0)

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"  Questions answered: {asked}")
    print(f"  Correct answers:   {score}")
    if asked > 0:
        print(f"  Accuracy:          {score / asked * 100:.0f}%")
    print(f"  Difficulty history: {' -> '.join(ctx.get('difficulty_history', []))}")

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    data = fsm.get_data(conversation_id)
    expected_keys = [
        "player_name",
        "current_question",
        "correct_answer",
        "wants_to_stop",
        "answer_correct",
        "user_answer",
        "player_feedback",
        "score",
        "questions_asked",
        "difficulty",
        "answer_history",
        "difficulty_history",
    ]
    extracted = 0
    for key in expected_keys:
        value = data.get(key)
        status = "EXTRACTED" if value is not None else "MISSING"
        if value is not None:
            extracted += 1
        print(f"  {key:25s}: {str(value)[:40]:40s} [{status}]")
    print(
        f"\nExtraction rate: {extracted}/{len(expected_keys)} ({100 * extracted / len(expected_keys):.0f}%)"
    )
    print(f"Final state: {fsm.get_current_state(conversation_id)}")

    fsm.end_conversation(conversation_id)


# ------------------------------------------------------------------
# Handler helper functions
# ------------------------------------------------------------------


def _update_score(ctx: dict) -> dict:
    """Update score based on whether the answer was correct."""
    correct = ctx.get("answer_correct", False)
    ctx["questions_asked"] = ctx.get("questions_asked", 0) + 1
    if correct:
        ctx["score"] = ctx.get("score", 0) + 1

    history = ctx.get("answer_history", [])
    history.append(bool(correct))
    return {
        "answer_history": history,
        "score": ctx.get("score", 0),
        "questions_asked": ctx["questions_asked"],
    }


def _adjust_difficulty(ctx: dict) -> dict:
    """Adjust difficulty based on the last 3 answers."""
    history = ctx.get("answer_history", [])
    current = ctx.get("difficulty", "medium")

    # Look at last 3 answers
    recent = history[-3:] if len(history) >= 3 else history
    if not recent:
        return {}

    correct_ratio = sum(recent) / len(recent)

    if correct_ratio >= 0.8 and current != "hard":
        new_difficulty = {"easy": "medium", "medium": "hard"}.get(current, current)
    elif correct_ratio <= 0.3 and current != "easy":
        new_difficulty = {"hard": "medium", "medium": "easy"}.get(current, current)
    else:
        new_difficulty = current

    diff_history = ctx.get("difficulty_history", [])
    diff_history.append(new_difficulty)
    return {"difficulty": new_difficulty, "difficulty_history": diff_history}


def _log_session(ctx: dict) -> dict:
    """Log session statistics (printed by the handler)."""
    history = ctx.get("answer_history", [])
    if history:
        streak = 0
        max_streak = 0
        for correct in history:
            if correct:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        print(f"\n  [Session Stats] Best streak: {max_streak} correct in a row")
    return {}


if __name__ == "__main__":
    main()
