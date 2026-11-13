"""The sentences every reasoning FSM prompt shares."""

# The no-questions rule keeps every FSM message-free: the sentences that carry it
# are written once here, never inline.
_NO_QUESTIONS = "Do not ask questions or request additional input from the user."
_IMPORTANT_NO_QUESTIONS = f"IMPORTANT: {_NO_QUESTIONS}"
_PRESENT_FINDINGS = f"Present your findings clearly and concisely. {_NO_QUESTIONS}"
_PRESENT_FINAL = f"Present the final result clearly. {_NO_QUESTIONS}"
