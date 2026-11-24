"""Identifier charset policy and the size caps the definition models share."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# The single source of truth for every identifier-shaped field in this module:
# `State.id`, `Transition.target_state`, `TransitionOption.target_state` and
# `IntentDefinition.name` (`FSMDefinition.initial_state` has only `min_length`; it
# is bound by `initial_state in states`). ASCII-only, letter-or-underscore first. Two spellings
# because pydantic's `Field(pattern=)` wants the source string while
# `IntentDefinition.validate_name_format` needs a compiled object -- see the
# D-025 anchor there for why that field must NOT migrate to `pattern=`.
ASCII_IDENTIFIER_PATTERN = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
_ASCII_IDENTIFIER = re.compile(ASCII_IDENTIFIER_PATTERN)

# Cap of the free text a model returns in a response model (``message``,
# ``reasoning``). ``llm/parsing.py`` truncates a recovered raw reply to it, so it must
# build the same capped model the pydantic field declares: one definition here.
RESPONSE_TEXT_MAX_LENGTH = 5000

#: Largest ``priority`` a transition or condition may carry (lower wins).
MAX_PRIORITY = 1000

#: Most intents a classification schema holds, the fallback included. The
#: ``intents`` field cap and the validator's tie warning (a tie of more
#: distinct targets than ``MAX_CLASSIFICATION_INTENTS - 1`` cannot be
#: classified) both read it; lives here, not in ``constants.py`` (pinned).
MAX_CLASSIFICATION_INTENTS = 15

#: Longest ``Conversation.summary`` (and ``ContextCompactor`` summary), in
#: characters; lives here, not in ``constants.py`` (pinned).
CONVERSATION_SUMMARY_MAX_CHARS = 2000

#: ``SynthesizeConfig`` default result key and caps (goal text, key names);
#: here, not in ``constants.py`` (pinned).
DEFAULT_SYNTHESIS_RESULT_KEY = "synthesis_result"
SYNTHESIZE_GOAL_MAX_LENGTH = 2000
SYNTHESIZE_KEY_MAX_LENGTH = 100

#: ``CallStateConfig`` default result key (the key a call state's record is written to);
#: here, not in ``constants.py`` (pinned), like the synthesis one.
DEFAULT_CALL_RESULT_KEY = "call_result"

#: The id of an executable, as the envelope spells it and as a ``call`` target names one: the
#: one spelling (the envelope model and the call models both read it).
EXECUTABLE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$"
EXECUTABLE_ID_MAX_LENGTH = 128

#: The caps of the executable envelope models (``executable`` and ``executable_pattern`` read
#: them): most entries in a list of names, longest single name.
EXECUTABLE_MAX_ITEMS = 256
EXECUTABLE_NAME_MAX_LENGTH = 128


def nests_deeper_than(value: Any, limit: int) -> bool:
    """True when a container of `value` sits deeper than `limit` (the root is 1).

    A container is any ``Mapping`` or any ``Sequence`` that is not text or bytes (a tuple
    counts), so a caller's ``MappingProxyType`` or tuple cannot slip under the bound.
    Iterative (an explicit stack): a recursive walk would hit the very limit it guards
    against.
    """
    stack = [(value, 1)]
    while stack:
        node, level = stack.pop()
        children: Iterable[Any]
        if isinstance(node, Mapping):
            children = node.values()
        elif isinstance(node, Sequence) and not isinstance(
            node, (str, bytes, bytearray)
        ):
            children = node
        else:
            continue
        if level > limit:
            return True
        stack.extend((child, level + 1) for child in children)
    return False
