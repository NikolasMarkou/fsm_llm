"""The conversation lifecycle manager: ``FSMManager`` and its helpers.

Layer ``engine`` (plan d454fd62, step 78): the old top-level ``fsm_llm.fsm``, split into
this package; the old dotted path is gone. Modules:

- ``manager.py``: ``FSMManager``, the shell (constructor, definition cache, start, end,
  context update), composed of the two mixins below.
- ``turns.py``: ``TurnsMixin``, the turn path (the turn guard, the locked sync turn, the
  lazy locked stream, the ERROR handlers).
- ``snapshots.py``: ``SnapshotsMixin``, the reads, snapshots and the restore seed, all
  under ``conv_lock``.
- ``helpers.py``: the caller-visible context filter, ``with_conversation_context``, the
  turn-failure mapping and the conversation lookup.

A private helper is imported from its defining module and a patch aims at the module
that performs the lookup (``fsm_llm.engine.fsm.snapshots.END_CONVERSATION_LOCK_TIMEOUT_SECONDS``),
never at this package.
"""

from .helpers import with_conversation_context
from .manager import FSMManager

__all__ = ["FSMManager", "with_conversation_context"]
