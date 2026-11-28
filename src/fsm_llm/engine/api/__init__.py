"""The conversation API: ``API`` and the stack types.

Layer ``engine`` (plan d454fd62, step 81): the old top-level ``fsm_llm.api``, split into this
package; the old dotted path is gone. Modules:

- ``conversation.py``: ``API``, the shell (constructor, factories, ``close``), composed of the
  mixins below.
- ``turn_entries.py``: ``TurnEntriesMixin``, ``start_conversation``, the turn entries
  ``converse`` and ``advance`` with their streams, and the two helpers every entry shares
  (``_turn_errors``, ``_auto_save_session``).
- ``run.py``: ``RunMixin``, ``run_until_terminal`` and its stream, the budgets and the
  ``before_step`` hook rounds.
- ``stack.py``: ``StackMixin``, ``push_fsm``, ``pop_fsm``, the context merge, the top-of-stack
  lookup, and the public ``FSMStackFrame`` and ``ContextMergeStrategy``.
- ``lifecycle.py``: ``LifecycleMixin``, the ended-conversation cache, the getters,
  ``end_conversation`` and the idle sweep.
- ``sessions.py``: ``SessionsMixin``, ``save_session``, ``load_session``, ``restore_session``.
- ``ledger.py``: ``release_dropped_frames``, the one cleanup of "these frames are gone" (the call
  marks, the delegated runs, the temp definitions) that ``pop_fsm`` and the refused branch of
  ``end_conversation`` share, and ``register_pending_push`` and ``release_pending_push``, the count
  of pushes in flight per definition id.
- ``calls.py``, ``delegated.py``: ``settle`` (the call record applied after the turn) and its
  delegated part (start the run after the commit, the run events, the one bounded wait).
- ``forks.py``, ``events.py``, ``budget.py``, ``registration.py``: ``fork`` and ``resume``, the
  event-log opt-in, the lifetime turn budget of a conversation, and the handler registration.
- ``helpers.py``: the module-level helpers (``handle_conversation_errors``, ``llm_settings_for``
  and the private error and read helpers).
- ``surface.py``: ``_ApiSurface``, the mypy-only declarations the mixins share.

``from fsm_llm import API`` is unchanged. A private helper is imported from its defining module
and a patch aims at the module that performs the lookup (``fsm_llm.engine.api.conversation.
LiteLLMInterface``, ``fsm_llm.engine.api.run.time``), never at this package.
"""

from .conversation import API
from .helpers import handle_conversation_errors, llm_settings_for
from .lifecycle import MAX_ENDED_CONVERSATIONS_CACHE
from .stack import ContextMergeStrategy, FSMStackFrame

__all__ = [
    "API",
    "MAX_ENDED_CONVERSATIONS_CACHE",
    "ContextMergeStrategy",
    "FSMStackFrame",
    "handle_conversation_errors",
    "llm_settings_for",
]
