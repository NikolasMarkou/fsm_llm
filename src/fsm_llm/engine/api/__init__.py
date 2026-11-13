"""The conversation API: ``API`` and the stack types.

Layer ``engine`` (plan d454fd62, step 81): the old top-level ``fsm_llm.api``, split into this
package; the old dotted path is gone. Modules:

- ``conversation.py``: ``API``, the shell (constructor, factories, ``start_conversation``, the
  turn entries ``converse`` and ``advance`` with their streams, handler registration, ``close``),
  composed of the four mixins below.
- ``run.py``: ``RunMixin``, ``run_until_terminal`` and its stream, the budgets and the
  ``before_step`` hook rounds.
- ``stack.py``: ``StackMixin``, ``push_fsm``, ``pop_fsm``, the context merge, the top-of-stack
  lookup, and the public ``FSMStackFrame`` and ``ContextMergeStrategy``.
- ``lifecycle.py``: ``LifecycleMixin``, the ended-conversation cache, the getters,
  ``end_conversation`` and the idle sweep.
- ``sessions.py``: ``SessionsMixin``, ``save_session``, ``load_session``, ``restore_session``.
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
