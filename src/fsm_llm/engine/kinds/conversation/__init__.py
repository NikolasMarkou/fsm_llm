"""The conversation kind's Pass 1 (plan d454fd62, step 80, D-030).

- ``executor.py``: ``ConversationExecutor`` (the registered, stateless executor) and
  ``ConversationPass1`` (the machinery below, bound to one pipeline's services for one call).
- ``extraction.py``: ``ConversationExtractionMixin``, the bulk-then-field data extraction.
- ``bulk.py``: ``ConversationBulkMixin``, the one bulk call and ``_BulkFailed``.
- ``fields.py``: ``ConversationFieldsMixin``, the field configs, the per-field calls, the
  validation and the type coercers.
- ``classification.py``: ``ConversationClassificationMixin``, the ``classification_extractions``.
- ``post_transition.py``: ``ConversationPostTransitionMixin``, what runs after a transition.
- ``surface.py``: ``_ConversationSurface``, the mypy-only declarations the mixins share.

A private helper is imported from its defining module and a patch aims at the module that
performs the lookup (``fsm_llm.engine.kinds.conversation.fields.logger``), never at this package.
"""

from .executor import ConversationExecutor

__all__ = ["ConversationExecutor"]
