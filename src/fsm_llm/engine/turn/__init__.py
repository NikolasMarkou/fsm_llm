"""The per-turn engine: ``MessagePipeline``, the 2-pass message processing.

Layer ``engine`` (plan d454fd62, step 79): the old top-level ``fsm_llm.pipeline``, split into
this package; the old dotted path is gone. Modules:

- ``message_pipeline.py``: ``MessagePipeline``, the shell (constructor, ``get_state``, the
  handler bridge ``execute_handlers``), composed of the mixins below.
- ``driver.py``: ``DriverMixin``, the sync and stream turn drivers for a message and for
  ``advance``.
- ``pass1.py``: ``Pass1Mixin``, the one Pass-1 seam (extraction through the state's executor,
  then the transition).
- ``transitions.py``: ``TransitionsMixin``, rule evaluation, the classifier tie-break and cache,
  the state transition.
- ``pass2.py``: ``Pass2Mixin``, the reply generation, the greeting and the stream variant.
- ``conversation_*.py``: the conversation kind's Pass-1 machinery (post-transition extraction,
  data extraction, bulk extraction, per-field extraction, classification extraction); step 80
  of the layout plan moves these into the kind executor.
- ``records.py``: the turn snapshot and record and the small module-level helpers.
- ``surface.py``: ``_PipelineSurface``, the mypy-only declarations the mixins share.

A private name is imported from its defining module and a patch aims at the module that performs
the lookup (``fsm_llm.engine.turn.transitions.Classifier``), never at this package.
"""

from .message_pipeline import MessagePipeline

__all__ = ["MessagePipeline"]
