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
- ``records.py``: the turn snapshot and record and the small module-level helpers.
- ``surface.py``: ``_PipelineSurface``, the mypy-only declarations the mixins share.

The conversation kind's Pass-1 machinery (post-transition, data, bulk, per-field and
classification extraction) is NOT here: since step 80 the conversation executor owns it
(``fsm_llm.engine.kinds.conversation``) and reaches this package's pipeline only through the
``_PipelineServices`` protocol; this package imports ``fsm_llm.engine.kinds``, never the reverse.

A private name is imported from its defining module and a patch aims at the module that performs
the lookup (``fsm_llm.engine.turn.transitions.Classifier``), never at this package.
"""

from .message_pipeline import MessagePipeline

__all__ = ["MessagePipeline"]
