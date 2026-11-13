"""State executors: what a state's kind does in Pass 1 of a turn.

The turn pipeline (``engine/turn/``) is kind-agnostic: at the one Pass-1 seam it
asks ``executor_for(state)`` for the state's executor and calls it. An executor
owns the three things that differ between kinds (roadmap row 0.14, plan
d454fd62, D-004, D-015):

- ``extract``: the state's Pass-1 extraction, a ``DataExtractionResponse``.
- ``cleans_extracted``: whether the extracted data goes through
  ``clean_context_keys`` before it is committed.
- ``after_transition``: what runs after a transition INTO a state of the kind
  (the conversation kind re-extracts what the same message holds for the new
  state; the other kinds do nothing).

Three kinds, two registrations: ``conversation`` (the field, bulk and classification
extraction, owned by the executor since step 80: ``conversation/``), ``completion``
(one ``LLMInterface.complete`` call over a consumer-owned transcript, ``completion.py``) and
``synthesize`` (passive, no registration: it runs as conversation, because a runner outside core
pushes the compiled procedure and a bare ``converse`` on such a state stays BLOCKED, D-002 of
plan 2cefeb61; a later synthesize executor registers under its own name and replaces the
fallback).

Layer ``engine`` (plan d454fd62, steps 53 to 56 and 80): the old flat ``fsm_llm.executors`` is
this package; the old dotted path is gone. Modules:

- ``base.py``: ``StateExecutor``.
- ``services.py``: ``_PipelineServices``, the protocol of what an executor may ask of the pipeline.
- ``registry.py``: the module table, ``register_executor``, ``check_executors``, ``executor_for``.
- ``completion.py``: ``CompletionExecutor``.
- ``conversation/``: ``ConversationExecutor`` and the conversation Pass-1 machinery.
- ``records.py``: the records and digests both the executors and the turn pipeline use.

Interface contract:

- ``StateExecutor.extract(pipeline, instance, user_message, conversation_id,
  state)``: ``pipeline`` is the calling ``MessagePipeline``, seen through ``_PipelineServices``
  (the executor reads ``pipeline.llm_interface`` and the other members listed there, nothing
  else, and never imports ``engine/turn``); ``state`` is the instance's current state. Must
  set ``instance.last_extraction_response`` to what it returns, take no snapshot
  of its own (Pass 1 runs outside the PRE and POST atomicity blocks of the turn
  body; the anchors in ``engine/turn/driver.py`` hold the rule) and may raise
  ``LLMResponseError``.
- ``StateExecutor.after_transition(pipeline, instance, user_message,
  conversation_id, extraction_response, previous_state, new_state)``: called
  once after a transition that ended in ``new_state`` (not for an agent-managed
  FSM); may update ``extraction_response`` and the context; returns ``None``.
- ``executor_for(state, executors=None)``: the executor of
  ``kind_of(state).name``: the per-instance ``executors`` map first, then the
  module table; a kind with neither runs as ``conversation`` (a passive kind),
  resolved through the per-instance ``conversation`` entry first, then the
  module's, so an override of ``conversation`` covers ``synthesize`` and a
  runtime kind nobody registered.
- ``register_executor(kind_name, executor)``: add or replace the executor of a
  kind in the MODULE table, which every pipeline in the process reads (a test
  kind registers its own beside its ``KindSpec``; the caller removes it again).
- ``check_executors(executors)``: validate the ``executors=`` argument of
  ``MessagePipeline``, ``FSMManager`` and ``API`` and return a read-only copy.
  A ``{kind_name: StateExecutor}`` mapping, or ``None`` (empty). ``TypeError``
  for a value that is not a mapping or a mapping value that is not a
  ``StateExecutor``; ``ValueError`` for a name no registered ``KindSpec``
  carries (a typo would otherwise be a silent no-op). The map belongs to one
  pipeline, is never written after construction (safe to read from any
  thread) and overrides the module table for that pipeline only; the spec
  registry (``state_kinds.KINDS``) stays static and process-wide.

This package sits BELOW ``engine/turn/``: it imports core, llm, context and handlers modules and
never ``engine/turn/`` (the pipeline imports it; the contract ``engine-kinds-imports-no-turn-or-
manager-or-api`` of ``.importlinter`` holds it). The helpers both need live here
(``records.py``, ``fsm_llm.context``'s cut-off record). A private name is imported from its
defining module and a patch aims at the module that performs the lookup, never at this package.
``fsm_llm/__init__.py`` does not import it; the registry is internal until the kind extension
point is public (roadmap C10).
"""

from .base import StateExecutor
from .completion import CompletionExecutor
from .conversation import ConversationExecutor
from .registry import check_executors, executor_for, register_executor

__all__ = [
    "CompletionExecutor",
    "ConversationExecutor",
    "StateExecutor",
    "check_executors",
    "executor_for",
    "register_executor",
]
