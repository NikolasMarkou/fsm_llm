"""
MetaBuilderAgent: builds an FSM, workflow or agent artifact from a description.

The agent drives the meta-builder FSM (``build_meta_builder_fsm``) on a core
``API``: ``classify`` (the artifact type, a ``classification_extractions``
call), ``collect`` (the reply that gathers requirements, Pass 2), ``build``
(one structured completion that returns the whole artifact spec as JSON),
``build_failed`` and ``done``. Every model call is core's: this module makes
no LLM call of its own.

The conversation is the one source of session state: the FSM state says
whether the build is done, and the context holds the artifact type, the
requirements, and every build output (the assembled artifact, its validation
errors, progress and summary). Python keeps what an FSM cannot express: the
keyword hints the driver computes from each message (``build_requested``,
``type_switch``, ``keyword_type``) and writes before the turn, the handlers
that write the build request and assemble each build reply on a fresh
``ArtifactBuilder``, the send counter behind ``max_turns``, and the canned
texts (welcome, build complete, build failed, collect fallback, and the build
prompt that closes every collect reply when the model dropped it).

The tool registries in ``meta_tools.py`` (``create_fsm_tools`` and friends)
are a separate public API for driving a builder programmatically; this agent
does not use them.

Package layout (plan d454fd62 step 90; the old ``agents/meta_builder.py``, same dotted path):

- ``agent`` - ``MetaBuilderAgent``, the shell (construction and the session attributes), composed of
  the six mixins below
- ``hints`` - ``HintsMixin``: the keyword hints, pure functions of one message
- ``handlers`` - ``HandlersMixin``: the core ``API`` over the meta FSM and its four handlers
- ``assembly`` - ``AssemblyMixin``: one build reply assembled on a fresh builder
- ``runs`` - ``RunMixin``: ``run`` and the ``build`` step
- ``session`` - ``SessionMixin``: the turn-by-turn session and the build prompt
- ``results`` - ``ResultsMixin``: completion, the result, the monitor's state, ``run_interactive``
- ``specs`` - the typed specs of a build reply (module level)
- ``names`` - the three aliases every module reads (``_S``, ``_K``, ``_TYPE_VALUES``)
- ``surface`` - ``_MetaBuilderSurface``, the declarations the mixins share (mypy only)

This ``__init__`` re-exports the public names with a static ``__all__``. A private name is imported from
its defining module (``fsm_llm.agents.meta_builder.hints._normalize_message``); a patch aims at the
module that performs the lookup (``fsm_llm.agents.meta_builder.handlers.API``), never this package; a
loguru record is named after its module (``fsm_llm.agents.meta_builder.handlers``, ``.assembly``,
``.runs``, ``.session``), not ``fsm_llm.agents.meta_builder``.
"""

from __future__ import annotations

from fsm_llm.agents.definitions import MetaBuilderConfig

from .agent import MetaBuilderAgent

__all__ = ["MetaBuilderAgent", "MetaBuilderConfig"]
