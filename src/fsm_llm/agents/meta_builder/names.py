"""The three names every module of the meta-builder agent reads (plan d454fd62 step 90; the old ``agents/meta_builder.py``, same dotted path).

``_S`` (the FSM state names), ``_K`` (the context keys) and ``_TYPE_VALUES`` (the artifact type values)
are the module aliases the old module defined once at its top; the methods spell them by these names,
so they are defined here once and imported by name.
"""

from __future__ import annotations

from fsm_llm.agents.constants import MetaBuilderStates, MetaContextKeys
from fsm_llm.agents.definitions import ArtifactType

_S = MetaBuilderStates
_K = MetaContextKeys
_TYPE_VALUES = frozenset(t.value for t in ArtifactType)
