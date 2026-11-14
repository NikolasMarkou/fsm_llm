"""
Artifact builders for incrementally constructing FSMs, Workflows, and Agents.

Each builder maintains partial state and provides:
- Mutation methods (add/remove/update)
- Incremental validation
- Progress tracking
- Summary generation for LLM context injection
- Final serialization to validated definitions

Package layout (plan d454fd62 step 89, D-041; the old ``agents/meta_builders.py``, same dotted path):
``base`` (``ArtifactBuilder``, the abstract builder), ``fsm`` (``FSMArtifactBuilder``), ``workflow``
(``WorkflowArtifactBuilder``) and ``agent`` (``AgentArtifactBuilder``). This ``__init__`` re-exports the
four public classes with a static ``__all__``; a patch aims at the module that performs the lookup
(``fsm_llm.agents.meta_builders.agent.AgentConfig``), never the package, and a loguru record is named
after its module (``fsm_llm.agents.meta_builders.fsm``).
"""

from __future__ import annotations

from .agent import AgentArtifactBuilder
from .base import ArtifactBuilder
from .fsm import FSMArtifactBuilder
from .workflow import WorkflowArtifactBuilder

__all__ = [
    "AgentArtifactBuilder",
    "ArtifactBuilder",
    "FSMArtifactBuilder",
    "WorkflowArtifactBuilder",
]
