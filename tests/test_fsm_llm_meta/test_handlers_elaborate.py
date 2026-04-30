from __future__ import annotations

"""Elaborate tests for meta-agent handlers.

The handlers module was removed when meta-agent moved to fsm_llm_agents.
"""


class TestHandlersElaborateModule:
    def test_importable(self):
        from fsm_llm.stdlib.agents.meta_builder import MetaBuilderAgent  # noqa: F401
