from __future__ import annotations

"""Tests for meta-agent handlers module.

The handlers module was removed when meta-agent moved to fsm_llm_agents.
This test verifies the new location is importable.
"""


class TestHandlersModule:
    def test_importable(self):
        from fsm_llm_agents.meta_builder import MetaBuilderAgent  # noqa: F401
