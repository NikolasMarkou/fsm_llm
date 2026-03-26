from __future__ import annotations

"""Elaborate tests for meta-agent handlers.

Most handler logic has been replaced by the ReactAgent-based architecture.
Handler-level tests are now covered by test_tools.py (builder tool tests).
"""


class TestHandlersElaborateModule:
    def test_importable(self):
        import fsm_llm_meta.handlers  # noqa: F401
