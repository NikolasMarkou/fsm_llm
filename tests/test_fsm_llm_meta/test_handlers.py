from __future__ import annotations

"""Tests for meta-agent handlers module.

Most handler logic has been replaced by the ReactAgent-based architecture.
This module verifies the handlers module is importable and the module
is present for backward compatibility.
"""


class TestHandlersModule:
    def test_importable(self):
        import fsm_llm_meta.handlers  # noqa: F401
