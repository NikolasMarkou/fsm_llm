from __future__ import annotations

"""Tests for meta-agent FSM definitions module.

The meta-agent no longer uses a custom FSM. The module is kept for
backward compatibility. This test verifies the module is importable.
"""


class TestFSMDefinitionsModule:
    def test_importable(self):
        import fsm_llm_meta.fsm_definitions  # noqa: F401
