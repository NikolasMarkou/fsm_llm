"""Plan 51fde95b step 13: the Builder improvements (D-038), one class per sub-step."""

from __future__ import annotations

import re
from pathlib import Path

from fsm_llm.monitor.constants import MAX_AGENT_ITERATIONS, MAX_AGENT_TIMEOUT_SECONDS

_MONITOR = Path(__file__).parent.parent.parent / "src" / "fsm_llm" / "monitor"


def _html() -> str:
    return (_MONITOR / "templates" / "index.html").read_text(encoding="utf-8")


def _input_attrs(element_id: str) -> str:
    match = re.search(rf'<input[^>]*id="{element_id}"[^>]*>', _html())
    assert match is not None, element_id
    return match.group(0)


class TestTheLaunchFormLimitsMatchTheServer:
    """13.1: the editable limits of the built-agent launch form mirror the request bounds."""

    def test_the_iterations_and_timeout_inputs_carry_the_server_maximums(self):
        iters = re.search(r'\bmax="(\d+)"', _input_attrs("builder-agent-iters"))
        timeout = re.search(r'\bmax="(\d+)"', _input_attrs("builder-agent-timeout"))
        assert iters and int(iters.group(1)) == MAX_AGENT_ITERATIONS
        assert timeout and int(timeout.group(1)) == int(MAX_AGENT_TIMEOUT_SECONDS)
