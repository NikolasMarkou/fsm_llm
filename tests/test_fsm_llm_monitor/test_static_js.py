"""Node probes of the monitor's plain ES modules (plan 52ce5464 D-009).

`static_probes/*.mjs` import the real modules from a copy of `monitor/static` that carries
a `package.json` of `"type": "module"`, run on a DOM stand-in, and print one JSON line each.
No bundler, no npm, no network. Without `node` the probes skip; on a Linux GitHub Actions
leg a missing `node` fails, so a CI skip cannot hide a regression.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_STATIC = Path(__file__).parent.parent.parent / "src" / "fsm_llm" / "monitor" / "static"
_PROBES = Path(__file__).parent / "static_probes"
_NODE = shutil.which("node")


def test_node_is_present_on_linux_ci():
    if os.environ.get("GITHUB_ACTIONS") == "true" and sys.platform == "linux":
        assert _NODE, "node is missing on a Linux CI leg: the JS probes would all skip"


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    if _NODE is None:
        pytest.skip("node is not installed")
    root = tmp_path_factory.mktemp("monitor_js")
    (root / "package.json").write_text('{"type": "module"}')
    shutil.copytree(
        _STATIC,
        root / "static",
        ignore=shutil.ignore_patterns("*.json", "*.md", "*.css"),
    )
    shutil.copytree(_PROBES, root / "probes")

    def run(name: str) -> dict:
        done = subprocess.run(
            [_NODE, "--no-warnings", str(root / "probes" / f"{name}.mjs")],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=root,
        )
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout.strip().splitlines()[-1])

    return run


class TestMarkdown:
    def test_code_survives_the_inline_passes(self, probe):
        out = probe("markdown")
        assert out["fence_keeps_code"]
        assert out["inline_keeps_code"] == (
            '<p>use <code class="md-code-inline">snake_case_a</code> and '
            '<code class="md-code-inline">a*b*c</code> now</p>'
        )

    def test_ordered_runs_are_an_ol(self, probe):
        out = probe("markdown")
        assert "<ol><li>first</li><li>second</li></ol>" in out["ordered"]
        assert out["unordered"] == "<ul><li>a</li><li>b</li></ul>"

    def test_escape_still_comes_first(self, probe):
        out = probe("markdown")
        assert "<img" not in out["xss_fence"] and "&lt;img" in out["xss_fence"]
        assert "<script" not in out["xss_inline"]
        assert "\x00" not in out["nul_forged"] and out["nul_forged"].startswith(
            "<p>a0b "
        )


class TestChatAndTargets:
    def test_a_reply_changes_only_the_conversation_still_shown(self, probe):
        out = probe("chat_switch")
        assert out["stay"] == {"badge": "END_A", "input_display": "none", "bubbles": 1}
        assert out["switched"] == {
            "badge": "stateB",
            "input_display": "block",
            "bubbles": 0,
        }

    def test_send_event_targets_are_the_runs_that_can_take_one(self, probe):
        options = probe("send_targets")["options"]
        assert options == [["", False], ["live-1", True], ["wait-1", False]]

    def test_an_activity_row_opens_its_conversation_once(self):
        app = (_STATIC / "app.js").read_text(encoding="utf-8")
        block = app.split("'activity-row-click'")[1].split("// Control Center")[0]
        assert "setTimeout" not in block
        assert block.count("showConversationInDrawer(instanceId, itemId)") == 1


class TestDashboard:
    def test_a_new_config_replaces_the_whole_custom_section(self, probe):
        out = probe("custom_panels")
        assert out["one"] == {
            "section": True,
            "title": "Config ONE",
            "desc": True,
            "panels": ["P1"],
            "alerts": ["ALERT ONE"],
        }
        assert out["two"] == {
            "section": True,
            "title": "Config TWO",
            "desc": False,
            "panels": ["P2"],
            "alerts": None,
        }
        assert out["three"]["section"] is False
