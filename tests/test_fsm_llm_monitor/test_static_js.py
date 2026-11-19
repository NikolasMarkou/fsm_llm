"""Node probes of the monitor's plain ES modules (plan 52ce5464 D-009).

`static_probes/*.mjs` import the real modules from a copy of `monitor/static` that carries
a `package.json` of `"type": "module"`, run on a DOM stand-in, and print one JSON line each.
No bundler, no npm, no network. Without `node` 18 or newer the probes skip; on a Linux GitHub Actions
leg that fails instead, so a CI skip cannot hide a regression.
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


def _usable_node() -> str | None:
    """The `node` on PATH when it is 18 or newer (the modules use ?. and top-level await)."""
    path = shutil.which("node")
    if path is None:
        return None
    try:
        out = subprocess.run(
            [path, "--version"], capture_output=True, encoding="utf-8", timeout=20
        ).stdout
        return path if int(out.strip().lstrip("v").split(".")[0]) >= 18 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


_NODE = _usable_node()


def test_node_is_present_on_linux_ci():
    if os.environ.get("GITHUB_ACTIONS") == "true" and sys.platform == "linux":
        assert _NODE, (
            "node 18 or newer is missing on a Linux CI leg: the JS probes would all skip"
        )


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    if _NODE is None:
        pytest.skip("node 18 or newer is not installed")
    root = tmp_path_factory.mktemp("monitor_js")
    (root / "package.json").write_text('{"type": "module"}', encoding="utf-8")
    shutil.copytree(
        _STATIC,
        root / "static",
        ignore=shutil.ignore_patterns("*.json", "*.md", "*.css"),
    )
    shutil.copytree(_PROBES, root / "probes")

    seen: dict[str, dict] = {}

    def run(name: str) -> dict:
        if name in seen:
            return seen[name]
        done = subprocess.run(
            [_NODE, "--no-warnings", str(root / "probes" / f"{name}.mjs")],
            capture_output=True,
            encoding="utf-8",
            timeout=60,
            cwd=root,
        )
        assert done.returncode == 0, done.stderr
        seen[name] = json.loads(done.stdout.strip().splitlines()[-1])
        return seen[name]

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


class TestLaunchModal:
    def test_the_button_stays_disabled_until_the_modal_closes(self, probe):
        out = probe("launch_modal")
        for name, posts in (("agent", 1), ("fsm", 2), ("workflow", 1)):
            assert out[f"{name}_ok"]["returned"] is True
            assert out[f"{name}_ok"]["window"] == [True, posts]
            assert out[f"{name}_ok"]["end"][0] is False

    def test_a_refused_launch_re_enables_the_button_at_once(self, probe):
        out = probe("launch_modal")
        for name in ("agent", "fsm", "workflow"):
            assert out[f"{name}_failed"]["returned"] is False


class TestBuilderAgentLaunch:
    def test_a_refused_launch_keeps_the_form_open_for_a_retry(self, probe):
        out = probe("builder_launch")
        assert out["opened"] == ["block", "Confirm Launch"]
        assert out["failed"] == {
            "form": "block",
            "label": "Confirm Launch",
            "disabled": False,
            "posts": 1,
        }

    def test_an_accepted_launch_closes_the_form(self, probe):
        retried = probe("builder_launch")["retried"]
        assert retried == {
            "form": "none",
            "label": "Launch AGENT",
            "disabled": False,
            "posts": 2,
        }


class TestChatInputAndMarkdownBlocks:
    """Plan 51fde95b step 12.1: the two quirks plan 52ce5464 carried."""

    def test_the_input_is_disabled_only_for_the_conversation_being_answered(
        self, probe
    ):
        out = probe("input_state")
        assert out["sendingA"] is True
        assert out["shownB"] is False
        assert out["after"] is False

    def test_a_send_in_another_conversation_is_refused_and_keeps_the_text(self, probe):
        refused = probe("input_state")["refused"]
        assert refused == {
            "value": "typed in B",
            "posts": 1,
            "toast": "Another message is still being answered",
        }

    def test_text_after_a_block_opens_its_own_paragraph(self, probe):
        out = probe("markdown_blocks")
        assert out["html"]["list_then_text"] == (
            "<ul><li>a</li><li>b</li></ul><p>After the list</p>"
        )
        assert out["html"]["ordered_then_text"].endswith("</ol><p>rest</p>")
        assert out["html"]["code_then_text"].endswith("</pre><p>after</p>")
        assert out["html"]["rule_then_text"] == '<hr class="md-hr"><p>foo</p>'
        assert all(
            out["balanced"][name]
            for name in ("list_then_text", "ordered_then_text", "code_then_text")
        )

    def test_blocks_alone_and_plain_paragraphs_are_unchanged(self, probe):
        out = probe("markdown_blocks")
        assert out["html"]["list_only"] == "<ul><li>a</li></ul>"
        assert out["html"]["plain_paragraphs"] == "<p>Para one</p><p>Para two</p>"

    def test_no_paragraph_is_opened_where_one_is_open_or_a_block_follows(self, probe):
        html = probe("markdown_blocks")["html"]
        assert html["list_gap_text"] == "<ul><li>a</li></ul><p>After</p>"
        assert html["block_run"] == (
            '<pre class="md-code-block"><code>x</code></pre><ul><li>a</li></ul>'
            '<ol><li>b</li></ol><hr class="md-hr">'
            '<pre class="md-code-block"><code>y</code></pre>'
        )


class TestUsageTiles:
    """Plan 51fde95b step 12.2: the LLM usage row of the dashboard."""

    def test_the_tiles_show_once_a_meter_reports_and_hide_when_none_does(self, probe):
        out = probe("usage_tiles")
        assert out["withUsage"] == {"display": "", "tiles": ["7", "12", "2"]}
        assert out["without"]["display"] == "none"


class TestConversationSignals:
    """Plan 51fde95b step 12.3a: the drawer shows the cut-off flag and the classifier records."""

    def test_the_drawer_shows_them_escaped_and_only_when_present(self, probe):
        out = probe("conv_signals")
        assert out["none"] == [False, False, False]
        assert out["some"] == [True, True, True, True]
        assert out["raw_tags"] is False


class TestEventLogView:
    """Plan 51fde95b step 12.3b: the Event log panel of the Logs page."""

    def test_the_list_says_when_there_is_no_log_and_escapes_ids(self, probe):
        out = probe("eventlog_view")
        assert out["unavailable"] is True
        assert out["list"] == [True, True, False]

    def test_the_timeline_is_escaped_summarised_and_says_when_cut(self, probe):
        out = probe("eventlog_view")
        # summary, cut note, escaped text, no raw tag, header row plus two events
        assert out["view"] == [True, True, True, False, 3]
        assert out["complete"] is True

    def test_a_refusal_and_the_id_in_the_path(self, probe):
        out = probe("eventlog_view")
        assert out["failure"] is True
        assert out["path"] == "/api/eventlog/a%20b%2Fc"
