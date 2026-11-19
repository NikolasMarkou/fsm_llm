"""Node probes of the monitor's plain ES modules (plan 52ce5464 D-009).

`static_probes/*.mjs` import the real modules from a copy of `monitor/static` that carries
a `package.json` of `"type": "module"`, run on a DOM stand-in, and print one JSON line each.
No bundler, no npm, no network. Without `node` 18 or newer the probes skip; on a Linux GitHub Actions
leg that fails instead, so a CI skip cannot hide a regression.
"""

import json
import os
import re
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


class TestBuilderAgentLaunchFields:
    """Plan 51fde95b step 13.1: the model of a launched built agent is the
    Builder field, then the launch form value, never ``artifact.config.model``
    (a template value of every built agent); the label and the two limits are
    editable fields pre-filled from the artifact."""

    def test_the_model_is_the_builder_field_not_the_artifact_config(self, probe):
        out = probe("builder_agent_fields")["fromBuilderField"]
        assert out["prefilled"]["model"] == "ollama_chat/my-local-model"
        assert out["sent"]["model"] == "ollama_chat/my-local-model"

    def test_the_form_value_wins_and_every_field_is_sent_as_typed(self, probe):
        sent = probe("builder_agent_fields")["typedInForm"]["sent"]
        assert sent["model"] == "ollama_chat/other"
        assert (sent["label"], sent["max_iterations"], sent["timeout_seconds"]) == (
            "Mine",
            3,
            45,
        )

    def test_an_empty_model_sends_none_so_the_server_default_applies(self, probe):
        sent = probe("builder_agent_fields")["emptyField"]["sent"]
        assert "model" not in sent  # not gpt-4o-mini from the artifact

    def test_the_fields_are_pre_filled_from_the_artifact(self, probe):
        pre = probe("builder_agent_fields")["emptyField"]["prefilled"]
        assert pre == {
            "label": "Weather bot",
            "model": "",
            "iters": 7,
            "timeout": 300,
        }

    def test_a_bad_artifact_limit_falls_back_to_the_default(self, probe):
        pre = probe("builder_agent_fields")["badConfig"]["prefilled"]
        assert (pre["iters"], pre["timeout"]) == (10, 120)


class TestBuilderAgentLaunchGate:
    """Plan 51fde95b step 13.2: Launch is disabled, with a note, for an agent
    type ``/api/capabilities`` does not list; unknown capabilities leave it on."""

    def test_a_listed_type_stays_launchable(self, probe):
        out = probe("builder_agent_gate")
        assert out["listed"]["disabled"] is False
        assert out["listed"]["noteShown"] == "none"
        assert out["backToListed"]["disabled"] is False  # a later result re-enables it

    def test_an_unlisted_type_is_disabled_with_a_note_and_no_request(self, probe):
        out = probe("builder_agent_gate")
        for key in ("unlisted", "unlistedPromptChain"):
            assert out[key]["disabled"] is True
            assert out[key]["noteShown"] == "block"
            assert "cannot be launched" in out[key]["note"]
            assert out[key]["posts"] == 0

    def test_a_capabilities_reply_after_a_reset_is_dropped(self, probe):
        stale = probe("builder_agent_gate")["staleAfterReset"]
        assert stale == {"disabled": False, "note": ""}

    def test_a_missing_list_or_a_failed_lookup_leaves_launch_on(self, probe):
        out = probe("builder_agent_gate")
        assert out["noList"]["disabled"] is False
        assert out["capsDown"]["disabled"] is False


class TestBuilderFsmLaunchOpensTheConversation:
    """Plan 51fde95b step 13.3: the conversation the launch starts opens in the drawer."""

    def test_the_started_conversation_is_selected_after_the_switch_to_control(
        self, probe
    ):
        out = probe("builder_fsm_launch")["withConversation"]
        assert (out["instance"], out["conv"]) == ("inst-1", "conv-77")
        assert "control" in out["pages"]

    def test_a_start_reply_with_no_conversation_only_switches_page(self, probe):
        out = probe("builder_fsm_launch")["withoutConversation"]
        assert (out["instance"], out["conv"], out["pages"]) == (None, None, ["control"])


class TestBuilderStartSendsMaxTokens:
    """Plan 51fde95b step 13.4: the Max Tokens field reaches the start request."""

    def test_the_field_is_sent_and_an_unusable_value_falls_back_to_the_default(
        self, probe
    ):
        assert probe("builder_start")["tokens"] == [8000, 4096, 4096]


class TestBuilderCancel:
    """Plan 51fde95b step 13.5: Cancel aborts the send and drops the session with
    ``?force=true``; the elapsed timer runs only while a send is in flight."""

    def test_a_running_send_shows_cancel_and_the_elapsed_time(self, probe):
        out = probe("builder_cancel")["whileSending"]
        assert out["cancelShown"] == ""
        assert re.search(r"Building artifact\.\.\. \d+ s", out["status"])

    def test_cancel_aborts_the_request_and_force_deletes_the_session(self, probe):
        out = probe("builder_cancel")["cancelled"]
        assert out["deletes"] == ["/api/builder/s1?force=true"]
        assert (out["aborted"], out["cancelBtn"], out["input"]) == (
            True,
            "none",
            "none",
        )
        assert out["bubble"] and "Cancelled" in out["status"]
        assert out["noAbortBubble"]  # the aborted request is not shown as an error

    def test_a_reset_during_a_send_aborts_and_force_deletes_too(self, probe):
        out = probe("builder_cancel")["reset"]
        assert out["deletes"] == ["/api/builder/s2?force=true"]
        assert (out["aborted"], out["cancelBtn"]) == (True, "none")

    def test_starting_again_during_a_send_aborts_and_force_deletes_the_old_session(
        self, probe
    ):
        out = probe("builder_cancel")["restart"]
        assert out == {"deletes": ["/api/builder/s3?force=true"], "aborted": True}

    def test_a_finished_send_stops_the_timer_and_cancel_is_then_a_no_op(self, probe):
        out = probe("builder_cancel")
        assert out["finished"] == {"cancelBtn": "none", "timerStopped": True}
        assert out["deletesAfterNoop"] == 4  # cancel, reset, restart, the next start


class TestBuilderEditJson:
    """Plan 51fde95b step 13.6: Edit JSON, Apply, then Launch sends the edited artifact."""

    def test_edit_opens_an_editor_on_the_artifact_text(self, probe):
        out = probe("builder_edit")["opened"]
        assert out == {
            "label": "Apply JSON",
            "editor": "block",
            "pre": "none",
            "text": "Built",
        }

    def test_a_parse_error_or_a_non_object_is_shown_and_the_editor_stays_open(
        self, probe
    ):
        out = probe("builder_edit")
        assert "Invalid JSON" in out["bad"]["status"]
        assert (out["bad"]["stillEditing"], out["bad"]["preUnchanged"]) == (
            "block",
            True,
        )
        assert "must be a JSON object" in out["array"]["status"]

    def test_launch_waits_for_the_edit_and_then_sends_the_edited_object(self, probe):
        out = probe("builder_edit")
        assert (
            out["openLaunch"]["posts"] == 0
            and "Apply the JSON edit" in out["openLaunch"]["status"]
        )
        assert out["applied"] == {
            "label": "Edit JSON",
            "editor": "none",
            "pre": "Edited",
            "launchLabel": "Launch FSM",  # the launch button is re-derived from the edit
            "summary": True,
        }
        assert (out["launched"], out["label"]) == (["Edited"], "Edited")


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


class TestJsonColouring:
    """Plan 51fde95b step 14 (D-040): the shared JSON colouring of `utils/json_highlight.js`."""

    def test_tokens_keep_every_byte_and_tell_a_key_from_a_string(self, probe):
        out = probe("json_highlight")
        assert out["flat"] == [[True, True, True]] * 15
        assert out["one"][1:4] == ['key:"s"', "punct:: ", 'string:"x"']
        assert [t for t in out["one"] if t[:4] in ("bool", "null")] == [
            "bool:true",
            "bool:false",
            "null:null",
        ]
        keyed = out["keyed"]
        assert 'key:"k"' in keyed and 'key:"sp"' in keyed and 'string:"b:c"' in keyed
        assert keyed.count('string:"k"') == 1 and 'key:"b:c"' not in keyed
        assert out["nums"] == [
            f"number:{n}" for n in "-1 0 1.5 -2.5e-3 1E+10 10 -0 0.25e2".split()
        ]
        assert out["esc"][:3] == [r'string:"a\"b"', r'string:"\\"', r'string:"C:\\"']

    def test_invalid_text_and_a_text_over_the_cap_stay_plain(self, probe):
        out = probe("json_highlight")
        assert out["fallback"] == [[True, True, False, 0]] * 9
        assert out["nonString"] == [True] * 4 and out["nullShown"] == ""
        assert out["max"] == 200000 and out["capRow"] == [True, True, True, False]

    def test_html_in_a_string_stays_text_and_deep_nesting_is_one_scan(self, probe):
        out = probe("json_highlight")
        assert out["xss"] == {"text": True, "spansOnly": True, "innerHTMLWrites": 0}
        assert out["writes"] == 0 and out["deepFast"]
        assert out["deepRow"] == [[True, 1, True], [True, 60003, True]]
        assert out["into"] == {
            "asked": ".json-view",
            "a": ['{"a": 1}', 5],
            "b": ["not json", 0],
        }
