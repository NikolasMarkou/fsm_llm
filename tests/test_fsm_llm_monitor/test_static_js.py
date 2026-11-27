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

    def test_the_launch_body_carries_the_selected_executable_id(self, probe):
        assert probe("launch_modal")["agent_body"]["agent_type"] == "debate"


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

    def test_the_posted_type_is_the_artifact_id_as_it_is(self, probe):
        sent = probe("builder_agent_fields")["fromBuilderField"]["sent"]
        assert sent["agent_type"] == "react"  # an executable id, never a class name

    def test_a_type_that_needs_a_tool_is_refused_without_one_and_sends_nothing(
        self, probe
    ):
        assert probe("builder_agent_fields")["noTool"]["sent"] is None


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

    def test_an_inherited_key_is_no_agent_type_and_names_itself_in_the_note(
        self, probe
    ):
        out = probe("builder_agent_gate")
        assert out["inherited"]["disabled"] is True
        assert out["inherited"]["note"].startswith("constructor cannot be launched")
        assert out["protoKey"]["disabled"] is True
        assert out["protoKey"]["note"].startswith("__proto__ cannot be launched")

    def test_a_capabilities_reply_for_the_pre_edit_artifact_is_dropped(self, probe):
        stale = probe("builder_agent_gate")["staleAfterEdit"]
        assert stale == {"disabled": False, "note": ""}

    def test_a_missing_list_or_a_failed_lookup_leaves_launch_on(self, probe):
        out = probe("builder_agent_gate")
        assert out["noList"]["disabled"] is False
        assert out["capsDown"]["disabled"] is False


_HOSTILE = "<img src=x onerror=alert(1)>"


class TestAgentSelectsAreFilledFromTheTypesRoute:
    """Plan 659c07c3 step 6: both agent selects are built from GET /api/agent/types
    with createElement and textContent, never from a list in the page or markup."""

    def test_the_launch_select_offers_only_launchable_ids_in_two_groups(self, probe):
        launch = probe("agent_select")["launchSelect"]
        assert launch["shape"] == [
            {
                "group": "Needs tools",
                "options": [["auto_memory", "auto_memory"], ["react", "react"]],
            },
            {
                "group": "No tools needed",
                "options": [
                    ["debate", "debate"],
                    ["adapt", "adapt"],
                    [_HOSTILE, _HOSTILE],
                ],
            },
        ]
        assert launch["value"] == "react"  # the default id

    def test_a_hostile_id_is_text_and_never_markup(self, probe):
        out = probe("agent_select")
        assert out["launchSelect"]["innerHTML"] == ""
        assert out["viz"]["innerHTML"] == ""
        assert [_HOSTILE, _HOSTILE] in [o for o in out["viz"]["shape"]]

    def test_the_visualizer_select_offers_the_drawable_ids_and_defaults_to_react(
        self, probe
    ):
        viz = probe("agent_select")["viz"]
        assert [o[0] for o in viz["shape"]] == [
            "debate",
            "auto_memory",
            "react",
            "adapt",
            _HOSTILE,
            "evaluator_optimizer",
        ]  # swarm is listed with drawable false: absent
        assert viz["value"] == "react"
        assert viz["drawn"] == ["/api/agent/visualize?agent_type=react"]

    def test_the_tools_wrapper_follows_takes_tools(self, probe):
        out = probe("agent_select")
        assert out["launchSelect"]["wrapper"] == ""  # react
        assert out["noTools"] == "none"  # debate
        assert out["optionalTools"] == ""  # adapt takes optional tools

    def test_a_second_opening_asks_once_and_keeps_the_choice(self, probe):
        assert probe("agent_select")["reopened"] == {"value": "adapt", "fetches": 1}

    def test_a_type_that_needs_a_tool_is_refused_unless_the_file_names_its_tools(
        self, probe
    ):
        out = probe("agent_select")
        assert out["needsOne"]["posts"] == 0
        assert "Add at least one tool for react" in out["needsOne"]["status"]
        assert out["namedTools"]["posts"] == 1
        assert out["namedTools"]["body"]["agent_type"] == "auto_memory"

    @pytest.mark.parametrize("mode", ["empty", "junk", "badrows", "fail"])
    def test_no_rows_leave_one_disabled_option_and_no_stale_choice(self, probe, mode):
        out = probe("agent_select_empty")[mode]
        if mode != "fail":
            assert out["toast"] == []  # a payload with no usable row is no failure
        none = [["", "No agent types available", True]]
        assert out["launch"] == none
        assert out["viz"] == none
        assert (out["launchValue"], out["vizValue"]) == ("", "")

    def test_a_failed_fetch_toasts_once_and_the_next_opening_asks_again(self, probe):
        out = probe("agent_select_empty")
        assert out["fail"]["toast"] == ["Agent types failed to load: down"]
        assert out["recovered"]["asked"] == 1
        assert out["recovered"]["launchValue"] == "react"
        assert out["recovered"]["vizValue"] == "react"

    def test_with_no_agent_type_nothing_is_posted(self, probe):
        out = probe("agent_select_empty")["nothingToLaunch"]
        assert out["posted"] == 0
        assert "No agent type available" in out["status"]
        none = [
            ["", "No agent types available", True]
        ]  # after a filled select: no stale choice
        assert (out["launch"], out["viz"]) == (none, none)
        assert (out["launchValue"], out["vizValue"]) == ("", "")


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

    def test_a_damaged_log_says_where_the_reading_stopped(self, probe):
        assert probe("eventlog_view")["damaged"] == [True, False]

    def test_a_refusal_and_the_id_in_the_path(self, probe):
        out = probe("eventlog_view")
        assert out["failure"] is True
        assert out["path"] == "/api/eventlog/a%20b%2Fc"


class TestLaunchFlows:
    """Plan 1f0925d2 step 5 (F1, D-006): a created instance is never a failed launch; no stale timer."""

    def test_a_failed_first_start_says_the_instance_was_launched(self, probe):
        out = probe("stale_state")["fsm_partial"]
        text = out["after"]["status"]
        assert "Launch failed" not in text
        assert "Launched: L1, but starting the first conversation failed" in text
        assert "LLM operation timed out" in text
        # G4: the button stays disabled in the window; one launch POST; then Control, modal closed.
        assert out["after"]["disabled"] is True
        assert (out["launches"], out["ran"], out["pages"]) == (1, 1, ["control"])
        assert (out["modal"], out["disabled"]) == ("none", False)

    def test_a_refused_first_post_is_still_launch_failed(self, probe):
        out = probe("stale_state")["fsm_first_failed"]
        assert "Launch failed: bad preset" in out["status"]
        assert (out["disabled"], out["timers"], out["pages"], out["posts"]) == (
            False,
            0,
            [],
            1,
        )

    @pytest.mark.parametrize("name", ["fsm", "workflow", "agent"])
    def test_a_close_in_the_window_cancels_the_navigation(self, probe, name):
        out = probe("stale_state")
        assert out[f"{name}_stays_open"]["pages"] == ["control"]
        for mode in ("closed", "closed_and_reopened"):
            row = out[f"{name}_{mode}"]
            assert row["pages"] == [] and row["drawer"] == 0, mode
            assert row["disabled_in_window"] is True and row["disabled"] is False
        assert out[f"{name}_closed"]["modal"] == "none"
        assert out[f"{name}_closed_and_reopened"]["modal"] == "flex"

    def test_a_close_while_the_start_is_in_flight_navigates_nowhere(self, probe):
        out = probe("stale_state")["fsm_closed_during_start"]
        assert (out["pages"], out["drawer"], out["disabled"]) == ([], 0, False)

    def test_a_failed_capabilities_fetch_blocks_nothing_and_is_shown_once(self, probe):
        out = probe("stale_state")["capabilities"]
        assert out["defaults"] == {"fsm": True, "workflows": True, "agents": True}
        assert out["kept"] == out["defaults"]
        assert out["toasts"] == 1
        assert out["workflow_posts"] == 1
        assert "not installed" not in out["workflow_status"]

    def test_the_tool_template_options_are_built_with_textcontent(self, probe):
        out = probe("stale_state")["tool_templates"]
        assert out["count"] == 16 and out["innerHTML"] == ""
        assert out["first"] == ["", "-- Add a tool from templates --"]
        assert out["search"] == [
            "search",
            "search \u2014 Search the web for information on a topic",
        ]

    def test_the_builder_fsm_launch_keeps_the_launched_wording(self, probe):
        out = probe("stale_state")["builder_partial"]
        assert "Launch failed" not in out["status"]
        assert (
            "Launched: L1, but starting the first conversation failed" in out["status"]
        )
        assert (out["disabled"], out["pages_before"]) == (True, [])
        assert (out["pages"], out["disabled_after"], out["launches"]) == (
            ["control"],
            False,
            1,
        )

    def test_two_clicks_on_the_builder_launch_make_one_fsm(self, probe):
        out = probe("stale_state")["builder_double"]
        assert out["inFlight"] == {"disabled": True, "launches": 1}
        assert (out["launches"], out["disabled_after"]) == (1, False)

    def test_a_refused_builder_launch_re_enables_the_button_for_a_retry(self, probe):
        out = probe("stale_state")["builder_first_failed"]
        assert out["disabled"] is True and out["disabled_after"] is False
        assert "Launch failed: bad definition" in out["status"]
        assert (out["timers"], out["retry_launches"]) == (0, 2)

    def test_the_launch_page_has_one_timer_and_builds_no_options_with_markup(self):
        text = (_STATIC / "pages" / "launch.js").read_text(encoding="utf-8")
        assert (
            text.count("setTimeout(") == 1
        )  # _afterLaunch: every path shares its guard
        assert "innerHTML +=" not in text


class TestConversationDetailRaces:
    """Plan 1f0925d2 step 6 (F2, D-007): a superseded failure paints nothing, a failure says why, focus follows the view."""

    def test_a_superseded_failure_does_not_replace_the_newer_view(self, probe):
        out = probe("stale_state")["detail_superseded"]
        assert out["before"]["live"] is True and out["before"]["input"] == "block"
        assert out["after"] == out["before"]
        assert out["selected"] == "B"

    def test_the_current_failure_shows_the_server_reason_escaped_and_hides_the_input(
        self, probe
    ):
        out = probe("stale_state")["detail_current_fails"]
        assert out["error"] == "Failed to load conversation: ended &lt;b&gt;A&lt;/b&gt;"
        assert out["input"] == "none"

    def test_a_failure_after_the_drawer_went_back_paints_nothing(self, probe):
        out = probe("stale_state")["detail_after_back"]
        assert out["prior"] is True and out["error"] is None and out["input"] == "block"

    def test_focus_moves_only_while_the_same_visible_conversation_is_shown(self, probe):
        out = probe("stale_state")["focus"]
        assert out["stays"] == {"focused": 1, "disabled": False}
        # the shared input is released in every case, focused only when it is this reply's to take
        assert out["switched"] == {"focused": 0, "disabled": False}
        assert out["terminal"] == {"focused": 0, "disabled": False}


class TestEventLogLoadMore:
    """Plan 1f0925d2 step 6 (F2, D-007): Load more by `since_seq`, a request token, errors keep the earlier pages."""

    def test_load_more_appends_the_next_pages_and_ends_with_a_marker(self, probe):
        out = probe("stale_state")["eventlog_more"]
        # the first page is not asked again; each later page starts after the last seq shown
        assert out["asked"] == [
            "GET /api/eventlog/abc",
            "GET /api/eventlog/abc?since_seq=200",
            "GET /api/eventlog/abc?since_seq=400",
        ]
        assert out["first"] == {
            "seqs": 200,
            "more": True,
            "notice": True,
            "end": False,
        }
        assert out["second"]["seqs"] == 400 and out["second"]["more"] is True
        assert out["second"]["notice"] is True and out["second"]["headers"] == 1
        third = out["third"]
        assert (third["rows"], third["unique"], third["last"]) == (450, 450, 450)
        assert third["more"] is False and third["notice"] is False
        assert third["end"] is True and third["headers"] == 1
        assert out["asked_after_end"] == 0

    def test_hostile_text_in_a_later_page_is_escaped(self, probe):
        second = probe("stale_state")["eventlog_more"]["second"]
        assert second["escaped"] is True and second["raw"] is False

    def test_a_second_click_while_a_page_is_in_flight_asks_once(self, probe):
        out = probe("stale_state")["eventlog_double_click"]
        assert out["asked"] == [
            "GET /api/eventlog/abc",
            "GET /api/eventlog/abc?since_seq=200",
        ]
        assert out["rows"] == 400

    def test_a_failed_page_keeps_the_earlier_pages_and_can_be_retried(self, probe):
        out = probe("stale_state")["eventlog_error"]
        assert out["rows"] == 200 and out["error"] is True and out["raw"] is False
        assert out["more"] is True
        assert out["rows_after"] == 400 and out["error_after"] is False

    def test_a_slow_log_never_paints_over_a_newer_one(self, probe):
        out = probe("stale_state")["eventlog_open_race"]
        assert out["b_shown"] is True and out["final_b"] is True
        assert out["final_rows"] == 200

    def test_a_late_page_of_a_replaced_log_is_dropped_and_logs_are_never_mixed(
        self, probe
    ):
        out = probe("stale_state")["eventlog_more_race"]
        assert out["rows"] == 200 and out["b_only"] is True
        assert out["next_rows"] == 400 and out["next_b_only"] is True
        assert out["asked"] == [
            "GET /api/eventlog/A?since_seq=200",
            "GET /api/eventlog/B?since_seq=200",
        ]

    def test_a_stale_failed_open_paints_nothing(self, probe):
        assert probe("stale_state")["eventlog_stale_failure"] == {"b_only": True}

    def test_every_column_of_a_row_and_the_head_are_escaped(self, probe):
        out = probe("stale_state")["eventlog_hostile"]
        assert out == {"raw": [], "escaped": 5}

    def test_a_load_more_click_while_another_log_opens_does_not_supersede_that_open(
        self, probe
    ):
        out = probe("stale_state")["eventlog_more_during_open"]
        assert out["b_only"] is True
        assert out["asked"] == ["GET /api/eventlog/A", "GET /api/eventlog/B"]

    def test_the_rendered_rows_are_bounded_and_the_view_says_where_the_log_continues(
        self, probe
    ):
        out = probe("stale_state")["eventlog_cap"]
        assert out["rows"] == 2000 and out["more"] is False
        assert (
            out["line"]
            == "Showing the first 2000 events; the log continues after seq 2000"
        )
        assert out["asked"] == 10  # the open and nine pages: no request past the cap

    def test_a_failed_open_leaves_no_load_more_to_click(self, probe):
        out = probe("stale_state")["eventlog_failed_open"]
        assert out == {"error": True, "more": False, "asked_more": 0}

    def test_the_action_is_wired_and_the_row_cap_is_one_named_constant(self):
        app = (_STATIC / "app.js").read_text(encoding="utf-8")
        assert "'eventlog-more':" in app and "eventlog.loadMoreEventLog()" in app
        text = (_STATIC / "pages" / "eventlog.js").read_text(encoding="utf-8")
        assert text.count("EVENTLOG_MAX_ROWS") >= 2
        assert "innerHTML +=" not in text


class TestControlDrawerStepDataAndDetails:
    """Plan 1f0925d2 step 7 (F3, D-009): step data whole or a labelled raw cut, the details state across a repaint."""

    def test_a_small_step_data_is_shown_whole_and_parses(self, probe):
        out = probe("stale_state")["step_data_small"]
        assert out["length"] > 300  # the old cut was 300 characters
        assert out["whole"] is True and out["parses"] is True and out["note"] is False

    def test_the_cap_is_inclusive_and_one_more_character_is_the_labelled_cut(
        self, probe
    ):
        out = probe("stale_state")["step_data_edge"]
        assert out["edge"] == 20000
        assert out["at_cap_whole"] is True and out["over_cut"] is True

    def test_a_huge_step_data_is_a_labelled_escaped_raw_cut_never_json(self, probe):
        out = probe("stale_state")["step_data_huge"]
        assert out["label"] is True and out["count"] is True and out["shown"] is True
        assert out["in_json_view"] is False  # the cut text is never presented as JSON
        assert out["raw_tags"] is False and out["escaped"] is True

    def test_an_opened_details_survives_a_repaint_and_a_closed_one_stays_closed(
        self, probe
    ):
        out = probe("stale_state")["details_state"]
        assert out["count"] == 1 and out["repainted"] is True
        assert out["closed_before"] == [False] and out["closed_after"] == [False]
        assert out["opened_after"] == [True] and out["unchanged"] == [True]
        assert out["same"] is True


class TestBuilderStartInputAndStatusTiming:
    """Plan 1f0925d2 step 7 (F3, D-009): the input waits for the start reply, an edit with no states hides the graph."""

    def test_the_input_area_is_shown_only_after_the_start_reply(self, probe):
        out = probe("stale_state")["builder_input_start"]
        assert out == {"pending": "none", "after": "flex"}

    def test_a_failed_start_leaves_the_input_hidden_and_shows_the_error(self, probe):
        out = probe("stale_state")["builder_input_failed"]
        assert out == {"input": "none", "error": True}

    def test_a_superseded_start_reply_never_shows_the_input(self, probe):
        out = probe("stale_state")["builder_input_superseded"]
        assert out == {"after_stale": "none", "after_current": "flex"}

    def test_a_finished_build_still_hides_the_input(self, probe):
        assert probe("stale_state")["builder_input_complete"] == "none"

    def test_the_build_status_paints_at_once_and_the_reply_replaces_it(self, probe):
        out = probe("stale_state")["builder_status_first"]
        assert re.search(r"Building artifact\.\.\. 0 s", out["first"])
        assert "Session active" in out["after"]

    def test_an_edit_with_no_states_hides_the_old_result_graph(self, probe):
        out = probe("stale_state")["builder_edit_graph"]
        assert out == {
            "before": "block",
            "with_states": "block",
            "no_states": "none",
            "again": "block",
            "empty_states": "none",
        }
        assert probe("stale_state")["builder_edit_workflow_graph"] == "none"


class TestJsonColouring:
    """Plan 51fde95b step 14 (D-040): `utils/json_highlight.js` and every page that shows JSON."""

    @pytest.mark.parametrize("name", ["json_highlight", "json_surfaces"])
    def test_every_check_of_the_probe_holds(self, probe, name):
        # json_highlight: bytes kept, token types, key vs string, numbers, escapes, invalid text plain,
        # the size cap, deep nesting, HTML stays text. json_surfaces: the event log, the conversation
        # drawer, the Builder result and state, the workflow and agent drawers.
        assert [k for k, v in probe(name).items() if v is not True] == []

    def test_the_colours_have_4_5_to_1_contrast_on_every_surface(self):
        # style.css has one (dark) theme: no prefers-color-scheme and no data-theme rule, so the six
        # variables live in :root. A second theme must restate them, and this test must then cover it.
        css = (_STATIC / "style.css").read_text(encoding="utf-8")
        assert "prefers-color-scheme" not in css and "data-theme" not in css
        root = css.split(":root {")[1].split("\n}")[0]

        def colour(name: str) -> float:
            hex_ = re.search(rf"--{name}:\s*#([0-9a-fA-F]{{6}})", root)[1]
            rgb = [int(hex_[i : i + 2], 16) / 255 for i in (0, 2, 4)]
            lin = [
                c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb
            ]
            return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

        surfaces = ("bg", "bg-secondary", "surface", "surface-alt", "surface-hover")
        for kind in ("key", "string", "number", "bool", "null", "punct"):
            assert f".json-{kind}" in css and f"color: var(--json-{kind})" in css
            lum = colour(f"json-{kind}")
            for surface in surfaces:
                hi, lo = sorted((lum, colour(surface)), reverse=True)
                assert (hi + 0.05) / (lo + 0.05) >= 4.5, (kind, surface)
