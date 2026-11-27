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

    def test_an_attached_api_conversation_opens_read_only_and_binds_to_no_fsm(
        self, probe
    ):
        """Plan 9283c1fd step 6 (O3, D-010): no instance id means no chat input, no End and no
        POST, with a managed FSM listed (the first-FSM fallback bound it to that FSM)."""
        out = probe("chat_switch")["attached"]
        assert out["chat_input"] == "none" and out["end_button"] is False
        assert out["conv_inst"] == "" and out["read_only"] is True
        assert out["posts"] == []
        assert out["detail_id"] is None  # no stale instance stays selected

    def test_the_read_only_flag_is_up_before_the_snapshot_is_back_and_stops_both_senders(
        self, probe
    ):
        out = probe("chat_switch")
        assert out["during_load"] is True
        assert out["read_only_guards"] == []  # no POST of a message, no POST of an end

    def test_back_from_an_attached_conversation_shows_no_stale_panel(self, probe):
        out = probe("chat_switch")["attached"]
        assert out["after_back"] == {
            "read_only": False,
            "detail_id": None,
            "panel": "",
            "events": "",
            "gets": [],
        }
        assert out["after_open_drawer"] is False

    def test_a_managed_conversation_keeps_chat_and_end_on_its_own_instance(self, probe):
        out = probe("chat_switch")["managed"]
        assert out == {
            "chat_input": "block",
            "end_button": True,
            "end_target": "managedFSM",
            "conv_inst": "managedFSM",
            "read_only": False,
            "detail_id": "managedFSM",
            "posts": ["/api/fsm/managedFSM/converse", "/api/fsm/managedFSM/end"],
        }

    def test_without_an_explicit_id_the_owner_is_the_snapshots_instance(self, probe):
        out = probe("chat_switch")["snapshot_owner"]
        assert out["chat_input"] == "block" and out["read_only"] is False
        assert out["end_target"] == "OWN" and out["conv_inst"] == "OWN"
        assert out["posts"] == ["/api/fsm/OWN/converse", "/api/fsm/OWN/end"]
        assert out["detail_id"] is None

    def test_a_failed_send_is_a_toast_and_the_text_returns_only_where_the_user_is(
        self, probe
    ):
        """B4: the toast is unconditional; the error bubble and the restored text need the
        conversation to be the one shown, and typed text is never overwritten."""
        send = probe("chat_switch")["send"]
        for name in ("stay", "stay_typed", "away", "away_typed"):
            assert len(send[name]["toasts"]) == 1, name
            assert "LLM down" in send[name]["toasts"][0], name
            assert send[name]["disabled"] is False, name
        assert (send["stay"]["error_bubbles"], send["stay"]["input"]) == (1, "hello")
        assert (send["stay_typed"]["error_bubbles"], send["stay_typed"]["input"]) == (
            1,
            "draft",
        )
        assert (send["away"]["error_bubbles"], send["away"]["input"]) == (0, "")
        assert (send["away_typed"]["error_bubbles"], send["away_typed"]["input"]) == (
            0,
            "draft B",
        )

    def test_a_successful_send_is_unchanged(self, probe):
        assert probe("chat_switch")["send"]["ok"] == {
            "error_bubbles": 0,
            "reply_bubbles": 1,
            "toasts": [],
            "input": "",
            "disabled": False,
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

    def test_an_accepted_launch_closes_the_form_and_holds_the_button(self, probe):
        # Plan 9283c1fd step 8: the button stays disabled until the redirect (it was freed at once).
        out = probe("builder_launch")
        assert out["retried"] == {
            "form": "none",
            "label": "Launch AGENT",
            "disabled": True,
            "posts": 2,
        }
        assert out["g3a"]["window_over"] == {"disabled": False, "pages": ["control"]}

    def test_four_clicks_inside_the_window_make_one_post(self, probe):
        g3a = probe("builder_launch")["g3a"]
        assert g3a["four_clicks"] == {"posts": 1, "disabled": True, "form": "none"}
        assert g3a["after_window"] == {"disabled": False, "pages": ["control"]}

    def test_a_refused_launch_frees_the_button_and_the_guard_at_once(self, probe):
        g3a = probe("builder_launch")["g3a"]
        assert g3a["refused"]["disabled"] is False
        assert g3a["refused"]["form"] == "block"
        assert g3a["retry_posts"] == 1  # the guard is released, a retry posts

    def test_the_redirect_follows_only_the_session_and_page_it_came_from(self, probe):
        g3a = probe("builder_launch")["g3a"]
        for kind in ("agent", "fsm"):
            assert g3a[f"{kind}_stays"]["pages"] == ["control"]
            for left in ("reset", "navigated"):
                moved = g3a[f"{kind}_{left}"]
                assert (
                    moved["pages"] == []
                )  # no redirect after a Reset or a page change
                assert moved["disabled"] is False  # and the button is free again
        assert (
            g3a["fsm_stays"]["conv"] == "c1"
        )  # the conversation opens only with the redirect
        assert g3a["fsm_reset"]["conv"] is None
        assert g3a["fsm_navigated"]["conv"] is None


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


class TestBuilderRefusalNamesEditJson:
    """Plan 9283c1fd step 8 (G3b): the refusal of a tool-less artifact names the control that adds a tool."""

    def test_the_refusal_names_edit_json_and_posts_nothing(self, probe):
        out = probe("builder_agent_gate")["needsTools"]
        assert out["posts"] == 0
        assert "requires at least one tool" in out["status"]
        assert "Edit JSON" in out["status"]

    def test_the_name_is_the_label_of_the_button_in_the_template(self):
        html = (_STATIC.parent / "templates" / "index.html").read_text(encoding="utf-8")
        label = re.search(r'id="builder-edit-btn"[^>]*>([^<]*)<', html).group(1)
        assert label == "Edit JSON"  # the refusal text quotes it as written


class TestKeyPromptLateClose:
    """Plan 9283c1fd step 8 (G7): a 4401 close after the user dismissed the prompt does not open it again."""

    def test_a_4401_after_cancel_keeps_the_modal_closed(self, probe):
        late = probe("dialog_focus")["late"]
        assert late == {"open": False, "opens": 1, "status": "API key required"}

    def test_a_4401_before_the_boot_prompt_is_one_modal(self, probe):
        early = probe("dialog_focus")["early"]
        assert early == {
            "open_after_4401": True,
            "opens": 1,
            "status": "API key required",
            "open_after_cancel": False,
        }

    def test_saving_a_key_after_the_dismissal_reconnects(self, probe):
        assert probe("dialog_focus")["save"] == {"reconnects": 1, "open": False}

    def test_a_4401_with_no_dismissal_still_prompts(self, probe):
        undismissed = probe("dialog_focus")["undismissed"]
        assert undismissed == {
            "open": True,
            "opens": 1,
            "status": "API key required",
        }

    def test_the_settings_page_keeps_its_own_key_field(self):
        html = (_STATIC.parent / "templates" / "index.html").read_text(encoding="utf-8")
        assert 'id="set-api-key"' in html  # the way back in after a dismissal


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
    """A created instance is never a failed launch; no stale timer."""

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
    """A superseded failure paints nothing, a failure says why, focus follows the view."""

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
    """Load more by `since_seq`, a request token, errors keep the earlier pages."""

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
    """Step data whole or a labelled raw cut, the details state across a repaint."""

    def test_a_small_step_data_is_shown_whole_and_parses(self, probe):
        out = probe("stale_state")["step_data_small"]
        assert out["length"] > 300  # no 300 character cut
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
    """The input waits for the start reply, an edit with no states hides the graph."""

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


class TestSmallUiFixes:
    """Graph rows, empty graph, CRLF markdown, six metric columns."""

    def test_a_row_starts_below_the_previous_rows_own_height(self, probe):
        out = probe("stale_state")["graph_rows"]
        assert out == {"overlaps": [], "row1_minus_row0": 860, "row2_minus_row1": 220}

    def test_a_taller_second_row_does_not_move_the_row_above_it(self, probe):
        out = probe("stale_state")["graph_rows_tall_second"]
        assert out == {"overlaps": [], "row1_minus_row0": 220, "row2_minus_row1": 540}

    def test_graphs_of_one_row_and_of_equal_rows_keep_every_byte(self, probe):
        # The sha1 of the whole svg markup and the viewBox were computed once for the layout of a one-row graph.
        assert probe("stale_state")["graph_unmoved"] == {
            "one": ["f263de7bd06eb1280c2d5f36c1708c720105b48d", "0 0 900 400"],
            "two": ["776132f4d6d901c5be55d72d187e5e1091a2511a", "0 0 1500 460"],
        }

    def test_an_empty_graph_clears_the_svg_and_never_writes_an_infinite_viewbox(
        self, probe
    ):
        out = probe("stale_state")["graph_empty"]
        assert out["html"] == "" and out["viewBox"] is None and out["bad"] is False
        assert out["threw"] is None and out["no_nodes_key"] is None

    def test_crlf_and_cr_text_render_as_lf_text(self, probe):
        out = probe("stale_state")["markdown_crlf"]
        assert out["fence"] and out["same_as_lf"] and out["cr_same_as_lf"]
        assert out["stray_cr"] is False and out["lists"] == [True, True]
        assert out["paragraphs"] is True

    def test_the_escape_still_comes_first_in_a_crlf_fence(self, probe):
        out = probe("stale_state")["markdown_crlf_hostile"]
        assert out == {"raw": False, "fence": True, "img": True}

    def test_the_markdown_normalisation_runs_after_the_escape(self):
        text = (_STATIC / "utils" / "markdown.js").read_text(encoding="utf-8")
        line = next(ln for ln in text.splitlines() if "let s = esc(text)" in ln)
        assert line.index("esc(text)") < line.index(r"/\r\n?/g")

    def test_the_first_metrics_row_has_a_column_for_every_card(self):
        html = (_STATIC.parent / "templates" / "index.html").read_text(encoding="utf-8")
        row = html.split('<div class="metrics-row">')[1].split(
            '<div class="metrics-row"'
        )[0]
        cards = row.count('<div class="metric-card')
        css = (_STATIC / "style.css").read_text(encoding="utf-8")
        rule = re.search(r"\n\.metrics-row \{([^}]*)\}", css)[1]
        template = re.search(r"grid-template-columns:\s*([^;]+);", rule)[1]

        def tracks(text: str) -> int:
            total = 0
            for repeat, _token in re.findall(r"repeat\((\d+),\s*[^)]+\)|(\S+)", text):
                total += int(repeat) if repeat else 1
            return total

        assert cards == 6 and tracks(template) == cards
        # the two breakpoint rules are unchanged: 3 columns hold six cards in two rows, 2 columns the hero apart
        assert (
            "@media (max-width: 1200px) {\n    .metrics-row { grid-template-columns: repeat(3, 1fr); }"
            in css
        )
        assert ".metrics-row { grid-template-columns: repeat(2, 1fr); }" in css


class TestVisualizerEmptyOption:
    """The empty option clears the view and a late reply cannot undo it."""

    def test_the_empty_workflow_option_clears_the_svg_info_and_table(self, probe):
        out = probe("stale_state")["viz_empty_workflow"]
        assert out["drawn"] is True and out["drew"] is True and out["cleared"] is None
        assert out["svg"] == "" and out["rows"] == ""
        assert out["info"] == (
            '<span class="key">Select a workflow to visualize</span><span class="val"></span>'
        )

    def test_the_agent_twin_clears_too(self, probe):
        out = probe("stale_state")["viz_empty_agent"]
        assert out["drew"] is True and out["svg"] == "" and out["rows"] == ""
        assert "Select an agent type to visualize" in out["info"]

    def test_a_reply_that_lands_after_the_clear_paints_nothing(self, probe):
        out = probe("stale_state")["viz_late_after_clear"]
        assert out == {"svg": "", "rows": "", "info": False, "returned": None}

    def test_a_reply_of_an_older_request_never_paints_over_a_newer_one(self, probe):
        out = probe("stale_state")["viz_late_after_newer"]
        assert out == {"info_is_b": True, "info_is_a": False}

    def test_the_empty_info_texts_are_the_words_of_the_template(self):
        html = (_STATIC.parent / "templates" / "index.html").read_text(encoding="utf-8")
        text = (_STATIC / "pages" / "visualizer.js").read_text(encoding="utf-8")
        for words in (
            "Select an agent type to visualize",
            "Select a workflow to visualize",
        ):
            assert words in html and words in text


class TestLogsClearCursor:
    """Clear survives every rebuild of the stream; a later record shows."""

    def test_cleared_records_stay_hidden_after_a_pill_a_search_and_a_page_show(
        self, probe
    ):
        out = probe("stale_state")["logs_clear"]
        assert out["refreshed"] == 3 and out["cleared"] == 0
        for key in (
            "refresh_after_clear",
            "pill_off",
            "pill_on",
            "search",
            "page_show",
            "sync",
        ):
            assert out[key] == 0, key

    def test_a_record_that_arrives_after_the_clear_shows_and_stays(self, probe):
        out = probe("stale_state")["logs_clear"]
        assert out["new_by_refresh"] == 1 and out["new_by_append"] == 2
        assert out["new_after_toggles"] == 2 and out["pill_text"] == "Info (2)"

    def test_a_second_clear_hides_those_too_and_a_later_record_shows(self, probe):
        out = probe("stale_state")["logs_clear"]
        assert out["second_clear"] == 0 and out["after_second_clear"] == 1


class TestLaunchAndVisualizerNotes:
    def test_the_first_start_failure_is_also_a_toast(self, probe):
        out = probe("stale_state")
        assert out["fsm_partial"]["toast"] == 1
        assert out["builder_partial"]["toast"] == 1

    def test_the_window_of_an_earlier_launch_never_frees_the_button_of_a_newer_one(
        self, probe
    ):
        out = probe("stale_state")["launch_newer_owns_button"]
        assert out["posts"] == 2 and out["before"] is True
        assert out["afterA"] == {"disabled": True, "text": "Launching..."}
        assert out["afterB"] == {"disabled": False, "text": "Launch Workflow"}

    def test_a_superseded_visualizer_failure_shows_no_toast(self, probe):
        out = probe("stale_state")["viz_late_failure"]
        assert out["returned"] is None and out["late_toasts"] == 0
        assert out["current_returned"] is False and out["current_toasts"] == 1


class TestLogsClearAcrossLevels:
    """Clear holds across a pill that lowers the minimum level (the server filters by level)."""

    def test_a_debug_record_from_before_the_clear_stays_hidden(self, probe):
        out = probe("stale_state")["logs_clear_levels"]
        assert out["shown_before"] == 2 and out["after_clear"] == 0
        assert (
            out["cursor_request"] == 1
        )  # one unfiltered request for the newest record
        assert out["debug_on"] == 0

    def test_an_info_record_from_before_the_clear_stays_hidden(self, probe):
        out = probe("stale_state")["logs_clear_levels"]
        assert out["info_off_shown"] == 1 and out["info_on"] == 0

    def test_a_record_after_the_clear_shows_at_any_level(self, probe):
        out = probe("stale_state")["logs_clear_levels"]
        assert out["new_debug"] == 1 and out["new_info"] == 2
        assert out["info_new"] == 1

    def test_a_late_reply_of_an_earlier_clear_never_moves_the_cursor_back(self, probe):
        assert probe("stale_state")["logs_clear_levels"]["late_reply"] == 0

    def test_a_failed_cursor_request_leaves_the_keys_as_the_guard(self, probe):
        assert probe("stale_state")["logs_clear_levels"]["cursor_failed"] == 0


class TestLogsClearWaitsAndSearchMatchesWhatIsShown:
    """Plan 9283c1fd step 7 (B5a, B5b): Clear waits for its cursor before a rebuild filters, and the
    search reads the one text the entry shows (module:line, [conversation id], message)."""

    def test_a_pill_toggled_while_the_clear_cursor_is_held_shows_nothing_old(
        self, probe
    ):
        out = probe("stale_state")["logs_clear_pending"]
        assert out["shown_before"] == 1
        assert out["during"] == 0  # the rebuild waits for the cursor
        assert out["after_landing"] == 0  # and drops the record from before the Clear

    def test_the_search_matches_the_module_and_the_conversation_id_and_the_message(
        self, probe
    ):
        out = probe("stale_state")["logs_search"]
        assert out["module"] == 2 and out["module_highlight"] is True
        assert out["conversation"] == 1 and out["conversation_upper"] == 1
        assert out["conversation_highlight"] is True
        assert out["message"] == 1 and out["line_and_message"] == 1
        assert out["no_match"] == 0

    def test_a_search_with_regex_characters_matches_and_highlights_and_never_fails(
        self, probe
    ):
        out = probe("stale_state")["logs_search"]
        assert out["regex_chars"] == 1 and out["regex_highlight"] is True
        assert out["regex_toasts"] == 0

    def test_the_websocket_append_path_filters_with_the_same_text(self, probe):
        out = probe("stale_state")["logs_search"]
        assert out["append_before"] == 0
        assert out["append_by_conversation"] == 1 and out["append_by_module"] == 2

    def test_the_highlight_and_both_filters_read_one_function(self):
        text = (_STATIC / "pages" / "logs.js").read_text(encoding="utf-8")
        assert (
            text.count("_logText(") >= 4
        )  # the definition, the highlight, two filters
        assert "r.message ?? ''" not in text  # no filter reads the message alone


class TestMarkdownFencesAndItalicRuns:
    """Plan 9283c1fd step 7 (B6a, B6b); `markdown.mjs` and `markdown_blocks.mjs` stay as they were."""

    CODE = '<pre class="md-code-block"><code>*a* **b** _c_ __d__</code></pre>'

    @pytest.mark.parametrize(
        "info",
        ["c++", "c#", "ts-node", "objective-c", "js  ", "js", ""],
    )
    def test_a_fence_with_any_info_string_is_a_code_block_and_the_body_stays_raw(
        self, probe, info
    ):
        assert probe("markdown_fences")["fence"][info] == self.CODE

    @pytest.mark.parametrize(
        "info",
        ["<script>alert(1)</script>", "<img src=x onerror=alert(1)>"],
    )
    def test_a_hostile_info_string_is_dropped_never_emitted(self, probe, info):
        html = probe("markdown_fences")["fence"][info]
        assert html == self.CODE
        assert "<script" not in html and "<img" not in html and "alert" not in html

    def test_a_fence_info_string_with_a_trailing_space_before_crlf_is_a_block(
        self, probe
    ):
        out = probe("markdown_fences")
        assert out["fence_crlf_info"] == (
            '<pre class="md-code-block"><code>int x;</code></pre>'
        )
        assert "<code>int *p;</code>" in out["fence_after_text"]
        assert "<em>" not in out["fence_after_text"]

    def test_a_backtick_in_the_info_string_is_not_a_fence(self, probe):
        html = probe("markdown_fences")["backtick_in_info"]
        assert "md-code-block" not in html and html.endswith("```</p>")

    def test_the_nested_fence_inside_backticks_is_unchanged(self, probe):
        assert probe("markdown_fences")["nested_unchanged"] == (
            '<p>`a <pre class="md-code-block"><code>x&lt;b&gt;</code></pre><p> b`</p>'
        )

    def test_runs_of_italic_all_render(self, probe):
        out = probe("markdown_fences")
        assert out["italic_pair"] == "<p><em>a</em> <em>b</em></p>"
        assert out["italic_three"] == "<p><em>a</em> <em>b</em> <em>c</em></p>"
        assert out["italic_lines"] == "<p><em>a</em><br><em>b</em></p>"
        assert out["italic_punct"] == "<p>(<em>a</em>), <em>b</em>.</p>"

    def test_identifiers_with_underscores_stay_text(self, probe):
        assert probe("markdown_fences")["identifiers"] == [
            "<p>snake_case_here</p>",
            "<p>foo_bar_baz</p>",
            "<p>http://x/a_b_c</p>",
            "<p>a_b_ c</p>",
        ]


class TestDrawerDirectOpen:
    """A conversation opened straight in the drawer fetches no hidden detail."""

    def test_the_request_census_is_the_conversation_alone(self, probe):
        out = probe("stale_state")["drawer_direct"]
        assert out["census"] == ["/api/conversations/c"]
        # the owning FSM is still the drawer selection and the 2 s poll still runs
        assert out["chosen"] == {"id": "I", "type": "fsm", "conv": "c", "polling": True}

    def test_back_asks_for_the_instance_detail(self, probe):
        out = probe("stale_state")["drawer_back"]
        assert out["census"] == [
            "/api/fsm/I/conversations",
            "/api/instances/I/events?limit=100",
        ]
        assert out["chosen"]["id"] == "I" and out["chosen"]["conv"] is None

    def test_a_plain_open_and_an_already_selected_instance_are_as_before(self, probe):
        out = probe("stale_state")
        assert out["drawer_plain_open"] == [
            "/api/fsm/I/conversations",
            "/api/instances/I/events?limit=100",
        ]
        assert out["drawer_already_selected"] == ["/api/conversations/c"]


class TestInstanceList:
    """Plan 9283c1fd step 5 (B3, O1, the dead trace path): `refreshInstances`, `hashInstances`
    and the drawer of a finished agent, over the stub DOM (no browser)."""

    def test_a_failed_first_fetch_leaves_no_spinner_and_one_toast(self, probe):
        out = probe("instance_list")["failed_first"]
        assert out == {
            "grid": "",
            "spinner": False,
            "toasts": ["Failed to refresh instances"],
        }

    def test_an_empty_monitor_polled_twice_never_keeps_the_spinner(self, probe):
        out = probe("instance_list")
        assert out["empty_first"] == "" and out["empty_second"] == ""
        assert out["empty_hint"] == "block"

    def test_an_older_slow_reply_never_overwrites_a_newer_one(self, probe):
        out = probe("instance_list")
        assert out["late_ok"] == {
            "ids": ["new"],
            "grid_old": False,
            "grid_new": True,
            "spinner": False,
        }

    def test_an_older_failure_after_a_newer_success_toasts_nothing(self, probe):
        out = probe("instance_list")["late_fail"]
        assert out == {"ids": ["new2"], "toasts": [], "spinner": False}

    def test_only_the_latest_request_clears_the_spinner(self, probe):
        out = probe("instance_list")
        assert out["latest_clears"] == {"grid": "", "toasts": []}
        assert out["superseded_keeps_spinner"] == {
            "whileLatestOut": True,
            "afterLatest": "",
        }

    def test_a_count_change_alone_refreshes_the_grid_and_the_table(self, probe):
        out = probe("instance_list")
        assert out["counts_before"] == {
            "grid": ["0 conversations", "0 active"],
            "table": ["0 convs", "0 active"],
        }
        assert out["conversation_count_change"] == {
            "grid": ["3 conversations", "0 active"],
            "table": ["3 convs", "0 active"],
        }
        assert out["active_workflows_change"] == {
            "grid": ["3 conversations", "2 active"],
            "table": ["3 convs", "2 active"],
        }

    def test_an_attached_api_conversation_row_has_an_action_and_others_keep_theirs(
        self, probe
    ):
        """O3: a row of `fsm_conversation` with no instance id is clickable (read-only view); an
        agent or workflow row with none stays inert; the item id is escaped."""
        rows = probe("instance_list")["activity_rows"]
        assert [(r["action"], r["instance"], r["item"]) for r in rows] == [
            (True, "", "att1"),
            (True, "F1", "c1"),
            (True, "", "&lt;b&gt;x&lt;/b&gt;"),
            (False, None, None),
        ]
        assert not any(r["raw_tag"] for r in rows)

    def test_the_websocket_gate_passes_a_count_change(self, probe):
        out = probe("instance_list")["ws_gate"]
        # first list, the same list again (gated), a count change, a new instance, its count change
        assert out == {"paints": [1, 1, 2, 4], "count": 2}

    def test_a_finished_agent_drawer_has_the_tool_call_fallback_and_no_trace_block(
        self, probe
    ):
        out = probe("instance_list")["agent_drawer"]
        assert out["tool_calls"] and out["escaped"]
        assert out["same_with_steps_in_result"] is True
        assert out["trace_block"] is False
        assert out["result_fetches"] == 0  # the result route fed only the trace block

    def test_no_page_or_action_table_names_the_deleted_trace_handlers(self):
        names = ("toggle-trace-step", "expand-all-trace", "collapse-all-trace")
        names += ("toggleTraceStep", "toggleAllTraceSteps", "_TRACE_STATES")
        for path in _STATIC.rglob("*.js"):
            text = path.read_text(encoding="utf-8")
            assert [n for n in names if n in text] == [], path.name


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
