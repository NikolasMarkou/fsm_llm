from __future__ import annotations

"""
Main Textual application for FSM-LLM Monitor.

Green-on-black retro 90s terminal dashboard.
"""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)

from .bridge import MonitorBridge
from .constants import (
    KEY_AGENTS,
    KEY_CONVERSATION,
    KEY_DASHBOARD,
    KEY_FSM_VIEWER,
    KEY_LOGS,
    KEY_QUIT,
    KEY_SETTINGS,
    KEY_WORKFLOWS,
)
from .definitions import MonitorConfig
from .theme import RETRO_THEME_CSS


class MonitorApp(App):
    """FSM-LLM Monitor — Retro Terminal Dashboard.

    A 90s-style green-on-black terminal UI for monitoring
    FSM conversations, agents, workflows, and system logs.
    """

    TITLE = "FSM-LLM MONITOR"
    SUB_TITLE = "v0.3.0"
    CSS = RETRO_THEME_CSS

    BINDINGS = [
        Binding(KEY_DASHBOARD, "switch_tab('dashboard')", "Dashboard", priority=True),
        Binding(KEY_FSM_VIEWER, "switch_tab('fsm')", "FSM Viewer", priority=True),
        Binding(KEY_CONVERSATION, "switch_tab('conversations')", "Conversations", priority=True),
        Binding(KEY_AGENTS, "switch_tab('agents')", "Agents", priority=True),
        Binding(KEY_WORKFLOWS, "switch_tab('workflows')", "Workflows", priority=True),
        Binding(KEY_LOGS, "switch_tab('logs')", "Logs", priority=True),
        Binding(KEY_SETTINGS, "switch_tab('settings')", "Settings", priority=True),
        Binding(KEY_QUIT, "quit", "Quit", priority=True),
    ]

    def __init__(
        self,
        bridge: MonitorBridge | None = None,
        config: MonitorConfig | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bridge = bridge or MonitorBridge(config=config)

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(
            initial="dashboard",
        ):
            with TabPane("Dashboard", id="dashboard"):
                yield _DashboardPane()
            with TabPane("FSM Viewer", id="fsm"):
                yield _FSMViewerPane()
            with TabPane("Conversations", id="conversations"):
                yield _ConversationPane()
            with TabPane("Agents", id="agents"):
                yield _AgentPane()
            with TabPane("Workflows", id="workflows"):
                yield _WorkflowPane()
            with TabPane("Logs", id="logs"):
                yield _LogPane()
            with TabPane("Settings", id="settings"):
                yield _SettingsPane()
        yield Footer()

    def action_switch_tab(self, tab_id: str) -> None:
        """Switch to the specified tab."""
        tabbed = self.query_one(TabbedContent)
        tabbed.active = tab_id


# --- Inline pane widgets (compose directly, no separate screens) ---


class _DashboardPane(Static):
    """Dashboard content rendered inline."""

    def compose(self) -> ComposeResult:
        from textual.containers import Horizontal, Vertical
        from textual.widgets import DataTable, Label, RichLog
        from textual.widgets import Static as St

        with Vertical():
            with Horizontal(classes="full-width"):
                with Vertical(classes="panel"):
                    yield Label("[b]SYSTEM METRICS[/b]", classes="panel-title")
                    yield St(
                        "Conversations: 0\n"
                        "Total Events:  0\n"
                        "Transitions:   0\n"
                        "Errors:        0",
                        id="dash-metrics",
                    )
                with Vertical(classes="panel"):
                    yield Label("[b]STATE DISTRIBUTION[/b]", classes="panel-title")
                    yield St("No data yet", id="dash-states")

            with Vertical(classes="panel"):
                yield Label("[b]ACTIVE CONVERSATIONS[/b]", classes="panel-title")
                yield DataTable(id="dash-conv-table")

            with Vertical(classes="panel"):
                yield Label("[b]RECENT EVENTS[/b]", classes="panel-title")
                yield RichLog(id="dash-events", max_lines=100)

    def on_mount(self) -> None:
        table = self.query_one("#dash-conv-table", DataTable)
        table.add_columns("ID", "State", "Messages", "Stack", "Status")
        self._refresh()
        self.set_interval(2.0, self._refresh)

    def _refresh(self) -> None:
        app = self.app
        if not hasattr(app, "bridge"):
            return
        bridge = app.bridge
        metrics = bridge.get_metrics()

        self.query_one("#dash-metrics", Static).update(
            f"Conversations: [b]{metrics.active_conversations}[/b]\n"
            f"Total Events:  [b]{metrics.total_events}[/b]\n"
            f"Transitions:   [b]{metrics.total_transitions}[/b]\n"
            f"Errors:        [b]{metrics.total_errors}[/b]"
        )

        states_disp = self.query_one("#dash-states", Static)
        if metrics.states_visited:
            lines = [f"  {s}: {c}" for s, c in sorted(metrics.states_visited.items(), key=lambda x: -x[1])]
            states_disp.update("\n".join(lines))
        else:
            states_disp.update("No transitions recorded")

        table = self.query_one("#dash-conv-table", DataTable)
        table.clear()
        for snap in bridge.get_all_conversation_snapshots():
            status = "[red]ENDED[/red]" if snap.is_terminal else "[green]ACTIVE[/green]"
            table.add_row(
                snap.conversation_id[:16],
                snap.current_state,
                str(len(snap.message_history)),
                str(snap.stack_depth),
                status,
            )

        events_log = self.query_one("#dash-events", RichLog)
        events_log.clear()
        for event in bridge.get_recent_events(limit=30):
            ts = event.timestamp.strftime("%H:%M:%S")
            color = {"ERROR": "red", "WARNING": "yellow", "DEBUG": "dim green"}.get(event.level, "green")
            events_log.write(f"[{color}]{ts}[/] [{color}]{event.event_type:<20}[/] {event.message}")


class _FSMViewerPane(Static):
    """FSM viewer content."""

    def compose(self) -> ComposeResult:
        from textual.containers import Horizontal, Vertical
        from textual.widgets import Button, DataTable, Input, Label, Tree
        from textual.widgets import Static as St

        with Vertical():
            with Horizontal(classes="panel"):
                yield Input(placeholder="Path to FSM JSON file...", id="fsm-path")
                yield Button("Load", id="fsm-load-btn", variant="primary")
            with Horizontal(classes="full-width"):
                with Vertical(classes="panel"):
                    yield Label("[b]STATE GRAPH[/b]", classes="panel-title")
                    yield Tree("FSM States", id="fsm-tree")
                with Vertical(classes="panel"):
                    yield Label("[b]STATE DETAILS[/b]", classes="panel-title")
                    yield St("Load an FSM to view details", id="fsm-details")
            with Vertical(classes="panel"):
                yield Label("[b]TRANSITIONS[/b]", classes="panel-title")
                yield DataTable(id="fsm-trans-table")

    def on_mount(self) -> None:
        table = self.query_one("#fsm-trans-table", DataTable)
        table.add_columns("From", "To", "Priority", "Description", "Conditions", "Logic")

    def on_button_pressed(self, event) -> None:
        if event.button.id == "fsm-load-btn":
            from textual.widgets import Input
            from textual.widgets import Static as St
            path = self.query_one("#fsm-path", Input).value.strip()
            if not path or not hasattr(self.app, "bridge"):
                return
            fsm = self.app.bridge.load_fsm_from_file(path)
            if fsm is None:
                self.query_one("#fsm-details", St).update("[red]Failed to load FSM[/red]")
                return
            self._render_fsm(fsm)

    def _render_fsm(self, fsm) -> None:
        from textual.widgets import DataTable, Tree
        from textual.widgets import Static as St
        tree = self.query_one("#fsm-tree", Tree)
        tree.clear()
        tree.root.label = f"[b]{fsm.name}[/b] (v{fsm.version})"
        for state in fsm.states:
            marker = " [green][>>][/green]" if state.is_initial else (" [red][XX][/red]" if state.is_terminal else "")
            node = tree.root.add(f"{state.state_id}{marker}", data=state)
            for trans in state.transitions:
                node.add_leaf(f"-> {trans.target_state} (p={trans.priority})", data=trans)
        tree.root.expand_all()

        table = self.query_one("#fsm-trans-table", DataTable)
        table.clear()
        for state in fsm.states:
            for trans in state.transitions:
                table.add_row(state.state_id, trans.target_state, str(trans.priority), trans.description[:40], str(trans.condition_count), "Yes" if trans.has_logic else "No")

        self.query_one("#fsm-details", St).update(
            f"[b]Name:[/b] {fsm.name}\n[b]Description:[/b] {fsm.description}\n[b]Version:[/b] {fsm.version}\n"
            f"[b]Initial:[/b] {fsm.initial_state}\n[b]Persona:[/b] {fsm.persona or 'None'}\n[b]States:[/b] {fsm.state_count}"
        )

    def on_tree_node_selected(self, event) -> None:
        from textual.widgets import Static as St
        data = event.node.data
        if data is None:
            return
        details = self.query_one("#fsm-details", St)
        if hasattr(data, "state_id"):
            details.update(
                f"[b]State:[/b] {data.state_id}\n[b]Description:[/b] {data.description}\n"
                f"[b]Purpose:[/b] {data.purpose}\n[b]Initial:[/b] {'Yes' if data.is_initial else 'No'}\n"
                f"[b]Terminal:[/b] {'Yes' if data.is_terminal else 'No'}\n[b]Transitions:[/b] {data.transition_count}"
            )
        elif hasattr(data, "target_state"):
            details.update(
                f"[b]Target:[/b] {data.target_state}\n[b]Description:[/b] {data.description}\n"
                f"[b]Priority:[/b] {data.priority}\n[b]Conditions:[/b] {data.condition_count}\n"
                f"[b]Has Logic:[/b] {'Yes' if data.has_logic else 'No'}"
            )


class _ConversationPane(Static):
    """Conversation monitor content."""

    def compose(self) -> ComposeResult:
        from textual.containers import Horizontal, Vertical
        from textual.widgets import DataTable, Label, RichLog, Select
        from textual.widgets import Static as St

        with Vertical():
            with Horizontal(classes="panel"):
                yield Label("Conversation: ", classes="panel-title")
                yield Select([], prompt="Select conversation...", id="conv-select")
            with Horizontal(classes="full-width"):
                with Vertical(classes="panel"):
                    yield Label("[b]CURRENT STATE[/b]", classes="panel-title")
                    yield St("No conversation selected", id="conv-state")
                    yield Label("[b]CONTEXT DATA[/b]", classes="panel-title")
                    yield DataTable(id="conv-ctx-table")
                with Vertical(classes="panel"):
                    yield Label("[b]MESSAGE HISTORY[/b]", classes="panel-title")
                    yield RichLog(id="conv-messages", max_lines=200)
            with Vertical(classes="panel"):
                yield Label("[b]LLM DETAILS[/b]", classes="panel-title")
                yield St("No data", id="conv-llm")

    def on_mount(self) -> None:
        table = self.query_one("#conv-ctx-table", DataTable)
        table.add_columns("Key", "Value")
        self.set_interval(2.0, self._refresh_list)

    def _refresh_list(self) -> None:
        if not hasattr(self.app, "bridge"):
            return
        from textual.widgets import Select
        convs = self.app.bridge.get_active_conversations()
        select = self.query_one("#conv-select", Select)
        select.set_options([(c, c) for c in convs])

    def on_select_changed(self, event) -> None:
        from textual.widgets import Select
        if event.value and event.value != Select.BLANK:
            self._load(str(event.value))

    def _load(self, conv_id: str) -> None:
        if not hasattr(self.app, "bridge"):
            return
        from textual.widgets import DataTable, RichLog
        from textual.widgets import Static as St
        snap = self.app.bridge.get_conversation_snapshot(conv_id)
        if not snap:
            return

        status = "[red]TERMINAL[/red]" if snap.is_terminal else "[green]ACTIVE[/green]"
        self.query_one("#conv-state", St).update(
            f"[b]State:[/b] {snap.current_state} {status}\n[b]Description:[/b] {snap.state_description}\n"
            f"[b]Stack:[/b] {snap.stack_depth}\n[b]ID:[/b] {conv_id}"
        )

        table = self.query_one("#conv-ctx-table", DataTable)
        table.clear()
        for k, v in sorted(snap.context_data.items()):
            vs = str(v)[:60]
            table.add_row(k, vs)

        log = self.query_one("#conv-messages", RichLog)
        log.clear()
        for ex in snap.message_history:
            for role, msg in ex.items():
                color = "cyan" if role == "user" else "green"
                log.write(f"[bold {color}]{role.upper()}:[/bold {color}] {msg}")
            log.write("---")

        parts = []
        if snap.last_extraction:
            parts.append(f"[b]Extraction Confidence:[/b] {snap.last_extraction.get('confidence', 'N/A')}")
        if snap.last_transition:
            parts.append(f"[b]Last Transition:[/b] {snap.last_transition.get('selected_transition', 'N/A')}")
        if snap.last_response:
            parts.append(f"[b]Response Type:[/b] {snap.last_response.get('message_type', 'N/A')}")
        self.query_one("#conv-llm", St).update("\n".join(parts) if parts else "No LLM data")


class _AgentPane(Static):
    """Agent monitor content."""

    def compose(self) -> ComposeResult:
        from textual.containers import Horizontal, Vertical
        from textual.widgets import DataTable, Label, RichLog
        from textual.widgets import Static as St

        with Vertical():
            with Horizontal(classes="full-width"):
                with Vertical(classes="panel"):
                    yield Label("[b]AGENT STATUS[/b]", classes="panel-title")
                    yield St("No agent data available.\nConnect to see agent executions.", id="agent-info")
                with Vertical(classes="panel"):
                    yield Label("[b]TOOL CALLS[/b]", classes="panel-title")
                    yield DataTable(id="agent-tools-table")
            with Vertical(classes="panel"):
                yield Label("[b]EXECUTION TRACE[/b]", classes="panel-title")
                yield RichLog(id="agent-trace", max_lines=200)

    def on_mount(self) -> None:
        table = self.query_one("#agent-tools-table", DataTable)
        table.add_columns("Tool", "Status", "Time (ms)", "Summary")


class _WorkflowPane(Static):
    """Workflow monitor content."""

    def compose(self) -> ComposeResult:
        from textual.containers import Horizontal, Vertical
        from textual.widgets import DataTable, Label, RichLog
        from textual.widgets import Static as St

        with Vertical():
            with Horizontal(classes="full-width"):
                with Vertical(classes="panel"):
                    yield Label("[b]WORKFLOW INSTANCES[/b]", classes="panel-title")
                    yield DataTable(id="wf-table")
                with Vertical(classes="panel"):
                    yield Label("[b]INSTANCE DETAILS[/b]", classes="panel-title")
                    yield St("No workflow data available.\nConnect to see workflows.", id="wf-details")
            with Vertical(classes="panel"):
                yield Label("[b]EXECUTION HISTORY[/b]", classes="panel-title")
                yield RichLog(id="wf-history", max_lines=200)

    def on_mount(self) -> None:
        table = self.query_one("#wf-table", DataTable)
        table.add_columns("Instance", "Status", "Step", "Updated")


class _LogPane(Static):
    """Log viewer content."""

    def compose(self) -> ComposeResult:
        from textual.containers import Horizontal, Vertical
        from textual.widgets import Button, Checkbox, Input, Label, RichLog, Select
        from textual.widgets import Static as St

        with Vertical():
            with Horizontal(classes="panel"):
                yield Label("Level: ")
                yield Select(
                    [("DEBUG", "DEBUG"), ("INFO", "INFO"), ("WARNING", "WARNING"), ("ERROR", "ERROR"), ("CRITICAL", "CRITICAL")],
                    value="INFO", id="log-level",
                )
                yield Label(" Filter: ")
                yield Input(placeholder="Search...", id="log-filter")
                yield Button("Apply", id="log-apply-btn")
                yield Checkbox("Auto-scroll", value=True, id="log-autoscroll")
            with Vertical(classes="panel"):
                yield Label("[b]LOG STREAM[/b]", classes="panel-title")
                yield RichLog(id="log-stream", max_lines=5000)
            with Horizontal(classes="panel"):
                yield St("Total: 0 | Shown: 0", id="log-stat")

    def on_button_pressed(self, event) -> None:
        if event.button.id == "log-apply-btn":
            self._refresh()

    def on_mount(self) -> None:
        self.set_interval(2.0, self._refresh)

    def _refresh(self) -> None:
        if not hasattr(self.app, "bridge"):
            return
        from textual.widgets import Input, RichLog, Select
        from textual.widgets import Static as St

        level = str(self.query_one("#log-level", Select).value or "INFO")
        search = self.query_one("#log-filter", Input).value.strip().lower()

        logs = self.app.bridge.collector.get_logs(level=level)
        if search:
            logs = [r for r in logs if search in r.message.lower()]

        stream = self.query_one("#log-stream", RichLog)
        stream.clear()
        colors = {"DEBUG": "dim green", "INFO": "green", "WARNING": "yellow", "ERROR": "red", "CRITICAL": "bold red"}
        for r in reversed(logs[:500]):
            ts = r.timestamp.strftime("%H:%M:%S.%f")[:12]
            c = colors.get(r.level, "green")
            conv = f" [{r.conversation_id}]" if r.conversation_id else ""
            stream.write(f"[{c}]{ts} {r.level:<8}[/] [dim]{r.module}:{r.line}[/dim]{conv} {r.message}")

        total = len(self.app.bridge.collector.get_logs())
        self.query_one("#log-stat", St).update(f"Total: {total} | Shown: {min(len(logs), 500)} | Level: >= {level}")


class _SettingsPane(Static):
    """Settings content."""

    def compose(self) -> ComposeResult:
        from textual.containers import Horizontal, Vertical
        from textual.widgets import Button, Checkbox, Input, Label, Select
        from textual.widgets import Static as St

        with Vertical():
            with Vertical(classes="panel"):
                yield Label("[b]MONITOR SETTINGS[/b]", classes="panel-title")
                with Horizontal():
                    yield Label("Refresh Interval (s): ")
                    yield Input(value="1.0", id="set-refresh", type="number")
                with Horizontal():
                    yield Label("Max Events:           ")
                    yield Input(value="1000", id="set-max-events", type="integer")
                with Horizontal():
                    yield Label("Max Log Lines:        ")
                    yield Input(value="5000", id="set-max-logs", type="integer")
                with Horizontal():
                    yield Label("Log Level:            ")
                    yield Select([("DEBUG", "DEBUG"), ("INFO", "INFO"), ("WARNING", "WARNING"), ("ERROR", "ERROR")], value="INFO", id="set-level")
                with Horizontal():
                    yield Checkbox("Show Internal Keys", value=False, id="set-internal")
                with Horizontal():
                    yield Checkbox("Auto-scroll Logs", value=True, id="set-autoscroll")
                with Horizontal():
                    yield Button("Save", id="set-save-btn", variant="primary")
                    yield Button("Reset", id="set-reset-btn")

            with Vertical(classes="panel"):
                yield Label("[b]SYSTEM INFO[/b]", classes="panel-title")
                yield St("Loading...", id="set-sysinfo")
            with Vertical(classes="panel"):
                yield Label("[b]CONNECTION STATUS[/b]", classes="panel-title")
                yield St("Not connected", id="set-conn")

    def on_mount(self) -> None:
        self._load()
        self._update_info()

    def on_button_pressed(self, event) -> None:
        if event.button.id == "set-save-btn":
            self._save()
        elif event.button.id == "set-reset-btn":
            self._reset()

    def _load(self) -> None:
        if not hasattr(self.app, "bridge"):
            return
        from textual.widgets import Checkbox, Input, Select
        cfg = self.app.bridge.config
        self.query_one("#set-refresh", Input).value = str(cfg.refresh_interval)
        self.query_one("#set-max-events", Input).value = str(cfg.max_events)
        self.query_one("#set-max-logs", Input).value = str(cfg.max_log_lines)
        self.query_one("#set-level", Select).value = cfg.log_level
        self.query_one("#set-internal", Checkbox).value = cfg.show_internal_keys
        self.query_one("#set-autoscroll", Checkbox).value = cfg.auto_scroll_logs

    def _save(self) -> None:
        if not hasattr(self.app, "bridge"):
            return
        from textual.widgets import Checkbox, Input, Select
        try:
            self.app.bridge.config = MonitorConfig(
                refresh_interval=float(self.query_one("#set-refresh", Input).value or "1.0"),
                max_events=int(self.query_one("#set-max-events", Input).value or "1000"),
                max_log_lines=int(self.query_one("#set-max-logs", Input).value or "5000"),
                log_level=str(self.query_one("#set-level", Select).value or "INFO"),
                show_internal_keys=self.query_one("#set-internal", Checkbox).value,
                auto_scroll_logs=self.query_one("#set-autoscroll", Checkbox).value,
            )
            self.notify("Settings saved")
        except (ValueError, TypeError) as e:
            self.notify(f"Invalid: {e}", severity="error")

    def _reset(self) -> None:
        if not hasattr(self.app, "bridge"):
            return
        self.app.bridge.config = MonitorConfig()
        self._load()
        self.notify("Settings reset")

    def _update_info(self) -> None:
        from textual.widgets import Static as St
        parts = []
        try:
            from fsm_llm_monitor.__version__ import __version__
            parts.append(f"[b]Monitor:[/b] {__version__}")
        except ImportError:
            pass
        try:
            from fsm_llm import __version__ as cv
            parts.append(f"[b]FSM-LLM:[/b] {cv}")
        except ImportError:
            pass
        try:
            import textual
            parts.append(f"[b]Textual:[/b] {textual.__version__}")
        except ImportError:
            pass
        if parts:
            self.query_one("#set-sysinfo", St).update("\n".join(parts))

        if hasattr(self.app, "bridge"):
            conn = self.query_one("#set-conn", St)
            if self.app.bridge.connected:
                m = self.app.bridge.get_metrics()
                conn.update(f"[green]Connected[/green]\nConversations: {m.active_conversations}\nEvents: {m.total_events}")
            else:
                conn.update("[yellow]Disconnected[/yellow]")
