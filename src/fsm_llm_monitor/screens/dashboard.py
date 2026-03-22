from __future__ import annotations

"""Dashboard screen — system overview with metrics, conversations, and events."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Label, RichLog, Static


class DashboardScreen(Screen):
    """Main dashboard showing system overview."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            # Top row: metrics
            with Horizontal(classes="full-width"):
                with Vertical(classes="panel"):
                    yield Label("[b]SYSTEM METRICS[/b]", classes="panel-title")
                    yield Static(
                        "Conversations: 0\n"
                        "Total Events:  0\n"
                        "Transitions:   0\n"
                        "Errors:        0",
                        id="metrics-display",
                    )
                with Vertical(classes="panel"):
                    yield Label("[b]STATE DISTRIBUTION[/b]", classes="panel-title")
                    yield Static("No data yet", id="states-display")

            # Middle: active conversations table
            with Vertical(classes="panel"):
                yield Label("[b]ACTIVE CONVERSATIONS[/b]", classes="panel-title")
                yield DataTable(id="conversations-table")

            # Bottom: recent events
            with Vertical(classes="panel"):
                yield Label("[b]RECENT EVENTS[/b]", classes="panel-title")
                yield RichLog(id="events-log", max_lines=100)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#conversations-table", DataTable)
        table.add_columns("ID", "State", "Messages", "Stack", "Status")
        self._refresh_data()

    def action_refresh(self) -> None:
        self._refresh_data()

    def _refresh_data(self) -> None:
        """Refresh all dashboard data from the bridge."""
        app = self.app
        if not hasattr(app, "bridge"):
            return

        bridge = app.bridge

        # Update metrics
        metrics = bridge.get_metrics()
        metrics_display = self.query_one("#metrics-display", Static)
        metrics_display.update(
            f"Conversations: [b]{metrics.active_conversations}[/b]\n"
            f"Total Events:  [b]{metrics.total_events}[/b]\n"
            f"Transitions:   [b]{metrics.total_transitions}[/b]\n"
            f"Errors:        [b]{metrics.total_errors}[/b]"
        )

        # Update state distribution
        states_display = self.query_one("#states-display", Static)
        if metrics.states_visited:
            lines = []
            for state, count in sorted(
                metrics.states_visited.items(), key=lambda x: -x[1]
            ):
                lines.append(f"  {state}: {count}")
            states_display.update("\n".join(lines))
        else:
            states_display.update("No transitions recorded")

        # Update conversations table
        table = self.query_one("#conversations-table", DataTable)
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

        # Update events log
        events_log = self.query_one("#events-log", RichLog)
        events_log.clear()
        for event in bridge.get_recent_events(limit=50):
            ts = event.timestamp.strftime("%H:%M:%S")
            level_color = {
                "ERROR": "red",
                "WARNING": "yellow",
                "DEBUG": "dim green",
            }.get(event.level, "green")
            events_log.write(
                f"[{level_color}]{ts}[/] [{level_color}]{event.event_type:<20}[/] {event.message}"
            )
