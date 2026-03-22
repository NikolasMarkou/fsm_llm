from __future__ import annotations

"""Agent monitor screen — agent execution tracking."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Label, RichLog, Static


class AgentScreen(Screen):
    """Agent execution monitor."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            with Horizontal(classes="full-width"):
                # Left: agent status
                with Vertical(classes="panel"):
                    yield Label("[b]AGENT STATUS[/b]", classes="panel-title")
                    yield Static(
                        "No agent data available.\n\n"
                        "Connect to an API instance with active\n"
                        "agent executions to see data here.",
                        id="agent-status",
                    )

                # Right: tool calls
                with Vertical(classes="panel"):
                    yield Label("[b]TOOL CALLS[/b]", classes="panel-title")
                    yield DataTable(id="tool-calls-table")

            # Bottom: execution trace
            with Vertical(classes="panel"):
                yield Label("[b]EXECUTION TRACE[/b]", classes="panel-title")
                yield RichLog(id="agent-trace-log", max_lines=200)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#tool-calls-table", DataTable)
        table.add_columns("Tool", "Status", "Time (ms)", "Summary")

    def action_refresh(self) -> None:
        self._refresh_data()

    def _refresh_data(self) -> None:
        app = self.app
        if not hasattr(app, "bridge"):
            return

        # Display agent-related events from the collector
        events = app.bridge.collector.get_events(limit=100)
        agent_events = [
            e for e in events if "agent" in e.event_type or "tool" in e.event_type
        ]

        trace_log = self.query_one("#agent-trace-log", RichLog)
        trace_log.clear()

        if agent_events:
            for event in agent_events:
                ts = event.timestamp.strftime("%H:%M:%S")
                trace_log.write(
                    f"[green]{ts}[/green] [{event.level.lower()}]{event.event_type}[/] {event.message}"
                )
        else:
            trace_log.write("[dim]No agent events captured yet[/dim]")
