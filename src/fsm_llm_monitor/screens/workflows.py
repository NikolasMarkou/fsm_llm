from __future__ import annotations

"""Workflow monitor screen — workflow instance tracking."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Label, RichLog, Static


class WorkflowScreen(Screen):
    """Workflow instance monitor."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            with Horizontal(classes="full-width"):
                # Left: workflow instances
                with Vertical(classes="panel"):
                    yield Label("[b]WORKFLOW INSTANCES[/b]", classes="panel-title")
                    yield DataTable(id="workflow-table")

                # Right: instance details
                with Vertical(classes="panel"):
                    yield Label("[b]INSTANCE DETAILS[/b]", classes="panel-title")
                    yield Static(
                        "No workflow data available.\n\n"
                        "Connect to an API instance with active\n"
                        "workflow executions to see data here.",
                        id="workflow-details",
                    )

            # Bottom: history timeline
            with Vertical(classes="panel"):
                yield Label("[b]EXECUTION HISTORY[/b]", classes="panel-title")
                yield RichLog(id="workflow-history-log", max_lines=200)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#workflow-table", DataTable)
        table.add_columns("Instance", "Status", "Step", "Updated")

    def action_refresh(self) -> None:
        self._refresh_data()

    def _refresh_data(self) -> None:
        app = self.app
        if not hasattr(app, "bridge"):
            return

        history_log = self.query_one("#workflow-history-log", RichLog)
        history_log.clear()

        events = app.bridge.collector.get_events(limit=100)
        workflow_events = [
            e for e in events if "workflow" in e.event_type
        ]

        if workflow_events:
            for event in workflow_events:
                ts = event.timestamp.strftime("%H:%M:%S")
                history_log.write(
                    f"[green]{ts}[/green] [{event.level.lower()}]{event.event_type}[/] {event.message}"
                )
        else:
            history_log.write("[dim]No workflow events captured yet[/dim]")
