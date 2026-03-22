from __future__ import annotations

"""Log viewer screen — real-time log streaming with filters."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
)


class LogScreen(Screen):
    """Real-time log viewer with level and text filters."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("x", "clear_logs", "Clear"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            # Filter bar
            with Horizontal(classes="panel"):
                yield Label("Level: ")
                yield Select(
                    [
                        ("DEBUG", "DEBUG"),
                        ("INFO", "INFO"),
                        ("WARNING", "WARNING"),
                        ("ERROR", "ERROR"),
                        ("CRITICAL", "CRITICAL"),
                    ],
                    value="INFO",
                    id="level-select",
                )
                yield Label(" Filter: ")
                yield Input(
                    placeholder="Search text...",
                    id="text-filter",
                )
                yield Button("Apply", id="apply-filter-btn")
                yield Checkbox("Auto-scroll", value=True, id="auto-scroll-check")

            # Log display
            with Vertical(classes="panel"):
                yield Label("[b]LOG STREAM[/b]", classes="panel-title")
                yield RichLog(id="log-display", max_lines=5000)

            # Stats bar
            with Horizontal(classes="panel"):
                yield Static("Total: 0 | Shown: 0", id="log-stats")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_logs()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply-filter-btn":
            self._refresh_logs()

    def action_refresh(self) -> None:
        self._refresh_logs()

    def action_clear_logs(self) -> None:
        log_display = self.query_one("#log-display", RichLog)
        log_display.clear()

    def _refresh_logs(self) -> None:
        app = self.app
        if not hasattr(app, "bridge"):
            return

        # Get filter values
        level_select = self.query_one("#level-select", Select)
        text_filter = self.query_one("#text-filter", Input)
        level = str(level_select.value) if level_select.value != Select.BLANK else "INFO"
        search_text = text_filter.value.strip().lower()

        # Get filtered logs
        logs = app.bridge.collector.get_logs(level=level)
        if search_text:
            logs = [r for r in logs if search_text in r.message.lower()]

        # Display
        log_display = self.query_one("#log-display", RichLog)
        log_display.clear()

        level_colors = {
            "DEBUG": "dim green",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold red",
        }

        for record in reversed(logs[:500]):  # Show oldest first, limit 500
            ts = record.timestamp.strftime("%H:%M:%S.%f")[:12]
            color = level_colors.get(record.level, "green")
            conv = f" [{record.conversation_id}]" if record.conversation_id else ""
            log_display.write(
                f"[{color}]{ts} {record.level:<8}[/] "
                f"[dim]{record.module}:{record.line}[/dim]{conv} "
                f"{record.message}"
            )

        # Update stats
        total = len(app.bridge.collector.get_logs())
        shown = min(len(logs), 500)
        stats = self.query_one("#log-stats", Static)
        stats.update(f"Total: {total} | Shown: {shown} | Level: >= {level}")
