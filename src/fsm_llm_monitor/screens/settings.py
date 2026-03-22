from __future__ import annotations

"""Settings screen — monitor configuration CRUD."""

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
    Select,
    Static,
)

from fsm_llm_monitor.definitions import MonitorConfig


class SettingsScreen(Screen):
    """Monitor settings CRUD."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            with Vertical(classes="panel"):
                yield Label("[b]MONITOR SETTINGS[/b]", classes="panel-title")

                with Horizontal():
                    yield Label("Refresh Interval (s): ")
                    yield Input(
                        value="1.0",
                        id="refresh-interval-input",
                        type="number",
                    )

                with Horizontal():
                    yield Label("Max Events:           ")
                    yield Input(
                        value="1000",
                        id="max-events-input",
                        type="integer",
                    )

                with Horizontal():
                    yield Label("Max Log Lines:        ")
                    yield Input(
                        value="5000",
                        id="max-log-lines-input",
                        type="integer",
                    )

                with Horizontal():
                    yield Label("Log Level:            ")
                    yield Select(
                        [
                            ("DEBUG", "DEBUG"),
                            ("INFO", "INFO"),
                            ("WARNING", "WARNING"),
                            ("ERROR", "ERROR"),
                        ],
                        value="INFO",
                        id="log-level-select",
                    )

                with Horizontal():
                    yield Checkbox(
                        "Show Internal Keys",
                        value=False,
                        id="show-internal-keys-check",
                    )

                with Horizontal():
                    yield Checkbox(
                        "Auto-scroll Logs",
                        value=True,
                        id="auto-scroll-logs-check",
                    )

                with Horizontal():
                    yield Button("Save", id="save-btn", variant="primary")
                    yield Button("Reset", id="reset-btn")

            with Vertical(classes="panel"):
                yield Label("[b]SYSTEM INFO[/b]", classes="panel-title")
                yield Static("Loading...", id="system-info")

            with Vertical(classes="panel"):
                yield Label("[b]CONNECTION STATUS[/b]", classes="panel-title")
                yield Static("Not connected", id="connection-status")
        yield Footer()

    def on_mount(self) -> None:
        self._load_settings()
        self._update_system_info()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            self._save_settings()
        elif event.button.id == "reset-btn":
            self._reset_settings()

    def action_refresh(self) -> None:
        self._load_settings()
        self._update_system_info()

    def _load_settings(self) -> None:
        app = self.app
        if not hasattr(app, "bridge"):
            return

        config = app.bridge.config
        self.query_one("#refresh-interval-input", Input).value = str(
            config.refresh_interval
        )
        self.query_one("#max-events-input", Input).value = str(config.max_events)
        self.query_one("#max-log-lines-input", Input).value = str(
            config.max_log_lines
        )

        level_select = self.query_one("#log-level-select", Select)
        level_select.value = config.log_level

        self.query_one("#show-internal-keys-check", Checkbox).value = (
            config.show_internal_keys
        )
        self.query_one("#auto-scroll-logs-check", Checkbox).value = (
            config.auto_scroll_logs
        )

    def _save_settings(self) -> None:
        app = self.app
        if not hasattr(app, "bridge"):
            return

        try:
            config = MonitorConfig(
                refresh_interval=float(
                    self.query_one("#refresh-interval-input", Input).value or "1.0"
                ),
                max_events=int(
                    self.query_one("#max-events-input", Input).value or "1000"
                ),
                max_log_lines=int(
                    self.query_one("#max-log-lines-input", Input).value or "5000"
                ),
                log_level=str(
                    self.query_one("#log-level-select", Select).value or "INFO"
                ),
                show_internal_keys=self.query_one(
                    "#show-internal-keys-check", Checkbox
                ).value,
                auto_scroll_logs=self.query_one(
                    "#auto-scroll-logs-check", Checkbox
                ).value,
            )
            app.bridge.config = config
            self.notify("Settings saved", severity="information")
        except (ValueError, TypeError) as e:
            self.notify(f"Invalid setting: {e}", severity="error")

    def _reset_settings(self) -> None:
        app = self.app
        if not hasattr(app, "bridge"):
            return

        app.bridge.config = MonitorConfig()
        self._load_settings()
        self.notify("Settings reset to defaults", severity="information")

    def _update_system_info(self) -> None:
        try:
            from fsm_llm_monitor.__version__ import __version__

            info_parts = [
                f"[b]Monitor Version:[/b] {__version__}",
            ]

            try:
                from fsm_llm import __version__ as core_version

                info_parts.append(f"[b]FSM-LLM Version:[/b] {core_version}")
            except ImportError:
                pass

            try:
                import textual

                info_parts.append(f"[b]Textual Version:[/b] {textual.__version__}")
            except ImportError:
                pass

            info = self.query_one("#system-info", Static)
            info.update("\n".join(info_parts))
        except Exception:
            pass

        # Connection status
        app = self.app
        if hasattr(app, "bridge"):
            status = self.query_one("#connection-status", Static)
            if app.bridge.connected:
                metrics = app.bridge.get_metrics()
                status.update(
                    f"[green]Connected[/green]\n"
                    f"Active conversations: {metrics.active_conversations}\n"
                    f"Total events: {metrics.total_events}"
                )
            else:
                status.update("[yellow]Disconnected[/yellow] — No API instance")
