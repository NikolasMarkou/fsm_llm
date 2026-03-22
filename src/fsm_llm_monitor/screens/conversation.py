from __future__ import annotations

"""Conversation monitor screen — live state, context, and message history."""


from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    RichLog,
    Select,
    Static,
)


class ConversationScreen(Screen):
    """Live conversation monitor."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            # Conversation selector
            with Horizontal(classes="panel"):
                yield Label("Conversation: ", classes="panel-title")
                yield Select(
                    [],
                    prompt="Select conversation...",
                    id="conversation-select",
                )

            with Horizontal(classes="full-width"):
                # Left: state + context
                with Vertical(classes="panel"):
                    yield Label("[b]CURRENT STATE[/b]", classes="panel-title")
                    yield Static("No conversation selected", id="state-display")
                    yield Label("[b]CONTEXT DATA[/b]", classes="panel-title")
                    yield DataTable(id="context-table")

                # Right: message history
                with Vertical(classes="panel"):
                    yield Label("[b]MESSAGE HISTORY[/b]", classes="panel-title")
                    yield RichLog(id="message-log", max_lines=200)

            # Bottom: LLM details
            with Vertical(classes="panel"):
                yield Label("[b]LLM DETAILS[/b]", classes="panel-title")
                yield Static("No data", id="llm-details")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#context-table", DataTable)
        table.add_columns("Key", "Value")
        self._refresh_conversation_list()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.value and event.value != Select.BLANK:
            self._load_conversation(str(event.value))

    def action_refresh(self) -> None:
        self._refresh_conversation_list()
        select = self.query_one("#conversation-select", Select)
        if select.value and select.value != Select.BLANK:
            self._load_conversation(str(select.value))

    def _refresh_conversation_list(self) -> None:
        app = self.app
        if not hasattr(app, "bridge"):
            return

        convs = app.bridge.get_active_conversations()
        select = self.query_one("#conversation-select", Select)
        options = [(c, c) for c in convs]
        select.set_options(options)

    def _load_conversation(self, conversation_id: str) -> None:
        app = self.app
        if not hasattr(app, "bridge"):
            return

        snap = app.bridge.get_conversation_snapshot(conversation_id)
        if snap is None:
            return

        # Update state display
        state_display = self.query_one("#state-display", Static)
        status = "[red]TERMINAL[/red]" if snap.is_terminal else "[green]ACTIVE[/green]"
        state_display.update(
            f"[b]State:[/b] {snap.current_state} {status}\n"
            f"[b]Description:[/b] {snap.state_description}\n"
            f"[b]Stack Depth:[/b] {snap.stack_depth}\n"
            f"[b]ID:[/b] {conversation_id}"
        )

        # Update context table
        table = self.query_one("#context-table", DataTable)
        table.clear()
        for key, value in sorted(snap.context_data.items()):
            val_str = str(value)
            if len(val_str) > 60:
                val_str = val_str[:57] + "..."
            table.add_row(key, val_str)

        # Update message history
        log = self.query_one("#message-log", RichLog)
        log.clear()
        for exchange in snap.message_history:
            for role, msg in exchange.items():
                if role == "user":
                    log.write(f"[bold cyan]USER:[/bold cyan] {msg}")
                else:
                    log.write(f"[bold green]SYSTEM:[/bold green] {msg}")
            log.write("---")

        # Update LLM details
        llm_details = self.query_one("#llm-details", Static)
        parts = []
        if snap.last_extraction:
            confidence = snap.last_extraction.get("confidence", "N/A")
            parts.append(f"[b]Extraction Confidence:[/b] {confidence}")
            extracted = snap.last_extraction.get("extracted_data", {})
            if extracted:
                parts.append(f"[b]Extracted Keys:[/b] {', '.join(extracted.keys())}")
        if snap.last_transition:
            selected = snap.last_transition.get("selected_transition", "N/A")
            parts.append(f"[b]Last Transition:[/b] {selected}")
        if snap.last_response:
            msg_type = snap.last_response.get("message_type", "N/A")
            parts.append(f"[b]Response Type:[/b] {msg_type}")

        llm_details.update("\n".join(parts) if parts else "No LLM data available")
