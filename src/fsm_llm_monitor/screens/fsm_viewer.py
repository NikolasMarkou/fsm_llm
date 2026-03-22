from __future__ import annotations

"""FSM Viewer screen — load and visualize FSM definitions."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Static,
    Tree,
)

from fsm_llm_monitor.definitions import FSMSnapshot


class FSMViewerScreen(Screen):
    """FSM definition viewer with state graph."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_fsm: FSMSnapshot | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            # File loader
            with Horizontal(classes="panel"):
                yield Input(
                    placeholder="Path to FSM JSON file...",
                    id="fsm-path-input",
                )
                yield Button("Load", id="load-btn", variant="primary")

            with Horizontal(classes="full-width"):
                # Left: state graph tree
                with Vertical(classes="panel"):
                    yield Label("[b]STATE GRAPH[/b]", classes="panel-title")
                    yield Tree("FSM States", id="state-tree")

                # Right: state details
                with Vertical(classes="panel"):
                    yield Label("[b]STATE DETAILS[/b]", classes="panel-title")
                    yield Static("Select a state to view details", id="state-details")

            # Bottom: transitions table
            with Vertical(classes="panel"):
                yield Label("[b]TRANSITIONS[/b]", classes="panel-title")
                yield DataTable(id="transitions-table")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#transitions-table", DataTable)
        table.add_columns("From", "To", "Priority", "Description", "Conditions", "Logic")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "load-btn":
            self._load_fsm()

    def _load_fsm(self) -> None:
        path_input = self.query_one("#fsm-path-input", Input)
        path = path_input.value.strip()
        if not path:
            return

        app = self.app
        if not hasattr(app, "bridge"):
            return

        fsm = app.bridge.load_fsm_from_file(path)
        if fsm is None:
            details = self.query_one("#state-details", Static)
            details.update("[red]Failed to load FSM file[/red]")
            return

        self._current_fsm = fsm
        self._render_fsm(fsm)

    def _render_fsm(self, fsm: FSMSnapshot) -> None:
        # Populate state tree
        tree = self.query_one("#state-tree", Tree)
        tree.clear()
        tree.root.label = f"[b]{fsm.name}[/b] (v{fsm.version})"

        for state in fsm.states:
            marker = ""
            if state.is_initial:
                marker = " [green][>>][/green]"
            elif state.is_terminal:
                marker = " [red][XX][/red]"

            node = tree.root.add(
                f"{state.state_id}{marker}",
                data=state,
            )
            for trans in state.transitions:
                node.add_leaf(
                    f"-> {trans.target_state} (p={trans.priority})",
                    data=trans,
                )

        tree.root.expand_all()

        # Populate transitions table
        table = self.query_one("#transitions-table", DataTable)
        table.clear()
        for state in fsm.states:
            for trans in state.transitions:
                table.add_row(
                    state.state_id,
                    trans.target_state,
                    str(trans.priority),
                    trans.description[:40],
                    str(trans.condition_count),
                    "Yes" if trans.has_logic else "No",
                )

        # Show FSM summary in details
        details = self.query_one("#state-details", Static)
        details.update(
            f"[b]Name:[/b] {fsm.name}\n"
            f"[b]Description:[/b] {fsm.description}\n"
            f"[b]Version:[/b] {fsm.version}\n"
            f"[b]Initial State:[/b] {fsm.initial_state}\n"
            f"[b]Persona:[/b] {fsm.persona or 'None'}\n"
            f"[b]States:[/b] {fsm.state_count}\n"
        )

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        node_data = event.node.data
        if node_data is None:
            return

        details = self.query_one("#state-details", Static)

        if hasattr(node_data, "state_id"):
            # StateInfo
            details.update(
                f"[b]State:[/b] {node_data.state_id}\n"
                f"[b]Description:[/b] {node_data.description}\n"
                f"[b]Purpose:[/b] {node_data.purpose}\n"
                f"[b]Initial:[/b] {'Yes' if node_data.is_initial else 'No'}\n"
                f"[b]Terminal:[/b] {'Yes' if node_data.is_terminal else 'No'}\n"
                f"[b]Transitions:[/b] {node_data.transition_count}\n"
            )
        elif hasattr(node_data, "target_state"):
            # TransitionInfo
            details.update(
                f"[b]Target:[/b] {node_data.target_state}\n"
                f"[b]Description:[/b] {node_data.description}\n"
                f"[b]Priority:[/b] {node_data.priority}\n"
                f"[b]Conditions:[/b] {node_data.condition_count}\n"
                f"[b]Has Logic:[/b] {'Yes' if node_data.has_logic else 'No'}\n"
            )

    def action_refresh(self) -> None:
        if self._current_fsm:
            self._render_fsm(self._current_fsm)
