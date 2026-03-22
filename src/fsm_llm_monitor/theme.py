from __future__ import annotations

"""
Retro 90s terminal theme for fsm_llm_monitor.

Green-on-black CRT aesthetic with Textual CSS.
"""

RETRO_THEME_CSS = """
/* === RETRO 90s TERMINAL THEME === */

Screen {
    background: #000000;
    color: #00ff00;
}

Header {
    background: #003300;
    color: #00ff00;
    dock: top;
    height: 3;
}

Footer {
    background: #003300;
    color: #00ff00;
    dock: bottom;
}

/* --- Panels & Containers --- */

.panel {
    border: solid #004400;
    background: #0a0a0a;
    padding: 1;
    margin: 0 1;
}

.panel-title {
    color: #33ff33;
    text-style: bold;
}

/* --- Data Tables --- */

DataTable {
    background: #0a0a0a;
    color: #00ff00;
}

DataTable > .datatable--header {
    background: #003300;
    color: #33ff33;
    text-style: bold;
}

DataTable > .datatable--cursor {
    background: #004400;
    color: #00ff00;
}

DataTable > .datatable--even-row {
    background: #050505;
}

/* --- Tabs --- */

Tabs {
    background: #000000;
    dock: top;
}

Tab {
    background: #001100;
    color: #006600;
    padding: 0 2;
}

Tab.-active {
    background: #003300;
    color: #00ff00;
    text-style: bold;
}

Tab:hover {
    background: #002200;
    color: #00ff00;
}

Underline {
    color: #00ff00;
}

/* --- Log & Rich Log --- */

RichLog {
    background: #000000;
    color: #00ff00;
    border: solid #004400;
    scrollbar-color: #006600;
    scrollbar-color-hover: #00ff00;
    scrollbar-background: #001100;
}

/* --- Input & Forms --- */

Input {
    background: #0a0a0a;
    color: #00ff00;
    border: solid #004400;
}

Input:focus {
    border: solid #00ff00;
}

Button {
    background: #003300;
    color: #00ff00;
    border: solid #004400;
    min-width: 12;
}

Button:hover {
    background: #004400;
    color: #33ff33;
}

Button:focus {
    background: #004400;
    border: solid #00ff00;
}

Button.-primary {
    background: #006600;
    color: #00ff00;
}

Select {
    background: #0a0a0a;
    color: #00ff00;
    border: solid #004400;
}

Switch {
    background: #001100;
}

Switch.-on {
    background: #006600;
}

Checkbox {
    background: #000000;
    color: #00ff00;
}

/* --- Tree --- */

Tree {
    background: #0a0a0a;
    color: #00ff00;
    border: solid #004400;
}

Tree > .tree--cursor {
    background: #003300;
    color: #33ff33;
}

Tree > .tree--highlight {
    background: #002200;
}

/* --- Scrollbar --- */

* {
    scrollbar-color: #006600;
    scrollbar-color-hover: #00ff00;
    scrollbar-background: #001100;
    scrollbar-corner-color: #001100;
    scrollbar-color-active: #33ff33;
}

/* --- Labels & Static --- */

Label {
    color: #00ff00;
}

Static {
    color: #00ff00;
}

.muted {
    color: #006600;
}

.accent {
    color: #33ff33;
    text-style: bold;
}

.warning {
    color: #ffff00;
}

.error {
    color: #ff0000;
}

.success {
    color: #00ff00;
}

/* --- Status Indicators --- */

.status-active {
    color: #00ff00;
    text-style: bold;
}

.status-idle {
    color: #006600;
}

.status-error {
    color: #ff0000;
    text-style: bold;
}

.status-waiting {
    color: #ffff00;
}

/* --- Layout Helpers --- */

.full-width {
    width: 100%;
}

.half-width {
    width: 50%;
}

.metric-value {
    color: #33ff33;
    text-style: bold;
}

.metric-label {
    color: #006600;
}

/* --- Header Bar --- */

.header-title {
    color: #00ff00;
    text-style: bold;
}

.header-subtitle {
    color: #006600;
}

/* --- ProgressBar --- */

ProgressBar Bar > .bar--bar {
    color: #00ff00;
    background: #001100;
}

/* --- Rule --- */

Rule {
    color: #004400;
}

/* --- Collapsible --- */

Collapsible {
    background: #0a0a0a;
    border: solid #004400;
}

CollapsibleTitle {
    color: #00ff00;
    background: #001100;
}

CollapsibleTitle:hover {
    background: #002200;
}
"""
