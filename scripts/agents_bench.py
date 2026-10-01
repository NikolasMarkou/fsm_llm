#!/usr/bin/env python3
"""Pre-registered live bench for agent tool loops (E1 of docs/agents_roadmap.md).

Same discipline as ``scripts/harness_bench.py``: raw jsonl rows, a manifest
written before row 1, blocks run once at fixed n, and a ``report`` that
recounts every number from the committed rows.

What it measures: ~38 deterministic tasks in 7 categories, each with pure
in-file tools and a non-LLM grader, run for ``--trials`` (3) fresh trials per
task on one agent arm. Graded on ``AgentResult.answer`` only; ``success`` and
``stop_reason`` are recorded beside it, so the success-vs-correct cross-tab
shows runs that claim success on a wrong answer (E5 at bench level).

Usage (always the venv; see scripts/bench_data/README.md):
    .venv/bin/python scripts/agents_bench.py list-tasks --verify
    .venv/bin/python scripts/agents_bench.py register \\
        --bench-id agents-react --block B1 --arm fsm_advance
    .venv/bin/python scripts/agents_bench.py run \\
        --bench-id agents-react --block B1 --arm fsm_advance --trials 3
    .venv/bin/python scripts/agents_bench.py report agents-react \\
        --blocks B0 B1 --pair B1/fsm_advance:B0/legacy

Arm labels name the code an arm ran, so a label is never reused for rows of
changed code: ``legacy`` and ``native_fc`` are B0's arms (d4b1626), retired
from ``ARMS``; their rows still recount (``report`` reads labels from file
names).

Call meter (``wrapper_version`` in every manifest): "2" for new blocks, core's
own counters: each trial gets one ``LiteLLMInterface`` built from the arm's
config, injected as ``llm_interface=``, and its ``usage()`` fills the row.
B0 and B1 rows were counted by "1", ``CallMeter`` patching the litellm
bindings; it stays only as the reference ``meter_parity`` checks "2" against,
so a "2" block is comparable with B0/B1. ``report`` never needs either meter.

Import hygiene: the module top level is stdlib only (pinned by an AST test
and a socket-disabled subprocess import). ``fsm_llm``, ``litellm`` and the
agents load lazily inside ``run``; ``report`` and ``list-tasks`` never load
them, so they work offline and after the measured code is deleted.
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import importlib.util
import inspect
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BENCH_DATA = ROOT / "scripts" / "bench_data"
MODEL = "ollama_chat/qwen3.5:4b"
TRIALS = 3
WRAPPER_VERSION = "2"
ANSWER_CHARS = 500
ERROR_CHARS = 300

#: Per-trial agent limits, recorded in every manifest. Temperature is the
#: package default (``fsm_llm.agents.constants.Defaults.TEMPERATURE``), so the
#: bench measures the shipped sampling, not a tuned one.
LIMITS: dict[str, Any] = {
    "max_iterations": 8,
    "timeout_seconds": 180.0,
    "temperature": 0.5,
    "max_tokens": 1000,
}


def _load_harness_bench() -> Any:
    """harness_bench, loaded by file path (stdlib only, inert at import).

    One module object per process: a copy already imported as
    ``harness_bench`` (the tests put scripts/ on sys.path) is reused, so
    ``BenchDataError`` is one class, not two.
    """
    # DECISION plan-2026-09-29T184639-65baa765/D-002
    # Do NOT copy wilson_ci, fisher_exact_two_sided, append_row, read_rows,
    # _model_digest or BenchDataError into this file, and do NOT import them
    # from fsm_llm.eval: a third copy would need its own parity test, and
    # fsm_llm pulls litellm, which opens a socket at import (the offline
    # `report` must not). harness_bench is the one stdlib source; its copies
    # are held equal to fsm_llm.eval by test_bench_parity.py. See D-002.
    cached = sys.modules.get("harness_bench")
    if cached is not None:
        return cached
    path = Path(__file__).resolve().parent / "harness_bench.py"
    spec = importlib.util.spec_from_file_location("harness_bench", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["harness_bench"] = module
    spec.loader.exec_module(module)
    return module


hb = _load_harness_bench()
BenchDataError = hb.BenchDataError
wilson_ci = hb.wilson_ci
fisher_exact_two_sided = hb.fisher_exact_two_sided
append_row = hb.append_row
read_rows = hb.read_rows

#: harness_bench's six comparability fields plus the agent-bench extras.
EXTRA_MANIFEST_FIELDS = (
    "tasks_sha256",
    "trials",
    "temperature",
    "limits",
    "wrapper_version",
)
#: What a block's requests carried beyond the task text, so a pair can show
#: every difference besides the model (``request_disclosure``,
#: ``llm_request_settings``). Recorded B0/B1 manifests predate them; ``report
#: --pair`` prints a field a manifest lacks as "not recorded".
DISCLOSURE_FIELDS = (
    "llm_request",
    "agent_class",
    "run_cap",
    "tool_schemas_sha256",
    "first_request",
)
MANIFEST_FIELDS = tuple(hb.MANIFEST_FIELDS) + EXTRA_MANIFEST_FIELDS + DISCLOSURE_FIELDS
#: How ``report --pair`` prints a field one manifest does not carry.
NOT_RECORDED = "not recorded"


# --- BEGIN TASKS (hashed: tools, tasks, graders, reference solvers) ----------
# Everything between the BEGIN/END markers is the fixture: `tasks_sha256` is
# the sha256 of these lines, so any edit here makes a new, incomparable task
# set. Arms live outside the markers, so registering one changes nothing here.

COUNTRIES: dict[str, dict[str, Any]] = {
    "veloria": {"capital": "Maskett", "population": 4218000, "area_km2": 51200,
                "currency": "VEL"},
    "orrin": {"capital": "Dunmere", "population": 1175500, "area_km2": 20900,
              "currency": "ORC"},
    "tessaly": {"capital": "Brisk", "population": 9832000, "area_km2": 120400,
                "currency": "TSL"},
    "quandor": {"capital": "Halvik", "population": 612300, "area_km2": 8700,
                "currency": "QDR"},
    "marrowind": {"capital": "Ostrel", "population": 2750000, "area_km2": 33300,
                  "currency": "MWD"},
}  # fmt: skip
#: Value of one unit of each currency in USD.
USD_VALUE: dict[str, float] = {
    "USD": 1.0,
    "VEL": 0.25,
    "ORC": 1.6,
    "TSL": 0.04,
    "QDR": 3.2,
    "MWD": 0.5,
}
EMPLOYEES: dict[str, dict[str, Any]] = {
    "E-101": {"name": "Ana Ruiz", "department": "Logistics", "salary": 58000,
              "manager": "E-205"},
    "E-205": {"name": "Bo Lindqvist", "department": "Logistics", "salary": 81000,
              "manager": "E-300"},
    "E-300": {"name": "Chidi Okafor", "department": "Operations",
              "salary": 120000, "manager": None},
    "E-412": {"name": "Dana Wu", "department": "Research", "salary": 97000,
              "manager": "E-300"},
    "E-518": {"name": "Eli Park", "department": "Research", "salary": 64000,
              "manager": "E-205"},
}  # fmt: skip
DEPARTMENTS: dict[str, dict[str, Any]] = {
    "logistics": {"budget": 1250000, "floor": 3, "head": "E-205"},
    "research": {"budget": 2400000, "floor": 5, "head": "E-412"},
    "operations": {"budget": 900000, "floor": 1, "head": "E-300"},
}
PRICES: dict[str, float] = {
    "AB-100": 12.5,
    "AB-200": 7.25,
    "CD-300": 3.1,
    "EF-400": 45.0,
}
STOCK: dict[str, int] = {"AB-100": 140, "AB-200": 0, "CD-300": 2750, "EF-400": 18}
STATIONS: dict[str, float] = {"ST-7": 18.4, "ST-9": 22.1}
#: Stations whose first reading per trial fails (error_recovery).
FLAKY_STATIONS = frozenset({"ST-7"})
#: Unit -> (dimension, value of one unit in the dimension's base unit).
UNITS: dict[str, tuple[str, float]] = {
    "m": ("length", 1.0),
    "km": ("length", 1000.0),
    "ft": ("length", 0.3048),
    "mi": ("length", 1609.344),
    "kg": ("mass", 1.0),
    "lb": ("mass", 0.45359237),
}
_SKU_RE = re.compile(r"^[A-Z]{2}-\d{3}$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CALC_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}


def _fmt_number(value: float) -> str:
    """Integral values without a decimal point; others rounded to 6 places."""
    if float(value).is_integer():
        return str(int(value))
    return str(round(value, 6))


def _calc_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _calc_eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _calc_eval(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _CALC_BINOPS:
        left, right = _calc_eval(node.left), _calc_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 64:
            raise ValueError("exponent too large")
        return _CALC_BINOPS[type(node.op)](left, right)
    raise ValueError(
        "unsupported expression: use only numbers (no thousands separators, "
        "no units, no %) and + - * / // % ** ( )"
    )


def _convert(value: float, from_unit: str, to_unit: str) -> float:
    src, dst = from_unit.strip().lower(), to_unit.strip().lower()
    if {src, dst} <= {"c", "f"}:
        if src == dst:
            return value
        return value * 9 / 5 + 32 if src == "c" else (value - 32) * 5 / 9
    if src not in UNITS or dst not in UNITS:
        bad = src if src not in UNITS and src not in ("c", "f") else dst
        raise ValueError(
            f"unsupported unit {bad!r}; use one of: m, km, ft, mi, kg, lb, c, f"
        )
    (dim_a, fa), (dim_b, fb) = UNITS[src], UNITS[dst]
    if dim_a != dim_b:
        raise ValueError(f"cannot convert {dim_a} to {dim_b}")
    return value * fa / fb


def make_tools() -> dict[str, Callable[..., Any]]:
    """A FRESH set of every bench tool (stateful tools start clean per trial).

    Interface contract (callers: ``run_block`` once per trial, the reference
    solvers in the tests, ``list-tasks --verify``):
        - Returns name -> plain callable with type hints and a docstring (the
          tool registry infers each schema from them). Pure and deterministic
          except ``station_reading``, whose FLAKY stations fail on the first
          call of each fresh set.
        - A tool signals bad input by raising ``ValueError``/``RuntimeError``
          (the agent sees the message as the observation); an unknown record
          is a normal "not found" return, not an exception.
    """
    first_read: set[str] = set()

    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression and return the result."""
        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"cannot parse expression: {exc.msg}") from exc
        return _fmt_number(_calc_eval(tree))

    def lookup_country(name: str) -> str:
        """Look up a country's capital, population, area and currency code."""
        record = COUNTRIES.get(name.strip().lower())
        if record is None:
            return f"not found: no country named {name!r} in the database"
        return json.dumps({"country": name.strip().title(), **record})

    def get_employee(employee_id: str) -> str:
        """Get an employee's record: name, department, salary and manager id."""
        record = EMPLOYEES.get(employee_id.strip().upper())
        if record is None:
            return f"not found: no employee with id {employee_id!r}"
        return json.dumps({"id": employee_id.strip().upper(), **record})

    def get_department(name: str) -> str:
        """Get a department's annual budget, floor and head (an employee id)."""
        record = DEPARTMENTS.get(name.strip().lower())
        if record is None:
            return f"not found: no department named {name!r}"
        return json.dumps({"department": name.strip().title(), **record})

    def convert_units(value: float, from_unit: str, to_unit: str) -> str:
        """Convert a value between units of length, mass or temperature."""
        return _fmt_number(round(_convert(float(value), from_unit, to_unit), 6))

    def days_between(start_date: str, end_date: str) -> str:
        """Count the days from one date to another."""
        for raw in (start_date, end_date):
            if not _ISO_DATE_RE.match(raw.strip()):
                raise ValueError(
                    f"invalid date {raw!r}: dates must be YYYY-MM-DD, e.g. 2024-03-05"
                )
        start = _dt.date.fromisoformat(start_date.strip())
        end = _dt.date.fromisoformat(end_date.strip())
        return str((end - start).days)

    def list_stats(numbers: list[float]) -> str:
        """Summary statistics (count, sum, mean, median, min, max) of numbers."""
        values = [float(v) for v in numbers]
        if not values:
            raise ValueError("numbers must be a non-empty list")
        return json.dumps(
            {
                "count": len(values),
                "sum": round(sum(values), 6),
                "mean": round(statistics.mean(values), 6),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
        )

    def order_total(quantities: dict, discount_percent: float) -> str:
        """Price an order: quantities maps SKU to units; returns the USD total."""
        total = 0.0
        for sku, qty in quantities.items():
            if sku not in PRICES:
                raise ValueError(f"unknown SKU {sku!r}; known: {sorted(PRICES)}")
            total += PRICES[sku] * int(qty)
        return _fmt_number(round(total * (1 - float(discount_percent) / 100), 2))

    def check_inventory(sku: str) -> str:
        """Units in stock for a product SKU."""
        if not _SKU_RE.match(sku.strip()):
            raise ValueError(
                f"invalid SKU {sku!r}: use two UPPERCASE letters, a dash and "
                "three digits, e.g. AB-100"
            )
        if sku.strip() not in STOCK:
            return f"not found: no product with SKU {sku.strip()}"
        return json.dumps({"sku": sku.strip(), "in_stock": STOCK[sku.strip()]})

    def exchange_rate(from_currency: str, to_currency: str) -> str:
        """How many units of to_currency one unit of from_currency buys."""
        src, dst = from_currency.strip().upper(), to_currency.strip().upper()
        for code in (src, dst):
            if code not in USD_VALUE:
                return f"not found: unknown currency code {code!r}"
        return _fmt_number(round(USD_VALUE[src] / USD_VALUE[dst], 6))

    def station_reading(station_id: str) -> str:
        """Current temperature in Celsius at a weather station."""
        sid = station_id.strip().upper()
        if sid not in STATIONS:
            return f"not found: no station {sid}"
        if sid in FLAKY_STATIONS and sid not in first_read:
            first_read.add(sid)
            raise RuntimeError("sensor network busy; retry the same call")
        return json.dumps({"station": sid, "celsius": STATIONS[sid]})

    def repeat_text(text: str, times: int, separator: str) -> str:
        """Repeat text a number of times, joined by a separator."""
        return separator.join([text] * int(times))

    def count_letter(word: str, letter: str) -> str:
        """Count how many times a letter occurs in a word."""
        return str(word.lower().count(letter.strip().lower()))

    def archive_search(query: str) -> str:
        """Search the company history archive."""
        return f"0 documents match {query!r}"

    # Distractors: plausible, deterministic, irrelevant to every task.
    def get_weather(city: str) -> str:
        """Today's weather forecast for a city."""
        return f"{city}: partly cloudy"

    def send_email(to: str, subject: str, body: str) -> str:
        """Send an email."""
        return "queued"

    def translate_text(text: str, language: str) -> str:
        """Translate text into another language."""
        return f"[{language}] {text}"

    def search_news(topic: str) -> str:
        """Latest news headlines on a topic."""
        return f"No major headlines about {topic} today."

    def get_stock_price(ticker: str) -> str:
        """Latest share price for a stock ticker."""
        return f"{ticker.upper()}: market closed"

    def set_reminder(text: str, minutes: int) -> str:
        """Set a reminder."""
        return "reminder set"

    def play_music(song: str) -> str:
        """Play a song."""
        return "playing"

    def book_flight(origin: str, destination: str, date: str) -> str:
        """Book a flight."""
        return "no seats available"

    def get_horoscope(sign: str) -> str:
        """Daily horoscope for a star sign."""
        return "A calm day."

    def random_fact(topic: str) -> str:
        """A fun fact about a topic."""
        return f"{topic} is interesting."

    return {fn.__name__: fn for fn in (
        calculator, lookup_country, get_employee, get_department, convert_units,
        days_between, list_stats, order_total, check_inventory, exchange_rate,
        station_reading, repeat_text, count_letter, archive_search,
        get_weather, send_email, translate_text, search_news, get_stock_price,
        set_reminder, play_music, book_flight, get_horoscope, random_fact,
    )}  # fmt: skip


DISTRACTORS = (
    "get_weather",
    "send_email",
    "translate_text",
    "search_news",
    "get_stock_price",
    "set_reminder",
    "play_music",
    "book_flight",
    "get_horoscope",
    "random_fact",
)
CATEGORIES = (
    "single_tool",
    "multi_step_chain",
    "no_tool_needed",
    "error_recovery",
    "typed_args",
    "distractor_tools",
    "unanswerable",
)
#: Phrases (normalised) any of which marks an answer as "the data is absent".
NOT_FOUND_PHRASES = (
    "not found",
    "no record",
    "no country",
    "no employee",
    "no product",
    "no station",
    "no such",
    "does not exist",
    "doesnt exist",
    "not exist",
    "unknown",
    "could not find",
    "couldnt find",
    "cannot find",
    "cant find",
    "unable to find",
    "unable to determine",
    "no information",
    "not available",
    "no results",
    "no documents",
    "0 documents",
    "no data",
    "not in the",
)


@dataclass(frozen=True)
class Task:
    """One bench task. ``grader`` is data; ``reference`` solves it with tools."""

    id: str
    category: str
    prompt: str
    tools: tuple[str, ...]
    grader: dict[str, Any]
    reference: Callable[[dict[str, Callable[..., Any]]], str]


def _j(text: str) -> dict[str, Any]:
    return json.loads(text)


_BASIC = ("calculator", "lookup_country", "get_employee")
_NF: dict[str, Any] = {"kind": "contains", "any_of": list(NOT_FOUND_PHRASES)}

TASKS: tuple[Task, ...] = (
    # single_tool: one call answers it (fictional data: memory cannot).
    Task("st-capital", "single_tool",
         "What is the capital of Veloria? Use the tools. Reply with only the "
         "city name.",
         ("lookup_country",), {"kind": "exact", "value": "Maskett"},
         lambda t: _j(t["lookup_country"]("Veloria"))["capital"]),
    Task("st-multiply", "single_tool",
         "Use the calculator to compute 48213 * 977. Give the final number.",
         ("calculator",), {"kind": "numeric", "values": [47104101], "tol": 0.5},
         lambda t: t["calculator"]("48213 * 977")),
    Task("st-convert", "single_tool",
         "Convert 26.2 miles to kilometres with the unit tool. Give the result "
         "rounded to two decimals.",
         ("convert_units",), {"kind": "numeric", "values": [42.16], "tol": 0.011},
         lambda t: t["convert_units"](26.2, "mi", "km")),
    Task("st-days", "single_tool",
         "How many days are there from 2024-01-15 to 2024-03-01? Use the date "
         "tool.",
         ("days_between",), {"kind": "numeric", "values": [46], "tol": 0},
         lambda t: t["days_between"]("2024-01-15", "2024-03-01")),
    Task("st-letter", "single_tool",
         "How many times does the letter 'r' occur in the word 'strawberryrr'? "
         "Use the counting tool.",
         ("count_letter",), {"kind": "numeric", "values": [5], "tol": 0},
         lambda t: t["count_letter"]("strawberryrr", "r")),
    Task("st-stock", "single_tool",
         "How many units of SKU CD-300 are in stock? Use the inventory tool.",
         ("check_inventory",), {"kind": "numeric", "values": [2750], "tol": 0},
         lambda t: str(_j(t["check_inventory"]("CD-300"))["in_stock"])),
    # multi_step_chain: 2-4 dependent calls; a shortcut gives a wrong number.
    Task("ch-density", "multi_step_chain",
         "What is the population density of Tessaly in people per square "
         "kilometre? Look the country up, then use the calculator. Round to one "
         "decimal.",
         ("lookup_country", "calculator"),
         {"kind": "numeric", "values": [81.7], "tol": 0.06},
         lambda t: t["calculator"]("{population} / {area_km2}".format(
             **_j(t["lookup_country"]("Tessaly"))))),
    Task("ch-manager-budget", "multi_step_chain",
         "Find the manager of employee E-518, then report the annual budget of "
         "the department that the MANAGER works in.",
         ("get_employee", "get_department"),
         {"kind": "numeric", "values": [1250000], "tol": 0},
         lambda t: str(_j(t["get_department"](_j(t["get_employee"](
             _j(t["get_employee"]("E-518"))["manager"]))["department"]))["budget"])),
    Task("ch-currency", "multi_step_chain",
         "How much is 250 QDR worth in VEL? Get the exchange rate with the tool, "
         "then use the calculator.",
         ("exchange_rate", "calculator"),
         {"kind": "numeric", "values": [3200], "tol": 0.5},
         lambda t: t["calculator"](f"250 * {t['exchange_rate']('QDR', 'VEL')}")),
    Task("ch-salaries", "multi_step_chain",
         "What is the combined annual salary of employees E-101 and E-412 plus "
         "their two managers (four people in total)?",
         ("get_employee", "calculator"),
         {"kind": "numeric", "values": [356000], "tol": 0},
         lambda t: t["calculator"](" + ".join(
             str(_j(t["get_employee"](eid))["salary"]) for eid in (
                 "E-101", _j(t["get_employee"]("E-101"))["manager"],
                 "E-412", _j(t["get_employee"]("E-412"))["manager"])))),
    Task("ch-fahrenheit", "multi_step_chain",
         "Read the temperature at station ST-9 and report it in Fahrenheit.",
         ("station_reading", "convert_units"),
         {"kind": "numeric", "values": [71.78], "tol": 0.05},
         lambda t: t["convert_units"](
             _j(t["station_reading"]("ST-9"))["celsius"], "c", "f")),
    Task("ch-order-local", "multi_step_chain",
         "An order of 3 units of AB-100 and 10 units of CD-300 with no discount "
         "is priced in USD. What does it cost in the currency of Veloria?",
         ("order_total", "lookup_country", "exchange_rate", "calculator"),
         {"kind": "numeric", "values": [274], "tol": 0.05},
         lambda t: t["calculator"]("{} * {}".format(
             t["order_total"]({"AB-100": 3, "CD-300": 10}, 0),
             t["exchange_rate"]("USD", _j(t["lookup_country"]("Veloria"))[
                 "currency"])))),
    # no_tool_needed: tools are offered but the answer needs none.
    Task("nt-ready", "no_tool_needed",
         "Reply with exactly the word: ready",
         _BASIC, {"kind": "exact", "value": "ready"}, lambda t: "ready"),
    Task("nt-sum", "no_tool_needed",
         "What is 12 plus 30? Answer with just the number.",
         _BASIC, {"kind": "numeric", "values": [42], "tol": 0}, lambda t: "42"),
    Task("nt-reverse", "no_tool_needed",
         "Write the word 'planet' backwards. Reply with only that word.",
         _BASIC, {"kind": "exact", "value": "tenalp"}, lambda t: "tenalp"),
    Task("nt-paint", "no_tool_needed",
         "Which colour do you get by mixing blue and yellow paint? One word.",
         _BASIC, {"kind": "exact", "value": "green"}, lambda t: "Green"),
    Task("nt-rgb", "no_tool_needed",
         "Name the three primary colours of light used in RGB screens.",
         _BASIC, {"kind": "tokens", "all_of": ["red", "green", "blue"]},
         lambda t: "red, green and blue"),
    # error_recovery: the natural first call fails with an instructive error.
    Task("er-date-format", "error_recovery",
         "How many days are there between March 5, 2024 and April 25, 2024? "
         "Use the date tool.",
         ("days_between",), {"kind": "numeric", "values": [51], "tol": 0},
         lambda t: t["days_between"]("2024-03-05", "2024-04-25")),
    Task("er-sku-case", "error_recovery",
         "How many units of product ef-400 are in stock? Use the inventory tool.",
         ("check_inventory",), {"kind": "numeric", "values": [18], "tol": 0},
         lambda t: str(_j(t["check_inventory"]("EF-400"))["in_stock"])),
    Task("er-flaky", "error_recovery",
         "What is the current temperature at station ST-7 in Celsius? Use the "
         "station tool.",
         ("station_reading",), {"kind": "numeric", "values": [18.4], "tol": 0.01},
         lambda t: _flaky_twice(t)),
    Task("er-unit-name", "error_recovery",
         "Convert 150 pounds to kilograms with the unit tool, rounded to one "
         "decimal.",
         ("convert_units",), {"kind": "numeric", "values": [68.0], "tol": 0.06},
         lambda t: t["convert_units"](150, "lb", "kg")),
    Task("er-calc-separators", "error_recovery",
         "Use the calculator to compute 2,480 times 0.15.",
         ("calculator",), {"kind": "numeric", "values": [372], "tol": 0.01},
         lambda t: t["calculator"]("2480 * 0.15")),
    # typed_args: list, dict, int and float parameters.
    Task("ty-mean", "typed_args",
         "What is the mean of 14, 3, 27, 8, 19, 11 and 22? Use the statistics "
         "tool and round to two decimals.",
         ("list_stats",), {"kind": "numeric", "values": [14.86], "tol": 0.006},
         lambda t: str(_j(t["list_stats"]([14, 3, 27, 8, 19, 11, 22]))["mean"])),
    Task("ty-range", "typed_args",
         "What is the difference between the largest and the smallest of 5.5, "
         "12.25, 3.75 and 9.0? Use the statistics tool and the calculator.",
         ("list_stats", "calculator"),
         {"kind": "numeric", "values": [8.5], "tol": 0.001},
         lambda t: t["calculator"]("{max} - {min}".format(
             **_j(t["list_stats"]([5.5, 12.25, 3.75, 9.0]))))),
    Task("ty-order", "typed_args",
         "Use the order tool to price 4 units of AB-200 and 2 units of EF-400 "
         "with a 15 percent discount.",
         ("order_total",), {"kind": "numeric", "values": [101.15], "tol": 0.01},
         lambda t: t["order_total"]({"AB-200": 4, "EF-400": 2}, 15)),
    Task("ty-repeat", "typed_args",
         "Use the repeat tool to repeat the text 'ab' 4 times separated by '-'. "
         "Reply with only the result.",
         ("repeat_text",), {"kind": "exact", "value": "ab-ab-ab-ab"},
         lambda t: t["repeat_text"]("ab", 4, "-")),
    Task("ty-two-conversions", "typed_args",
         "Convert 37.5 degrees Celsius to Fahrenheit and 250 feet to metres. "
         "Report both numbers.",
         ("convert_units",), {"kind": "numeric", "values": [99.5, 76.2],
                              "tol": 0.01},
         lambda t: "{} F and {} m".format(t["convert_units"](37.5, "c", "f"),
                                          t["convert_units"](250, "ft", "m"))),
    # distractor_tools: one relevant tool among ten irrelevant ones.
    Task("di-currency", "distractor_tools",
         "Which currency code does Quandor use? Reply with the code only.",
         ("lookup_country", *DISTRACTORS), {"kind": "exact", "value": "QDR"},
         lambda t: _j(t["lookup_country"]("Quandor"))["currency"]),
    Task("di-floor", "distractor_tools",
         "On which floor is the Research department?",
         ("get_department", *DISTRACTORS),
         {"kind": "numeric", "values": [5], "tol": 0},
         lambda t: str(_j(t["get_department"]("Research"))["floor"])),
    Task("di-stock", "distractor_tools",
         "How many units of AB-100 are in stock?",
         ("check_inventory", *DISTRACTORS),
         {"kind": "numeric", "values": [140], "tol": 0},
         lambda t: str(_j(t["check_inventory"]("AB-100"))["in_stock"])),
    Task("di-rate", "distractor_tools",
         "How many ORC does one TSL buy?",
         ("exchange_rate", *DISTRACTORS),
         {"kind": "numeric", "values": [0.025], "tol": 0.0005},
         lambda t: t["exchange_rate"]("TSL", "ORC")),
    Task("di-name", "distractor_tools",
         "What is the full name of employee E-300?",
         ("get_employee", *DISTRACTORS),
         {"kind": "tokens", "all_of": ["chidi", "okafor"]},
         lambda t: _j(t["get_employee"]("E-300"))["name"]),
    Task("di-days", "distractor_tools",
         "How many days are there from 2023-11-20 to 2024-02-10?",
         ("days_between", *DISTRACTORS),
         {"kind": "numeric", "values": [82], "tol": 0},
         lambda t: t["days_between"]("2023-11-20", "2024-02-10")),
    # unanswerable: the tools report the record absent; inventing one fails.
    Task("un-country", "unanswerable",
         "What is the capital of Zandoria? Use the tools.",
         ("lookup_country",), _NF,
         lambda t: t["lookup_country"]("Zandoria")),
    Task("un-employee", "unanswerable",
         "Which department does employee E-999 work in?",
         ("get_employee", "get_department"),
         {**_NF, "none_of": ["logistics", "research", "operations"]},
         lambda t: t["get_employee"]("E-999")),
    Task("un-sku", "unanswerable",
         "How many units of SKU ZZ-999 are in stock?",
         ("check_inventory",), _NF,
         lambda t: t["check_inventory"]("ZZ-999")),
    Task("un-founder", "unanswerable",
         "Who founded the Veloria Glassworks company? Search the archive.",
         ("archive_search",), _NF,
         lambda t: t["archive_search"]("Veloria Glassworks founder")),
    Task("un-station", "unanswerable",
         "What is the temperature at station ST-42?",
         ("station_reading",), {**_NF, "none_of": ["18.4", "22.1"]},
         lambda t: t["station_reading"]("ST-42")),
)  # fmt: skip


def _flaky_twice(tools: dict[str, Callable[..., Any]]) -> str:
    """Reference for er-flaky: the first read fails, the retry succeeds."""
    try:
        tools["station_reading"]("ST-7")
    except RuntimeError:
        pass
    return str(_j(tools["station_reading"]("ST-7"))["celsius"])


_ISO_IN_TEXT_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def normalize(text: str) -> str:
    """Casefold, drop punctuation, collapse whitespace."""
    cleaned = re.sub(r"[^\w\s]", "", str(text).casefold())
    return " ".join(cleaned.split())


def numbers_in(text: str) -> list[float]:
    """Every number in *text* (thousands commas allowed; ISO dates skipped)."""
    out: list[float] = []
    for token in _NUMBER_RE.findall(_ISO_IN_TEXT_RE.sub(" ", str(text))):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return out


def grade(grader: dict[str, Any], answer: str) -> bool:
    """Whether *answer* satisfies *grader*; never calls a model.

    Kinds: ``exact`` (normalised equality), ``numeric`` (every value in
    ``values`` is within ``tol`` of some number in the answer; task prompts
    never contain their expected numbers), ``tokens`` (every normalised word
    present), ``contains`` (some ``any_of`` phrase present). ``none_of``
    applies to every kind: any listed phrase present fails the answer.
    """
    norm = normalize(answer)
    padded = f" {norm} "
    if any(f" {normalize(p)} " in padded for p in grader.get("none_of", ())):
        return False
    kind = grader["kind"]
    if kind == "exact":
        return norm == normalize(grader["value"])
    if kind == "numeric":
        found = numbers_in(answer)
        tol = float(grader["tol"])
        return all(
            any(abs(n - float(v)) <= tol + 1e-9 for n in found)
            for v in grader["values"]
        )
    if kind == "tokens":
        words = set(norm.split())
        return all(normalize(tok) in words for tok in grader["all_of"])
    if kind == "contains":
        return any(f" {normalize(p)} " in padded for p in grader["any_of"])
    raise ValueError(f"unknown grader kind {kind!r}")


# --- END TASKS ---------------------------------------------------------------


def tasks_sha256() -> str:
    """sha256 of the fixture region of this file (tools, tasks, graders).

    Independent of ``ARMS``, ``LIMITS`` and the run machinery, which sit
    outside the markers; so registering an arm never changes it.
    """
    text = Path(__file__).read_text(encoding="utf-8")
    start = text.index("# --- BEGIN TASKS")
    end = text.index("# --- END TASKS")
    return hashlib.sha256(text[start:end].encode("utf-8")).hexdigest()


def tasks_by_id() -> dict[str, Task]:
    return {task.id: task for task in TASKS}


def prompt_sha256() -> str:
    """sha256 of every task prompt in bench order (what the bench sends)."""
    payload = "\x00".join(f"{t.id}\n{t.prompt}" for t in TASKS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tool_surface(arm_name: str) -> dict[str, Any]:
    return {
        "arm_factory": arm_name,
        "limits": dict(LIMITS),
        "tools_per_task": {t.id: sorted(t.tools) for t in TASKS},
    }


# --- Arms --------------------------------------------------------------------


def _agent_config(model: str) -> Any:
    from fsm_llm.agents import AgentConfig

    return AgentConfig(
        model=model,
        max_iterations=LIMITS["max_iterations"],
        timeout_seconds=LIMITS["timeout_seconds"],
        temperature=LIMITS["temperature"],
        max_tokens=LIMITS["max_tokens"],
    )


def build_registry(tools: dict[str, Callable[..., Any]]) -> Any:
    """A ToolRegistry holding *tools*; schemas inferred from the type hints."""
    from fsm_llm.agents import ToolRegistry

    registry = ToolRegistry()
    for name, fn in tools.items():
        registry.register_function(fn, name=name)
    return registry


def _fsm_advance_arm(
    tools: dict[str, Callable[..., Any]], model: str, llm_interface: Any
) -> Any:
    """create_agent("react", tools, config=...): ReactAgent on core steps."""
    # B0's `legacy` construction, run on the code that drives the FSM through
    # core's advance/run_until_terminal; the docstring is the manifest's
    # `arm.factory` text. `llm_interface` is the trial's metered interface.
    from fsm_llm.agents import create_agent

    return create_agent(
        "react",
        build_registry(tools),
        config=_agent_config(model),
        llm_interface=llm_interface,
    )


#: Arm label -> factory ``(tools, model, llm_interface) -> agent with
#: .run(task)``; the factory must hand ``llm_interface`` to the agent, or the
#: row counts no call (``meter_parity`` shows it). Recorded rows stay
#: recountable after an arm is removed: ``report`` reads arm labels from the
#: ``rows_<arm>.jsonl`` file names, never from this table. A label names the
#: code it ran: changed code gets a NEW label, never an old one.
ARMS: dict[str, Callable[[dict[str, Callable[..., Any]], str, Any], Any]] = {
    "fsm_advance": _fsm_advance_arm,
}


# --- Call meters -------------------------------------------------------------


def metered_interface(model: str) -> Any:
    """The trial's ``LiteLLMInterface``: the meter of ``wrapper_version`` "2".

    Built from the same ``AgentConfig`` the arm uses (model, temperature,
    max_tokens; every other setting the default ``API`` would use), so
    injecting it changes no request. One per trial: its ``usage()`` is the
    trial's count, no delta or reset needed.
    """
    # DECISION plan-2026-10-01T093600-944e2692/D-004: wrapper_version "2" reads
    # core's per-instance counters off an interface the bench builds and
    # injects. Do NOT meter a new block by patching litellm bindings (that is
    # "1", kept only for meter_parity), and do NOT share one interface across
    # trials (its count would no longer be one trial's). See D-004.
    from fsm_llm import LiteLLMInterface

    config = _agent_config(model)
    return LiteLLMInterface(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )


def llm_request_settings(model: str) -> dict[str, Any]:
    """The per-request transport settings of the trial interface (manifest
    ``llm_request``): ``timeout`` (seconds, ``None`` = litellm's default),
    ``retries`` (0 = the SDK default) and the names of any extra provider
    kwargs (a ``seed`` would show here)."""
    llm = metered_interface(model)
    return {
        "timeout": llm.timeout,
        "retries": llm.retries,
        "extra_kwargs": sorted(llm.kwargs),
    }


# --- Request disclosure ------------------------------------------------------


def sha256_json(obj: Any) -> str:
    """sha256 of ``json.dumps(obj)`` with key order kept, never sorted: a
    schema whose keys were reordered is a different digest."""
    return hashlib.sha256(json.dumps(obj).encode("utf-8")).hexdigest()


def _request_recorder(model: str) -> Any:
    """An ``LLMInterface`` that records every request and refuses it.

    Each method appends ``(method_name, request)`` to ``requests`` and raises
    ``LLMResponseError``, so an agent run on it ends after its first refused
    request(s) without any provider call. ``model`` is the arm's model, so the
    agent sees the interface it would be given in a trial.
    """
    from fsm_llm import LLMInterface, LLMResponseError

    class _RequestRecorder(LLMInterface):
        def __init__(self) -> None:
            self.model = model
            self.requests: list[tuple[str, Any]] = []

        def _refuse(self, kind: str, request: Any) -> Any:
            self.requests.append((kind, request))
            raise LLMResponseError("agents_bench request probe: no provider")

        def generate_response(self, request: Any) -> Any:
            return self._refuse("generate_response", request)

        def generate_response_stream(self, request: Any) -> Any:
            return self._refuse("generate_response_stream", request)

        def extract_field(self, request: Any) -> Any:
            return self._refuse("extract_field", request)

        def extract_bulk_data(self, request: Any) -> Any:
            return self._refuse("extract_bulk_data", request)

        def complete(self, request: Any) -> Any:
            return self._refuse("complete", request)

    return _RequestRecorder()


def _request_digest(kind: str, request: Any) -> dict[str, Any]:
    """``kind`` plus digests of a request's system text(s) and ``tools``."""
    messages = getattr(request, "messages", None)
    if messages is not None:
        system = [m.get("content") for m in messages if m.get("role") == "system"]
    else:
        system = [request.system_prompt]
    tools = getattr(request, "tools", None)
    return {
        "kind": kind,
        "system_sha256": sha256_json(system),
        "tools_sha256": None if tools is None else sha256_json(tools),
    }


def request_disclosure(arm_name: str, model: str) -> dict[str, Any]:
    """What *arm_name*'s agent hands its interface, per task, with no provider.

    Interface contract (callers: ``build_manifest``; the tests):
        - Builds the arm for every task on a ``_request_recorder`` and runs
          it; the recorder refuses every request, so nothing is sent and the
          run ends in an error, which is expected and recorded by name.
        - Returns ``{"agent_class", "run_cap", "tool_schemas_sha256",
          "first_request"}``: the agent's ``module.qualname``; its core step
          ceiling for ``LIMITS["max_iterations"]`` as ``{"max_steps",
          "formula", "max_seconds"}`` (``None`` for an agent without
          ``_step_ceiling``); per task the ``sha256_json`` of the registry's
          ``get_json_schemas()`` (the bytes a native ``tools=`` carries, and
          the one schema source a prompt-mode arm renders); per task the
          first request's ``{"kind", "system_sha256", "tools_sha256",
          "run_error"}`` (``None`` when the run sent no request).
        - A prompt-mode arm's system prompt carries today's date, so its
          ``system_sha256`` is specific to the day it was computed.
        - Raises whatever the arm factory raises (a broken arm fails the
          registration, not a trial). Writes nothing.
    """
    tool_schemas: dict[str, str] = {}
    first_request: dict[str, dict[str, Any] | None] = {}
    agent: Any = None
    for task in TASKS:
        tools = make_tools()
        selected = {name: tools[name] for name in task.tools}
        tool_schemas[task.id] = sha256_json(build_registry(selected).get_json_schemas())
        recorder = _request_recorder(model)
        agent = ARMS[arm_name](selected, model, recorder)
        run_error = None
        try:
            agent.run(task.prompt)
        except Exception as exc:  # every request is refused: the run must fail
            run_error = type(exc).__name__
        if not recorder.requests:
            first_request[task.id] = None
            continue
        digest = _request_digest(*recorder.requests[0])
        first_request[task.id] = {**digest, "run_error": run_error}
    step_ceiling = getattr(agent, "_step_ceiling", None)
    run_cap = None
    if callable(step_ceiling):
        max_steps, formula = step_ceiling(LIMITS["max_iterations"])
        run_cap = {
            "max_steps": max_steps,
            "formula": formula,
            "max_seconds": LIMITS["timeout_seconds"],
        }
    return {
        "agent_class": f"{type(agent).__module__}.{type(agent).__qualname__}",
        "run_cap": run_cap,
        "tool_schemas_sha256": tool_schemas,
        "first_request": first_request,
    }


def usage_fields(usage: Any) -> dict[str, int]:
    """The row's meter fields from a core ``LLMUsage`` snapshot.

    Same keys and meanings as ``CallMeter.snapshot`` (the "1" rows), so
    ``report`` recounts "1" and "2" rows alike.
    """
    return {
        "llm_calls": usage.calls,
        "llm_errors": usage.errors,
        "usage_missing": usage.usage_missing,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


class CallMeter:
    """``wrapper_version`` "1": counts completion calls and token usage
    between two ``reset`` calls by wrapping litellm bindings.

    B0 and B1 rows were counted by it. New blocks are metered by "2"
    (``metered_interface``); this class is kept only as the independent
    reference ``meter_parity`` checks "2" against. Sequential use only: one
    trial at a time, so the counters are unambiguous.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.errors = 0
        self.usage_missing = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def record(self, response: Any) -> None:
        self.calls += 1
        usage = _usage_of(response)
        if usage is None:
            self.usage_missing += 1
            return
        prompt, completion, total = usage
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total

    def snapshot(self) -> dict[str, int]:
        return {
            "llm_calls": self.calls,
            "llm_errors": self.errors,
            "usage_missing": self.usage_missing,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _as_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _usage_of(response: Any) -> tuple[int, int, int] | None:
    """(prompt, completion, total) tokens, read defensively; None if absent.

    litellm renames response fields across its supported range, and streamed
    responses carry no usage at all, so every read tolerates absence.
    """
    usage = _field(response, "usage")
    if usage is None:
        return None
    prompt = _as_int(_field(usage, "prompt_tokens"))
    completion = _as_int(_field(usage, "completion_tokens"))
    total = _as_int(_field(usage, "total_tokens")) or prompt + completion
    return prompt, completion, total


def _metered(fn: Callable[..., Any], meter: CallMeter) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(fn):

        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                response = await fn(*args, **kwargs)
            except Exception:
                meter.calls += 1
                meter.errors += 1
                raise
            meter.record(response)
            return response

        async_wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
        return async_wrapper

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            response = fn(*args, **kwargs)
        except Exception:
            meter.calls += 1
            meter.errors += 1
            raise
        meter.record(response)
        return response

    wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    return wrapper


def install_meter(
    meter: CallMeter, targets: list[tuple[Any, str]]
) -> Callable[[], None]:
    """Wrap each ``getattr(module, attr)`` with *meter*; returns the undo.

    Interface contract (callers: ``run_block``, the tests with fake modules):
        - Each binding is wrapped separately, sync or async by the original's
          kind; a response's usage is read defensively (``_usage_of``).
        - A call that raises is counted as a call and an error, then re-raised.
        - The returned callable restores every original binding; call it in
          a ``finally``.
    """
    originals = [(module, attr, getattr(module, attr)) for module, attr in targets]
    for module, attr, fn in originals:
        setattr(module, attr, _metered(fn, meter))

    def restore() -> None:
        for module, attr, fn in originals:
            setattr(module, attr, fn)

    return restore


def _completion_targets() -> list[tuple[Any, str]]:
    """Every binding an agent can reach litellm's completion through (the
    "1" meter's targets).

    ``fsm_llm.llm`` binds ``completion`` by name at import (every
    ``LiteLLMInterface`` request, the classifier's included, is one count
    there); a caller still on its own path (native_fc until it runs on core)
    looks ``litellm.completion`` up at call time. Loaded here, BEFORE any arm
    imports ``fsm_llm.agents``.
    """
    import litellm

    import fsm_llm.llm as llm

    return [
        (litellm, "completion"),
        (litellm, "acompletion"),
        (llm, "completion"),
    ]


# --- Run ---------------------------------------------------------------------


def _model_tag(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def build_manifest(
    *, bench_id: str, block: str, arm_name: str, trials: int, model: str
) -> dict[str, Any]:
    """The pre-registration record, written BEFORE row 1.

    Besides the comparability fields it discloses what the arm's requests
    carry (``DISCLOSURE_FIELDS``): transport settings, agent class, run cap,
    tool-schema and first-request digests, computed offline with no provider
    call (``request_disclosure``).
    """
    # DECISION plan-2026-10-01T093600-944e2692/D-036: a manifest records what
    # its requests carried (timeout, tool-schema bytes, system prompt, agent
    # class, run cap), computed by running the arm on a refusing recorder.
    # Do NOT derive these from constants or source text (they drift from
    # what is sent: the step-14 schema diff hid behind a key-sorted
    # comparison), do NOT sort keys before hashing, and do NOT edit a
    # recorded manifest to add them (report prints "not recorded"). D-036.
    fixture = tasks_sha256()
    disclosure = request_disclosure(arm_name, model)
    return {
        "bench_id": bench_id,
        "block": block,
        "n_tasks": len(TASKS),
        "trials": trials,
        "n_preregistered": len(TASKS) * trials,
        "order": "trial-major: every task once per trial, tasks in file order",
        "seed": None,
        "model": model,
        "created_at": hb._utc_now(),
        "prompt_bytes_sha256": prompt_sha256(),
        "tool_surface": tool_surface(arm_name),
        "fixture_hash": fixture,
        "tasks_sha256": fixture,
        "model_digest": hb._model_digest(_model_tag(model)),
        "arm": {"name": arm_name, "factory": (ARMS[arm_name].__doc__ or "").strip()},
        "git_commit": hb._git_commit(),
        "temperature": LIMITS["temperature"],
        "limits": dict(LIMITS),
        "wrapper_version": WRAPPER_VERSION,
        "llm_request": llm_request_settings(model),
        **disclosure,
    }


def _arm_paths(bdir: Path, arm: str) -> tuple[Path, Path, Path]:
    names = (f"manifest_{arm}.json", f"rows_{arm}.jsonl", f"summary_{arm}.json")
    return (bdir / names[0], bdir / names[1], bdir / names[2])


#: Manifest keys a pre-registered block must still match when ``run`` starts.
#: ``first_request`` is not one: a prompt-mode arm's system prompt carries
#: the date, so it moves overnight without any code change.
_PINNED_AT_RUN = (
    "tasks_sha256",
    "trials",
    "model",
    "limits",
    "wrapper_version",
    "llm_request",
    "agent_class",
    "run_cap",
    "tool_schemas_sha256",
)


def register_block(
    bench_id: str, block: str, arm_name: str, trials: int = TRIALS, model: str = MODEL
) -> Path:
    """Write the manifest only (to commit it before any row exists)."""
    if arm_name not in ARMS:
        raise BenchDataError(
            f"unknown arm {arm_name!r}; expected one of {sorted(ARMS)}"
        )
    manifest_path, rows_path, summary_path = _arm_paths(
        BENCH_DATA / bench_id / block, arm_name
    )
    for path in (manifest_path, rows_path, summary_path):
        if path.exists():
            raise BenchDataError(f"{path} exists -- a block is registered ONCE")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        bench_id=bench_id, block=block, arm_name=arm_name, trials=trials, model=model
    )
    hb._write_json(manifest_path, manifest)
    return manifest_path


def _checked_manifest(
    manifest_path: Path,
    *,
    bench_id: str,
    block: str,
    arm_name: str,
    trials: int,
    model: str,
) -> dict[str, Any]:
    """The existing pre-registered manifest, refused if it no longer matches."""
    current = build_manifest(
        bench_id=bench_id, block=block, arm_name=arm_name, trials=trials, model=model
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    drift = [k for k in _PINNED_AT_RUN if manifest.get(k) != current[k]]
    if manifest.get("model_digest", {}).get("digest") != current["model_digest"].get(
        "digest"
    ):
        drift.append("model_digest")
    if drift:
        raise BenchDataError(
            f"{manifest_path} was registered with different {drift}; "
            "pre-register a NEW block"
        )
    return manifest


def _trial_row(task: Task, trial: int, arm_name: str, model: str) -> dict[str, Any]:
    """Run one fresh agent on one task; any exception is an incorrect row.

    The meter fields come from the trial's own ``metered_interface``.
    """
    tools = make_tools()
    selected = {name: tools[name] for name in task.tools}
    llm_interface = metered_interface(model)
    answer, error = "", None
    success, stop_reason = False, None
    iterations = tool_calls = 0
    tools_used: list[str] = []
    started = time.perf_counter()
    try:
        agent = ARMS[arm_name](selected, model, llm_interface)
        result = agent.run(task.prompt)
        answer = str(getattr(result, "answer", "") or "")
        success = bool(getattr(result, "success", False))
        stop_reason = getattr(result, "stop_reason", None)
        trace = getattr(result, "trace", None)
        calls = list(getattr(trace, "tool_calls", []) or [])
        tool_calls = len(calls)
        tools_used = sorted({getattr(c, "tool_name", str(c)) for c in calls})
        iterations = int(getattr(trace, "total_iterations", 0) or 0)
    except Exception as exc:  # a crashed or timed-out trial is a failed row
        name = type(exc).__name__
        error = f"{name}: {exc}"[:ERROR_CHARS]
        stop_reason = "timeout" if "Timeout" in name else "error"
    latency = time.perf_counter() - started
    return {
        "task_id": task.id,
        "category": task.category,
        "arm": arm_name,
        "trial": trial,
        "correct": error is None and grade(task.grader, answer),
        "success": success,
        "stop_reason": stop_reason,
        "iterations": iterations,
        "tool_calls": tool_calls,
        "tools_used": tools_used,
        **usage_fields(llm_interface.usage()),
        "latency_s": round(latency, 3),
        "error": error,
        "answer": answer[:ANSWER_CHARS],
    }


def run_block(
    bench_id: str, block: str, arm_name: str, trials: int = TRIALS, model: str = MODEL
) -> dict[str, Any]:
    """Manifest (or its check), every task x trial in trial-major order,
    summary. An abort keeps its rows and is summarised ``status: aborted``."""
    if arm_name not in ARMS:
        raise BenchDataError(
            f"unknown arm {arm_name!r}; expected one of {sorted(ARMS)}"
        )
    bdir = BENCH_DATA / bench_id / block
    manifest_path, rows_path, summary_path = _arm_paths(bdir, arm_name)
    if rows_path.exists() or summary_path.exists():
        # DECISION plan-2026-07-22T114536-879d04a0/D-002
        # Do NOT add a --force/overwrite flag: a block runs ONCE at its fixed
        # n; a new question needs a NEW pre-registered block. See D-002.
        raise BenchDataError(
            f"{rows_path} exists -- a block is run ONCE (D-002); no re-sampling"
        )
    if manifest_path.exists():
        manifest = _checked_manifest(
            manifest_path,
            bench_id=bench_id,
            block=block,
            arm_name=arm_name,
            trials=trials,
            model=model,
        )
    else:
        bdir.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(
            bench_id=bench_id,
            block=block,
            arm_name=arm_name,
            trials=trials,
            model=model,
        )
        hb._write_json(manifest_path, manifest)
    print(f"pre-registered {manifest_path} (digest {manifest['model_digest']})")
    started, status = hb._utc_now(), "aborted"
    try:
        total = len(TASKS) * trials
        done = 0
        for trial in range(1, trials + 1):
            for task in TASKS:
                row = _trial_row(task, trial, arm_name, model)
                row.update(bench_id=bench_id, block=block, ts=hb._utc_now())
                append_row(rows_path, row)
                done += 1
                print(
                    f"  {block}/{arm_name} {done}/{total} {task.id} t{trial}: "
                    f"correct={row['correct']} success={row['success']} "
                    f"stop={row['stop_reason']} calls={row['llm_calls']} "
                    f"{row['latency_s']}s",
                    flush=True,
                )
        status = "complete"
    finally:
        summary = write_summary(bdir, arm_name, status=status, started_at=started)
        print(f"wrote {summary_path} (status: {status})")
    return summary


def meter_parity(
    task_ids: list[str], arm_name: str, model: str = MODEL
) -> list[dict[str, Any]]:
    """Run each task once with both meters; their counts side by side.

    Interface contract (callers: the offline parity test, the live smoke
    before a "2" block is registered):
        - Each task runs once through ``_trial_row`` (the "2" meter: the
          trial's injected interface) while the "1" ``CallMeter`` wraps
          ``_completion_targets()``; the bindings are restored on return.
        - Returns, in the given order, ``{"task_id", "v1", "v2", "equal",
          "error"}``: ``v1``/``v2`` hold the six row meter fields, ``equal``
          is True when all six agree, ``error`` is the trial's row error
          (``None`` when it ran). A call the arm sends outside the injected
          interface shows as ``v1`` above ``v2``.
        - Writes no rows and no manifest. Raises ``BenchDataError`` for an
          unknown arm or task id, before any task runs.
    """
    if arm_name not in ARMS:
        raise BenchDataError(
            f"unknown arm {arm_name!r}; expected one of {sorted(ARMS)}"
        )
    by_id = tasks_by_id()
    unknown = [task_id for task_id in task_ids if task_id not in by_id]
    if unknown:
        raise BenchDataError(f"unknown task ids {unknown}")
    meter = CallMeter()
    restore = install_meter(meter, _completion_targets())
    results: list[dict[str, Any]] = []
    try:
        for task_id in task_ids:
            meter.reset()
            row = _trial_row(by_id[task_id], 1, arm_name, model)
            v1 = meter.snapshot()
            v2 = {key: row[key] for key in v1}
            results.append(
                {
                    "task_id": task_id,
                    "v1": v1,
                    "v2": v2,
                    "equal": v1 == v2,
                    "error": row["error"],
                }
            )
    finally:
        restore()
    return results


# --- Metrics -----------------------------------------------------------------


def percentile_nearest_rank(values: list[float], q: float) -> float | None:
    """The nearest-rank percentile: the ceil(q/100 * n)-th smallest value."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def _median(values: list[float]) -> float | None:
    return round(float(statistics.median(values)), 4) if values else None


def _wilson(k: int, n: int) -> list[float]:
    lo, hi = wilson_ci(k, n)
    return [round(lo, 4), round(hi, 4)]


def compute_metrics(rows: list[dict[str, Any]], trials: int) -> dict[str, Any]:
    """Every summary number, from raw rows alone (no task table, no arms).

    pass@1 (primary) is first-trial correctness per task: one independent
    unit per task. ``mean_over_trials`` counts every row (trials of a task
    are correlated: indicative only). ``pass_hat_k`` counts tasks with at
    least *trials* rows, all correct.
    """
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row.get("task_id")), []).append(row)
    firsts = {
        tid: min(rs, key=lambda r: int(r.get("trial", 0)))
        for tid, rs in by_task.items()
    }
    n_tasks, n_rows = len(by_task), len(rows)
    k_first = sum(bool(r.get("correct")) for r in firsts.values())
    k_rows = sum(bool(r.get("correct")) for r in rows)
    k_hat = sum(
        1
        for rs in by_task.values()
        if len(rs) >= trials and all(bool(r.get("correct")) for r in rs)
    )
    cross = Counter(
        ("success" if r.get("success") else "fail")
        + "_"
        + ("correct" if r.get("correct") else "incorrect")
        for r in rows
    )
    per_category: dict[str, dict[str, int]] = {}
    for tid, rs in sorted(by_task.items()):
        cat = str(rs[0].get("category"))
        slot = per_category.setdefault(
            cat, {"tasks": 0, "first_trial_correct": 0, "rows": 0, "rows_correct": 0,
                  "pass_hat_k": 0},
        )  # fmt: skip
        slot["tasks"] += 1
        slot["first_trial_correct"] += bool(firsts[tid].get("correct"))
        slot["rows"] += len(rs)
        slot["rows_correct"] += sum(bool(r.get("correct")) for r in rs)
        slot["pass_hat_k"] += len(rs) >= trials and all(
            bool(r.get("correct")) for r in rs
        )

    def col(key: str) -> list[float]:
        return [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]

    latency = col("latency_s")
    p50 = percentile_nearest_rank(latency, 50)
    p95 = percentile_nearest_rank(latency, 95)
    return {
        "n_rows": n_rows,
        "n_tasks": n_tasks,
        "trials": trials,
        "k_pass1_first_trial": k_first,
        "wilson_pass1_first_trial": _wilson(k_first, n_tasks),
        "k_correct_rows": k_rows,
        "wilson_correct_rows": _wilson(k_rows, n_rows),
        "k_pass_hat_k": k_hat,
        "wilson_pass_hat_k": _wilson(k_hat, n_tasks),
        "k_success_rows": sum(bool(r.get("success")) for r in rows),
        "success_vs_correct": {
            key: cross.get(key, 0)
            for key in (
                "success_correct",
                "success_incorrect",
                "fail_correct",
                "fail_incorrect",
            )
        },
        "k_error_rows": sum(1 for r in rows if r.get("error")),
        "llm_calls_mean": _mean(col("llm_calls")),
        "llm_calls_median": _median(col("llm_calls")),
        "total_tokens_mean": _mean(col("total_tokens")),
        "total_tokens_median": _median(col("total_tokens")),
        "prompt_tokens_mean": _mean(col("prompt_tokens")),
        "completion_tokens_mean": _mean(col("completion_tokens")),
        "tool_calls_mean": _mean(col("tool_calls")),
        "latency_p50_s": None if p50 is None else round(p50, 3),
        "latency_p95_s": None if p95 is None else round(p95, 3),
        "stop_reasons": dict(
            sorted(Counter(str(r.get("stop_reason")) for r in rows).items())
        ),
        "per_category": per_category,
    }


def write_summary(
    bdir: Path, arm_name: str, *, status: str, started_at: str | None = None
) -> dict[str, Any]:
    """Summary from rows; REFUSES without a complete manifest (not evidence)."""
    manifest_path, rows_path, summary_path = _arm_paths(bdir, arm_name)
    if not manifest_path.is_file():
        raise BenchDataError(
            f"refusing {summary_path.name}: no manifest -- not evidence"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = [f for f in MANIFEST_FIELDS if f not in manifest]
    if missing:
        raise BenchDataError(f"refusing {summary_path.name}: manifest lacks {missing}")
    summary = {
        "bench_id": manifest["bench_id"],
        "block": manifest["block"],
        "arm": arm_name,
        "status": status,
        "started_at": started_at,
        "finished_at": hb._utc_now(),
        "manifest": manifest_path.name,
        "metrics": compute_metrics(read_rows(rows_path), int(manifest["trials"])),
    }
    hb._write_json(summary_path, summary)
    return summary


# --- Report ------------------------------------------------------------------


def _fmt_k(k: int, n: int, ci: list[float]) -> str:
    rate = f"{k / n:.1%}" if n else "n/a"
    return f"{k}/{n} ({rate}) wilson95=[{ci[0]:.3f}, {ci[1]:.3f}]"


def _print_metrics(m: dict[str, Any]) -> None:
    n_t, n_r = m["n_tasks"], m["n_rows"]
    print(
        f"  pass@1 first trial: "
        f"{_fmt_k(m['k_pass1_first_trial'], n_t, m['wilson_pass1_first_trial'])}"
    )
    print(
        f"  mean over trials:   "
        f"{_fmt_k(m['k_correct_rows'], n_r, m['wilson_correct_rows'])}"
    )
    print(
        f"  pass^{m['trials']}:             "
        f"{_fmt_k(m['k_pass_hat_k'], n_t, m['wilson_pass_hat_k'])}"
    )
    cross = m["success_vs_correct"]
    print(
        "  success x correct:  "
        + ", ".join(f"{key}={val}" for key, val in cross.items())
    )
    print(
        f"  llm calls mean/median: {m['llm_calls_mean']}/{m['llm_calls_median']}; "
        f"tokens mean/median: {m['total_tokens_mean']}/{m['total_tokens_median']}; "
        f"tool calls mean: {m['tool_calls_mean']}"
    )
    print(
        f"  latency p50/p95 (nearest rank): "
        f"{m['latency_p50_s']}s/{m['latency_p95_s']}s; errors: {m['k_error_rows']}"
    )
    print(f"  stop reasons: {m['stop_reasons']}")
    print("  category          tasks  pass@1  rows_correct  pass^k")
    for cat, slot in m["per_category"].items():
        print(
            f"  {cat:<17} {slot['tasks']:>5}  {slot['first_trial_correct']:>6}  "
            f"{slot['rows_correct']:>5}/{slot['rows']:<6}  {slot['pass_hat_k']:>6}"
        )


def _read_manifest(bdir: Path, arm: str) -> dict[str, Any] | None:
    path = bdir / f"manifest_{arm}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(
    side: str, recomputed: dict[tuple[str, str], dict[str, Any]]
) -> tuple[str, str]:
    """``block/arm`` or a bare ``arm`` that exists in exactly one block."""
    if "/" in side:
        block, arm = side.split("/", 1)
        if (block, arm) not in recomputed:
            raise BenchDataError(
                f"--pair side {side!r}: no rows_{arm}.jsonl in {block}"
            )
        return block, arm
    hits = [key for key in recomputed if key[1] == side]
    if len(hits) != 1:
        raise BenchDataError(
            f"--pair side {side!r} matches {len(hits)} blocks; write BLOCK/ARM"
        )
    return hits[0]


#: Marks a manifest key one side of a pair lacks (distinct from a JSON null).
_ABSENT = object()


def _show(value: Any) -> str:
    return NOT_RECORDED if value is _ABSENT else json.dumps(value)


def manifest_differences(
    a: dict[str, Any], b: dict[str, Any], prefix: str = ""
) -> list[str]:
    """One ``"key: A vs B"`` line per manifest leaf that differs, A first.

    Interface contract (callers: ``report`` for ``--pair``; the tests):
        - Walks the union of both manifests' keys in sorted order, recursing
          into keys whose values are dicts on both sides (dotted paths, e.g.
          ``model_digest.digest``); every other differing value is one line
          with both values as JSON.
        - A key one side lacks prints as ``not recorded`` (recorded B0/B1
          manifests predate ``DISCLOSURE_FIELDS``); nothing is skipped or
          truncated. Equal manifests give ``[]``. Never raises.
    """
    lines: list[str] = []
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key, _ABSENT), b.get(key, _ABSENT)
        if va == vb:
            continue
        path = f"{prefix}{key}"
        if isinstance(va, dict) and isinstance(vb, dict):
            lines.extend(manifest_differences(va, vb, f"{path}."))
        else:
            lines.append(f"{path}: {_show(va)} vs {_show(vb)}")
    return lines


def report(
    bench_id: str, blocks: list[str] | None = None, pairs: list[str] | None = None
) -> int:
    """Recount every block arm from RAW rows; cross-check committed summaries;
    Fisher per requested ``A:B`` pair. Returns 0 ok, 1 on mismatch/refusal."""
    bench_dir = BENCH_DATA / bench_id
    if not bench_dir.is_dir():
        raise BenchDataError(f"no such bench: {bench_dir}")
    names = blocks or sorted(d.name for d in bench_dir.iterdir() if d.is_dir())
    ok = True
    recomputed: dict[tuple[str, str], dict[str, Any]] = {}
    for block in names:
        bdir = bench_dir / block
        if not bdir.is_dir():
            raise BenchDataError(f"no such block: {bdir}")
        for rows_path in sorted(bdir.glob("rows_*.jsonl")):
            arm = rows_path.stem.split("_", 1)[1]  # from the file, not ARMS
            rows = read_rows(rows_path)
            manifest = _read_manifest(bdir, arm)
            trials = int(
                manifest["trials"]
                if manifest and "trials" in manifest
                else max((int(r.get("trial", 1)) for r in rows), default=1)
            )
            metrics = compute_metrics(rows, trials)
            recomputed[(block, arm)] = metrics
            print(f"{bench_id} {block} [{arm}] rows={metrics['n_rows']}")
            _print_metrics(metrics)
            summary_path = bdir / f"summary_{arm}.json"
            if not summary_path.is_file():
                print("  (no committed summary to cross-check)")
                continue
            committed = json.loads(summary_path.read_text(encoding="utf-8"))
            want = committed.get("metrics", {})
            for key in sorted(set(metrics) | set(want)):
                if want.get(key) != metrics.get(key):
                    ok = False
                    print(
                        f"  MISMATCH {key}: committed {want.get(key)} "
                        f"!= recount {metrics.get(key)}"
                    )
    for pair in pairs or []:
        if ":" not in pair:
            raise BenchDataError(f"--pair {pair!r}: expected A:B")
        left, right = pair.split(":", 1)
        a, b = _resolve(left, recomputed), _resolve(right, recomputed)
        manifest_a = _read_manifest(bench_dir / a[0], a[1]) or {}
        manifest_b = _read_manifest(bench_dir / b[0], b[1]) or {}
        # Every disclosed difference comes before any number (D-036).
        diffs = manifest_differences(manifest_a, manifest_b)
        print(
            f"Manifest differences, {a[0]}/{a[1]} vs {b[0]}/{b[1]}: "
            f"{len(diffs)} field(s)"
        )
        for line in diffs:
            print(f"  {line}")
        da = manifest_a.get("model_digest", {})
        db = manifest_b.get("model_digest", {})
        if not da or not db or da.get("digest") != db.get("digest"):
            ok = False
            print(
                f"REFUSING {a[0]}/{a[1]} vs {b[0]}/{b[1]}: model digests differ "
                f"or a manifest is absent ({da.get('digest')} vs {db.get('digest')})"
            )
            continue
        ma, mb = recomputed[a], recomputed[b]
        print(f"Fisher two-sided, {a[0]}/{a[1]} vs {b[0]}/{b[1]}:")
        for key, label in (
            ("k_pass1_first_trial", "pass@1 first trial"),
            ("k_pass_hat_k", "pass^k"),
        ):
            ka, kb, na, nb = ma[key], mb[key], ma["n_tasks"], mb["n_tasks"]
            if na == 0 or nb == 0:
                print(f"  {label}: empty arm, no test")
                continue
            p = fisher_exact_two_sided(ka, na, kb, nb)
            print(f"  {label}: {ka}/{na} vs {kb}/{nb} p={p:.4f}")
    return 0 if ok else 1


# --- list-tasks ----------------------------------------------------------------


def verify_tasks() -> list[str]:
    """Problems with the task set; empty means every reference answer grades
    correct with fresh tools and every numeric answer is absent from its prompt."""
    problems: list[str] = []
    seen: set[str] = set()
    for task in TASKS:
        if task.id in seen:
            problems.append(f"{task.id}: duplicate id")
        seen.add(task.id)
        if task.category not in CATEGORIES:
            problems.append(f"{task.id}: unknown category {task.category!r}")
        tools = make_tools()
        missing = [name for name in task.tools if name not in tools]
        if missing:
            problems.append(f"{task.id}: unknown tools {missing}")
            continue
        answer = task.reference({name: tools[name] for name in task.tools})
        if not grade(task.grader, answer):
            problems.append(f"{task.id}: reference answer {answer!r} grades wrong")
        if task.grader["kind"] == "numeric":
            in_prompt = numbers_in(task.prompt)
            for value in task.grader["values"]:
                if any(
                    abs(n - float(value)) <= float(task.grader["tol"])
                    for n in in_prompt
                ):
                    problems.append(
                        f"{task.id}: expected {value} appears in the prompt"
                    )
    return problems


def list_tasks(verify: bool = False) -> int:
    for task in TASKS:
        tools = ",".join(t for t in task.tools if t not in DISTRACTORS)
        extra = (
            f"+{len(DISTRACTORS)} distractors" if DISTRACTORS[0] in task.tools else ""
        )
        print(
            f"{task.id:<20} {task.category:<17} {task.grader['kind']:<8} {tools}{extra}"
        )
    counts = Counter(task.category for task in TASKS)
    print(f"{len(TASKS)} tasks: " + ", ".join(f"{c}={counts[c]}" for c in CATEGORIES))
    print(f"tasks_sha256 {tasks_sha256()}")
    if not verify:
        return 0
    problems = verify_tasks()
    for problem in problems:
        print(f"PROBLEM {problem}")
    print("verify: ok" if not problems else f"verify: {len(problems)} problem(s)")
    return 0 if not problems else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (
        ("register", "write ONE block arm's manifest only (commit it before rows)"),
        ("run", "LIVE: run ONE pre-registered block arm, once"),
    ):
        cmd = sub.add_parser(name, help=text)
        cmd.add_argument("--bench-id", required=True)
        cmd.add_argument("--block", required=True, help="block name, e.g. B0")
        cmd.add_argument("--arm", required=True, choices=sorted(ARMS))
        cmd.add_argument("--trials", type=int, default=TRIALS)
        cmd.add_argument("--model", default=MODEL)
    rep = sub.add_parser("report", help="recount every block arm from raw jsonl")
    rep.add_argument("bench_id")
    rep.add_argument("--blocks", nargs="+", default=None)
    rep.add_argument(
        "--pair", action="append", default=None,
        help="A:B, each side ARM or BLOCK/ARM; Fisher on first-trial pass@1",
    )  # fmt: skip
    lst = sub.add_parser("list-tasks", help="print the task set and its sha256")
    lst.add_argument("--verify", action="store_true", help="grade reference answers")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "register":
            path = register_block(
                args.bench_id, args.block, args.arm, args.trials, args.model
            )
            print(f"registered {path}")
        elif args.command == "run":
            run_block(args.bench_id, args.block, args.arm, args.trials, args.model)
        elif args.command == "report":
            return report(args.bench_id, args.blocks, args.pair)
        else:
            return list_tasks(verify=args.verify)
        return 0
    except BenchDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
