#!/usr/bin/env python3
"""Pre-registered live bench for runtime FSM synthesis: ONE model, three scaffolds.

Does a procedure a model COMPILED into an FSM (``synth-fsm``) beat the same model
as an in-context ReAct agent (``react``) on a conversational procedure, with a
hand-written FSM (``hand-fsm``) as the check that the harness can pass at all?
Every role of every arm is ``ollama_chat/bonsai-2:27b``: NO strong/cheap split
(plan 2cefeb61, D-039, D-040, D-054).

    .venv/bin/python scripts/synthesis_bench.py register|run|report s0
    .venv/bin/python scripts/synthesis_bench.py smoke [--tasks c1 ...]

``register`` (reads ``/api/tags``) and ``report`` touch no model; ``smoke`` and
``run`` first pass the preflight (exit 3, ``BLOCKED``). Manifest before row 1, a
block runs once, ``report`` recounts raw rows. Top level is stdlib only; raw
requests and replies go to ``logs/synthesis-bench/`` (gitignored).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple


def _load_bench_common() -> Any:
    """bench_common, loaded by file path (one module object per process)."""
    if "bench_common" in sys.modules:
        return sys.modules["bench_common"]
    path = Path(__file__).resolve().with_name("bench_common.py")
    spec = importlib.util.spec_from_file_location("bench_common", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_common"] = module
    spec.loader.exec_module(module)
    return module


bc = _load_bench_common()  # wilson_ci, fisher_exact_two_sided, append_row, read_rows

ROOT = Path(__file__).resolve().parent.parent
BENCH_DATA = ROOT / "scripts" / "bench_data"
RAW_DIR = ROOT / "logs" / "synthesis-bench"  # logs/ is gitignored
BENCH_ID, MODEL_TAG = "synthesis", "bonsai-2:27b"
MODEL = f"ollama_chat/{MODEL_TAG}"
SYNTH, HAND, REACT = "synth-fsm", "hand-fsm", "react"
BLOCKS = {"s0": (SYNTH, HAND, REACT)}
TRIALS = 3  # per task; fixed at registration from the smoke
TIMEOUTS = {"compile": 600.0, "sim": 300.0, "run": 300.0}
ROLE_BY_TIMEOUT = {TIMEOUTS["compile"]: "compile", TIMEOUTS["run"]: "run"}
POLICY = {"max_turns": 10, "max_repair_rounds": 3, "sim_trials": 2, "max_sim_turns": 8, "max_simulation_seconds": 300, "max_simulation_calls": 400, "max_simulation_rounds": 2, "min_pass_rate": 0.5}  # fmt: skip
LIMITS = {"temperature": 0.5, "run_max_tokens": 1000, "max_user_turns": 14, "react_max_iterations": 12, "react_timeout_seconds": 900.0, "timeouts": TIMEOUTS, "policy": POLICY}  # fmt: skip
# DECISION plan-2026-11-10T194139-2cefeb61/D-067
# SRC_PINS is the ONE list of what the verdict depends on (this whole script, bench_common,
# the compile, simulate and run code, the judges: validator, G6b's eval driver and its stats,
# the JSON reader, the exception classes, the modules react.py imports): hashed at register,
# checked at run AND report. The engine is NOT pinned (a test names every file left out, D-075).
# Do NOT hash only the fixture region and do NOT keep a second list in the tests.
_SYNTH = "dryrun simulation traces gates policy synthesizer runner library prompts repair constants exceptions"  # fmt: skip
_AGENTS = "__init__ react base constants definitions exceptions fsm_definitions handlers hitl meta_builders meta_prompts prompts summarization tools truncation"  # fmt: skip
_CORE = "flow_analysis utilities validator eval/cases eval/config eval/stats"
SRC_PINS = (
    "scripts/synthesis_bench.py",
    "scripts/bench_common.py",
    *(f"src/{m}.py" for m in _CORE.split()),
    *(f"src/synthesis/{m}.py" for m in _SYNTH.split()),
    *(f"src/agents/{m}.py" for m in _AGENTS.split()),
)
PINNED_AT_RUN = (
    "fixture_hash",
    "trials",
    "model",
    "limits",
    "src_sha256",
    "arm",
    "wire",
)
OUTAGE_ROWS, EXIT_BLOCKED = 3, 3  # consecutive errored rows abort a block
# D-040: the preflight and the per-trial guard
DEFAULT_CONTEXT, QUIET_MINUTES = 32768, 10
PROBES, PROBE_SPACING, MAX_QUEUE_WAIT = 3, 5.0, 3.0
CHAT_LINE = re.compile(r"/api/(chat|generate)|/v1/")


# --- BEGIN FIXTURE (hashed: tasks, user, builders, grader, the rule) ----------
# fixture_hash is the sha256 of these lines; any edit is a new, incomparable
# fixture. A user rule is (regex over the last assistant text, utterance or a
# sequence of them); ``other`` is a question about the OTHER branch.
FALLBACK, OPENER = "Sorry, could you say that again?", "Hello."
PRIMARY_N, ALPHA, H_FLOOR, CEILING, MAX_INFRA = 16, 0.05, 10, 15, 2
ONE_MODEL = (
    "NOTE: compiler, scenario writer, executor, ReAct agent and hand-written arm are "
    "ONE 2-bit 27B model (bonsai-2:27b, Q2_0): there is NO strong/cheap split, the "
    "comparison is the same model under three scaffolds, so no strong-compile/cheap-run "
    "cost asymmetry is tested, and a G6b pass is partly the model agreeing with itself "
    "(a weaker signal than a separate judge); one run, one model, n = 16 tasks."
)
RULE = (
    "Unit: the first trial of each task, all required keys correct; s, r, h are the "
    "tasks passed by synth-fsm, react, hand-fsm. H1: s > r with Fisher exact two-sided "
    f"p < {ALPHA}. SUPPORTED if valid, p < {ALPHA} and s > r; REFUTED if valid, "
    f"p < {ALPHA} and r > s; NOT SUPPORTED if valid and p >= {ALPHA}. INCONCLUSIVE if "
    f"invalid: an arm block aborted or incomplete, or h < {H_FLOOR} of {PRIMARY_N}, or "
    f"r and s both at least {CEILING} of {PRIMARY_N}, or more than {MAX_INFRA} of "
    f"{PRIMARY_N} first-trial S failures have phase provider or host."
)


class Task(NamedTuple):
    id: str
    family: str  # collect | route | branch | retry
    goal: str
    gold: dict[str, str]  # return key -> gold value, compared after squash
    users: tuple[tuple[str, Any], ...]
    other: str
    spec: Any  # route: labels; branch: (labels, ask, ask); retry: (lo, hi)


_TRIP = (("business", "leisure"), "Ask for the name of the company the user works for.", "Ask for the destination the user is travelling to.")  # fmt: skip
_BUY = (("delivery", "pickup"), "Ask for the street address to deliver to.", "Ask which store the user will collect the order from.")  # fmt: skip
_TRIP_GOAL = "Find out whether the trip is business or leisure. For a business trip ask for the company name, for a leisure trip ask for the destination. Return the kind and the detail."  # fmt: skip
_BUY_GOAL = "Find out whether the order is for delivery or pickup. For delivery ask for the street address, for pickup ask for the store name. Return the method and the detail."  # fmt: skip
_RANGE = "If the number is outside that range, say so and ask again until it is valid."
# fmt: off
TASKS = tuple(Task(*row) for row in (
    ("c1", "collect", "Collect a visa applicant's passport number, travel dates and purpose of travel.",
     {"passport_no": "X1234567", "travel_dates": "2026-05-01", "purpose": "tourism"},
     ((r"passport", "My passport number is X1234567."), (r"date|when|arriv|depart", "I travel on 2026-05-01."), (r"purpose|reason|why", "The purpose is tourism.")), "", None),
    ("c2", "collect", "Collect a hotel guest's name, check-in date and room type for a booking.",
     {"guest_name": "Ada Lovelace", "checkin_date": "2026-07-14", "room_type": "double"},
     ((r"name", "My name is Ada Lovelace."), (r"check|date|arriv", "I check in on 2026-07-14."), (r"room", "I would like a double room.")), "", None),
    ("c3", "collect", "Collect a support ticket's order id, the customer's contact email and the affected product.",
     {"order_id": "ORD-5521", "contact_email": "ada@example.com", "product": "router"},
     ((r"order", "My order id is ORD-5521."), (r"email|contact|reach", "You can reach me at ada@example.com."), (r"product|item", "The product is the router.")), "", None),
    ("c4", "collect", "Collect a clinic patient's name, birth year and main symptom.",
     {"patient_name": "Grace Hopper", "birth_year": "1985", "symptom": "migraine"},
     ((r"name", "My name is Grace Hopper."), (r"born|birth|year", "I was born in 1985."), (r"symptom|problem|complain", "My main symptom is migraine.")), "", None),
    ("r1", "route", "Ask what the customer needs and route them to billing, technical, sales or other.", {"department": "billing"}, ((r"", "My invoice shows a charge twice this month."),), "", ("billing", "technical", "sales", "other")),
    ("r2", "route", "Ask what the problem is and rate its priority as urgent, normal or low.", {"priority": "urgent"}, ((r"", "The production server is down and customers cannot log in."),), "", ("urgent", "normal", "low")),
    ("r3", "route", "Ask what the customer wants and file it as refund, shipping, complaint or compliment.", {"topic": "shipping"}, ((r"", "Where is my parcel? It was due on Monday."),), "", ("refund", "shipping", "complaint", "compliment")),
    ("r4", "route", "Ask what the employee needs and send them to hr, legal, it or finance.", {"team": "it"}, ((r"", "I need a new charger for my laptop."),), "", ("hr", "legal", "it", "finance")),
    ("b1", "branch", _TRIP_GOAL, {"trip_type": "business", "detail": "Acme"},
     ((r"business|leisure", "It is a business trip."), (r"company|employer", "My company is Acme.")), r"destination|where are you|which city", _TRIP),
    ("b2", "branch", _TRIP_GOAL, {"trip_type": "leisure", "detail": "Lisbon"},
     ((r"business|leisure", "It is a leisure trip."), (r"destination|where|city", "I am going to Lisbon.")), r"company|employer", _TRIP),
    ("b3", "branch", _BUY_GOAL, {"method": "delivery", "detail": "12 Elm Street"},
     ((r"delivery|pickup", "I would like delivery."), (r"address|street", "My address is 12 Elm Street.")), r"store|branch|which shop", _BUY),
    ("b4", "branch", _BUY_GOAL, {"method": "pickup", "detail": "Harbor"},
     ((r"delivery|pickup", "I want pickup."), (r"store|branch|shop", "The Harbor store.")), r"address|street", _BUY),
    ("y1", "retry", f"Ask for the party size, from 1 to 8 guests. {_RANGE} Return party_size.", {"party_size": "6"}, ((r"party|guests|how many", ("We are 12 people.", "We are 6 people.")),), "", (1, 8)),
    ("y2", "retry", f"Ask how many tickets to buy, from 1 to 10. {_RANGE} Return quantity.", {"quantity": "4"}, ((r"tickets|how many|quantity", ("I want 25 tickets.", "I want 4 tickets.")),), "", (1, 10)),
    ("y3", "retry", f"Ask how many nights to stay, from 1 to 14. {_RANGE} Return nights.", {"nights": "5"}, ((r"nights|stay|how long", ("I will stay 30 nights.", "I will stay 5 nights.")),), "", (1, 14)),
    ("y4", "retry", f"Ask how many seats to reserve, from 1 to 6. {_RANGE} Return seats.", {"seats": "2"}, ((r"seats|how many", ("I need 9 seats.", "I need 2 seats.")),), "", (1, 6)),
))
# fmt: on


class User:
    """The scripted user, no model: ``reply`` is the next utterance of EVERY rule whose regex
    matches the assistant text (the last repeats), else ``FALLBACK`` (``confused``).
    ``note`` keeps the assistant texts produced after rule 0 (the branch value) answered."""

    def __init__(self, rules: tuple[tuple[str, Any], ...]) -> None:
        self.rules = [
            (re.compile(p, re.I), (s,) if isinstance(s, str) else tuple(s))
            for p, s in rules
        ]
        self.used, self.confused = [0] * len(rules), 0  # used[i]: answers rule i gave
        self.late: list[str] = []

    def note(self, text: str) -> None:
        self.late += [text] * bool(self.used[0])

    def reply(self, text: str) -> str:
        said = []
        for i, (pattern, lines) in enumerate(self.rules):
            if pattern.search(text):
                said.append(lines[min(self.used[i], len(lines) - 1)])
                self.used[i] += 1
        self.confused += not said
        return " ".join(said) or FALLBACK


# fmt: off
def _edge(target: str, keys: Any, logic: Any = None) -> dict[str, Any]:
    cond = {"description": f"to {target}", "requires_context_keys": list(keys)} | ({"logic": logic} if logic else {})
    return {"target_state": target, "description": f"to {target}", "priority": 1, "conditions": [cond]}


def _state(sid: str, say: str, fields: Any = (), edges: Any = (), extract: str = "") -> dict[str, Any]:
    state: dict[str, Any] = {"id": sid, "description": sid, "purpose": f"{sid} step", "response_instructions": say}
    state |= {"extraction_instructions": extract} if extract else {}
    state |= {"field_extractions": [{"field_name": k, "field_type": t, "extraction_instructions": h} for k, t, h in fields]} if fields else {}
    return state | ({"transitions": list(edges)} if edges else {})


def hand_fsm(task: Task) -> dict[str, Any]:
    """The reference child of *task*, one builder per family; it passes the compiler's
    own gates (``check_fixture``) and every task offline."""
    keys = tuple(task.gold)
    key, var = keys[0], {"var": keys[0]}
    if task.family == "collect":
        words = [k.replace("_", " ") for k in keys]
        fields = [(k, "str", f"the {w}") for k, w in zip(keys, words, strict=True)]
        states = {"ask": _state("ask", "Ask the user for: " + ", ".join(words) + ".", fields, [_edge("done", keys)])}
    elif task.family == "route":
        hint = "exactly one of: " + ", ".join(task.spec)
        states = {"ask": _state("ask", "Ask the user what they need.", [(key, "str", hint)], [_edge("done", keys, {"in": [var, list(task.spec)]})])}
    elif task.family == "branch":
        (one, two), ask_one, ask_two = task.spec
        edges = [_edge("first", [key], {"==": [var, one]}), _edge("second", [key], {"==": [var, two]})]
        detail, done = [("detail", "str", "the detail the user gave")], [_edge("done", ["detail"])]
        states = {"ask": _state("ask", f"Ask whether it is {one} or {two}.", [(key, "str", f"exactly one of: {one}, {two}")], edges),
                  "first": _state("first", ask_one, detail, done), "second": _state("second", ask_two, detail, done)}
    else:
        lo, hi = task.spec
        ask = f"Ask the user for the {key.replace('_', ' ')} (from {lo} to {hi}). If it is outside that range, say it is not allowed and ask again."
        # a later-turn correction comes only from the bulk pass, which needs these instructions
        why = f"Extract {key}: the whole number the user gives, even when it is out of range."
        edge = _edge("done", keys, {"and": [{">=": [var, lo]}, {"<=": [var, hi]}]})
        states = {"ask": _state("ask", ask, [(key, "int", "a whole number")], [edge], why)}
    states["done"] = _state("done", "Thank the user and say it is noted.")
    return {"name": f"Bench {task.id}", "description": task.goal[:250], "initial_state": "ask", "states": states}


def host_definition(task: Task) -> dict[str, Any]:
    """The host every FSM arm shares: welcome, a silent ``synthesize`` state, then
    ``wrapup`` on ``status == ok`` or ``sorry`` on ``failed`` or ``aborted``."""
    status = {"var": "synthesis_result.status"}

    def edge(target: str, priority: int, logic: Any) -> dict[str, Any]:
        cond = {"description": target, "requires_context_keys": ["synthesis_result"], "logic": logic}
        return {"target_state": target, "description": target, "priority": priority, "conditions": [cond]}

    work = _state("work", "", edges=[edge("wrapup", 1, {"==": [status, "ok"]}), edge("sorry", 2, {"in": [status, ["failed", "aborted"]]})])
    work["synthesize"] = {"goal": task.goal, "inputs": [], "return_keys": list(task.gold), "on_failure": "sorry"}
    states = {"welcome": _state("welcome", "Greet the user in one short sentence.", edges=[{"target_state": "work", "description": "go"}]), "work": work,
              "wrapup": _state("wrapup", "Tell the user it is done."), "sorry": _state("sorry", "Apologise briefly.")}
    return {"name": "Bench host", "description": "Welcomes, runs the procedure, wraps up.", "persona": "A calm, brief assistant", "initial_state": "welcome", "states": states}


def squash(value: Any) -> str:
    """Case, whitespace and punctuation dropped; a whole float reads as an int."""
    value = int(value) if isinstance(value, float) and value.is_integer() else value
    return re.sub(r"[\W_]+", "", str(value)).lower()


# DECISION plan-2026-11-10T194139-2cefeb61/D-065
# One grader for every arm. R's fallback reads only whole ``key: value`` pairs and
# the branch check only the texts AFTER the branch value was given (``User.late``).
# Do NOT match a gold token inside the answer (a label list or the word "it"
# passes) and do NOT read every reply for the branch check (a first question that
# names both options fails a right run).
def pairs(answer: str, keys: Any) -> dict[str, str]:
    """R's fallback: ``key: value``, ``key = value`` or ``key is value``, key a whole word; the
    value runs to a ``;`` or newline, a ``,`` or ``;`` before another key, or the end."""
    names = "|".join(map(re.escape, keys))
    found = {k: re.search(rf"\b{re.escape(k)}\b\W{{0,3}}(?::|=|\bis\b)\s*(.*?)\s*(?=[,;]\s*(?:{names})\b|[;\n]|$)", answer, re.I) for k in keys}  # fmt: skip
    return {k: m[1] for k, m in found.items() if m}


def react_values(answer: str, keys: Any) -> dict[str, Any]:
    """R's return values: the JSON object of its answer, else its ``pairs`` (never prose)."""
    from fsm_llm.utilities import extract_json_from_text

    return extract_json_from_text(answer) or pairs(answer, keys)


def grade(task: Task, values: dict[str, Any] | None, late: list[str], used: int) -> dict[str, Any]:
    """Correct = every key equals its gold after ``squash`` AND the family check: branch, no
    question about the other branch after the branch value; retry, the rule answered twice."""
    values = values or {}
    per_key = {k: values.get(k) is not None and squash(values[k]) == squash(g) for k, g in task.gold.items()}
    other = bool(task.other) and any(re.search(task.other, a, re.I) for a in late)
    checks = {"other_branch": not other, "reask": task.family != "retry" or used >= 2}
    return {"correct": all(per_key.values()) and all(checks.values()), "per_key": per_key, "checks": checks}


def decide(arms: dict[str, dict[str, Any]], complete: bool) -> tuple[str, list[str]]:
    """The pre-registered rule (``RULE``): ``(verdict, reasons)`` from each arm's metrics
    (``k`` tasks passed on the first trial, ``n_units``, ``infra_first``)."""
    s, r, h = arms[SYNTH], arms[REACT], arms[HAND]
    why = ["a block is aborted or incomplete"] * (not complete or any(a["n_units"] != PRIMARY_N for a in arms.values()))
    why += [f"h = {h['k']} of {PRIMARY_N}, below {H_FLOOR}"] * (h["k"] < H_FLOOR)
    why += [f"ceiling: s = {s['k']} and r = {r['k']}"] * (min(s["k"], r["k"]) >= CEILING)
    why += [f"{s['infra_first']} S first-trial failures are provider or host"] * (s["infra_first"] > MAX_INFRA)
    if why:
        return "INCONCLUSIVE", why
    p = bc.fisher_exact_two_sided(s["k"], PRIMARY_N, r["k"], PRIMARY_N)
    at = [f"s = {s['k']}, r = {r['k']}, h = {h['k']} of {PRIMARY_N}; Fisher p = {p:.4f}"]
    return ("SUPPORTED" if s["k"] > r["k"] else "REFUTED", at) if p < ALPHA else ("NOT SUPPORTED", at)


def phase(row: dict[str, Any]) -> str:
    """Where a trial failed: pass, provider, compile, run_budget, contract, wrong_route,
    wrong_value or host. Pure, from the row's own fields."""
    error, reason, status = row["error"] or "", row["runner_reason"], row["status"]
    wrong = "wrong_route" if row["family"] == "route" else "wrong_value"
    if row["correct"]:
        return "pass"
    if row["arm"] == REACT:
        return "provider" if error else wrong if row["stop_reason"] == "answered" else "run_budget"
    if reason == "provider_error" or any(w in error for w in ("LLMResponseError", "Timeout", "Connection")):
        return "provider"
    if reason == "compile_failed":
        return "compile"
    if reason == "budget_spent" or status in ("running", "aborted"):
        return "run_budget"
    return "contract" if reason == "return_keys_missing" else wrong if status == "ok" else "host"
# fmt: on


# --- END FIXTURE -------------------------------------------------------------


def fixture_sha256() -> str:
    text = Path(__file__).read_text(encoding="utf-8")
    start, end = text.index("# --- BEGIN FIXTURE"), text.index("# --- END FIXTURE")
    return hashlib.sha256(text[start:end].encode("utf-8")).hexdigest()


def code_pins() -> dict[str, Any]:
    """The hashes of what the verdict depends on: the fixture region and each ``SRC_PINS`` file."""
    return {"fixture_hash": fixture_sha256(), "src_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SRC_PINS}}  # fmt: skip


def edited_pins(recorded: dict[str, Any]) -> list[str]:
    """The pinned files (and the fixture region) that differ from a registered manifest."""
    now, was = code_pins(), recorded.get("src_sha256") or {}
    return [p for p, h in now["src_sha256"].items() if was.get(p) != h] + ["the fixture region"] * (recorded.get("fixture_hash") != now["fixture_hash"])  # fmt: skip


def min_gaps(n: int = PRIMARY_N) -> dict[int, int]:
    """For each r, the smallest s with Fisher p < ALPHA (printed in the pre-registration)."""  # fmt: skip
    return {r: next((s for s in range(r, n + 1) if bc.fisher_exact_two_sided(s, n, r, n) < ALPHA), n + 1) for r in range(n)}  # fmt: skip


def check_fixture() -> list[str]:
    """Register-time problems (empty = runnable): every hand child passes the
    compiler's gates and every host loads."""
    import fsm_llm.synthesis as syn
    from fsm_llm.definitions import FSMDefinition

    policy, problems = syn.SynthesisPolicy(**POLICY), []
    problems += ["a task id repeats"] * (len({t.id for t in TASKS}) != len(TASKS))
    for t in TASKS:
        report = syn.run_gates(hand_fsm(t), policy, return_keys=tuple(t.gold))
        problems += [f"{t.id}: hand child fails the gates: {report.repair_text(300)}"] * (not report.passed)  # fmt: skip
        try:
            FSMDefinition(**host_definition(t))
        except ValueError as exc:
            problems.append(f"{t.id}: host does not load: {str(exc)[:100]}")
    return problems


# --- Models, wire, server -----------------------------------------------------


def require_model(model: str) -> str:
    """Every role is ``MODEL`` (D-039): any other string is refused."""
    if model != MODEL:
        raise bc.BenchDataError(f"one model only: {model!r} is not {MODEL!r}")
    return model


def make_llm(role: str, model: str = MODEL) -> Any:
    """THE interface construction site (the tests replace it). ``role`` is compile,
    sim or run (own timeouts); NO ``num_ctx`` is sent (D-040: it reloads the model)."""
    from fsm_llm import LiteLLMInterface

    return LiteLLMInterface(model=require_model(model), temperature=LIMITS["temperature"], max_tokens=LIMITS["run_max_tokens"], timeout=TIMEOUTS[role])  # fmt: skip


def wire_disclosure() -> dict[str, Any]:
    """The request settings of each role, taken at the wire with the provider refused."""
    from fsm_llm.definitions import CompletionRequest

    out = {}
    for role in ("compile", "run"):
        with bc.captured_wire() as seen, contextlib.suppress(Exception):
            make_llm(role).complete(CompletionRequest(messages=[{"role": "user", "content": "hi"}]))  # fmt: skip
        out[role] = bc.wire_settings(seen[0])
    return out


@contextmanager
def wire_spy() -> Iterator[list[dict[str, Any]]]:
    """Record every request at core's one send binding while still sending: role (by the
    request's timeout), seconds, tokens, raw params and response."""
    import fsm_llm.llm as core_llm

    seen: list[dict[str, Any]] = []
    original = core_llm.completion

    def spy(**params: Any) -> Any:
        role = ROLE_BY_TIMEOUT.get(params.get("timeout"), "other")
        seen.append(rec := {"role": role, "seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "params": copy.deepcopy(params)})  # fmt: skip
        started = time.perf_counter()
        try:
            response = original(**params)
        except Exception as exc:
            rec["error"] = type(exc).__name__
            raise
        finally:
            rec["seconds"] = round(time.perf_counter() - started, 3)
        used = getattr(response, "usage", None)
        rec["prompt_tokens"] = int(getattr(used, "prompt_tokens", 0) or 0)
        rec["completion_tokens"] = int(getattr(used, "completion_tokens", 0) or 0)
        dump = getattr(response, "model_dump", None)
        rec["response"] = dump() if callable(dump) else str(response)
        return response

    core_llm.completion = spy
    try:
        yield seen
    finally:
        core_llm.completion = original


class Blocked(bc.BenchDataError):
    """The preflight refused (exit 3): the step ends BLOCKED, no model call is made."""


class ForeignLoad(bc.BenchDataError):
    """The server changed under a running block (status ``aborted: foreign-load``)."""


class OllamaClient:
    """The read-only calls of the preflight; tests inject a fake with the same
    three methods, so no test opens a socket. ``urllib`` loads lazily."""

    base = "http://localhost:11434"

    def _call(self, path: str, body: dict[str, Any] | None, timeout: float) -> Any:
        from urllib.request import Request, urlopen

        data = None if body is None else json.dumps(body).encode()
        request = Request(self.base + path, data=data, headers={"Content-Type": "application/json"})  # fmt: skip
        with urlopen(request, timeout=timeout) as resp:
            return json.load(resp)

    def ps(self) -> dict[str, Any]:
        """``/api/ps``: the loaded models and their contexts."""
        return self._call("/api/ps", None, 10)  # type: ignore[no-any-return]

    @staticmethod
    def _lines(cmd: list[str]) -> list[str] | None:
        """The stdout lines of a ``journalctl`` query, ``None`` if it could not be read."""
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        bad = out.returncode or "No journal files" in out.stderr or "permission" in out.stderr  # fmt: skip
        return None if bad else out.stdout.splitlines()

    # DECISION plan-2026-11-10T194139-2cefeb61/D-068
    # An empty window is quiet ONLY if the unit has an entry at all. `journalctl -u
    # <missing unit>` exits 0 with no output, so silence alone passed on a missing
    # unit, an unreadable journal or a Docker server. Do NOT read empty as quiet.
    def journal(self, minutes: int) -> list[str] | None:
        """The ollama unit's journal lines of the last *minutes*; ``None`` (unreadable, the
        preflight refuses) unless an empty window is backed by an entry the unit ever wrote."""
        cmd = ["journalctl", "-u", "ollama", "--no-pager", "-o", "cat"]
        lines = self._lines([*cmd, "--since", f"{minutes} minutes ago"])
        return None if lines == [] and not self._lines([*cmd, "-n", "1"]) else lines

    def probe(self) -> tuple[float, float]:
        """One one-token chat, no ``num_ctx``: ``(client wall seconds, server
        seconds)``; the difference is the queue wait."""
        body = {"model": MODEL_TAG, "stream": False, "think": False, "options": {"num_predict": 1}, "messages": [{"role": "user", "content": "Reply with one word."}]}  # fmt: skip
        started = time.perf_counter()
        reply = self._call("/api/chat", body, 120)
        return time.perf_counter() - started, float(reply.get("total_duration", 0)) / 1e9  # fmt: skip


def _loaded(models: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """The loaded models as ``{name, context_length}`` and what is wrong with them."""
    seen, wrong = [], []
    for m in models:
        name, ctx = m.get("name") or m.get("model"), m.get("context_length")
        seen.append({"name": name, "context_length": ctx})
        if name != MODEL_TAG:
            wrong.append(f"model {name} is loaded")
        elif ctx != DEFAULT_CONTEXT:
            wrong.append(f"{name} is loaded with context {ctx}, not {DEFAULT_CONTEXT}")
    return seen, wrong


def guard(client: Any) -> list[dict[str, Any]]:
    """The per-trial read-only check (D-040 item 2): the loaded models, or ``ForeignLoad``."""  # fmt: skip
    try:
        seen, wrong = _loaded(client.ps().get("models", []))
    except Exception as exc:
        raise ForeignLoad(f"cannot read /api/ps: {type(exc).__name__}") from exc
    if wrong:
        raise ForeignLoad("; ".join(wrong))
    return seen


def preflight(client: Any, *, confirm_exclusive: bool = False, sleep: Callable[[float], None] = time.sleep) -> list[str]:  # fmt: skip
    """The refusals (empty = pass), D-040: (a) no other model loaded, ours at the default
    context; (b) a quiet journal, or ``confirm_exclusive`` when it is unreadable (only the
    orchestrator passes it, after the USER says the server is free); (c) three probes
    with a short queue wait, sent only after (a) and (b) pass."""
    # DECISION plan-2026-11-10T194139-2cefeb61/D-054
    # Do NOT send num_ctx, do NOT probe before (a) and (b) pass, and NEVER act on a
    # process: this file only reads (/api/ps, the journal) and refuses. A queued
    # run measures the queue, not the model.
    try:
        _, refusals = _loaded(client.ps().get("models", []))
    except Exception as exc:
        return [f"cannot read /api/ps: {type(exc).__name__}"]
    lines = client.journal(QUIET_MINUTES)
    if lines is None:
        refusals += [] if confirm_exclusive else ["the journal is unreadable and --confirm-exclusive was not given"]  # fmt: skip
    elif busy := [x for x in lines if CHAT_LINE.search(x)]:
        refusals.append(f"{len(busy)} request line(s) in the journal of the last {QUIET_MINUTES} minutes")  # fmt: skip
    for i in range(PROBES * (not refusals)):
        if i:
            sleep(PROBE_SPACING)
        try:
            wall, served = client.probe()
        except Exception as exc:
            return [f"probe failed: {type(exc).__name__}"]
        if wall - served >= MAX_QUEUE_WAIT:
            return [f"probe {i + 1}: queue wait {wall - served:.1f} s, at least {MAX_QUEUE_WAIT} s"]  # fmt: skip
    return refusals


# --- One trial ----------------------------------------------------------------


def meter(llms: dict[str, Any]) -> dict[str, int]:
    """Calls and tokens per role off each interface's ``usage()`` (0 for a role not used)."""  # fmt: skip
    use = {role: llms[role].usage() if role in llms else None for role in TIMEOUTS}
    return {f"{role}_{f}": int(getattr(u, f, 0) or 0) for role, u in use.items() for f in ("calls", "prompt_tokens", "completion_tokens")}  # fmt: skip


def drive_fsm(arm: str, task: Task, lib_dir: Path, llms: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    """Arms S and H on one host, one runner, one library path: user turns until the
    conversation ends or ``max_user_turns``. Returns the row's FSM fields."""
    import fsm_llm.synthesis as syn
    from fsm_llm import API
    from fsm_llm.synthesis.gates import GATES

    hand = arm == HAND
    policy = syn.SynthesisPolicy(**POLICY, require_simulation=not hand)
    library, keys = syn.FSMLibrary(lib_dir), tuple(task.gold)
    if hand:
        # DECISION plan-2026-11-10T194139-2cefeb61/D-054
        # H is seeded through the compiler's own gates, with gates_passed = GATES
        # only: G6b did not run on a hand-written child and is never claimed. Do
        # NOT write G6b here and do NOT compile: H makes zero strong calls (a
        # trial with one is an errored row).
        gates = syn.run_gates(hand_fsm(task), policy, return_keys=keys)
        if gates.definition is None:
            raise bc.BenchDataError(f"{task.id}: the hand child fails the gates")
        library.put(gates.definition, goal=task.goal, inputs_required=(), return_keys=keys, synthesizing_model="hand-written", gates_passed=GATES, lifecycle="verified")  # fmt: skip
    sim = None if hand else syn.Simulator(llms["sim"], llms["compile"])
    synth = syn.FSMSynthesizer(llms["compile"], policy=policy, library=library, simulator=sim)  # fmt: skip
    events: list[tuple[str, dict[str, Any]]] = []
    host = host_definition(task)
    api = API.from_definition(definition=host, llm_interface=llms["run"])
    runner = syn.ProcedureRunner(api, host, synth, trace_sink=lambda e, f: events.append((e, f)))  # fmt: skip
    conv, _ = api.start_conversation()
    user, said, turns = User(task.users), OPENER, []
    try:
        for _ in range(LIMITS["max_user_turns"]):
            reply = runner.converse(conv, said)
            user.note(reply)
            turns.append([said[:300], reply[:300]])
            if api.has_conversation_ended(conv):
                break
            said = user.reply(reply)
        data = api.get_data(conv)
    finally:
        with contextlib.suppress(Exception):
            runner.end_conversation(conv)
    result = data.get("synthesis_result") or {}
    got = next((f for e, f in events if e in ("obtained", "failed")), {})
    grades = grade(task, {k: data.get(k) for k in keys}, user.late, user.used[0])
    # fmt: off
    return {**grades, "status": result.get("status"), "runner_reason": result.get("reason"), "source": got.get("source"),
            "rounds": [c.split(",") for c in (got.get("rounds") or "").split(";") if c], "compile_seconds": got.get("seconds", 0.0),
            "turns": len(turns), "confused_turns": user.confused, "transcript": turns,
            "child_states": sorted({f["state"] for e, f in events if e == "stepped" and f.get("state")})}
    # fmt: on


def drive_react(task: Task, llm: Any) -> dict[str, Any]:
    """Arm R: ``create_agent("react")`` with ``ask_user`` backed by the same user rules; the
    answer is a JSON object (else ``key: value`` pairs, D-065)."""
    from fsm_llm.agents import AgentConfig, ToolRegistry, create_agent

    user, asked = User(task.users), []

    def ask_user(question: str) -> str:
        """Ask the user one question and return the user's reply."""
        asked.append(question)
        user.note(question)
        return user.reply(question) if len(asked) <= LIMITS["max_user_turns"] else "The user has left."  # fmt: skip

    registry = ToolRegistry().register_function(ask_user, name="ask_user")
    config = AgentConfig(model=MODEL, max_iterations=LIMITS["react_max_iterations"], timeout_seconds=LIMITS["react_timeout_seconds"], temperature=LIMITS["temperature"], max_tokens=LIMITS["run_max_tokens"])  # fmt: skip
    prompt = f"{task.goal}\nTalk to the user only with the ask_user tool. When you have everything, answer with ONLY a JSON object with the keys {', '.join(task.gold)}."  # fmt: skip
    result = create_agent("react", registry, config=config, llm_interface=llm).run(prompt)  # fmt: skip
    answer = str(result.answer or "")
    return {**grade(task, react_values(answer, task.gold), user.late, user.used[0]), "stop_reason": result.stop_reason, "turns": len(asked), "confused_turns": user.confused, "answer": answer[:500]}  # fmt: skip


def run_trial(arm: str, task: Task, trial: int, workdir: Path, raw: Path | None = None) -> dict[str, Any]:  # fmt: skip
    """One trial of *arm* on *task*; an exception is a failed row, never a raise.

    S keeps its library in ``workdir/synth`` across a task's trials (2 and 3 reuse
    the stored procedure), H seeds a fresh one per trial. The row holds grades,
    ``phase``, the runner's status and reason, ``source``, compile ``rounds``,
    meters per role, one entry per request, ``models`` and ``models_ok`` (a role
    on another model is an errored row and nothing runs).
    """
    row: dict[str, Any] = {"task_id": task.id, "family": task.family, "arm": arm, "trial": trial, "correct": False, "per_key": {}, "error": None, "status": None, "runner_reason": None, "stop_reason": None, "source": None, "rounds": [], "compile_seconds": 0.0, "turns": 0, "confused_turns": 0, "child_states": [], "transcript": [], "answer": "", "models": {}, "models_ok": True}  # fmt: skip
    llms: dict[str, Any] = {}
    started = time.perf_counter()
    with wire_spy() as wire:
        try:
            llms = {role: make_llm(role) for role in (("run",) if arm == REACT else TIMEOUTS)}  # fmt: skip
            row["models"] = {role: getattr(llm, "model", None) for role, llm in llms.items()}  # fmt: skip
            if foreign := [r for r, m in row["models"].items() if m != MODEL]:
                row.update(models_ok=False, error=f"one model only: {foreign} are not {MODEL}")  # fmt: skip
            elif arm == REACT:
                row.update(drive_react(task, llms["run"]))
            else:
                row.update(drive_fsm(arm, task, workdir / ("synth" if arm == SYNTH else f"hand{trial}"), llms))  # fmt: skip
        except Exception as exc:  # a crashed or timed-out trial is a failed row
            row.update(correct=False, error=f"{type(exc).__name__}: {exc}"[:300])
    row.update(meter(llms), latency_s=round(time.perf_counter() - started, 3))
    if arm == HAND and row["compile_calls"]:
        row.update(correct=False, error="hand-fsm made a strong call")
    row["wire"] = [{k: w[k] for k in ("role", "seconds", "prompt_tokens", "completion_tokens")} for w in wire]  # fmt: skip
    row["max_prompt_tokens"] = max((w["prompt_tokens"] for w in wire), default=0)
    row["phase"] = phase(row)
    if raw is not None:
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(json.dumps({"row": row, "wire": wire}, default=str))
    return row


# --- Manifest, register, run ---------------------------------------------------


def _paths(bdir: Path, arm: str) -> tuple[Path, Path, Path]:
    return bdir / f"manifest_{arm}.json", bdir / f"rows_{arm}.jsonl", bdir / f"summary_{arm}.json"  # fmt: skip


def plan(block: str = "s0") -> list[tuple[str, Task, int]]:
    """The fixed run order: task-major (S reuses its library across a task's
    trials), arms alternate and the first arm flips from task to task."""
    arms = BLOCKS[block]
    return [(a, t, n) for i, t in enumerate(TASKS) for n in range(1, TRIALS + 1) for a in arms[:: 1 - i % 2 * 2]]  # fmt: skip


def build_manifest(block: str, arm: str) -> dict[str, Any]:
    return {
        "bench_id": BENCH_ID, "block": block, "model": require_model(MODEL), "trials": TRIALS,
        "arm": {"name": arm}, "order": "task-major, arms alternate, first arm flips per task",
        "n_tasks": len(TASKS), "limits": copy.deepcopy(LIMITS), **code_pins(),
        "model_digest": bc._model_digest(MODEL_TAG), "wire": wire_disclosure(),
        "git_commit": bc._git_commit(), "created_at": bc._utc_now(),
    }  # fmt: skip


def register_block(block: str) -> list[Path]:
    """Write every arm's manifest of *block* (nothing else), once."""
    bdir = BENCH_DATA / BENCH_ID / block
    if existing := [p for a in BLOCKS[block] for p in _paths(bdir, a) if p.exists()]:
        raise bc.BenchDataError(f"{existing[0]} exists -- a block is registered ONCE")
    if problems := check_fixture():
        raise bc.BenchDataError("fixture: " + "; ".join(problems))
    bdir.mkdir(parents=True, exist_ok=True)
    for arm in BLOCKS[block]:
        bc._write_json(_paths(bdir, arm)[0], build_manifest(block, arm))
    return [_paths(bdir, arm)[0] for arm in BLOCKS[block]]


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every summary number from raw rows alone. Unit: the first trial of each task."""
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row)
    first = [min(rs, key=lambda r: r["trial"]) for rs in by_task.values()]
    k, lib = sum(bool(r["correct"]) for r in first), [r for r in rows if r.get("source") == "library"]  # fmt: skip

    def total(key: str) -> int:
        return sum(int(r.get(key) or 0) for r in rows)

    # fmt: off
    return {
        "n_rows": len(rows), "n_units": len(first), "k": k, "wilson": [round(x, 4) for x in bc.wilson_ci(k, len(first))],
        "pass_all": sum(len(rs) >= TRIALS and all(r["correct"] for r in rs) for rs in by_task.values()),
        "families": {f: [sum(bool(r["correct"]) for r in first if r["family"] == f), sum(r["family"] == f for r in first)] for f in sorted({r["family"] for r in first})},
        "phases_first": dict(sorted(Counter(r["phase"] for r in first if not r["correct"]).items())),
        "infra_first": sum(r["phase"] in ("provider", "host") for r in first),
        "errored_rows": sum(bool(r.get("error")) for r in rows), "foreign_model_rows": sum(not r.get("models_ok", True) for r in rows),
        "reuse_rows": len(lib), "reuse_strong_calls": sum(int(r.get("compile_calls") or 0) for r in lib),
        "confused_turns": total("confused_turns"), "wall_s": round(sum(float(r.get("latency_s") or 0) for r in rows), 1),
        "meters": {f"{role}_{f}": total(f"{role}_{f}") for role in TIMEOUTS for f in ("calls", "prompt_tokens", "completion_tokens")},
    }
    # fmt: on


def write_summary(bdir: Path, arm: str, status: str, started: str, run: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    """Summary from the arm's raw rows; ``status`` is complete or aborted: reason."""
    _, rows_path, summary_path = _paths(bdir, arm)
    summary = {"arm": arm, "status": status, "started_at": started, "finished_at": bc._utc_now(), "run": run, "metrics": metrics(bc.read_rows(rows_path))}  # fmt: skip
    bc._write_json(summary_path, summary)
    return summary


def _gate(client: Any, confirm_exclusive: bool, sleep: Callable[[float], None]) -> Any:
    """The preflight of ``run`` and ``smoke``: the client, or ``Blocked``."""
    client = client or OllamaClient()
    if refusals := preflight(client, confirm_exclusive=confirm_exclusive, sleep=sleep):
        raise Blocked("; ".join(refusals))
    return client


def run_block(block: str, *, client: Any = None, confirm_exclusive: bool = False, sleep: Callable[[float], None] = time.sleep) -> dict[str, dict[str, Any]]:  # fmt: skip
    """LIVE. Refuse (no network) a block with rows or no manifests, then the preflight,
    then a drifted manifest; run ``plan`` (a read-only ``guard`` before each trial)."""
    bdir, arms = BENCH_DATA / BENCH_ID / block, BLOCKS[block]
    for arm in arms:
        manifest_path, rows_path, summary_path = _paths(bdir, arm)
        if rows_path.exists() or summary_path.exists():
            raise bc.BenchDataError(f"{rows_path} exists -- a block is run ONCE")
        if not manifest_path.is_file():
            raise bc.BenchDataError(f"{manifest_path} missing -- register first")
    client = _gate(client, confirm_exclusive, sleep)
    for arm in arms:
        recorded = json.loads(_paths(bdir, arm)[0].read_text(encoding="utf-8"))
        if drift := bc.pinned_drift(recorded, build_manifest(block, arm), PINNED_AT_RUN):  # fmt: skip
            raise bc.BenchDataError(f"{arm} was registered with different {drift}; edited after register: {', '.join(edited_pins(recorded)) or 'no pinned file'}")  # fmt: skip
    started, ran, status, streak, todo = bc._utc_now(), bc.run_commit(), "aborted: not finished", 0, plan(block)  # fmt: skip
    try:
        with tempfile.TemporaryDirectory(prefix="synthesis-bench-") as tmp:
            for done, (arm, task, trial) in enumerate(todo, 1):
                ps = guard(client)
                raw = RAW_DIR / BENCH_ID / block / f"{arm}__{task.id}__t{trial}.json"
                row = run_trial(arm, task, trial, Path(tmp) / task.id, raw)
                row.update(bench_id=BENCH_ID, block=block, ps=ps, ts=bc._utc_now())
                bc.append_row(_paths(bdir, arm)[1], row)
                print(f"  {block}/{arm} {done}/{len(todo)} {task.id} t{trial}: {row['phase']} calls={row['compile_calls']}/{row['sim_calls']}/{row['run_calls']} {row['latency_s']}s", flush=True)  # fmt: skip
                streak = streak + 1 if row["error"] or row["phase"] == "provider" else 0
                if streak >= OUTAGE_ROWS:
                    raise bc.BenchDataError(f"{OUTAGE_ROWS} consecutive errored rows (provider outage?)")  # fmt: skip
        status = "complete"
    except BaseException as exc:  # keep the rows, summarise, re-raise
        status = "aborted: foreign-load" if isinstance(exc, ForeignLoad) else f"aborted: {type(exc).__name__}: {exc}"[:200]  # fmt: skip
        raise
    finally:
        out = {a: write_summary(bdir, a, status, started, {**ran, "preflight": "passed"}) for a in arms}  # fmt: skip
        print(f"wrote summaries for {block} (status: {status})\n{ONE_MODEL}")
    return out


def smoke(ids: list[str], *, client: Any = None, confirm_exclusive: bool = False, sleep: Callable[[float], None] = time.sleep) -> list[dict[str, Any]]:  # fmt: skip
    """LIVE, after the preflight: one trial per arm per task in *ids*; raw requests in
    ``logs/synthesis-bench/smoke/``, nothing under ``scripts/bench_data/``."""
    if not (wanted := [t for t in TASKS if t.id in ids]):
        raise bc.BenchDataError(f"no such task: {ids}")
    client, rows = _gate(client, confirm_exclusive, sleep), []
    with tempfile.TemporaryDirectory(prefix="synthesis-smoke-") as tmp:
        for task in wanted:
            for arm in BLOCKS["s0"]:
                row = run_trial(arm, task, 1, Path(tmp) / task.id, RAW_DIR / "smoke" / f"{arm}__{task.id}.json")  # fmt: skip
                rows.append(row | {"ps": guard(client)})
                print(f"  smoke {arm} {task.id}: {row['phase']} correct={row['correct']} rounds={row['rounds']} calls={row['compile_calls']}/{row['sim_calls']}/{row['run_calls']} max_prompt={row['max_prompt_tokens']} {row['latency_s']}s ps={rows[-1]['ps']}", flush=True)  # fmt: skip
    print(ONE_MODEL)
    return rows


# --- Report -------------------------------------------------------------------


def report(block: str) -> int:
    """Recount every arm from RAW rows, cross-check the committed summaries, apply
    the pre-registered rule once all three arms are there. 0 ok; 1 on a mismatch."""
    bdir, ok, recount, complete = BENCH_DATA / BENCH_ID / block, True, {}, True
    if not bdir.is_dir():
        raise bc.BenchDataError(f"no such block: {bdir}")
    for arm in BLOCKS[block]:
        manifest_path, rows_path, summary_path = _paths(bdir, arm)
        if manifest_path.is_file() and (edited := edited_pins(json.loads(manifest_path.read_text(encoding="utf-8")))):  # fmt: skip
            raise bc.BenchDataError(
                f"{arm}: edited after register: {', '.join(edited)}"
            )
        if not rows_path.is_file():
            continue
        recount[arm] = m = json.loads(json.dumps(metrics(bc.read_rows(rows_path))))
        print(f"{BENCH_ID} {block} [{arm}] rows={m['n_rows']}: first trial {m['k']}/{m['n_units']} wilson95={m['wilson']}; pass^{TRIALS} {m['pass_all']}; families {m['families']}; failures {m['phases_first']}; reuse rows {m['reuse_rows']} (strong calls {m['reuse_strong_calls']}); meters {m['meters']}; wall {m['wall_s']} s")  # fmt: skip
        committed = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None  # fmt: skip
        complete &= committed is not None and committed.get("status") == "complete"
        want = (committed or {}).get("metrics", m)
        for key in sorted(want.keys() | m.keys()):
            if want.get(key) != m.get(key):
                ok = False
                print(f"  MISMATCH {key}: committed {want.get(key)} != recount {m.get(key)}")  # fmt: skip
    if not recount:
        print(f"{BENCH_ID} {block}: registered, NOT RUN (no rows); smallest s with p < {ALPHA} for each r of {PRIMARY_N}: {min_gaps()}")  # fmt: skip
    elif set(recount) == set(BLOCKS[block]):
        verdict, why = decide(recount, complete)
        print(f"verdict: {verdict} ({'; '.join(why)})\nrule: {RULE}")
    print(ONE_MODEL)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("command", choices=("register", "run", "report", "smoke"))  # fmt: skip
    parser.add_argument("block", nargs="?", default="s0", choices=sorted(BLOCKS))
    parser.add_argument("--tasks", nargs="+", default=["c1"], help="smoke: task ids")
    parser.add_argument("--confirm-exclusive", action="store_true", help="ONLY after the user says the server is free")  # fmt: skip
    args = parser.parse_args(argv)
    try:
        if args.command == "register":
            print(*(f"registered {p}" for p in register_block(args.block)), sep="\n")
        elif args.command == "run":
            run_block(args.block, confirm_exclusive=args.confirm_exclusive)
        elif args.command == "smoke":
            smoke(args.tasks, confirm_exclusive=args.confirm_exclusive)
        else:
            return report(args.block)
        return 0
    except Blocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return EXIT_BLOCKED
    except bc.BenchDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
