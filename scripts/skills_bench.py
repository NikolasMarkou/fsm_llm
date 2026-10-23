#!/usr/bin/env python3
"""Pre-registered live bench for Agent Skills progressive disclosure.

Question (I-132, I-133, I-097): with 16 skills, does a small model load the
right skill, then read the right bundled file, and does the fixed read hint
(``constants.SKILL_READ_HINT``) on the first ``skill_load`` page matter?

Block B0 (decision): ``qwen3.5:4b`` ReAct, arms ``hint_on`` and ``hint_off``.
Block B1 (informational): ``native_4b``, ``native_4b_hint_off``, ``react_9b``,
``native_9b`` (hint on except the one named off).
12 questions x 2 trials per arm. Grading is offline, no judge (``grade``).
The rule is in ``bench_data/skills-disclosure/PRE_REGISTRATION_B*.md``.

    .venv/bin/python scripts/skills_bench.py register skills-disclosure B0
    .venv/bin/python scripts/skills_bench.py run skills-disclosure B0
    .venv/bin/python scripts/skills_bench.py report skills-disclosure

Same discipline as ``agents_bench.py`` (manifest before row 1, a block runs
once, ``report`` recounts raw rows); it reuses that module's helpers through
``ab``/``hb``: no copy of wilson, fisher, append_row or read_rows. The module
top level is stdlib only; ``fsm_llm`` loads lazily. Raw requests and responses
of every trial go to ``logs/skills-bench/`` (gitignored).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agents_bench as ab  # loads harness_bench as ab.hb

hb = ab.hb  # wilson_ci, fisher_exact_two_sided, append_row, read_rows, pinned_drift

ROOT = Path(__file__).resolve().parent.parent
BENCH_DATA = ROOT / "scripts" / "bench_data"
RAW_DIR = ROOT / "logs" / "skills-bench"  # logs/ is gitignored
TRIALS = 2
M4B, M9B = "ollama_chat/qwen3.5:4b", "ollama_chat/qwen3.5:9b-q8_0"
LIMITS = {"max_iterations": 8, "timeout_seconds": 300.0, "temperature": 0.5, "max_tokens": 1000}  # fmt: skip
#: arm -> (agent pattern, model, hint shown). A label names what ran.
ARMS = {
    "hint_on": ("react", M4B, True), "hint_off": ("react", M4B, False),
    "native_4b": ("native_fc", M4B, True), "react_9b": ("react", M9B, True),
    "native_9b": ("native_fc", M9B, True), "native_4b_hint_off": ("native_fc", M4B, False),
}  # fmt: skip
BLOCKS = {"B0": ("hint_on", "hint_off"), "B1": ("native_4b", "native_4b_hint_off", "react_9b", "native_9b")}  # fmt: skip
SRC_PINS = ("src/fsm_llm/agent_skills.py", "src/fsm_llm/agents/agent_skills.py", "src/fsm_llm/constants.py")  # fmt: skip
PINNED_AT_RUN = ("fixture_hash", "trials", "model", "limits", "src_sha256", "arm")
PRIMARY_N = 10  # read-needed questions per arm
OUTAGE_ROWS = 3  # consecutive errored rows abort a block


# --- BEGIN FIXTURE (hashed: skills, questions, builder, grader) ---------------
# fixture_hash is the sha256 of these lines; any edit is a new, incomparable
# fixture. Each skill: (name, description, SKILL.md body sentence, files).

SKILLS: tuple[tuple[str, str, str, dict[str, str]], ...] = (
    ("gateway-ops", "Operate the Zephyr gateway: restarts, health checks, payload and rate limits.",
     "For the numeric limits of the gateway, see references/limits.md.",
     {"references/limits.md": "The Zephyr gateway accepts payloads up to 7341 KB."}),
    ("gateway-billing", "Invoice customers of the Zephyr gateway: pricing tiers, credits, refunds.",
     "Pricing uses volume tiers. Tier prices are in references/tiers.md.",
     {"references/tiers.md": "The first volume tier ends at 2204 KB."}),
    ("sensor-calibration", "Calibrate Vantor probes and record the result.",
     "Recalibration intervals of each probe are listed in references/intervals.md.",
     {"references/intervals.md": "A Vantor probe must be recalibrated every 4185 operating hours."}),
    ("sensor-wiring", "Wire Vantor probes to a controller: pins, cable lengths.",
     "Cable limits are in references/cables.md.",
     {"references/cables.md": "A Vantor probe cable may be at most 1692 m long."}),
    ("queue-tuning", "Tune the Quillon job queue: retries, backoff, concurrency.",
     "The retry and backoff settings of the queue are kept in the files bundled with this skill.",
     {"references/backoff.md": "The Quillon queue never waits longer than 8264 ms between two retries.",
      "references/concurrency.md": "A Quillon worker runs at most 64 jobs at once."}),
    ("queue-monitoring", "Monitor the Quillon job queue: depth alerts, dashboards, paging.",
     "Alert thresholds are in references/alerts.md.",
     {"references/alerts.md": "Alert when the Quillon queue holds more than 9120 waiting jobs."}),
    ("report-builder", "Build Marlow warehouse reports and exports: columns, rows, formats.",
     "Column caps, row caps and formats are described in the bundled files.",
     {"references/rows.md": "A Marlow report export holds at most 12673 rows.",
      "references/columns.md": "A Marlow report holds at most 41 columns."}),
    ("report-scheduler", "Schedule Marlow reports: cron times, retries, delivery lists.",
     "Delivery limits are in references/delivery.md.",
     {"references/delivery.md": "A scheduled Marlow report is sent to at most 2350 recipients."}),
    ("fleet-routing", "Plan Corvid fleet routes by region: depots, handovers, windows.",
     "Each region has its own depot notes under references.",
     {"references/regions/north.md": "The north depot of the Corvid fleet is coded NRD-5529.",
      "references/regions/south.md": "The south depot of the Corvid fleet is coded SRD-6403."}),
    ("audit-export", "Export Halden audit logs: formats, filters, retention.",
     "Per-format notes are bundled under references.",
     {"references/formats/csv.md": "A Halden CSV audit export is cut into chunks of 6082 rows.",
      "references/formats/json.md": "A Halden JSON audit export is cut into chunks of 1930 records."}),
    ("vault-rotation", "Rotate Sable vault keys: schedule, policy, contacts, limits.",
     "Read the bundled files for the rotation facts.",
     {"references/policy.md": "A rotated Sable key is kept for 14 days before it is destroyed.",
      "references/schedule.md": "Sable vault keys rotate every 1473 hours.",
      "references/contacts.md": "Rotation incidents go to the desk coded RT-27.",
      "references/limits.md": "A Sable vault holds at most 3658 active keys."}),
    ("vault-access", "Request and approve access to the Sable vault: roles, tickets.",
     "Access expiry rules are in references/expiry.md.",
     {"references/expiry.md": "Sable vault access expires after 5890 minutes."}),
    ("label-printing", "Print Brindle shipping labels: sizes, margins, ribbons, printers.",
     "Sizes, margins, ribbons and printer limits are in the bundled files.",
     {"references/sizes.md": "A Brindle label is 100 mm wide.",
      "references/margins.md": "A Brindle label keeps a margin of 9407 microns.",
      "references/ribbons.md": "A Brindle ribbon roll is 5216 mm long.",
      "references/printers.md": "A Brindle printer holds 3 rolls."}),
    ("label-templates", "Design Brindle label templates: fields, fonts, barcodes.",
     "Barcode limits are in references/barcodes.md.",
     {"references/barcodes.md": "A Brindle template holds at most 7318 barcodes."}),
    ("onboarding-checklist", "Welcome new staff: accounts, badges, first-week checklist.",
     "Give every new hire the welcome pack coded WP-8801 on day one.",
     {"references/forms.md": "The badge form is HR-12."}),
    ("holiday-calendar", "Look up company holidays and shutdown days.",
     "Each employee has a floating holiday balance of 3190 minutes per year.",
     {"references/regions.md": "Holidays differ by region."}),
)  # fmt: skip


class Question(NamedTuple):
    id: str
    kind: str  # pointer | unnamed | nested | multi | control
    skill: str
    path: str  # the file holding the answer; "" for a control (its SKILL.md)
    text: str
    token: str

    @property
    def control(self) -> bool:
        return self.kind == "control"


QUESTIONS = tuple(Question(*row) for row in (
    ("q01", "pointer", "gateway-ops", "references/limits.md", "What is the maximum payload size, in KB, that the Zephyr gateway accepts?", "7341"),
    ("q02", "unnamed", "queue-tuning", "references/backoff.md", "What is the longest wait, in milliseconds, between two retries of the Quillon queue?", "8264"),
    ("q03", "nested", "fleet-routing", "references/regions/north.md", "What is the depot code of the Corvid fleet in the north region?", "5529"),
    ("q04", "multi", "vault-rotation", "references/schedule.md", "How many hours pass between two rotations of the Sable vault keys?", "1473"),
    ("q05", "control", "onboarding-checklist", "", "What is the code of the welcome pack for new hires?", "8801"),
    ("q06", "pointer", "sensor-calibration", "references/intervals.md", "After how many operating hours must a Vantor probe be recalibrated?", "4185"),
    ("q07", "unnamed", "report-builder", "references/rows.md", "What is the maximum number of rows in a Marlow report export?", "12673"),
    ("q08", "nested", "audit-export", "references/formats/csv.md", "How many rows are in one chunk of a Halden CSV audit export?", "6082"),
    ("q09", "multi", "label-printing", "references/margins.md", "What margin, in microns, does a Brindle label keep?", "9407"),
    ("q10", "control", "holiday-calendar", "", "How many minutes is the floating holiday balance of an employee per year?", "3190"),
    ("q11", "multi", "vault-rotation", "references/limits.md", "How many active keys can a Sable vault hold at most?", "3658"),
    ("q12", "multi", "label-printing", "references/ribbons.md", "How long, in mm, is one Brindle ribbon roll?", "5216"),
))  # fmt: skip


def skill_md(name: str, desc: str, body: str) -> str:
    return f"---\nname: {name}\ndescription: {json.dumps(desc)}\n---\n\n# {name}\n\n{body}\n"


def build_skills(root: Path) -> None:
    """Write every skill of ``SKILLS`` under *root* (one folder each)."""
    for name, desc, body, files in SKILLS:
        (root / name).mkdir()
        (root / name / "SKILL.md").write_text(
            skill_md(name, desc, body), encoding="utf-8"
        )
        for rel, text in files.items():
            (root / name / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / name / rel).write_text(
                f"# {Path(rel).stem}\n\n{text}\n", encoding="utf-8"
            )


def squash(text: str) -> str:
    """Lower-cased *text* without commas and whitespace ("7,341 KB" -> "7341kb")."""
    return re.sub(r"[,\s]", "", text).lower()


def _called(trace: list[dict[str, Any]], tool: str, q: Question, exact: bool) -> bool:
    return any(
        c["tool"] == tool and c["params"].get("name") == q.skill
        and (not exact or str(c["params"].get("path", "")).strip() == q.path)
        for c in trace
    )  # fmt: skip


def grade(q: Question, answer: str, trace: list[dict[str, Any]]) -> dict[str, Any]:
    """G1 (primary): token in the answer AND skill_load of the target AND, for a
    non-control, skill_read of the exact path. G2: token only. G3: the exact
    skill_read (None for a control). G4: G1 and no other question's token."""
    said = squash(answer)
    g2 = q.token in said
    g3 = None if q.control else _called(trace, "skill_read", q, True)
    g1 = g2 and _called(trace, "skill_load", q, False) and (q.control or bool(g3))
    return {"g1": g1, "g2": g2, "g3": g3,
            "g4": g1 and not any(o.token in said for o in QUESTIONS if o.id != q.id)}  # fmt: skip


def shape(
    q: Question, answer: str, trace: list[dict[str, Any]], g: dict, stop: Any
) -> str:
    """Failure shape: pass, budget, wrong skill, wrong file, load-loop, other
    (token said, trace incomplete), invented number, refusal."""
    mine = [c for c in trace if c["params"].get("name") == q.skill]
    reads = [c for c in mine if c["tool"] == "skill_read"]
    loads = [c for c in mine if c["tool"] == "skill_load"]
    for found, label in (
        (g["g1"], "pass"), (stop not in ("answered", "evidence"), "budget"),
        (not mine, "wrong skill"), (reads and not q.control and not g["g3"], "wrong file"),
        (len(loads) >= 3 and not reads, "load-loop"), (g["g2"], "other"),
    ):  # fmt: skip
        if found:
            return label
    return "invented number" if re.search(r"\d", answer) else "refusal"


# --- END FIXTURE -------------------------------------------------------------


def fixture_sha256() -> str:
    text = Path(__file__).read_text(encoding="utf-8")
    start, end = text.index("# --- BEGIN FIXTURE"), text.index("# --- END FIXTURE")
    return hashlib.sha256(text[start:end].encode("utf-8")).hexdigest()


def check_fixture(catalog: Any) -> list[str]:
    """Register-time problems of the fixture; empty means it can be run.

    Each token sits in exactly its own file (the SKILL.md for a control), never
    in ``render_catalog()`` or its question, and no token contains another; the
    target path is in ``skill.resources``; the loader reports no issue.
    """
    from fsm_llm.constants import SKILL_READ_HINT

    shown = catalog.render_catalog()
    problems = [f"loader issue: {i}" for i in catalog.issues]
    problems += ["the read hint is in render_catalog()"] * (SKILL_READ_HINT in shown)
    texts = {(n, "SKILL.md"): skill_md(n, d, b) for n, d, b, _ in SKILLS}
    texts |= {(n, rel): t for n, _, _, fs in SKILLS for rel, t in fs.items()}
    for q in QUESTIONS:
        homes = [key for key, text in texts.items() if q.token in squash(text)]
        skill = catalog.get(q.skill)
        for bad, why in (
            (skill is None, "no such skill"),
            (homes != [(q.skill, q.path or "SKILL.md")], f"token is in {homes}"),
            (q.token in squash(shown) or q.token in squash(q.text), "token is in the catalog or question"),
            (q.path and skill and q.path not in skill.resources, "path is not in skill.resources"),
            (any(o.id != q.id and q.token in o.token for o in QUESTIONS), "token is inside another token"),
        ):  # fmt: skip
            problems += [f"{q.id}: {why}"] * bool(bad)
    return problems


# --- Instruments -------------------------------------------------------------


@contextmanager
def hint_ablation(hint_on: bool) -> Iterator[None]:
    """``hint_off``: drop exactly the ``SKILL_READ_HINT`` line from the first
    ``skill_load`` page for the duration of the block, then restore."""
    # DECISION plan-2026-10-23T113904-93d78536/D-013
    # Do NOT ablate with `SKILL_READ_HINT = ""` in constants: it leaves an empty
    # line in the reply and shrinks the file-name budget by the hint's 76
    # characters, so the two arms would differ by more than the one line.
    # Wrapping `_first_page` removes only that line (the tests pin it).
    if hint_on:
        yield
        return
    import fsm_llm.agents.agent_skills as adapter
    from fsm_llm.constants import SKILL_READ_HINT

    original = adapter._first_page
    adapter._first_page = lambda skill, spare: [
        line for line in original(skill, spare) if line != SKILL_READ_HINT
    ]
    try:
        yield
    finally:
        adapter._first_page = original


@contextmanager
def wire_spy() -> Iterator[list[dict[str, Any]]]:
    """Record every request and response at core's one send binding
    (``fsm_llm.llm.completion``, the binding ``hb.captured_wire`` replaces)
    while still sending. Yields the list; restores the binding on exit."""
    import fsm_llm.llm as core_llm

    seen: list[dict[str, Any]] = []
    original = core_llm.completion

    def spy(**params: Any) -> Any:
        seen.append(record := {"params": copy.deepcopy(params)})
        response = original(**params)
        dump = getattr(response, "model_dump", None)
        record["response"] = dump() if callable(dump) else str(response)
        return response

    core_llm.completion = spy
    try:
        yield seen
    finally:
        core_llm.completion = original


def spy_registry(registry: Any, sink: list[dict[str, Any]]) -> None:
    """Record every tool call of *registry* (name, arguments, reply) in *sink*."""
    original = registry.execute

    def execute(tool_call: Any, **kwargs: Any) -> Any:
        result = original(tool_call, **kwargs)
        sink.append({"tool": tool_call.tool_name, "params": dict(tool_call.parameters), "ok": bool(result.success),
                     "reply": str(result.result if result.success else result.error)})  # fmt: skip
        return result

    registry.execute = execute


def run_trial(
    arm: str, q: Question, trial: int, catalog: Any, raw: Path | None = None
) -> dict[str, Any]:
    """One fresh agent on one question; an exception is a failed row.

    Interface contract (callers: ``run_block``, the tests, the live smoke):
    returns the row (grades, shape, trace with arguments, answer[:500], meter
    fields, ``hint_seen`` = the hint is in a request message at the send
    binding, ``hint_in_reply``, ``hint_valid``, ``first_load_reply``[:1500]);
    writes the raw requests and responses to *raw* when given. Never raises.
    """
    from fsm_llm.agents import AgentConfig, create_agent
    from fsm_llm.agents.agent_skills import create_skill_tools
    from fsm_llm.constants import SKILL_READ_HINT

    kind, model, hint_on = ARMS[arm]
    llm, registry, trace = ab.metered_interface(model), create_skill_tools(catalog), []
    spy_registry(registry, trace)
    answer, error, result, started = "", None, None, time.perf_counter()
    with hint_ablation(hint_on), wire_spy() as wire:
        try:
            agent = create_agent(
                kind,
                registry,
                config=AgentConfig(model=model, **LIMITS),
                llm_interface=llm,
            )
            result = agent.run(catalog.render_catalog() + "\n\n" + q.text)
            answer = str(result.answer or "")
        except Exception as exc:  # a crashed or timed-out trial is a failed row
            error = f"{type(exc).__name__}: {exc}"[:300]
    latency = time.perf_counter() - started
    stop = "error" if error else getattr(result, "stop_reason", None)
    loads = [c for c in trace if c["tool"] == "skill_load"]
    expected = hint_on and any(
        (s := catalog.get(c["params"].get("name"))) and s.resources for c in loads
    )
    seen = any(
        SKILL_READ_HINT in json.dumps(w["params"].get("messages"), default=str)
        for w in wire
    )
    in_reply = any(SKILL_READ_HINT in c["reply"] for c in loads)
    g = grade(q, answer, trace)
    row = {
        "task_id": q.id, "kind": q.kind, "control": q.control, "arm": arm, "trial": trial, **g,
        "shape": shape(q, answer, trace, g, stop), "stop_reason": stop, "error": error,
        "success": bool(getattr(result, "success", False)), "answer": answer[:500],
        "trace": [{k: c[k] for k in ("tool", "params", "ok")} for c in trace],
        "iterations": int(getattr(getattr(result, "trace", None), "total_iterations", 0) or 0),
        **ab.usage_fields(llm.usage()), "latency_s": round(latency, 3),
        "hint_seen": seen, "hint_in_reply": in_reply, "hint_valid": seen == expected == in_reply,
        "first_load_reply": loads[0]["reply"][:1500] if loads else "",
    }  # fmt: skip
    if raw is not None:
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(
            json.dumps({"row": row, "wire": wire, "tools": trace}, default=str)
        )
    return row


# --- Manifest, register, run --------------------------------------------------


def _paths(bdir: Path, arm: str) -> tuple[Path, Path, Path]:
    return (
        bdir / f"manifest_{arm}.json",
        bdir / f"rows_{arm}.jsonl",
        bdir / f"summary_{arm}.json",
    )


def plan(block: str) -> list[tuple[str, Question, int]]:
    """The fixed run order of a block: (arm, question, trial) triples.

    Arms sharing an agent and a model form one group; groups run one after the
    other (no model reloads); inside a group the arms alternate for each
    question and trial and the first arm flips from question to question."""
    groups: dict[tuple[str, str], list[str]] = {}
    for arm in BLOCKS[block]:
        groups.setdefault(ARMS[arm][:2], []).append(arm)
    return [
        (a, q, t)
        for grp in groups.values()
        for i, q in enumerate(QUESTIONS)
        for t in range(1, TRIALS + 1)
        for a in grp[:: 1 - i % 2 * 2]
    ]


def build_manifest(bench_id: str, block: str, arm: str) -> dict[str, Any]:
    kind, model, hint_on = ARMS[arm]
    return {
        "bench_id": bench_id, "block": block, "model": model, "trials": TRIALS,
        "arm": {"name": arm, "agent": kind, "hint": hint_on}, "order": "model-major, arms of one model alternate" if block == "B1" else "question-major, arms alternate, first arm flips",
        "n_questions": len(QUESTIONS), "limits": dict(LIMITS), "fixture_hash": fixture_sha256(),
        "src_sha256": {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in SRC_PINS},
        "model_digest": hb._model_digest(model.split("/", 1)[1]),
        "git_commit": hb._git_commit(), "created_at": hb._utc_now(),
    }  # fmt: skip


def fixture_catalog(root: Path) -> Any:
    """Build the skills under *root*, load them, refuse a broken fixture."""
    from fsm_llm.agent_skills import load_skills

    build_skills(root)
    catalog = load_skills(root)
    if problems := check_fixture(catalog):
        raise hb.BenchDataError("fixture: " + "; ".join(problems))
    return catalog


def register_block(bench_id: str, block: str) -> list[Path]:
    """Write every arm's manifest of *block* (nothing else), once."""
    bdir = BENCH_DATA / bench_id / block
    if existing := [p for a in BLOCKS[block] for p in _paths(bdir, a) if p.exists()]:
        raise hb.BenchDataError(f"{existing[0]} exists -- a block is registered ONCE")
    with tempfile.TemporaryDirectory(prefix="skills-bench-") as tmp:
        fixture_catalog(Path(tmp))
    bdir.mkdir(parents=True, exist_ok=True)
    for arm in BLOCKS[block]:
        hb._write_json(_paths(bdir, arm)[0], build_manifest(bench_id, block, arm))
    return [_paths(bdir, arm)[0] for arm in BLOCKS[block]]


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every summary number from raw rows alone. Unit: the first-trial row of
    each read-needed question (10 per arm); controls are reported apart."""
    by_q: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_q.setdefault(row["task_id"], []).append(row)
    first = {q: min(rs, key=lambda r: r["trial"]) for q, rs in by_q.items()}
    need = [r for r in first.values() if not r["control"]]
    ctrl = [r for r in first.values() if r["control"]]
    k = {g: sum(bool(r[g]) for r in need) for g in ("g1", "g2", "g3", "g4")}
    return {
        "n_rows": len(rows), "n_units": len(need), **{f"k_{g}": v for g, v in k.items()},
        "wilson_g1": [round(x, 4) for x in hb.wilson_ci(k["g1"], len(need))],
        "k_pass_hat2": sum(not first[q]["control"] and len(rs) >= TRIALS and all(r["g1"] for r in rs) for q, rs in by_q.items()),
        "controls_n": len(ctrl), "controls_k_g1": sum(bool(r["g1"]) for r in ctrl),
        "invalid_cells": sum(not r["hint_valid"] for r in need), "invalid_rows": sum(not r["hint_valid"] for r in rows),
        "shapes": dict(sorted(Counter(r["shape"] for r in rows).items())),
    }  # fmt: skip


def write_summary(bdir: Path, arm: str, status: str, started: str, run: dict) -> dict:
    """Summary from the arm's raw rows; ``status`` is complete or aborted: reason."""
    _, rows_path, summary_path = _paths(bdir, arm)
    summary = {"arm": arm, "status": status, "started_at": started, "finished_at": hb._utc_now(),
               "run": run, "metrics": metrics(hb.read_rows(rows_path))}  # fmt: skip
    hb._write_json(summary_path, summary)
    return summary


def run_block(bench_id: str, block: str) -> dict[str, dict[str, Any]]:
    """LIVE. Refuse a block with rows, an unregistered arm or a drifted manifest;
    run the fixed ``plan``; summarise each arm (complete, or aborted with reason)."""
    bdir, arms = BENCH_DATA / bench_id / block, BLOCKS[block]
    for arm in arms:
        manifest_path, rows_path, summary_path = _paths(bdir, arm)
        if rows_path.exists() or summary_path.exists():
            raise hb.BenchDataError(f"{rows_path} exists -- a block is run ONCE")
        if not manifest_path.is_file():
            raise hb.BenchDataError(f"{manifest_path} missing -- register first")
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if drift := hb.pinned_drift(
            recorded, build_manifest(bench_id, block, arm), PINNED_AT_RUN
        ):
            raise hb.BenchDataError(f"{arm} was registered with different {drift}")
    started, ran, status, streak, todo = (
        hb._utc_now(),
        hb.run_commit(),
        "aborted: not finished",
        0,
        plan(block),
    )
    try:
        with tempfile.TemporaryDirectory(prefix="skills-bench-") as tmp:
            catalog = fixture_catalog(Path(tmp))
            for done, (arm, q, trial) in enumerate(todo, 1):
                raw = RAW_DIR / bench_id / block / f"{arm}__{q.id}__t{trial}.json"
                row = run_trial(arm, q, trial, catalog, raw)
                row.update(bench_id=bench_id, block=block, ts=hb._utc_now())
                hb.append_row(_paths(bdir, arm)[1], row)
                print(f"  {block}/{arm} {done}/{len(todo)} {q.id} t{trial}: {row['shape']} g1={row['g1']} "
                      f"hint_seen={row['hint_seen']} calls={row['llm_calls']} {row['latency_s']}s", flush=True)  # fmt: skip
                streak = streak + 1 if row["error"] else 0
                if streak >= OUTAGE_ROWS:
                    raise hb.BenchDataError(
                        f"{OUTAGE_ROWS} consecutive errored rows (provider outage?)"
                    )
        status = "complete"
    except BaseException as exc:  # keep the rows, summarise, re-raise
        status = f"aborted: {type(exc).__name__}: {exc}"[:200]
        raise
    finally:
        out = {a: write_summary(bdir, a, status, started, ran) for a in arms}
        print(f"wrote summaries for {block} (status: {status})")
    return out


# --- Report -------------------------------------------------------------------


def decide_b0(
    on: dict[str, Any], off: dict[str, Any], complete: bool
) -> tuple[str, str]:
    """The pre-registered B0 rule: (verdict, action) from the two arms' metrics.

    Interface contract: the rule text is PRE_REGISTRATION_B0.md. An incomplete
    block (aborted, or fewer than 10 units) or one with more than 2 invalid
    primary cells is invalid and the hint stays. diff = on - off in first-trial
    G1 cells; FLOOR first (D-032): both arms <= 1 of 10 -> inconclusive, KEEP
    (the run never reached skill_read, so the hint could not act); Fisher
    two-sided p: diff >= 4 and p < .05 -> effect shown, KEEP;
    -diff >= 4 and p < .05 -> harmful, REMOVE; |diff| <= 1 -> no effect shown,
    REMOVE; anything else -> inconclusive, KEEP.
    """
    if not complete or on["n_units"] != PRIMARY_N or off["n_units"] != PRIMARY_N:
        return "block incomplete", "KEEP"
    if on["invalid_cells"] + off["invalid_cells"] > 2:
        return "block invalid (more than 2 invalid cells)", "KEEP"
    diff = on["k_g1"] - off["k_g1"]
    p = hb.fisher_exact_two_sided(on["k_g1"], PRIMARY_N, off["k_g1"], PRIMARY_N)
    at = f"(diff {diff}, p={p:.4f})"
    if max(on["k_g1"], off["k_g1"]) <= 1:
        return f"floor, inconclusive {at}; both arms <= 1 of 10, the runs did not reach skill_read, so the hint could not act", "KEEP"  # fmt: skip
    if diff >= 4 and p < 0.05:
        return f"effect shown {at}", "KEEP"
    if -diff >= 4 and p < 0.05:
        return f"hint harmful {at}", "REMOVE"
    return (
        (f"no effect shown {at}", "REMOVE")
        if abs(diff) <= 1
        else (f"inconclusive {at}", "KEEP")
    )


def report(bench_id: str, blocks: list[str] | None = None) -> int:
    """Recount every arm from RAW rows, cross-check the committed summaries,
    apply the B0 rule. 0 ok; 1 on a mismatch."""
    bench_dir, ok = BENCH_DATA / bench_id, True
    if not bench_dir.is_dir():
        raise hb.BenchDataError(f"no such bench: {bench_dir}")
    for block in blocks or sorted(d.name for d in bench_dir.iterdir() if d.is_dir()):
        recount, complete = {}, True
        for rows_path in sorted((bench_dir / block).glob("rows_*.jsonl")):
            arm = rows_path.stem.split("_", 1)[1]
            recount[arm] = m = json.loads(json.dumps(metrics(hb.read_rows(rows_path))))
            print(f"{bench_id} {block} [{arm}] rows={m['n_rows']}: G1 first trial {m['k_g1']}/{m['n_units']} "
                  f"wilson95={m['wilson_g1']}; G2 {m['k_g2']} G3 {m['k_g3']} G4 {m['k_g4']}; pass^2 "
                  f"{m['k_pass_hat2']}; controls {m['controls_k_g1']}/{m['controls_n']}; shapes {m['shapes']}; "
                  f"invalid cells {m['invalid_cells']} rows {m['invalid_rows']}")  # fmt: skip
            summary_path = rows_path.with_name(f"summary_{arm}.json")
            committed = (
                json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.is_file()
                else None
            )
            complete &= committed is not None and committed.get("status") == "complete"
            want = (committed or {}).get("metrics", m)
            for key in sorted(want.keys() | m.keys()):
                if want.get(key) != m.get(key):
                    ok = False
                    print(
                        f"  MISMATCH {key}: committed {want.get(key)} != recount {m.get(key)}"
                    )
        if block == "B0" and {"hint_on", "hint_off"} <= recount.keys():
            verdict, action = decide_b0(
                recount["hint_on"], recount["hint_off"], complete
            )
            print(f"B0 rule: {verdict} -> {action} the hint")
        if block == "B1" and {"native_4b", "native_4b_hint_off"} <= recount.keys():
            on, off = recount["native_4b"], recount["native_4b_hint_off"]
            p = hb.fisher_exact_two_sided(
                on["k_g1"], on["n_units"], off["k_g1"], off["n_units"]
            )
            print(f"B1 native 4b hint on {on['k_g1']}/{on['n_units']} vs off {off['k_g1']}/{off['n_units']}: Fisher p={p:.4f} (informational, no gate)")  # fmt: skip
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("register", "run", "report"):
        cmd = sub.add_parser(name)
        cmd.add_argument("bench_id")
        if name == "report":
            cmd.add_argument("--blocks", nargs="+", default=None)
        else:
            cmd.add_argument("block", choices=sorted(BLOCKS))
    args = parser.parse_args(argv)
    try:
        if args.command == "register":
            print(
                *(f"registered {p}" for p in register_block(args.bench_id, args.block)),
                sep="\n",
            )
        elif args.command == "run":
            run_block(args.bench_id, args.block)
        else:
            return report(args.bench_id, args.blocks)
        return 0
    except hb.BenchDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
