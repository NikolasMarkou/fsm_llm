"""
Synthesis gates: G1 to G5 and G6a composed into one report, and stored-form checks.

G1 loads the compile as an ``FSMDefinition``, G2 is ``FSMValidator``, G3 and G4
are ``fsm_llm.flow_analysis``, G5 is ``policy.check_policy`` and G6a is the
dead-edge dry run of ``dryrun``. Every finding
becomes one ``GateFinding`` (converted, no shared base class). Nothing here
raises on a hostile compile, logs or does I/O.

# DECISION plan-2026-11-10T194139-2cefeb61/D-027
# Order is fixed. G1 and G2 first (G2 only when G1 loaded: its stage 0 repeats
# the loader's errors), then G3, G4, G6a and G5 together on the RAW compile, so one
# repair message lists everything and a G1 failure runs no typed analysis. Only
# after all pass is the abort injected, and then only G1 and G2 re-run on it.
# Do NOT run G3 or G4 on the injected form (the abort edge would read as a
# path), do NOT stop at the first failing gate (the repair loop would need one
# round per gate), and do NOT make G2 stricter than the loader.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from fsm_llm.constants import SKILL_DEFANG
from fsm_llm.definitions import FSMDefinition
from fsm_llm.flow_analysis import analyze_data_flow, analyze_liveness
from fsm_llm.okf import capped_line
from fsm_llm.validator import FSMValidator

from .dryrun import analyze_dryrun
from .exceptions import SynthesisError
from .policy import SynthesisPolicy, check_policy, inject_abort, strip_abort

GATES = ("G1", "G2", "G3", "G4", "G5", "G6a")
#: Schema findings kept per compile, and characters kept of each.
_MAX_SCHEMA_FINDINGS = 20
_MESSAGE_CHARS = 300


@dataclass(frozen=True)
class GateFinding:
    """One defect of one gate: ``gate`` is G1 to G5 or G6a, ``code`` the defect,
    ``state`` the state id it sits at (``""`` for the document). ``state`` and
    ``message`` may quote model-authored text: ``repair_text`` is the only way
    they reach a prompt."""

    gate: str
    code: str
    state: str
    message: str


@dataclass(frozen=True)
class GateReport:
    """The verdict of ``run_gates`` or ``verify_stored``.

    ``findings`` is empty on a pass. ``skipped`` names gates that did not run
    because an earlier one failed. On a pass ``definition`` is the loaded
    abort-injected procedure and ``injected`` the same as a plain dict; both
    are ``None`` otherwise.
    """

    findings: tuple[GateFinding, ...] = ()
    definition: FSMDefinition | None = None
    injected: dict[str, Any] | None = None
    skipped: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.findings

    @property
    def verdicts(self) -> dict[str, str]:
        """Each gate: ``"fail"``, ``"skipped"`` or ``"pass"``."""
        failed = {f.gate for f in self.findings}
        return {
            g: "fail" if g in failed else "skipped" if g in self.skipped else "pass"
            for g in GATES
        }

    def repair_text(self, cap: int = 2000) -> str:
        """The failures as text for a repair turn: one line per finding,
        ``G3 undeclared-read [state]: message``, at most ``cap`` characters,
        then ``and K more``. Control-stripped, one line each, and ``[ ] < >``
        defanged, so a quoted model-authored name cannot spell a fence or tag.
        A G4 ``cycle-no-exit`` at a state G3 already reported as
        ``undeclared-read`` and a G6a ``edge-unfireable`` at a state G4 already
        reported as ``shadowed-edge`` are left out (the same cause);
        ``findings`` keeps both.
        """

        def at(gate: str, code: str) -> set[str]:
            return {f.state for f in self.findings if (f.gate, f.code) == (gate, code)}

        covered = {
            ("G4", "cycle-no-exit"): at("G3", "undeclared-read"),
            ("G6a", "edge-unfireable"): at("G4", "shadowed-edge"),
        }
        shown = [
            f for f in self.findings if f.state not in covered.get((f.gate, f.code), ())
        ]

        def safe(text: str) -> str:
            return capped_line(text, _MESSAGE_CHARS).translate(SKILL_DEFANG)

        lines: list[str] = []
        used = 0
        for i, f in enumerate(shown):
            line = f"{f.gate} {safe(f.code)} [{safe(f.state)}]: {safe(f.message)}"
            if used + len(line) + 1 > cap - 30:  # room for the "and K more" line
                lines.append(f"and {len(shown) - i} more")
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)[:cap]


def _load(data: Any) -> tuple[FSMDefinition | None, list[GateFinding]]:
    """G1: the typed load, the same one ``API`` does."""
    try:
        return FSMDefinition(**data), []
    except ValidationError as e:
        rows = e.errors(include_url=False, include_input=False, include_context=False)
        found = []
        for row in rows[:_MAX_SCHEMA_FINDINGS]:
            loc = row["loc"]
            sid = str(loc[1]) if len(loc) > 1 and loc[0] == "states" else ""
            where = ".".join(str(p) for p in loc) or "document"
            found.append(
                GateFinding(
                    "G1",
                    row["type"],
                    capped_line(sid, 60),
                    capped_line(f"{where}: {row['msg']}", _MESSAGE_CHARS),
                )
            )
        return None, found
    except (TypeError, ValueError, RecursionError) as e:
        text = f"the compile cannot be loaded ({type(e).__name__})"
        return None, [GateFinding("G1", "not-loadable", "", text)]


def _structure(data: dict[str, Any]) -> list[GateFinding]:
    """G2: the validator's errors; its warnings never fail a compile."""
    errors = FSMValidator(data).validate().errors
    return [
        GateFinding("G2", "structure", "", capped_line(m, _MESSAGE_CHARS))
        for m in errors
    ]


def run_gates(
    data: Any,
    policy: SynthesisPolicy,
    *,
    inputs: Iterable[str] = (),
    return_keys: Iterable[str] = (),
) -> GateReport:
    """Run G1 to G5 and G6a on a raw compile and, on a pass, inject the abort path.

    Interface contract (callers: the compiler's repair loop, the library path,
    the runner via ``verify_stored``):
        data: the compile as parsed, any shape (never mutated). ``inputs`` are
        the key names the driver provides, ``return_keys`` the keys the
        procedure must hand back (G3 requires each guaranteed). Returns a
        ``GateReport``: every failure of G1 to G6a together (G2 to G4 and
        G6a are skipped when G1 fails), or, when all pass, the injected definition
        re-checked by G1 and G2. Never raises.
    """
    inputs, return_keys = tuple(inputs), tuple(return_keys)
    findings: list[GateFinding] = []
    skipped: tuple[str, ...] = ()
    definition, found = _load(data)
    findings += found
    if definition is None:
        skipped = ("G2", "G3", "G4", "G6a")
    else:
        findings += _structure(data)
        findings += [
            GateFinding("G3", f.code, f.state, f.message)
            for f in analyze_data_flow(
                definition, inputs=inputs, return_keys=return_keys
            )
        ]
        findings += [
            GateFinding("G4", f.code, f.state, f.message)
            for f in analyze_liveness(
                definition, inputs=inputs, max_turns=policy.max_turns
            )
        ]
        findings += [
            GateFinding("G6a", f.code, f.state, f.message)
            for f in analyze_dryrun(definition, inputs=inputs)
        ]
    findings += [
        GateFinding("G5", f.code, f.state, f.message)
        for f in check_policy(data, policy)
    ]
    if findings:
        return GateReport(tuple(findings), skipped=skipped)
    try:
        injected = inject_abort(data)
    except SynthesisError:  # G5 passed but the shape is not injectable
        text = "the abort path cannot be added to this compile"
        return GateReport((GateFinding("G5", "abort-not-injectable", "", text),))
    loaded, again = _load(injected)
    if loaded is not None:
        again = _structure(injected)
    if again:
        return GateReport(tuple(again))
    return GateReport(definition=loaded, injected=injected)


def verify_stored(
    definition: FSMDefinition | dict[str, Any],
    policy: SynthesisPolicy,
    *,
    inputs: Iterable[str] = (),
    return_keys: Iterable[str] = (),
) -> GateReport:
    """Re-verify a stored, abort-injected procedure before it runs.

    Interface contract (callers: the library path, the runner, the tools):
        definition: the stored form, a dict or the loaded ``FSMDefinition``
        (dumped with ``exclude_defaults``, the library's canonical form); any
        other value is not canonical.
        ``inputs`` are the signature's required inputs. The abort path is
        stripped (``G5 stored-not-canonical`` when it is not exactly the
        canonical one), G1 to G6a run on the stripped form (a stored dead edge
        is refused), and its re-injection must equal the stored dict
        (``G5 stored-abort-mismatch``). Returns the
        report of ``run_gates`` on a pass; never raises.
    """
    stored = (
        definition.model_dump(exclude_defaults=True)
        if isinstance(definition, BaseModel)
        else definition
    )
    stripped = strip_abort(stored)
    if stripped is None:
        text = "the stored procedure is not the canonical abort-injected form"
        return GateReport(
            (GateFinding("G5", "stored-not-canonical", "", text),),
            skipped=("G1", "G2", "G3", "G4", "G6a"),
        )
    report = run_gates(stripped, policy, inputs=inputs, return_keys=return_keys)
    if report.passed and report.injected != stored:
        text = "the stored procedure differs from its own re-injected form"
        return GateReport((GateFinding("G5", "stored-abort-mismatch", "", text),))
    return report
