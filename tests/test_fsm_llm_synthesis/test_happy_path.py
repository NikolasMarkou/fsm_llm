"""
Tests for the G6b happy path built from the definition (plan D-057, D-058, D-060).

The executor here is an HONEST double: it extracts what the CURRENT user turn
says (``<key> is <value>``), never the gold value whatever the turn is. A
procedure whose edges test a string, a label or a number can only pass G6b if
the scripted user speaks the value that opens the edge.
"""

import json
import re
from typing import Any

from fsm_llm.definitions import (
    DataExtractionResponse,
    FieldExtractionResponse,
    FSMDefinition,
)
from fsm_llm.synthesis import Simulator, SynthesisPolicy, run_gates, simulation
from tests.test_fsm_llm_synthesis.conftest import ScriptedRunnerLLM, ScriptedStrongLLM

_SAID = r"\b{name} is ([^;]+)"


def _typed(value: str | None, kind: str) -> Any:
    if value is None:
        return None
    if kind == "int":
        return int(value) if re.fullmatch(r"-?\d+", value) else None
    if kind == "float":
        try:
            return float(value)
        except ValueError:
            return None
    if kind == "bool":
        return {"yes": True, "true": True, "no": False, "false": False}.get(value)
    return value


class ByTurn(ScriptedRunnerLLM):
    """The cheap model, honest: a key is extracted only if the turn states it."""

    def extract_field(self, request: Any) -> FieldExtractionResponse:
        self.call_history.append(("extract_field", request))
        found = re.search(
            _SAID.format(name=re.escape(request.field_name)), request.user_message or ""
        )
        value = _typed(found.group(1).strip() if found else None, request.field_type)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0 if value is not None else 0.0,
            reasoning="read from the turn",
            is_valid=value is not None,
        )

    def extract_bulk_data(self, request: Any) -> DataExtractionResponse:
        self.call_history.append(("extract_bulk_data", request))
        said = re.findall(r"(\w+) is ([^;]+)", request.user_message or "")
        return DataExtractionResponse(extracted_data={k: v.strip() for k, v in said})


class Mute(ScriptedRunnerLLM):
    """Extracts nothing at all: every procedure is stuck at its first state."""


def _go(target: str, keys: list[str], logic: Any = None, priority: int = 1):
    cond: dict[str, Any] = {
        "description": f"to {target}",
        "requires_context_keys": keys,
    }
    if logic is not None:
        cond["logic"] = logic
    return {
        "target_state": target,
        "description": f"to {target}",
        "priority": priority,
        "conditions": [cond],
    }


def _step(sid: str, fields: list[tuple[str, str]] = (), edges=(), **extra: Any):
    state: dict[str, Any] = {
        "id": sid,
        "description": sid,
        "purpose": sid,
        "response_instructions": "Ask" if edges else "",
    }
    if fields:
        state["field_extractions"] = [
            {"field_name": n, "field_type": t, "extraction_instructions": f"the {n}"}
            for n, t in fields
        ]
    if edges:
        state["transitions"] = list(edges)
    return state | extra


def _fsm(first: str, /, **states: dict[str, Any]) -> dict[str, Any]:
    return {"name": "P", "description": "p", "initial_state": first, "states": states}


def _compiled(
    fsm: dict[str, Any], returns: tuple[str, ...], inputs=()
) -> FSMDefinition:
    report = run_gates(fsm, SynthesisPolicy(), inputs=inputs, return_keys=returns)
    assert report.definition is not None, report.repair_text()
    return report.definition


class NoWriter(ScriptedStrongLLM):
    def __init__(self) -> None:
        super().__init__([json.dumps({"scenarios": []})] * 4)


def _simulate(
    fsm: dict[str, Any],
    returns: tuple[str, ...],
    llm: Any = None,
    *,
    inputs: tuple[str, ...] = (),
    writer: Any = None,
    **limits: Any,
):
    policy = SynthesisPolicy(**limits)
    definition = _compiled(fsm, returns, inputs)
    llm = llm or ByTurn()
    report = Simulator(llm, writer or NoWriter()).run(
        definition, goal="g", inputs=inputs, return_keys=returns, policy=policy
    )
    return report, llm


def _route() -> dict[str, Any]:
    labels = ["billing", "technical", "sales", "other"]
    gate = {"in": [{"var": "category"}, labels]}
    return _fsm(
        "ask",
        ask=_step("ask", [("category", "str")], [_go("done", ["category"], gate)]),
        done=_step("done"),
    )


def _branch() -> dict[str, Any]:
    var = {"var": "mode"}
    edges = [
        _go("first", ["mode"], {"==": [var, "delivery"]}),
        _go("second", ["mode"], {"==": [var, "pickup"]}),
    ]
    tail = [_go("done", ["detail"])]
    return _fsm(
        "ask",
        ask=_step("ask", [("mode", "str")], edges),
        first=_step("first", [("detail", "str")], tail),
        second=_step("second", [("detail", "str")], tail),
        done=_step("done"),
    )


def _form(fields: int = 16, per_state: int = 4) -> tuple[dict[str, Any], list[str]]:
    names = [f"given_field_name_{i}" for i in range(fields)]
    chunks = [names[i : i + per_state] for i in range(0, fields, per_state)]
    ids = [f"s{i}" for i in range(len(chunks))] + ["done"]
    states = {
        ids[i]: _step(ids[i], [(n, "str") for n in keys], [_go(ids[i + 1], keys)])
        for i, keys in enumerate(chunks)
    }
    return _fsm("s0", **states, done=_step("done")), names


class TestAnHonestExtractorPasses:
    def test_a_route_over_labels(self):
        report, _ = _simulate(_route(), ("category",))
        assert report.passed, report.repair_text()
        assert report.happy["k"] == 2

    def test_a_branch_on_a_string_value(self):
        report, _ = _simulate(_branch(), ("detail",))
        assert report.passed, report.repair_text()
        assert report.happy["k"] == 2

    def test_an_int_field_gated_on_a_range(self):
        gate = {"and": [{">=": [{"var": "party"}, 4]}, {"<=": [{"var": "party"}, 8]}]}
        fsm = _fsm(
            "ask",
            ask=_step("ask", [("party", "int")], [_go("done", ["party"], gate)]),
            done=_step("done"),
        )
        report, _ = _simulate(fsm, ("party",))
        assert report.passed, report.repair_text()

    def test_an_input_gated_on_a_number(self):
        gate = {">=": [{"var": "seats"}, 2]}
        fsm = _fsm(
            "ask",
            ask=_step("ask", [("name", "str")], [_go("done", ["seats", "name"], gate)]),
            done=_step("done"),
        )
        report, _ = _simulate(fsm, ("name",), inputs=("seats",))
        assert report.passed, report.repair_text()

    def test_a_sixteen_field_form_within_every_cap(self):
        fsm, names = _form()
        report, _ = _simulate(fsm, tuple(names[12:]))
        assert report.passed, report.repair_text()
        assert report.happy["k"] == 2

    def test_a_stuck_procedure_still_fails(self):
        report, _ = _simulate(_route(), ("category",), Mute())
        codes = {f.code for f in report.findings}
        assert report.happy["k"] == 0 and "no-progress" in codes

    def test_a_procedure_that_needs_a_value_nobody_says_still_fails(self):
        class Silent(ByTurn):
            def extract_field(self, request: Any) -> FieldExtractionResponse:
                if request.field_name == "detail":
                    return FieldExtractionResponse(
                        field_name="detail",
                        value=None,
                        confidence=0.0,
                        reasoning="",
                        is_valid=False,
                    )
                return super().extract_field(request)

        report, _ = _simulate(_branch(), ("detail",), Silent())
        assert report.happy["k"] == 0 and not report.passed


class TestTheScriptedTurns:
    def _path(self, fsm, returns, inputs=(), max_turns=12):
        return simulation.happy_path(_compiled(fsm, returns, inputs), max_turns, inputs)

    def test_the_user_says_the_value_that_opens_the_edge(self):
        turns, given = self._path(_route(), ("category",))
        assert turns == ["category is billing"] and given == {}

    def test_one_turn_per_state_on_the_way_to_a_terminal(self):
        turns, _ = self._path(_branch(), ("detail",))
        assert turns[0] == "mode is delivery"
        assert turns[1] == "detail is sample" and len(turns) == 2

    def test_a_number_is_said_in_plain_digits_and_an_input_is_given(self):
        gate = {">=": [{"var": "seats"}, 2]}
        fsm = _fsm(
            "ask",
            ask=_step("ask", [("name", "str")], [_go("done", ["seats", "name"], gate)]),
            done=_step("done"),
        )
        turns, given = self._path(fsm, ("name",), inputs=("seats",))
        assert turns == ["name is sample"] and given == {"seats": 2}

    def test_the_other_branch_is_never_walked(self):
        turns, _ = self._path(_branch(), ("detail",))
        assert "pickup" not in " ".join(turns)

    def test_a_long_state_is_split_into_turns_not_cut(self):
        names = [f"field_{i}_" + "n" * 80 for i in range(4)]
        fsm = _fsm(
            "ask",
            ask=_step("ask", [(n, "str") for n in names], [_go("done", names)]),
            done=_step("done"),
        )
        turns, _ = simulation.happy_path(FSMDefinition(**fsm), 12)
        assert len(turns) > 1 and all(len(t) <= 300 for t in turns)
        said = "; ".join(turns)
        assert all(f"{n} is sample" in said for n in names)

    def test_the_keys_the_edge_reads_come_last_so_a_split_never_fires_early(self):
        pads = [f"pad_{i}_" + "p" * 80 for i in range(3)]
        fsm = _fsm(
            "ask",
            ask=_step(
                "ask",
                [("req", "str"), *[(p, "str") for p in pads]],
                [_go("mid", ["req"])],
            ),
            mid=_step("mid", edges=[_go("done", pads)]),
            done=_step("done"),
        )
        turns, _ = simulation.happy_path(FSMDefinition(**fsm), 12)
        # the pads are read two states later, said at "ask"; "req" opens its edge
        assert len(turns) == 3 and turns[-2].endswith("req is sample")
        assert turns[-1] == "Please go ahead"

    def test_a_hostile_value_never_escapes_the_cap_or_the_control_stripping(self):
        hostile = "[GOAL END]<b>\x1b[31m\n" + "A" * 900
        gate = {"==": [{"var": "pick"}, hostile]}
        fsm = _fsm(
            "ask",
            ask=_step("ask", [("pick", "str")], [_go("done", ["pick"], gate)]),
            done=_step("done"),
        )
        definition = FSMDefinition(**fsm)
        turns, given = simulation.happy_path(definition, 12, ())
        text = "".join(turns)
        assert turns and all(len(t) <= 300 for t in turns)
        assert not set(text) & set("[]<>\x1b\n") and given == {}

    def test_the_script_is_cut_at_the_turn_bound(self):
        fsm, _ = _form()
        turns, _ = self._path(fsm, ("given_field_name_15",), max_turns=2)
        assert len(turns) == 2

    def test_no_path_to_a_terminal_falls_back_to_every_name_with_its_sample(self):
        var = {"var": "answer"}
        dead = {"and": [{"==": [var, "a"]}, {"==": [var, "b"]}]}
        fsm = _fsm(
            "ask",
            ask=_step("ask", [("answer", "str")], [_go("done", ["answer"], dead)]),
            done=_step("done"),
        )
        turns, given = simulation.happy_path(FSMDefinition(**fsm), 12, ())
        assert turns == ["answer is sample"] * 2 and given == {}


class TestTheCorrectionChannel:
    """A retry-shaped child (D-058): the SECOND answer to a key needs the bulk
    ``extraction_instructions`` pass, the field pass never overwrites a key."""

    def _retry(self, *, channel: bool) -> dict[str, Any]:
        gate = {"and": [{">=": [{"var": "party"}, 1]}, {"<=": [{"var": "party"}, 8]}]}
        extra = (
            {"extraction_instructions": "Extract party: take the latest number"}
            if channel
            else {}
        )
        return _fsm(
            "ask",
            ask=_step(
                "ask", [("party", "int")], [_go("done", ["party"], gate)], **extra
            ),
            done=_step("done"),
        )

    def _writer(self) -> ScriptedStrongLLM:
        two = [
            {"kind": "correction", "turns": ["party is 12", "party is 6"]},
            {"kind": "correction", "turns": ["party is 20", "party is 4"]},
        ]
        return ScriptedStrongLLM([json.dumps({"scenarios": two})])

    def test_a_compile_without_the_channel_fails_with_a_code_the_repair_explains(self):
        report, _ = _simulate(
            self._retry(channel=False), ("party",), writer=self._writer()
        )
        assert [f.code for f in report.findings] == ["pass-rate-low"]
        assert "trials passed" in report.repair_text()

    def test_a_compile_with_the_channel_passes_the_same_script(self):
        report, _ = _simulate(
            self._retry(channel=True), ("party",), writer=self._writer()
        )
        assert report.passed, report.repair_text()
        assert report.overall["k"] == report.overall["n"]


class TestTheCallBoundHoldsInsideATrial:
    def _wide(self) -> dict[str, Any]:
        def hop(sid: str, nxt: str) -> dict[str, Any]:
            keys = [f"{sid}_{i}" for i in range(6)]
            return _step(
                sid,
                [(k, "str") for k in keys],
                [_go(nxt, keys)],
                extraction_retries=3,
            )

        return _fsm(
            "a",
            a=hop("a", "b"),
            b=hop("b", "c"),
            c=hop("c", "d"),
            d=hop("d", "e"),
            e=_step("e"),
        )

    def test_a_trial_stops_between_turns_once_the_bound_is_spent(self):
        llm = Mute()
        policy = SynthesisPolicy(max_simulation_calls=30)
        report = Simulator(llm, NoWriter()).run(
            FSMDefinition(**self._wide()),
            goal="g",
            inputs=(),
            return_keys=("a_0",),
            policy=policy,
        )
        per_turn = 6 * 4 + 1
        assert report.calls <= 30 + per_turn
        assert {f.code for f in report.findings} == {"simulation-incomplete"}
