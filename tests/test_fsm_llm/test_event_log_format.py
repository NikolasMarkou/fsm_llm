"""The event log format: the vocabulary, the safe encoder and the rules of the runtime layer.

Plan 21e6f00d step 1. Everything here is about bytes and vocabulary, with no clock and no
file: one event is one ASCII line ending in one newline, a hostile value cannot make the
encoder or the decoder recurse, and a source scan keeps the banned file and JSON idioms out
of ``src/fsm_llm/runtime`` (every module, including those later steps add).
"""

from __future__ import annotations

import ast
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from fsm_llm.core.definitions import FSMError
from fsm_llm.runtime import events
from fsm_llm.runtime.constants import (
    BLOB_REF_RE,
    CONVERSATION_ID_RE,
    ENVELOPE_KEYS,
    LOG_FORMAT_VERSION,
    MAX_BLOB_BYTES,
    MAX_EVENT_BYTES,
    MAX_INT_BITS,
    MAX_PAYLOAD_DEPTH,
    WINDOWS_RESERVED_NAMES,
)
from fsm_llm.runtime.events import (
    DECLARED_TYPES,
    PRODUCED_TYPES,
    EventLogError,
    EventPayloadError,
    EventType,
    LogCorruptError,
    LogLockedError,
    LogWriteError,
    TurnOutcome,
    canonical_json,
    check_depth,
    decode_event,
    encode_event,
    to_jsonable,
)

RUNTIME_DIR = Path(__file__).resolve().parents[2] / "src" / "fsm_llm" / "runtime"

#: A literal copy of the 21 types of roadmap row 1.1 (docs/2026_roadmap.md), in its order.
ROADMAP_TYPES = [
    "turn/start",
    "llm/request",
    "llm/response",
    "provider/error",
    "extraction",
    "extraction/miss",
    "intent/out_of_scope",
    "transition",
    "decision",
    "handler",
    "tool/call",
    "tool/result",
    "approval/request",
    "approval/outcome",
    "executor/start",
    "executor/result",
    "stack/push",
    "stack/pop",
    "context/record",
    "compaction/epoch",
    "turn/end",
]


def _envelope(**fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "v": 1,
        "seq": 1,
        "ts": "2026-11-17T10:00:00.000Z",
        "conversation_id": "c1",
        "frame": 0,
        "type": EventType.TURN_START,
    }
    base.update(fields)
    return base


def _deep_list(depth: int) -> list[Any]:
    """A list nested *depth* levels (depth 0 is None), built without recursion."""
    value: Any = None
    for _ in range(depth):
        value = [value]
    return value


class TestVocabulary:
    def test_the_21_strings_equal_the_roadmap_list(self) -> None:
        assert [member.value for member in EventType] == ROADMAP_TYPES
        assert len(EventType) == 21
        assert len(set(ROADMAP_TYPES)) == 21

    def test_members_are_upper_snake_and_values_are_the_wire_strings(self) -> None:
        for member in EventType:
            assert member.name == member.value.replace("/", "_").upper()

    def test_a_member_name_is_not_a_wire_string(self) -> None:
        with pytest.raises(ValueError, match="llm_request"):
            EventType.from_string("llm_request")

    def test_from_string_is_closed(self) -> None:
        for member in EventType:
            assert EventType.from_string(member.value) is member
            assert EventType.from_string(member) is member
        for bad in ("", "LLM/REQUEST", " llm/request", "llm/request\n", None, 3, "x"):
            with pytest.raises(ValueError, match="Invalid event type"):
                EventType.from_string(bad)  # type: ignore[arg-type]

    def test_the_declared_set_is_the_whole_enum(self) -> None:
        assert DECLARED_TYPES == frozenset(EventType)
        assert len(DECLARED_TYPES) == 21

    def test_the_produced_set_is_13_and_a_subset_of_the_declared_set(self) -> None:
        assert len(PRODUCED_TYPES) == 13
        assert PRODUCED_TYPES < DECLARED_TYPES
        assert {t.value for t in PRODUCED_TYPES} == {
            "turn/start",
            "turn/end",
            "llm/request",
            "llm/response",
            "provider/error",
            "tool/call",
            "tool/result",
            "context/record",
            "stack/push",
            "stack/pop",
            "executor/start",
            "executor/result",
            "transition",
        }

    def test_the_eight_declared_but_not_produced_types(self) -> None:
        assert {t.value for t in DECLARED_TYPES - PRODUCED_TYPES} == {
            "extraction",
            "extraction/miss",
            "intent/out_of_scope",
            "decision",
            "handler",
            "approval/request",
            "approval/outcome",
            "compaction/epoch",
        }

    def test_turn_outcome_is_the_four_values_and_closed(self) -> None:
        assert [m.value for m in TurnOutcome] == [
            "committed",
            "rolled_back",
            "abandoned",
            "interrupted",
        ]
        assert TurnOutcome.from_string("rolled_back") is TurnOutcome.ROLLED_BACK
        assert TurnOutcome.from_string(TurnOutcome.ABANDONED) is TurnOutcome.ABANDONED
        with pytest.raises(ValueError, match="Invalid turn outcome"):
            TurnOutcome.from_string("ROLLED_BACK")


class TestErrors:
    def test_the_family_roots_at_fsm_error(self) -> None:
        assert issubclass(EventLogError, FSMError)
        for sub in (LogCorruptError, LogLockedError, LogWriteError, EventPayloadError):
            assert issubclass(sub, EventLogError)
        assert not issubclass(EventLogError, RuntimeError)

    def test_an_error_carries_its_message_and_details(self) -> None:
        error = LogCorruptError("bad", details={"seq": 3})
        assert str(error) == "bad"
        assert error.details == {"seq": 3}


class TestConstants:
    def test_the_format_version_and_caps(self) -> None:
        assert LOG_FORMAT_VERSION == 1
        assert MAX_EVENT_BYTES == 65_536
        assert MAX_PAYLOAD_DEPTH == 32
        assert MAX_BLOB_BYTES >= MAX_EVENT_BYTES
        assert ENVELOPE_KEYS == ("v", "seq", "ts", "conversation_id", "frame", "type")

    @pytest.mark.parametrize(
        "name", ["abc_1-2", "c1", "a" * 64, "0f5a6c1e-3b7d-4c8e-9a1f-2b3c4d5e6f70"]
    )
    def test_a_usable_conversation_id_matches(self, name: str) -> None:
        assert CONVERSATION_ID_RE.fullmatch(name)

    @pytest.mark.parametrize(
        "name", ["", "Abc", "a" * 65, "../x", "a/b", "a.b", "a b", "ab\n", "é"]
    )
    def test_an_unusable_conversation_id_does_not(self, name: str) -> None:
        assert not CONVERSATION_ID_RE.fullmatch(name)

    def test_the_reserved_names_are_the_windows_devices(self) -> None:
        assert {"con", "prn", "aux", "nul", "com1", "com9", "lpt1", "lpt9"} <= (
            WINDOWS_RESERVED_NAMES
        )
        assert len(WINDOWS_RESERVED_NAMES) == 4 + 9 + 9
        assert "com0" not in WINDOWS_RESERVED_NAMES
        assert all(name == name.lower() for name in WINDOWS_RESERVED_NAMES)

    def test_a_blob_ref_is_sha256_and_64_lowercase_hex(self) -> None:
        good = "sha256:" + "0123456789abcdef" * 4
        assert BLOB_REF_RE.fullmatch(good)
        for bad in (
            good + "\n",
            good.upper(),
            "sha256:" + "A" * 64,
            "sha256:" + "a" * 63,
            "sha256:../..",
            "sha1:" + "a" * 40,
            "blob:" + good,
        ):
            assert not BLOB_REF_RE.fullmatch(bad)


class TestEncoding:
    def test_an_envelope_is_one_ascii_line_ending_in_one_newline(self) -> None:
        data = encode_event(_envelope(note="café \u2603"))
        assert data.endswith(b"\n")
        assert data.count(b"\n") == 1
        data.decode("ascii")
        assert b"\r" not in data

    def test_the_encoding_is_compact_and_sorted(self) -> None:
        data = encode_event(_envelope(b=1, a=2))
        assert b" " not in data
        assert (
            data.index(b'"a"') < data.index(b'"b"') < data.index(b'"conversation_id"')
        )
        assert data.index(b'"conversation_id"') < data.index(b'"v"')

    def test_a_str_enum_type_is_written_as_its_wire_string(self) -> None:
        line = encode_event(_envelope(type=EventType.LLM_REQUEST))
        assert json.loads(line)["type"] == "llm/request"

    @pytest.mark.parametrize(
        "text",
        [
            "a\u2028b",
            "a\u0085b",
            "a\x1cb",
            "a\r\nb",
            "a\nb",
            "lone \ud800 surrogate",
            "emoji \U0001f600",
            "cp1252 cannot encode \u4e2d\u6587 \u2603",
            "",
        ],
    )
    def test_hostile_text_round_trips_in_one_line(self, text: str) -> None:
        data = encode_event(_envelope(text=text))
        assert len(data.split(b"\n")) == 2  # the line and the empty rest
        assert data.split(b"\n")[0] != b""
        assert decode_event(data.split(b"\n")[0])["text"] == text
        assert decode_event(data)["text"] == text  # a trailing newline is allowed

    def test_non_finite_floats_and_unjsonable_leaves_are_refused(self) -> None:
        for value in (float("nan"), float("inf"), {1, 2}, b"raw", object()):
            with pytest.raises(EventPayloadError, match="not JSON serializable"):
                encode_event(_envelope(x=value))

    def test_a_missing_envelope_key_is_refused(self) -> None:
        for key in ENVELOPE_KEYS:
            envelope = _envelope()
            del envelope[key]
            with pytest.raises(EventPayloadError, match=key):
                encode_event(envelope)
        with pytest.raises(EventPayloadError):
            encode_event([1, 2])  # type: ignore[arg-type]

    def test_the_size_cap_counts_the_newline(self) -> None:
        base = len(encode_event(_envelope(pad="")))
        fits = encode_event(_envelope(pad="x" * (MAX_EVENT_BYTES - base)))
        assert len(fits) == MAX_EVENT_BYTES
        with pytest.raises(EventPayloadError, match="blob"):
            encode_event(_envelope(pad="x" * (MAX_EVENT_BYTES - base + 1)))

    def test_a_100000_deep_value_is_refused_without_recursion(self) -> None:
        deep = _deep_list(100_000)
        with pytest.raises(EventPayloadError, match="deeper"):
            encode_event(_envelope(x=deep))

    def test_depth_is_bounded_at_the_cap(self) -> None:
        # the envelope is depth 1, so the payload gets MAX_PAYLOAD_DEPTH - 1 more levels
        encode_event(_envelope(x=_deep_list(MAX_PAYLOAD_DEPTH - 1)))
        with pytest.raises(EventPayloadError):
            encode_event(_envelope(x=_deep_list(MAX_PAYLOAD_DEPTH)))

    def test_a_cycle_is_refused(self) -> None:
        loop: list[Any] = []
        loop.append(loop)
        with pytest.raises(EventPayloadError):
            encode_event(_envelope(x=loop))

    def test_check_depth_runs_before_json_dumps(self, monkeypatch) -> None:
        def boom(*_args: Any, **_kwargs: Any) -> str:
            raise AssertionError("json.dumps reached before the depth check")

        monkeypatch.setattr(events.json, "dumps", boom)
        with pytest.raises(EventPayloadError, match="deeper"):
            encode_event(_envelope(x=_deep_list(MAX_PAYLOAD_DEPTH + 5)))


class TestDecoding:
    def test_a_written_event_decodes_to_itself(self) -> None:
        envelope = _envelope(type="llm/request", payload={"a": [1, 2, {"b": None}]})
        assert decode_event(encode_event(envelope)) == envelope

    def test_a_line_over_the_cap_is_refused_before_json_loads(
        self, monkeypatch
    ) -> None:
        def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("json.loads reached before the size check")

        monkeypatch.setattr(events.json, "loads", boom)
        with pytest.raises(LogCorruptError, match="byte cap"):
            decode_event(b"x" * (MAX_EVENT_BYTES + 1))
        with pytest.raises(LogCorruptError, match="byte cap"):
            decode_event(b"[" * 10_000_000)

    def test_a_10_mb_line_without_a_newline_is_refused_by_size(self) -> None:
        with pytest.raises(LogCorruptError, match="byte cap"):
            decode_event(b"{" + b" " * (10 * 1024 * 1024))

    def test_a_deep_line_within_the_byte_cap_is_corruption_not_a_crash(self) -> None:
        # 30,000 levels is 60,000 bytes, under the cap: Python 3.11 gives up in json.loads
        # (RecursionError), 3.14 and 3.15 parse it; both end as LogCorruptError.
        depth = 30_000
        head = (
            b'{"conversation_id":"c","frame":0,"seq":1,"ts":"t","type":"x","v":1,"x":'
        )
        line = head + b"[" * depth + b"]" * depth + b"}"
        assert len(line) < MAX_EVENT_BYTES
        with pytest.raises(LogCorruptError):
            decode_event(line)

    def test_a_line_just_over_the_depth_cap_is_corruption(self) -> None:
        head = (
            b'{"conversation_id":"c","frame":0,"seq":1,"ts":"t","type":"x","v":1,"x":'
        )
        ok = (
            head
            + b"[" * (MAX_PAYLOAD_DEPTH - 1)
            + b"]" * (MAX_PAYLOAD_DEPTH - 1)
            + b"}"
        )
        decode_event(ok)
        over = head + b"[" * MAX_PAYLOAD_DEPTH + b"]" * MAX_PAYLOAD_DEPTH + b"}"
        with pytest.raises(LogCorruptError, match="deeper"):
            decode_event(over)

    def test_memory_error_and_recursion_error_are_corruption(self, monkeypatch) -> None:
        for error in (MemoryError, RecursionError):

            def fail(*_args: Any, _error: type = error, **_kwargs: Any) -> Any:
                raise _error()

            monkeypatch.setattr(events.json, "loads", fail)
            with pytest.raises(LogCorruptError, match=error.__name__):
                decode_event(b"{}")

    @pytest.mark.parametrize(
        "line",
        [b"", b"not json", b"[]", b"3", b'"s"', b"null", b"{", b"{}", "é".encode()],
    )
    def test_a_bad_line_is_corruption(self, line: bytes) -> None:
        with pytest.raises(LogCorruptError):
            decode_event(line)

    def test_a_non_ascii_line_is_corruption(self) -> None:
        line = encode_event(_envelope(text="x")).replace(b'"x"', '"é"'.encode())
        with pytest.raises(LogCorruptError):
            decode_event(line)

    @pytest.mark.parametrize("version", [0, 2, 99, -1, True, "1", 1.0, None])
    def test_an_unsupported_version_is_refused(self, version: Any) -> None:
        line = json.dumps(_envelope(v=version, type="x")).encode()
        with pytest.raises(LogCorruptError, match="version"):
            decode_event(line)

    def test_a_missing_envelope_key_is_corruption(self) -> None:
        for key in ENVELOPE_KEYS:
            envelope = _envelope(type="x")
            del envelope[key]
            with pytest.raises(LogCorruptError, match=key):
                decode_event(json.dumps(envelope).encode())


class TestCheckDepth:
    def test_scalars_and_flat_containers(self) -> None:
        assert check_depth(1, 0)
        assert check_depth("x", 0)
        assert check_depth([], 1)
        assert not check_depth([], 0)
        assert check_depth({"a": [1]}, 2)
        assert not check_depth({"a": [1]}, 1)

    def test_a_tuple_counts_as_a_container(self) -> None:
        assert not check_depth(((1,),), 1)
        assert check_depth(((1,),), 2)

    def test_dict_keys_are_not_walked(self) -> None:
        assert check_depth({(1, (2, (3,))): 1}, 1)

    def test_the_limit_is_exact_for_a_deep_list(self) -> None:
        assert check_depth(_deep_list(MAX_PAYLOAD_DEPTH))
        assert not check_depth(_deep_list(MAX_PAYLOAD_DEPTH + 1))
        assert not check_depth(_deep_list(100_000))

    def test_a_cycle_through_a_dict_and_a_list_is_too_deep(self) -> None:
        a: dict[str, Any] = {}
        a["x"] = [a]
        assert not check_depth(a)

    def test_shared_children_cost_distinct_nodes_not_paths(self) -> None:
        # 30 levels, two references to the level below: 2**30 paths, 30 nodes
        node: list[Any] = []
        for _ in range(MAX_PAYLOAD_DEPTH - 2):
            node = [node, node]
        assert check_depth(node)

    def test_a_node_first_seen_shallow_is_rechecked_when_met_deeper(self) -> None:
        shared = _deep_list(MAX_PAYLOAD_DEPTH - 2)
        # reached at depth 2 and again at depth MAX_PAYLOAD_DEPTH - 1: the second path is too deep
        tall: Any = [shared]
        for _ in range(MAX_PAYLOAD_DEPTH - 3):
            tall = [tall]
        assert not check_depth([shared, tall])
        assert check_depth([shared])


class TestToJsonable:
    def test_json_values_are_copied_equal(self) -> None:
        value = {"a": [1, 2.5, "x", None, True], "b": {"c": {}}}
        out = to_jsonable(value)
        assert out == value
        assert out is not value and out["a"] is not value["a"]

    def test_a_scalar_is_returned_as_is(self) -> None:
        assert to_jsonable(3) == 3
        assert to_jsonable("s") == "s"
        assert to_jsonable(None) is None

    def test_a_tuple_becomes_a_list(self) -> None:
        assert to_jsonable((1, (2, 3))) == [1, [2, 3]]

    def test_key_order_is_kept(self) -> None:
        out = to_jsonable({"z": 1, "a": 2, "m": {"y": 1, "b": 2}})
        assert list(out) == ["z", "a", "m"]
        assert list(out["m"]) == ["y", "b"]

    def test_datetime_decimal_uuid_become_text(self) -> None:
        import datetime
        import decimal
        import uuid

        stamp = datetime.datetime(2026, 11, 17, 10, 0, tzinfo=datetime.UTC)
        uid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        out = to_jsonable(
            {
                "t": stamp,
                "d": datetime.date(2026, 11, 17),
                "n": decimal.Decimal("1.50"),
                "u": uid,
            }
        )
        assert out == {
            "t": "2026-11-17T10:00:00+00:00",
            "d": "2026-11-17",
            "n": "1.50",
            "u": str(uid),
        }

    def test_other_leaves_become_a_typed_marker_never_a_repr(self) -> None:
        class Holder:
            def __repr__(self) -> str:
                return "SECRET-IN-REPR"

            def __str__(self) -> str:
                return "SECRET-IN-STR"

        out = to_jsonable(
            {
                "s": {1, 2},
                "b": b"abc",
                "ba": bytearray(5),
                "o": Holder(),
                "f": float("nan"),
            }
        )
        assert out == {
            "s": "<set>",
            "b": "<bytes:3>",
            "ba": "<bytes:5>",
            "o": "<Holder>",
            "f": "<float:nan>",
        }
        assert "SECRET" not in json.dumps(out)

    def test_keys_that_are_not_strings_become_strings(self) -> None:
        out = to_jsonable({1: "a", 2.5: "b", None: "c", False: "d", (1,): "e"})
        assert set(out) == {"1", "2.5", "None", "False", "<tuple>"}

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(1 << 5000, id="power-of-two"),
            pytest.param(-(1 << 5000), id="negative"),
            pytest.param(10**4400, id="power-of-ten"),
            pytest.param(math.factorial(2000), id="factorial"),
        ],
    )
    def test_an_int_too_big_for_json_text_is_a_typed_marker_value_or_key(
        self, value: int
    ) -> None:
        marker = f"<int:{value.bit_length()} bits>"
        out = to_jsonable({"n": value, value: 1, "l": [value]})
        assert out == {"n": marker, marker: 1, "l": [marker]}
        canonical_json(out)  # the encoder never raises for an honest Python int

    def test_an_int_within_the_bound_stays_under_the_smallest_digit_limit(self) -> None:
        small, before = (1 << MAX_INT_BITS) - 1, sys.get_int_max_str_digits()
        sys.set_int_max_str_digits(640)  # the smallest limit Python accepts
        try:
            data = to_jsonable({"n": small, "t": True, "f": False})
            assert data == {"n": small, "t": True, "f": False}
            assert json.loads(canonical_json(data)) == data
        finally:
            sys.set_int_max_str_digits(before)
        assert to_jsonable(1 << MAX_INT_BITS) == f"<int:{MAX_INT_BITS + 1} bits>"

    def test_a_cycle_is_cut(self) -> None:
        loop: dict[str, Any] = {"name": "x"}
        loop["self"] = loop
        assert to_jsonable(loop) == {"name": "x", "self": "<cycle>"}
        shared = [1]
        assert to_jsonable([shared, shared]) == [[1], [1]]  # shared is not a cycle

    def test_a_value_deeper_than_the_cap_is_cut_without_recursion(self) -> None:
        out = to_jsonable(_deep_list(100_000))
        assert check_depth(out, MAX_PAYLOAD_DEPTH - 1)
        assert not check_depth(out, MAX_PAYLOAD_DEPTH - 2)  # cut exactly at the default
        assert "<truncated:depth>" in json.dumps(out)

    def test_max_depth_is_a_parameter(self) -> None:
        out = to_jsonable(_deep_list(50), max_depth=3)
        assert check_depth(out, 3) and not check_depth(out, 2)

    def test_the_result_encodes(self) -> None:
        hostile = {"x": _deep_list(100_000), "s": {1}, "b": b"1", 7: float("inf")}
        line = encode_event(_envelope(payload=to_jsonable(hostile)))
        assert decode_event(line)["payload"]["s"] == "<set>"

    def test_a_cyclic_value_with_two_references_per_level_terminates(self) -> None:
        a: list[Any] = []
        b: list[Any] = [a, a]
        a.extend([b, b])
        out = to_jsonable(a)
        assert check_depth(out, MAX_PAYLOAD_DEPTH - 1)


# ---------------------------------------------------------------------------------------------
# The source scan: banned idioms in every module of the runtime layer.
# ---------------------------------------------------------------------------------------------


def _violations(source: str) -> list[str]:
    """The banned idioms in *source* as ``"<line> <idiom>"`` strings (empty when clean).

    Banned in the runtime layer: ``indent=`` (``json.dumps`` of a deep value allocates
    gigabytes on 3.14 and 3.15), ``default=str`` (hides an object's ``str`` in the log),
    a text-mode builtin ``open`` (newline translation and the locale codec on Windows),
    ``Path.read_text`` and ``write_text`` (the same), ``splitlines`` (it splits on U+2028
    and U+0085, which would cut one event into three) and ``perf_counter`` (the cpu_timer
    scan of ``test_cpu_timer.py`` bans it outside its list).
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.keyword):
            if node.arg == "indent":
                found.append(f"{node.value.lineno} indent=")
            if node.arg == "default" and (
                isinstance(node.value, ast.Name) and node.value.id == "str"
            ):
                found.append(f"{node.value.lineno} default=str")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "open":
                mode: ast.expr | None = node.args[1] if len(node.args) > 1 else None
                for keyword in node.keywords:
                    if keyword.arg == "mode":
                        mode = keyword.value
                binary = (
                    isinstance(mode, ast.Constant)
                    and isinstance(mode.value, str)
                    and "b" in mode.value
                )
                if not binary:
                    found.append(f"{node.lineno} text-mode open")
        if isinstance(node, ast.Attribute) and node.attr in {
            "splitlines",
            "perf_counter",
            "read_text",
            "write_text",
        }:
            found.append(f"{node.lineno} {node.attr}")
    return found


class TestSourceScan:
    def test_the_scan_sees_every_module_of_the_layer(self) -> None:
        names = {p.name for p in RUNTIME_DIR.rglob("*.py")}
        assert {"__init__.py", "constants.py", "events.py"} <= names

    def test_the_runtime_layer_has_no_banned_idiom(self) -> None:
        problems = {
            str(path.relative_to(RUNTIME_DIR)): _violations(
                path.read_text(encoding="utf-8")
            )
            for path in sorted(RUNTIME_DIR.rglob("*.py"))
        }
        assert {name: found for name, found in problems.items() if found} == {}

    @pytest.mark.parametrize(
        ("source", "idiom"),
        [
            ("import json\njson.dumps(x, indent=2)\n", "indent="),
            ("import json\njson.dumps(x, default=str)\n", "default=str"),
            ("open(p)\n", "text-mode open"),
            ("open(p, 'w')\n", "text-mode open"),
            ("open(p, mode='a')\n", "text-mode open"),
            ("text.splitlines()\n", "splitlines"),
            ("time.perf_counter()\n", "perf_counter"),
            ("p.read_text()\n", "read_text"),
            ("p.write_text('x')\n", "write_text"),
        ],
    )
    def test_the_scan_flags_what_it_bans(self, source: str, idiom: str) -> None:
        assert [v.split(" ", 1)[1] for v in _violations(source)] == [idiom]

    @pytest.mark.parametrize(
        "source",
        [
            "open(p, 'rb')\n",
            "open(p, mode='ab')\n",
            "os.open(p, os.O_RDONLY)\n",
            "json.dumps(x, default=_refuse)\n",
            "text.split('\\n')\n",
        ],
    )
    def test_the_scan_passes_what_it_allows(self, source: str) -> None:
        assert _violations(source) == []

    def test_the_encoder_pins_its_dumps_arguments(self) -> None:
        tree = ast.parse((RUNTIME_DIR / "events.py").read_text(encoding="utf-8"))
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "dumps"
        ]
        assert len(calls) == 1
        kwargs = {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords}
        assert kwargs == {
            "separators": (",", ":"),
            "ensure_ascii": True,
            "sort_keys": True,
            "allow_nan": False,
        }

    def test_no_runtime_module_imports_a_private_name_of_another_runtime_module(
        self,
    ) -> None:
        offenders = [
            f"{path.name}: {alias.name}"
            for path in RUNTIME_DIR.rglob("*.py")
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.ImportFrom)
            and (node.level or (node.module or "").startswith("fsm_llm.runtime"))
            for alias in node.names
            if alias.name.startswith("_")
        ]
        assert offenders == []

    def test_the_provenance_key_is_a_constant_in_each_layer_and_a_literal_nowhere_else(
        self,
    ) -> None:
        src = RUNTIME_DIR.parent
        spelled = sorted(
            path.relative_to(src).as_posix()
            for path in src.rglob("*.py")
            if any(
                isinstance(n, ast.Constant) and n.value == "pipeline_extracted"
                for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            )
        )
        assert spelled == ["engine/session.py", "runtime/stores/session.py"]

    def test_the_package_surface_names_only_things_that_exist(self) -> None:
        import fsm_llm.runtime as runtime

        assert len(runtime.__all__) == len(set(runtime.__all__))
        for name in runtime.__all__:
            assert hasattr(runtime, name), name
        assert re.fullmatch(r"[A-Za-z_]+", "".join(runtime.__all__))
