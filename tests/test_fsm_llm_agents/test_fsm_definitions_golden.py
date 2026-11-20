"""Golden hashes of the fragment builders in ``agents/fsm_definitions/``.

Each case calls a builder with one representative argument set and pins the
canonical JSON (``json.dumps(d, sort_keys=True)``) sha256. The full FSM cases
are pinned by ``tests/fixtures/executables/index.json`` and its text fixtures
(same sha256 and ``fsm_id``, and a readable diff); only the fragments, which
have no ``fsm_id``, keep a row here. Captured at 7cb4c86, before the dict shapes
were collapsed into helpers (plan 5980fbb7 step 13).

The case table itself lives in ``tests/executables_corpus.py`` (plan ace7a8c8,
F1.2), beside the readable text fixtures built from it.

Do NOT regenerate a value to make a failure go away: a difference means a
builder now emits a different prompt or routing contract.

Re-pinned once for F-AR-05 (plan 0068b055 step 7, D-013): ``adapt``,
``adapt_depth``, ``orchestrator`` and ``orchestrator_keys`` moved because the
six intermediate states (orchestrate, delegate, collect; attempt, assess,
decompose) lost their ``response_instructions``. The dicts were diffed before
the re-pin: only those six fields changed, nothing else.
"""

from __future__ import annotations

import json

import pytest

from fsm_llm.core.definitions import FSMDefinition, fsm_id_of
from tests.executables_corpus import CASES, INDEX_PATH, canonical_sha256

#: The FRAGMENT cases only: case name -> (sha256 of the canonical JSON, "" for a fragment).
#: Every full FSM case is pinned by ``tests/fixtures/executables/index.json`` (same sha256,
#: ``fsm_id`` and digest, plus a text fixture that shows a diff), checked by
#: ``tests/test_fsm_llm/test_executables_corpus.py``; their rows were dropped here (plan
#: ace7a8c8 step 4, G2.3) after a one by one comparison found every literal equal.
_GOLDEN: dict[str, tuple[str, str]] = {
    "await_approval_state": (
        "393e8623fb20ed25ddea4f976ed2feae6905132dc4606d18ad0ec26c485bd08a",
        "",
    ),
    "conclude_logic": (
        "0dbd0c38d9d910979b5f2eb9b3ea6370e0969b8d1eabd43b6cba0618f3b1e529",
        "",
    ),
    "conclude_logic_flag": (
        "9f44111b718849d4ed185fdab4d43b81907baa958b5883fef9703e763388dae3",
        "",
    ),
    "finalize": (
        "6f3a0eb19050a46126fcdc6ac6ec9959343609d92746a2fb26ada398b839e4d7",
        "",
    ),
    "finalize_default_description": (
        "af241c6abe316938a9f78e6908c0df0c343d93529f1c5adc53a2f7e821ef2c0e",
        "",
    ),
    "flag_condition": (
        "4c3e2fe7e763cfb8c6d187f6c717fad62551a50c4c7fd59e20816ab24641ad1b",
        "",
    ),
    "loop_field_context_keys": (
        "479befedfae79cf900993283f1fd2bc6041ceec6342de05c6413b0dcb360378c",
        "",
    ),
    "loop_field_context_keys_extra": (
        "3ab03b0c58a5c064ecedf093cfcd13f12f3c2f8af72cef3de498547a025bcecf",
        "",
    ),
    "reply_kind_condition": (
        "6d86bb241f242498f02ef25398d0c3dc5fa3d1e946854307e660c2e379553a73",
        "",
    ),
    "think_fields": (
        "63a1b85d464f03fdd089906ba900f8ecffea824a91858973fdf7cee3b651a98d",
        "",
    ),
    "think_fields_no_name": (
        "d54e2dbaa4ae50822270e6626eb9f3f0d3a3d03308041d71ab70ef7a052b70ec",
        "",
    ),
    "tool_selection": (
        "5784ab0b6a088748fae20b2c7ad23fac0cdaced80fc9cb6d4be0b5b71e1d0050",
        "",
    ),
    "tool_selection_no_name": (
        "3ab71852345a3ef8844d9212965d9da8fb80981ed8e02233ce0c53dd8bd912ba",
        "",
    ),
}


def _measure(name: str) -> tuple[str, str]:
    build, is_fsm = CASES[name]
    value = build()
    return canonical_sha256(value), fsm_id_of(FSMDefinition(**value)) if is_fsm else ""


def test_every_case_has_a_pin():
    indexed = {
        key.split("/", 1)[1]
        for key in json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        if key.startswith("agents/")
    }
    assert not set(_GOLDEN) & indexed
    assert set(_GOLDEN) | indexed == set(CASES)


@pytest.mark.parametrize("name", sorted(_GOLDEN))
def test_builder_output_is_unchanged(name):
    assert _measure(name) == _GOLDEN[name]
