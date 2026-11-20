"""Readable text fixtures of every built-in FSM (plan ace7a8c8, F1.2).

``tests/fixtures/executables/{agents,reasoning}/<case>.json`` hold the canonical
JSON of every full-FSM built-in in insertion order. The hash goldens use
``sort_keys=True`` and show only a changed hash; these show a diff, and they see a
state or key order the hashes cannot. The test only compares: regenerate with
``python -m tests.executables_corpus --write`` and review the diff.
"""

from __future__ import annotations

import difflib
import json
import os
import subprocess
import sys

import pytest

from fsm_llm.agents import fsm_definitions as fd
from fsm_llm.core.definitions import FSMDefinition, definition_digest, fsm_id_of
from fsm_llm.reasoning.reasoning_modes import ALL_REASONING_FSMS
from tests import executables_corpus as corpus

_CASES = corpus.fsm_cases()
_REGENERATE = "regenerate: python -m tests.executables_corpus --write"


@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_fixture_text_is_the_rebuilt_text(case):
    path = corpus.fixture_path(case)
    assert path.is_file(), f"no fixture for {case}; {_REGENERATE}"
    stored, built = path.read_text(encoding="utf-8"), corpus.render(_CASES[case])
    diff = difflib.unified_diff(
        stored.splitlines(), built.splitlines(), "fixture", "built", lineterm=""
    )
    assert stored == built, "\n".join([*list(diff)[:40], _REGENERATE])


def test_the_index_is_the_recomputed_identity_of_every_case():
    stored = corpus.INDEX_PATH.read_text(encoding="utf-8")
    assert stored == corpus.index_text()
    for case, row in json.loads(stored).items():
        definition = FSMDefinition(**_CASES[case])
        assert row["digest64"] == definition_digest(definition)
        assert row["fsm_id"] == fsm_id_of(definition)


def test_every_fixture_is_ascii_and_the_escape_is_exercised():
    paths = [*sorted(corpus.FIXTURES.glob("*/*.json")), corpus.INDEX_PATH]
    raw = {p: p.read_bytes() for p in paths}
    assert [p for p, b in raw.items() if not b.isascii()] == []
    assert any(b"\\u2014" in b for b in raw.values())  # an em dash is written escaped


def test_no_orphan_fixture_and_no_case_without_a_file():
    on_disk = {f"{p.parent.name}/{p.stem}" for p in corpus.FIXTURES.glob("*/*.json")}
    expected = {f"reasoning/{name}" for name in ALL_REASONING_FSMS} | {
        f"agents/{name}" for name, (_, full) in corpus.CASES.items() if full
    }
    assert on_disk == expected
    assert set(_CASES) == expected


def test_every_exported_builder_has_a_case():
    used = {n for build, _ in corpus.CASES.values() for n in build.__code__.co_names}
    builders = {n for n in fd.__all__ if n.startswith("build_")}
    assert builders - used == set()


def test_the_text_is_the_same_under_three_hash_seeds():
    here = corpus.texts_digest()
    code = "from tests.executables_corpus import texts_digest; print(texts_digest())"
    for seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        done = subprocess.run(
            [sys.executable, "-c", code],
            cwd=corpus.ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert done.stdout.strip() == here, f"PYTHONHASHSEED={seed}"
