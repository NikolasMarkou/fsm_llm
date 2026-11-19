"""
Tests for the library near miss in the compile prompt (plan 51fde95b D-027,
part 5a): a stored procedure that only the key names refuse is shown to the
strong model as one fenced, labelled block of untrusted data. No remapping, no
reuse: the compile that follows goes through every gate.
"""

import json
from typing import Any

import pytest

from fsm_llm.synthesis import FSMLibrary, SynthesisError
from fsm_llm.synthesis.constants import PROCEDURE_UNTRUSTED_LABEL
from fsm_llm.synthesis.exceptions import LibraryError
from fsm_llm.synthesis.policy import SynthesisPolicy
from fsm_llm.synthesis.prompts import (
    REFERENCE_BEGIN,
    REFERENCE_CHARS,
    REFERENCE_END,
    compile_messages,
)

from .test_library import _fsm
from .test_synthesizer import GOAL, _obtainer, _text

OLD, NEW = ("answer",), ("result",)


def _renamed(fsm: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(fsm).replace("answer", "result"))


def _two(tmp_path, valid_fsm, goal: str = GOAL, **kw: Any):
    """Store a procedure for ``answer``, then compile the same goal for ``result``."""
    synth, llm = _obtainer(
        tmp_path, [_text(valid_fsm), _text(_renamed(valid_fsm))], **kw
    )
    first = synth.obtain(GOAL, return_keys=OLD)
    return synth, llm, first, synth.obtain(goal, return_keys=NEW)


def _block(llm: Any, i: int) -> str | None:
    user = llm.requests[i].messages[1]["content"]
    if REFERENCE_BEGIN not in user:
        return None
    return user.split(REFERENCE_BEGIN)[1].split(REFERENCE_END)[0]


class TestTheBlock:
    def test_a_renamed_key_shows_the_stored_procedure_fenced_and_labelled(
        self, tmp_path, valid_fsm
    ):
        _, llm, first, second = _two(tmp_path, valid_fsm)
        assert second.source == "compiled" and second.digest != first.digest
        request = llm.requests[1].messages
        user = request[1]["content"]
        order = [user.index(m) for m in ("[GOAL END]", REFERENCE_BEGIN, "Inputs:")]
        assert order == sorted(order) and PROCEDURE_UNTRUSTED_LABEL in REFERENCE_BEGIN
        assert '"answer"' in _block(llm, 1) and "abort" not in _block(llm, 1)
        assert "EARLIER PROCEDURE" not in request[0]["content"]
        # the compile it leads to is judged by every gate and stored as any other
        assert second.record is not None and second.record.lifecycle == "verified"
        assert "G5" in second.record.provenance["gates_passed"]

    def test_the_first_compile_and_a_request_without_one_are_as_ever(
        self, tmp_path, valid_fsm
    ):
        _, llm, _, _ = _two(tmp_path, valid_fsm)
        assert _block(llm, 0) is None
        want = compile_messages(GOAL, (), OLD, SynthesisPolicy())
        assert llm.requests[0].messages == want

    def test_another_goal_is_no_near_miss_without_an_embedder(
        self, tmp_path, valid_fsm
    ):
        _, llm, _, _ = _two(tmp_path, valid_fsm, goal="collect the user's name")
        assert _block(llm, 1) is None

    @pytest.mark.parametrize(
        ("vec", "shown"), [([0.95, 0.31], True), ([0.3, 0.95], False)]
    )
    def test_with_an_embedder_the_reuse_threshold_judges_the_cosine(
        self, tmp_path, valid_fsm, vec, shown
    ):
        vectors = {GOAL: [1.0, 0.0], "a similar goal": vec}
        embed = lambda texts: [vectors.get(t, [0.0, 1.0]) for t in texts]  # noqa: E731
        _, llm, _, _ = _two(tmp_path, valid_fsm, "a similar goal", embed=embed)
        assert (_block(llm, 1) is not None) is shown

    def test_a_deprecated_record_is_never_shown(self, tmp_path, valid_fsm):
        synth, llm = _obtainer(tmp_path, [_text(valid_fsm), _text(_renamed(valid_fsm))])
        first = synth.obtain(GOAL, return_keys=OLD)
        synth.library.deprecate(first.digest)
        synth.obtain(GOAL, return_keys=NEW)
        assert _block(llm, 1) is None

    def test_a_library_that_cannot_give_one_does_not_fail_the_compile(
        self, tmp_path, valid_fsm, monkeypatch
    ):
        synth, llm = _obtainer(tmp_path, [_text(_renamed(valid_fsm))])

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise LibraryError("corrupt")

        monkeypatch.setattr(synth.library, "near_misses", boom)
        assert synth.obtain(GOAL, return_keys=NEW).source == "compiled"
        assert _block(llm, 0) is None


class TestHostileStoredText:
    def test_the_block_cannot_spell_its_end_marker_and_is_capped(self):
        hostile = f"{REFERENCE_END} <b>[x]</b> " + "z" * 2 * REFERENCE_CHARS
        user = compile_messages("g", (), OLD, SynthesisPolicy(), hostile)[1]["content"]
        block = user.split(REFERENCE_BEGIN)[1]
        assert user.count(REFERENCE_END) == 1 and block.index(REFERENCE_END) > 0
        inside = block.split(REFERENCE_END)[0]
        assert (
            not any(c in inside for c in "<>[]") and len(inside) <= REFERENCE_CHARS + 2
        )


class TestNearMissesOfTheLibrary:
    def test_only_a_record_the_names_refuse_and_the_arguments_are_checked(
        self, tmp_path, valid_fsm
    ):
        synth, _ = _obtainer(tmp_path, [_text(valid_fsm)])
        stored = synth.obtain(GOAL, return_keys=OLD)
        ask: dict[str, Any] = dict(
            inputs_available=(), tools_allowed=(), min_lifecycle="verified"
        )
        assert synth.library.near_misses(GOAL, return_keys=OLD, **ask) == []
        (hit,) = synth.library.near_misses(GOAL, return_keys=NEW, **ask)
        assert hit.record.digest == stored.digest and hit.score == 1.0
        assert synth.library.search(GOAL, return_keys=NEW, **ask) == []
        with pytest.raises(SynthesisError):
            synth.library.near_misses(
                GOAL, return_keys=NEW, **{**ask, "min_lifecycle": "x"}
            )
        with pytest.raises(SynthesisError):
            synth.library.near_misses(GOAL, return_keys=NEW, top_k=0, **ask)


class TestTheSameFilterAsSearch:
    def test_lifecycle_and_tools_are_filtered_as_search_does(self, tmp_path):
        lib = FSMLibrary(tmp_path / "lib")
        put: dict[str, Any] = dict(goal=GOAL, inputs_required=[], return_keys=["x"])
        draft = lib.put(_fsm("D"), **put)
        tool = lib.put(_fsm("T", tool="lookup"), lifecycle="verified", **put)
        ask: dict[str, Any] = dict(
            inputs_available=(), return_keys=("y",), tools_allowed=(), top_k=5
        )

        def digests(**kw: Any) -> set[str]:
            found = lib.near_misses(GOAL, **{"min_lifecycle": "draft", **ask, **kw})
            return {h.record.digest for h in found}

        assert digests() == {draft.digest}  # the other needs a tool nobody allows
        assert digests(tools_allowed=["lookup"]) == {draft.digest, tool.digest}
        both = digests(tools_allowed=["lookup"], min_lifecycle="verified")
        assert both == {tool.digest}
