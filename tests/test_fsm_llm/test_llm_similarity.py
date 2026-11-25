"""``fsm_llm.llm.similarity``: ``EmbedTextsFn`` and ``cosine_similarity`` (plan c720601d, step 1b).

The two names moved from ``fsm_llm.agents.semantic_tools`` so the synthesis library and
synthesizer read them below the pattern runtimes. These rows pin the values, the one-WARNING
rule for vectors of different length, the single static ``__all__`` surface of ``fsm_llm.llm``
and the clean break at the old path (no alias, no shim).
"""

from __future__ import annotations

import importlib

import pytest

import fsm_llm.llm as llm
from fsm_llm.logging import logger


def _warnings_of(fn) -> tuple[object, list[str]]:
    logger.enable("fsm_llm")
    captured: list[str] = []
    sink_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
    try:
        result = fn()
    finally:
        logger.remove(sink_id)
        logger.disable("fsm_llm")
    return result, captured


class TestCosineSimilarity:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ([1.0, 0.0], [1.0, 0.0], 1.0),
            ([1.0, 0.0], [0.0, 1.0], 0.0),
            ([1.0, 0.0], [-1.0, 0.0], -1.0),
            ([0.0, 0.0], [1.0, 1.0], 0.0),
            ([1.0, 1.0], [0.0, 0.0], 0.0),
            ([], [], 0.0),
            ([3.0, 4.0], [6.0, 8.0], 1.0),
        ],
    )
    def test_values(self, a, b, expected):
        assert llm.cosine_similarity(a, b) == pytest.approx(expected)

    def test_different_widths_score_zero_with_one_warning(self):
        score, captured = _warnings_of(
            lambda: llm.cosine_similarity([1.0, 0.0], [1.0, 0.0, 5.0])
        )

        assert score == 0.0
        assert len(captured) == 1 and "dimension" in captured[0].lower()

    def test_equal_widths_log_nothing(self):
        _, captured = _warnings_of(lambda: llm.cosine_similarity([1.0], [2.0]))

        assert captured == []


class TestTheSurface:
    def test_both_names_are_in_the_one_static_all(self):
        assert {"EmbedTextsFn", "cosine_similarity"} <= set(llm.__all__)
        assert (
            llm.cosine_similarity
            is importlib.import_module("fsm_llm.llm.similarity").cosine_similarity
        )

    def test_the_old_module_defines_neither_name(self):
        old = importlib.import_module("fsm_llm.agents.semantic_tools")

        assert not hasattr(old, "cosine_similarity")
        assert not hasattr(old, "EmbedTextsFn")

    def test_the_old_import_path_raises(self):
        with pytest.raises(ImportError):
            exec("from fsm_llm.agents.semantic_tools import cosine_similarity")
        with pytest.raises(ImportError):
            exec("from fsm_llm.agents.semantic_tools import EmbedTextsFn")
