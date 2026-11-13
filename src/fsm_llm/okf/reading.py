"""
Reading helpers shared by the OKF agent tools and the CLI: a concept body page and
the close-id hint of a not-found answer.
"""

from __future__ import annotations

import difflib
from typing import Any

from fsm_llm.text import strip_unsafe_text, text_page

from .bundle import OKFBundle
from .models import OKFConcept


def body_page(
    concept: OKFConcept, offset: Any, size: int
) -> tuple[str, int | None, int]:
    """``text_page`` of a concept's stripped body."""
    return text_page(strip_unsafe_text(concept.body, multiline=True), offset, size)


def close_ids(
    bundle: OKFBundle, wanted: str, *, include_deprecated: bool = False, n: int = 5
) -> list[str]:
    """Up to *n* ids of *bundle* close to *wanted* (the not-found hint)."""
    ids = [c.id for c in bundle.list(include_deprecated=include_deprecated)]
    return difflib.get_close_matches(wanted[:200], ids, n=n)
