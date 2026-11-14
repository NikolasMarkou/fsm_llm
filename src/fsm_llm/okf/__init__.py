"""
Open Knowledge Format (OKF): a directory of markdown files with YAML frontmatter.

The format layer, a layer package: ``models`` (the trust tier, an issue, the load
limits, a concept), ``parser`` (the tolerant parser), ``links`` (ids and links),
``bundle`` (the in-memory bundle, link graph and search), ``loader`` (bounded
loading), ``checker`` (conformance checks), ``writer`` (rendering and writing),
``reading`` (the helpers of the agent tools and the CLI), ``memory_export``
(``bundle_from_memory_entries``, not re-exported here: import it by that module's name) and
``cli`` (the ``fsm-llm-okf`` command). The plain-FSM handler is in
``fsm_llm.engine.okf_bridge``, which also re-exports ``bundle_from_memory_entries``. The text helpers (``strip_unsafe_text``,
``capped_line``, ``capped_issues``, ``capped_note``, ``text_page``, ``iso_utc``) live
in the kernel module ``fsm_llm.text`` and are re-exported here as documented public
names. OKF text is untrusted third-party prose: content never raises (a problem
becomes an ``OKFIssue``), nothing a document names is executed, unknown keys are kept.

Spec status: OKF v0.2 at commit ad30107 of GoogleCloudPlatform/open-knowledge-format,
pre-1.0 and mutable. The trust tier is a CLAIM the bundle makes about itself
(a free ``verified[].by`` string), advisory only, never authorization.

This ``__init__`` is the one import surface of the package (a static ``__all__``);
a private name is imported from the module that defines it.
"""

from __future__ import annotations

from fsm_llm.core.definitions import OKFError
from fsm_llm.text import (
    capped_issues,
    capped_line,
    capped_note,
    iso_utc,
    strip_unsafe_text,
    text_page,
)

from .bundle import OKFBundle, incomplete_note
from .checker import check_bundle
from .loader import load_bundle
from .models import (
    INCOMPLETE_CODES,
    OKFConcept,
    OKFIssue,
    OKFLimits,
    TrustTier,
    parse_okf_datetime,
)
from .parser import parse_concept_text
from .reading import body_page, close_ids
from .writer import has_lone_surrogate, write_bundle, writes_within_caps

__all__ = [
    "INCOMPLETE_CODES",
    "OKFBundle",
    "OKFConcept",
    "OKFError",
    "OKFIssue",
    "OKFLimits",
    "TrustTier",
    "body_page",
    "capped_issues",
    "capped_line",
    "capped_note",
    "check_bundle",
    "close_ids",
    "has_lone_surrogate",
    "incomplete_note",
    "iso_utc",
    "load_bundle",
    "parse_concept_text",
    "parse_okf_datetime",
    "strip_unsafe_text",
    "text_page",
    "write_bundle",
    "writes_within_caps",
]
