"""
OKF conformance checks (spec section 11) over a bundle; never raises.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Literal

from fsm_llm.constants import OKF_MAX_ISSUES

from .bundle import _CHECK_DEADLINE, OKFBundle
from .links import _body_links, _is_reserved, _link_candidates
from .models import OKFIssue, _capped
from .parser import _check_timestamps

_MAX_LINK_ISSUES = 20  # link-broken issues per concept


def check_bundle(bundle: OKFBundle) -> list[OKFIssue]:
    """Spec section 11 conformance as a list of issues. Never raises.

    The load-time issues plus checks that also cover concepts added with
    ``put``: non-empty ``type``, no reserved id, timestamps without an explicit
    UTC offset, unknown ``okf_version`` (info), broken links (info; at most
    ``_MAX_LINK_ISSUES`` per concept and twice ``OKF_MAX_ISSUES`` in all, then
    one ``issues-truncated``). Duplicates dropped, result capped as in
    ``load_bundle``; it stops at ``max_seconds`` with ``check-deadline``.
    """
    out = list(bundle.issues)
    seen = {(i.code, i.path, i.id, i.message) for i in out}
    typed = {i.id for i in out if i.code in ("type-missing", "type-empty")}

    def add(
        severity: Literal["error", "warning", "info"],
        code: str,
        message: str,
        cid: str | None = None,
    ) -> None:
        issue = OKFIssue(severity=severity, code=code, message=message, id=cid)
        if (key := (code, None, issue.id, issue.message)) not in seen:
            seen.add(key)
            out.append(issue)

    deadline, broken, cut = bundle._deadline(), 0, False
    if (version := bundle.okf_version) not in (None, "0.1", "0.2"):
        add("info", "okf-version-unknown", f"okf_version {version!r} unknown")
    for c in bundle:
        if time.monotonic() > deadline:
            if _CHECK_DEADLINE not in out:  # backlinks may have left it behind
                out.append(_CHECK_DEADLINE)
            break
        note = partial(add, cid=c.id)

        if _is_reserved(c.id.rsplit("/", 1)[-1]):
            note("error", "reserved-id", "index and log are reserved, never concepts")
        if not c.type.strip() and c.id not in typed:
            note("error", "type-empty", "'type' must be a non-empty string")
        _check_timestamps(c.stale_after, c.generated, c.verified, note)
        first = broken
        for target in _body_links(c.body):
            if (cands := _link_candidates(target, c.id)) and not any(
                x in bundle for x in cands
            ):
                if broken - first >= _MAX_LINK_ISSUES or broken >= 2 * OKF_MAX_ISSUES:
                    cut = True
                    break
                broken += 1
                note("info", "link-broken", f"link target {target!r} not in the bundle")
    if cut:
        add("info", "issues-truncated", "more broken links are not shown")
    return _capped(out)
