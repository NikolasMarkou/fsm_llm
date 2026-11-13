"""The API-key check shared by the monitor server and `AgentServer`."""

from __future__ import annotations

import hmac
from collections.abc import Mapping

# --------------------------------------------------------------
# API Key Check (monitor server, AgentServer)
# --------------------------------------------------------------


def checked_api_key(api_key: str | None) -> str | None:
    """Return ``api_key`` unchanged; ``ValueError`` if it is empty or blank.

    Interface contract: ``None`` means "no auth" and passes through; a ``str``
    that is empty or whitespace-only raises ``ValueError``.
    """
    # DECISION plan-2026-09-24T091842-c1d5bfbc/D-006
    # Reject an empty key. Do NOT accept it (compare_digest(b"", b"") is True,
    # so an empty header would authenticate) and do NOT map it to None (an
    # operator who passed os.getenv("KEY", "") would silently get an open
    # server). An empty ENV VAR still means "unset", decided by the caller.
    if api_key is not None and not api_key.strip():
        raise ValueError("api_key must be a non-empty string, or None for no auth")
    return api_key


def presented_api_key(headers: Mapping[str, str]) -> str | None:
    """The key a request presents: ``Authorization: Bearer <key>``, else ``X-API-Key``.

    Interface contract: ``headers`` is a case-insensitive mapping (Starlette's
    ``request.headers``) or a dict with lower-case names. A Bearer value is
    stripped; ``Authorization`` with another scheme falls through to
    ``X-API-Key``. Returns ``None`` when neither is present.
    """
    auth_header = headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[len("bearer ") :].strip()
    return headers.get("x-api-key")


def api_key_matches(token: str | None, expected: str) -> bool:
    """True when ``token`` equals ``expected``, compared in constant time.

    Interface contract: ``expected`` is a configured, already-checked key
    (``checked_api_key``). Anything but a ``str`` ``token`` (``None`` for an
    absent header, a ``bytes`` value) is refused. Never raises for any ``str``,
    non-ASCII and lone-surrogate text included.
    """
    # DECISION plan-2026-09-12T065608-089d0ec7/D-016
    # hmac.compare_digest, never `!=`: a plain comparison short-circuits on the
    # first differing byte and leaks the key through response timing. The
    # `isinstance` check comes first because compare_digest(None, key) raises
    # TypeError instead of returning False.
    #
    # DECISION plan-2026-09-12T065608-089d0ec7/D-020
    # Do NOT pass `str` to compare_digest: its str overload raises TypeError
    # unless BOTH sides are ASCII, which would turn an attacker-sent
    # non-ASCII header into a 500. Encode both sides to bytes first, with
    # `surrogateescape` so an undecodable header byte encodes losslessly
    # instead of raising UnicodeEncodeError.
    return isinstance(token, str) and hmac.compare_digest(
        token.encode("utf-8", "surrogateescape"),
        expected.encode("utf-8", "surrogateescape"),
    )
