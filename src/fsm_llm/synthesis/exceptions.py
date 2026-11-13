"""
Exception hierarchy for the fsm_llm.synthesis package.

Rooted at ``fsm_llm.definitions.FSMError`` so a caller catching the framework's
base error also catches every synthesis failure.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.definitions import FSMError


class SynthesisError(FSMError):
    """Base exception for all synthesis errors (a compile that never passed
    its gates, a provider outage during a compile, a bad limit)."""


class SynthesisRefusedError(SynthesisError):
    """A procedure was refused on purpose (policy, approval, depth).

    ``code`` is one of the fixed reason codes in ``constants`` (for example
    ``approval_denied``); callers branch on it, never on the message.
    """

    def __init__(
        self,
        message: str,
        code: str,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)
        self.code = code


class PatchError(SynthesisError):
    """A patch was refused whole (nothing was applied).

    ``code`` is one of ``constants.PATCH_CODES`` and ``op`` the index of the op
    at fault (``None`` for a refusal of the whole patch). The message holds the
    code and the index only, never any text of the patch.
    """

    def __init__(self, code: str, op: int | None = None):
        super().__init__(
            f"patch refused: {code}" + ("" if op is None else f" (op {op})")
        )
        self.code = code
        self.op = op


class LibraryError(SynthesisError):
    """The procedure library is unreadable, corrupt, unwritable or inconsistent.

    A corrupt file is never overwritten; a human fixes it.
    """
