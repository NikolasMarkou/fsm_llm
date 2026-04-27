"""Schemas for the retry_loop pipeline example.

retry_loop uses host-callables only (no Leaf nodes), so this file is
present for shape parity with other pipeline examples but defines no
schemas referenced from run.py.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class _Unused(BaseModel):
    placeholder: str = Field(default="")
