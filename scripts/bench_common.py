#!/usr/bin/env python3
"""Shared stdlib helpers of the pre-registered live benches (no CLI).

``scripts/agents_bench.py`` and ``scripts/skills_bench.py`` load this module
and reuse its statistics (``wilson_ci``, ``fisher_exact_two_sided``), jsonl
row files, manifest comparison and drift checks, wire capture and digests.
They keep no copy: this file is the one source (plan 65baa765, D-002), and
``tests/test_fsm_llm_eval/test_bench_parity.py`` holds its seven mirrored
helpers equal to their ``fsm_llm.eval`` twins.

Import hygiene: the top level imports the standard library only; anything
else (``fsm_llm.llm``, ``fsm_llm.agents.tools``) is imported LAZILY inside the
function that needs it, so importing this module opens no socket. Pinned by a
socket-disabled subprocess import in the tests.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
MODEL_TAG = "qwen3.5:4b"

#: The six fields a manifest MUST carry for two blocks to be comparable.
_FIELDS = "prompt_bytes_sha256 tool_surface fixture_hash model_digest arm git_commit"
MANIFEST_FIELDS = tuple(_FIELDS.split())

#: How a pair prints a field one manifest does not carry.
NOT_RECORDED = "not recorded"
#: The date line a prompt-mode system prompt carries (core prompts.py); a
#: wire digest masks it so a digest does not move overnight or with TZ.
_DATE_LINE_RE = re.compile(r"Today's date: \d{4}-\d{2}-\d{2}")
_DATE_MASK = "Today's date: <DATE>"
#: Stands in for the task text in a digested user turn (``tasks_sha256`` /
#: ``fixture_hash`` pin the task text itself).
_TASK_MASK = "<TASK>"


class BenchDataError(RuntimeError):
    """A bench invariant would be violated; refuse rather than degrade."""


# DECISION plan-2026-09-29T061903-581c2634/D-008
# wilson_ci, fisher_exact_two_sided, _utc_now, _git_commit, _write_json,
# append_row and read_rows are stdlib copies of their fsm_llm.eval twins. Do NOT
# delegate them to fsm_llm.eval, not even lazily: importing fsm_llm pulls
# litellm, which fetches its model-cost map over HTTP, so an offline bench
# command (a report recount) would open a socket.
# tests/test_fsm_llm_eval/test_bench_parity.py keeps the two copies equal;
# change both together. See decisions.md D-008.
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval (95% default); math-only, scipy is not in venv."""
    if k < 0 or n < 0 or k > n:
        raise ValueError(f"impossible count: k={k}, n={n}")
    if n == 0:
        return (0.0, 1.0)
    p, zz = k / n, z * z
    denom = 1 + zz / n
    center = (p + zz / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + zz / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def fisher_exact_two_sided(k1: int, n1: int, k2: int, n2: int) -> float:
    """Fisher exact p for [[k1,n1-k1],[k2,n2-k2]]: sum of pmf <= observed pmf."""
    for k, n in ((k1, n1), (k2, n2)):
        if n <= 0 or k < 0 or k > n:
            raise ValueError(f"impossible arm: k={k}, n={n}")
    r1, denom = k1 + k2, math.comb(n1 + n2, k1 + k2)

    def pmf(a: int) -> float:
        return math.comb(n1, a) * math.comb(n2, r1 - a) / denom

    p_obs = pmf(k1)
    span = range(max(0, r1 - n2), min(r1, n1) + 1)
    return min(1.0, sum(pmf(a) for a in span if pmf(a) <= p_obs * (1 + 1e-9)))


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str:
    cmd = ("git", "rev-parse", "HEAD")
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    return res.stdout.strip()


def _git_dirty() -> bool:
    """Whether tracked files differ from HEAD (the run's code is not HEAD's)."""
    cmd = ("git", "status", "--porcelain", "--untracked-files=no")
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    return bool(res.stdout.strip())


def run_commit() -> dict[str, Any]:
    """The code a ``run`` actually executes: ``{"git_commit", "git_dirty"}``,
    read when the run starts (a block may be registered at an older commit)."""
    return {"git_commit": _git_commit(), "git_dirty": _git_dirty()}


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _model_digest(tag: str = MODEL_TAG) -> dict[str, str]:
    """The digest Ollama serves for *tag* RIGHT NOW -- queried, never assumed."""
    from urllib.request import urlopen  # lazy: no sockets at import time

    try:
        with urlopen(OLLAMA_TAGS_URL, timeout=5) as resp:
            data = json.load(resp)
    except Exception as exc:  # daemon down, refused, malformed
        raise BenchDataError(f"cannot query {OLLAMA_TAGS_URL}: {exc}") from exc
    for entry in data.get("models", []):
        if tag in entry.get("name", ""):
            return {"tag": entry["name"], "digest": entry.get("digest", "")}
    raise BenchDataError(f"model {tag!r} not in ollama /api/tags -- pull it first")


# --- Wire disclosure (shared by the benches) -----------------------------------


class WireCaptured(Exception):
    """Raised by the capture stub instead of sending a provider request."""


@contextmanager
def captured_wire() -> Iterator[list[dict[str, Any]]]:
    """Capture every request at core's one send binding; send nothing.

    Interface contract (callers: ``request_disclosure`` in agents_bench;
    the tests):
        - Replaces ``fsm_llm.llm.completion`` (the binding every
          ``LiteLLMInterface`` request goes through, ``_send``) for the
          duration of the block with a stub that appends a deep copy of the
          request kwargs to the yielded list and raises ``WireCaptured``, so
          the caller's run fails at its first request with no provider call.
        - The captured kwargs are the FINAL request: Ollama preparation,
          temperature rules, ``max_tokens``, timeout, retries and every extra
          provider kwarg included.
        - Restores the binding on exit, also when the block raises.
        - Imports fsm_llm lazily (this module's top level stays stdlib-only).
    """
    # DECISION plan-2026-10-01T093600-944e2692/D-040: disclosure is taken at
    # core's send binding, not at the LLMInterface boundary. Do NOT digest
    # the request objects an agent hands its interface (core's request
    # preparation, temperature 0 rules, /nothink and max_tokens then move no
    # manifest field), and do NOT call a provider. See decisions.md D-040.
    import fsm_llm.llm as core_llm

    captured: list[dict[str, Any]] = []

    def capture(**call_params: Any) -> Any:
        captured.append(copy.deepcopy(call_params))
        raise WireCaptured("bench wire probe: no provider request is sent")

    original = core_llm.completion
    core_llm.completion = capture
    try:
        yield captured
    finally:
        core_llm.completion = original


def sha256_json(obj: Any) -> str:
    """sha256 of ``json.dumps(obj)`` with key order kept, never sorted: a
    schema whose keys were reordered is a different digest."""
    return hashlib.sha256(json.dumps(obj).encode("utf-8")).hexdigest()


def _masked(content: Any, task_text: str | None) -> Any:
    """``content`` with the date line and the task text masked (str only)."""
    if not isinstance(content, str):
        return content
    text = _DATE_LINE_RE.sub(_DATE_MASK, content)
    if task_text:
        text = text.replace(task_text, _TASK_MASK)
    return text


def wire_settings(call_params: dict[str, Any]) -> dict[str, Any]:
    """Every request kwarg except ``messages`` and ``tools``, secrets redacted.

    Values are kept (a ``seed`` 7 vs 8, an ``api_base`` change, temperature,
    ``max_tokens``, ``reasoning_effort``, timeout, ``max_retries``), sorted
    by name; an entry ``fsm_llm.agents.tools.redact_secret_entries`` (core's
    ``is_forbidden_context_entry`` at every level) calls secret keeps its key
    with the value ``"<redacted>"``.
    """
    from fsm_llm.agents.tools import redact_secret_entries

    settings = {
        key: call_params[key]
        for key in sorted(call_params)
        if key not in ("messages", "tools")
    }
    redacted: dict[str, Any] = redact_secret_entries(settings)
    return redacted


def wire_digest(call_params: dict[str, Any], task_text: str | None) -> dict[str, Any]:
    """Digests of one captured request, comparable across days and TZs.

    Interface contract (callers: ``request_disclosure`` in agents_bench):
        - ``system_sha256``: the system message contents in order;
          ``user_sha256``: every other message (role and content), with the
          ``Today's date:`` line and ``task_text`` masked, so the Ollama
          ``/nothink`` prefix, a schema echo or the neutral turn still count;
          ``tools_sha256``: the ``tools`` kwarg bytes (key order kept), or
          ``None``; ``settings_sha256``: ``wire_settings`` (values included).
        - Never raises on a strange message shape (non-dict entries are
          digested as given).
    """
    messages = call_params.get("messages") or []
    system: list[Any] = []
    others: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            others.append(message)
            continue
        content = _masked(message.get("content"), task_text)
        if message.get("role") == "system":
            system.append(content)
        else:
            others.append({**message, "content": content})
    tools = call_params.get("tools")
    return {
        "system_sha256": sha256_json(system),
        "user_sha256": sha256_json(others),
        "tools_sha256": None if tools is None else sha256_json(tools),
        "settings_sha256": sha256_json(wire_settings(call_params)),
    }


def _leaf_text(value: Any) -> str:
    """A leaf as JSON, type-exact (``1`` vs ``true``, ``120`` vs ``120.0``),
    key order kept (as every digest here keeps it)."""
    return json.dumps(value)


_ABSENT = object()


def manifest_differences(
    a: dict[str, Any], b: dict[str, Any], prefix: str = ""
) -> list[str]:
    """One ``"key: A vs B"`` line per manifest leaf that differs, A first.

    Interface contract (callers: ``report`` pairs in agents_bench; the tests):
        - Walks the union of both manifests' keys in sorted order, recursing
          into keys whose values are dicts on both sides (dotted paths, e.g.
          ``model_digest.digest``); every other differing value is one line
          with both values as JSON.
        - Values are compared as JSON text, so a type change is a difference
          (``1`` vs ``True``, ``120`` vs ``120.0``), unlike Python ``==``.
        - A key one side lacks prints as ``NOT_RECORDED`` (recorded B0/B1
          manifests predate the disclosure fields); nothing is skipped or
          truncated. Equal manifests give ``[]``. Never raises.
    """
    lines: list[str] = []
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key, _ABSENT), b.get(key, _ABSENT)
        path = f"{prefix}{key}"
        if isinstance(va, dict) and isinstance(vb, dict):
            lines.extend(manifest_differences(va, vb, f"{path}."))
            continue
        ta = NOT_RECORDED if va is _ABSENT else _leaf_text(va)
        tb = NOT_RECORDED if vb is _ABSENT else _leaf_text(vb)
        if ta != tb:
            lines.append(f"{path}: {ta} vs {tb}")
    return lines


def pinned_drift(
    recorded: dict[str, Any], current: dict[str, Any], keys: tuple[str, ...]
) -> list[str]:
    """The pinned keys a registered manifest no longer matches at ``run``.

    Interface contract (callers: ``_checked_manifest`` in agents_bench, the
    run gate of skills_bench; the tests):
        - ``recorded`` is the manifest read back from disk, ``current`` the
          one ``build_manifest`` returns now. ``current`` is normalised the
          way ``_write_json`` stores a manifest (keys sorted at every level)
          before the comparison, so key order inside a list never reads as
          drift (review pass 12 note 3); values are compared as JSON text
          (``manifest_differences``), so a type change is drift.
        - Returns the drifted names in ``keys`` order, then
          ``"model_digest"`` when the served digest moved. ``[]`` = no drift.
          A key ``current`` lacks is compared as ``None``. Never raises.
    """
    # DECISION plan-2026-10-01T093600-944e2692/D-049: one drift check for
    # agents_bench and skills_bench, against the stored (sort_keys) form. Do NOT compare the
    # in-memory manifest as built (list items of dicts then drift on key
    # order) and do NOT let `run` overwrite a registered manifest instead of
    # checking it (the committed pre-registration would vanish). See D-049.
    normal = json.loads(json.dumps(current, sort_keys=True))
    drift = [
        key
        for key in keys
        if manifest_differences({key: recorded.get(key)}, {key: normal.get(key)})
    ]
    recorded_digest = (recorded.get("model_digest") or {}).get("digest")
    if recorded_digest != (normal.get("model_digest") or {}).get("digest"):
        drift.append("model_digest")
    return drift


def append_row(path: Path, row: dict[str, Any]) -> None:
    """Append ONE jsonl row and flush -- rows survive an aborted block."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
