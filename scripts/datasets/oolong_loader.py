#!/usr/bin/env python3
"""
OOLONG → fsm_llm internal JSONL converter (M5 slice 7).

Pulls OOLONG records from HuggingFace via the `datasets` library
(streaming mode — no 12 GB bulk download), converts to the slice-6
internal JSONL schema, and writes to a local file the bench harness
can consume:

    {id, task, question, answer, document, metadata}

# DECISION D-S7-001: streaming-only. We never trigger a bulk parquet
# download — `datasets.load_dataset(..., streaming=True)` returns an
# IterableDataset that we iterate one record at a time.
#
# DECISION D-S7-002: license-conservative. HF dataset cards for
# `oolongbench/oolong-{synth,real}` do not state a license. We do NOT
# redistribute records — the converted JSONL stays in the user's
# working tree, gitignored. Loader writes only to user-supplied paths.
#
# DECISION D-S7-003: every OOLONG task maps to `task: "aggregate"` in
# our internal schema. Original task preserved in `metadata.oolong_task`.

Usage:
    python scripts/datasets/oolong_loader.py \\
        --subset synth --split validation \\
        --limit-per-task 6 \\
        --out evaluation/datasets/oolong_synth_real_subset.jsonl

The output file is gitignored by `evaluation/datasets/.gitignore`.

Requires: pip install -e ".[oolong]"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_DEFAULT_LIMIT_PER_TASK = 6
_SYNTH_DATASET = "oolongbench/oolong-synth"
_REAL_DATASET = "oolongbench/oolong-real"
_DEFAULT_REAL_CONFIG = "dnd"


def convert_record(
    rec: dict[str, Any], subset: str, split: str, idx: int
) -> dict[str, Any] | None:
    """Convert one OOLONG record to the slice-6 internal JSONL schema.

    Returns None if required fields are missing/empty.

    Per D-S7-003, ``task`` is always ``"aggregate"``; the OOLONG
    per-record task string is preserved in ``metadata.oolong_task``.
    """
    document = rec.get("context_window_text")
    question = rec.get("question")
    answer = rec.get("answer")
    if not document or not question or answer is None or answer == "":
        return None

    if subset == "synth":
        oolong_id = rec.get("id", idx)
        oolong_task_group = rec.get("task_group", "unknown")
        oolong_task = rec.get("task", "unknown")
        oolong_answer_type = rec.get("answer_type")
        context_len = rec.get("context_len")
        namespaced_id = (
            f"oolong-synth-{split}-{oolong_task_group}-{oolong_task}-{oolong_id}"
        )
        metadata = {
            "source": _SYNTH_DATASET,
            "oolong_id": oolong_id,
            "oolong_task_group": oolong_task_group,
            "oolong_task": oolong_task,
            "oolong_answer_type": oolong_answer_type,
            "context_len": context_len,
        }
    elif subset == "real":
        oolong_id = rec.get("id", str(idx))
        oolong_task = rec.get("question_type", "unknown")
        campaign = rec.get("campaign")
        namespaced_id = f"oolong-real-{split}-{oolong_task}-{oolong_id}"
        metadata = {
            "source": _REAL_DATASET,
            "oolong_id": oolong_id,
            "oolong_task": oolong_task,
            "oolong_campaign": campaign,
        }
    else:
        raise ValueError(f"unknown subset: {subset!r}")

    return {
        "id": namespaced_id,
        "task": "aggregate",
        "question": str(question),
        "answer": str(answer),
        "document": str(document),
        "metadata": metadata,
    }


def _task_key(rec: dict[str, Any], subset: str) -> str:
    """Per-task grouping key for subsetting."""
    if subset == "synth":
        return str(rec.get("task", "unknown"))
    return str(rec.get("question_type", "unknown"))


def load_and_convert(
    subset: str,
    split: str,
    *,
    limit_per_task: int,
    seed: int = 42,
    config: str | None = None,
    max_context_len: int | None = None,
    _records_iter: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Stream OOLONG records, group by task, take first N per task, convert.

    The ``_records_iter`` parameter is a test-only escape hatch for
    monkeypatching `datasets.load_dataset`. In production it is None and
    we call `load_dataset(..., streaming=True)`.
    """
    if limit_per_task < 1:
        return []

    if _records_iter is None:
        # D-S7-001: streaming mode only.
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "OOLONG loader requires `datasets`. Install via: "
                'pip install -e ".[oolong]"'
            ) from e

        if subset == "synth":
            ds_name = _SYNTH_DATASET
            cfg = config
        elif subset == "real":
            ds_name = _REAL_DATASET
            cfg = config or _DEFAULT_REAL_CONFIG
        else:
            raise ValueError(f"unknown subset: {subset!r}")

        load_kwargs: dict[str, Any] = {"split": split, "streaming": True}
        if cfg is not None:
            ds = load_dataset(ds_name, cfg, **load_kwargs)
        else:
            ds = load_dataset(ds_name, **load_kwargs)
        ds = ds.shuffle(seed=seed, buffer_size=1024)
        _records_iter = ds

    by_task: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    converted: list[dict[str, Any]] = []
    skipped = 0
    skipped_too_long = 0
    for idx, rec in enumerate(_records_iter):
        # Pre-filter on context_len when supplied (synth records have it).
        if max_context_len is not None:
            ctx_len = rec.get("context_len")
            if isinstance(ctx_len, int) and ctx_len > max_context_len:
                skipped_too_long += 1
                continue
        key = _task_key(rec, subset)
        if len(by_task[key]) >= limit_per_task:
            # Already full for this task; check if all groups are full
            # — but we don't know the cardinality of task keys ahead of
            # time, so just keep iterating until we've seen enough total
            # records to be confident. Cap iteration at limit_per_task *
            # 12 (max 5 synth task types + slack) to bound work.
            if (
                all(len(by_task[k]) >= limit_per_task for k in by_task)
                and len(by_task) >= 5
            ):
                break
            continue
        out = convert_record(rec, subset, split, idx)
        if out is None:
            skipped += 1
            continue
        by_task[key].append(out)
        converted.append(out)

    print(
        f"[oolong_loader] subset={subset} split={split} "
        f"converted={len(converted)} skipped={skipped} "
        f"skipped_too_long={skipped_too_long} "
        f"task_groups={dict((k, len(v)) for k, v in by_task.items())}",
        file=sys.stderr,
    )
    return converted


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """One record per line, sorted-keys JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    p.add_argument(
        "--subset",
        choices=("synth", "real"),
        required=True,
        help="OOLONG subset. 'synth' = oolong-synth (1024-tok ICL "
        "tasks); 'real' = oolong-real (D&D transcripts).",
    )
    p.add_argument(
        "--split",
        default="validation",
        help="HF split name. Default 'validation'.",
    )
    p.add_argument(
        "--limit-per-task",
        type=int,
        default=_DEFAULT_LIMIT_PER_TASK,
        help=f"Records per task type. Default {_DEFAULT_LIMIT_PER_TASK}.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed (HF buffer_size=1024). Default 42.",
    )
    p.add_argument(
        "--config",
        default=None,
        help="HF config name. Default: synth uses no config; real uses 'dnd'.",
    )
    p.add_argument(
        "--max-context-len",
        type=int,
        default=None,
        help=(
            "Skip records whose `context_len` field exceeds this value "
            "(synth-only). OOLONG-synth includes contexts from 1024 to "
            "4M tokens; cap to keep bench wall-time manageable. Default: "
            "no cap."
        ),
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output JSONL path (e.g. evaluation/datasets/oolong_synth_real_subset.jsonl).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(
            f"ERROR: output path {out_path} exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        return 4

    if args.limit_per_task < 1:
        print("ERROR: --limit-per-task must be >= 1", file=sys.stderr)
        return 5

    try:
        records = load_and_convert(
            args.subset,
            args.split,
            limit_per_task=args.limit_per_task,
            seed=args.seed,
            config=args.config,
            max_context_len=args.max_context_len,
        )
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except (ConnectionError, OSError) as e:
        print(
            f"ERROR: HF connection failed ({type(e).__name__}: {e}). "
            f"Check network or run `huggingface-cli login` if the dataset "
            f"requires authentication.",
            file=sys.stderr,
        )
        return 2
    except PermissionError as e:
        print(
            f"ERROR: HF dataset access denied ({e}). The dataset may be "
            f"gated; run `huggingface-cli login` and accept terms on the "
            f"dataset page.",
            file=sys.stderr,
        )
        return 2

    if not records:
        print(
            f"WARNING: no records produced for subset={args.subset} "
            f"split={args.split}. Output not written.",
            file=sys.stderr,
        )
        return 3

    write_jsonl(records, out_path)
    print(
        f"Wrote {len(records)} records to {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
