"""The one cleanup of "these frames are gone" for ``API`` (plan 8525c797, step 5).

Three exits drop frames from a conversation stack and each used to hand-write a part of the same
bookkeeping: ``pop_fsm`` (``stack.py``), the refused branch of ``end_conversation`` (``lifecycle.py``)
and the settle pop (``calls.py``, through ``pop_fsm``). ``release_dropped_frames`` is that
bookkeeping, once.

Layer ``engine``; imports ``delegated`` and ``surface`` only.
"""

from __future__ import annotations

from collections.abc import Iterable

from .delegated import cancel_delegated_runs
from .surface import _ApiSurface


# DECISION plan-2026-11-28T050702-8525c797/D-006
# The frames-are-gone bookkeeping lives HERE and nowhere else. Do NOT re-inline it at a fourth exit
# (the three copies drifted: a failing END handler after the teardown skipped it in `pop_fsm`, a
# busy lower frame skipped the cancel in `end_conversation`, a hand pop never forgot the marks), and
# do NOT call it for a frame whose end was REFUSED (its instance is alive: the frame, its marks and
# its runs stay, D-050 and D-053 of plan 8a03483a). Do NOT cancel under `_stack_lock`: the launcher
# is foreign code and the lock is a plain `Lock` that no manager call or launcher may sit behind
# (D-050 and D-056 of plan baa37f40); the marks are forgotten under the lock, then the lock is
# released, then the runs are cancelled. Do NOT cancel the surviving root: the cancel is for the
# given frame ids only, so a root that stays keeps its run.
def release_dropped_frames(
    api: _ApiSurface, frame_ids: Iterable[str], *, keep_run_marks: bool = False
) -> None:
    """Forget what settle kept for *frame_ids*, tell their delegated runs to stop and release the
    temp definitions no live frame references any more.

    Contract: call it once the frames are OUT of ``api.conversation_stacks``, with no lock held.
    ``frame_ids`` are the dropped frames only (never a surviving frame); an empty iterable does
    nothing. The call marks are dropped in one ``_stack_lock`` hold (``keep_run_marks=True`` leaves
    the ``runs`` marks for ``end_runs_of``, which writes the ``run/end`` that mark owes: a pop);
    each frame's runs are cancelled through ``cancel_delegated_runs`` outside the lock; the temp
    definitions go through the reference-aware release. Never raises: a launcher that fails costs
    one WARNING naming its exception type (``cancel_delegated_runs``)."""
    ids = list(dict.fromkeys(frame_ids))
    if not ids:
        return
    with api._stack_lock:
        api._call_marks.forget(ids, runs=not keep_run_marks)
    cancel_delegated_runs(api._targets, ids[0], ids[1:])
    api._release_unreferenced_temp_definitions()
