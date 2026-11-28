"""The temp definition ledger of ``API``: the one cleanup of "these frames are gone" and the count
of pushes in flight (plan 8525c797, steps 5 and 6).

The count is ``register_pending_push`` and ``release_pending_push`` (bottom of the module). The
cleanup: three exits drop frames from a conversation stack and each used to hand-write a part of the same
bookkeeping: ``pop_fsm`` (``stack.py``), the refused branch of ``end_conversation`` (``lifecycle.py``)
and the settle pop (``calls.py``, through ``pop_fsm``). ``release_dropped_frames`` is that
bookkeeping, once.

Layer ``engine``; imports ``core.definitions`` (a type), ``delegated`` and ``surface`` only.
"""

from __future__ import annotations

from collections.abc import Iterable

from fsm_llm.core.definitions import FSMDefinition

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


# DECISION plan-2026-11-28T050702-8525c797/D-006
# `_pending_push_ids` is a COUNT of pushes in flight per content-hash id, never a set. Two pushes of
# one definition share one id, so a set let the first to finish discard the guard of the second
# (a cleanup then evicted the definition mid-greeting and the second child failed with
# `FSMDefinitionNotFoundError`). Do NOT go back to `add` and `discard`, and do NOT uncount an id this
# call did not count: a caller keeps the id it registered in a local (None before the register and
# after the release) and hands only that to `_rollback_push`; an unconditional decrement removes
# another push's guard. A count that reaches zero is deleted, so `in` and the cleanup read it as a set.
def register_pending_push(
    api: _ApiSurface, fsm_id: str, definition: FSMDefinition
) -> None:
    """Register *definition* as the temp definition of *fsm_id* and count one push in flight.

    Contract: the caller holds ``api._stack_lock`` and releases the push exactly once, through
    ``release_pending_push``, in the hold that appends its frame or in the rollback; it keeps the id
    in a local so a failure before this call never uncounts a push that is not its own."""
    api._temp_fsm_definitions[fsm_id] = definition
    api._pending_push_ids[fsm_id] += 1


def release_pending_push(api: _ApiSurface, fsm_id: str) -> None:
    """Uncount one push in flight for *fsm_id*; the entry goes at zero (never a negative count).

    Contract: the caller holds ``api._stack_lock`` and counted this push through
    ``register_pending_push``. An id with no count is left alone."""
    count = api._pending_push_ids.get(fsm_id, 0) - 1
    if count > 0:
        api._pending_push_ids[fsm_id] = count
    else:
        api._pending_push_ids.pop(fsm_id, None)
