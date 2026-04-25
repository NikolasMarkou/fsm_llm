from __future__ import annotations

"""
Needle-in-haystack (NIAH) factory.

Given a question, returns a λ-term that recursively splits an input
document of any size into k-ary chunks until each chunk fits the leaf
threshold τ, calls the oracle on each chunk to extract a candidate answer,
and reduces with a caller-supplied "best" op to select the final answer.

Stub at scaffolding step — body is implemented at step 2.
"""

from fsm_llm.lam import Term


def niah(
    question: str,
    *,
    tau: int = 512,
    k: int = 2,
    reduce_op_name: str = "best",
) -> Term:
    """Build a needle-in-haystack λ-term.

    Parameters
    ----------
    question:
        The question to ask each leaf-level chunk. Bound into the leaf
        prompt template.
    tau:
        Leaf-size threshold (characters). Inputs ≤ τ go to a single leaf
        oracle call; larger inputs are SPLIT into k pieces and recursed.
    k:
        Branching factor for SPLIT. Default 2 (paper Theorem 4 optimum
        under linear cost).
    reduce_op_name:
        Name of the REDUCE op to look up in env. Caller must bind a
        ``ReduceOp`` instance under this name when calling
        ``Executor.run(program, env)``. Default ``"best"``.

    Returns
    -------
    A ``Term`` that, when run with env containing the bound document,
    a ``size_bucket`` callable, and a ``reduce_op_name`` ReduceOp,
    produces the best-matching answer.

    Notes
    -----
    Implementation deferred to step 2 of the M5 slice-1 plan.
    """
    raise NotImplementedError(
        "niah() body lands at plan step 2; this is the step-1 scaffold."
    )


__all__ = ["niah"]
