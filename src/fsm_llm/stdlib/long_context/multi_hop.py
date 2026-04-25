from __future__ import annotations

"""
Multi-hop factory — chain N independent niah-style sweeps over a long
document, threading each hop's result into the next hop's leaf prompt.

Builds a ``Let`` chain of ``hops`` independent ``Fix`` calls. Each hop is a
full niah-shaped recursive sweep (``fix(λself. λP. case size_bucket(P) of
"small" → leaf(prompt_i, P, extra_inputs) | _ → reduce_(<op>, fmap(self,
split(P, k))))``). Hop ``0``'s leaf prompt embeds the user's question.
Hop ``i ≥ 1``'s leaf prompt closes over the question AND references the
prior hop's result via the env variable ``hop_{i-1}_result``, which is
bound by the surrounding ``Let``. The innermost ``Let`` body returns
``var("hop_{hops-1}_result")`` — the final hop's answer.

Semantics::

    let hop_0_result = fix(λself. ...)(document) in
    let hop_1_result = fix(λself. ...)(document) in
    ...
    let hop_{N-1}_result = fix(λself. ...)(document) in
    hop_{N-1}_result

Each hop re-traverses the full document; per-hop cost equality follows the
same Theorem-2 invariant as ``niah``. Total cost across all hops::

    ex.oracle_calls == hops * plan(...).predicted_calls

The factory closes over no Python state. The caller's executor env must
bind:

- ``<input_var>`` — the document string (default name: ``"document"``).
- ``size_bucket`` — a callable ``str → {"small","big"}`` deciding base case.
- ``<reduce_op_name>`` — a ``ReduceOp`` (or bare callable) for combining
  candidate answers (default name: ``"best"``). The standard helper is
  ``best_answer_op()`` from ``stdlib.long_context.niah``.

The per-hop result bindings (``hop_0_result``, ``hop_1_result``, ...) are
auto-managed by the ``Let`` chain — callers do NOT need to bind them.

Slice 3 limitations
-------------------

- ``hops`` is a fixed factory argument; confidence-gated dynamic
  termination is deferred to slice 4.
- Each hop re-traverses the full document. Sharing oracle calls across
  hops is out of scope.
- The ``{question}`` placeholder is substituted at *factory build time*
  (Python f-string), not at runtime; each hop's leaf prompt template is
  statically baked.
"""

from fsm_llm.lam import Term, let_, var

from ._recursive import _recursive_long_context

_HOP0_PROMPT_TEMPLATE = (
    "You are searching a portion of a long document to answer a question by "
    "finding the most relevant entity or fact.\n\n"
    "Question: {question}\n\n"
    "Text:\n{{P}}\n\n"
    "If the text contains an entity or fact relevant to the question, output "
    "it verbatim and nothing else. Otherwise output exactly: NOT_FOUND"
)

_HOPN_PROMPT_TEMPLATE = (
    "You are searching a portion of a long document for further detail "
    "building on a prior finding.\n\n"
    "Prior finding: {{{prev_var}}}\n"
    "Question: {question}\n\n"
    "Text:\n{{P}}\n\n"
    "If the text contains further detail relevant to the question and the "
    "prior finding, output it verbatim and nothing else. Otherwise output "
    "exactly: NOT_FOUND"
)


def multi_hop(
    question: str,
    hops: int,
    *,
    tau: int = 512,
    k: int = 2,
    reduce_op_name: str = "best",
    input_var: str = "document",
) -> Term:
    # DECISION D-S3-002: hops as Let-chain of independent Fix calls.
    # theorem2: ex.oracle_calls == hops * predicted.predicted_calls.
    # Confidence-gated dynamic hops deferred to slice 4. See decisions.md.
    """Build a multi-hop λ-term over a long document.

    Parameters
    ----------
    question:
        The question used at every hop. Baked into each hop's leaf prompt
        template at factory-build time (Python f-string).
    hops:
        Number of hop sweeps. Must be ``>= 1``. ``hops=1`` reduces to a
        single niah-shaped sweep (the degenerate case).
    tau:
        Leaf-size threshold (characters). Same semantics as ``niah``.
        Must be ``>= 1``.
    k:
        Branching factor for SPLIT. Default 2. Must be ``>= 2``.
    reduce_op_name:
        Name of the REDUCE op to look up in env. Default ``"best"`` — bind
        via ``best_answer_op()`` from ``stdlib.long_context.niah``. Used
        identically by every hop.
    input_var:
        Name of the env variable that holds the document string. Default
        ``"document"``. Every hop reads the SAME document.

    Returns
    -------
    A closed λ-term ready for ``Executor.run(program, env)``. The caller's
    env must bind ``input_var``, ``"size_bucket"``, and ``reduce_op_name``.
    Per-hop result bindings (``hop_0_result``, ``hop_1_result``, ...) are
    auto-managed by the ``Let`` chain.

    Raises
    ------
    ValueError
        If ``hops < 1``, ``tau < 1``, or ``k < 2``.

    Notes
    -----
    Cost equality across all hops (``ex.oracle_calls == hops *
    plan(...).predicted_calls``) holds when ``len(document) == τ · k^d``
    for some integer ``d ≥ 0``. The planner records one ``Plan`` entry
    per hop in ``ex.plans``.
    """
    if hops < 1:
        raise ValueError(f"hops must be >= 1, got {hops}")
    if tau < 1:
        raise ValueError(f"tau must be >= 1, got {tau}")
    if k < 2:
        raise ValueError(f"k must be >= 2 for non-degenerate recursion, got {k}")

    # Build each hop term. Hop 0 closes only over P; hop i>=1 also closes
    # over the prior hop's result variable.
    hop_terms: list[Term] = []
    for i in range(hops):
        if i == 0:
            leaf_prompt = _HOP0_PROMPT_TEMPLATE.format(question=question)
            hop_term = _recursive_long_context(
                leaf_prompt,
                tau=tau,
                k=k,
                reduce_op_name=reduce_op_name,
                input_var=input_var,
            )
        else:
            prev_var = f"hop_{i - 1}_result"
            # _HOPN_PROMPT_TEMPLATE uses {prev_var} as a build-time slot
            # for the variable NAME (so the runtime placeholder is
            # {hop_{i-1}_result}); {question} is build-time too; {{P}}
            # survives as the runtime chunk placeholder.
            leaf_prompt = _HOPN_PROMPT_TEMPLATE.format(
                question=question, prev_var=prev_var
            )
            hop_term = _recursive_long_context(
                leaf_prompt,
                tau=tau,
                k=k,
                reduce_op_name=reduce_op_name,
                input_var=input_var,
                extra_input_vars=(prev_var,),
            )
        hop_terms.append(hop_term)

    # Build the Let chain bottom-up: innermost body is the final hop's
    # result var; wrap outward binding hop_{N-1}_result, ..., hop_0_result.
    body: Term = var(f"hop_{hops - 1}_result")
    for i in range(hops - 1, -1, -1):
        body = let_(f"hop_{i}_result", hop_terms[i], body)
    return body


__all__ = ["multi_hop"]
