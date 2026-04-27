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
  termination historical: shipped slice 6 — see ``multi_hop_dynamic``
  below (D-S3-001 anchor preservation pattern).
- Each hop re-traverses the full document. Sharing oracle calls across
  hops is out of scope.
- The ``{question}`` placeholder is substituted at *factory build time*
  (Python f-string), not at runtime; each hop's leaf prompt template is
  statically baked.
"""

from collections.abc import Callable
from typing import Any

from fsm_llm.lam import Term, app, let_, var

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


# M5 slice 6 — confidence-gated dynamic-hop variant.
#
# DECISION D-S6-001: host-callable orchestrator (not Fix-at-hop-level).
# Each hop stays a niah-shaped Fix; iteration is lifted to Python so the
# confidence gate can short-circuit. Slice-5 ``oracle_compare_op``
# precedent. Tight coupling to ``Executor._eval`` is documented.
#
# DECISION D-S6-002: Theorem-2 reformulated as upper bound:
# ``actual == actual_hops * predicted_calls`` (strict per actual hops);
# ``actual <= max_hops * predicted_calls`` (loose). Both reported via
# ``actual_hops_cell`` + ``executor.oracle_calls``.

_GATE_SENTINEL_NOT_FOUND = "NOT_FOUND"


def not_found_gate(
    *, sentinel: str = _GATE_SENTINEL_NOT_FOUND
) -> Callable[[Any, int], bool]:
    """Default gate: STOP iff hop result does NOT start with the sentinel.

    Returns a callable ``(result, hop_index) -> bool``. Returns ``True``
    (STOP) on a concrete answer; ``False`` (CONTINUE) on a sentinel-like
    result. Match is case-insensitive after ``strip()``.
    """
    sentinel_norm = sentinel.strip().upper()

    def _gate(result: Any, hop_index: int) -> bool:
        s = str(result).strip().upper()
        return not s.startswith(sentinel_norm)

    return _gate


def make_dynamic_hop_runner(
    executor: Any,
    question: str,
    *,
    max_hops: int,
    peer_env: dict[str, Any],
    confidence_gate: Callable[[Any, int], bool] | None = None,
    tau: int = 512,
    k: int = 2,
    reduce_op_name: str = "best",
    input_var: str = "document",
    actual_hops_cell: list[int] | None = None,
) -> Callable[[str], Any]:
    """Build a host-callable ``(document) -> final_answer`` orchestrator.

    Iterates up to ``max_hops`` independent niah-shaped sweeps. Stops
    early when ``confidence_gate(result, i)`` returns ``True``. Default
    gate is ``not_found_gate()``.

    Invokes ``executor._eval`` directly (NOT ``executor.run`` — the
    public entry point resets ``_oracle_calls``); this aggregates the
    counter across all hops. Per D-S6-001.

    The ``peer_env`` dict MUST contain ``size_bucket`` and the binding
    named by ``reduce_op_name`` (e.g. ``best``); the runner merges it
    with each hop's per-hop env (document + threaded ``hop_{i-1}_result``
    bindings). If ``actual_hops_cell`` is supplied, the runner writes
    ``actual_hops_cell[0]`` after each hop.

    Raises
    ------
    ValueError
        If ``max_hops < 1``, ``tau < 1``, or ``k < 2``.
    """
    if max_hops < 1:
        raise ValueError(f"max_hops must be >= 1, got {max_hops}")
    if tau < 1:
        raise ValueError(f"tau must be >= 1, got {tau}")
    if k < 2:
        raise ValueError(f"k must be >= 2 for non-degenerate recursion, got {k}")

    gate = confidence_gate if confidence_gate is not None else not_found_gate()
    sentinel_norm = _GATE_SENTINEL_NOT_FOUND.upper()

    def _run_hops(document: str) -> Any:
        env: dict[str, Any] = {input_var: document, **peer_env}
        last_concrete: Any = None
        result: Any = None
        for i in range(max_hops):
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
            # D-S6-001: bypass executor.run() so _oracle_calls aggregates.
            result = executor._eval(hop_term, env, _fix_depth=0)
            if actual_hops_cell is not None:
                actual_hops_cell[0] = i + 1
            try:
                if gate(result, i):
                    return result
            except Exception:
                pass  # E4: gate exception → continue
            env = {**env, f"hop_{i}_result": result}
            if not str(result).strip().upper().startswith(sentinel_norm):
                last_concrete = result
        return last_concrete if last_concrete is not None else result

    return _run_hops


def multi_hop_dynamic(
    question: str,
    *,
    max_hops: int,
    runner_var: str = "dynamic_hop_runner",
    input_var: str = "document",
) -> Term:
    """Build a confidence-gated dynamic-hop λ-term.

    Term shape: ``app(var(runner_var), var(input_var))``. Caller binds
    ``runner_var`` to the callable returned by
    ``make_dynamic_hop_runner(executor, question, max_hops=..., peer_env=...)``
    and ``input_var`` to the document string. Theorem-2 contract per
    D-S6-002.

    ``question`` is accepted for signature parity with ``multi_hop``;
    the runner already closes over it.

    Raises
    ------
    ValueError
        If ``max_hops < 1``.
    """
    if max_hops < 1:
        raise ValueError(f"max_hops must be >= 1, got {max_hops}")
    _ = question  # signature parity — runner closes over it
    return app(var(runner_var), var(input_var))


__all__ = [
    "multi_hop",
    "multi_hop_dynamic",
    "make_dynamic_hop_runner",
    "not_found_gate",
]
