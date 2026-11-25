"""The three models of a simulated compile and the rule that keeps them apart."""

from __future__ import annotations

from dataclasses import dataclass

from fsm_llm.llm import LLMInterface, interface_model

from .exceptions import CODE_JUDGE_NOT_DISTINCT, SimulationError


@dataclass(frozen=True)
class Roles:
    """Who compiled, who ran and who judged: model names, ``None`` when unnamed.

    ``same_model_allowed`` says the distinctness check was waived by the caller.
    """

    compiler: str | None
    executor: str | None
    judge: str | None
    same_model_allowed: bool = False


def _name(llm: LLMInterface | None) -> str | None:
    return (interface_model(llm) or "").strip() or None


# DECISION plan-2026-11-25T074601-c720601d/D-009
# A guard against the default (judge is executor), not a proof: it compares names, so
# an alias or two servers behind one name pass. It fails closed: an unnamed model
# cannot be shown distinct. A compiler equal to the executor is allowed. Roles ride
# the report, the compile result and the receipt, NEVER ``LibraryRecord.provenance``;
# the refusal is raised at synthesizer construction and in ``repair_procedure`` only,
# never from ``reason_of``. Do NOT add a ``Reason`` or a policy field for it.
def roles_of(
    compiler: LLMInterface | None,
    executor: LLMInterface,
    judge: LLMInterface,
    *,
    allow: bool = False,
) -> Roles:
    """The ``Roles`` of a run, refusing a judge that is not distinct.

    Interface contract (caller: ``Simulator.roles``): ``judge`` must differ from
    ``executor`` and, when given, from ``compiler``: the same object, equal stripped
    model names (``interface_model``) or an unnamed side is refused with
    ``SimulationError(code=CODE_JUDGE_NOT_DISTINCT)``; the message names no model.
    ``allow=True`` skips the check. Never calls a model.
    """
    if not allow:
        for other in (executor, compiler):
            if other is None:
                continue
            a, b = _name(judge), _name(other)
            if judge is other or a is None or b is None or a == b:
                raise SimulationError(
                    "the judge must be a different model from the executor and the "
                    f"compiler ({CODE_JUDGE_NOT_DISTINCT}); name each model or pass "
                    "allow_same_model=True",
                    code=CODE_JUDGE_NOT_DISTINCT,
                )
    return Roles(_name(compiler), _name(executor), _name(judge), allow)
