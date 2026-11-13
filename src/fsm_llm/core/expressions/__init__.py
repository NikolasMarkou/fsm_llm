"""
JsonLogic Expression Evaluator for FSM-LLM.

This package provides a lightweight implementation of JsonLogic for evaluating
transition conditions in Finite State Machines (FSMs) driven by Large Language Models.

JsonLogic is a way to write logical expressions as JSON objects, making them:

1. Easy to read and write
2. Easy to serialize and store with FSM definitions
3. Expressive enough for complex conditions
4. Programming language-agnostic

This implementation is specifically tailored for FSM-LLM and includes:
    * All standard JsonLogic operators (comparison, logical, arithmetic)
    * Special operators for FSM context data access
    * Data validation operators for checking required fields

Example:
    Basic usage example::

        from fsm_llm.core.expressions import evaluate_logic

        # Sample condition: If customer is VIP and issue is high priority
        logic = {
            "and": [
                {"==": [{"var": "customer.status"}, "vip"]},
                {"==": [{"var": "issue.priority"}, "high"]}
            ]
        }

        # Sample context data
        context = {
            "customer": {"status": "vip"},
            "issue": {"priority": "high"}
        }

        # Evaluate the logic
        result = evaluate_logic(logic, context)
        # result is True

None / missing rule (one rule, applied by ``evaluate_logic``):

    * ``{"var": ...}`` resolves an absent key to ``None``, so "unset" and
      "explicitly None" behave the same in every operator below.
    * Arithmetic (``+``, ``-``, ``*``, ``/``, ``%``, ``min``, ``max``) on
      a ``None`` operand yields a module-private UNDEFINED value, which
      propagates through nested arithmetic. Every comparison, equality and
      membership operator (``==``, ``!=``, ``===``, ``!==``, ``<``, ``<=``,
      ``>``, ``>=``, ``in``, ``contains``) is ``False`` when an operand is
      UNDEFINED, and ``!``, ``!!``, ``and``, ``or``, ``if`` treat it as
      falsy. So ``{"<": [{"-": [total, discount]}, 100]}`` and
      ``{"!=": [{"-": [balance, paid]}, 0]}`` with the second operand unset
      are both ``False``. ``evaluate_logic`` returns a top-level UNDEFINED as
      ``None``, so a bare arithmetic condition on an unset operand is falsy
      (the transition does not fire). Any other operator (``cat``) sees it
      as ``None``. Gate arithmetic operands with ``requires_context_keys``
      when "unset" should mean something specific. An arithmetic operator
      that cannot compute (a non-numeric operand such as an extracted
      ``"N/A"``, a wrong arity) is UNDEFINED too, never ``False``
      (D-007 of plan-2026-11-05T134105-5ed71cf2); a division or modulo by
      zero still raises.
    * Ordering (``<``, ``<=``, ``>``, ``>=``) returns ``False`` when ANY
      operand is ``None``. ``-`` is unary only when it has exactly one
      operand (``{"-": [5]}`` is ``-5``); ``{"-": [5, null]}`` is undefined.
    * ``==`` / ``!=`` keep ``null == null`` True (JsonLogic), a ``None``
      never equals a non-``None`` value (so an unset var never equals the
      string ``"None"``), and a number equals a numeric string
      (``{"==": [1.0, "1"]}`` is True).
      Only plain ASCII decimal/scientific strings are numeric (``"1"``,
      ``"-2.5"``, ``"1e3"``, ``".5"``); Python-only forms (``"1_000"``,
      ``" 1 "``, ``"inf"``, ``"nan"``) and non-ASCII digits (Arabic-Indic,
      fullwidth) are not. ``null == null`` is about real nulls (a
      ``{"var"}`` miss), never about UNDEFINED.
      A ``bool`` operand is never numerically coerced by ``==``
      (``true == "1"`` is False; ``true == 1`` stays True via ``bool()``),
      and neither are two strings (``"01" == "1"`` is False, as in JS).
      ``<=`` and ``>=`` do NOT mirror that: they also accept numerically
      equal operands that ``==`` refuses, so ``"01" <= "1"`` and
      ``"01" >= "1"`` are True, and so is ``true <= "1"``, while ``"01" ==
      "1"`` and ``true == "1"`` are False. ``<`` and ``>`` coerce two numeric
      strings to numbers (``"10" < "9"`` is False). A gate on a numeric string
      therefore reads the same through ``<=``/``>=`` however the number is
      spelled (``>= "50"`` accepts ``"50.0"``); use ``==`` only where the
      spelling is the identity (zip codes, ids).
    * "Missing" means absent, ``None`` or ``""`` (``is_missing``). It is the
      one predicate behind ``missing``, ``missing_some`` and a transition
      condition's ``requires_context_keys``. ``0``, ``False`` and ``[]`` are
      present. ``has_context`` is different on purpose: pure key existence.
"""

from __future__ import annotations

from .access import get_var, is_missing, missing, missing_some
from .comparison import (
    greater,
    greater_or_equal,
    hard_equals,
    less,
    less_or_equal,
    soft_equals,
)
from .evaluator import evaluate_logic
from .logic_and_text import cat
from .registry import operations

__all__ = [
    "cat",
    "evaluate_logic",
    "get_var",
    "greater",
    "greater_or_equal",
    "hard_equals",
    "is_missing",
    "less",
    "less_or_equal",
    "missing",
    "missing_some",
    "operations",
    "soft_equals",
]
