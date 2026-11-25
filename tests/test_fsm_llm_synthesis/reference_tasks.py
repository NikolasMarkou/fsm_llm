"""The 16 reference tasks of the retired synthesis bench, and the user-turn loop that drives one.

Ported from ``scripts/synthesis_bench.py`` (plan baa37f40, step 4, D-012) when the bench, its
stdlib helpers and its recorded blocks were deleted. What stays is what an offline test reads:

- ``TASKS``, ``hand_fsm`` and ``host_definition``: the task table, the reference hand-written
  child of each task and the host every FSM arm shares. They are text for text what the bench
  built, because the kind-equality baseline holds ``bench/hand`` and ``bench/host`` records of
  them (``tests/test_fsm_llm/kind_corpus.py``): an edit moves the baseline.
- ``User``: the scripted user (a rule table, no model).
- ``grade`` (with ``holds``, ``squash``, ``NEGATORS``): one grader for the values a run returned.
- ``drive_fsm``: arms S (a compiled child) and H (a hand-written child) on one host, one
  bare ``API`` with a ``SynthesizeExecutor`` and one library, user turn by user turn.

Interface of ``drive_fsm(arm, task, lib_dir, llms)``: ``arm`` is ``SYNTH`` or ``HAND``;
``llms`` maps ``compile``, ``sim`` (arm S only) and ``run`` to ``LLMInterface`` objects;
``lib_dir`` is the library directory. It returns the row fields (``correct``, ``per_key``,
``checks``, ``values``, ``status``, ``reason``, ``source``, ``rounds``,
``compile_seconds``, ``turns``, ``confused_turns``, ``transcript``, ``child_states``). It raises
``AssertionError`` when the hand child fails the compiler's own gates; any other exception of
the run propagates.
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import Any, NamedTuple

SYNTH, HAND = "synth-fsm", "hand-fsm"

POLICY = {"max_turns": 10, "max_repair_rounds": 3, "sim_trials": 2, "max_sim_turns": 8, "max_simulation_seconds": 300, "max_simulation_calls": 400, "max_simulation_rounds": 2, "min_pass_rate": 0.5}  # fmt: skip
MAX_USER_TURNS = 14  # the bench's LIMITS["max_user_turns"]
FALLBACK, OPENER = "Sorry, could you say that again?", "Hello."


class Task(NamedTuple):
    id: str
    family: str  # collect | route | branch | retry
    goal: str
    gold: dict[str, str]  # return key -> gold value, compared after squash
    users: tuple[tuple[str, Any], ...]
    other: str
    spec: Any  # route: labels; branch: (labels, ask, ask); retry: (lo, hi)


_TRIP = (("business", "leisure"), "Ask for the name of the company the user works for.", "Ask for the destination the user is travelling to.")  # fmt: skip
_BUY = (("delivery", "pickup"), "Ask for the street address to deliver to.", "Ask which store the user will collect the order from.")  # fmt: skip
_TRIP_GOAL = "Find out whether the trip is business or leisure. For a business trip ask for the company name, for a leisure trip ask for the destination. Return the kind and the detail."  # fmt: skip
_BUY_GOAL = "Find out whether the order is for delivery or pickup. For delivery ask for the street address, for pickup ask for the store name. Return the method and the detail."  # fmt: skip
_RANGE = "If the number is outside that range, say so and ask again until it is valid."
# fmt: off
TASKS = tuple(Task(*row) for row in (
    ("c1", "collect", "Collect a visa applicant's passport number, travel dates and purpose of travel.",
     {"passport_no": "X1234567", "travel_dates": "2026-05-01", "purpose": "tourism"},
     ((r"passport", "My passport number is X1234567."), (r"date|when|arriv|depart", "I travel on 2026-05-01."), (r"purpose|reason|why", "The purpose is tourism.")), "", None),
    ("c2", "collect", "Collect a hotel guest's name, check-in date and room type for a booking.",
     {"guest_name": "Ada Lovelace", "checkin_date": "2026-07-14", "room_type": "double"},
     ((r"name", "My name is Ada Lovelace."), (r"check|date|arriv", "I check in on 2026-07-14."), (r"room", "I would like a double room.")), "", None),
    ("c3", "collect", "Collect a support ticket's order id, the customer's contact email and the affected product.",
     {"order_id": "ORD-5521", "contact_email": "ada@example.com", "product": "router"},
     ((r"order", "My order id is ORD-5521."), (r"email|contact|reach", "You can reach me at ada@example.com."), (r"product|item", "The product is the router.")), "", None),
    ("c4", "collect", "Collect a clinic patient's name, birth year and main symptom.",
     {"patient_name": "Grace Hopper", "birth_year": "1985", "symptom": "migraine"},
     ((r"name", "My name is Grace Hopper."), (r"born|birth|year", "I was born in 1985."), (r"symptom|problem|complain", "My main symptom is migraine.")), "", None),
    ("r1", "route", "Ask what the customer needs and route them to billing, technical, sales or other.", {"department": "billing"}, ((r"", "My invoice shows a charge twice this month."),), "", ("billing", "technical", "sales", "other")),
    ("r2", "route", "Ask what the problem is and rate its priority as urgent, normal or low.", {"priority": "urgent"}, ((r"", "The production server is down and customers cannot log in."),), "", ("urgent", "normal", "low")),
    ("r3", "route", "Ask what the customer wants and file it as refund, shipping, complaint or compliment.", {"topic": "shipping"}, ((r"", "Where is my parcel? It was due on Monday."),), "", ("refund", "shipping", "complaint", "compliment")),
    ("r4", "route", "Ask what the employee needs and send them to hr, legal, it or finance.", {"team": "it"}, ((r"", "I need a new charger for my laptop."),), "", ("hr", "legal", "it", "finance")),
    ("b1", "branch", _TRIP_GOAL, {"trip_type": "business", "detail": "Acme"},
     ((r"business|leisure", "It is a business trip."), (r"company|employer", "My company is Acme.")), r"destination|where are you|which city", _TRIP),
    ("b2", "branch", _TRIP_GOAL, {"trip_type": "leisure", "detail": "Lisbon"},
     ((r"business|leisure", "It is a leisure trip."), (r"destination|where|city", "I am going to Lisbon.")), r"company|employer", _TRIP),
    ("b3", "branch", _BUY_GOAL, {"method": "delivery", "detail": "12 Elm Street"},
     ((r"delivery|pickup", "I would like delivery."), (r"address|street", "My address is 12 Elm Street.")), r"store|branch|which shop", _BUY),
    ("b4", "branch", _BUY_GOAL, {"method": "pickup", "detail": "Harbor"},
     ((r"delivery|pickup", "I want pickup."), (r"store|branch|shop", "The Harbor store.")), r"address|street", _BUY),
    ("y1", "retry", f"Ask for the party size, from 1 to 8 guests. {_RANGE} Return party_size.", {"party_size": "6"}, ((r"party|guests|how many", ("We are 12 people.", "We are 6 people.")),), "", (1, 8)),
    ("y2", "retry", f"Ask how many tickets to buy, from 1 to 10. {_RANGE} Return quantity.", {"quantity": "4"}, ((r"tickets|how many|quantity", ("I want 25 tickets.", "I want 4 tickets.")),), "", (1, 10)),
    ("y3", "retry", f"Ask how many nights to stay, from 1 to 14. {_RANGE} Return nights.", {"nights": "5"}, ((r"nights|stay|how long", ("I will stay 30 nights.", "I will stay 5 nights.")),), "", (1, 14)),
    ("y4", "retry", f"Ask how many seats to reserve, from 1 to 6. {_RANGE} Return seats.", {"seats": "2"}, ((r"seats|how many", ("I need 9 seats.", "I need 2 seats.")),), "", (1, 6)),
))
# fmt: on


class User:
    """The scripted user, no model: ``reply`` is the next utterance of EVERY rule whose regex
    matches the assistant text (the last repeats), else ``FALLBACK`` (``confused``). ``late`` keeps
    the texts answered AFTER rule 0 (the branch value) had been given: the one branch-check input."""

    def __init__(self, rules: tuple[tuple[str, Any], ...]) -> None:
        self.rules = [
            (re.compile(p, re.I), (s,) if isinstance(s, str) else tuple(s))
            for p, s in rules
        ]
        self.used, self.confused = [0] * len(rules), 0  # used[i]: answers rule i gave
        self.late: list[str] = []

    def reply(self, text: str) -> str:
        self.late += [text] * bool(self.used[0])
        said = []
        for i, (pattern, lines) in enumerate(self.rules):
            if pattern.search(text):
                said.append(lines[min(self.used[i], len(lines) - 1)])
                self.used[i] += 1
        self.confused += not said
        return " ".join(said) or FALLBACK


# fmt: off
def _edge(target: str, keys: Any, logic: Any = None) -> dict[str, Any]:
    cond = {"description": f"to {target}", "requires_context_keys": list(keys)} | ({"logic": logic} if logic else {})
    return {"target_state": target, "description": f"to {target}", "priority": 1, "conditions": [cond]}


def _state(sid: str, say: str, fields: Any = (), edges: Any = (), extract: str = "") -> dict[str, Any]:
    state: dict[str, Any] = {"id": sid, "description": sid, "purpose": f"{sid} step", "response_instructions": say}
    state |= {"extraction_instructions": extract} if extract else {}
    state |= {"field_extractions": [{"field_name": k, "field_type": t, "extraction_instructions": h} for k, t, h in fields]} if fields else {}
    return state | ({"transitions": list(edges)} if edges else {})


def hand_fsm(task: Task) -> dict[str, Any]:
    """The reference child of *task*, one builder per family; it passes the compiler's
    own gates (``check_fixture``) and every task offline."""
    keys = tuple(task.gold)
    key, var = keys[0], {"var": keys[0]}
    if task.family == "collect":
        words = [k.replace("_", " ") for k in keys]
        fields = [(k, "str", f"the {w}") for k, w in zip(keys, words, strict=True)]
        states = {"ask": _state("ask", "Ask the user for: " + ", ".join(words) + ".", fields, [_edge("done", keys)])}
    elif task.family == "route":
        hint = "the one of these that fits what the user needs: " + ", ".join(task.spec)
        states = {"ask": _state("ask", "Ask the user what they need.", [(key, "str", hint)], [_edge("done", keys, {"in": [var, list(task.spec)]})])}
    elif task.family == "branch":
        (one, two), ask_one, ask_two = task.spec
        edges = [_edge("first", [key], {"==": [var, one]}), _edge("second", [key], {"==": [var, two]})]
        own = "Take only the user's answer to that question; if the user has not answered it yet, leave it empty."  # the state's own question, then this
        done = [_edge("done", ["detail"])]
        states = {"ask": _state("ask", f"Ask whether it is {one} or {two}.", [(key, "str", f"exactly one of: {one}, {two}")], edges),
                  "first": _state("first", ask_one, [("detail", "str", f"{ask_one} {own}")], done), "second": _state("second", ask_two, [("detail", "str", f"{ask_two} {own}")], done)}
    else:
        lo, hi = task.spec
        ask = f"Ask the user for the {key.replace('_', ' ')} (from {lo} to {hi}). If it is outside that range, say it is not allowed and ask again."
        # a later-turn correction comes only from the bulk pass, which needs these instructions
        why = f"Extract {key}: the whole number the user gives, even when it is out of range."
        edge = _edge("done", keys, {"and": [{">=": [var, lo]}, {"<=": [var, hi]}]})
        states = {"ask": _state("ask", ask, [(key, "int", "a whole number")], [edge], why)}
    states["done"] = _state("done", "Thank the user and say it is noted.")
    return {"name": f"Bench {task.id}", "description": task.goal[:250], "initial_state": "ask", "states": states}


def host_definition(task: Task) -> dict[str, Any]:
    """The host every FSM arm shares: welcome, a silent ``synthesize`` state, then
    ``wrapup`` on ``status == ok`` or ``sorry`` on ``failed`` or ``aborted``."""
    status = {"var": "synthesis_result.status"}

    def edge(target: str, priority: int, logic: Any) -> dict[str, Any]:
        cond = {"description": target, "requires_context_keys": ["synthesis_result"], "logic": logic}
        return {"target_state": target, "description": target, "priority": priority, "conditions": [cond]}

    work = _state("work", "", edges=[edge("wrapup", 1, {"==": [status, "ok"]}), edge("sorry", 2, {"in": [status, ["failed", "aborted"]]})])
    work["synthesize"] = {"goal": task.goal, "inputs": [], "return_keys": list(task.gold), "on_failure": "sorry"}
    states = {"welcome": _state("welcome", "Greet the user in one short sentence.", edges=[{"target_state": "work", "description": "go"}]), "work": work,
              "wrapup": _state("wrapup", "Tell the user it is done."), "sorry": _state("sorry", "Apologise briefly.")}
    return {"name": "Bench host", "description": "Welcomes, runs the procedure, wraps up.", "persona": "A calm, brief assistant", "initial_state": "welcome", "states": states}


def squash(value: Any) -> str:
    """Case, whitespace and punctuation dropped; a whole float reads as an int."""
    value = int(value) if isinstance(value, float) and value.is_integer() else value
    return re.sub(r"[\W_]+", "", str(value)).lower()


NEGATORS = frozenset("not no never nor neither without except instead rather or and but".split())


def holds(gold: str, value: Any, banned: frozenset[str]) -> bool:
    """*value* equals *gold* after ``squash``, or its word tokens hold the gold's
    as ONE contiguous run with at most two other tokens, none of them in *banned*."""
    have, want = (re.findall(r"\w+", str(s).lower()) for s in (value, gold))
    spans = [(have[:i], have[i + len(want) :]) for i in range(len(have) - len(want) + 1) if have[i : i + len(want)] == want]  # fmt: skip
    return squash(value) == squash(gold) or any(len(a + b) <= 2 and not banned & {*a, *b} for a, b in spans)  # fmt: skip


def grade(task: Task, values: dict[str, Any] | None, late: list[str], used: int) -> dict[str, Any]:
    """Correct = every key is its gold (``holds``, the grader of every arm) AND the family check: branch,
    no question about the other branch after the branch value; retry, the rule answered twice. ``values``
    is what each key came back as, cut at 80 characters, for a later offline re-grade."""
    values = values or {}
    labels = task.spec if task.family == "route" else task.spec[0] if task.family == "branch" else ()
    banned = NEGATORS | {w for lab in labels for w in re.findall(r"\w+", lab.lower())}  # a list of the task's labels fails
    per_key = {k: values.get(k) is not None and holds(g, values[k], banned) for k, g in task.gold.items()}
    other = bool(task.other) and any(re.search(task.other, a, re.I) for a in late)
    checks = {"other_branch": not other, "reask": task.family != "retry" or used >= 2}
    kept = {k: None if values.get(k) is None else str(values[k])[:80] for k in task.gold}
    return {"correct": all(per_key.values()) and all(checks.values()), "per_key": per_key, "checks": checks, "values": kept}
# fmt: on


def drive_fsm(
    arm: str, task: Task, lib_dir: Path, llms: dict[str, Any]
) -> dict[str, Any]:
    """Arms S and H on one host, one bare ``API`` with a ``SynthesizeExecutor``, one library
    path: user turns until the conversation ends or ``MAX_USER_TURNS``. Returns the row's FSM
    fields."""
    import fsm_llm.synthesis as syn
    from fsm_llm import API
    from fsm_llm.simulation import Simulator
    from fsm_llm.synthesis.gates import GATES
    from tests.test_fsm_llm_synthesis.rig import ExecutorDriver

    hand = arm == HAND
    policy = syn.SynthesisPolicy(**POLICY, require_simulation=not hand)
    library, keys = syn.FSMLibrary(lib_dir), tuple(task.gold)
    if hand:
        # H is seeded through the compiler's own gates, with gates_passed = GATES
        # only: G6b did not run on a hand-written child and is never claimed. Do
        # NOT write G6b here and do NOT compile: H makes zero strong calls.
        gates = syn.run_gates(hand_fsm(task), policy, return_keys=keys)
        assert gates.definition is not None, (
            f"{task.id}: the hand child fails the gates"
        )
        library.put(gates.definition, goal=task.goal, inputs_required=(), return_keys=keys, synthesizing_model="hand-written", gates_passed=GATES, lifecycle="verified")  # fmt: skip
    sim = None if hand else Simulator(llms["sim"], llms["compile"], allow_same_model=True)  # fmt: skip
    synth = syn.FSMSynthesizer(llms["compile"], policy=policy, library=library, simulator=sim)  # fmt: skip
    events: list[tuple[str, dict[str, Any]]] = []
    host = host_definition(task)
    executor = syn.SynthesizeExecutor(synth, trace_sink=lambda e, f: events.append((e, f)))  # fmt: skip
    api = API.from_definition(definition=host, llm_interface=llms["run"], executors={"synthesize": executor})  # fmt: skip
    driver = ExecutorDriver(api)
    conv, _ = api.start_conversation()
    user, said, turns = User(task.users), OPENER, []
    try:
        for _ in range(MAX_USER_TURNS):
            reply = driver.converse(conv, said)
            turns.append([said[:300], reply[:300]])
            if api.has_conversation_ended(conv):
                break
            said = user.reply(reply)
        data = api.get_data(conv)
    finally:
        with contextlib.suppress(Exception):
            api.end_conversation(conv)
    result = data.get("synthesis_result") or {}
    got = next((f for e, f in events if e in ("obtained", "failed")), {})
    grades = grade(task, {k: data.get(k) for k in keys}, user.late, user.used[0])
    # fmt: off
    return {**grades, "status": result.get("status"), "reason": result.get("reason"), "source": got.get("source"),
            "rounds": [c.split(",") for c in (got.get("rounds") or "").split(";") if c], "compile_seconds": got.get("seconds", 0.0),
            "turns": len(turns), "confused_turns": user.confused, "transcript": turns,
            "child_states": sorted({f["state"] for e, f in events if e == "stepped" and f.get("state")})}
    # fmt: on
