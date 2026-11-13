"""Key-name decisions: the internal-prefix test and `is_forbidden_context_entry`.

`has_internal_prefix` is the one internal-key test; `is_forbidden_context_entry`
is the one secret-looking-entry decision, with the name-scan data of layer 2,
the policy-suffix value judge and the container walk it needs."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .patterns import (
    _KEY_TRIGGER,
    _PASSWORD_POLICY_SUFFIXES,
    _PLAIN_WORD_END,
    _SAFE_TOKEN_QUALIFIERS,
    _SEP_RUN,
    _TOKEN_TRIGGER,
    _WORD_END,
    COMPILED_FORBIDDEN_CONTEXT_PATTERNS,
)
from .shape_data import _VALUE_SCAN_LIMIT
from .shapes import _looks_like_credential_value

# --------------------------------------------------------------
# Internal Context Key Prefixes
# --------------------------------------------------------------

INTERNAL_KEY_PREFIXES = ["_", "system_", "internal_", "__"]


# DECISION plan-2026-07-19T191147-4b664252/D-009 [STALE]
# This helper exists because the identical `any(key.startswith(p) ...)`
# expression was copy-pasted to FIVE call sites (context.py, fsm.py,
# core/definitions x2, prompts.py) and NONE of them case-folded -- so
# `SYSTEM_foo` bypassed the filter at every site at once (F-13).
# Do NOT re-inline `key.startswith(...)` at a call site, and do NOT hardcode
# the prefix list: five hand-maintained copies is what caused this defect.
# Import this helper instead. See decisions.md D-009.
def has_internal_prefix(key: str, prefixes: Iterable[str] | None = None) -> bool:
    """Return True if ``key`` carries an internal context-key prefix.

    Matching is CASE-INSENSITIVE: ``SYSTEM_foo`` and ``Internal_x`` are
    internal keys just as ``system_foo`` and ``internal_x`` are.

    Args:
        key: The context key to test.
        prefixes: Prefixes to test against. Defaults to
            :data:`INTERNAL_KEY_PREFIXES`. Callers with a configurable
            prefix list (e.g. ``PromptConfig.internal_key_prefixes``) pass
            their own; the case-folding rule is applied either way.

    Returns:
        True if the key should be treated as internal (and therefore hidden
        from user-visible data and from LLM prompts). Never raises for any
        ``str`` *key*, including a hostile subclass (D-006).
    """
    if prefixes is None:
        prefixes = INTERNAL_KEY_PREFIXES
    # DECISION plan-2026-07-20T103203-b8a6b855/D-006 [STALE] -- unbound `str.lower`, not
    # `key.lower()`: a `str` SUBCLASS overriding `.lower` ran arbitrary code here
    # and out of every prompt builder (probed at `a414601`). `lowered` is then an
    # exact `str`, so bound `.startswith` is safe. Do NOT re-bind either call.
    lowered = str.lower(key)
    return any(lowered.startswith(str.lower(prefix)) for prefix in prefixes)


# Names for which layer 2 is consulted: a `key`/`keys` trigger that layer 1 did
# not already resolve. The concatenated form is INCLUDED on purpose -- `passkey`
# is a credential and `monkey_species` is not, and it is the VALUE that separates
# them, which is the whole point of this layer.
_VALUE_SCAN_NAME_RE = re.compile(
    rf"(?:^|.*[\W_])[a-z0-9]*{_SEP_RUN}{_KEY_TRIGGER}{_WORD_END}", re.IGNORECASE
)

# Genuinely ambiguous credential abbreviations that contain no `key` substring at
# all, so no `key`-shaped pattern can structurally reach them. These are NOT
# stripped on the name -- `pk`/`sk` are DynamoDB partition/sort keys as often as
# they are secret/public keys -- they are merely referred to layer 2, which
# decides on the value. This is the clearest single case of the value separating
# what the name cannot.
_AMBIGUOUS_CREDENTIAL_ABBREVIATIONS = frozenset(
    {"sk", "pk", "mk", "dk", "ek", "psk", "skey", "ckey", "privkey", "seckey"}
)

# DECISION plan-2026-07-20T040150-876e7164/D-021 [STALE]
# Names for which layer 2 decides the TOKEN arm. This regex is CHARACTER-FOR-
# CHARACTER the middle alternative D-015 shipped inside
# `FORBIDDEN_CONTEXT_PATTERNS` -- a `<qualifier>_token` shape whose qualifiers
# do not ALL decompose into `_SAFE_TOKEN_QUALIFIERS`. Keeping it identical is
# the point of the change and the honest statement of it: the set of names is
# unchanged, and what changed is that the arm now ASKS THE VALUE instead of
# stripping unconditionally. `billed_tokens`, `prefill_tokens`, `subword_token`
# and `truncation_token` are in this set and are now kept.
_TOKEN_VALUE_SCAN_NAME_RE = re.compile(
    rf"(?!(?:(?:{_SAFE_TOKEN_QUALIFIERS}){_SEP_RUN})+{_TOKEN_TRIGGER}{_PLAIN_WORD_END})"
    rf"(?:^|.*[\W_])[a-z0-9]+{_SEP_RUN}{_TOKEN_TRIGGER}{_WORD_END}",
    re.IGNORECASE,
)


# DECISION plan-2026-09-21T203800-8a03483a/D-012
# DECISION plan-2026-09-21T203800-8a03483a/D-031
# Credential names with no `password`/`secret`/`key`/`token` substring, which
# layer 1 never reached (audit D5). Each term matches ONLY as a whole
# `[\W_]`-delimited segment (camelCase via the snake re-test below): `pin`
# strips `pin`/`user_pin`/`newPin` but never `shipping`/`spin`/`opinion`, and
# `pass` never `passenger`/`compass`/`bypass`/`passport`. Do NOT drop the
# segment anchors to catch `dbpass`/`userpin` (that re-opens every substring
# false positive above); those stay open by design, like `dbpassword`.
# Two KEEP rules bound the over-strip, both reused rather than invented:
#   1. a tail that decomposes ENTIRELY into `_PASSWORD_POLICY_SUFFIXES`
#      (`pin_attempts`, `otp_enabled`, `criteria_pass_count`) is metadata,
#      by the same measured rule `password_*` already uses (D-030);
#   2. a `bool` value is never credential material (`all_criteria_pass:
#      True`, a pass/fail flag). Numbers are NOT exempt: a PIN,
#      CVV, OTP or card number is routinely stored as an int.
# A name-only caller (`value=None`) strips. See decisions.md D-012, D-031.
#
# DECISION plan-2026-09-21T203800-8a03483a/D-046
# Rule 1 is NAME-only no longer: a policy-suffix name KEEPS only a value that
# cannot be credential material (`_policy_tail_value_is_credential`), so
# `pin_enabled: "1234"` and `authorization_status: "Bearer ..."` strip while
# `pin_attempts: 3` and `pin_status: "locked"` stay. A term may carry a digit
# suffix (`cvv2`, `pin2`), `cvc` is a term, and an acronym run is split
# (`PINCode` -> `PIN_Code`) for THIS rule only. Do NOT apply the acronym split
# to layer 1 (unmeasured there) and do NOT make the policy-suffix tail
# name-only again. See decisions.md D-046.
_CREDENTIAL_NAME_TERMS = (
    r"passwd|pwd|pass|passcode|passphrase|pin|otp|mfa[\W_]?code|cvv|cvc|ssn"
    r"|credit[\W_]?card|card[\W_]?number|cookie|jwt|bearer|authorization"
    r"|auth[\W_]?header|recovery[\W_]?code"
)
_CREDENTIAL_NAME_HEAD = rf"(?:^|.*[\W_])(?:{_CREDENTIAL_NAME_TERMS})(?:s|\d+)?"
# Matches a credential name whose tail is NOT only policy suffixes.
_CREDENTIAL_NAME_RE = re.compile(
    rf"{_CREDENTIAL_NAME_HEAD}"
    rf"(?!(?:[-_.]?(?:{_PASSWORD_POLICY_SUFFIXES}))+$){_PLAIN_WORD_END}",
    re.IGNORECASE,
)
# Matches every credential name, policy-suffix tails included.
_CREDENTIAL_NAME_ANY_RE = re.compile(
    rf"{_CREDENTIAL_NAME_HEAD}{_PLAIN_WORD_END}", re.IGNORECASE
)
# Upper-case run followed by a capitalised word: `PINCode` -> `PIN_Code`.
_ACRONYM_BOUNDARY = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
# DECISION plan-2026-09-21T203800-8a03483a/D-052
# What a policy-suffix credential name (`pin_status`, `cookie_max_age`) may
# hold and still reach a prompt. Do NOT reopen the D-046 allowances these
# replace: "any number below 1,000" kept every 3-digit CVV (`cvv_status:
# 737`) and "any plain letter words" kept letter-only secrets (`pin_status:
# "hunter"`). A number is kept only under a count suffix (below
# `_POLICY_TAIL_MAX_COUNT`) or a duration suffix (any size), a string only
# when EVERY word is in `_POLICY_TAIL_STATE_WORDS` or it is an ISO date, and a
# dict (or a list of dicts / safe scalars) is kept so the walker filters its
# inner keys by their own names. See decisions.md D-052.
_POLICY_TAIL_RE = re.compile(
    rf"{_CREDENTIAL_NAME_HEAD}(?P<tail>(?:[-_.]?(?:{_PASSWORD_POLICY_SUFFIXES}))+)$",
    re.IGNORECASE,
)
# Tail endings that make a number a count (small) or a duration (any size).
_POLICY_TAIL_COUNT_END_RE = re.compile(
    r"(?:^|[-_.])(?:attempts?|retry|retries|count|length|len|min|max|minimum"
    r"|maximum|limit|days?|score|version)$",
    re.IGNORECASE,
)
_POLICY_TAIL_DURATION_END_RE = re.compile(
    r"(?:^|[-_.])(?:age|ttl|timeout|expires[-_.]?in)$", re.IGNORECASE
)
_POLICY_TAIL_MAX_COUNT = 1000
# Closed vocabulary of state/outcome words a policy-suffix value may use.
_POLICY_TAIL_STATE_WORDS = frozenset(
    {
        "enabled", "disabled", "active", "inactive", "on", "off", "true",
        "false", "yes", "no", "set", "unset", "required", "optional",
        "pending", "verified", "expired", "locked", "unlocked", "none",
        # Outcome and consent words already pinned by the step-8 guards
        # (`pass_status: "ok"`, `authorization_status: "approved"`,
        # `cookie_banner: "shown"`, `cookie_settings: "accept all"`).
        "ok", "approved", "denied", "rejected", "accepted", "declined",
        "shown", "hidden", "sent", "failed", "passed", "complete",
        "completed", "blocked", "allowed", "valid", "invalid", "accept",
        "reject", "all", "only", "necessary", "essential",
        # Delivery channels (`mfa_code_sent: "sms"`).
        "sms", "email", "voice", "push",
    }
)  # fmt: skip
_POLICY_TAIL_WORD_SPLIT = re.compile(r"[ _-]")
_POLICY_TAIL_WORD_VALUE_MAX_CHARS = 40
_ISO_DATE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?",
    re.ASCII,
)


def _token_value_is_credential(value: object, name: str = "") -> bool:
    """Layer-2 verdict for the TOKEN arm: is *value* credential material?

    Differs from :func:`_looks_like_credential_value` in its DEFAULT, and only
    in that. The token arm's safe space is dominated by metering counts, so a
    numeric value is decisive evidence AGAINST a credential; everything that is
    not a number and not a string is decided fail-CLOSED, which is exactly the
    posture the name-only arm had before this layer existed.

    The pre-design probe that authorised this split measured the signal at 18/18
    metering names numeric and 0/18 credential-shaped, and 8/8 tokenizer symbols
    short and low-entropy. It ALSO measured the case that breaks it -- 4 of 6
    pagination cursors are indistinguishable from bearer tokens by value -- and
    those are handled by name in ``_SAFE_TOKEN_QUALIFIERS`` instead.

    NEVER logs *value*. Never raises, for ANY *value* and ANY *name* of ANY
    type -- it delegates to :func:`_looks_like_credential_value`, so it
    inherits that function's guards and nothing else (D-006, D-009).
    """
    # A number cannot carry key material. `bool` is an `int` subclass and is
    # covered here on purpose: `True` carries no material either.
    if isinstance(value, (bool, int, float)):
        return False
    # `None`, `bytes`, dicts, lists and every other type: fail CLOSED. This is
    # what preserves the pre-layer posture for a name-only caller, which gets
    # `value=None` and therefore the old unconditional strip.
    if not isinstance(value, str):
        return True
    return _looks_like_credential_value(value, name)


def _policy_tail_scalar_is_credential(value: object, tail: str) -> bool:
    """Scalar half of :func:`_policy_tail_value_is_credential` (no containers)."""
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        if _POLICY_TAIL_DURATION_END_RE.search(tail):
            return False
        if _POLICY_TAIL_COUNT_END_RE.search(tail):
            return not abs(value) < _POLICY_TAIL_MAX_COUNT
        return True
    if type(value) is not str or len(value) > _POLICY_TAIL_WORD_VALUE_MAX_CHARS:
        return True
    if _ISO_DATE_RE.fullmatch(value):
        return False
    words = _POLICY_TAIL_WORD_SPLIT.split(str.lower(value))
    return not all(word in _POLICY_TAIL_STATE_WORDS for word in words)


def _policy_tail_value_is_credential(value: object, tail: str) -> bool:
    """Is *value* credential material under a policy-suffix credential name
    whose policy tail is *tail* (``"_status"`` for ``pin_status``)?

    Contract:
        - KEEP (False): ``None``; a ``bool``; a number when *tail* ends in a
          duration suffix (``age``, ``ttl``, ``timeout``, ``expires_in``) or
          ends in a count suffix and is below ``_POLICY_TAIL_MAX_COUNT`` in
          magnitude; an exact ``str`` of at most 40 chars that is an ISO date
          or whose every word is in ``_POLICY_TAIL_STATE_WORDS``; a ``dict``
          (the walker then filters its keys by their own names); a
          ``list``/``tuple`` whose every element is a ``dict`` or a scalar
          this rule keeps.
        - Everything else is True (strip): other numbers, digit or letter
          strings outside the word set, ``str`` subclasses, nested lists,
          other types.
        - Never raises; never calls anything on a ``str`` subclass value.
    """
    if isinstance(value, dict):
        return False
    if isinstance(value, (list, tuple)):
        return not all(
            isinstance(item, dict)
            or (
                not isinstance(item, (list, tuple))
                and not _policy_tail_scalar_is_credential(item, tail)
            )
            for item in value
        )
    return _policy_tail_scalar_is_credential(value, tail)


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Bounds of the key-arm container judge below. A context value can be arbitrarily
# large and this runs on every prompt build with the per-conversation lock held.
_CONTAINER_NODE_BUDGET = 256
_CONTAINER_MAX_DEPTH = 16


# DECISION plan-2026-11-05T134105-5ed71cf2/D-004
# The key arm used to KEEP every non-`str` value, so `partner_key: ["sk_live_..."]`,
# a one-key dict or `b"-----BEGIN ..."` reached the prompt (audit F-SU-02); walkers
# test names of dict keys only, never a container held under an ambiguous name.
# This judges the STRING LEAVES with the same value test, and fails CLOSED when it
# cannot finish (node budget, depth bound, a cycle, a container subclass that
# raises). Do NOT mirror the token arm (strip every container under an ambiguous
# key name): that over-strips honest `order_key: [...]` lists the D-019 corpus
# keeps. Do NOT loosen the exact-`str` rule of `_looks_like_credential_value`
# (D-006): a `str` subclass leaf is still credential material. Do NOT apply this
# to the token arm, whose non-`str` default is already fail-closed.
def _key_value_is_credential(value: object, name: str) -> bool:
    """Layer-2 verdict for the key arm: does *value* hold credential material?

    Interface contract:
        - ``value``: any object. An exact or subclassed ``str`` is judged by
          :func:`_looks_like_credential_value` (unchanged). A ``bytes``/
          ``bytearray`` is judged on its first ``_VALUE_SCAN_LIMIT`` bytes decoded
          as latin-1, read through ``memoryview`` so a subclass cannot hook it.
          A ``list``/``tuple``/``set``/``frozenset``/``dict`` (dict VALUES; its
          keys are filtered by the walkers) is walked for string and bytes
          leaves. Every other type is KEPT (False), as before.
        - ``name``: the context key, forwarded to the leaf judge.
        - Returns True when any leaf is credential-shaped, or when the walk
          cannot finish: more than ``_CONTAINER_NODE_BUDGET`` nodes, deeper than
          ``_CONTAINER_MAX_DEPTH``, a reference cycle, or any exception.
        - Never raises; never logs or returns *value*.
    """
    try:
        budget = [_CONTAINER_NODE_BUDGET]
        return _walk_for_credential(value, name, 0, set(), budget)
    except Exception:
        return True


def _walk_for_credential(
    value: object, name: str, depth: int, active: set[int], budget: list[int]
) -> bool:
    """Recursive half of :func:`_key_value_is_credential` (raises are caught there)."""
    budget[0] -= 1
    if budget[0] < 0:
        return True
    if isinstance(value, str):
        return _looks_like_credential_value(value, name)
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(memoryview(value)[:_VALUE_SCAN_LIMIT])
        return _looks_like_credential_value(raw.decode("latin-1"), name)
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        if depth >= _CONTAINER_MAX_DEPTH or id(value) in active:
            return True
        active.add(id(value))
        try:
            items = value.values() if isinstance(value, dict) else value
            for item in items:
                if _walk_for_credential(item, name, depth + 1, active, budget):
                    return True
        finally:
            active.discard(id(value))
    return False


def is_forbidden_context_entry(key: object, value: object = None) -> bool:
    """Return True if this context ENTRY must never reach an LLM prompt.

    The single decision point for the context-key security filter, shared by
    ``context.clean_context_keys``, ``prompts.BasePromptBuilder``'s
    ``_is_forbidden_context_key`` and ``context.filtering.redact_context``'s log redaction. It combines:

    * **Layer 1 (name)** -- ``COMPILED_FORBIDDEN_CONTEXT_PATTERNS``. Decides on
      the name alone and is authoritative in the STRIP direction.
    * **Layer 2 (value)** -- consulted for the two shapes layer 1 leaves open,
      with OPPOSITE defaults because their safe spaces differ:

      - the ambiguous ``<qualifier>_key`` shape plus
        ``_AMBIGUOUS_CREDENTIAL_ABBREVIATIONS`` go to
        :func:`_looks_like_credential_value` and default to KEEP;
      - the non-allowlisted ``<qualifier>_token`` shape goes to
        :func:`_token_value_is_credential` and defaults to STRIP.

      The key arm also judges the string leaves of a list, tuple, set, dict or
      ``bytes`` value (:func:`_key_value_is_credential`, bounded, fail-closed).
      A digit run directly after ``key``/``token``/``secret``/``credential``
      is part of the same name (``key1``, ``private_key2``).

    Args:
        key: The context key. A non-``str`` key carries no name to match; the
            caller is responsible for logging that, since only the caller knows
            whether it is about to build a prompt or write a log line.
        value: The value stored under *key*. Optional so that name-only callers
            keep working -- but note the asymmetry that costs: omitting it makes
            the ``key`` arm KEEP and the ``token`` arm STRIP, so
            ``is_forbidden_context_entry("billed_tokens")`` is True while
            ``is_forbidden_context_entry("billed_tokens", 137)`` is False. Every
            caller in this package passes the value; a name-only caller gets the
            pre-D-021 behaviour on the token arm, which is the safe direction.

    Returns:
        True if the entry must be stripped or redacted. Never raises, for ANY
        *key* and *value* including hostile ``str`` subclasses of either
        (D-006), and never re-raises with the value attached.
    """
    if not isinstance(key, str):
        return False

    if any(pattern.match(key) for pattern in COMPILED_FORBIDDEN_CONTEXT_PATTERNS):
        return True

    # DECISION plan-2026-09-19T175721-21cd7f8e/D-040
    # camelCase names (`newPassword`, `clientSecret`) carry no `_` boundary for
    # the layer-1 patterns, so they are re-tested in snake_case form. Do NOT
    # widen the patterns to plain substrings instead (that strips `tokenizer`,
    # `secretary`, `keyboardLayout`); `dbpassword`/`mysecret` stay open by
    # design. Exact `str` only: `.sub` on a hostile `str` subclass would
    # dispatch to its overrides (D-006 polarity).
    snake_key = _CAMEL_BOUNDARY.sub("_", key) if type(key) is str else key
    if snake_key != key and any(
        pattern.match(snake_key) for pattern in COMPILED_FORBIDDEN_CONTEXT_PATTERNS
    ):
        return True

    # DECISION plan-2026-09-21T203800-8a03483a/D-031 -- tested BEFORE the token
    # referral so a name matching both (`pin_token`) gets the stricter verdict
    # (same rule as D-021 below). Only a `bool` value keeps a matched name.
    # DECISION plan-2026-09-21T203800-8a03483a/D-046 -- a policy-suffix tail
    # keeps only a non-credential-shaped value; acronym split for this rule only.
    # A tuple, not a set: hashing a hostile `str` subclass key would dispatch
    # to its `__hash__` (D-006 polarity). The acronym split is exact-`str` only.
    name_forms: tuple[str, ...] = (key, snake_key)
    if type(key) is str:
        name_forms += (_ACRONYM_BOUNDARY.sub("_", snake_key),)
    if any(_CREDENTIAL_NAME_RE.match(form) for form in name_forms):
        return not isinstance(value, bool)
    if any(_CREDENTIAL_NAME_ANY_RE.match(form) for form in name_forms):
        # The value is kept if it clears the rule under some form's tail
        # (`pinMaxAge` parses only in its snake form); no parseable policy
        # tail fails closed.
        for form in name_forms:
            tail_match = _POLICY_TAIL_RE.match(form)
            if tail_match is None:
                continue
            if not _policy_tail_value_is_credential(value, tail_match["tail"]):
                return False
        return True

    # DECISION plan-2026-07-20T040150-876e7164/D-021 [STALE] -- the TOKEN referral is
    # tested FIRST, and deliberately: a name matching both shapes (`foo_key_token`)
    # must get the STRICTER of the two defaults, and the token arm's is
    # fail-CLOSED where the key arm's is fail-open. Do not reorder these.
    if _TOKEN_VALUE_SCAN_NAME_RE.match(key):
        return _token_value_is_credential(value, key)

    if value is None:
        return False

    # DECISION plan-2026-07-20T103203-b8a6b855/D-006 [STALE] -- site B, the key-arm read.
    # Unbound `str.lower`, not `key.lower()`. The polarity here is NOT the
    # value arm's: failing closed on every `str` subclass KEY would strip
    # unrelated names wholesale, so this site reads the underlying buffer, as the
    # layer-1 regexes (`patterns.py`) already do. `lowered_key` is then an exact `str`, so
    # the frozenset lookup cannot reach a subclass `__eq__` either.
    lowered_key = str.lower(key)
    if not (
        lowered_key in _AMBIGUOUS_CREDENTIAL_ABBREVIATIONS
        or _VALUE_SCAN_NAME_RE.match(key)
    ):
        return False

    # DECISION plan-2026-11-05T134105-5ed71cf2/D-004 -- containers and bytes are
    # judged by their string leaves; see `_key_value_is_credential`.
    return _key_value_is_credential(value, key)
