"""Layer 2 detectors of the forbidden-context-entry filter: value shapes.

`_looks_like_credential_value` and the helpers it calls. The tables they read
are in `shape_data.py`."""

from __future__ import annotations

import math
import re
from urllib.parse import unquote

from .shape_data import (
    _AUTH_SCHEME_WORDS,
    _CREDENTIAL_VALUE_CHARSET_RE,
    _CREDENTIAL_VALUE_PREFIXES,
    _IDENTIFIER_NOUN_VOCABULARY,
    _MIN_CREDENTIAL_VALUE_ENTROPY,
    _MIN_CREDENTIAL_VALUE_LENGTH,
    _NAME_TOKEN_SPLIT_RE,
    _PATH_SEGMENT_TRIGGER,
    _PATH_VALUE_RE,
    _PEM_PREFIX,
    _ULID_VALUE_RE,
    _UUID_VALUE_RE,
    _VALUE_SCAN_LIMIT,
)

# DECISION plan-2026-07-20T040150-876e7164/D-021 [STALE]
# COLON-COMPOSITE CREDENTIALS. `:` is deliberately OUTSIDE the charset in `shape_data.py`,
# because `cache_key: "sha256:<digest>"` is integrity data that SC-5 pins as
# KEPT. That exclusion also let two real composite credentials through
# unmeasured -- an Asana PAT is `<numeric id>:<secret>` and a Cloudinary key is
# `<numeric id>:<secret>`; both were fail-open leaks in attempt 3's held-out
# slice (D-019). So a colon-bearing value is not dismissed: it is SPLIT on the
# LAST colon and the tail is judged on its own.
#
# The one carve-out is `<label>:<pure hex>`, which is the `sha256:`/`md5:`
# convention (Docker image digests, Subresource Integrity, ETags,
# content-addressed cache keys) and is integrity data, not a credential.
#
# THE CARVE-OUT IS PURE-HEX, NOT A LENGTH SET, AND THAT IS A CORRECTION MADE
# UNDER MEASUREMENT. The first cut of this rule keyed off the published digest
# output sizes (32/40/56/64/96/128 hex characters) on the theory that a fixed
# externally-published list is safer than a shape. It is not: content-addressed
# cache keys routinely TRUNCATE the digest, and the primary corpus's own
# `cache_key: "sha256:9f86...822c"` carries a 48-character truncation, so the
# length set stripped the one name SC-5 pins individually. Truncation length is
# an application choice with no canonical set; hex-ness is the actual signal.
# Do NOT reintroduce a length set here.
#
# NOTE THE DELIBERATE REVERSAL OF DIRECTION. For a BARE value D-019 resolves
# ambiguous hex toward STRIP, because nothing distinguishes an AES-256 key from
# a SHA-256 digest. Here it resolves toward KEEP, because the `<label>:` prefix
# is evidence the bare form lacked -- credentials are not published with an
# algorithm label in front of them; digests are. THE COST IS REAL AND IS NOT
# HIDDEN: a credential deliberately stored as `v1:<hex>` under a name layer 1
# does not reach is KEPT. That is the same inseparable-hex corner D-019 already
# disclosed, decided the other way because the evidence differs.
# Do NOT "simplify" this by adding `:` to the charset in `shape_data.py` -- that strips
# `cache_key: "sha256:..."`, which SC-5 pins. See decisions.md D-021.
_PURE_HEX_RE = re.compile(r"^[0-9a-f]+$", re.IGNORECASE)


def _colon_composite_tail(stripped: str) -> str | None:
    """The credential-bearing tail of a ``<prefix>:<tail>`` value, or ``None``.

    Returns ``None`` when *stripped* carries no colon, and when the value is
    the ``<label>:<hex digest>`` integrity shape. NEVER logs *stripped*.
    Never raises for any EXACT ``str`` *stripped* -- which is all it is ever
    given, by the containment argument in the D-006 block below. See that block
    for why the qualifier is there and why the guard is not repeated here.
    """
    if ":" not in stripped:
        return None
    tail = stripped.rsplit(":", 1)[1]
    if _PURE_HEX_RE.match(tail):
        return None
    return tail


def _shannon_entropy(text: str) -> float:
    """Bits of Shannon entropy per character of *text*.

    Never raises for any EXACT ``str`` *text* (D-006 containment argument).
    """
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for character in text:
        counts[character] = counts.get(character, 0) + 1
    total = len(text)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


# DECISION plan-2026-07-20T103203-b8a6b855/D-005 [STALE]
# THE CHARSET/SHAPE RULE IS ONE RULE. Internal whitespace, percent-encoding,
# `count("/") >= 3` and the 24-character length floor were four separate
# failure modes of the single test below, and they were patched, disclosed and
# reasoned about separately for three plans. They are now one rule with ONE
# accepted-gaps list. Do NOT re-split it.
#
# WHAT CHANGED, AND WHY EACH CHANGE IS THE ONE IT IS
#
# 1. TRANSPORT UNWRAPPING BEFORE JUDGEMENT (`_normalise_credential_value`).
#    Percent-decode once, then unwrap `<scheme> <credential>` once, BEFORE the
#    vendor-prefix and charset tests. Both shapes previously fell outside the
#    charset and were KEPT, so `gateway_key: "Bearer 9dR2...uI"` and
#    `upload_key: "Atzr%2FIQEB..."` reached the prompt intact. The result is
#    never longer than the input, so `_VALUE_SCAN_LIMIT` still bounds the path
#    and there is no decode bomb. `errors="ignore"` DROPS undecodable bytes
#    rather than substituting U+FFFD, which would land outside the charset and
#    buy the value a KEEP -- the fail-OPEN direction (S-2).
#
# 2. THE PATH CARVE-OUT NEEDS POSITIVE SEGMENT EVIDENCE (`_is_path_shaped`).
#    The old rule was `_PATH_VALUE_RE.match(v) or v.count("/") >= 3`, defended
#    by an in-code comment claiming "the extension test alone carries almost all
#    of it -- random base64 does not end in `.pdf`". MEASURED FALSE: standard
#    base64 emits `/` and PASETO/JWT segments are `.`-joined, so real credentials
#    satisfy BOTH triggers --
#      distribution_key = "9dR2pQ7xL4mZ8vN3bK6tY1wJ5hG0sF2/aD8cE4rT7uI.Xk3f9a"
#      mailer_key       = "a8f3/d9c2b1e4/f7a0d3c6/b9e2f5a8d1c4e7b0a3d6f9Xk"
#    Neither trigger is evidence of a PATH. The evidence is what the SEGMENTS
#    are: a path's segments are human-authored words (`invoices`, `2024`,
#    `invoice-10482.pdf`), none credential material alone; a base64 blob's are.
#    So the triggers stay and a NECESSARY condition is added -- no segment may be
#    credential-shaped by the generic arm's own test. That is why
#    `_generic_shape_is_credential` is called on the whole value AND each
#    segment: one threshold, one definition, two call sites. Do NOT reintroduce
#    the bare `count("/") >= 3`; it is what handed `mailer_key` to the prompt.
#
# 3. THE 24-CHARACTER LENGTH FLOOR STAYS AT 24 -- a MEASURED refusal, not an
#    omission. Sweeping it against both corpora (both axes, per-arm and
#    slice-total) gives, on the independent holdout's key arm:
#      floor 24 -> over-strip  2.0% | floor 22 -> 4.0% | floor 20 -> 10.0%
#      floor 18 -> 12.0%            | floor 14 -> 18.0% | floor 13 -> 22.0%
#    The shortest pinned credential is 13 characters and ANY floor reaching it
#    puts holdout over-strip at 18-22%, past the 15% bound -- destroying
#    `alert_rule_key`, `dashboard_panel_key`, `shipping_zone_key` and a dozen
#    more ordinary values to catch one. Floor 22 is inside both bounds and was
#    still REFUSED: the sole credential it catches is a leak observed in the
#    holdout AFTER measurement, and tuning a threshold to instances you just
#    watched leak is the one move `plans/LESSONS.md [I:4]` forbids outright.
#    Do NOT lower this floor without re-running the sweep.
#
# ACCEPTED GAPS -- THE COMPLETE RESIDUAL FAILURE SURFACE OF THIS RULE.
# Every credential shape below is still KEPT. This list is the whole of it, in
# one place on purpose, so an editor sees the failure surface at once instead of
# rediscovering it one carve-out at a time. Each gap names its pin.
#
#  G1 LENGTH. Credentials under `_MIN_CREDENTIAL_VALUE_LENGTH` (24) are
#     invisible to this entire arm. The NAME layer is their only control.
#     Pinned: `short/credential` (`terminal_key`, 13 chars) in
#     `CARVE_OUT_KNOWN_FAIL_OPEN`, two-sided against `short/safe`
#     (`order_key: "ORD-10482"`). See point 3 for why it is not lowered.
#  G2 UNWRAPPING DEPTH. Normalisation is exactly two shapes deep and runs each
#     ONCE. Double-encoding (`%252F`), a three-or-more-field whitespace value
#     (`Bearer <cred> trailing`), and a two-field value whose head is not in
#     `_AUTH_SCHEME_WORDS` (`Sigv4Custom <cred>`) all stay outside the charset
#     and are KEPT. Iterating to a fixed point was rejected: it converts a
#     bounded normaliser into an unbounded one and reopens the decode surface
#     `_VALUE_SCAN_LIMIT` exists to close. Pinned by
#     `test_the_unwrapping_is_two_shapes_deep_and_that_is_disclosed`.
#  G3 CHARSET. A credential carrying any character outside
#     `_CREDENTIAL_VALUE_CHARSET_RE` after normalisation -- `#`, `@`, or a
#     non-ASCII character from a VALID escape such as `%E2%82%AC` -- is KEPT.
#     Widening the charset is not available: `:` is excluded so that
#     `cache_key: "sha256:..."` survives (D-021, SC-5-pinned). Pinned by
#     `test_a_credential_outside_the_charset_is_still_kept`.
#  G4 PATH SEGMENTS. A slash-bearing credential whose segments are EACH below
#     the floor or single-class reads as a path and is KEPT
#     (`a8f3/d9c2/b1e4/f7a0/d3c6/b9e2/f5a8`). This is the residual of fix 2 and
#     is strictly smaller than the `count("/") >= 3` it replaced. Pinned by
#     `test_a_slash_credential_with_only_short_segments_is_still_kept`.
#  G5 CHARACTER CLASSES and G6 ENTROPY -- `classes < 2` and the 3.0-bit floor in
#     `_generic_shape_is_credential`. Pinned: `single_class/credential`
#     (`legacy_key`, 30 lowercase chars) and `low_entropy/credential`
#     (`staging_key`, `aaaa...AAAA1111`), both in `CARVE_OUT_KNOWN_FAIL_OPEN`.
#     MAINTAINER RULING, RECORDED VERBATIM: "these are the generic arm's CORE
#     THRESHOLDS, not carve-outs. Tightening them is exactly the over-correction
#     Pre-Mortem 4 names, and `staging_key`'s ground truth as a credential is
#     itself questionable -- by this seam's own foundational lesson
#     (LESSONS [I:5], 'a credential's VALUE is high-entropy by definition; a
#     low-entropy secret is already broken'), a 3-bit-entropy string is not
#     credential material in any meaningful sense. Disclosing them is correct;
#     tightening a threshold to catch them would trade a security defect for a
#     usability defect across the entire generic arm." DISCLOSE, DO NOT FIX.
#  G7 COLON COMPOSITES. `signer_key: "v1:<hex>"` rides the `<label>:<pure hex>`
#     carve-out that exists so `cache_key: "sha256:..."` survives. Prior-plan
#     D-021's disclosed gap, restated here so the surface is complete. Pinned:
#     `hex_composite/credential` in `CARVE_OUT_KNOWN_FAIL_OPEN`.
#
# DECISION plan-2026-07-20T103203-b8a6b855/D-009 [STALE]
# ACCEPTED GAPS OF THE **OTHER** RULES ON THIS CHAIN. G1-G7 above are
# this rule's residual and only this rule's. G8-G9 belong to the D-003
# identifier-noun carve-out and to the layer-1 qualifier alternation
# respectively, and are listed HERE, next to G1-G7, for one reason: an editor
# looking for "what does this chain still get wrong" must find the whole answer
# in one place. Their OWNERS are elsewhere; do not fold them into the rule above.
#
#  G8 DOTTED / HIERARCHICAL IDENTIFIERS OVER-STRIP. The D-003 carve-out only
#     applies under a UUID/ULID-shaped value, so an identifier in any OTHER
#     common real format falls through to the generic arm and is STRIPPED even
#     when its name carries a LISTED noun. MEASURED at HEAD:
#       metric_key        "svc.checkout.latency.p99.eu-central-1"      STRIP
#       routing_key       "orders.eu-central-1.v2.created.high-..."     STRIP
#       trace_token       "00-4bf92f35...-00f067aa0ba902b7-01" (W3C)   STRIP
#       correlation_token "4bf92f3577b34da6a3ce929d0e0e4736" (32 hex)  STRIP
#     `trace` and `correlation` ARE in `_IDENTIFIER_NOUN_VOCABULARY`, so the
#     vocabulary does not protect the identifiers it was named for in their most
#     common wire formats -- which is the honest limit of the D-003 result and
#     is stated here rather than left for a fourth adversarial pass to find.
#     This is OVER-strip, the axis with headroom (S-1: 3.4% against a 15%
#     bound), which is why it is disclosed rather than fixed: widening the
#     carve-out to non-UUID shapes would hand the fail-open axis a class it
#     cannot separate by value (A-2). Pinned two-sided by `metric_key` in
#     `CRYPTO_KEY_KNOWN_OVER_STRIPPED` and `trace_token`/`correlation_token` in
#     `TOKEN_KNOWN_OVER_STRIPPED`. Raised as pass-3 concern 11 and carried.
#  G9 DIGEST-FUNCTION QUALIFIERS ARE NOT CRYPTO-KEY QUALIFIERS -- DELIBERATE.
#     `sha256_key`, `sha1_key`, `sha512_key`, `md5_key` and `blake3_key` are
#     KEPT on the NAME while `aes256_key`, `aes128_key`, `rsa2048_key`,
#     `rsa4096_key`, `hmac256_key`, `ecdsa_key` and `ed25519_key` are STRUCK on
#     it. That boundary is not an omission from `_CRYPTO_KEY_QUALIFIERS`: SHA,
#     MD5 and BLAKE name DIGEST functions, and a `sha256_key` is as often a
#     cache or content-address handle as key material -- `cache_key:
#     "sha256:..."` is SC-5-pinned to be KEPT by the prior plan for exactly this
#     reason. HMAC is struck because HMAC genuinely is keyed. MEASURED, and this
#     is why it is not "fixed": all five digest names STRIP anyway once they
#     carry a real 64-hex digest value, so adding `sha|md5|blake` as qualifiers
#     buys ZERO fail-open and costs five names' worth of pure over-strip --
#     "spending headroom that buys nothing is not a trade, it is waste"
#     (D-003's own argument against disposition (a)). Raised as pass-3
#     concern 12 and answered here. Do NOT add digest names to
#     `_CRYPTO_KEY_QUALIFIERS`.
#
# MEASURED (both corpora, both axes, per-arm/slice-total; bounds 15%/5%): shipped
# over-strip 2.6/4.4/3.4 and holdout 2.0/2.0/2.0 BOTH UNCHANGED to the entry --
# the `LESSONS [I:4]` check this edit most needed, since it pushes toward STRIP on
# three fronts; shipped fail-open 7.9/1.4/5.3 -> 4.0/1.4/2.9, holdout 2.4/5.0/3.7
# unchanged. All 24 cells inside both bounds. See decisions.md D-005.
def _normalise_credential_value(stripped: str) -> str:
    """Unwrap transport syntax from *stripped*, returning the payload to judge.

    Two normalisations, applied in this order and at most once each:
    percent-decoding, then `<auth-scheme> <credential>` unwrapping. A value
    carrying neither is returned unchanged. The result is never longer than the
    input, so the ``_VALUE_SCAN_LIMIT`` bound still holds. NEVER logs
    *stripped*. Never raises for any EXACT ``str`` *stripped* (D-006
    containment argument).
    """
    if "%" in stripped:
        # `errors="ignore"` DROPS undecodable bytes rather than substituting
        # U+FFFD, which keeps the result inside the credential charset. That is
        # the fail-CLOSED direction (S-2): a mangled escape must not buy a KEEP.
        stripped = unquote(stripped, errors="ignore")
    fields = str.split(stripped)
    if len(fields) == 2 and str.lower(fields[0]) in _AUTH_SCHEME_WORDS:
        stripped = fields[1]
    return stripped


def _generic_shape_is_credential(text: str) -> bool:
    """The generic shape arm: length, charset, character-class mix, entropy.

    Vocabulary-free by construction -- it reads only the shape of *text*. Split
    out of :func:`_looks_like_credential_value` because `_is_path_shaped` needs
    to ask the same question of a path's individual segments, and two
    hand-maintained copies of a security threshold is a defect, not a pattern.
    NEVER logs *text*. Never raises for any EXACT ``str`` *text* (D-006
    containment argument).
    """
    if len(text) < _MIN_CREDENTIAL_VALUE_LENGTH:
        return False
    if not _CREDENTIAL_VALUE_CHARSET_RE.match(text):
        return False
    # A credential mixes character classes; human-readable identifiers
    # (`checkout.button.submit`, `idx_users_email`, `region-eu-central-1`) are
    # overwhelmingly single-case. Require at least two of lower/upper/digit.
    classes = 0
    if any(character.islower() for character in text):
        classes += 1
    if any(character.isupper() for character in text):
        classes += 1
    if any(character.isdigit() for character in text):
        classes += 1
    if classes < 2:
        return False
    return _shannon_entropy(text) >= _MIN_CREDENTIAL_VALUE_ENTROPY


def _is_path_shaped(stripped: str) -> bool:
    """True if *stripped* is an object-storage / filesystem path.

    A trigger (file extension, or four-plus `/`-separated fields) is necessary
    but NOT sufficient: the carve-out applies only when no individual segment is
    itself credential material. NEVER logs *stripped*. Never raises for any
    EXACT ``str`` *stripped* (D-006 containment argument).
    """
    if "/" not in stripped:
        return False
    segments = stripped.split("/")
    # DECISION plan-2026-07-20T103203-b8a6b855/D-005 [STALE] -- ReDoS ordering.
    # Segment count first, THEN the regex: `or` short-circuits, and
    # `_PATH_VALUE_RE`'s two adjacent `[^\s]*` runs have a measured backtracking
    # cliff on slash-and-dot-heavy input (~490us at the `_VALUE_SCAN_LIMIT`
    # boundary, vs ~46us when the count test answers first). Verdict-identical;
    # cost is not. Do NOT swap these back.
    triggered = (
        len(segments) >= _PATH_SEGMENT_TRIGGER
        or _PATH_VALUE_RE.match(stripped) is not None
    )
    if not triggered:
        return False
    return not any(_generic_shape_is_credential(segment) for segment in segments)


def _looks_like_credential_value(value: object, name: str = "") -> bool:
    """Return True if *value* has the SHAPE of credential material.

    This is layer 2 of the context-key security filter and it is only ever
    consulted for a key whose NAME layer 1 left ambiguous (see
    :func:`is_forbidden_context_entry`). It reads the value's shape -- charset,
    character-class mix, length and Shannon entropy, plus published vendor
    prefixes and PEM armour -- and never its meaning.

    Non-``str`` values are never credentials by this test: an ``int``, ``bool``
    or ``float`` cannot carry key material, and containers are recursed into by
    the callers rather than inspected here. A ``str`` SUBCLASS is credential
    material fail-CLOSED and is never inspected -- see the D-006 block below.

    *name* is the context key the value was stored under, and is read by ONE
    rule only: the fail-closed UUID/ULID identifier-noun carve-out (D-003). It
    defaults to ``""``, which is the fail-closed direction -- a caller that
    supplies no name gets no carve-out. Read through the unbound
    ``str.lower``, so a hostile ``str`` subclass cannot hook it; a non-``str``
    *name* is blanked to ``""`` first, which is the same fail-closed direction
    (D-009).

    NEVER logs *value* and never re-raises with it attached. Never raises, for
    ANY *value* and ANY *name* of ANY type, including hostile ``str``
    subclasses of either (D-006, D-009). Inspects at most
    ``_VALUE_SCAN_LIMIT`` leading characters, so cost is bounded regardless of
    how large the context value is.
    """
    if not isinstance(value, str):
        return False

    # DECISION plan-2026-07-20T103203-b8a6b855/D-006 [STALE]
    # THE `str`-SUBCLASS RULE, THIRD SITE ON THIS SEAM. `isinstance` admits a
    # subclass whose `__getitem__`/`.lower`/`__len__`/`__bool__` runs arbitrary
    # code; the slice below propagated `RuntimeError` out of the live prompt path
    # (probed at `a414601`). EXACT type, and fail CLOSED -- an unreadable value
    # is credential material (S-2). Unbound `str.__getitem__` is ALSO raise-proof
    # and was REJECTED: it reads the underlying buffer, so a subclass hiding its
    # payload from `str` while revealing it to the prompt via `__str__` would be
    # KEPT. Raise-proof is not the property that matters here; fail-closed is.
    # Do NOT "simplify" this to unbound slicing. Everything downstream then holds
    # an EXACT `str` (every `str` method returns one), which is what makes
    # "Never raises" TRUE rather than aspirational on the helpers below.
    #
    # DECISION plan-2026-07-20T103203-b8a6b855/D-009 [STALE] -- the containment
    # argument's EXACT SCOPE, measured rather than asserted. This guard is what
    # makes the five helpers below raise-proof, so their claim is scoped to what
    # it actually buys: "never raises for any EXACT `str`". Called DIRECTLY with
    # a hostile subclass -- which nothing in this package does, and which the
    # sweep confirms is the only way to reach them unguarded -- they DO raise
    # (measured: 147 cells across `_shannon_entropy`, `_colon_composite_tail`,
    # `_normalise_credential_value`, `_generic_shape_is_credential`,
    # `_is_path_shaped`). Repeating this guard five times was REJECTED: it would
    # give five choke points where the design's whole claim is that there is
    # one, and it buys nothing a caller inside this module can reach. The
    # honest move is the narrowed docstring, not a bare "Never raises" that a
    # direct caller can falsify. Do NOT widen those claims back.
    # Pinned two-sided by `test_a_hostile_str_subclass_value_fails_closed` and
    # `test_a_benign_str_subclass_value_also_fails_closed`.
    # See decisions.md D-006; `memory.py` D-016 is sites one and two.
    if type(value) is not str:
        return True

    sample = value[:_VALUE_SCAN_LIMIT]
    lowered = sample.lstrip().lower()

    # PEM armour first: it is the one credential shape that legitimately
    # contains whitespace, so the whitespace rejection below would lose it.
    if lowered.startswith(_PEM_PREFIX):
        return True

    # DECISION plan-2026-07-20T103203-b8a6b855/D-005 [STALE] -- unwrapping runs BEFORE
    # the vendor-prefix test on purpose, so `Bearer sk_live_...` reaches the
    # enumerated arm. See the D-005 block above `_normalise_credential_value`.
    stripped = _normalise_credential_value(sample.strip())

    # Published vendor prefixes (enumerated arm). This is tested BEFORE the
    # length floor on purpose: a self-identifying vendor prefix is conclusive
    # regardless of how short the credential is, and several real formats sit
    # under the floor -- an AWS access key id (`AKIA...`) is exactly 20
    # characters. Ordering these the other way round leaked all three.
    if stripped.lower().startswith(_CREDENTIAL_VALUE_PREFIXES):
        return True

    # DECISION plan-2026-07-20T040150-876e7164/D-021 [STALE] -- colon composites. Done
    # AFTER the vendor-prefix test (a prefixed credential is conclusive whole)
    # and BEFORE the length floor (the tail is shorter than the whole).
    composite_tail = _colon_composite_tail(stripped)
    if composite_tail is not None:
        stripped = composite_tail

    # Generic arm. Everything below here is shape, not vocabulary.
    if not _generic_shape_is_credential(stripped):
        return False
    # DECISION plan-2026-07-20T103203-b8a6b855/D-003 [STALE] -- the carve-out is
    # name-conditioned and fail-CLOSED. An unlisted name gets no carve-out and
    # falls through to the rules below, which strip a canonical UUID/ULID. Do
    # NOT drop the vocabulary test to "simplify" this back to unconditional;
    # that is the measured disposition (c) and it leaks. Full table, the N=4
    # limit on the claim, and what NOT to do: see the D-003 block above
    # `_UUID_VALUE_RE` in `shape_data.py`.
    if _UUID_VALUE_RE.match(stripped) or _ULID_VALUE_RE.match(stripped):
        # DECISION plan-2026-07-20T103203-b8a6b855/D-006 [STALE] -- site C, deleted.
        # The `if name else ()` short-circuit that stood here read
        # `bool(name)` -- the subclass's `__len__`/`__bool__`, a latent raise
        # site reachable ONLY under a UUID/ULID value, which is why it survived
        # step 4. Deleting it is verdict-identical (`str.lower("")` splits to
        # `[""]`, not in the vocabulary). Do NOT restore it as an optimisation.
        #
        # DECISION plan-2026-07-20T103203-b8a6b855/D-009 [STALE] -- site E, the FIFTH
        # raise site, and the reason a sweep must be re-run after the last edit
        # rather than after the edit that motivated it. `str.lower(name)` is
        # raise-proof for a `str` SUBCLASS (that is why D-006 chose it) but
        # raises `TypeError` for a NON-`str` name -- and only when the value is
        # UUID/ULID-shaped, the same latency that hid site C from step 4. Not
        # live-path reachable (`is_forbidden_context_entry` guards
        # `isinstance(key, str)` first), but the docstring above claims "for ANY
        # input" and a non-`str` name is an input.
        #
        # `isinstance`, NOT `type(name) is str`, and the difference is a
        # deliberate polarity call. Blanking a str SUBCLASS name would delete
        # its carve-out and STRIP it -- which contradicts D-006 site B, where
        # failing closed on subclass KEYS was rejected precisely because it
        # would strip unrelated names wholesale. A subclass name is read safely
        # by the unbound `str.lower` below (probed: 13 hostile subclasses x 18
        # value shapes x 2 entry points, zero raises), so there is nothing to
        # fail closed ABOUT. A non-`str` name has no name to read, and `""` is
        # the fail-CLOSED direction: no token matches the identifier-noun
        # vocabulary, so no carve-out applies and a UUID/ULID value STRIPS.
        # Pinned by `test_a_non_str_name_fails_closed_under_a_uuid_value`.
        if not isinstance(name, str):
            name = ""
        tokens = _NAME_TOKEN_SPLIT_RE.split(str.lower(name))
        if any(token in _IDENTIFIER_NOUN_VOCABULARY for token in tokens):
            return False
    # DECISION plan-2026-07-20T103203-b8a6b855/D-005 [STALE] -- positive path evidence.
    # Do NOT restore the bare `stripped.count("/") >= 3`; standard base64 emits
    # slash runs and that test alone handed `mailer_key` to the prompt.
    if _is_path_shaped(stripped):
        return False

    return True
