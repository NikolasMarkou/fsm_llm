"""Layer 2 data of the forbidden-context-entry filter: value-shape tables.

Limits, known credential prefixes, identifier vocabulary and the regexes the
detectors in `shapes.py` read, with the reasoning that shaped them."""

from __future__ import annotations

import re

# ==============================================================
# LAYER 2 -- VALUE SHAPE
# ==============================================================
# DECISION plan-2026-07-20T040150-876e7164/D-019 [STALE]
# WHY THIS LAYER EXISTS AT ALL. Two previous designs used ONLY the key NAME and
# both were measured defective on one axis or the other: a fail-closed allowlist
# over `key` over-stripped 35.5% of ordinary application vocabulary, and the
# crypto DENYLIST that replaced it fails open on 87% of an independently-built
# credential corpus. Neither failure was an implementation slip. They are the two
# faces of ONE structural fact, recorded as D-017: `stripe_key` and `order_key`
# are structurally identical strings, so `*_key` is NOT separable into credential
# vs. ordinary by name shape alone. Any pure-name control must therefore choose
# which direction to be wrong in.
#
# The VALUE is an orthogonal signal and it is where the separation actually
# lives. `stripe_key: "sk_live_4eC39Hq..."` and `order_key: "ORD-12345"` are not
# structurally identical. This layer reads it.
#
# WHAT IS AND IS NOT A CLASS CONTROL HERE -- stated precisely, because overstating
# this is the defect that killed both previous attempts:
#   - The GENERIC arm (charset + character-class mix + length + Shannon entropy)
#     IS a class control, and it is one for a real reason rather than a hopeful
#     one: a credential's value is high-entropy BY DEFINITION -- a low-entropy
#     secret is already a broken secret. That property is intrinsic to what a
#     credential IS, which is exactly what no name vocabulary could ever be.
#   - The PREFIX arm (`sk_live_`, `ghp_`, `AKIA`, `AIza`, `xoxb-`, ...) is an
#     ENUMERATED DENYLIST and fails open on any vendor not listed. It is
#     tolerable to enumerate ONLY because these are externally-fixed PUBLISHED
#     protocol constants that the vendors deliberately made self-identifying so
#     that scanners can find them -- they are not vocabulary this author invented.
#     It is still a list. It is disclosed as one and it is NOT the arm the class
#     claim rests on. HISTORY, kept because it is the reason to distrust an
#     uncited claim here: this comment once cited
#     `test_the_generic_arm_carries_the_control` as pinning that claim while no
#     such test existed (pass-3 adversarial review, CRITICAL 3), and the
#     citation was retracted at CLOSE of plan-2026-07-20T040150-876e7164.
#     The test NOW EXISTS (plan-2026-07-20T103203-b8a6b855 step 7) and the claim
#     is a checked invariant again, measured rather than asserted: neutralising
#     `_CREDENTIAL_VALUE_PREFIXES` to `()` loses **0 of 148** credential
#     detections on the shipped corpus -- the generic arm carries 100%, not the
#     73% the retracted citation claimed. The prefix arm's real and only
#     contribution is the SHORT tail: a published vendor credential under the
#     24-character length floor (`AKIA...` at 20 chars, `xoxb-...`) is detected
#     by it ALONE, and the test pins that too. So do NOT delete this list on the
#     strength of the 100% figure -- it buys the tail, and the 100% is measured
#     over a corpus whose vendor-prefixed credential happens to be 30 characters.
#
# SCOPE IS THE BLAST-RADIUS BOUND AND IT IS DELIBERATE. This layer runs ONLY for
# names that carry a `key` trigger which layer 1 left UNRESOLVED, plus a short
# list of genuinely ambiguous credential abbreviations. It does NOT run for names
# layer 1 positively allowlisted (`page_token`, `max_tokens`, `bos_token` hold
# opaque high-entropy cursors and would be destroyed by it), and it does NOT run
# for names with no trigger at all (`request_id`, `trace_id`, `image_b64`,
# `git_sha` are high-entropy and legitimately belong in a prompt). Do NOT
# "generalise" this to every context key -- that is a very large over-strip and
# it is the first thing a future editor will be tempted to do.
#
# KNOWN AND ACCEPTED GAPS, enumerated rather than papered over:
#   - `authorization: "Bearer eyJ..."` still reaches the prompt: no trigger in
#     the name, so this layer never activates. Accepted to bound blast radius.
#   - BARE HEX IS GENUINELY INSEPARABLE, AND IT IS RESOLVED IN THE STRIP
#     DIRECTION. An AES-128 key and an MD5 digest are both exactly 32 hex
#     characters; a SHA-256 digest and a 256-bit key are both exactly 64. There
#     is no signal in the name OR the value that tells them apart -- this is the
#     one subspace where D-017's structural claim survives this layer intact.
#     Because it cannot be separated, it must be DECIDED, and the plan's own
#     asymmetry decides it: SC-20 bounds fail-open at 5% while SC-5 bounds
#     over-strip at 15%, precisely because a leaked credential is materially
#     worse than a degraded prompt. So a hex value under an ambiguous `*_key`
#     name STRIPS. The measured price is enumerated and real: a hex ETag, git
#     SHA, checksum, trace id or content digest stored under a `*_key` name is
#     stripped from the prompt (`etag_key`, `commit_key`, `checksum_key`,
#     `correlation_key`, `trace_key`, `cache_key_digest` are all pinned as known
#     over-strips). This was NOT instead closed with an "integrity noun"
#     allowlist (`etag|commit|digest|trace|checksum`), even though that scores
#     better on both axes at once, because that list would have been co-authored
#     with the very corpus it is measured against -- the exact tautology D-015
#     shipped and pass-2 caught. A worse honest number beat a better rigged one.
#   - A credential short or low-entropy enough to pass (`api_key: "test"`) is
#     caught by layer 1 on the NAME, which is why both layers ship, not one.
#
# THE VALUE IS NEVER LOGGED. Every log line in this control names the KEY and the
# reason only. Adding the value to a log message would turn a filter that exists
# to keep secrets out of prompts into a mechanism that writes them to disk.
# Do NOT add `{value!r}` to any message here. See decisions.md D-019.

# Only this many leading characters of a value are inspected. A context value can
# be arbitrarily large (a pasted document, a base64 image) and this runs on every
# prompt build with the per-conversation lock held; entropy over a megabyte is a
# DoS, and a credential is never longer than this.
_VALUE_SCAN_LIMIT = 512

# Below this length a string cannot carry enough entropy to be key material, and
# above it the false-positive risk from ordinary identifiers collapses.
_MIN_CREDENTIAL_VALUE_LENGTH = 24

# Shannon entropy floor in bits per character. Random base64 measures ~5.0,
# random hex ~4.0; dotted human identifiers (`checkout.button.submit`) and
# snake_case names measure well below 3.0.
_MIN_CREDENTIAL_VALUE_ENTROPY = 3.0

# Published, vendor-fixed credential prefixes. Sourced from each vendor's own key
# format documentation. ENUMERATED AND INCOMPLETE BY CONSTRUCTION -- see the
# block above; the class claim rests on the generic arm, not on this list.
_CREDENTIAL_VALUE_PREFIXES = (
    "sk_live_",
    "sk_test_",
    "pk_live_",
    "rk_live_",
    "rk_test_",
    "whsec_",
    "sk-ant-",
    "sk-proj-",
    "sk-",
    "shpat_",
    "shpss_",
    "ghp_",
    "gho_",
    "ghu_",
    "ghs_",
    "ghr_",
    "github_pat_",
    "glpat-",
    "xoxb-",
    "xoxp-",
    "xoxa-",
    "xoxs-",
    "xapp-",
    "akia",
    "asia",
    "aiza",
    "ya29.",
    "sg.",
    "nrak-",
    "rzp_live_",
    "rzp_test_",
    "dop_v1_",
    "doo_v1_",
    "dor_v1_",
    "eyj",
    "1//0",
    "eaaa",
    "pat-na",
    "key-",
    "npm_",
    "figd_",
    "sl.u.",
    "hf_",
    "r8_",
    "gsk_",
)

# PEM / PKCS armour. Multi-line and space-bearing, so it must be tested BEFORE
# the whitespace rejection in `shapes.py`.
_PEM_PREFIX = "-----begin"

# DECISION plan-2026-07-20T103203-b8a6b855/D-003 [STALE]
# THE UUID/ULID CARVE-OUT IS NAME-CONDITIONED AND FAIL-CLOSED. It fires only
# when the NAME carries a token from `_IDENTIFIER_NOUN_VOCABULARY` below. An
# unlisted name does NOT get the carve-out: it falls through to the generic
# shape arm and STRIPS. That polarity is the design, not an implementation
# detail (plan S-2).
#
# WHY. The comment this block replaces justified an UNCONDITIONAL carve-out
# with "idempotency keys, dedup keys, correlation keys and request ids are
# UUIDs". Three of those four cannot reach this code at all: `request_id`,
# `correlation_id` and `tenant_id` carry no `key`/`token` trigger, so layer 1
# never refers them to layer 2 (A-1/H-2, probe-confirmed). A census of both
# corpora put the carve-out's ENTIRE true reach at 12 entries out of 557
# (2.2%) -- 7 safe, 5 credential. Unconditional, it bought 7 correct KEEPs and
# paid 5 fail-open leaks on the axis that was already over budget.
#
# MEASURED -- all three dispositions x both axes x per-arm AND slice-total, on
# the shipped regression-probe corpus and on an independently-authored
# 181-entry holdout. Bounds: over-strip <= 15%, fail-open <= 5%.
#
#   disposition | corpus  | over-strip key/token/total | fail-open key/token/total
#   ------------|---------|----------------------------|--------------------------
#   (c) keep    | shipped |      2.6 / 4.4 / 3.4       |     9.9 / 1.4 / 6.5
#   (c) keep    | holdout |      2.0 / 2.0 / 2.0       |     2.4 / 12.5 / 7.4
#   (a) delete  | shipped |      7.8 / 4.4 / 6.3       |     7.9 / 1.4 / 5.3
#   (a) delete  | holdout |      4.0 / 2.0 / 3.0       |     2.4 / 5.0 / 3.7
#   (b) narrow  | shipped |      2.6 / 4.4 / 3.4       |     7.9 / 1.4 / 5.3
#   (b) narrow  | holdout |      2.0 / 2.0 / 2.0       |     2.4 / 5.0 / 3.7
#
# (b) IS WHAT SHIPPED, and it is set inclusion rather than a judgement call:
# (b)'s verdict-change set is a proper SUBSET of (a)'s. Both fix the same 5
# credentials; (a) additionally over-strips 7 safe identifiers and buys nothing
# for them. (c) is dominated outright -- there is no cell where it beats (b).
#
# WHAT THIS CLAIM IS WORTH. READ THIS BEFORE CITING IT.
# The vocabulary was authored BEFORE any disposition was measured and nothing
# was added to it in response to a measurement. Reconciled after the fact it is
# 12/12 correct on the carve-out's true population -- BUT THAT 12/12 IS PARTLY
# CIRCULAR AND MUST BE DISCOUNTED. Four of the shipped corpus's safe UUID names
# are the same nouns the comment this block replaces already listed, so the
# corpus, the old comment and the vocabulary share ancestry and their agreement
# is not evidence. THE INDEPENDENT EVIDENCE IS N=4, ON THE HOLDOUT ONLY:
# `replay_guard_key` kept via `replay` (a noun absent from the old comment) and
# three non-identifier credential UUIDs stripped. So the alignment between
# identifier-noun naming and safe/credential ground truth is MEASURED AND
# NARROW at N=4. It is NOT a general property of the world and must never be
# cited as one.
# Coverage of the list itself, measured not remembered: 11 of the 17 nouns
# appear on some name in some corpus, and only 5 (`idempotency`, `dedupe`,
# `correlation`, `request`, `replay`) ever gate a UUID/ULID-shaped value, which
# is the only population this rule reaches. The other 12 are carried on
# published-convention argument alone and are DISCLOSED as unmeasured, not
# claimed. (decisions.md D-003 says "six ... exercised by NEITHER corpus" and
# then lists nine; both figures are wrong and the ones here are the measured
# replacement.)
# The standing cost is A-2 -- UUID credentials and UUID identifiers are not
# separable by value -- so an adversary who names a UUID-valued credential
# `batch_key`, `trace_key` or `cursor_key` gets it KEPT (probed, all three).
# That is disclosed, not fixed. Note that `session_key` is NOT such an example
# even though `session` is listed: layer 1 strikes it on the NAME and it never
# reaches this rule.
#
# DO NOT:
#  * restore the unconditional carve-out. That is disposition (c), and it
#    leaked all three of the holdout's independent token-arm UUID credentials.
#  * add a vendor name to a denylist to catch an observed leak. Fixing a class
#    by naming its members is this seam's recurring failure shape
#    (plans/LESSONS.md [I:5]), and it is precisely why this list is
#    fail-CLOSED: an omitted noun costs OVER-STRIP, the axis with headroom
#    (S-1), and can never cost fail-open. A denylist inverts that.
#  * split UUID from ULID. Measured separately they behave identically in kind
#    -- each carries both a safe and a credential instance, each fails open on
#    its credential instance under (c), each is fully repaired by (b) at zero
#    over-strip cost. They differ only in magnitude. Narrowing one alone leaves
#    `partner_key: 01J9ZQ4T7XKD3M8VYB2NHF6CWE` live for no measured gain.
#
# Pinned two-sided (both arms, both alphabets, both polarities) by
# `test_a_uuid_valued_credential_under_an_unlisted_name_now_strips`,
# `test_a_uuid_valued_identifier_under_a_listed_noun_is_still_kept` and
# `test_the_identifier_noun_carve_out_is_fail_closed_for_unknown_names` in
# tests/test_fsm_llm/test_context_unit.py. See decisions.md D-003.

# Canonical UUID.
_UUID_VALUE_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)

# ULID (Crockford base32, 26 chars) -- the other standard opaque request id.
_ULID_VALUE_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")

# The identifier nouns that earn the carve-out above. Authored before any
# disposition was measured; see the D-003 block for what its 12/12 fit is and
# is not worth.
#
# DECISION plan-2026-07-20T144233-47e8c662/D-018 [STALE]
# SECOND MEASURED NEGATIVE (the first is D-006 below, on `_AUTH_SCHEME_WORDS`).
# Splitting these nouns by POLARITY -- observability (a UUID is an identifier)
# vs resource/principal (it may be a credential) -- was priced against a
# purpose-built exposure population authored blind. It bought 12 fail-open
# shapes and charged 4 new over-strips: `job_definition_id_key`,
# `lease_holder_ref_key`, `session_handle_token`, `tenant_account_key` -- 4 of
# the 4 rows capable of testing it. Holds under both contested-row
# counterfactuals (re-truthed, dropped).
# Splitting CANNOT repair it: the read site (`shapes.py`) is `any(token in ...)`, a
# DISJUNCTION, and `run_replay_grant_key` carries `run` AND `replay`, so one
# observability token restores the carve-out and the credential still leaks.
# A polarity dimension over a list does not repair a disjunction over it.
# TWO-AXIS CHARGE: 10 fail-open shapes AND 2 over-strip shapes on the census
# (D-014). Extending worsens the first, shrinking worsens the second -- the
# signature of a WRONG PREDICATE, not a short list.
# DO NOT: split by polarity (measured, rejected); extend; shrink; delete the
# carve-out (prior plan's D-003 disposition (a)); or make it unconditional
# (disposition (c) -- measured, strictly dominated; F-06). Working: D-014,
# D-017, D-018. Visible only to
# `tests/test_fsm_llm/fixtures/census_key_corpus.py` (role: INDEPENDENCE).
_IDENTIFIER_NOUN_VOCABULARY = frozenset(
    {
        "idempotency",
        "dedup",
        "dedupe",
        "partition",
        "trace",
        "correlation",
        "request",
        "tenant",
        "shard",
        "lease",
        "span",
        "replay",
        "cursor",
        "batch",
        "job",
        "run",
        "session",
    }
)

# Names are matched token-wise, not by substring: `session_key` matches and
# `obsession_key` does not.
_NAME_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")

# Object-storage / filesystem paths. `/` cannot simply be dropped from the
# credential charset below, because standard (non-url-safe) base64 uses it, so
# paths need an explicit carve-out instead. `s3_key`, `blob_key`, `object_key`
# and `upload_key` hold paths and are ordinary application data.
# This regex is only the TRIGGER for the path carve-out, never the whole of it:
# see `_is_path_shaped` and the D-005 block in `shapes.py` for why an extension is not
# sufficient evidence on its own.
_PATH_VALUE_RE = re.compile(r"^[^\s]*/[^\s]*\.[A-Za-z0-9]{1,6}$")

# `<n>` or more `/`-separated fields is the other path trigger. Pulled out of
# the expression it used to be inlined in so the two triggers read as one rule.
_PATH_SEGMENT_TRIGGER = 4

# The charset a credential value is drawn from: base64 / base64url / base32 / hex
# plus the separators vendors use. Anything outside it (a space, `#`, `:`, `@`)
# means the value is structured application data, not a secret. Values that
# merely WRAP a credential in transport syntax are normalised into this charset
# by `_normalise_credential_value` before the test, rather than being dismissed.
_CREDENTIAL_VALUE_CHARSET_RE = re.compile(r"^[A-Za-z0-9+/=_.\-]+$")

# HTTP authentication scheme words (RFC 7235 / the IANA registry, plus the
# non-registered spellings that appear in the wild). A value of the form
# `<scheme> <credential>` is a credential wearing an `Authorization` header's
# wire syntax, not a sentence. DATA, not control flow: one word per line.
#
# DECISION plan-2026-07-20T144233-47e8c662/D-006 [STALE]
# MEASURED NEGATIVE. Replacing this frozenset with a bare positive head-shape
# test at its single read site (`shapes.py`) -- `re.compile(r"[A-Za-z][A-Za-z0-9\-]{0,19}")`
# fullmatching `fields[0]` -- was implemented, measured on a 274-entry
# shape-census corpus, and REVERTED. Fail-open fell hard: census key arm
# 38.8% -> 13.4% (-25.4 pp), token arm 45.5% -> 15.2% (-30.3 pp), with ZERO
# fail-open regressions on any of the three corpora. Over-strip paid for it and
# breached a pre-registered trigger: census key 13.2% -> 16.2% (+2.9 pp), token
# 12.3% -> 15.1% (+2.7 pp). Full tables in D-003 (baseline), D-004 (delta),
# D-005 (two framings retracted on review), D-006 (the pre-change prediction).
#
# WHY it cannot move alone: this frozenset does DOUBLE DUTY. Duty 1 is the
# stated one -- a fail-OPEN scheme detector, done badly (47.5% coverage of
# realistic scheme words; 37 census leaks, 20 distinct values, wear unlisted
# heads). Duty 2 is accidental and was never written down: by refusing to unwrap
# an unlisted head, it keeps `<Noun> <UUID>` values away from the fail-CLOSED
# `_IDENTIFIER_NOUN_VOCABULARY` name gate. Both duties are the same predicate
# read once, so deleting it alone charges the over-strip cost to that second
# vocabulary -- measured as exactly 4 rows (`acl_policy_*`, `ledger_*`),
# predicted by name from pre-change data before the change was made.
#
# DO NOT:
#  - delete this frozenset without SIMULTANEOUSLY redesigning
#    `_IDENTIFIER_NOUN_VOCABULARY`. Measured: they cannot move independently.
#  - "fix" it by adding more scheme words. Its omissions charge fail-open, the
#    axis with no headroom, and naming a class's members is not fixing the class.
#  - skip the unwrap when the tail is a bare UUID. That rescue was built and
#    measured: 0 over-strips, no trigger fires -- and 4 SECURITY REGRESSIONS.
#  - add `not _generic_shape_is_credential(fields[0])`. Under a 20-char head cap
#    the 24-char length floor is unreachable, so it is provably dead code that
#    READS like a safety check.
#
# This class is visible only to `tests/test_fsm_llm/fixtures/census_key_corpus.py`
# (role: INDEPENDENCE); both shipped corpora are BURNED against it.
_AUTH_SCHEME_WORDS = frozenset(
    {
        "apikey",
        "basic",
        "bearer",
        "concealed",
        "digest",
        "dpop",
        "gnap",
        "hmac",
        "hoba",
        "jwt",
        "mac",
        "mutual",
        "negotiate",
        "ntlm",
        "oauth",
        "privatetoken",
        "scram-sha-1",
        "scram-sha-256",
        "signature",
        "ssws",
        "token",
        "vapid",
    }
)
