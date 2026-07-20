"""Shape-census independence corpus for the `*_key` / `*_token` prompt-path filter.

ROLE: **INDEPENDENCE**.
================================================================================
This is the THIRD corpus on this seam, and its role is deliberately distinct from
the other two. Do not merge it into either of them.

  * `context_key_corpus.py`  -- role: REGRESSION-PROBE + SHAPE-COVERAGE. It has been
    read against the filter repeatedly and carries two-sided pin sets that encode
    the filter's measured verdicts. It cannot answer "does this filter generalise?"
    because its rows were selected in the filter's presence.
  * `holdout_key_corpus.py`  -- role: BURNED. It was authored blind, measured once,
    and that single measurement spent its independence. Re-measuring it yields no
    new information about the classes it has already scored.
  * `census_key_corpus.py`   -- role: INDEPENDENCE (this file). Authored WITHOUT
    opening `src/fsm_llm/constants.py`, `context_key_corpus.py`,
    `holdout_key_corpus.py`, or any prior plan content about this seam.

DERIVATION
================================================================================
Every entry below is derived from
`plans/plan-2026-07-20T144233-47e8c662/findings/shape-census-blind.md` -- a 50-shape
(S01..S50) census of realistic `*_key`/`*_token` VALUE shapes that was itself
authored without reading the filter, from RFC/spec knowledge plus a grep of the
repo OUTSIDE the protected paths.

Ground-truth calls came from exactly two sources:
  1. the census's own KEEP / STRIP verdict column, and
  2. NAME semantics -- the semantic root of the field name (`secret`, `private`,
     `admin`, `acl`, `root` vs `request`, `trace`, `correlation`, `replay`,
     `dedupe`, `commit`).
Never from what the filter does. No ground-truth call in this file was decided by
running the filter, reading `constants.py`, or consulting either sibling corpus.
That independence IS the artifact; it is the only property that makes a
post-change measurement on this corpus mean anything.

FREQUENCY WEIGHTING -- READ THIS BEFORE QUOTING ANY RATE FROM THIS CORPUS
================================================================================
The census carries a `high`/`med`/`low` frequency column. The census author labels
it, in its own words, "a SAMPLING GUIDE, not ground truth" -- it is that author's
personal judgment calibrated against public API-design conventions, NOT a
measurement of anything in this repo or in the wild.

How it was used here, plainly: **it was used only to decide which shapes get more
than one entry, and never to set a target proportion.** High-frequency shapes
(Bearer wraps, JWTs, opaque hex/base64url, UUID identifiers, small numeric ids,
enum words) appear several times; low-frequency shapes (base32, PEM, percent-
encoded, unicode, whitespace-only) usually appear once or twice per cell. The four
(arm x ground_truth) cells were then filled to roughly equal size by construction
so that no cell is vacuous.

CONSEQUENCE, STATED SO NOBODY IS MISLED: **this corpus is NOT frequency-calibrated.**
Its cell rates are rates over ITS OWN population, not estimates of any real-world
population. A 10% fail-open rate here means "10% of the shapes this corpus chose to
sample", not "10% of credentials in the wild". Report every rate with its n and its
95% Wilson interval, and never present a census rate as a field prevalence.

THIS CORPUS IS HALF MIRROR -- THE ARM SPLIT IS NOT INDEPENDENT EVIDENCE
================================================================================
Measured, not estimated (`test_the_census_mirror_is_measured_and_pinned` pins all
four numbers and fails if they drift):

    321 rows  /  176 distinct VALUES  /  124 values present in BOTH arms
    182 distinct (value, ground_truth) SHAPES

    (iteration 2 step 2 measured 274 / 129 / 124 / 135; step 3 added 47 rows, every
     one a NEW value in exactly ONE arm, so the both-arms count did not move and the
     distinct/row RATCHET improved from 129/274 = 0.471 to 176/321 = 0.548.)

So the `*_key` arm and the `*_token` arm are near-perfect mirrors of one another for
the ORIGINAL 274 rows: the SAME value was typed under a `*_key` name and again under
a `*_token` name for 124 of the first 129 values here. The consequences, stated
plainly because iteration 1 of plan-2026-07-20T144233-47e8c662 reported 18 cells
without knowing them:

  * **A per-arm rate and the slice-total are largely the same observations
    recounted.** "The key arm AND the token arm AND the total all breached" is ONE
    finding reported three times, not three corroborating findings.
  * **The corpus's effective n is roughly half its stated n.** Per-cell Wilson
    intervals assume independent trials; on mirrored rows they are anticonservative.
  * **Any n >= 60 per-cell adequacy guard is satisfied partly BY the mirroring**,
    because it counts rows.

THE UNIT OF MEASUREMENT IS THEREFORE THE DISTINCT `(value, ground_truth)` SHAPE,
counted by `distinct_shape_count()` in `test_context_unit.py`. Every cell reported
from this corpus carries `n` (rows) AND `d` (shapes), with the SHAPE-level rate as
the primary figure; a row-level rate is never quoted alone. `d` (182) exceeds the
distinct-value count (176) because six values are ground-truthed BOTH ways -- the
F-02 class, where one string is an identifier under one name and a credential under
another. All six are original rows; step 3 added none, because a value that appears
in exactly one arm under exactly one truth cannot be one.

Any row added to this file must be a value that appears NOWHERE else in the corpus
and in exactly ONE arm, so that the distinct/row ratio improves. That direction is
mechanically enforced by the ratchet in the mirror guard.

NOT EVERY ROW HERE IS CENSUS-DERIVED: THE F-05 BLOCK WAS PLAN-SPECIFIED
================================================================================
The independence claim above is about how ground truth was decided. It is NOT a
claim that every SHAPE here was chosen blind, and one block was not.

The two-field OAuth/JWT metadata rows -- `scope readonly`, `grant_type refresh`,
`realm production`, `token_type Bearer`, `Zone eu-central-1`, `Region us-east-1`,
`Type Standard`, `Tier Gold`, `Order 12345` and their siblings -- were transcribed
from `plan.md` step 3, which took them from finding F-05, which derived them BY
READING `_generic_shape_is_credential`'s thresholds. They were selected **because
their author had read the filter and PREDICTED they would KEEP**.

The bias this introduces is real, is one-directional, and is favourable rather than
convenient:

  * They all KEEP, so they contribute **0 to the over-strip NUMERATOR**.
  * They are ~37% of each safe cell (25/68 key, 28/73 token), so they **inflate the
    over-strip DENOMINATOR** substantially.
  * Net: **they bias the measured over-strip rate DOWNWARD.** Any over-strip figure
    from this corpus is therefore a LOWER bound on what the same filter would score
    against a safe population chosen without sight of the thresholds. Excluding this
    block, the iteration-1 over-strip baseline was 15.2% key / 18.8% token rather
    than the 13.9% / 13.2% the whole-cell figures reported.

The rows STAY -- deleting rows to improve a rate is the forbidden move this seam is
named for. They are labelled instead. The operational consequence: **no absolute
over-strip ceiling computed on this corpus is defensible**, and none is used as a
gate anywhere in iteration 2. Every over-strip trigger is a DELTA over a fixed
population, denominated in distinct shapes.

"LISTED" / "UNLISTED" IN THE POPULATION GUARD IS A DIAGNOSTIC LABEL
================================================================================
`test_the_census_contains_every_population_it_was_built_for` (verified present at
`tests/test_fsm_llm/test_context_unit.py`) names concrete scheme heads --
`listed_heads = ("Bearer ", "Basic ", "token ", "apikey ", "digest ")` and a longer
`unlisted_heads` tuple -- and that labelling IS derived from knowledge of the
shipped `_AUTH_SCHEME_WORDS`. It is a DIAGNOSTIC label used to check that this
corpus covers both sides of the class the plan exists to measure. It is **not** a
ground-truth call: no `ground_truth` value in this file was decided by whether a
head is listed. This file itself remains filter-blind and never imports
`fsm_llm.constants` (mechanically guarded). Same diagnostic-vs-ground-truth
distinction D-003 already draws for the measurement harness.

GROUND-TRUTH CALLS OPEN TO CHALLENGE (disclosed, not hidden)
================================================================================
* `<auth-scheme> <uuid>` is ground-truthed **credential under BOTH name families** --
  under secret-ish names (`admin_key`, `root_token`) and under identifier-ish names
  (`request_auth_key`, `correlation_ctx_key`, `session_bearer_token`). The reasoning: a value that
  literally carries an RFC 6750 / RFC 7235 scheme head is an Authorization header
  value, and that is true regardless of the name holding it. This is a CALL, made
  from name+value semantics, and it is the single call in this corpus most open to
  challenge. It is disclosed here rather than buried.
* The safe half of the wrapped-UUID interaction (F-09) is populated with NON-auth
  heads -- `Sequence <uuid>`, `Request <uuid>`, `Batch <ulid>`, `Policy <uuid>` --
  under both name families. Same two-field-with-UUID-tail SHAPE, honestly safe.
* A UUID under a secret-ish name (`acl_token`, `management_token`, `offline_token`,
  `subject_private_key`) is ground-truthed **credential**: real products issue canonical
  UUID-shaped bearer and ACL tokens. A UUID under an identifier-ish name
  (`trace_key`, `message_token`, `transaction_key`) is ground-truthed **safe**.
  The census flags this pair (S09-S13) as structurally indistinguishable by shape
  alone; the name is the only signal, and this corpus uses it honestly in both
  directions rather than resolving the class one way to flatter a rate.
* PEM PUBLIC key blocks and certificates are **safe** (RFC 7468 does not make the
  envelope sensitive); PEM PRIVATE key blocks are **credential**. The census names
  the public half as a hard false-positive class for any PEM-shape-triggers-strip
  rule.

THE DECISIVE CELLS (iteration 2, step 3) -- WHY 47 ROWS WERE ADDED
================================================================================
D-006 measured the cell that decides this whole seam at **n = 0**: a BARE canonical
UUID/ULID, under a name carrying a LISTED identifier noun, whose ground truth is
`credential`. That is exactly where an identifier-noun carve-out must fail, and the
corpus could not speak to it. Three populations were authored to fix that, blind,
from name semantics and public product knowledge only:

  (i)   **The empty cell.** 12 bare UUID/ULID rows under the LISTED resource nouns
        `session`, `job`, `run`, `lease`, `tenant`, `batch`, ground-truthed
        `credential`, each grounded in a shipping product that issues a
        UUID-shaped capability (Consul session IDs, Vault AppRole/lease
        credentials, Nomad ACL SecretIDs, Keycloak client credentials, GitLab CI
        job credentials, runner enrolment credentials). d = 12 (floor: 8).
  (ii)  **The `<head> <uuid>` 2x2x2 grid**, {listed head, unlisted head} x {listed
        noun, unlisted noun} x {credential, safe}. D-006 could measure ONE of these
        cells at n = 1 and found two EMPTY. 27 rows added, >= 4 distinct shapes in
        every cell that can be populated at all.
  (iii) **`<English noun> <UUID>` safe rows, widened.** The +2.9 pp that fired
        STOP-2 in iteration 1 came from exactly TWO distinct shapes
        (`Policy <uuid>`, `Request <uuid>`) -- an UNBOUNDED class sampled at two.
        8 further distinct English-noun heads added so the class is MEASURED
        rather than sampled.

TWO GRID CELLS ARE STRUCTURALLY EMPTY, AND THAT IS A RESULT, NOT AN OMISSION.
`<listed scheme head> <uuid>` ground-truthed **safe** cannot be authored without
contradicting this corpus's own disclosed rule (below): a value that literally
carries an RFC 6750 / RFC 7235 scheme head is an Authorization header value under
either name family. So `head LISTED / noun LISTED / safe` and
`head LISTED / noun UNLISTED / safe` are empty BY THE RULE. Filling them would have
meant inventing a row to make a table look complete. They stay empty and are
disclosed here instead.

THE GROUND-TRUTH RULE FOR AMBIGUOUS RESOURCE NOUNS -- CONTESTABLE BY CONSTRUCTION
================================================================================
Written down BEFORE any row of the decisive cells was authored, because the cell is
defined by the exact ambiguity it adjudicates. F-02 derived the trap blind: the SAME
36-character string is a session identifier and a session credential, and no shape
predicate can tell them apart.

    `session_token`  = <uuid>  is a CREDENTIAL.
    `session_id_key` = <uuid>  is an IDENTIFIER.
    The same 36-character string is both.

    **THE RULE: the name's semantic root, plus the `*_token` / `*_key` trigger,
    decides. Never the value. Never the filter.**

A name whose root names a CAPABILITY over a resource (`..._grant`, `..._ticket`,
`..._capability`, `..._enrolment`, `..._dispatch`, `..._client`, `..._provisioning`,
`..._handoff`, a CI job credential) is `credential`. A name whose root names WHICH
resource (`..._id`, `..._offset`, `..._page`, `..._step`, `..._placement`,
`..._window`, `..._slot`) is `safe`.

**This is the one place in this corpus where a reasonable author could differ**, and
it is disclosed rather than defended. The rows where the contest is live are marked
mechanically in `CONTESTED_RESOURCE_NOUN_ROWS` below -- a module-level tuple, so a
later step SELECTS them rather than re-deriving them by grep.

**Step 7 of plan-2026-07-20T144233-47e8c662 is pre-committed to a counterfactual
re-score over exactly that tuple**: every contested row (a) re-truthed to the
opposite label and (b) dropped entirely, with each stop trigger's verdict stated
under each counterfactual. That is the technique the iteration-1 adversarial review
used to retire the previously-contested `<auth-scheme> <uuid>` call, and it is what
stops a contestable authoring choice from silently selecting a design.

STOP-4 DROPS
================================================================================
STOP-4 (plan.md Pre-Mortem 4) requires dropping any entry whose ground truth cannot
be decided without reading the filter or either sibling corpus.

  **Entries dropped under STOP-4: none.** (Iteration 1: none. Iteration 2 step 3:
  none -- every one of the 47 decisive-cell rows was decidable from the rule above
  plus public product knowledge, without opening `constants.py` or either sibling
  corpus. The two structurally-empty grid cells described above are NOT STOP-4
  drops: they are decided BY the rule, not blocked by it.)

Every census shape that reached this file was decidable from the census verdict
column plus the name's semantic root. The census's own AMBIGUOUS shapes
(S09-S13 identifiers, S45 git-SHA, S49 short high-entropy, S50 unicode) were each
resolved by NAME semantics as A-7 prescribes -- e.g. S45 hex-40 under `commit_key`
is safe, S49 `aB3!kX9` under `pin_key` is credential -- not by peeking. S39
(stray wrapping quotes) inherits the inner shape's verdict, per the census's own
"(inherit inner shape)" note.

CONSTRAINTS ON THIS FILE
================================================================================
* MUST NOT import `fsm_llm.constants` -- mechanically guarded (SC-1). The corpus is
  worthless the moment its content is derived from the thing it measures.
* Contains **no vendor-real secret prefixes**. Every credential-shaped literal uses
  an invented prefix (`xk_live_`, `zc_test_`) that matches no real vendor regex.
  This file must not re-block the push (F-15).
* Name-disjoint from both sibling corpora -- mechanically guarded. Overlap is fixed
  by renaming HERE, never by weakening the guard. 43 names authored blind collided
  with `context_key_corpus.py` and were renamed by ADDING A QUALIFIER that preserves
  the semantic root the ground truth rests on (`private_key` -> `subject_private_key`,
  `trace_token` -> `trace_id_token`, `refresh_token` -> `refresh_grant_token`, ...).
  Not one ground-truth verdict changed as a result: the qualifier never crosses the
  identifier-ish / secret-ish boundary that decides the call. Collisions were detected
  mechanically, by set intersection, without reading a single sibling-corpus entry.

INTERFACE
================================================================================
`(name, value, ground_truth)` with `ground_truth in {"credential", "safe"}`, and
`arm_of(name)` reused from `holdout_key_corpus.py` per A-8 rather than
reimplementing arm classification a third time. Exports mirror the holdout's shape:
four flat per-arm lists plus the combined `CENSUS`.
"""

from tests.test_fsm_llm.fixtures.holdout_key_corpus import arm_of

__all__ = [
    "CENSUS",
    "CONTESTED_RESOURCE_NOUN_ROWS",
    "GROUND_TRUTH_VALUES",
    "KEY_ARM_CREDENTIAL",
    "KEY_ARM_SAFE",
    "TOKEN_ARM_CREDENTIAL",
    "TOKEN_ARM_SAFE",
    "arm_of",
]

GROUND_TRUTH_VALUES = frozenset({"credential", "safe"})

# The rows where the resource-noun ground-truth rule in the banner is genuinely
# open to challenge -- a reasonable author reading the same name could call each of
# these an IDENTIFIER instead of a CREDENTIAL. Every one is currently ground-truthed
# `credential`.
#
# INTERFACE CONTRACT: a tuple of context-key NAMES present in `CENSUS`. Consumers
# select rows by membership, never by re-deriving the list from a grep or a regex
# over the file. Two consumers exist by construction: the corpus-adequacy guard in
# `test_context_unit.py` (which asserts every name here is a real `credential` row),
# and step 7's counterfactual re-score, which must re-run its measurement twice --
# once with these rows re-truthed to `safe`, once with them dropped -- and report
# whether either stop trigger's verdict changes. Adding a row here WIDENS that
# counterfactual; it never changes a measured number on its own.
CONTESTED_RESOURCE_NOUN_ROWS: tuple[str, ...] = (
    "batch_ingest_token",
    "consul_session_grant_key",
    "cursor_resume_token",
    "job_dispatch_wrapped_key",
    "lease_capability_token",
    "lease_renewal_wrapped_token",
    "nomad_job_dispatch_token",
    "partition_rebalance_key",
    "session_ticket_token",
    "shard_rebalance_key",
    "vault_lease_renewal_key",
)

# A >200-character unstructured high-entropy blob (census S48). Held as a named
# constant because it is used in both arms and a wrapped literal would be unreadable.
_LONG_OPAQUE_BLOB = (
    "aZ3kQm9RpT2sLxW7vNcBqW8eR4tY6uI0oP2aS5dF7gH9jK1lZ3xC6vB8nM4qW7eR2tY5uI8"
    "oP1aS4dF6gH8jK0lZ2xC5vB7nM3qW6eR1tY4uI7oP0aS3dF5gH7jK9lZ1xC4vB6nM2qW5eR"
    "0tY3uI6oP9aS2dF4gH6jK8lZ0xC3vB5nM1qW4eR9tY2uI5oP8aS1dF3gH5jK7lZ9xC2vB4nM"
)

_PEM_PRIVATE = (
    "-----BEGIN PRIVATE KEY-----\n"
    "MIIBVgIBADANBgkqhkiG9w0BAQEFAASCAUAwggE8AgEAAkEAqW8eR4tY6uI0oP2a\n"
    "-----END PRIVATE KEY-----"
)
_PEM_RSA_PRIVATE = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEowIBAAKCAQEAx7Vn9kQm3sLpT2RwqW8eR4tY6uI0oP2aS5dF7gH9jK1lZ3xC\n"
    "-----END RSA PRIVATE KEY-----"
)
_PEM_PUBLIC = (
    "-----BEGIN PUBLIC KEY-----\n"
    "MFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBAKx7Vn9kQm3sLpT2RwqW8eR4tY6uI0oP\n"
    "-----END PUBLIC KEY-----"
)
_PEM_CERT = (
    "-----BEGIN CERTIFICATE-----\n"
    "MIIBkTCB+wIJANlqZ3kQm9RpT2sLxW7vNcBqW8eR4tY6uI0oP2aS5dF7gH9jK1lZ\n"
    "-----END CERTIFICATE-----"
)

_JWT3 = (
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0."
    "4pC1x9k3vQ7Z2mN8fJ1bH6yT0aR5cE9wL3sV7uK2dQ"
)
_JWT2_MALFORMED = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"

_UUID_LOWER = "3fa85f64-5717-4562-b3fc-2c963f66afa6"
_UUID_UPPER = "3FA85F64-5717-4562-B3FC-2C963F66AFA6"
_UUID_V7 = "018f3c9a-7b21-7c44-9e05-2b1d6a4f8e33"
_ULID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
_KSUID = "0ujsswThIGTUYm2K8FjOOfXtY1K"
_NANOID = "V1StGXR8_Z5jdHi6B-myT"

_HEX32 = "9f86d081884c7d659a2feaa0c55ad015"
_HEX64 = "2c26b46b68ffc68ff99b453c1d304134b45d7d3f2e1a6c98705f43b21c0d8e67"
_HEX40_GIT = "a94a8fe5ccb19ba61c4c0873d391e987982fbbd3"
_B64 = "Tm90UmVhbFNlY3JldFZhbHVlSGVyZQ=="
_B64_USERPASS = "dXNlcjpwYXNzd29yZFZhbHVl"
_B64URL43 = "dGhpc2lzYW5vcGFxdWV0b2tlbnZhbHVlZm9ydGVzdA"
_B64_OPAQUE = "dGhpc2lzYW5vcGFxdWV0b2tlbnZhbHVl"
_B32 = "ORSXG5BAORSXG5BAORSXG5BAOR"
_ALNUM20 = "aZ3kQm9RpT2sLxW7vNcB"
_ALNUM24 = "aZ3kQm9RpT2sLxW7vNcB1234"
_ALNUM_LOWER24 = "a1b2c3d4e5f6g7h8i9j0k1l2"


# ============================================================================
# KEY ARM -- credential (census verdict STRIP, or name-semantics secret-ish)
# ============================================================================
KEY_ARM_CREDENTIAL: list[tuple[str, str, str]] = [
    # -- scheme-word class, heads that a reader would call "obviously listed" (S27-S29)
    ("platform_api_key", f"Bearer {_ALNUM24}", "credential"),
    ("service_key", f"Basic {_B64_USERPASS}", "credential"),
    ("gateway_key", f"token {_ALNUM24}", "credential"),
    ("partner_key", f"apikey {_HEX32}", "credential"),
    ("proxy_auth_key", f"digest {_HEX32}", "credential"),
    # -- scheme-word class, heads from the IANA HTTP Authentication Scheme Registry
    #    and from real transport protocols. These are the class this plan exists for.
    ("federation_key", f"OAuth2 {_ALNUM24}", "credential"),
    ("request_signing_key", f"AWS4-HMAC-SHA256 {_HEX32}", "credential"),
    ("blob_storage_key", f"SharedKey {_B64_OPAQUE}", "credential"),
    ("caster_key", f"ntrip {_ALNUM24}", "credential"),
    ("vault_key", f"secret {_ALNUM24}", "credential"),
    ("edge_key", f"Sigv4Custom {_HEX32}", "credential"),
    ("realm_auth_key", f"GSSAPI {_B64_OPAQUE}", "credential"),
    ("domain_key", f"Kerberos5 {_ALNUM24}", "credential"),
    ("header_key", f"X-Api-Key {_HEX32}", "credential"),
    ("broker_key", f"SASL {_ALNUM24}", "credential"),
    ("smtp_key", f"PLAIN {_B64_USERPASS}", "credential"),
    ("charger_key", f"OCPP16 {_ALNUM24}", "credential"),
    ("sso_key", f"Negotiate {_B64_OPAQUE}", "credential"),
    ("device_auth_key", f"HOBA {_HEX32}", "credential"),
    ("mtls_key", f"Mutual {_ALNUM24}", "credential"),
    ("sasl_scram_key", f"SCRAM-SHA-256 {_B64_OPAQUE}", "credential"),
    ("dpop_key", f"DPoP {_JWT3}", "credential"),
    ("http_sig_key", f"Signature {_ALNUM24}", "credential"),
    ("hawk_key", f"Hawk {_HEX32}", "credential"),
    ("cdn_key", f"EG1-HMAC-SHA256 {_B64_OPAQUE}", "credential"),
    ("ws_key", f"Concealed {_ALNUM24}", "credential"),
    ("mq_key", f"AMQPLAIN {_ALNUM24}", "credential"),
    ("ldap_key", f"SIMPLE {_B64_USERPASS}", "credential"),
    ("radius_key", f"PAP {_ALNUM24}", "credential"),
    ("grpc_key", f"MetadataAuth {_B64_OPAQUE}", "credential"),
    ("iot_key", f"SigV2 {_ALNUM24}", "credential"),
    ("mqtt_key", f"MQTTv5 {_HEX32}", "credential"),
    # -- opaque high-entropy shapes (S01-S07, S48)
    ("client_app_key", _HEX32, "credential"),
    ("hmac_digest_key", _HEX64, "credential"),
    ("upload_key", _B64, "credential"),
    ("rotation_key", _B64URL43, "credential"),
    ("otp_seed_key", _B32, "credential"),
    ("session_secret_key", _ALNUM20, "credential"),
    ("webhook_key", _ALNUM_LOWER24, "credential"),
    ("dump_key", _LONG_OPAQUE_BLOB, "credential"),
    # -- structured credential shapes (S35, S36, S44)
    ("jwt_key", _JWT3, "credential"),
    ("legacy_jwt_key", _JWT2_MALFORMED, "credential"),
    ("tls_private_key", _PEM_PRIVATE, "credential"),
    ("rsa_pair_key", _PEM_RSA_PRIVATE, "credential"),
    # -- invented vendor-prefix shapes (S42, S43). NOT real vendor prefixes.
    ("billing_key", f"xk_live_{_HEX32}", "credential"),
    ("sandbox_key", f"zc_test_{_HEX32}", "credential"),
    # -- colon composites carrying a credential (S30, S31)
    ("db_key", "svcuser:Tr0ub4dor3xKq", "credential"),
    ("oauth_client_key", f"client-42:{_ALNUM20}", "credential"),
    # -- URLs that embed a credential (S21, S22)
    ("callback_key", "https://svc:p4ssW0rdXyz@host.example/api", "credential"),
    ("reset_link_key", "https://x.example/reset?token=Ab12Cd34Ef56Gh78", "credential"),
    # -- census AMBIGUOUS shapes resolved by NAME semantics (S49, S50, S39, pct-enc)
    ("pin_key", "aB3!kX9", "credential"),
    ("master_unlock_key", "cle-secrete-\N{KEY}", "credential"),
    ("encoded_secret_key", f"{_ALNUM24}%3D%3D", "credential"),
    ("quoted_key", f'"{_ALNUM24}"', "credential"),
    # -- identifier-shaped values under SECRET-ish names (S09-S13).
    #    Real products issue canonical UUID-shaped bearer and ACL tokens.
    ("acl_key", _UUID_LOWER, "credential"),
    ("management_key", _UUID_UPPER, "credential"),
    ("offline_key", _ULID, "credential"),
    ("subject_private_key", _KSUID, "credential"),
    ("bootstrap_key", _NANOID, "credential"),
    ("agent_secret_key", _UUID_V7, "credential"),
    # -- F-09: scheme-wrapped UUID under SECRET-ish names
    ("admin_key", f"Bearer {_UUID_LOWER}", "credential"),
    ("root_key", f"Bearer {_ULID}", "credential"),
    ("consul_acl_key", f"Bearer {_UUID_V7}", "credential"),
    # -- F-09: scheme-wrapped UUID under IDENTIFIER-ish names. See the disclosed
    #    ground-truth call in the banner: the scheme head makes it a credential.
    ("request_auth_key", f"Bearer {_UUID_LOWER}", "credential"),
    ("correlation_ctx_key", f"Bearer {_ULID}", "credential"),
    ("dedupe_ctl_key", f"OAuth2 {_UUID_LOWER}", "credential"),
    ("idempotency_ctl_key", f"SharedKey {_UUID_V7}", "credential"),
    # ========================================================================
    # ITERATION 2, STEP 3 -- THE DECISIVE CELL (i). D-006 measured this at n = 0.
    # BARE canonical UUID/ULID, name carries a LISTED resource noun (`session`,
    # `job`, `run`, `lease`, `tenant`, `batch`), ground truth `credential`.
    # Grounded in shipping products that issue UUID-shaped capabilities; decided
    # by the banner's resource-noun rule (semantic root + `*_key` trigger), never
    # by the value and never by the filter. Every value below is NEW and appears
    # in exactly ONE arm.
    # ========================================================================
    # Consul creates sessions via /v1/session/create and returns a UUID; holding
    # it renews and releases locks. CONTESTED: it is also a session identifier.
    ("consul_session_grant_key", "36fd1dfa-19a5-4345-85c5-b0a2d2ee8235", "credential"),
    # A GitLab CI job credential authenticates API calls AS the job.
    ("gitlab_job_ci_key", "97c18e97-ef6c-4497-b0b2-10bd7c5633ab", "credential"),
    # Vault lease renewal: presenting it extends the lease. CONTESTED -- a lease
    # id reads as an identifier to a reader who has not tried to renew one.
    ("vault_lease_renewal_key", "fdba6878-b9f6-49f7-a9e7-3a7791603f01", "credential"),
    # Per-tenant API credential issued at provisioning time.
    ("tenant_provisioning_key", "7a3237a3-c11e-416c-b68c-66774098a037", "credential"),
    ("batch_upload_grant_key", "01FHTQDXQGPC1917467W1XYK2S", "credential"),
    ("run_replay_grant_key", "872ddbe6-3d8c-4a71-8290-35ba0dfbe305", "credential"),
    # ========================================================================
    # ITERATION 2, STEP 3 -- GRID (ii). `<head> <uuid>`, key arm, credential.
    # Heads are labelled listed/unlisted DIAGNOSTICALLY (same status as the head
    # labels in `test_the_census_contains_every_population_it_was_built_for`);
    # no ground truth below was decided by whether a head is listed. The truth
    # is the banner's disclosed call: a value carrying an RFC 6750 / RFC 7235
    # scheme head is an Authorization header value under either name family.
    # ========================================================================
    # (A) head LISTED x noun LISTED x credential -- D-006 measured this at n = 1.
    (
        "tenant_scoped_access_key",
        "Bearer 860805af-e709-4a4b-937d-cecdcdf9d101",
        "credential",
    ),
    ("job_dispatch_wrapped_key", "token 01EQAZ88A4GFSTTHQMKD7PYX1Z", "credential"),
    (
        "shard_rebalance_key",
        "digest 576297d4-1347-4f89-9326-a6be1fba5592",
        "credential",
    ),
    # (B) head LISTED x noun UNLISTED x credential
    (
        "manifest_signing_key",
        "Bearer 71056dab-8092-4770-9bad-b8409fc80346",
        "credential",
    ),
    ("docket_access_key", "token 01X4MAETWH5NXZTYRZMD02ZEX5", "credential"),
    # (C) head UNLISTED x noun LISTED x credential
    (
        "partition_rebalance_key",
        "OAuth2 59259872-fa09-4109-8244-700e712dd042",
        "credential",
    ),
    (
        "lease_handoff_key",
        "Kerberos5 d89cb7db-2821-4fa9-b218-50489c5510a1",
        "credential",
    ),
    # (E) head UNLISTED x noun UNLISTED x credential -- D-006 found this EMPTY.
    (
        "dossier_release_key",
        "Sigv4Custom 494d89cf-182e-41da-862e-0a11042e72f5",
        "credential",
    ),
    ("escrow_release_key", "PLAIN 2e594f42-8346-4812-88da-bf1ca3fff8f2", "credential"),
]


# ============================================================================
# KEY ARM -- safe (census verdict KEEP, or name-semantics identifier-ish)
# ============================================================================
KEY_ARM_SAFE: list[tuple[str, str, str]] = [
    # -- F-05 OVER-STRIP CASUALTIES: two-field values that have EXACTLY the shape of
    #    a scheme wrap but are ordinary metadata. Without these rows the over-strip
    #    cost of the shape test is unmeasured by construction.
    ("subscription_key", "scope readonly", "safe"),
    ("grant_key", "grant_type refresh", "safe"),
    ("tenant_realm_key", "realm production", "safe"),
    ("auth_scheme_key", "token_type Bearer", "safe"),
    ("availability_key", "Zone eu-central-1", "safe"),
    ("cluster_region_key", "Region us-east-1", "safe"),
    ("sku_key", "Type Standard", "safe"),
    ("account_key", "Tier Gold", "safe"),
    ("invoice_order_key", "Order 12345", "safe"),
    ("lifecycle_key", "Status active", "safe"),
    ("env_mode_key", "Mode sandbox", "safe"),
    ("schema_key", "Version 2", "safe"),
    ("i18n_locale_key", "Locale en US", "safe"),
    ("billing_plan_key", "Plan premium", "safe"),
    ("shipping_key", "Carrier DHL", "safe"),
    ("currency_key", "Currency EUR", "safe"),
    ("priority_key", "Priority high", "safe"),
    ("notify_key", "Channel email", "safe"),
    ("encoding_key", "Format json", "safe"),
    ("retention_key", "Retention 30d", "safe"),
    ("ratelimit_key", "Limit 1000", "safe"),
    # -- same class, but the trailing field is LONG and wordy. Length alone must not
    #    buy a STRIP when the content is plainly prose.
    ("catalog_key", "Category Home and Garden Furnishings", "safe"),
    (
        "policy_key",
        "Description Standard commercial retention policy for archived invoices",
        "safe",
    ),
    (
        "template_key",
        "Subject Your order has shipped and is on its way",
        "safe",
    ),
    (
        "audit_key",
        "Reason customer requested a manual review of the pricing tier",
        "safe",
    ),
    # -- identifier nouns holding identifier values (S09-S14). Name semantics say
    #    these record WHICH request/message/span, not a credential.
    ("trace_key", _UUID_LOWER, "safe"),
    ("span_key", _UUID_UPPER, "safe"),
    ("message_id_key", _ULID, "safe"),
    ("event_id_key", _KSUID, "safe"),
    ("transaction_key", _UUID_V7, "safe"),
    ("replay_key", _NANOID, "safe"),
    ("partition_id_key", _UUID_LOWER, "safe"),
    ("shard_id_key", _ULID, "safe"),
    # -- F-09 safe half: two-field values with a UUID/ULID tail whose head is NOT an
    #    auth scheme. Same shape as `Bearer <uuid>`, honestly safe.
    ("batch_key", f"Sequence {_UUID_LOWER}", "safe"),
    ("ledger_key", f"Request {_UUID_V7}", "safe"),
    ("job_key", f"Batch {_ULID}", "safe"),
    ("acl_policy_key", f"Policy {_UUID_LOWER}", "safe"),
    # -- ordinary scalars (S08, S14-S18, S23-S26, S47)
    ("row_number_key", "48291", "safe"),
    ("index_row_key", "42", "safe"),
    ("discord_key", "1234567890123456789", "safe"),
    ("tier_key", "premium-tier", "safe"),
    ("status_key", "active", "safe"),
    ("desk_key", "Customer Support Desk", "safe"),
    ("created_key", "2026-07-20T10:00:00Z", "safe"),
    ("enabled_key", "true", "safe"),
    ("disabled_key", "false", "safe"),
    ("optional_key", "None", "safe"),
    ("missing_key", "null", "safe"),
    ("empty_key", "", "safe"),
    ("blank_key", " ", "safe"),
    ("port_key", "8080", "safe"),
    # -- paths, URLs, dotted namespaces, semver (S19, S20, S32, S33, S34)
    ("cert_path_key", "/etc/app/certs/service.pem", "safe"),
    ("data_path_key", "/var/lib/app/data", "safe"),
    ("status_url_key", "https://api.example.internal/v1/status", "safe"),
    ("config_namespace_key", "app.settings.theme", "safe"),
    ("version_key", "1.4.2", "safe"),
    ("endpoint_key", "10.0.0.12:8080", "safe"),
    # -- hash-shaped values separated only by name context or an algo prefix (S45, S46)
    ("commit_key", _HEX40_GIT, "safe"),
    ("digest_key", f"sha256:{_HEX64}", "safe"),
    # -- PEM public half: the census's named hard false-positive class (S37)
    ("subject_public_key", _PEM_PUBLIC, "safe"),
    ("verify_key", _PEM_CERT, "safe"),
    # -- references and templates -- pointers to a secret, not the secret (S38, S40, S41)
    ("menu_key", "Caf%C3%A9%20Menu", "safe"),
    ("placeholder_key", "${API_KEY}", "safe"),
    ("mustache_key", "{{token}}", "safe"),
    ("envref_key", "$OPENAI_API_KEY", "safe"),
    ("envvar_key", "env:OPENAI_API_KEY", "safe"),
    # -- unicode label under a `*_key` name (S50 resolved safe by name semantics)
    ("city_key", "Z\N{LATIN SMALL LETTER U WITH DIAERESIS}rich Hauptbahnhof", "safe"),
    ("locale_code_key", "fr-CA", "safe"),
    # ========================================================================
    # ITERATION 2, STEP 3 -- GRID (ii), safe half, key arm.
    # (D) head UNLISTED (non-auth English noun) x noun LISTED x safe
    # (F) head UNLISTED (non-auth English noun) x noun UNLISTED x safe
    # The name root says WHICH resource, not a capability over it, so the
    # banner's rule makes these `safe`.
    # ========================================================================
    ("run_step_key", "Segment d4aaee4b-9638-4e64-b34a-53ac1c72923e", "safe"),
    ("shard_placement_key", "Record 71099285-e48b-44c5-8dea-c22f3a71bdff", "safe"),
    ("replay_window_key", "Draft b0dfc5b1-5791-40f7-af39-ca9cc78c3719", "safe"),
    ("manifest_line_key", "Shipment f251e24c-8a6f-4f00-9e0d-8ed053508434", "safe"),
    ("roster_slot_key", "Position 58dbe200-870d-4d4e-983f-7499b9a4f7ce", "safe"),
    # ========================================================================
    # ITERATION 2, STEP 3 -- POPULATION (iii): `<English noun> <UUID>` WIDENED.
    # Iteration 1's entire +2.9 pp over-strip figure came from TWO distinct
    # shapes, `Policy <uuid>` and `Request <uuid>`. The class of English nouns
    # that can precede an identifier is UNBOUNDED, so a two-shape sample sets
    # the measured cost by accident. These rows make the class measured.
    # DO NOT delete them to improve a rate; that is the forbidden move.
    # ========================================================================
    ("catalogue_item_key", "Item 01E8HZV2KK249KWS0DGBW0SK2C", "safe"),
    ("contract_clause_key", "Clause 02ac9f84-a057-4bb8-a268-ee7774471edb", "safe"),
    ("asset_serial_key", "Asset c43080f0-a535-41d4-a125-f7e137d5f774", "safe"),
    ("article_revision_key", "Revision f06a71f2-68ec-486c-991c-4ddbd3b571b2", "safe"),
]


# ============================================================================
# TOKEN ARM -- credential
# ============================================================================
TOKEN_ARM_CREDENTIAL: list[tuple[str, str, str]] = [
    # -- scheme-word class, "obviously listed" heads
    ("platform_access_token", f"Bearer {_ALNUM24}", "credential"),
    ("gateway_auth_token", f"Basic {_B64_USERPASS}", "credential"),
    ("wrapped_bearer_token", f"token {_ALNUM24}", "credential"),
    ("service_account_token", f"apikey {_HEX32}", "credential"),
    ("proxy_token", f"digest {_HEX32}", "credential"),
    # -- scheme-word class, registry / transport-protocol heads
    ("federation_token", f"OAuth2 {_ALNUM24}", "credential"),
    ("sigv4_token", f"AWS4-HMAC-SHA256 {_HEX32}", "credential"),
    ("blob_token", f"SharedKey {_B64_OPAQUE}", "credential"),
    ("caster_token", f"ntrip {_ALNUM24}", "credential"),
    ("vault_token", f"secret {_ALNUM24}", "credential"),
    ("edge_token", f"Sigv4Custom {_HEX32}", "credential"),
    ("krb_token", f"GSSAPI {_B64_OPAQUE}", "credential"),
    ("domain_token", f"Kerberos5 {_ALNUM24}", "credential"),
    ("header_token", f"X-Api-Key {_HEX32}", "credential"),
    ("broker_token", f"SASL {_ALNUM24}", "credential"),
    ("smtp_token", f"PLAIN {_B64_USERPASS}", "credential"),
    ("charger_token", f"OCPP16 {_ALNUM24}", "credential"),
    ("sso_negotiate_token", f"Negotiate {_B64_OPAQUE}", "credential"),
    ("device_auth_token", f"HOBA {_HEX32}", "credential"),
    ("mtls_token", f"Mutual {_ALNUM24}", "credential"),
    ("scram_token", f"SCRAM-SHA-256 {_B64_OPAQUE}", "credential"),
    ("dpop_token", f"DPoP {_JWT3}", "credential"),
    ("httpsig_token", f"Signature {_ALNUM24}", "credential"),
    ("hawk_token", f"Hawk {_HEX32}", "credential"),
    ("akamai_token", f"EG1-HMAC-SHA256 {_B64_OPAQUE}", "credential"),
    ("ws_token", f"Concealed {_ALNUM24}", "credential"),
    ("amqp_token", f"AMQPLAIN {_ALNUM24}", "credential"),
    ("ldap_token", f"SIMPLE {_B64_USERPASS}", "credential"),
    ("radius_token", f"PAP {_ALNUM24}", "credential"),
    ("grpc_token", f"MetadataAuth {_B64_OPAQUE}", "credential"),
    ("iot_token", f"SigV2 {_ALNUM24}", "credential"),
    ("mqtt_token", f"MQTTv5 {_HEX32}", "credential"),
    # -- opaque high-entropy shapes
    ("refresh_grant_token", _HEX32, "credential"),
    ("oidc_id_token", _HEX64, "credential"),
    ("csrf_form_token", _B64, "credential"),
    ("opaque_token", _B64URL43, "credential"),
    ("totp_seed_token", _B32, "credential"),
    ("upload_slot_token", _ALNUM20, "credential"),
    ("webhook_sig_token", _ALNUM_LOWER24, "credential"),
    ("dump_token", _LONG_OPAQUE_BLOB, "credential"),
    # -- structured credential shapes
    ("jwt_assertion_token", _JWT3, "credential"),
    ("truncated_token", _JWT2_MALFORMED, "credential"),
    ("pem_token", _PEM_PRIVATE, "credential"),
    ("rsa_token", _PEM_RSA_PRIVATE, "credential"),
    # -- invented vendor-prefix shapes. NOT real vendor prefixes.
    ("billing_token", f"xk_live_{_HEX32}", "credential"),
    ("sandbox_token", f"zc_test_{_HEX32}", "credential"),
    # -- colon composites
    ("db_token", "svcuser:Tr0ub4dor3xKq", "credential"),
    ("client_pair_token", f"client-42:{_ALNUM20}", "credential"),
    # -- URLs embedding a credential
    ("callback_token", "https://svc:p4ssW0rdXyz@host.example/api", "credential"),
    ("reset_token", "https://x.example/reset?secret=Ab12Cd34Ef56Gh78", "credential"),
    # -- census AMBIGUOUS shapes resolved by NAME semantics
    ("pin_token", "aB3!kX9", "credential"),
    ("unicode_secret_token", "cle-secrete-\N{KEY}", "credential"),
    ("encoded_token", f"{_ALNUM24}%3D%3D", "credential"),
    ("quoted_token", f'"{_ALNUM24}"', "credential"),
    # -- identifier-shaped values under SECRET-ish names
    ("acl_token", _UUID_LOWER, "credential"),
    ("management_token", _UUID_UPPER, "credential"),
    ("offline_token", _ULID, "credential"),
    ("private_token", _KSUID, "credential"),
    ("bootstrap_token", _NANOID, "credential"),
    ("agent_secret_token", _UUID_V7, "credential"),
    # -- F-09: scheme-wrapped UUID under SECRET-ish names
    ("admin_root_token", f"Bearer {_UUID_LOWER}", "credential"),
    ("root_token", f"Bearer {_ULID}", "credential"),
    ("nomad_acl_token", f"Bearer {_UUID_V7}", "credential"),
    # -- F-09: scheme-wrapped UUID under IDENTIFIER-ish names
    ("session_bearer_token", f"Bearer {_UUID_LOWER}", "credential"),
    ("trace_ctx_token", f"OAuth2 {_UUID_LOWER}", "credential"),
    ("dedupe_ctl_token", f"SharedKey {_UUID_V7}", "credential"),
    # ========================================================================
    # ITERATION 2, STEP 3 -- THE DECISIVE CELL (i), token arm.
    # BARE canonical UUID/ULID under a LISTED resource noun, `credential`.
    # ========================================================================
    ("session_handoff_token", "739e7ff9-e69e-466c-b4cc-4b4abc86e316", "credential"),
    # Nomad ACL SecretIDs are canonical UUIDs. CONTESTED: "dispatch" could name
    # the dispatch rather than the capability to dispatch.
    ("nomad_job_dispatch_token", "b693b8eb-c0d5-44ab-8de3-6377e8e51da2", "credential"),
    # Keycloak issues UUID-shaped client credentials by default.
    (
        "keycloak_tenant_client_token",
        "e3368415-855e-40d1-b382-a3f418082508",
        "credential",
    ),
    # CONTESTED: a lease capability handle is also a lease identifier.
    ("lease_capability_token", "013TGVCMBCR4PGH6HHFVY5RNC6", "credential"),
    (
        "run_worker_enrolment_token",
        "02c33d86-deb4-46fc-8c5d-3181fee37282",
        "credential",
    ),
    # A TLS/Kerberos session ticket resumes a session on presentation.
    # CONTESTED: "ticket" also reads as an identifier in issue-tracker English.
    ("session_ticket_token", "05a5c559-ce35-42ff-a6b1-5803990500a7", "credential"),
    # ========================================================================
    # ITERATION 2, STEP 3 -- GRID (ii), credential half, token arm.
    # ========================================================================
    # (A) head LISTED x noun LISTED x credential
    (
        "lease_renewal_wrapped_token",
        "Basic c4e87120-24a7-42f6-92cd-1eeb373be391",
        "credential",
    ),
    ("run_attach_token", "apikey a99184f5-1457-4b38-8887-9e46e61759c4", "credential"),
    # (B) head LISTED x noun UNLISTED x credential
    (
        "voucher_redeem_token",
        "Basic 2b91304d-6f48-42b0-8a13-96f741f395cb",
        "credential",
    ),
    ("roster_sync_token", "apikey 23e7c536-6472-43a9-a91b-5b97cc291697", "credential"),
    # (C) head UNLISTED x noun LISTED x credential
    (
        "cursor_resume_token",
        "SharedKey ce40bcea-f557-4ea3-a61d-f6331839f384",
        "credential",
    ),
    ("tenant_federation_token", "GSSAPI 019B0ZAW2W2HHKFHEZZAWEPN56", "credential"),
    ("batch_ingest_token", "SASL d36ef446-21f7-4ee2-ad1d-5e22b61edac8", "credential"),
    # (E) head UNLISTED x noun UNLISTED x credential -- D-006 found this EMPTY.
    (
        "parcel_carrier_token",
        "X-Api-Key 6bbf646e-684b-447d-b3b4-47e7e2fb49bf",
        "credential",
    ),
    (
        "warrant_issue_token",
        "AWS4-HMAC-SHA256 0188X3DZ6Z66GQ81P8CDEGE2QX",
        "credential",
    ),
]


# ============================================================================
# TOKEN ARM -- safe
# ============================================================================
TOKEN_ARM_SAFE: list[tuple[str, str, str]] = [
    # -- F-05 OVER-STRIP CASUALTIES under `*_token` names. These are the exact rows
    #    the shape test puts at risk; they must exist BEFORE the filter is touched.
    ("scope_token", "scope readonly", "safe"),
    ("grant_token", "grant_type refresh", "safe"),
    ("realm_token", "realm production", "safe"),
    ("type_token", "token_type Bearer", "safe"),
    ("zone_token", "Zone eu-central-1", "safe"),
    ("region_token", "Region us-east-1", "safe"),
    ("class_token", "Type Standard", "safe"),
    ("tier_token", "Tier Gold", "safe"),
    ("order_token", "Order 12345", "safe"),
    ("status_token", "Status active", "safe"),
    ("mode_token", "Mode sandbox", "safe"),
    ("version_token", "Version 2", "safe"),
    ("locale_token", "Locale en US", "safe"),
    ("plan_token", "Plan premium", "safe"),
    ("carrier_token", "Carrier DHL", "safe"),
    ("currency_token", "Currency EUR", "safe"),
    ("priority_token", "Priority high", "safe"),
    ("channel_token", "Channel email", "safe"),
    ("format_token", "Format json", "safe"),
    ("retention_token", "Retention 30d", "safe"),
    ("limit_token", "Limit 1000", "safe"),
    ("expiry_token", "expires_in 3600", "safe"),
    ("audience_token", "audience https://api.example.internal", "safe"),
    ("issuer_token", "issuer https://idp.example.internal", "safe"),
    # -- same class, long wordy trailing field
    ("category_token", "Category Home and Garden Furnishings", "safe"),
    (
        "policy_desc_token",
        "Description Standard commercial retention policy for archived invoices",
        "safe",
    ),
    (
        "subject_token",
        "Subject Your order has shipped and is on its way",
        "safe",
    ),
    (
        "reason_token",
        "Reason customer requested a manual review of the pricing tier",
        "safe",
    ),
    # -- identifier nouns holding identifier values
    ("trace_id_token", _UUID_LOWER, "safe"),
    ("span_token", _UUID_UPPER, "safe"),
    ("message_token", _ULID, "safe"),
    ("event_token", _KSUID, "safe"),
    ("transaction_token", _UUID_V7, "safe"),
    ("replay_token", _NANOID, "safe"),
    ("correlation_id_token", _UUID_LOWER, "safe"),
    ("idempotency_token", _ULID, "safe"),
    ("request_token", _UUID_V7, "safe"),
    ("dedupe_token", _KSUID, "safe"),
    # -- F-09 safe half: UUID tail under a NON-auth head
    ("batch_token", f"Sequence {_UUID_LOWER}", "safe"),
    ("ledger_token", f"Request {_UUID_V7}", "safe"),
    ("job_token", f"Batch {_ULID}", "safe"),
    ("acl_policy_token", f"Policy {_UUID_LOWER}", "safe"),
    # -- ordinary scalars
    ("row_token", "48291", "safe"),
    ("index_token", "42", "safe"),
    ("snowflake_token", "1234567890123456789", "safe"),
    ("slug_token", "premium-tier", "safe"),
    ("state_token", "active", "safe"),
    ("desk_token", "Customer Support Desk", "safe"),
    ("created_token", "2026-07-20T10:00:00Z", "safe"),
    ("enabled_token", "true", "safe"),
    ("bool_token", "false", "safe"),
    ("optional_token", "None", "safe"),
    ("null_token", "null", "safe"),
    ("empty_token", "", "safe"),
    ("blank_token", " ", "safe"),
    ("port_token", "8080", "safe"),
    # -- paths, URLs, dotted namespaces, semver
    ("certpath_token", "/etc/app/certs/service.pem", "safe"),
    ("datapath_token", "/var/lib/app/data", "safe"),
    ("status_url_token", "https://api.example.internal/v1/status", "safe"),
    ("config_token", "app.settings.theme", "safe"),
    ("semver_token", "1.4.2", "safe"),
    ("endpoint_token", "10.0.0.12:8080", "safe"),
    # -- hash-shaped values
    ("commit_token", _HEX40_GIT, "safe"),
    ("digest_token", f"sha256:{_HEX64}", "safe"),
    # -- PEM public half
    ("pubkey_token", _PEM_PUBLIC, "safe"),
    ("cert_token", _PEM_CERT, "safe"),
    # -- references and templates
    ("menu_token", "Caf%C3%A9%20Menu", "safe"),
    ("placeholder_token", "${API_KEY}", "safe"),
    ("mustache_token", "{{token}}", "safe"),
    ("envref_token", "$OPENAI_API_KEY", "safe"),
    ("envvar_token", "env:OPENAI_API_KEY", "safe"),
    # -- unicode label
    (
        "city_token",
        "Z\N{LATIN SMALL LETTER U WITH DIAERESIS}rich Hauptbahnhof",
        "safe",
    ),
    ("region_name_token", "fr-CA", "safe"),
    # ========================================================================
    # ITERATION 2, STEP 3 -- GRID (ii), safe half, token arm.
    # (D) noun LISTED x safe, (F) noun UNLISTED x safe.
    # ========================================================================
    ("cursor_page_token", "Entry 019cbf7a-feba-4e73-9952-f54c6290ed3a", "safe"),
    ("partition_offset_token", "Sequence 0171CXXTNQB8V5G4G46DV8SGTF", "safe"),
    ("docket_entry_token", "Invoice 7dbe1ed5-9379-4dad-b507-277d7da4fa97", "safe"),
    ("dossier_index_token", "Folder 72f4906f-d2f6-450c-8d46-49c808e1d122", "safe"),
    # ========================================================================
    # ITERATION 2, STEP 3 -- POPULATION (iii), token arm. See the key-arm block.
    # ========================================================================
    ("warehouse_bin_token", "Bin 89404757-a396-4118-bf3d-397321768f72", "safe"),
    ("survey_response_token", "Response 94eb4bf5-5c3e-4a54-ac95-5a13d2ff8cad", "safe"),
    ("booking_slot_token", "Booking 01T43DSTJ86P1D736XTAK0R2C2", "safe"),
    ("route_leg_token", "Leg 7de6a7a8-139f-45a0-b714-80867b42b892", "safe"),
]


CENSUS: list[tuple[str, str, str]] = (
    KEY_ARM_CREDENTIAL + KEY_ARM_SAFE + TOKEN_ARM_CREDENTIAL + TOKEN_ARM_SAFE
)
