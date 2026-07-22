from __future__ import annotations

"""
Constants and configuration values for the FSM-LLM framework.
"""

import math
import re
from collections.abc import Iterable
from urllib.parse import unquote

# --------------------------------------------------------------
# Internal Context Key Prefixes
# --------------------------------------------------------------

INTERNAL_KEY_PREFIXES = ["_", "system_", "internal_", "__"]


# DECISION plan-2026-07-19T191147-4b664252/D-009 [STALE]
# This helper exists because the identical `any(key.startswith(p) ...)`
# expression was copy-pasted to FIVE call sites (context.py, fsm.py,
# definitions.py x2, prompts.py) and NONE of them case-folded -- so
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


# DECISION plan-2026-07-19T191147-4b664252/D-011 [STALE]
# One bound for BOTH recursive context filters (`context.clean_context_keys`
# and `prompts.BasePromptBuilder._filter_context_for_security`). Do NOT
# re-declare a local depth limit in either module: two hand-maintained copies
# of a security bound is the same duplication that produced F-13, and a filter
# that stops one level shallower than its sibling is a bypass. The behaviour AT
# the bound is fail-CLOSED in both: a container nested deeper is DROPPED, never
# passed through unfiltered (D-010). See decisions.md D-010, D-011.
MAX_CONTEXT_FILTER_DEPTH = 16


# --------------------------------------------------------------
# LLM Configuration Defaults
# --------------------------------------------------------------

DEFAULT_LLM_MODEL = "ollama_chat/qwen3.5:4b"
DEFAULT_TEMPERATURE = 0.5

# --------------------------------------------------------------
# Conversation Management Constants
# --------------------------------------------------------------

# History management defaults
DEFAULT_MAX_HISTORY_SIZE = 5
DEFAULT_MAX_MESSAGE_LENGTH = 1000

# Message processing constants
MESSAGE_TRUNCATION_SUFFIX = "... [truncated]"

# FSM stacking depth limit
DEFAULT_MAX_STACK_DEPTH = 10

# --------------------------------------------------------------
# LLM Interface Constants
# --------------------------------------------------------------

# Log preview lengths
LOG_MESSAGE_PREVIEW_LENGTH = 100
LOG_RESPONSE_PREVIEW_LENGTH = 200

# FSM ID generation
FSM_ID_HASH_LENGTH = 8

# --------------------------------------------------------------
# Transition Evaluation Constants
# --------------------------------------------------------------

# Transition evaluation internals
PRIORITY_SCALING_DIVISOR = 1000.0
MIN_BASE_CONFIDENCE = 0.1
CONDITION_SUCCESS_RATE_BOOST = 0.5
FLOAT_EQUALITY_EPSILON = 1e-9

# Classification-aware transition defaults
DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE = 0.6
TRANSITION_CLASSIFICATION_FALLBACK_INTENT = "_stay_in_state"
CONTEXT_KEY_CLASSIFICATION_RESULT = "_transition_classification_result"

# Classification extraction defaults
CLASSIFICATION_EXTRACTION_RESULT_SUFFIX = "_classification"

# --------------------------------------------------------------
# Environment Variable Keys
# --------------------------------------------------------------

ENV_LLM_MODEL = "LLM_MODEL"
ENV_LLM_TEMPERATURE = "LLM_TEMPERATURE"
ENV_LLM_MAX_TOKENS = "LLM_MAX_TOKENS"
ENV_FSM_PATH = "FSM_PATH"

# --------------------------------------------------------------
# Validation and Safety Constants
# --------------------------------------------------------------

# Security constants
ALLOWED_JSONLOGIC_OPERATIONS = {
    "==",
    "!=",
    "===",
    "!==",
    ">",
    ">=",
    "<",
    "<=",
    "and",
    "or",
    "!",
    "!!",
    "if",
    "var",
    "missing",
    "missing_some",
    "+",
    "-",
    "*",
    "/",
    "%",
    "min",
    "max",
    "cat",
    "in",
    "contains",
    "has_context",
    "context_length",
}

# DECISION plan-2026-07-19T191147-4b664252/D-009 [STALE]
# These patterns fail in BOTH directions, and both are real harm:
#   - Over-match STRIPS legitimate user data out of context and out of the
#     LLM prompt, silently degrading replies. The old password pattern's
#     trailing `(?:.*|$)` was VACUOUS (`.*` always matches), so every
#     "password"-containing flag -- `passwordless_login`,
#     `forgot_password_supported` -- was destroyed as collateral.
#   - Under-match LEAKS secrets into prompts. The plural forms `secrets`,
#     `access_tokens`, `private_keys`, `oauth_tokens` all passed through
#     untouched because only `credential(?:s)?` had been given the optional
#     plural.
# Every term therefore ends in an explicit boundary -- `[\W_]`, `$`, or an
# optional `(?:s)?`/digit-suffix -- and NEVER a bare `.*`. Do NOT "simplify"
# a trailing `(?:[\W_].*|$)` to `.*`: that re-creates the over-match, and
# `secretary`/`secretariat`/`access_tokenizer`/`private_keystone` are the
# pinned near-miss negatives that catch it.
# Any change here must be re-checked against the ADVERSARIAL negative set in
# tests/test_fsm_llm/test_context_unit.py, where each negative is maximally
# similar to a positive. A negative set of obviously-safe keys ("username",
# "email") validates whatever the implementation happens to do.
#
# DECISION plan-2026-07-19T191147-4b664252/D-016 [STALE] (SUPERSEDES the `password_hash`
# "accepted gap" that D-009 recorded here)
# D-009 claimed "no regex can separate `password_hash` from
# `password_reset_flow_enabled`" and therefore terminal-anchored the password
# pattern (`...password(?:s)?(?:[-_.]?\d+)?$`). That claim is FALSE and the
# anchoring under-matched a ~200-key class: EVERY `password_<non-numeric-suffix>`
# key was kept, and `password_hash`/`db_password_plaintext`/`user_password_salt`/
# `admin_password_encrypted` were measured reaching the LLM prompt through
# `prompts.BasePromptBuilder._filter_context_for_security`.
# The pattern below separates the two classes with a NEGATIVE LOOKAHEAD over a
# bounded allowlist of POLICY/STATUS suffix tokens. Read the direction
# carefully, because it is the whole point:
#   - suffix IS in the allowlist  -> KEPT  (`password_policy`, `passwordless_login`)
#   - suffix is NOT in the allowlist -> STRIPPED (`password_hash`, `password_pepper`)
# So an unrecognized suffix fails CLOSED. Do NOT invert this into a denylist of
# secret suffixes (`hash|salt|digest|...`) -- that is what the reviewer's own
# counter-example used, and it fails OPEN: `password_pepper`, `password_raw`,
# `password_hash2`, `password_argon2` would all be KEPT. In a security control
# the ambiguous case must be stripped; over-match costs a degraded LLM reply,
# under-match costs a credential disclosure.
# Only add a token here if it can NEVER name a credential VALUE. `hint`,
# `field`, `input`, `form` and `confirmation` were considered and DELIBERATELY
# excluded: `password_confirmation` is the re-typed password itself.
# `reset[-_.]?token` was added to the auth-token pattern below so that
# `password_reset_token` -- which this allowlist would otherwise keep via
# `reset` -- is still stripped.
# See decisions.md D-016.
#
# DECISION plan-2026-07-19T191147-4b664252/D-026 [STALE] (CORRECTS D-016's SHAPE, keeps
# its fail-CLOSED direction)
# D-016 got the DIRECTION right and the SHAPE wrong. Its lookahead tested only
# the token IMMEDIATELY following `password` and let the rest of the key run on
# unchecked, so the allowlist tokens behaved as unanchored PREFIXES with
# unbounded trailing content. One infix token defeated the whole control:
#   password_plaintext      -> STRIPPED   but password_last_plaintext -> KEPT
#   password_hash           -> STRIPPED   but password_reset_hash     -> KEPT
# Six such keys (`password_reset_code`, `password_reset_otp`,
# `password_reset_hash`, `password_last_plaintext`, `password_retrieval`,
# `password_policy_key`) were measured reaching the LLM prompt. D-016 found
# exactly ONE member of this class (`password_reset_token`) and closed it with a
# point fix in the auth-token pattern -- which is the same error as step 8:
# fixing a named key instead of the class.
# The lookahead below now matches the WHOLE remainder: a key is KEPT only when
# everything after `password` decomposes into allowlisted policy tokens, right
# up to `$`. Any unrecognized TRAILING token strips the key, no matter what
# preceded it. So `password_reset_flow_enabled` is KEPT (reset + flow + enabled
# are all policy tokens) while `password_reset_hash` is STRIPPED (`hash` is not).
# Two rules for editing this list, both load-bearing:
#  1. Tokens are COMPLETE WORDS, not prefixes. D-016 used `retr` and `polic` as
#     prefixes; `retr` then silently kept `password_retrieval` -- a secret --
#     because `retr` prefixes `retrieval` as happily as `retry`. Spell out
#     `retry|retries` and `policy|policies` instead. A prefix token re-opens the
#     bypass for every longer word it happens to prefix.
#  2. `[-_.]?` separators are OPTIONAL between tokens so `passwordless_login`
#     works, but the `+$` anchor is NOT optional -- dropping it restores the
#     unbounded-tail bypass exactly.
# See decisions.md D-026.
#
# DECISION plan-2026-07-19T191147-4b664252/D-030 [STALE] (bounds D-026's OVER-strictness;
# does NOT change its shape)
# D-026 closed the under-match and opened an over-match of the same class: the
# `+$` anchor requires EVERY trailing token to be allowlisted, so one ordinary
# English word strips the key. Measured against an independent-vocabulary corpus:
# 155 keys / 31 suffixes that BOTH `b00fade` and D-016 KEPT were newly stripped --
# `password_updated_at`, `password_changed_at`, `password_expires_at`,
# `password_policy_url`, `password_help_text`. These are timestamps and UI copy,
# not credentials. Fail-CLOSED, so a usability cost, not a leak.
# THE FIX IS DELIBERATELY PARTIAL, and the reason is the load-bearing part:
# because a key is KEPT when its WHOLE suffix decomposes into tokens, EVERY token
# added here is ALSO independently keepable as the ENTIRE suffix. Measured, one
# token at a time (probe `r3_token_safety.py`): adding `code` keeps
# `password_code` AND `password_reset_code` -- one of the six credential keys
# D-026 closed; `url` keeps `password_reset_url`, which embeds the reset token and
# so is credential-bearing by the SAME argument that excluded `link`; `text`,
# `value` and `string` keep `password_text`/`password_value`/`password_string`,
# i.e. the password itself. So `url`, `text`, `code`, `message`, `template`,
# `list`, `label`, `placeholder`, `value` and `string` are REFUSED, and keys
# ending in them stay stripped. That residual over-strip is DISCLOSED rather than
# fixed -- do not "complete" this group without re-running the probe.
# Only the temporal/ordinal/numeric group below is added, because those words
# cannot name a credential VALUE even standing alone as the whole suffix.
# See decisions.md D-030.
_PASSWORD_POLICY_SUFFIXES = (
    "less|reset|policy|policies|strength|expiry|expires|expiration|expired"
    "|require|required|requirement|requirements|rule|rules|support|supported"
    "|enable|enabled|disable|disabled|change|changed|update|updated|last"
    "|attempt|attempts|retry|retries|length|min|max|minimum|maximum"
    "|complexity|valid|validation|mismatch|manager|strategy|age|status|count"
    "|error|errors|help|instruction|instructions|setup|setting|settings"
    "|flow|login|day|days|meter|screen|page|banner|notice|reminder"
    # Notification-status tokens (`password_reset_email_sent`). `email` here
    # means an email ADDRESS or a sent-notification flag, never a credential.
    # `link` is deliberately NOT in this group: a `password_reset_link` embeds
    # the reset token, so it is a credential-bearing value and must strip.
    "|email|mail|sent|notification|notified|warning|enforced"
    # D-030 temporal/ordinal/numeric METADATA group. Every word here is safe as a
    # WHOLE suffix on its own (`password_at`, `password_score` cannot name a
    # credential), which is the test each candidate had to pass. This group is what
    # keeps `password_updated_at`, `password_expires_at`, `password_last_rotated`,
    # `password_strength_score` and `password_policy_version` reaching the prompt.
    "|at|on|in|date|time|today|after|since|until|rotated|rotation"
    "|score|version|len"
    # D-030 status group, added by the SAME rule that already admits
    # `enabled`/`disabled`/`status`/`changed`: these words denote an outcome or a
    # lifecycle event, never a value. `recovery` was probed and REFUSED --
    # `password_recovery` plausibly names a recovery code or security answer.
    "|fail|failed|failure|failures|complete|completed|create|created"
)

# DECISION plan-2026-07-20T040150-876e7164/D-015 [STALE]
# (CORRECTS D-014, which is superseded for the `key` trigger and AMENDED for the
#  `token` trigger. D-014's own in-code text made a class claim that was false
#  for the code it annotated; that text is replaced here rather than preserved.)
#
# WHAT D-014 GOT RIGHT. Before it, both triggers said "strip only when the
# qualifier is on this list", so every qualifier NOT on the list FAILED OPEN:
# 46 of 52 real crypto-key names leaked (`ssh_key`, `deploy_key`, `rsa_key`,
# `keypair`, `id_rsa`, ...) and 27 of 27 real auth-token names leaked
# (`csrf_token`, `session_token`, `jwt_token`, ...). That had to change.
#
# WHAT D-014 GOT WRONG, AND WHY IT IS FIXED HERE. Two defects, both found by
# adversarial review of the shipped code:
#
#   (1) THE SCAN WAS NOT ANCHORED. D-014's negative lookaheads inspected only
#       the qualifier IMMEDIATELY before the trigger, so prepending anything to
#       an allowlisted qualifier defeated the strip: `csrf_token` stripped but
#       `csrf_max_token` was KEPT; `ssh_key` stripped but `ssh_cache_key` was
#       KEPT; `private_key` stripped but `private_public_key` was KEPT. The
#       "whole decomposition" claim was therefore false. Fixed for `token` by
#       anchoring the lookahead to the START of the name (`(?:qual[-_.]?)+`, so
#       EVERY qualifier before the trigger must be allowlisted).
#
#   (2) THE POLARITY WAS WRONG FOR `key`. The two triggers have OPPOSITE
#       vocabulary structure and D-014 shipped the same polarity for both:
#         `token` -- the SAFE space is small and closed (LLM metering, NLP
#                    special tokens, pagination cursors) and the DANGEROUS space
#                    is open (any `<service>_token` bearer credential), so a
#                    fail-closed ALLOWLIST is the right shape. KEPT.
#         `key`   -- INVERTED. The SAFE space is unbounded (`s3_key`,
#                    `user_key`, `order_key`, `config_key`, `context_key`,
#                    `item_key`, any `<noun>_key`) while the DANGEROUS space is
#                    a finite, well-known crypto vocabulary. A fail-closed
#                    allowlist over an unbounded safe space produced a MEASURED
#                    42% over-strip against an independently-sourced corpus
#                    (31 names, all newly regressed by D-014) -- see
#                    findings/review-iter-1.md. So `key` is now a crypto
#                    qualifier DENYLIST.
#
# WHAT THIS IS, STATED PRECISELY -- NO CLASS CLAIM FOR `key`.
#   `key`   : an ENUMERATED DENYLIST of crypto vocabulary. It is NOT a class
#             control. It FAILS OPEN on any crypto qualifier nobody enumerated
#             (`foo_key` for some future algorithm `foo` reaches the prompt).
#             That is a real, permanent cost and it is exactly the failure shape
#             LESSONS [I:5] warns about; it is accepted here only because the
#             alternative -- fail-closed over an unbounded safe space -- was
#             measured and destroyed 42% of ordinary application vocabulary on
#             the live prompt path. The four STRONG triggers (`secret`,
#             `credential`, `password`, and `private` inside this list) remain
#             an INDEPENDENT backstop layer above it.
#   `token` : a fail-CLOSED, START-ANCHORED ALLOWLIST for the `<qual>_token`
#             suffix shape. Unrecognised qualifiers strip. This half IS a class
#             control and the claim is safe to make for it.
#             The `token_<head>` PREFIX shape is a denylist of value-bearing
#             heads, for the same reason `key` is: `token_count`,
#             `token_max_length`, `token_per_second`, `tokenizer_config` are
#             unbounded ordinary vocabulary.
#
# DO NOT convert the `token` SUFFIX allowlist into a denylist. That is what
# leaked 27/27, and bearer-token qualifiers are open-ended.
# DO NOT convert the `key` denylist back into an allowlist without re-running
# the independent-corpus measurement in
# tests/test_fsm_llm/test_context_unit.py::TestCryptoKeyAndTokenTriggers; that
# is the measurement D-014 skipped (its corpus was a 1:1 restatement of its own
# allowlist, 36/36 and 21/21, so it measured nothing).
# DO NOT relax `[a-z0-9]+` to `[a-z0-9]*` in the token suffix arm. G-14: with
# zero-or-more, the `.*[\W_]` skip-prefix consumes the safe qualifier and the
# qualifier then matches EMPTY against a bare trigger, silently defeating the
# lookahead for EVERY allowlisted word.
# DO NOT replace `_CRYPTO_GAP`'s bounded `{0,64}?` with `*?`. The gap sits
# inside an `(?:^|.*[\W_])` scan, so an unbounded lazy gap makes the pattern
# QUADRATIC on inputs like `"ssh_"*n` -- a DoS on the prompt path, where this
# runs with the per-conversation lock held. The bound is what keeps it linear
# (measured: per-character time varies 1.63x over 7 doublings).
#
# KNOWN AND ACCEPTED GAPS, enumerated rather than papered over:
#   - camelCase multi-word (`sshPublicKey`) is NOT reached: the gap scan needs
#     the crypto word to end at a separator or directly abut `key`.
#   - a crypto word that is not at a word boundary (`mysshkey`) is not reached.
#   - `kek`, `dek`, `id_rsa`, `id_dsa`, `id_ecdsa`, `id_ed25519` contain no
#     "key" substring at all and get their own explicit entry below.
#   - `product_key` is deliberately KEPT: it is genuinely ambiguous (Windows
#     product key vs. a product table's key) and the independent corpus counts
#     it as ordinary. `license_key`, `activation_key` and `serial_key` remain
#     stripped as the unambiguous license-credential spellings.
#   - `public` is NOT in the crypto denylist (a public key is not a secret;
#     invariant I-7, pinned by tests/test_fsm_llm_regression/test_regression_iter2.py).
#     `ssh_public_key` still strips, because `ssh` is.
# See decisions.md D-015.

# Consume a trailing plural `s` when -- and only when -- it is a plural. The
# alternation is NOT a stylistic `s?`: `s?` would let the engine backtrack to the
# no-`s` parse and re-match, so `tokens_used` would decompose as `token` + `s`
# + `_used` and strip. The second branch asserts there is no consumable `s`,
# which forecloses that parse while still letting `keystone` decompose as
# `key` + `stone` (the `s` there is followed by an alphanumeric, so it is part
# of the next word rather than a plural).
_TRIGGER_PLURAL = r"(?:s(?![a-z0-9])|(?!s(?![a-z0-9])))"
_KEY_TRIGGER = rf"key{_TRIGGER_PLURAL}"
_TOKEN_TRIGGER = rf"token{_TRIGGER_PLURAL}"

# Word boundary. Deliberately `(?:[\W_]|$)` and NOT D-026's `(?:[\W_].*|$)`:
# these patterns are applied with `.match()`, which never requires reaching the
# end of the string, so the trailing `.*` is pure backtracking surface. Dropping
# it removes an O(n) scan from inside two negative lookaheads that are themselves
# retried at every separator position.
_WORD_END = r"(?:[\W_]|$)"

# DECISION plan-2026-07-20T040150-876e7164/D-019 [STALE]
# A separator RUN, not a single optional separator. The token suffix arm used
# `[-_.]?`, which permits at most ONE separator between the qualifier and the
# trigger, so a second separator left `[a-z0-9]+` with nothing to match and the
# fail-CLOSED arm silently KEPT `csrf__token`, `session__token`, `bearer__token`,
# `jwt__token`, `api__token`, `csrf--token` and `x__csrf__token`. That is a
# bypass of the one half D-015 stakes a class claim on, and the `key` arm was
# never vulnerable because its bounded gap already accepts separators.
# The run is BOUNDED (`{1,4}` after a mandatory first separator is folded in as
# `{0,4}` here) for the same ReDoS reason the crypto gap is bounded -- an
# unbounded `[-_.]*` next to `[a-z0-9]+` inside an `(?:^|.*[\W_])` scan is
# gratuitous backtracking surface. Both the strip alternative AND the
# allowlist lookahead use it, so `max__tokens` is still KEPT.
# Do NOT revert this to `[-_.]?`. See decisions.md D-019.
_SEP_RUN = r"[-_.]{0,4}"

# STRIP-list (DENYLIST -- see the polarity paragraph in D-015 above) of crypto
# vocabulary. A `*key*` name strips when ANY of these words appears at a word
# boundary anywhere before the trigger, so `ssh_key`, `ssh_cache_key` and
# `ssh_public_key` all strip while `s3_key`, `cache_key` and `public_key` are
# kept. Grouped by source vocabulary so that additions land in the right group
# and the coverage gaps are visible.
_CRYPTO_KEY_QUALIFIERS = (
    # asymmetric algorithms and their file-name spellings
    "rsa|dsa|ecdsa|eddsa|ed25519|x25519|curve25519|ecc|elliptic"
    # symmetric primitives, MACs and password-hashing KDFs
    "|aes|des|3des|blowfish|chacha|chacha20|salsa20|rc4|idea|cipher|ciphers"
    "|hmac|mac|cmac|poly1305|kdf|hkdf|pbkdf2|scrypt|bcrypt|argon2"
    # transport security and PKI container formats
    "|tls|ssl|dtls|mtls|x509|pem|der|pkcs[0-9]*|pfx|p12|jks|csr"
    "|cert|certs|certificate|certificates|ca|cacert|truststore"
    # SSH, PGP and host identity
    "|ssh|sshd|openssh|gpg|pgp|openpgp|host|hosts|known"
    # the ROLE a key plays -- this group, not the algorithm group, is what
    # catches the majority of real names
    "|private|priv|secret|master|signing|sign|signature|verification"
    "|encryption|encrypt|encrypted|decryption|decrypt|symmetric|asymmetric"
    "|shared|derive|derived|derivation|wrap|wrapping|unwrap|unwrapping"
    "|kek|dek|envelope|seal|unseal|sealing"
    # credential-adjacent roles (these also have independent strong triggers,
    # which is deliberate -- defence in depth, not redundancy)
    "|api|access|auth|oauth|jwt|jwk|jwks|session|bearer|credential"
    "|deploy|deployment|authorized|authorised|root|admin|server|client"
    # secret-management platforms and key stores
    "|kms|hsm|vault|keystore|keyvault|managed|cmk|byok"
    # seeds, salts and account-recovery material
    "|seed|salt|entropy|nonce|mnemonic|passphrase|recovery|restore|backup"
    "|unlock|activation|license|licence|serial|subscription"
    # framework config spellings whose VALUE is key material
    "|rails|django|secretkeybase|webhook"
)

# STRIP-list of heads sitting immediately AFTER a leading `key`. Position is
# load-bearing and the asymmetry is intentional: `key_blob` is a blob OF a key
# (strip) while `blob_key` is a key OF a blob (keep); `keyfile` is a file
# holding key material (strip) while `file_key` identifies a file (keep).
# Anything not on this list keeps, so `keyword`, `keyboard_layout`,
# `keystone_species`, `key_value_pair` and `key_performance_indicator` need no
# allowlist of their own -- which is why D-014's `_SAFE_KEY_WORDS` and
# `_SAFE_KEY_HEADS` are gone rather than moved.
# DECISION plan-2026-07-20T040150-876e7164/D-019 [STALE]
# The two head lists are kept at PARITY deliberately. They drifted apart once
# already: `_TOKEN_MATERIAL_HEADS` omitted `material|content|body|payload|bytes|
# data` which its stated mirror `_KEY_MATERIAL_HEADS` carried, and omitted the
# raw-material and grant-type heads entirely, so `token_raw`, `token_plaintext`,
# `token_material`, `token_access`, `token_refresh` and `token_pem` reached the
# prompt while the `key_` equivalents did not. A head that names credential
# material under `key_` names credential material under `token_` too. If you add
# a head to either list, add it to BOTH or record why it is asymmetric.
# See decisions.md D-019.
_CREDENTIAL_MATERIAL_HEADS_SHARED = (
    "raw|plaintext|cleartext|material|content|contents|body|payload|data"
    "|blob|blobs|bytes|byte|b64|base64|hex|digest|str|string|strings|text"
    "|secret|secrets|hash|hashes|pem|der|cipher|encrypted"
    "|access|refresh|auth|session|response|jwt|bearer|credential|credentials"
)

# The ONE recorded asymmetry the parity note above demands. `value|values` is a
# token head but NOT a key head: `token_value` names the bearer string itself,
# while `key_value_pair`, `key_value_store` and `key_value_map` are the
# pervasive ordinary data-structure idiom and D-015 pins `key_value_pair` as
# KEPT. There is no `token_value_pair` idiom to protect on the other side.
_TOKEN_ONLY_MATERIAL_HEADS = "value|values"

# The container/handle heads are `key`-specific (a `key_file` holds key material;
# there is no `token_file` idiom worth the over-strip); the material heads are
# SHARED with `_TOKEN_MATERIAL_HEADS` -- see the parity note there.
_KEY_MATERIAL_HEADS = (
    "pair|pairs|file|files|store|stores|storage|ring|rings|chain|chains"
    "|seed|salt|phrase|passphrase|fingerprint|thumbprint|vault|vaults"
    "|holder|slot|private|rsa|ssh|" + _CREDENTIAL_MATERIAL_HEADS_SHARED
)

# DECISION plan-2026-07-20T040150-876e7164/D-019 [STALE]
# Bounded lazy gap between a crypto qualifier and the trigger, so that EVERY
# intervening word is skipped rather than only the last one (the D-014 defect).
#
# THE BOUND IS TWO THINGS AT ONCE AND BOTH MUST STAY DISCLOSED. It is a ReDoS
# control (the gap sits inside an `(?:^|.*[\W_])` scan, so an unbounded lazy gap
# makes the pattern QUADRATIC on `"ssh_"*n` -- a DoS on the prompt path, where
# this runs with the per-conversation lock held). It is ALSO a hard limit on the
# control's REACH: a crypto word further than this many characters before the
# trigger is NOT seen, and the name reaches the prompt. D-015 documented only the
# first role, which let an exact, reproducible bypass cliff ship undisclosed at
# 65 characters -- adversarial review demonstrated `ssh_` + `"a"*64` + `_key`
# STRIPS while `+65` KEEPS.
#
# The bound is raised 64 -> 192 because tenant/region/cluster-qualified names
# (`ssh_prod_us_east_1_cluster_07_replica_shard_0003_rotation_2026_q3_key`)
# routinely exceed 64 characters between the crypto word and the trigger. 192 was
# re-timed for linearity, not guessed. The cliff still EXISTS at 193 -- it is
# moved, not removed -- and it is pinned two-sided by
# `test_the_crypto_gap_reach_cliff_is_pinned_two_sided` so that a future editor
# changing this number sees what it costs. Do NOT replace it with `*?`.
# See decisions.md D-019.
_CRYPTO_GAP_REACH = 192
_CRYPTO_GAP = rf"[a-z0-9_.\-]{{0,{_CRYPTO_GAP_REACH}}}?"

# DECISION plan-2026-07-20T040150-876e7164/D-019 [STALE]
# The crypto word must either end at a word boundary (`ssh_...`) or abut the
# trigger directly (`sshkey`). Without this, the gap would let a crypto word
# match as a mere PREFIX of an unrelated word (`secretary_monkey`).
#
# The `[0-9]*` is LOAD-BEARING, not decoration. Without it a digit immediately
# after the crypto word defeated the boundary test, so the standard
# algorithm-plus-bit-size naming convention -- `aes256_key`, `aes128_key`,
# `rsa2048_key`, `rsa4096_key`, `hmac256_key` -- KEPT while the separated
# spellings `aes_256_key` and `chacha20_key` stripped. Those are unambiguous
# credential names and the difference between them was a missing separator.
# Do NOT remove the `[0-9]*`. See decisions.md D-019.
_CRYPTO_AT_WORD = rf"(?:{_CRYPTO_KEY_QUALIFIERS})[0-9]*(?:(?![a-z0-9])|(?=key))"

# KEEP-list for the qualifiers before `token`. THIS ONE IS AN ALLOWLIST and it
# is START-ANCHORED in the pattern below: every qualifier between the start of
# the name and the trigger must be on this list, so `csrf_max_token` strips even
# though `max` is allowlisted. Three crisply-ruled groups and no others:
# (1) METERING DIMENSIONS -- this is an LLM framework, so `max_tokens`/
# `prompt_tokens`/`completion_tokens`/`cached_tokens` are pervasive, ordinary
# and non-secret, and stripping them is the tightest collateral constraint in
# this control; (2) NLP SPECIAL TOKENS (`bos_token`, `eos_token`, `pad_token`,
# `mask_token` -- HuggingFace tokenizer vocabulary, a token here is a symbol,
# not a credential); (3) PAGINATION CURSORS (`nextPageToken`, `NextToken`,
# `continuationToken`, `syncToken`), opaque list-API offsets. Every bearer-style
# qualifier -- `csrf`, `session`, `jwt`, `id`, `device`, `magic`, `invite`,
# `activation`, `recovery`, `access`, `refresh`, `bearer`, `auth`, `api`,
# `reset` -- is absent and therefore strips.
_SAFE_TOKEN_QUALIFIERS = (
    "max|min|prompt|completion|input|output|total|count|usage|used|limit"
    "|budget|remaining|average|avg|estimated|estimate|cached|reasoning"
    "|audio|text|image|per|page|next|continuation|pagination|sync"
    "|bos|eos|eot|sos|pad|unk|mask|sep|cls|special|delimiter|stop"
    # DECISION plan-2026-07-20T040150-876e7164/D-021 [STALE] -- the cursor group is
    # extended to the other published spellings of the same concept
    # (Elasticsearch `scroll_id`, MongoDB change-stream `resumeToken`, AWS
    # `Marker`/`NextMarker`, Azure `continuationToken`, Relay `cursor`, stream
    # `checkpoint`/`watermark`). This group is HERE and not left to the value
    # layer for a measured reason: a pagination cursor's value is an opaque
    # high-entropy base64 blob that is CATEGORICALLY INDISTINGUISHABLE from a
    # bearer token by shape -- the pre-design probe measured 4 of 6 real
    # cursors as credential-shaped. The allowlist now holds exactly the names
    # whose safe values the value layer provably cannot judge, which is what
    # an allowlist is for. Do NOT delete these expecting layer 2 to cover them.
    "|cursor|scroll|resume|marker|watermark|checkpoint"
)

# DECISION plan-2026-07-20T040150-876e7164/D-021 [STALE]
# STRIP-list (DENYLIST) of unambiguous bearer-credential qualifiers, ADJACENT to
# the trigger. This is the token arm's counterpart to `_CRYPTO_KEY_QUALIFIERS`
# and it exists for exactly the reason that one does: the value layer below
# cannot see a credential that is SHORT or LOW-ENTROPY (`csrf_token: "Wm9wOTQx"`,
# `session_token: "sess-0001"`, `access_token: "expired"`, `otp_token: "483920"`),
# so the NAME has to carry those. It is a DENYLIST and it FAILS OPEN on bearer
# vocabulary nobody listed; it is NOT a class control and must not be described
# as one -- the class claim on this arm rests on the value layer, exactly as
# D-019 established for `key`.
#
# ADJACENCY IS DELIBERATE and it is a REVERSAL of the R1 fix, which used an
# unbounded qualifier scan so that `csrf_max_token` stripped. The unbounded scan
# also strips `session_max_tokens`, `api_total_tokens` and `auth_budget_tokens`
# -- plausible per-scope metering fields in an LLM framework, which is the
# tightest collateral constraint on this control. Infixed bearer names are NOT
# abandoned: they fall through to the value layer below, which strips them
# whenever they actually hold a credential (always high-entropy, by definition).
# The residual is the conjunction of an infixed bearer name AND a short or
# low-entropy value, which is disclosed rather than claimed closed.
#
# Vocabulary is RFC/framework-sourced, not corpus-sourced: RFC 6749 (`access`,
# `refresh`, `bearer`, `authorization`), RFC 7519 / OIDC (`jwt`, `id`),
# Django/Rails/ASP.NET CSRF (`csrf`, `xsrf`, `authenticity`, `antiforgery`),
# FCM/APNs push registration, SAML/SSO, magic-link and MFA vocabulary. Words
# that appear in this plan's held-out credential slice were deliberately
# WITHHELD from this list so that the value layer, not the name, has to catch
# them -- see decisions.md D-021.
_BEARER_TOKEN_QUALIFIERS = (
    "csrf|xsrf|antiforgery|authenticity|jwt|jwe|jws|oauth|oidc|saml|sso"
    "|access|refresh|bearer|auth|authorization|authentication|api|id|identity"
    "|session|magic|login|signin|signup|registration|activation|confirmation"
    "|verification|reset|recovery|invite|invitation|impersonation|consent"
    "|device|push|fcm|apns|sas|nonce|challenge|otp|totp|hotp|mfa"
    "|secret|private|credential"
)

# STRIP-list of heads immediately AFTER a leading `token`, mirroring
# `_KEY_MATERIAL_HEADS` and for the same reason: `token_<head>` has an unbounded
# ordinary tail (`token_count`, `token_max_length`, `token_per_second`,
# `token_ids`, `tokenizer_config`) and a small dangerous one. Only the heads
# that name the VALUE strip.
#
_TOKEN_MATERIAL_HEADS = (
    _CREDENTIAL_MATERIAL_HEADS_SHARED + "|" + _TOKEN_ONLY_MATERIAL_HEADS
)

FORBIDDEN_CONTEXT_PATTERNS = [
    # Password keys. Strips password/passwords/password123/user_password AND
    # every `password_<suffix>` whose suffix does not decompose ENTIRELY into
    # policy/status tokens. See the D-026 block above before editing.
    rf"(?:^|.*[\W_])passwords?(?!(?:[-_.]?(?:{_PASSWORD_POLICY_SUFFIXES}))+$)",
    r"(?:^|.*[\W_])secret(?:s)?(?:[\W_].*|$)",  # Secret-related keys (not "secretary")
    r".*(?:api[-_.]?key|key[-_.]?api).*",  # API key patterns (both orderings, with dash/underscore/dot)
    r"(?:^|.*[\W_])credential(?:s)?(?:[\W_].*|$)",  # Credential-related keys
    # DECISION plan-2026-07-20T040150-876e7164/D-015 [STALE] -- cryptographic key
    # material. This is an ENUMERATED CRYPTO DENYLIST and it FAILS OPEN on
    # vocabulary nobody listed; it is NOT a class control and must not be
    # described as one. Read the D-015 block above before editing.
    #   1. the bare article: `key`, `keys`
    #   2. a crypto qualifier at a word boundary ANYWHERE before the trigger:
    #      strips `ssh_key`, `sshkey`, `ssh_cache_key`, `ssh_public_key`,
    #      `private_public_key`; keeps `s3_key`, `cache_key`, `public_key`,
    #      `user_key`, `monkey_species`
    #   3. key-then-material-head: strips `keyfile`, `key_pair`, `keystore`,
    #      `key_material`; keeps `keyword`, `keyboard_layout`, `key_value_pair`,
    #      `key_performance_indicator`
    rf"(?:{_KEY_TRIGGER}$)"
    rf"|(?:^|.*[\W_]){_CRYPTO_AT_WORD}{_CRYPTO_GAP}{_KEY_TRIGGER}{_WORD_END}"
    rf"|(?:^|.*[\W_]){_KEY_TRIGGER}[-_.]?"
    rf"(?:{_KEY_MATERIAL_HEADS})s?{_WORD_END}",
    # DECISION plan-2026-07-20T040150-876e7164/D-015 [STALE] -- the ENUMERATED residual.
    # These name key material but contain no "key" substring, so the pattern
    # above cannot structurally reach them. This entry is a DENYLIST and is
    # therefore incomplete by construction; it is disclosed as a list, never
    # claimed as a class. Do not read it as evidence that the control above is
    # list-shaped -- it is the explicit admission that one narrow corner is.
    r"(?:^|.*[\W_])(?:kek|dek|id[-_.]?rsa|id[-_.]?dsa|id[-_.]?ecdsa"
    r"|id[-_.]?ed25519)(?:[\W_]|$)",
    # DECISION plan-2026-07-20T040150-876e7164/D-021 [STALE] -- auth tokens. This entry
    # is NAME-AUTHORITATIVE and it is a DENYLIST on both arms, mirroring the
    # `key` entry above. It replaces D-015's fail-CLOSED allowlist arm, which
    # D-019 finding (2) measured as an allowlist over an UNBOUNDED safe space:
    # 20 independently-sourced safe `*_token` names were 20/20 outside it and
    # 20/20 over-stripped, and widening it to fit drove combined over-strip to
    # 23.2%, past SC-5. The names it used to strip unconditionally are now
    # referred to the VALUE layer via `_TOKEN_VALUE_SCAN_NAME_RE` below.
    #   1. the bare article: `token`, `tokens`
    #   2. a bearer qualifier ADJACENT to the trigger: strips `csrf_token`,
    #      `csrftoken`, `x_csrf_token`, `csrf__token`, `session_token`,
    #      `jwt_token`, `id_token`, `api_token`; keeps `session_max_tokens`
    #      and `csrf_max_token` FOR THE NAME LAYER (both are then decided by
    #      the value layer)
    #   3. token-then-material-head: strips `token_value`, `token_secret`,
    #      `token_raw`; keeps `token_count`, `token_ids`, `tokenizer_config`,
    #      `token_max_length`
    # The `oauth[-_.]?token` entry that follows is retained deliberately as an
    # independently-pinned explicit statement, not because it is load-bearing.
    rf"(?:{_TOKEN_TRIGGER}$)"
    rf"|(?:^|.*[\W_])(?:{_BEARER_TOKEN_QUALIFIERS}){_SEP_RUN}"
    rf"{_TOKEN_TRIGGER}{_WORD_END}"
    rf"|(?:^|.*[\W_]){_TOKEN_TRIGGER}[-_.]?"
    rf"(?:{_TOKEN_MATERIAL_HEADS})s?{_WORD_END}",
    r"(?:^|.*[\W_])oauth[-_.]?token(?:s)?(?:[\W_].*|$)",  # OAuth token patterns
]

# Pre-compiled versions for performance (avoid recompiling in loops)
COMPILED_FORBIDDEN_CONTEXT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in FORBIDDEN_CONTEXT_PATTERNS
]


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
# the whitespace rejection below.
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
# Splitting CANNOT repair it: the read site below is `any(token in ...)`, a
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
# see `_is_path_shaped` and the D-005 block below for why an extension is not
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
# test at its single read site below -- `re.compile(r"[A-Za-z][A-Za-z0-9\-]{0,19}")`
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

# DECISION plan-2026-07-20T040150-876e7164/D-021 [STALE]
# COLON-COMPOSITE CREDENTIALS. `:` is deliberately OUTSIDE the charset above,
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
# Do NOT "simplify" this by adding `:` to the charset above -- that strips
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
    # `_UUID_VALUE_RE`.
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
    rf"(?!(?:(?:{_SAFE_TOKEN_QUALIFIERS}){_SEP_RUN})+{_TOKEN_TRIGGER}{_WORD_END})"
    rf"(?:^|.*[\W_])[a-z0-9]+{_SEP_RUN}{_TOKEN_TRIGGER}{_WORD_END}",
    re.IGNORECASE,
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


def is_forbidden_context_entry(key: object, value: object = None) -> bool:
    """Return True if this context ENTRY must never reach an LLM prompt.

    The single decision point for the context-key security filter, shared by
    ``context.clean_context_keys``, ``prompts.BasePromptBuilder``'s
    ``_is_forbidden_context_key`` and ``runner``'s log redaction. It combines:

    * **Layer 1 (name)** -- ``COMPILED_FORBIDDEN_CONTEXT_PATTERNS``. Decides on
      the name alone and is authoritative in the STRIP direction.
    * **Layer 2 (value)** -- consulted for the two shapes layer 1 leaves open,
      with OPPOSITE defaults because their safe spaces differ:

      - the ambiguous ``<qualifier>_key`` shape plus
        ``_AMBIGUOUS_CREDENTIAL_ABBREVIATIONS`` go to
        :func:`_looks_like_credential_value` and default to KEEP;
      - the non-allowlisted ``<qualifier>_token`` shape goes to
        :func:`_token_value_is_credential` and defaults to STRIP.

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
    # layer-1 regexes above already do. `lowered_key` is then an exact `str`, so
    # the frozenset lookup cannot reach a subclass `__eq__` either.
    lowered_key = str.lower(key)
    if not (
        lowered_key in _AMBIGUOUS_CREDENTIAL_ABBREVIATIONS
        or _VALUE_SCAN_NAME_RE.match(key)
    ):
        return False

    return _looks_like_credential_value(value, key)


# --------------------------------------------------------------
# Logging Configuration
# --------------------------------------------------------------

ENV_LOG_LEVEL = "FSM_LLM_LOG_LEVEL"
ENV_LOG_FORMAT = "FSM_LLM_LOG_FORMAT"

LOG_FORMAT_HUMAN = "human"
LOG_FORMAT_JSON = "json"

LOG_SINK_STDERR = "stderr"
LOG_SINK_STDOUT = "stdout"
LOG_SINK_FILE = "file"

LOG_DEFAULT_LEVEL = "DEBUG"
LOG_DEFAULT_ROTATION = "10 MB"
LOG_DEFAULT_RETENTION = "1 month"
LOG_DEFAULT_COMPRESSION = "zip"
LOG_DEFAULT_FILE_PATTERN = "fsm-llm_{time}.log"

LOG_HUMAN_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}:{function}:{line}</cyan> | "
    "{message}"
)

LOG_HUMAN_FORMAT_WITH_CONTEXT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "conv:<yellow>{extra[conversation_id]:<12}</yellow> | "
    "<cyan>{name}:{function}:{line}</cyan> | "
    "{message}"
)

LOG_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | "
    "{level: <8} | "
    "conv_id: {extra[conversation_id]:<12} | "
    "{name}:{function}:{line} | "
    "{message}"
)

# Standard context field names for structured logging
LOG_FIELD_TIMESTAMP = "timestamp"
LOG_FIELD_LEVEL = "level"
LOG_FIELD_MESSAGE = "message"
LOG_FIELD_MODULE = "module"
LOG_FIELD_FUNCTION = "function"
LOG_FIELD_LINE = "line"
LOG_FIELD_CONVERSATION_ID = "conversation_id"
LOG_FIELD_PACKAGE = "package"

# Default value for unbound context fields
LOG_DEFAULT_CONVERSATION_ID = "GENERAL"

# --------------------------------------------------------------
# Timeout Defaults
# --------------------------------------------------------------

# Handler execution timeout (seconds). None = no timeout.
# Recommended: set handler_timeout=DEFAULT_HANDLER_TIMEOUT when creating
# HandlerSystem for safety against handlers that block indefinitely.
DEFAULT_HANDLER_TIMEOUT = 30.0

# Workflow step execution timeout (seconds). None = no timeout.
DEFAULT_STEP_TIMEOUT = 120.0
