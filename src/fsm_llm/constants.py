from __future__ import annotations

"""
Constants and configuration values for the FSM-LLM framework.
"""

import re
from collections.abc import Iterable

# --------------------------------------------------------------
# Internal Context Key Prefixes
# --------------------------------------------------------------

INTERNAL_KEY_PREFIXES = ["_", "system_", "internal_", "__"]


# DECISION plan-2026-07-19T191147-4b664252/D-009
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
        from user-visible data and from LLM prompts). Never raises.
    """
    if prefixes is None:
        prefixes = INTERNAL_KEY_PREFIXES
    lowered = key.lower()
    return any(lowered.startswith(prefix.lower()) for prefix in prefixes)


# DECISION plan-2026-07-19T191147-4b664252/D-011
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

# DECISION plan-2026-07-19T191147-4b664252/D-009
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
# DECISION plan-2026-07-19T191147-4b664252/D-016 (SUPERSEDES the `password_hash`
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
# DECISION plan-2026-07-19T191147-4b664252/D-026 (CORRECTS D-016's SHAPE, keeps
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
# DECISION plan-2026-07-19T191147-4b664252/D-030 (bounds D-026's OVER-strictness;
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

# DECISION plan-2026-07-20T040150-876e7164/D-015
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
_KEY_MATERIAL_HEADS = (
    "pair|pairs|file|files|store|stores|storage|ring|rings|chain|chains"
    "|material|blob|blobs|bytes|byte|data|content|contents|body|payload"
    "|pem|der|b64|base64|hex|seed|salt|phrase|passphrase|fingerprint"
    "|thumbprint|digest|secret|secrets|vault|vaults|holder|slot"
)

# Bounded lazy gap between a crypto qualifier and the trigger, so that EVERY
# intervening word is skipped rather than only the last one (the D-014 defect).
# The `{0,64}` bound is a ReDoS control, NOT a style choice -- see D-015.
_CRYPTO_GAP = r"[a-z0-9_.\-]{0,64}?"

# The crypto word must either end at a word boundary (`ssh_...`) or abut the
# trigger directly (`sshkey`). Without this, the gap would let a crypto word
# match as a mere PREFIX of an unrelated word (`secretary_monkey`).
_CRYPTO_AT_WORD = rf"(?:{_CRYPTO_KEY_QUALIFIERS})(?:(?![a-z0-9])|(?=key))"

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
)

# STRIP-list of heads immediately AFTER a leading `token`, mirroring
# `_KEY_MATERIAL_HEADS` and for the same reason: `token_<head>` has an unbounded
# ordinary tail (`token_count`, `token_max_length`, `token_per_second`,
# `token_ids`, `tokenizer_config`) and a small dangerous one. Only the heads
# that name the VALUE strip.
_TOKEN_MATERIAL_HEADS = (
    "value|values|secret|secrets|string|strings|hash|hashes|data|blob|blobs"
    "|bytes|b64|base64|hex|digest|jwt|bearer|credential|credentials"
)

FORBIDDEN_CONTEXT_PATTERNS = [
    # Password keys. Strips password/passwords/password123/user_password AND
    # every `password_<suffix>` whose suffix does not decompose ENTIRELY into
    # policy/status tokens. See the D-026 block above before editing.
    rf"(?:^|.*[\W_])passwords?(?!(?:[-_.]?(?:{_PASSWORD_POLICY_SUFFIXES}))+$)",
    r"(?:^|.*[\W_])secret(?:s)?(?:[\W_].*|$)",  # Secret-related keys (not "secretary")
    r".*(?:api[-_.]?key|key[-_.]?api).*",  # API key patterns (both orderings, with dash/underscore/dot)
    r"(?:^|.*[\W_])credential(?:s)?(?:[\W_].*|$)",  # Credential-related keys
    # DECISION plan-2026-07-20T040150-876e7164/D-015 -- cryptographic key
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
    # DECISION plan-2026-07-20T040150-876e7164/D-015 -- the ENUMERATED residual.
    # These name key material but contain no "key" substring, so the pattern
    # above cannot structurally reach them. This entry is a DENYLIST and is
    # therefore incomplete by construction; it is disclosed as a list, never
    # claimed as a class. Do not read it as evidence that the control above is
    # list-shaped -- it is the explicit admission that one narrow corner is.
    r"(?:^|.*[\W_])(?:kek|dek|id[-_.]?rsa|id[-_.]?dsa|id[-_.]?ecdsa"
    r"|id[-_.]?ed25519)(?:[\W_]|$)",
    # DECISION plan-2026-07-20T040150-876e7164/D-015 -- auth tokens. The SUFFIX
    # arm is a fail-CLOSED allowlist and IS a class control; the negative
    # lookahead is anchored at the START of the name, so every qualifier before
    # the trigger must be allowlisted -- `csrf_max_token` strips even though
    # `max` is on the list. The PREFIX arm is a value-head denylist, for the
    # same unbounded-safe-tail reason as `key`.
    # Strips `csrf_token`, `csrftoken`, `session_token`, `jwt_token`,
    # `id_token`, `csrf_max_token`, `bearer_cached_token`, `token_value`;
    # keeps `max_tokens`, `prompt_tokens`, `token_count`, `tokenizer_config`,
    # `token_max_length`, `bos_token`, `next_page_token`.
    # This SUBSUMES the six-word `(api|auth|access|refresh|bearer|reset)_token`
    # list it replaces, and also the `oauth[-_.]?token` entry that follows it --
    # that entry is retained deliberately as an independently-pinned explicit
    # statement, not because it is still load-bearing.
    rf"(?:{_TOKEN_TRIGGER}$)"
    rf"|(?!(?:(?:{_SAFE_TOKEN_QUALIFIERS})[-_.]?)+{_TOKEN_TRIGGER}{_WORD_END})"
    rf"(?:^|.*[\W_])[a-z0-9]+[-_.]?{_TOKEN_TRIGGER}{_WORD_END}"
    rf"|(?:^|.*[\W_]){_TOKEN_TRIGGER}[-_.]?"
    rf"(?:{_TOKEN_MATERIAL_HEADS})s?{_WORD_END}",
    r"(?:^|.*[\W_])oauth[-_.]?token(?:s)?(?:[\W_].*|$)",  # OAuth token patterns
]

# Pre-compiled versions for performance (avoid recompiling in loops)
COMPILED_FORBIDDEN_CONTEXT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in FORBIDDEN_CONTEXT_PATTERNS
]

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
