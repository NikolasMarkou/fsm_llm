"""Layer 1 of the forbidden-context-entry filter: the key-NAME regexes, as data.

`FORBIDDEN_CONTEXT_PATTERNS` and the fragments it is built from, with the
reasoning that shaped them. The decisions are made in `names.py`."""

from __future__ import annotations

import re

# --------------------------------------------------------------
# Forbidden Context Entry Filter
# --------------------------------------------------------------

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
    # D-052 duration/limit group: a cookie or PIN TTL, timeout or limit is a
    # number of seconds or tries, never a credential, even as a whole suffix.
    "|ttl|timeout|limit"
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
#     invariant I-7, pinned by tests/test_fsm_llm_regression/test_regression_core_bugs.py,
#     TestForbiddenContextPatterns).
#     `ssh_public_key` still strips, because `ssh` is.
# See decisions.md D-015.

# Consume a trailing plural `s` when -- and only when -- it is a plural. The
# alternation is NOT a stylistic `s?`: `s?` would let the engine backtrack to the
# no-`s` parse and re-match, so `tokens_used` would decompose as `token` + `s`
# + `_used` and strip. The second branch asserts there is no consumable `s`,
# which forecloses that parse while still letting `keystone` decompose as
# `key` + `stone` (the `s` there is followed by an alphanumeric, so it is part
# of the next word rather than a plural). A DIGIT after the `s` still makes it
# a plural (`keys2`, `tokens2`), hence `[a-z]` and not `[a-z0-9]` (D-003).
_TRIGGER_PLURAL = r"(?:s(?![a-z])|(?!s(?![a-z])))"
_KEY_TRIGGER = rf"key{_TRIGGER_PLURAL}"
_TOKEN_TRIGGER = rf"token{_TRIGGER_PLURAL}"

# Word boundary. Deliberately `(?:[\W_]|$)` and NOT D-026's `(?:[\W_].*|$)`:
# these patterns are applied with `.match()`, which never requires reaching the
# end of the string, so the trailing `.*` is pure backtracking surface. Dropping
# it removes an O(n) scan from inside two negative lookaheads that are themselves
# retried at every separator position.
#
# DECISION plan-2026-11-05T134105-5ed71cf2/D-003 -- `_WORD_END` is the boundary
# that follows a TRIGGER WORD (`key`, `token`) and so tolerates a DIGIT RUN:
# `key1`, `ssh_key2`, `token3_old` and `order_key2` are the same names as `key`,
# `ssh_key`, `token_old` and `order_key` (audit F-SU-01: `secret1`, `token2`,
# `private_key2` reached the prompt). `\d*` is a plain greedy run over a class
# disjoint from `[\W_]`, so it adds no backtracking (the `"ssh_"*n` timing pin
# still holds). `_PLAIN_WORD_END` keeps the old boundary for the three places a
# digit must NOT count, each measured by the step-3 differential: after a
# material HEAD (`keyrsa1` would be a flip with no trigger word before the
# digit), inside the safe-qualifier lookahead of `_TOKEN_VALUE_SCAN_NAME_RE`
# (`max_tokens2_billed_tokens` would flip True to False, the quirk
# `max_tokens_billed_tokens` already has), and in the credential-name rule
# (its head already eats digits; `pins1` would flip). Do NOT also let a digit
# follow a separator (`key_2`): that stays a layer-2 name on purpose. Do NOT
# fold `\d*` into `_TRIGGER_PLURAL`.
# Step 3.1: the two `trigger + head` arms in `FORBIDDEN_CONTEXT_PATTERNS` take the
# same digit run BEFORE their optional separator (`{_KEY_TRIGGER}\d*[-_.]?heads`),
# so `token2_value`, `tokens2_hash`, `key2_file` and `keys2_pair` are the names
# `token_value`, `tokens_hash`, `key_file`, `keys_pair` (review WARNING 2). The
# digit still stops at the trigger: after the HEAD the boundary stays
# `_PLAIN_WORD_END`, and the digit-free tail (`token_value2`) is unchanged.
_PLAIN_WORD_END = r"(?:[\W_]|$)"
_WORD_END = rf"(?:\d*{_PLAIN_WORD_END})"

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
# and it exists for exactly the reason that one does: the value layer (`shapes.py`)
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
# abandoned: they fall through to the value layer (`shapes.py`), which strips them
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
    # Secret-related keys (not "secretary"); a digit run may follow (`secret1`).
    rf"(?:^|.*[\W_])secret(?:s)?{_WORD_END}",
    r".*(?:api[-_.]?key|key[-_.]?api)",  # API key patterns (both orderings, with dash/underscore/dot)
    rf"(?:^|.*[\W_])credential(?:s)?{_WORD_END}",  # Credential-related keys
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
    rf"(?:{_KEY_TRIGGER}\d*$)"
    rf"|(?:^|.*[\W_]){_CRYPTO_AT_WORD}{_CRYPTO_GAP}{_KEY_TRIGGER}{_WORD_END}"
    rf"|(?:^|.*[\W_]){_KEY_TRIGGER}\d*[-_.]?"
    rf"(?:{_KEY_MATERIAL_HEADS})s?{_PLAIN_WORD_END}",
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
    # referred to the VALUE layer via `_TOKEN_VALUE_SCAN_NAME_RE` (`names.py`).
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
    rf"(?:{_TOKEN_TRIGGER}\d*$)"
    rf"|(?:^|.*[\W_])(?:{_BEARER_TOKEN_QUALIFIERS}){_SEP_RUN}"
    rf"{_TOKEN_TRIGGER}{_WORD_END}"
    rf"|(?:^|.*[\W_]){_TOKEN_TRIGGER}\d*[-_.]?"
    rf"(?:{_TOKEN_MATERIAL_HEADS})s?{_PLAIN_WORD_END}",
    rf"(?:^|.*[\W_])oauth[-_.]?token(?:s)?{_PLAIN_WORD_END}",  # OAuth token patterns
]

# Pre-compiled versions for performance (avoid recompiling in loops)
COMPILED_FORBIDDEN_CONTEXT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in FORBIDDEN_CONTEXT_PATTERNS
]
