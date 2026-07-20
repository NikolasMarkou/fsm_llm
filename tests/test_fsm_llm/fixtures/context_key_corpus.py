"""Hand-authored context-key corpus for the forbidden-pattern security control.

======================================================================
WHAT THIS ARTIFACT IS -- AND WHAT IT IS NOT
======================================================================

    THIS IS A **REGRESSION-PROBE + SHAPE-COVERAGE** ARTIFACT.
    IT IS **NOT** AN INDEPENDENCE STATISTIC.

This corpus DELIBERATELY RESTATES prior bypass vocabulary. Every name and
value a previous round measured as leaking, or as over-stripped, is kept
here on purpose so that a closed bypass cannot silently reopen. That
property is exactly what makes it USELESS as an independence measurement:
a set assembled to regress known failures is, by construction, correlated
with the failures it was assembled from.

    Computing a "vocabulary-independence" figure from this file is a
    METRIC CATEGORY ERROR. Do not do it, and do not quote a number
    derived from it as evidence that the filter generalises.

The independence corpus is a SEPARATE artifact: `holdout_key_corpus.py`
(ships at plan step 8, banner-marked BURNED once it has been measured).
Keep the two files, and the two claims they support, apart -- collapsing
them is the error that invalidated a prior plan's headline figure
(`plans/LESSONS.md` [I:5]; H-7).

The second purpose of this file is SHAPE COVERAGE. The value layer of the
control carves out ~13 value SHAPES (canonical UUID, ULID, PEM armour,
published vendor prefix, colon composites, sub-length, whitespace,
percent-encoding, paths, slash runs, single character class, low entropy)
plus the token arm's numeric/non-str split. A corpus containing zero
instances of a shape its own carve-outs exist to handle cannot measure
those carve-outs at all -- which is precisely how all five of the defects
this plan closes stayed invisible. So every shape below carries at least
one CREDENTIAL instance and at least one SAFE instance, mechanically
enforced by the shape-coverage guard in `test_context_unit.py`.

======================================================================

THE CONSTRUCTION RULE IS THE POINT OF THIS FILE, so it is stated next:

    Every name below is written by hand from REAL-WORLD application vocabulary.
    Nothing here may be generated, derived or copied from `fsm_llm.constants` --
    not from `_PASSWORD_POLICY_SUFFIXES`, not from `FORBIDDEN_CONTEXT_PATTERNS`,
    not by importing the module at all. This module MUST NOT import
    `fsm_llm.constants`.

Why the rule exists. Three consecutive corpora in this plan (step 8's, D-016's,
D-026's 6330-key one) were enumerated by crossing the allowlist's own token list
with itself. Such a corpus is STRUCTURALLY INCAPABLE of containing a key whose
suffix the allowlist omits, so it reports 0 over-strips no matter what the
pattern does -- D-026's "107 policy-shaped keys: 0 over-stripped" was guaranteed
by construction, not measured. And for three consecutive rounds the residual
defect hid in exactly that blind spot: round 1 an under-match, round 2 an
over-match of the same class. A corpus drawn from the vocabulary under test can
only ever confirm the pattern against itself.

So: to add a key here, think of an application you have seen, and write the key
name that application would use. Do not read `constants.py` first.

See decisions.md D-030.
"""

# --------------------------------------------------------------------------
# MUST STRIP. Each of these names a credential VALUE or something from which a
# credential can be recovered. A key that appears here and reaches an LLM prompt
# is a disclosure bug, not a formatting preference.
# --------------------------------------------------------------------------
SECRET_KEYS: tuple[str, ...] = (
    # the bare article, and the shapes ORMs and auth libraries actually emit
    "password",
    "passwords",
    "user_password",
    "admin_password",
    "db_password",
    "password_hash",
    "password_salt",
    "password_digest",
    "password_encrypted",
    "password_bcrypt",
    "password_sha256",
    "password_argon2",
    "password_pepper",
    "password_raw",
    "password_plaintext",
    "password_cleartext",
    "password_ciphertext",
    "password_value",
    "password_string",
    "password_text",
    "password_material",
    "password_entropy",
    "password_vault_entry",
    "password_keyring_ref",
    # infix shapes -- an ordinary policy word in the MIDDLE must not launder the
    # credential token that follows it (this is the class D-026 closed)
    "password_last_plaintext",
    "password_reset_hash",
    "password_reset_code",
    "password_reset_otp",
    "password_reset_token",
    "password_reset_link",
    "password_reset_secret",
    "password_policy_key",
    "password_retrieval",
    "password_recovery_answer",
    "password_change_old_value",
    "password_history_hashes",
    "temporary_password_value",
    "initial_password_plaintext",
    # non-password credential families sharing this control
    "api_key",
    "api_keys",
    "apikey",
    "secret",
    "secrets",
    "client_secret",
    "access_token",
    "access_tokens",
    "refresh_token",
    "bearer_token",
    "auth_token",
    "session_secret",
    "private_key",
    "private_keys",
    "ssh_private_key",
    "signing_key",
    "encryption_key",
    "oauth_token",
    "oauth_tokens",
    "credential",
    "credentials",
    "aws_secret_access_key",
    "stripe_secret_key",
)

# --------------------------------------------------------------------------
# SHOULD REACH THE PROMPT. Policy, status, UI and telemetry keys whose VALUES are
# booleans, counts, timestamps or prose. Stripping these is not dangerous, but it
# silently degrades the model's context, so it is a real cost.
#
# The near-miss entries are deliberate: `secretary`, `access_tokenizer`,
# `private_keystone` and `passwordless_login` are maximally similar to a positive.
# A negative set of obviously-safe names ("username", "email") would validate
# whatever the implementation happens to do.
# --------------------------------------------------------------------------
SAFE_KEYS: tuple[str, ...] = (
    # policy / configuration flags
    "password_policy",
    "password_policies",
    "password_required",
    "password_requirements",
    "password_min_length",
    "password_max_length",
    "password_complexity",
    "password_expiry",
    "password_expired",
    "password_strength",
    "password_manager_enabled",
    "passwordless_login",
    "passwordless_supported",
    "forgot_password_supported",
    "password_reset_enabled",
    "password_reset_flow_enabled",
    "password_change_required",
    "password_validation_rules",
    "password_setup_complete",
    # status / counters
    "password_status",
    "password_attempt_count",
    "password_failed_attempts",
    "password_retry_count",
    "password_mismatch",
    "password_reset_email_sent",
    "password_change_notification_sent",
    "password_expiry_warning_enabled",
    # temporal metadata -- the largest real-world class, and the one D-026
    # regressed against BOTH `b00fade` and D-016
    "password_updated_at",
    "password_changed_at",
    "password_changed_on",
    "password_expires_at",
    "password_expired_at",
    "password_expires_in",
    "password_reset_at",
    "password_last_changed_at",
    "password_last_updated_at",
    "password_last_rotated",
    "password_updated_time",
    "password_expiry_date",
    "password_age_in_days",
    "password_retry_after",
    "password_valid_until",
    "password_reset_requested_at",
    # numeric / versioned metadata
    "password_strength_score",
    "password_policy_version",
    "password_min_len",
    "password_max_len",
    # UI copy and links (see KNOWN_OVER_STRIPPED -- these currently do NOT pass)
    "password_help_text",
    "password_policy_text",
    "password_policy_url",
    "password_rules_url",
    "password_reset_url",
    "password_error_message",
    "password_validation_message",
    "password_requirements_list",
    "password_reset_email_template",
    "password_status_code",
    # near-miss negatives for the non-password families
    "secretary",
    "secretariat",
    "access_tokenizer",
    "private_keystone",
    "keyboard_layout",
    "monkey_species",
    "tokenizer_config",
    "credentialing_body",
)

# --------------------------------------------------------------------------
# PINNED KNOWN OVER-STRIP. A subset of SAFE_KEYS that the control currently
# strips anyway. This set is pinned EXACTLY -- a key leaving it or joining it
# both fail the test -- so the cost of the fail-closed default stays visible and
# cannot drift silently in either direction.
#
# Why these specifically are NOT fixed (D-030): a key is kept when its WHOLE
# suffix decomposes into allowlisted tokens, so any token added to the allowlist
# is also keepable as the ENTIRE suffix. Adding `code` would keep
# `password_reset_code`; `url` would keep `password_reset_url`, which embeds the
# reset token; `text` would keep `password_text` -- i.e. the password. Those
# tokens are refused, and these keys are the price. `password_last_4_changed`
# needs a bare-digit token, which `b00fade` also refused.
#
# `password_reset_requested_at` is here for a different reason: admitting it
# needs `request`/`requested`, an EVENT noun. The two groups D-030 did admit have
# crisp rules ("denotes a time, an ordinal or a number" / "denotes an outcome or
# a lifecycle state"). Event nouns are a third rule, and widening the vocabulary
# rule one convenient word at a time is precisely how D-016 shipped its bypass.
# --------------------------------------------------------------------------
KNOWN_OVER_STRIPPED: frozenset[str] = frozenset(
    {
        "password_reset_requested_at",
        "password_help_text",
        "password_policy_text",
        "password_policy_url",
        "password_rules_url",
        "password_reset_url",
        "password_error_message",
        "password_validation_message",
        "password_requirements_list",
        "password_reset_email_template",
        "password_status_code",
    }
)


# ==========================================================================
# CRYPTO-KEY TRIGGER (`key`) -- added for plan-2026-07-20T040150-876e7164 D-014
#
# Same construction rule as everything above, restated because this is where
# it has failed three times: these names were written from REAL TOOLING before
# `constants.py` was opened in this session. Sources, per group, named inline:
# OpenSSH / GitHub deploy keys, OpenSSL & PKI tooling, GnuPG, AWS KMS + GCP KMS
# + Azure Key Vault envelope-encryption vocabulary, JOSE/JWK, Rails & Django
# config files, and (for the SAFE half) PostgreSQL/MySQL/DynamoDB/Cassandra
# documentation, Redis/S3 key idioms, Stripe's idempotency header, the GNU
# gettext / Rails-i18n translation-key idiom, and this framework's own
# generic-reference `*_key` convention.
#
# NOT sourced from `_CRYPTO_KEY_QUALIFIERS`, not from FORBIDDEN_CONTEXT_PATTERNS,
# not by importing anything.
# ==========================================================================

# --------------------------------------------------------------------------
# MUST STRIP. Every one of these names either IS private key material or is the
# handle/container from which private key material is read. Deliberately avoids
# the substrings `password`, `secret`, `credential` and `api_key`, so that a
# green result here is attributable to the `key` control itself and cannot be
# borrowed from a sibling pattern.
# --------------------------------------------------------------------------
CRYPTO_KEY_SECRET_KEYS: tuple[str, ...] = (
    # the bare article
    "key",
    "keys",
    # OpenSSH / git forge deploy keys
    "ssh_key",
    "ssh_keys",
    "sshkey",
    "ssh-key",
    "ssh.key",
    "ssh_host_key",
    "ssh_private_key",
    "deploy_key",
    "deployment_key",
    "authorized_key",
    "authorized_keys",
    "host_key",
    "private_key",
    "privatekey",
    "private-key",
    "privkey",
    "priv_key",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    # key-FIRST shapes: no `*_key` suffix pattern can structurally reach these
    "keypair",
    "key_pair",
    "keypairs",
    "keyfile",
    "key_file",
    "key_material",
    "key_seed",
    "key_blob",
    "key_bytes",
    "key_data",
    "keystore",
    "keyring",
    "keychain",
    # algorithm names (OpenSSL / libsodium vocabulary)
    "rsa_key",
    "rsakey",
    "dsa_key",
    "ecdsa_key",
    "ed25519_key",
    "aes_key",
    "cipher_key",
    "hmac_key",
    "mac_key",
    # role names
    "signing_key",
    "signing_keys",
    "encryption_key",
    "decryption_key",
    "master_key",
    "session_key",
    "symmetric_key",
    "shared_key",
    "derived_key",
    "derivation_key",
    "wrapping_key",
    "unwrap_key",
    # cloud KMS envelope encryption (AWS KMS, GCP KMS, Azure Key Vault)
    "data_encryption_key",
    "key_encryption_key",
    "kek",
    "dek",
    "kms_key",
    "customer_managed_key",
    "envelope_key",
    # TLS / PKI
    "tls_key",
    "ssl_key",
    "server_key",
    "client_key",
    "cert_key",
    "certificate_key",
    "pem_key",
    "pkcs8_key",
    # GnuPG / OpenPGP
    "gpg_key",
    "gpgkey",
    "pgp_key",
    # JOSE / framework config
    "jwt_signing_key",
    "jwk_private_key",
    "rails_master_key",
    "django_signing_key",
    "vault_unseal_key",
    # account-recovery and licensing material (credential-adjacent -- D-014
    # decides `license_key` STRIPS rather than being allowlisted)
    "recovery_key",
    "unlock_key",
    "license_key",
    "activation_key",
    "serial_key",
    # case variants -- the control is case-insensitive and must stay so
    "SSH_KEY",
    "PrivateKey",
    "Signing_Key",
)

# --------------------------------------------------------------------------
# SHOULD REACH THE PROMPT. Ordinary application vocabulary. The `key` trigger
# has by far the largest legitimate surface of any word in this control, which
# is why the collateral half is bigger than the secret half.
#
# REBUILT for D-015. The D-014 version of this list was a TAUTOLOGY: adversarial
# review measured that 36 of its 36 `*_key` qualifiers were themselves members
# of the shipped `_SAFE_KEY_QUALIFIERS` allowlist, so "0/100 over-strip" was a
# restatement of the pattern, not a measurement of it. The names below are drawn
# from vocabulary the pattern author did not choose:
#   - AWS SDK / boto3 (`s3_key`, `partition_key`, `sort_key`, `range_key`)
#   - PostgreSQL, MySQL and Cassandra DDL (`candidate_key`, `clustering_key`,
#     `cluster_key`, `column_key`)
#   - Redis / memcached / Kafka (`redis_key`, `memcache_key`, `message_key`,
#     `topic_key`, `queue_key`)
#   - Django / Rails / Spring / Laravel config and ORM identifiers
#     (`settings_key`, `route_key`, `model_key`, `form_key`, `label_key`)
#   - HTTP/REST conventions (`idempotency_key`, `correlation_key`,
#     `request_key`, `dedupe_key`)
#   - ordinary domain nouns (`user_key`, `tenant_key`, `customer_key`,
#     `order_key`, `product_key`, `invoice_key`, `document_key`)
# The 31 names the reviewer's independent corpus flagged as regressed by D-014
# are ALL present here, deliberately, so that the regression cannot recur
# silently. `tests/test_fsm_llm/test_context_unit.py` carries a MECHANICAL
# anti-tautology guard over this list -- see
# `test_the_safe_corpus_is_not_a_restatement_of_the_shipped_vocabulary`.
#
# The last group is the adversarial one: ordinary English words that merely
# CONTAIN the letters k-e-y. A pattern that matches concatenated shapes (needed
# for `sshkey`/`privkey`) will eat `monkey` and `keyboard` unless it is written
# carefully, and a safe set without them would validate whatever the
# implementation happens to do (LESSONS [I:5]).
#
# CONTESTED NAME, disclosed rather than silently resolved: `product_key` lives
# HERE and not in the strip list above, where D-014 put it. It is genuinely
# ambiguous -- a Windows product key is a license credential, a product table's
# key is ordinary data -- and the independently-sourced corpus counts it as
# ordinary. `license_key`, `activation_key` and `serial_key` remain in the strip
# list as the unambiguous license-credential spellings. See D-015.
# --------------------------------------------------------------------------
CRYPTO_KEY_SAFE_KEYS: tuple[str, ...] = (
    # AWS / GCP / Azure SDK field names
    "s3_key",
    "object_key",
    "bucket_key",
    "storage_key",
    "blob_key",
    "file_key",
    "partition_key",
    "sort_key",
    "range_key",
    "hash_key",
    "item_key",
    "table_key",
    "resource_key",
    # relational / NoSQL DDL vocabulary
    "primary_key",
    "primary_keys",
    "foreign_key",
    "foreign_keys",
    "composite_key",
    "natural_key",
    "surrogate_key",
    "unique_key",
    "candidate_key",
    "index_key",
    "shard_key",
    "row_key",
    "column_key",
    "cluster_key",
    "clustering_key",
    # cache, queue and stream infrastructure
    "redis_key",
    "memcache_key",
    "cache_key",
    "cache_keys",
    "message_key",
    "queue_key",
    "topic_key",
    "record_key",
    "stream_key",
    "entry_key",
    # ordinary domain nouns
    "user_key",
    "tenant_key",
    "customer_key",
    "order_key",
    "product_key",
    "event_key",
    "document_key",
    "invoice_key",
    # framework configuration and ORM identifiers
    "config_key",
    "env_key",
    "settings_key",
    "locale_key",
    "translation_key",
    "i18n_key",
    "metadata_key",
    "state_key",
    "context_key",
    "field_key",
    "name_key",
    "value_key",
    "label_key",
    "route_key",
    "model_key",
    "form_key",
    # HTTP / REST API conventions (Stripe's idempotency header, W3C trace
    # correlation, ordinary dedupe idioms)
    "idempotency_key",
    "request_key",
    "correlation_key",
    "dedupe_key",
    "lookup_key",
    "search_key",
    "group_key",
    # language / data-structure vocabulary
    "dict_key",
    "dict_keys",
    "map_key",
    "ref_key",
    # the PUBLIC half of an asymmetric pair -- pinned by an existing regression
    # test (tests/test_fsm_llm_regression/test_regression_iter2.py), invariant I-7
    "public_key",
    "public_keys",
    # this framework's own generic-reference `*_key` idiom
    "agent_key",
    "workflow_key",
    "conversation_key",
    "conv_key",
    "wf_key",
    "result_key",
    "timer_key",
    "evidence_key",
    "payload_key",
    # ADVERSARIAL: ordinary English words that merely contain "key", plus the
    # two `key_*` phrases whose head is NOT key material
    "key_value_pair",
    "key_performance_indicator",
    "keyword",
    "keywords",
    "keyword_list",
    "search_keywords",
    "keyboard_layout",
    "keyboard_shortcut",
    "keynote_speaker",
    "keystone_species",
    "keyspace_name",
    "keypad_enabled",
    "keyframe_index",
    "monkey_species",
    "donkey_count",
    "turkey_quantity",
    "hockey_team",
    "whiskey_brand",
    "jockey_name",
    # DOTTED / HIERARCHICAL IDENTIFIER (accepted gap G8; pass-3 concern 11).
    # Added at step 9 so this class stops being invisible. An ordinary metrics
    # path -- not secret, not generated, and STRIPPED, because the D-003
    # identifier-noun carve-out only applies under a UUID/ULID-shaped value and
    # this shape is neither. It is pinned in `CRYPTO_KEY_KNOWN_OVER_STRIPPED`
    # below, which was EMPTY before this entry: the key arm shipped with zero
    # disclosed over-strip cost while the class demonstrably existed.
    "metric_key",
)

# --------------------------------------------------------------------------
# PINNED KNOWN OVER-STRIP for the `key` trigger. Two-sided pin: a name leaving
# this set and a name joining it BOTH fail the test, so the measured cost of
# the fail-closed default cannot drift silently in either direction.
# Populated from the measurement, not from intent -- see D-014.
# --------------------------------------------------------------------------
CRYPTO_KEY_KNOWN_OVER_STRIPPED: frozenset[str] = frozenset(
    {
        # ACCEPTED GAP G8 (`constants.py`, D-009); pass-3 concern 11, open
        # across two plans and pinned here for the first time.
        #
        # `metric_key: "svc.checkout.latency.p99.eu-central-1"`. A dotted
        # metrics path clears the 24-character floor with a mixed character
        # class and enough entropy to read as generated -- it IS generated; it
        # is not secret. The D-003 identifier-noun carve-out cannot rescue it
        # because that carve-out only fires under a UUID/ULID-shaped value.
        #
        # This set was EMPTY until step 9, which meant the key arm shipped
        # claiming zero disclosed over-strip cost. That was never true; it was
        # unmeasured. The class is real and its siblings on the token arm
        # (`trace_token`, `correlation_token`) are pinned in
        # `TOKEN_KNOWN_OVER_STRIPPED` for the same reason.
        #
        # DISCLOSED, NOT FIXED, and the direction is the argument: this is
        # OVER-strip, the axis with headroom (3.4% slice-total against a 15%
        # bound), whereas widening the carve-out to non-UUID shapes would hand
        # the FAIL-OPEN axis a class no value test can separate (plan A-2).
        "metric_key",
    }
)


# ==========================================================================
# AUTH-TOKEN TRIGGER (`token`) -- added for the same decision, D-014.
#
# Sources: RFC 6749/6750 (OAuth 2.0), OpenID Connect Core, the Django and
# Rails/Devise CSRF and account-confirmation flows, Slack/GitHub/Discord bot
# credentials, Azure SAS, and SAML/SSO vocabulary for the SECRET half; the
# OpenAI/Anthropic usage-object field names, HuggingFace tokenizer API, and the
# GCP `nextPageToken` / AWS `NextToken` / Azure `continuationToken` list-API
# conventions for the SAFE half.
#
# This trigger is where the collateral risk is sharpest: this is an LLM
# framework, so `*_tokens` metering vocabulary is pervasive, ordinary and
# non-secret.
# ==========================================================================

# --------------------------------------------------------------------------
# MUST STRIP. Bearer-style artefacts: possession of the VALUE is sufficient to
# act as somebody.
# --------------------------------------------------------------------------
TOKEN_SECRET_KEYS: tuple[str, ...] = (
    "token",
    "tokens",
    # CSRF / XSRF (Django's cookie is literally `csrftoken`)
    "csrf_token",
    "csrftoken",
    "csrf-token",
    "x_csrf_token",
    "xsrf_token",
    "anti_forgery_token",
    "form_token",
    # session and identity
    "session_token",
    "sessiontoken",
    "jwt_token",
    "id_token",
    "idtoken",
    "user_token",
    "admin_token",
    "service_token",
    # device / client registration
    "device_token",
    "push_token",
    "client_token",
    "registration_token",
    # passwordless / MFA / recovery flows
    "magic_link_token",
    "login_token",
    "signin_token",
    "captcha_token",
    "recaptcha_token",
    "otp_token",
    "totp_token",
    "mfa_token",
    "two_factor_token",
    "recovery_token",
    "unlock_token",
    "remember_token",
    "remember_me_token",
    # account lifecycle (Devise / allauth)
    "signup_token",
    "verification_token",
    "confirmation_token",
    "invite_token",
    "invitation_token",
    "activation_token",
    # integrations and delegation
    "webhook_token",
    "impersonation_token",
    "sso_token",
    "saml_token",
    "delegation_token",
    "bot_token",
    "slack_token",
    "github_token",
    "sas_token",
    "security_token",
    "personal_access_token",
    "upload_token",
    "download_token",
    "share_token",
    # `token_<head>` where the head names the VALUE -- the prefix shape, which
    # the D-014 corpus did not exercise at all
    "token_value",
    "token_secret",
    "token_hash",
    "token_string",
    # case variants
    "AUTH_TOKEN",
    "SessionToken",
)

# --------------------------------------------------------------------------
# SHOULD REACH THE PROMPT. Metering counts, tokenizer word-forms, NLP special
# tokens and pagination cursors. None of these is a bearer credential; all of
# them are things an LLM genuinely benefits from seeing.
#
# REBUILT for D-015, same reason as the `key` half: 21 of 21 of the D-014
# version's qualifiers were members of the shipped `_SAFE_TOKEN_QUALIFIERS`
# allowlist. The names added here come from the HuggingFace `tokenizers` /
# `transformers` special-token API (`bos_token`, `eos_token`, `pad_token`,
# `unk_token`, `mask_token`, `sep_token`, `cls_token`, `token_type_ids`,
# `token_ids`) and from ordinary throughput/cost telemetry vocabulary
# (`token_per_second`, `token_throughput`, `token_ratio`, `token_cost`) --
# neither of which the allowlist author enumerated.
# --------------------------------------------------------------------------
TOKEN_SAFE_KEYS: tuple[str, ...] = (
    # LLM metering -- the tightest collateral constraint in the whole plan
    "max_tokens",
    "max_token",
    "min_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "token_count",
    "token_counts",
    "token_usage",
    "token_limit",
    "token_budget",
    "tokens_used",
    "tokens_remaining",
    "used_tokens",
    "remaining_tokens",
    "prompt_token_count",
    "completion_token_count",
    "total_token_usage",
    "output_token_limit",
    "average_tokens",
    "estimated_tokens",
    # OpenAI / Anthropic usage-object dimensions
    "cached_tokens",
    "reasoning_tokens",
    "audio_tokens",
    "text_tokens",
    "image_tokens",
    # throughput / cost telemetry -- `token_<head>` with an ordinary tail
    "token_max_length",
    "token_per_second",
    "token_throughput",
    "token_ratio",
    "token_cost",
    "token_index",
    "token_offset",
    "token_position",
    # HuggingFace tokenizer outputs and special tokens: a "token" here is a
    # vocabulary symbol, not a bearer credential
    "token_ids",
    "token_type_ids",
    "bos_token",
    "eos_token",
    "pad_token",
    "unk_token",
    "mask_token",
    "sep_token",
    "cls_token",
    "special_tokens",
    # tokenizer word-forms -- "token" is a morpheme here, not an artefact
    "tokenizer",
    "tokenizers",
    "tokenize",
    "tokenized",
    "tokenization",
    "tokenizer_config",
    "tokenizer_name",
    "tokenized_input",
    # list-API pagination cursors (GCP nextPageToken, AWS NextToken,
    # Azure continuationToken, Google Calendar syncToken)
    "next_page_token",
    "page_token",
    "next_token",
    "continuation_token",
    "pagination_token",
    "sync_token",
    # ----------------------------------------------------------------------
    # ADDED FOR D-021 -- THE NAMES THAT MAKE THIS CORPUS A MEASUREMENT.
    #
    # Everything above this line was authored alongside `_SAFE_TOKEN_QUALIFIERS`
    # and measures 35% of its qualifier words outside the shipped vocabulary,
    # against SC-21's 50% floor. That is the D-014 tautology, reduced but not
    # gone, and D-019 finding (2) is that it was hiding a 20/20 over-strip: the
    # arm was a fail-closed allowlist over an UNBOUNDED safe space, and no
    # corpus drawn from the allowlist could ever see it.
    #
    # These names could not be added before D-021, because before D-021 they
    # ALL stripped. They are not new vocabulary invented to pass a threshold --
    # they were authored as a held-out measurement slice BEFORE the D-021
    # pattern was written, from vLLM / TGI / llama.cpp inference-server
    # counters, token-bucket rate-limiter fields, billing meters, text-chunker
    # settings, and WordPiece/SentencePiece tokenizer outputs. Three of them
    # over-strip and are pinned in `TOKEN_KNOWN_OVER_STRIPPED` rather than
    # removed; removing them is how a corpus becomes a restatement again.
    # ----------------------------------------------------------------------
    # inference-server and rate-limiter metering dimensions
    "prefill_tokens",
    "decode_tokens",
    "speculative_tokens",
    "draft_tokens",
    "kv_tokens",
    "overage_tokens",
    "quota_tokens",
    "credit_tokens",
    "burst_tokens",
    "refill_tokens",
    "billable_tokens",
    "window_tokens",
    "overlap_tokens",
    "chunk_tokens",
    "token_headroom",
    "truncated_token_count",
    "max_output_tokens_str",
    # tokenizer word-forms and NLP symbols outside the special-token API
    "subword_token",
    "wordpiece_token",
    "sentencepiece_token",
    "boundary_token",
    "lemma_token",
    "fallback_token",
    "encoding_token",
    "truncation_token",
    "role_token",
    # pagination cursors OUTSIDE the cursor vocabulary the allowlist was
    # written from. Both over-strip; see TOKEN_KNOWN_OVER_STRIPPED.
    "bookmark_token",
    "seek_token",
    # DOTTED / HIERARCHICAL IDENTIFIERS in their real wire formats (accepted
    # gap G8; pass-3 concern 11). Added at step 9. Both carry a noun that IS in
    # `_IDENTIFIER_NOUN_VOCABULARY` and both over-strip anyway, because the
    # D-003 carve-out only fires under a UUID/ULID-shaped value and neither W3C
    # `traceparent` nor a bare 32-hex trace id is one. Pinned in
    # `TOKEN_KNOWN_OVER_STRIPPED`.
    "trace_token",
    "correlation_token",
)

# --------------------------------------------------------------------------
# PINNED KNOWN OVER-STRIP for the `token` trigger. Two-sided, same rule as
# above. Populated from the measurement -- see D-014, D-021.
#
# Empty until D-021, which is itself the tell: the previous corpus contained no
# name the arm could over-strip, because it was drawn from the allowlist. These
# three are the honest cost of the D-021 design and each is a DIFFERENT cost:
#
#   `bookmark_token`, `seek_token` -- pagination cursors whose qualifier is not
#     in `_SAFE_TOKEN_QUALIFIERS`. Their values are opaque high-entropy base64,
#     i.e. CATEGORICALLY indistinguishable from a bearer token by shape, so the
#     value layer cannot save them and the name layer does not know them. This
#     is the one subspace where D-021's value signal genuinely fails, and it
#     fails in the safe direction. Adding `bookmark`/`seek` to the allowlist
#     would fix these two and is deliberately NOT done: the allowlist was
#     sourced from published cursor vocabulary before these were measured, and
#     widening it afterwards to absorb the names that caught it out is exactly
#     the co-authoring that made D-014 and D-015 report tautologies.
#
#   `max_output_tokens_str` -- caught by the PREFIX arm on the `str` material
#     head (`tokens_str` names the token AS A STRING). Pre-existing D-019
#     behaviour, not introduced here, and the name is unusual enough that the
#     head is worth more than the name.
#
#   `trace_token`, `correlation_token` -- ACCEPTED GAP G8 (`constants.py`,
#     D-009); pass-3 concern 11, open across two plans and pinned here for the
#     first time at step 9. These are the sharpest form of the class, and the
#     reason they are worth two corpus entries rather than a sentence: BOTH
#     names carry a noun that IS a member of `_IDENTIFIER_NOUN_VOCABULARY`
#     (`trace`, `correlation`), so the D-003 carve-out was written FOR them --
#     and it does not reach them, because it fires only under a UUID/ULID
#     value and these hold a W3C `traceparent` and a bare 32-hex trace id
#     respectively. The vocabulary does not protect the identifiers it was
#     named for in their most common real formats. That is the honest limit of
#     the D-003 result, and it now costs two visible entries instead of none.
# --------------------------------------------------------------------------
TOKEN_KNOWN_OVER_STRIPPED: frozenset[str] = frozenset(
    {
        "bookmark_token",
        "seek_token",
        "max_output_tokens_str",
        "trace_token",
        "correlation_token",
    }
)


# ==========================================================================
# VALUES FOR THE TOKEN CORPUS -- added for D-021
#
# A NAME-ONLY TOKEN CORPUS CAN NO LONGER MEASURE THIS CONTROL, and that is the
# whole point of D-021 rather than an inconvenience. D-015's token arm was a
# fail-CLOSED allowlist that decided on the name alone; D-019 finding (2)
# measured it as an allowlist over an UNBOUNDED safe space (20 independently
# sourced safe `*_token` names, 20/20 outside it, 20/20 over-stripped), so
# D-021 refers every non-allowlisted `<qualifier>_token` name to the VALUE.
# `user_token` is a credential or a metering count depending ENTIRELY on
# whether it holds `9dR2pQ7...` or `500`, and a corpus of bare strings cannot
# express that distinction.
#
# So the two maps below are part of the corpus, not decoration. A name whose
# value is missing here falls back to a placeholder, and the test that consumes
# them asserts total coverage so a name added to either tuple without a value
# fails loudly instead of being measured against a placeholder.
#
# The VALUES are invented for this file; the NAMES are not (see the sourcing
# notes on each tuple above). That split is deliberate: value SHAPE is what is
# under test, and shape is a structural property, so it can be authored here
# without the tautology risk that authoring NAMES here would carry.
# ==========================================================================

# Realistic bearer-credential shapes. Five distinct ones, rotated across the
# corpus, so that no single shape carries the result.
_TOKEN_CREDENTIAL_SHAPES: tuple[str, ...] = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3In0.dBjftJeZ4CVP",
    "9dR2pQ7xL4mZ8vN3bK6tY1wJ5hG0sF2aD8cE4rT7uI",
    "a8f3d9c2b1e4f7a0d3c6b9e2f5a8d1c4e7b0a3d6f9",
    # NOT the URL-encoded spelling of this shape. A percent-encoded value
    # (`Atzr%2FIQEBLjAs...`) falls outside the credential charset and is KEPT --
    # a real, separate gap that is reported rather than fixed here, because
    # D-020 scopes attempt 4 to the token arm and the colon leaks. The corpus
    # carries the raw wire form, which is what a context key actually holds.
    "AtzrIQEBLjAsAhRmHNTV5xZ8pQwLmKjNbVcXsDfGhYtRe",
    "v2.local.QnJlYWtpbmdCYWRUb2tlblZhbHVlRm9yVGVzdHM",
)

TOKEN_SECRET_VALUES: dict[str, object] = {
    name: _TOKEN_CREDENTIAL_SHAPES[index % len(_TOKEN_CREDENTIAL_SHAPES)]
    for index, name in enumerate(TOKEN_SECRET_KEYS)
}

# Bearer credentials whose VALUE is short or low-entropy, i.e. the ones the
# value layer is structurally unable to see. These exist to pin that
# `_BEARER_TOKEN_QUALIFIERS` -- the NAME layer -- is still load-bearing after
# D-021 and was not quietly made redundant by the value layer.
TOKEN_SECRET_SHORT_VALUE_ENTRIES: tuple[tuple[str, object], ...] = (
    ("csrf_token", "Wm9wOTQx"),
    ("session_token", "sess-0001"),
    ("id_token", "short.jwt.x"),
    ("otp_token", "483920"),
    ("access_token", "expired"),
    ("api_token", "-"),
    ("jwt_token", ""),
)

# Ordinary values for the safe half: metering dimensions hold counts, tokenizer
# fields hold vocabulary symbols and containers, pagination fields hold opaque
# cursors. The cursors are the interesting entries -- their values are
# INDISTINGUISHABLE from bearer credentials by shape, which is exactly why
# `_SAFE_TOKEN_QUALIFIERS` still exists and still has to carry them by name.
_TOKEN_SAFE_NON_COUNT_VALUES: dict[str, object] = {
    "token_max_length": 512,
    "token_per_second": 38.4,
    "token_throughput": 1275.5,
    "token_ratio": 0.62,
    "token_cost": 0.0031,
    "token_ids": [101, 7592, 2088, 102],
    "token_type_ids": [0, 0, 1, 1],
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "[PAD]",
    "unk_token": "<unk>",
    "mask_token": "[MASK]",
    "sep_token": "[SEP]",
    "cls_token": "[CLS]",
    "special_tokens": ["<s>", "</s>", "[PAD]"],
    "tokenizer": "cl100k_base",
    "tokenizers": ["cl100k_base", "o200k_base"],
    "tokenize": True,
    "tokenized": True,
    "tokenization": "bpe",
    "tokenizer_config": {"model_max_length": 512},
    "tokenizer_name": "bert-base-uncased",
    "tokenized_input": ["hello", "world"],
    # Opaque pagination cursors, in the wire formats the four cited APIs
    # actually emit. High-entropy by construction and safe by name only.
    "next_page_token": "CAESBQiA1LQCGgwIABIIY3Vyc29yMDE",
    "page_token": "eyJvZmZzZXQiOjEyMCwibGltaXQiOjUwfQ==",
    "next_token": "AAMA-EFvcGFxdWUtY29udGludWF0aW9uLTAwMDE",
    "continuation_token": "1!128!MDAwMDI4IWZpbGUudHh0ITAwMDAyOA==",
    "pagination_token": "cGFnZTo0Mnxzb3J0OmNyZWF0ZWRfYXRfZGVzYw==",
    "sync_token": "CPDAlvWDx70CEPDAlvWDx70CGAU=",
    # D-021 additions. The metering names fall through to the numeric default
    # below, which is the point; only the non-count values are spelled out.
    "token_headroom": 2048,
    "max_output_tokens_str": "8192",
    "subword_token": "##ing",
    "wordpiece_token": "##tion",
    "sentencepiece_token": "▁the",
    "boundary_token": "[SEP]",
    "lemma_token": "running",
    "fallback_token": "<unk>",
    "encoding_token": "cl100k_base",
    "truncation_token": "<|endoftext|>",
    "role_token": "<|assistant|>",
    # Opaque cursors under non-allowlisted qualifiers -- the two pinned
    # over-strips. Their values are what makes them unsalvageable.
    "bookmark_token": "eyJvZmZzZXQiOjQyMCwic29ydCI6ImNyZWF0ZWQifQ==",
    "seek_token": "MDAwMDQyMHxjcmVhdGVkX2F0fGRlc2M=",
    # ACCEPTED GAP G8 / pass-3 concern 11. These MUST be spelled out here: the
    # numeric default below would make the token arm keep them on the `int`
    # rule and the class would stay invisible, which is exactly the defect
    # being closed. The formats are the published ones -- a W3C `traceparent`
    # header value and a bare W3C trace id -- not shapes chosen to strip.
    "trace_token": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
    "correlation_token": "4bf92f3577b34da6a3ce929d0e0e4736",
}

TOKEN_SAFE_VALUES: dict[str, object] = {
    name: _TOKEN_SAFE_NON_COUNT_VALUES.get(name, 1200 + index)
    for index, name in enumerate(TOKEN_SAFE_KEYS)
}


# ==========================================================================
# VALUES FOR THE CRYPTO-KEY CORPUS -- defect 5, this plan's step 1.
#
# UNTIL NOW THE `key` ARM HAD NO VALUE CORPUS AT ALL. Its two corpus tests
# called `_kept(name)` with the inert default `"v"` -- three characters, one
# character class, zero entropy -- so they measured LAYER 1 (the name) and
# nothing else. The shipped control decides `stripe_key` vs `order_key`
# ENTIRELY on the value (LESSONS [I:5]), so a green result there was a
# statement about NAMES wearing the costume of a statement about the control.
#
# Same split as the token arm above, for the same reason: the NAMES were
# sourced from real application vocabulary before the pattern existed; the
# VALUES are authored here, because value SHAPE is a structural property and
# can be written without the tautology risk that authoring NAMES carries.
#
# Every name in `CRYPTO_KEY_SECRET_KEYS` and `CRYPTO_KEY_SAFE_KEYS` has an
# entry here. The consuming tests assert total coverage, so a name added to
# either tuple without a value fails loudly instead of quietly reverting to a
# placeholder -- which is the exact regression this section exists to end.
#
# A few values deliberately begin with a PUBLISHED vendor prefix
# (`sk_live_`, `ghp_`, `xoxb-`, `AKIA`). The prefix IS the shape under test,
# so it has to appear; every tail is synthesized keyboard noise. Nothing here
# is, or ever was, a live credential.
# ==========================================================================
CRYPTO_KEY_SECRET_VALUES: dict[str, object] = {
    # the bare article -- layer 1 cannot resolve these, so the value is the
    # whole control for them
    "key": "9dR2pQ7xL4mZ8vN3bK6tY1wJ5hG0sF2aD8cE4rT7uI",
    "keys": "aK3mQ9xZ7pL4vN8bR2tY6wJ1hG5sF0dC4eT8uI2oP6",
    # OpenSSH / git forge deploy keys
    "ssh_key": (
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2\n"
        "-----END OPENSSH PRIVATE KEY-----\n"
    ),
    "ssh_keys": ("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIH7mQ2xZ9pL4vN8bR2tY6wJ1hG5sF0"),
    "sshkey": "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMw",
    "ssh-key": "AAAAB3NzaC1yc2EAAAADAQABAAABgQDh7mQ2xZ9pL4vN8bR2tY6wJ1hG5sF",
    "ssh.key": "AAAAC3NzaC1lZDI1NTE5AAAAIL4vN8bR2tY6wJ1hG5sF0dC4eT8uI2oP6zQ",
    "ssh_host_key": "AAAAB3NzaC1yc2EAAAADAQABAAABgQC9pL4vN8bR2tY6wJ1hG5sF0dC",
    "ssh_private_key": (
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2\n"
        "-----END OPENSSH PRIVATE KEY-----\n"
    ),
    # DESENSITIZED. GitHub push protection blocks synthetic values that are
    # byte-indistinguishable from live credentials (it already blocked this
    # repo once, on the `shpat_` body in `holdout_key_corpus.py`). The vendor
    # PREFIX is what these entries test -- `ghp_` is in
    # `_CREDENTIAL_VALUE_PREFIXES` -- so the prefix is preserved and only the
    # body is made obviously synthetic. Re-measured: still STRIPS. Do not
    # restore a realistic body.
    "deploy_key": "ghp_NOTAREALTOKENnotarealtokenZZZZZZZZZZ",
    "deployment_key": "glpat-9pL4vN8bR2tY6wJ1hG5s",
    "authorized_key": "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDh7mQ2xZ9pL4 ci@runner",
    "authorized_keys": (
        "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDh7mQ2xZ9pL4 ci@runner\n"
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIH7mQ2xZ9pL4 ops@bastion\n"
    ),
    "host_key": "AAAAB3NzaC1yc2EAAAADAQABAAABgQDh7mQ2xZ9pL4vN8bR2tY6wJ1hG5sF",
    "private_key": (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC9pL4vN8bR2tY6\n"
        "-----END PRIVATE KEY-----\n"
    ),
    "privatekey": "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC9pL4",
    "private-key": "MIIEowIBAAKCAQEAvaS+LzfG0drWOsCdYRuWRfHRdCPhOgi9jOgs",
    "privkey": "L4vN8bR2tY6wJ1hG5sF0dC4eT8uI2oP6zQ9mK3aX7yB1nV5rD",
    "priv_key": "8bR2tY6wJ1hG5sF0dC4eT8uI2oP6zQ9mK3aX7yB1nV5rD4jH",
    "id_rsa": (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEAvaS+LzfG0drWOsCdYRuWRfHRdCPhOgi9jOgsC1nV5rD4jH2k\n"
        "-----END RSA PRIVATE KEY-----\n"
    ),
    "id_ed25519": (
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2\n"
        "-----END OPENSSH PRIVATE KEY-----\n"
    ),
    "id_ecdsa": (
        "-----BEGIN EC PRIVATE KEY-----\n"
        "MHcCAQEEIL4vN8bR2tY6wJ1hG5sF0dC4eT8uI2oP6zQ9mK3aX7yBoAoGCCqGSM49\n"
        "-----END EC PRIVATE KEY-----\n"
    ),
    "id_dsa": (
        "-----BEGIN DSA PRIVATE KEY-----\n"
        "MIIBuwIBAAKBgQC9pL4vN8bR2tY6wJ1hG5sF0dC4eT8uI2oP6zQ9mK3aX7yB1nV5\n"
        "-----END DSA PRIVATE KEY-----\n"
    ),
    # key-FIRST shapes
    "keypair": "vN8bR2tY6wJ1hG5sF0dC4eT8uI2oP6zQ9mK3aX7yB1nV5rD4jH2k",
    "key_pair": "tY6wJ1hG5sF0dC4eT8uI2oP6zQ9mK3aX7yB1nV5rD4jH2kM7nL9",
    "keypairs": "wJ1hG5sF0dC4eT8uI2oP6zQ9mK3aX7yB1nV5rD4jH2kM7nL9pT3",
    "keyfile": "/etc/fsm/secrets/service-signing-2024.pem",
    "key_file": "/var/lib/fsm/keys/envelope-eu-west-1.der",
    "key_material": "hG5sF0dC4eT8uI2oP6zQ9mK3aX7yB1nV5rD4jH2kM7nL9pT3sW6",
    "key_seed": "sF0dC4eT8uI2oP6zQ9mK3aX7yB1nV5rD4jH2kM7nL9pT3sW6xE8",
    "key_blob": "0dC4eT8uI2oP6zQ9mK3aX7yB1nV5rD4jH2kM7nL9pT3sW6xE8yR2",
    "key_bytes": "C4eT8uI2oP6zQ9mK3aX7yB1nV5rD4jH2kM7nL9pT3sW6xE8yR2uF",
    "key_data": "eT8uI2oP6zQ9mK3aX7yB1nV5rD4jH2kM7nL9pT3sW6xE8yR2uF4c",
    "keystore": "MIIKPAIBAzCCCfYGCSqGSIb3DQEHAaCCCecEggnjMIIJ3zCCBesGCSqGSIb3",
    "keyring": "8uI2oP6zQ9mK3aX7yB1nV5rD4jH2kM7nL9pT3sW6xE8yR2uF4cG7t",
    "keychain": "I2oP6zQ9mK3aX7yB1nV5rD4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1",
    # algorithm names
    "rsa_key": "MIIEowIBAAKCAQEAvaS+LzfG0drWOsCdYRuWRfHRdCPhOgi9jOgsC1nV5",
    "rsakey": "oP6zQ9mK3aX7yB1nV5rD4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5",
    "dsa_key": "P6zQ9mK3aX7yB1nV5rD4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5q",
    "ecdsa_key": "MHcCAQEEIL4vN8bR2tY6wJ1hG5sF0dC4eT8uI2oP6zQ9mK3aX7yB",
    "ed25519_key": "6zQ9mK3aX7yB1nV5rD4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ",
    "aes_key": "zQ9mK3aX7yB1nV5rD4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5q",
    "cipher_key": "Q9mK3aX7yB1nV5rD4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ",
    "hmac_key": "9mK3aX7yB1nV5rD4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8",
    "mac_key": "mK3aX7yB1nV5rD4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8d",
    # role names
    "signing_key": "K3aX7yB1nV5rD4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dN",
    "signing_keys": "3aX7yB1nV5rD4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNp",
    "encryption_key": "aX7yB1nV5rD4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpS",
    "decryption_key": "X7yB1nV5rD4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSw",
    "master_key": "7yB1nV5rD4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwR",
    "session_key": "yB1nV5rD4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRt",
    "symmetric_key": "B1nV5rD4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtY",
    "shared_key": "1nV5rD4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYh",
    "derived_key": "nV5rD4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhF",
    "derivation_key": "V5rD4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFq",
    "wrapping_key": "5rD4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFqL",
    "unwrap_key": "rD4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFqLm",
    # cloud KMS envelope encryption
    "data_encryption_key": "D4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFqLmX",
    "key_encryption_key": "4jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFqLmXc",
    "kek": "jH2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFqLmXcV",
    "dek": "H2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFqLmXcVb",
    # a real AWS KMS key id IS a canonical UUID -- the carve-out's hardest case
    "kms_key": "8f1c62d7-4a03-4b58-9e2d-15c7a6b30f94",
    "customer_managed_key": "2kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFqLmXcVbN",
    "envelope_key": "kM7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFqLmXcVbNg",
    # TLS / PKI
    "tls_key": (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC9pL4vN8bR2tY6\n"
        "-----END PRIVATE KEY-----\n"
    ),
    "ssl_key": "M7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFqLmXcVbNgK",
    "server_key": "7nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFqLmXcVbNgKr",
    # Desensitized, same reason as `deploy_key` above; `sk_live_` prefix (the
    # shape under test) preserved, body made obviously synthetic.
    "client_key": "sk_live_NOTAREALTOKENnotarealtokenZZZZZZZZZZZZZZZZZZ",
    "cert_key": "nL9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFqLmXcVbNgKrT",
    "certificate_key": "L9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFqLmXcVbNgKrTd",
    "pem_key": (
        "-----BEGIN EC PRIVATE KEY-----\n"
        "MHcCAQEEIL4vN8bR2tY6wJ1hG5sF0dC4eT8uI2oP6zQ9mK3aX7yBoAoGCCqGSM49\n"
        "-----END EC PRIVATE KEY-----\n"
    ),
    "pkcs8_key": "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC9pL4v",
    # GnuPG / OpenPGP
    "gpg_key": (
        "-----BEGIN PGP PRIVATE KEY BLOCK-----\n"
        "lQOYBGXk1pkBCADh7mQ2xZ9pL4vN8bR2tY6wJ1hG5sF0dC4eT8uI2oP6zQ9mK3aX\n"
        "-----END PGP PRIVATE KEY BLOCK-----\n"
    ),
    "gpgkey": "9pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFqLmXcVbNgKrTdW",
    "pgp_key": "pT3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFqLmXcVbNgKrTdWj",
    # JOSE / framework config
    "jwt_signing_key": "T3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFqLmXcVbNgKrTdWjP",
    "jwk_private_key": "3sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFqLmXcVbNgKrTdWjPz",
    "rails_master_key": "sW6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFqLmXcVbNgKrTdWjPzC",
    "django_signing_key": "W6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFqLmXcVbNgKrTdWjPzCu",
    "vault_unseal_key": "6xE8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFqLmXcVbNgKrTdWjPzCuA",
    # account-recovery and licensing material
    "recovery_key": "xE8y-R2uF-4cG7-tB1v-Z5qJ-8dNp-SwRt-YhFq",
    "unlock_key": "E8yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFqLmXcVbNgKrTdWjPzCuAe",
    "license_key": "8YR2U-F4CG7-TB1VZ-5QJ8D-NPSWR",
    "activation_key": "yR2uF4cG7tB1vZ5qJ8dNpSwRtYhFqLmXcVbNgKrTdWjPzCuAeS",
    "serial_key": "R2UF4-CG7TB-1VZ5Q-J8DNP-SWRTY",
    # case variants
    "SSH_KEY": "2uF4cG7tB1vZ5qJ8dNpSwRtYhFqLmXcVbNgKrTdWjPzCuAeSv",
    "PrivateKey": "uF4cG7tB1vZ5qJ8dNpSwRtYhFqLmXcVbNgKrTdWjPzCuAeSvH",
    "Signing_Key": "F4cG7tB1vZ5qJ8dNpSwRtYhFqLmXcVbNgKrTdWjPzCuAeSvHn",
}

# Ordinary application values for the safe half. Every one is what the named
# field ACTUALLY holds in the system the name was sourced from: object-storage
# paths, DDL column lists, cache namespaces, i18n message ids, cursors,
# counts. Several deliberately carry a carve-out shape (UUID, ULID, path,
# whitespace, percent-encoding, colon composite), because a carve-out with no
# safe instance is a carve-out nobody has ever measured the benefit of.
CRYPTO_KEY_SAFE_VALUES: dict[str, object] = {
    # AWS / GCP / Azure SDK field names
    "s3_key": "invoices/2024/q3/invoice-10482.pdf",
    "object_key": "tenants/acme/exports/2024/09/orders",
    "bucket_key": "acme-prod-eu-west-1",
    "storage_key": "uploads/users/10482/avatar.png",
    "blob_key": "media/2024/09/14/clip-10482.mp4",
    "file_key": "reports/quarterly/q3-2024.xlsx",
    "partition_key": "USER#10482#PROFILE",
    "sort_key": "ORDER#2024-09-14#10482",
    "range_key": "2024-09-14T11:32:07Z",
    "hash_key": "TENANT#acme",
    "item_key": "PRODUCT#SKU-10482",
    "table_key": "orders_by_customer",
    "resource_key": "projects/acme/locations/eu-west1/instances/db-1",
    # relational / NoSQL DDL vocabulary
    "primary_key": 10482,
    "primary_keys": ["order_id", "line_no"],
    "foreign_key": "customer_id",
    "foreign_keys": ["customer_id", "warehouse_id"],
    "composite_key": "acme:orders:10482",
    "natural_key": "ISBN-978-0-13-235088-4",
    "surrogate_key": 90210,
    "unique_key": "email:jane@acme.example",
    "candidate_key": "(tenant_id, external_ref)",
    "index_key": "idx_orders_customer_created",
    "shard_key": "shard-07",
    "row_key": "acme#2024-09-14#10482",
    "column_key": "cf:metrics:latency_p99",
    "cluster_key": "eu-west-1a",
    "clustering_key": "created_at DESC",
    # cache, queue and stream infrastructure
    "redis_key": "session:user:10482:profile",
    "memcache_key": "frag/product/10482/v3",
    # SC-5 pins this one individually: a truncated content digest under the
    # `<label>:<pure hex>` carve-out. Do not "fix" the filter in a way that
    # strips it.
    "cache_key": (
        "sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    ),
    "cache_keys": ["frag/product/10482/v3", "frag/cart/10482"],
    "message_key": "orders.created.10482",
    "queue_key": "billing-retry-high",
    "topic_key": "acme.orders.v2",
    "record_key": "10482",
    "stream_key": "1726312327000-0",
    "entry_key": "line-3",
    # ordinary domain nouns
    "user_key": "usr_9241",
    "tenant_key": "acme-corp",
    "customer_key": "CUST-10482",
    "order_key": "ORD-10482",
    "product_key": "SKU-88231",
    "event_key": "checkout.completed",
    "document_key": "policies/privacy-2024-09.md",
    "invoice_key": "INV-2024-10482",
    # framework configuration and ORM identifiers
    "config_key": "server.timeout_seconds",
    "env_key": "FSM_LLM_DEFAULT_MODEL",
    "settings_key": "notifications.email.digest",
    "locale_key": "en-GB",
    "translation_key": "checkout.payment.method.selector",
    "i18n_key": "errors.validation.required",
    "metadata_key": "source_system",
    "state_key": "awaiting_payment",
    "context_key": "conversation_summary",
    "field_key": "shipping_address_line_1",
    "name_key": "display_name",
    "value_key": "amount_cents",
    "label_key": "Quarterly Revenue Report 2024",
    "route_key": "%2Fapi%2Fv2%2Forders%2Fsearch",
    "model_key": "billing.Invoice",
    "form_key": "step-1-step-1-step-1-step-1",
    # HTTP / REST API conventions -- the TRUE beneficiary population of the
    # UUID/ULID carve-out (G-1): trigger-carrying identifier names. `request_id`
    # and `correlation_id` carry no `key` trigger and never reach layer 2, so
    # they are NOT what that carve-out protects; these four are.
    "idempotency_key": "3f8b2c14-9d67-4a52-b0e3-7c1f5a94d208",
    "request_key": "01HZ8QK4PYRB6JT2WMXV3NCDGF",
    "correlation_key": "b7e41c3a-58d2-4f19-9a06-3ed85c2f7b41",
    "dedupe_key": "01J9ZQ4T7XKD3M8VYB2NHF6CWE",
    "lookup_key": "email:jane.doe@acme.example",
    "search_key": "quarterly revenue",
    "group_key": "region=eu-west-1",
    # language / data-structure vocabulary
    "dict_key": "user_profile",
    "dict_keys": ["user_profile", "billing"],
    "map_key": "eu-west-1",
    "ref_key": "#/components/schemas/Order",
    # the PUBLIC half of an asymmetric pair -- invariant I-7. A public key is
    # published BY DESIGN; the wire form is the one-line `authorized_keys`
    # spelling, which is what an application actually stores.
    "public_key": ("ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDh7mQ2xZ9pL4vN8 jane@laptop"),
    "public_keys": [
        "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDh7mQ2xZ9pL4vN8 jane@laptop",
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIH7mQ2xZ9pL4vN8bR ops@bastion",
    ],
    # this framework's own generic-reference `*_key` idiom
    "agent_key": "research_agent",
    "workflow_key": "order_processing",
    "conversation_key": "conv-10482",
    "conv_key": "conv-10482",
    "wf_key": "wf-2024-09",
    "result_key": "extracted_fields",
    "timer_key": "escalation_timeout",
    "evidence_key": "audit.step.3",
    "payload_key": "customer.address",
    # ADVERSARIAL: ordinary English that merely contains "key"
    "key_value_pair": "colour=blue",
    "key_performance_indicator": "monthly_active_users",
    "keyword": "invoice",
    "keywords": ["invoice", "refund", "chargeback"],
    "keyword_list": ["shipping", "returns"],
    "search_keywords": "refund policy",
    "keyboard_layout": "en-GB QWERTY",
    "keyboard_shortcut": "Ctrl+Shift+P",
    "keynote_speaker": "Dr Amara Okafor",
    "keystone_species": "Enhydra lutris",
    "keyspace_name": "analytics_eu",
    "keypad_enabled": True,
    "keyframe_index": 47,
    "monkey_species": "Macaca fascicularis",
    "donkey_count": 12,
    "turkey_quantity": 3,
    "hockey_team": "Toronto Maple Leafs",
    "whiskey_brand": "Ardbeg Uigeadail",
    "jockey_name": "Frankie Dettori",
    # ACCEPTED GAP G8 / pass-3 concern 11. A Prometheus-style metrics path:
    # 37 characters, mixed class, dotted and dashed. Entirely ordinary, and
    # STRIPPED. Pinned in `CRYPTO_KEY_KNOWN_OVER_STRIPPED`.
    "metric_key": "svc.checkout.latency.p99.eu-central-1",
}


# ==========================================================================
# CARVE-OUT SHAPE COVERAGE -- SC-2, this plan's step 1.
#
# THE ROOT CAUSE OF ALL FIVE DEFECTS THIS PLAN CLOSES IS ONE SENTENCE: the
# value layer's carve-outs were tuned against corpora containing ZERO
# instances of the shapes they carve out. A carve-out with no corpus instance
# is not a measured trade-off, it is an untested branch with a comment
# attached.
#
# So each entry below is a (entry id, context name, value) triple pinning ONE
# value SHAPE the layer-2 chain treats specially, and every shape appears on
# BOTH sides:
#
#   CARVE_OUT_CREDENTIAL_ENTRIES -- the value IS credential material and the
#       entry MUST STRIP. An entry here that is KEPT is a FAIL-OPEN.
#   CARVE_OUT_SAFE_ENTRIES       -- the value is ordinary application data and
#       the entry MUST BE KEPT. An entry here that is STRIPPED is an
#       OVER-STRIP.
#
# The NAMES are chosen so the probe actually reaches the value layer: a name
# layer 1 already strikes measures nothing about a value shape. That is why
# the credential side uses innocuous business qualifiers (`webhook_key`,
# `partner_key`, `gateway_key`) rather than `private_key` -- `stripe_key` and
# `order_key` are the same string to the name layer (LESSONS [I:5]), and this
# corpus is the instrument that finally says so in values.
#
# `test_context_unit.py` carries a MECHANICAL shape-coverage guard that
# re-derives each shape as its OWN predicate -- it does not read these entry
# ids -- and fails naming any shape missing a credential or a safe instance.
# The ids are documentation; the predicates are the check.
# ==========================================================================
CARVE_OUT_CREDENTIAL_ENTRIES: tuple[tuple[str, str, object], ...] = (
    # A generated API key formatted as a canonical UUID. Structurally
    # IDENTICAL to an idempotency key (A-2): no value test can separate them.
    ("uuid/credential", "merchant_key", "7c3f1a92-4be8-4d17-9f60-2ab5c8e10d34"),
    # Same dilemma one alphabet over.
    ("ulid/credential", "partner_key", "01J9ZQ4T7XKD3M8VYB2NHF6CWE"),
    (
        "pem/credential",
        "vendor_key",
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC9pL4vN8bR2tY6\n"
        "-----END PRIVATE KEY-----\n",
    ),
    # Body desensitized (see `deploy_key` above). The LENGTH is load-bearing for
    # the prose below, so the replacement is exactly 30 characters and was
    # re-measured: `_generic_shape_is_credential` still returns True for it
    # unaided, which is the claim the next paragraph rests on.
    ("vendor_prefix/credential", "billing_key", "sk_live_NOTAREALTOKENnotarealt"),
    # SHAPE-COVERAGE GAP CLOSED BY PLAN STEP 8. The entry above is 30
    # characters, so it clears the 24-character floor and the GENERIC shape arm
    # catches it unaided -- the enumerated `_CREDENTIAL_VALUE_PREFIXES` denylist
    # contributes nothing to its verdict. That left the corpus with ZERO
    # instances of the ONE region where the prefix arm is genuinely
    # load-bearing: a published vendor credential SHORTER than the floor, where
    # the prefix arm fires before the length test and is the only thing standing
    # between the value and the prompt.
    #
    # Its absence is why `test_the_generic_arm_carries_the_control` measured the
    # generic arm's share at a suspiciously clean 100%: the corpus held no
    # credential the prefix arm could uniquely claim. With this entry the figure
    # is 148/149, and the 1 is this row. That is the honest number, and the drop
    # is the POINT of adding it -- a 100% that came from a missing shape is the
    # same defect class this whole plan exists to close (`LESSONS [I:5]`).
    #
    # DO NOT delete `_CREDENTIAL_VALUE_PREFIXES` on the strength of the generic
    # arm's share. This row is the counter-example: 20 characters, `AKIA`
    # prefix, synthesized (not copied from vendor documentation).
    #
    # Body desensitized (see `deploy_key` above). The 20-character LENGTH is
    # load-bearing for the paragraph above -- it is what puts the row BELOW the
    # 24-character floor -- so the replacement is exactly 20 characters and was
    # re-measured: `_generic_shape_is_credential` still returns False for it,
    # and the entry still STRIPS via the `AKIA` prefix arm alone.
    ("vendor_prefix_short/credential", "kiosk_key", "AKIANOTAREALTOKENZZZ"),
    # `<label>:<pure hex>`. D-021 KEEPS this shape by design so that
    # `cache_key: "sha256:..."` survives; a credential deliberately stored as
    # `v1:<hex>` rides that carve-out out of the filter. Disclosed, not hidden.
    (
        "hex_composite/credential",
        "signer_key",
        "v1:4f9a2c7e1b8d3506af62c94e7d10b385",
    ),
    # `<numeric id>:<secret>` -- the Asana / Cloudinary PAT shape.
    (
        "id_secret_composite/credential",
        "integration_key",
        "1207439982556301:9fK2mQ7xL4pZ8vN3bR6tY1wJ5h",
    ),
    # Below the 24-character floor. The value layer is structurally blind
    # here; only the NAME layer can carry it.
    ("short/credential", "terminal_key", "aK9dQ2mZ7pL4x"),
    # Internal whitespace: the wire form an `Authorization` header value has.
    (
        "whitespace/credential",
        "gateway_key",
        "Bearer 9dR2pQ7xL4mZ8vN3bK6tY1wJ5hG0sF2aD8cE4rT7uI",
    ),
    # Percent-encoded. Same bytes as a credential, outside the charset.
    (
        "percent/credential",
        "upload_key",
        "Atzr%2FIQEBLjAsAhRmHNTV5xZ8pQwLmKjNbVcXsDfGhYtRe",
    ),
    # Standard (non-url-safe) base64 uses `/`, and `.` separates PASETO/JWT
    # segments -- so a real credential can match the path carve-out exactly.
    (
        "path_ext/credential",
        "distribution_key",
        "9dR2pQ7xL4mZ8vN3bK6tY1wJ5hG0sF2/aD8cE4rT7uI.Xk3f9a",
    ),
    (
        "many_slash/credential",
        "mailer_key",
        "a8f3/d9c2b1e4/f7a0d3c6/b9e2f5a8d1c4e7b0a3d6f9Xk",
    ),
    # Single character class: a lowercase-only base32 secret.
    ("single_class/credential", "legacy_key", "zqmwvjxrbtkdhngfplyscuearoibfe"),
    # Long but repetitive -- a weak generated secret is still a secret.
    ("low_entropy/credential", "staging_key", "aaaaaaaaaaaaAAAAAAAAAAAA1111"),
    # TOKEN ARM ONLY. A numeric OTP seed is credential material that the
    # token arm's `bool`/`int`/`float` KEEP rule cannot see.
    ("numeric/credential", "provisioning_token", 837465019283746501),
    # TOKEN ARM ONLY. Non-str, non-numeric -- fail-CLOSED by design.
    ("non_str/credential", "wrapped_token", b"9dR2pQ7xL4mZ8vN3bK6tY1wJ5hG0sF2a"),
    # ======================================================================
    # ADDED BY PLAN STEP 10 (adversarial-review concern 2). THESE EIGHT
    # ENTRIES EXIST TO MAKE A NUMBER WORSE, AND THAT IS THE POINT.
    #
    # Step 5 disclosed gaps G2, G3 and G4 in dedicated tests that assert the
    # gap is open (`test_the_unwrapping_is_two_shapes_deep_and_that_is_
    # disclosed` and its two siblings). Those tests exhibit EIGHT distinct
    # leaking credential values -- and not one of them was counted in any
    # corpus, so no fail-open rate in this plan could see them.
    #
    # The four gaps INHERITED from before step 5 (G1, G5, G6, G7) DO have
    # corpus instances and ARE counted. So the exclusion ran in exactly one
    # direction: pre-existing gaps counted against the headline, self-inflicted
    # gaps did not. plan.md step 5 required "every residual gap named there
    # must have a pinned corpus instance", and the single deviation from that
    # rule was the deviation that preserved a PASS.
    #
    # The values are NOT duplicated here -- the three disclosure tests now
    # SOURCE their literals from these entries by id prefix (`g2/`, `g3/`,
    # `g4/`), so an exhibit can never again be demonstrated by a test while
    # being invisible to the rate. That coupling is the structural fix; adding
    # the rows is only the arithmetic one.
    #
    # DO NOT "fix" the resulting fail-open figure by removing these rows,
    # re-classifying them, or choosing a friendlier denominator. If the number
    # is past bound, the number is past bound (plan Pre-Mortem 2).
    # ----------------------------------------------------------------------
    # G2, exhibit 1/3: normalisation percent-decodes ONCE. A doubly-encoded
    # credential survives the single pass and lands back outside the charset.
    ("g2/double_percent", "gateway_key", "Atzr%252FIQEBLjAsAhRmHNTV5xZ8pQwLmKj"),
    # G2, exhibit 2/3: scheme-unwrapping handles TWO fields. A third field
    # (a real `Authorization: Bearer <jwt> <nonce>` shape) defeats it.
    (
        "g2/three_fields",
        "gateway_key",
        "Bearer 9dR2pQ7xL4mZ8vN3bK6tY1wJ5hG0sF2aD8cE4rT7uI trailing",
    ),
    # G2, exhibit 3/3: a two-field value whose first field is NOT in
    # `_AUTH_SCHEME_WORDS`. See the unlisted-scheme-word block below -- this
    # single row was the ONLY member of that class in either corpus, and its
    # word is invented rather than realistic, which is why step 10 adds five
    # realistic ones beside it.
    (
        "g2/unlisted_scheme",
        "gateway_key",
        "Sigv4Custom 9dR2pQ7xL4mZ8vN3bK6tY1wJ5hG0sF2aD8cE4rT7uI",
    ),
    # G3, exhibit 1/3: `%E2%82%AC` is a VALID escape, so decoding does not
    # drop it -- it becomes a euro sign, outside the charset, and the value is
    # KEPT.
    (
        "g3/decoded_out_of_charset",
        "upload_key",
        "Atzr%E2%82%ACIQEBLjAsAhRmHNTV5xZ8pQwLmKj",
    ),
    # G3, exhibit 2/3 and 3/3: one character outside
    # `_CREDENTIAL_VALUE_CHARSET_RE` is enough. Widening the charset is not
    # available -- `:` is excluded so `cache_key: "sha256:..."` survives
    # (D-021) -- so these are disclosed, not closable here.
    ("g3/hash_character", "upload_key", "9dR2pQ7xL4mZ8vN3bK6tY1wJ5hG0sF2a#D8cE4rT7uI"),
    ("g3/at_character", "upload_key", "9dR2pQ7xL4mZ8vN3bK6tY1wJ5hG0sF2a@D8cE4rT7uI"),
    # G4, exhibit 1/2: the residual of the path fix. Every `/`-separated
    # segment is below the length floor, so no segment is credential-shaped
    # and the joined value reads as a path.
    ("g4/short_segments", "mailer_key", "a8f3/d9c2/b1e4/f7a0/d3c6/b9e2/f5a8"),
    # G4, exhibit 2/2: segments long enough but each single-character-class.
    (
        "g4/single_class_segments",
        "mailer_key",
        "zqmwvjxrbtkdh/ngfplyscuearoibfe/xkdhngfply",
    ),
    # ======================================================================
    # ADDED BY PLAN STEP 10 (adversarial-review concern 1). THE
    # UNLISTED-SCHEME-WORD CLASS, COUNTED FOR THE FIRST TIME.
    #
    # Step 5 shipped `_AUTH_SCHEME_WORDS`, a 22-word fail-OPEN allowlist: a
    # two-field value is re-judged on its trailing field only when field 1 is
    # a LISTED word. Before this block, the evidence validating that list was:
    #   - the independently-authored holdout: ZERO scheme-wrapped credentials;
    #   - this corpus: exactly ONE (`whitespace/credential`), and its word
    #     (`Bearer`) is a member of the very list being validated.
    # That is `LESSONS [I:5]`'s "allowlist corpus restating the allowlist",
    # recurring one layer below the name layer where it was finally fixed.
    #
    # MEASURED AT STEP 10 on an independently derived population -- 40
    # realistic scheme words taken from the IANA HTTP Authentication Scheme
    # Registry, vendor `Authorization:` conventions, non-HTTP auth protocols,
    # and the degenerate labels real config files use. 19 are listed and ALL
    # 19 are caught; 21 are unlisted and ALL 21 leak. So the leak rate GIVEN
    # an unlisted word is 100% by construction (that is what fail-open
    # polarity means), and the honest figure is the COVERAGE: the 22-word list
    # covers 19/40 = 47.5% of realistic scheme words, so 52.5% [95% Wilson
    # 37.5%, 67.1%] of realistic scheme-wrapped credentials reach the prompt.
    #
    # THE FIVE ROWS BELOW ARE NOT A FIX AND MUST NOT BECOME ONE. Extending
    # `_AUTH_SCHEME_WORDS` to catch them is `LESSONS [I:4]`'s forbidden move
    # (patching the specific names an instrument just showed you leaking) and
    # would repeat round 3 verbatim. The replacement the reviewer named -- a
    # positive SHAPE test on the trailing field, judging field 2 whenever
    # field 1 is short and alphabetic -- is a filter redesign, deliberately
    # NOT in this step's scope. It is the maintainer's PIVOT decision.
    #
    # One row per derivation source, chosen for spread rather than for effect:
    (
        "scheme_iana/credential",
        "gateway_key",
        "AWS4-HMAC-SHA256 9dR2pQ7xL4mZ8vN3bK6tY1wJ5h",
    ),
    # NOT `webhook_key`: that name is struck at LAYER 1, so it would measure
    # the name layer and say nothing about the scheme word. Caught by
    # `test_every_carve_out_probe_is_decided_by_its_VALUE_not_its_name`, which
    # is precisely the guard this plan added for it.
    (
        "scheme_vendor/credential",
        "dispatch_key",
        "SharedKey 4pT7yU1iO5aS9dF3gH8jK2lZ6xC",
    ),
    (
        "scheme_oauth2/credential",
        "partner_gateway_key",
        "OAuth2 7bN3mQ9wE5rT1yU4iO8pA2sD6fG",
    ),
    (
        "scheme_protocol/credential",
        "correction_key",
        "ntrip 2xZ8vC4bN6mQ1wE9rT5yU3iO7pA",
    ),
    # The literal word `secret` as field 1 buys a KEEP. So does `key`,
    # `password` and `auth` -- measured, all four.
    (
        "scheme_degenerate/credential",
        "handoff_key",
        "secret 5gH2jK7lZ9xC4vB1nM6qW8eR3tY",
    ),
)

CARVE_OUT_SAFE_ENTRIES: tuple[tuple[str, str, object], ...] = (
    ("uuid/safe", "idempotency_key", "3f8b2c14-9d67-4a52-b0e3-7c1f5a94d208"),
    ("ulid/safe", "dedupe_key", "01HZ8QK4PYRB6JT2WMXV3NCDGF"),
    # A CERTIFICATE is PEM-armoured and is published by design. The armour
    # test cannot tell it from a private key, so this is the carve-out's cost
    # measured rather than assumed.
    (
        "pem/safe",
        "trust_anchor_key",
        "-----BEGIN CERTIFICATE-----\n"
        "MIIDdzCCAl+gAwIBAgIEAgAAuTANBgkqhkiG9w0BAQUFADBaMQswCQYDVQQGEwJJ\n"
        "-----END CERTIFICATE-----\n",
    ),
    # A cache namespace that happens to start `sk-`. Ordinary data wearing a
    # published vendor prefix.
    ("vendor_prefix/safe", "fragment_key", "sk-user-profile-10482-v3"),
    (
        "hex_composite/safe",
        "content_cache_key",
        "sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    ),
    # A namespaced cursor whose tail is high-entropy and entirely ordinary.
    (
        "id_secret_composite/safe",
        "routing_key",
        "orders:4821:9fK2mQ7xL4pZ8vN3bR6tY1wJ5h",
    ),
    ("short/safe", "order_key", "ORD-10482"),
    ("whitespace/safe", "label_key", "Quarterly Revenue Report 2024"),
    ("percent/safe", "route_key", "%2Fapi%2Fv2%2Forders%2Fsearch"),
    ("path_ext/safe", "s3_key", "invoices/2024/q3/invoice-10482.pdf"),
    ("many_slash/safe", "object_key", "tenants/acme/exports/2024/09/orders"),
    ("single_class/safe", "translation_key", "checkout.payment.method.selector"),
    ("low_entropy/safe", "form_key", "step-1-step-1-step-1-step-1"),
    ("numeric/safe", "retry_token", 3),
    ("non_str/safe", "vendor_token", ["primary", "secondary"]),
    # ======================================================================
    # ADDED BY PLAN STEP 10 (adversarial-review concern 4). THE
    # UNLISTED-IDENTIFIER-NOUN OVER-STRIP CLASS, COUNTED FOR THE FIRST TIME.
    #
    # Step 4 narrowed the UUID/ULID carve-out to `_IDENTIFIER_NOUN_VOCABULARY`,
    # a 17-noun fail-CLOSED allowlist: a UUID-shaped value is kept only when
    # the NAME carries a listed identifier noun. D-003 argued the polarity was
    # safe because "an omitted noun costs OVER-STRIP, the axis with headroom
    # (S-1)". The DIRECTION of that argument is correct. Its MAGNITUDE was
    # never measured, and the anchor's supporting census -- "the carve-out's
    # ENTIRE true reach at 12 entries out of 557 (2.2%)" -- is a census of THE
    # CORPORA, not of the population. Neither corpus contained one instance of
    # the excluded class, so 2.2% is a statement about what had been collected,
    # not about what the rule reaches in a real application.
    #
    # MEASURED AT STEP 10 on an independently derived population -- 29
    # plausible trigger-carrying identifier names from everyday messaging,
    # workflow, storage and transactional vocabulary, each holding a canonical
    # UUID: **26/29 = 89.7% OVER-STRIPPED** [95% Wilson 73.6%, 96.4%], six
    # times the 15% bound. Only `pagination_token`, `continuation_token` and
    # `sync_token` survive. The same 29 names holding a ULID score identically
    # (26/29); holding the inert `"v"` all but two are KEPT, so the verdict is
    # value-attributable and this is genuinely layer 2's cost.
    #
    # `message_key`, `event_key` and `transaction_key` holding a UUID are
    # everyday Kafka/eventing context data. This is the largest cost step 4
    # incurred and it is landing on the axis that HAS headroom -- see the D-010
    # table for where it actually lands once counted.
    #
    # DO NOT extend `_IDENTIFIER_NOUN_VOCABULARY` to catch these nine. That is
    # `LESSONS [I:4]`'s forbidden move and `LESSONS [I:5]`'s "fixing a class by
    # naming its members" in one gesture. The two rows below excluded from my
    # 29-name probe (`envelope_key`, `subscription_key`) are excluded on
    # purpose: they strip at layer 1 on the NAME, so they would measure the
    # name layer and say nothing about this carve-out.
    (
        "unlisted_noun_message/safe",
        "message_key",
        "6b1f0c48-2a97-4e35-8d10-c47f9b3e2a56",
    ),
    ("unlisted_noun_event/safe", "event_key", "d4e9107a-3c05-42fb-91a8-6de207b4c193"),
    (
        "unlisted_noun_transaction/safe",
        "transaction_key",
        "b820ff3c-15d9-4e07-a6b2-90c4e1d738ab",
    ),
    (
        "unlisted_noun_workflow/safe",
        "workflow_key",
        "5a7c2e91-88b4-4f30-9d16-3ea0c5b72f4d",
    ),
    (
        "unlisted_noun_snapshot/safe",
        "snapshot_key",
        "0c96d4b1-72fa-4e58-8b03-d19e6a2c7458",
    ),
    (
        "unlisted_noun_document/safe",
        "document_key",
        "e31a5f70-6c28-4ba9-95d7-1f8b04e3c6a2",
    ),
    ("unlisted_noun_ledger/safe", "ledger_key", "97fb3d02-4a61-4c8e-b750-2d6ac9e15f83"),
    (
        "unlisted_noun_conversation/safe",
        "conversation_token",
        "1d40b8e6-59c3-4a72-8f91-7b52ce0d36a4",
    ),
    (
        "unlisted_noun_thread/safe",
        "thread_token",
        "af62c1e9-30d7-4b85-a2c6-84f19b0e5d37",
    ),
)

# --------------------------------------------------------------------------
# PINNED KNOWN-WRONG carve-out verdicts, two-sided, exactly like
# `TOKEN_KNOWN_OVER_STRIPPED` above: an entry LEAVING either set is a fix that
# must be recorded, an entry JOINING it is an undisclosed regression.
#
# These sets are DISCLOSURE, not absolution. Each id below names a live defect
# with the plan step that owns its fix. They are populated FROM MEASUREMENT,
# never from intent -- step 1 measures, steps 4 and 5 fix.
# --------------------------------------------------------------------------
#
# MEASURED AT aa284b7, THE FIRST TIME THIS SEAM WAS EVER PROBED WITH VALUES OF
# ITS OWN CARVE-OUT SHAPES: 11 of 15 credential shapes reach the prompt and 4
# of 15 safe shapes do not. Those two numbers are the reason this plan exists,
# and they were INVISIBLE to every prior round -- not because the tests were
# weak, but because the corpus had no instance of any shape to show them.
#
# NOTHING IN THIS FILE MAY BE "FIXED" BY EDITING THE FILTER IN STEP 1. The
# pins below make the defects visible and green-on-purpose; steps 4 and 5 move
# entries OUT of these sets, and the two-sided pin is what forces that
# movement to be recorded rather than absorbed.
CARVE_OUT_KNOWN_FAIL_OPEN: frozenset[str] = frozenset(
    {
        # --- FIXED by plan step 4 (D-003), no longer listed ----------------
        # `uuid/credential` (`merchant_key`) and `ulid/credential`
        # (`partner_key`) both leaked here until step 4 narrowed the UUID/ULID
        # carve-out to a fail-closed identifier-noun vocabulary. Neither name
        # carries an identifier noun, so both now fall through to the generic
        # shape arm and STRIP. Their absence from this set is the mechanical
        # proof of that fix; their safe counterparts (`uuid/safe`,
        # `ulid/safe`) are still kept and are pinned by
        # `CARVE_OUT_KNOWN_OVER_STRIPPED` NOT listing them.
        # --- FIXED by plan step 5 (D-005), no longer listed ----------------
        # `whitespace/credential` (`gateway_key`), `percent/credential`
        # (`upload_key`), `path_ext/credential` (`distribution_key`) and
        # `many_slash/credential` (`mailer_key`) all leaked here until step 5
        # unified the charset/shape rule: transport unwrapping now runs before
        # the charset test, and the path carve-out now requires that no
        # `/`-separated segment is itself credential-shaped. Their absence from
        # this set is the mechanical proof of that fix, and their safe
        # counterparts (`whitespace/safe`, `percent/safe`, `path_ext/safe`,
        # `many_slash/safe`) are still kept -- pinned by
        # `CARVE_OUT_KNOWN_OVER_STRIPPED` NOT listing them.
        # --- DISCLOSED ACCEPTED GAPS, no step fixes these -----------------
        # G1 of the D-005 accepted-gaps list. Under the 24-character floor, so
        # the value layer is structurally blind and the NAME layer is the only
        # control. Step 5 MEASURED every lower floor and refused all of them:
        # any floor reaching 13 characters puts holdout over-strip at 18-22%,
        # past the 15% bound. See the D-005 block in constants.py.
        "short/credential",
        # G7. `v1:<hex>` rides the `<label>:<pure hex>` carve-out that exists so
        # `cache_key: "sha256:..."` survives (D-021 states this cost outright).
        "hex_composite/credential",
        # G5 and G6. The generic arm's own floors. A single-character-class or
        # sub-3.0-bit secret is a WEAK secret; raising either floor to catch
        # it destroys ordinary identifiers (`idx_orders_customer_created`,
        # `checkout.payment.method.selector`) wholesale. The NAME layer is the
        # control for these, which is exactly what
        # `test_the_bearer_name_list_still_carries_short_credentials` pins.
        "single_class/credential",
        "low_entropy/credential",
        # The token arm KEEPS `bool`/`int`/`float` by design, so a numeric OTP
        # seed is kept. Reversing that strips every metering count in the
        # framework -- the cure is far worse (D-021).
        "numeric/credential",
        # --- ADDED BY STEP 10: the G2/G3/G4 exhibits, now COUNTED ----------
        # These eight were demonstrated by three shipped tests and counted by
        # nothing. Disclosing a gap in a test while excluding it from the rate
        # is disclosure that costs the headline nothing, which is the objection
        # adversarial-review concern 2 raised. They are pinned here on exactly
        # the same terms as the four gaps above: an id LEAVING this set is a
        # FIX and must be recorded; the set is DISCLOSURE, not absolution.
        "g2/double_percent",
        "g2/three_fields",
        "g2/unlisted_scheme",
        "g3/decoded_out_of_charset",
        "g3/hash_character",
        "g3/at_character",
        "g4/short_segments",
        "g4/single_class_segments",
        # --- ADDED BY STEP 10: the unlisted-scheme-word class, COUNTED -----
        # G2 by classification, but listed separately because they are a
        # DIFFERENT claim: the eight above show the gap is real, these five
        # show it is BROAD. 21 of 40 independently derived realistic scheme
        # words are unlisted, and every unlisted word leaks by construction.
        # DO NOT close these by adding words to `_AUTH_SCHEME_WORDS`.
        "scheme_iana/credential",
        "scheme_vendor/credential",
        "scheme_oauth2/credential",
        "scheme_protocol/credential",
        "scheme_degenerate/credential",
    }
)

CARVE_OUT_KNOWN_OVER_STRIPPED: frozenset[str] = frozenset(
    {
        # All four are the honest, deliberate cost of a rule that is doing its
        # job, not defects. Each is the safe side of a shape whose credential
        # side the same rule catches.
        # A published certificate wears the same armour as a private key.
        "pem/safe",
        # A cache namespace that happens to begin `sk-`.
        "vendor_prefix/safe",
        # D-021 splits on the last colon and judges the tail; an ordinary
        # high-entropy cursor tail looks exactly like an Asana PAT tail.
        "id_secret_composite/safe",
        # The token arm is fail-CLOSED for non-str, non-numeric values (S-2),
        # so a list-valued `*_token` is stripped.
        "non_str/safe",
        # --- ADDED BY STEP 10: the unlisted-identifier-noun class, COUNTED --
        # Nine ordinary eventing/workflow/storage identifiers holding a
        # canonical UUID, none carrying a noun in the 17-word
        # `_IDENTIFIER_NOUN_VOCABULARY`. All nine are destroyed. This is the
        # measured cost of step 4's fail-CLOSED polarity, on the axis D-003
        # correctly said it would land on -- see the block above them for the
        # 89.7% (26/29) figure from the independent probe, and D-010 for where
        # the over-strip rate lands once they are counted.
        #
        # These are NOT "defects to fix by adding nouns". Every noun added is
        # one more member of a class named member-by-member (`LESSONS [I:5]`).
        "unlisted_noun_message/safe",
        "unlisted_noun_event/safe",
        "unlisted_noun_transaction/safe",
        "unlisted_noun_workflow/safe",
        "unlisted_noun_snapshot/safe",
        "unlisted_noun_document/safe",
        "unlisted_noun_ledger/safe",
        "unlisted_noun_conversation/safe",
        "unlisted_noun_thread/safe",
    }
)
