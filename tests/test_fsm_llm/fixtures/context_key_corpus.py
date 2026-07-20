"""Hand-authored context-key corpus for the forbidden-pattern security control.

THE CONSTRUCTION RULE IS THE POINT OF THIS FILE, so it is stated first:

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
)

# --------------------------------------------------------------------------
# PINNED KNOWN OVER-STRIP for the `key` trigger. Two-sided pin: a name leaving
# this set and a name joining it BOTH fail the test, so the measured cost of
# the fail-closed default cannot drift silently in either direction.
# Populated from the measurement, not from intent -- see D-014.
# --------------------------------------------------------------------------
CRYPTO_KEY_KNOWN_OVER_STRIPPED: frozenset[str] = frozenset()


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
# --------------------------------------------------------------------------
TOKEN_KNOWN_OVER_STRIPPED: frozenset[str] = frozenset(
    {
        "bookmark_token",
        "seek_token",
        "max_output_tokens_str",
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
}

TOKEN_SAFE_VALUES: dict[str, object] = {
    name: _TOKEN_SAFE_NON_COUNT_VALUES.get(name, 1200 + index)
    for index, name in enumerate(TOKEN_SAFE_KEYS)
}
