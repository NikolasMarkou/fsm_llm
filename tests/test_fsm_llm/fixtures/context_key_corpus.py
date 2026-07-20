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
