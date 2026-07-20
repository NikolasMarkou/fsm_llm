"""BURNED independence holdout for the `key`/`token` context-value filter.

======================================================================
                            **BURNED**
======================================================================

THIS CORPUS HAS BEEN MEASURED AGAINST THE SHIPPED FILTER (steps 2-8 of
`plan-2026-07-20T103203-b8a6b855`). IT IS A **REGRESSION** ARTIFACT FROM
THIS POINT ON AND MUST **NOT** BE USED AS AN INDEPENDENCE STATISTIC AGAIN.

A future plan needing an independence number MUST derive a FRESH corpus,
exactly as this one was derived: authored in full BEFORE the pattern
source is opened in that working context. Re-quoting the figures below as
"independent" after the filter has been tuned in their sight is the metric
category error that invalidated a prior plan's headline number
(`plans/LESSONS.md` [I:5]; H-7, H-8).

    The regression-probe + shape-coverage artifact is the SEPARATE file
    `context_key_corpus.py`. Keep the two files, and the two claims they
    support, apart. Do not merge them.

----------------------------------------------------------------------
HOW THIS CORPUS WAS DERIVED -- the property that made it worth anything
----------------------------------------------------------------------

AUTHORING ORDER (H-8, `LESSONS [I:5]`). Every name and every value below
was written BEFORE `src/fsm_llm/constants.py` and BEFORE
`tests/test_fsm_llm/fixtures/context_key_corpus.py` were opened in the
authoring context. Neither file had been read in that context for any
other reason. Vocabulary was drawn from external SaaS-vendor, cloud/infra,
CI/CD, database/broker, observability, e-commerce, i18n and design-system
domains -- never from the filter's own alternations and never from any
prior corpus.

ALL credential values are SYNTHESIZED. No real secret appears here.

COMPOSITION -- 181 entries:

    key arm    41 credential / 50 safe
    token arm  40 credential / 50 safe

MEASURED VALUE-ATTRIBUTABLE SHARE AT STEP 2 (the number that makes this
corpus able to speak about the VALUE layer at all): **58.0%** of its
credential half was decided at layer 2 on the VALUE, not struck at layer 1
on the NAME -- key arm 51.2%, token arm 65.0%. A probe under a name layer 1
already strikes measures the NAME layer and says nothing about the value
layer; a majority-value-attributable corpus is the precondition for reading
its fail-open figure as a statement about the control this plan changed.
(For contrast, the shipped corpus's key arm is only **9.8%** value-attributable
and its key-arm fail-open figure must NEVER be read as a value-layer
measurement.)

    CORRECTION (step 9, adversarial review concern 8). This banner shipped
    at step 8 reading "only 3.0% value-attributable". That number is WRONG.
    3.0% was the step-2 BASELINE figure, measured at `034b0ec` BEFORE steps
    4, 5 and 8 moved five shipped key-arm credentials out of `leaked` and
    into value-attributable; it was carried into a step-8 file unchanged.
    The figure at the commit this banner ships in is **9.8% (10/102)**, and
    `test_context_unit.py`'s sibling comment ("~9%") and `decisions.md` D-008
    ("9.8%") both had it right -- this banner was the outlier of three.
    Recorded rather than silently overwritten, per the convention step 8
    established for its own two numeric corrections; the whole point of that
    convention is that a security artifact's numbers are auditable, and a
    number that was quietly swapped teaches nobody why it was wrong.
    The 3.0% -> 9.8% change does NOT weaken the sentence it sits in: at 9.8%
    the shipped key arm is still nowhere near a majority, so the contrast
    being drawn -- that this holdout can speak about layer 2 and that corpus
    cannot -- holds exactly as stated, and holds for the same reason.

    Every other number in this banner was re-verified against a live
    measurement at step 9 (entry counts, arm splits, the two intersection
    figures, the step-2 shares, and the byte lengths quoted in the residual-
    leak notes below). No further defect was found.

DISJOINTNESS vs `context_key_corpus.py`, mechanically enforced by
`test_the_burned_holdout_is_disjoint_from_the_regression_corpus`:

    NAME  intersection: 0 -- and it must stay 0.
    VALUE intersection: exactly 1 -- the scalar literal ``True``.

That one value overlap is DISCLOSED, not tolerated silently. `bool` has two
inhabitants, so it is structurally NOT re-derivable: re-deriving can only
change the NAME (done -- `streaming_enabled_token` became
`push_notifications_enabled_token`). Dropping the entry instead would have
deleted the bool arm's shape coverage, which is the worse error. Step 2
found five collisions in total and RE-DERIVED four of them rather than
trimming them, because trimming biases a corpus toward what the filter
already handles.

----------------------------------------------------------------------
RESIDUAL LEAKS -- disclosed, NOT patched
----------------------------------------------------------------------

Three credentials in this corpus still reach the prompt, and one safe value
on each arm is still stripped. They are pinned two-sided in
`test_context_unit.py` by `HOLDOUT_KNOWN_FAIL_OPEN` /
`HOLDOUT_KNOWN_OVER_STRIPPED` below -- the same mechanism as
`TOKEN_KNOWN_OVER_STRIPPED`. An id LEAVING either set is a fix that must be
recorded; an id JOINING one is an undisclosed regression.

NONE of them was patched after being observed. `LESSONS [I:4]`: declining
to patch a holdout leak after seeing it is what makes the measurement mean
anything, and a threshold tuned to an instance you just watched leak is the
one move that discipline forbids outright.

Entry shape: ``(name, value, ground_truth)`` with ground_truth in
{"credential", "safe"}. "credential" = MUST be stripped. "safe" = MUST be
kept.
"""

# --------------------------------------------------------------------------------------------
# KEY ARM -- credential
# --------------------------------------------------------------------------------------------
KEY_ARM_CREDENTIAL = [
    # SaaS vendor credentials
    (
        "algolia_admin_key",
        "b7f4c1d9e2a86530fb14cc97ae0d2b6183f5a0c74e9d1268",
        "credential",
    ),
    ("mailgun_signing_key", "mg-prv-9f3ab71ce4d0582716bd94ac3e5f0187", "credential"),
    ("pusher_secret_key", "c48e0b17ad926f3510e7cb84d2f60a9b", "credential"),
    (
        "contentful_delivery_key",
        "CFPAT-8Kq2mZr7vXn41LdBs0WgYtEuHcJi5AoP3fN",
        "credential",
    ),
    ("sendbird_master_key", "0d7a1f6b39ce85204817acfd6e2b930514cc7f8a", "credential"),
    ("zuora_rest_key", "zr_live_R6yQnT2wKp9XmB4dLc8VfHs1", "credential"),
    ("braintree_private_key", "9c1e73b48fd02a65e91cb7043fa8265d", "credential"),
    (
        "onesignal_rest_key",
        "NGY2ZTQ4YzMtOWIwMS00ZDdlLWE4ZjMtMmM1YjcxZDllMGE0",
        "credential",
    ),
    ("mux_signing_key", "Xj7Qm2Vp9RtLc4Yb8Kn1Wd6Fs3Hg5Za0Ue", "credential"),
    ("plaid_secret_key", "4f8c02b91ae7d365c0148bf2e69a7d31", "credential"),
    ("segment_write_key", "sgW_7Hn2Kq9Rt4Vb1Xm8Ld3Fc6Ya0Pz5Ju", "credential"),
    ("recurly_private_key", "e91c47af02db6835741ea9cf0b2d6837", "credential"),
    # Cloud / infrastructure credentials
    (
        "linode_api_key",
        "8f2a94c73be0d165a827fc4b90e35d17c6a04829bf7e3d12",
        "credential",
    ),
    ("hetzner_project_key", "hcloud_9KpR2mVt7XqL4bYn1Wd8Fs3Gc6Za0Ue5Jh", "credential"),
    ("scaleway_secret_key", "3b7f1e9c-4a02-4d68-b135-8ce7f0a49d2b", "credential"),
    ("ovh_consumer_key", "7Qm2Vp9RtLc4Yb8Kn1Wd6Fs3Hg5Za0Ue", "credential"),
    ("digitalocean_spaces_key", "DO00A7F2C91BE4D3685A0C", "credential"),
    ("exoscale_api_key", "EXOa1c7f394be02d5687f14ac9b", "credential"),
    ("vultr_account_key", "V5RQ2MPT7XKL4BYN1WD8FS3GC6ZA0UE5JH", "credential"),
    # CI/CD secrets
    ("drone_rpc_key", "2c9f47b1ea08d3657f04a1cb9e2d6853", "credential"),
    ("buildkite_agent_key", "bkat_6Yn1Wd8Fs3Gc0Za5Ue9Jh2Kp7Rm4Vt", "credential"),
    (
        "teamcity_auth_key",
        "eyJhbGciOiJIUzI1NiJ9.dGNhdXRoLXN0dWI.9Kp2Rm7Vt4Xq",
        "credential",
    ),
    (
        "argocd_repo_key",
        "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\n-----END OPENSSH PRIVATE KEY-----",
        "credential",
    ),
    ("woodpecker_agent_key", "wp_a71c4f93be0d2568a0c714fb9e3d", "credential"),
    # Database / message broker credentials
    ("clickhouse_cluster_key", "ch_9f3a71ce4d0582716bd94ac3e5f01872", "credential"),
    ("nats_seed_key", "SUAJ7KPR2MVT4XQL8BYN1WD3GC6ZA0UE5JHFS9RM2VT7XQ", "credential"),
    ("rabbitmq_shovel_key", "amqp-shovel-4f8c02b91ae7d365c0148bf2e69a", "credential"),
    ("cassandra_client_key", "0a4d7f19c62be38054f1ac7b9d2e6035", "credential"),
    (
        "etcd_peer_key",
        "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIQD3Kp2Rm7Vt4XqL8bYn1Wd\n-----END EC PRIVATE KEY-----",
        "credential",
    ),
    ("cockroach_cluster_key", "crdb_7Hn2Kq9Rt4Vb1Xm8Ld3Fc6Ya0Pz5", "credential"),
    ("influx_write_key", "influx-6Yn1Wd8Fs3Gc0Za5Ue9Jh2Kp7Rm4Vt-token", "credential"),
    # Observability / telemetry credentials
    ("honeycomb_ingest_key", "hcaik_01j9f3a71ce4d0582716bd94ac3e5f01", "credential"),
    ("lightstep_access_key", "LS_9Kp2Rm7Vt4XqL8bYn1Wd3Gc6Za0Ue5Jh", "credential"),
    ("victoriametrics_write_key", "vmw-3b7f1e9c4a024d68b1358ce7f0a49d2b", "credential"),
    ("loki_push_key", "loki_2c9f47b1ea08d3657f04a1cb9e2d6853", "credential"),
    # Raw crypto material
    (
        "wireguard_private_key",
        "aE7Kp2Rm7Vt4XqL8bYn1Wd3Gc6Za0Ue5JhFs9Rm2Vt=",
        "credential",
    ),
    (
        "age_identity_key",
        "AGE-SECRET-KEY-1QM2VP9RTLC4YB8KN1WD6FS3HG5ZA0UE7JX",
        "credential",
    ),
    (
        "minisign_secret_key",
        "RWQf6LRCGA9i53bXaBc0d1Ef2Gh3Ij4Kl5Mn6Op7Qr8St9Uv",
        "credential",
    ),
    (
        "luks_master_key",
        "9f3a71ce4d0582716bd94ac3e5f01872c6a04829bf7e3d1a",
        "credential",
    ),
    (
        "kerberos_keytab_key",
        "BQIAAABTAAIAB0VYQU1QTEUABGh0dHAAA2ZvbwAAAAE",
        "credential",
    ),
    ("jwt_hmac_key", "sYq2Kp9Rt4Vb1Xm8Ld3Fc6Ya0Pz5JuHn7Wg2Bs4Tk", "credential"),
]

# --------------------------------------------------------------------------------------------
# KEY ARM -- safe
# --------------------------------------------------------------------------------------------
KEY_ARM_SAFE = [
    # E-commerce / business domain data
    ("catalog_sort_key", "price_asc", "safe"),
    ("basket_group_key", "perishables", "safe"),
    ("fulfilment_route_key", "eu-west-ground", "safe"),
    ("warehouse_bin_key", "A12-R04-S07", "safe"),
    ("promo_bucket_key", "spring-clearance", "safe"),
    ("shipping_zone_key", "zone_3_islands", "safe"),
    ("tax_class_key", "reduced_rate_books", "safe"),
    ("refund_reason_key", "damaged_in_transit", "safe"),
    ("loyalty_tier_key", "silver", "safe"),
    ("invoice_series_key", "INV-2026-Q3", "safe"),
    ("supplier_contract_key", "contracts/2026/acme-supply.pdf", "safe"),
    ("price_book_key", "wholesale_eur", "safe"),
    ("return_window_key", "30_days", "safe"),
    ("bundle_offer_key", "buy2get1", "safe"),
    ("channel_listing_key", "marketplace_de", "safe"),
    # i18n / localization data
    ("locale_fallback_key", "pt-BR>pt>en", "safe"),
    ("plural_rule_key", "one|few|many|other", "safe"),
    ("currency_format_key", "de_DE.EUR.symbol_after", "safe"),
    ("translation_slug_key", "checkout.shipping.title", "safe"),
    ("rtl_layout_key", "mirror_icons", "safe"),
    ("date_pattern_key", "dd/MM/yyyy HH:mm", "safe"),
    ("collation_order_key", "und-u-co-phonebk", "safe"),
    ("message_catalog_key", "locales/fr/checkout.json", "safe"),
    ("numbering_system_key", "arabext", "safe"),
    ("transliteration_key", "Cyrl-Latn", "safe"),
    # Feature-flag / configuration data
    ("rollout_cohort_key", "beta_wave_2", "safe"),
    ("killswitch_flag_key", "disable_legacy_checkout", "safe"),
    ("experiment_variant_key", "variant_b", "safe"),
    ("config_profile_key", "staging-eu", "safe"),
    ("theme_palette_key", "high_contrast_dark", "safe"),
    ("segment_rule_key", "country in (DE, AT, CH)", "safe"),
    ("sampling_policy_key", "head_1_in_100", "safe"),
    ("retry_policy_key", "exponential_backoff_5", "safe"),
    ("quota_plan_key", "tier_enterprise", "safe"),
    ("migration_stage_key", "dual_write", "safe"),
    # Routing / partitioning / identifier data (the true UUID-carve-out beneficiaries)
    ("replay_guard_key", "8c1d4b7a-3e05-4f92-9d61-0ab7c25e4f83", "safe"),
    ("shard_route_key", "shard_07_of_64", "safe"),
    ("bucket_prefix_key", "events/2026/07/20/", "safe"),
    ("partition_route_key", "tenant-acme-eu", "safe"),
    ("lease_holder_key", "worker-node-14", "safe"),
    ("merge_conflict_key", "orders#4821", "safe"),
    ("cache_bucket_key", "product_listing_v3", "safe"),
    ("dedupe_window_key", "300s", "safe"),
    ("ordering_group_key", "customer_4821", "safe"),
    ("checkpoint_offset_key", "topic-orders:partition-3:offset-99120", "safe"),
    # Observability identifiers that happen to carry the trigger
    ("span_link_key", "trace-a1b2c3d4-span-0007", "safe"),
    ("dashboard_panel_key", "latency_p99_by_region", "safe"),
    ("alert_rule_key", "error_rate_above_2pct", "safe"),
    ("metric_series_key", "http.server.duration", "safe"),
    ("log_stream_key", "prod/api-gateway/stdout", "safe"),
]

# --------------------------------------------------------------------------------------------
# TOKEN ARM -- credential
# --------------------------------------------------------------------------------------------
TOKEN_ARM_CREDENTIAL = [
    # Developer-platform / registry tokens
    ("github_app_token", "ghs_7Kp2Rm4Vt9XqL8bYn1Wd3Gc6Za0Ue5JhFs2", "credential"),
    ("npm_publish_token", "npm_9f3a71ce4d0582716bd94ac3e5f01872c6a0", "credential"),
    ("crates_registry_token", "cioQ2mZr7vXn41LdBs0WgYtEuHcJi5AoP3", "credential"),
    ("docker_registry_token", "dckr_pat_6Yn1Wd8Fs3Gc0Za5Ue9Jh2Kp7Rm4Vt", "credential"),
    (
        "harbor_robot_token",
        "robot$ci+deploy:4f8c02b91ae7d365c0148bf2e69a7d31",
        "credential",
    ),
    ("quay_oauth_token", "1qm2vp9rtlc4yb8kn1wd6fs3hg5za0ue7jx4mb", "credential"),
    (
        "jfrog_identity_token",
        "eyJ2ZXIiOiIyIn0.anNvbi1zdHVi.7Hn2Kq9Rt4Vb1Xm8Ld",
        "credential",
    ),
    (
        "sonarqube_analysis_token",
        "sqa_0a4d7f19c62be38054f1ac7b9d2e6035c1e8",
        "credential",
    ),
    # Secret-management / orchestration tokens
    (
        "vault_wrapping_token",
        "hvs.CAESIJ7Kp2Rm4Vt9XqL8bYn1Wd3Gc6Za0Ue5Jh",
        "credential",
    ),
    ("consul_acl_token", "3b7f1e9c-4a02-4d68-b135-8ce7f0a49d2b", "credential"),
    ("nomad_management_token", "9c1e73b4-8fd0-42a6-b5e9-1cb7043fa826", "credential"),
    (
        "terraform_cloud_token",
        "atlasv1.Xj7Qm2Vp9RtLc4Yb8Kn1Wd6Fs3Hg5Za0Ue",
        "credential",
    ),
    ("boundary_recovery_token", "brt_2c9f47b1ea08d3657f04a1cb9e2d6853", "credential"),
    # Identity provider tokens
    ("okta_sso_token", "00Kp2Rm4Vt9XqL8bYn1Wd3Gc6Za0Ue5JhFs2Rm7", "credential"),
    (
        "auth0_management_token",
        "eyJhbGciOiJSUzI1NiJ9.bWdtdC1zdHVi.Vt4XqL8bYn1Wd3Gc",
        "credential",
    ),
    ("keycloak_offline_token", "9f3a71ce-4d05-4827-96bd-94ac3e5f0187", "credential"),
    (
        "dropbox_refresh_token",
        "sl.Bx7Kp2Rm4Vt9XqL8bYn1Wd3Gc6Za0Ue5JhFs2Rm7Vt",
        "credential",
    ),
    ("ping_federate_token", "pf_6Yn1Wd8Fs3Gc0Za5Ue9Jh2Kp7Rm4Vt1Xq", "credential"),
    # Deployment / edge / CDN tokens
    ("netlify_deploy_token", "nfp_4f8c02b91ae7d365c0148bf2e69a7d31c6a0", "credential"),
    ("vercel_bearer_token", "7Hn2Kq9Rt4Vb1Xm8Ld3Fc6Ya0Pz5Ju", "credential"),
    (
        "cloudflare_scoped_token",
        "v1.0-9Kp2Rm7Vt4XqL8bYn1Wd-3Gc6Za0Ue5JhFs2Rm7Vt4Xq",
        "credential",
    ),
    ("fastly_purge_token", "0a4d7f19c62be38054f1ac7b9d2e6035", "credential"),
    ("bunny_storage_token", "bn-7Kp2Rm4Vt9XqL8bYn1Wd3Gc6Za0Ue5Jh", "credential"),
    # Product / messaging / monitoring tokens
    ("pagerduty_events_token", "R02c9f47b1ea08d3657f04a1cb9e2d685", "credential"),
    (
        "slack_socket_token",
        "xapp-1-A0KP2RM4-9412876503-7hn2kq9rt4vb1xm8ld3fc6ya",
        "credential",
    ),
    ("sentry_release_token", "sntrys_6Yn1Wd8Fs3Gc0Za5Ue9Jh2Kp7Rm4Vt1Xq8", "credential"),
    (
        "grafana_service_token",
        "glsa_9f3a71ce4d0582716bd94ac3e5f0187_2c9f47b1",
        "credential",
    ),
    ("datadog_app_token", "8f2a94c73be0d165a827fc4b90e35d17c6a04829", "credential"),
    (
        "shopify_storefront_token",
        # Body deliberately contains non-hex characters. GitHub push protection
        # flagged the original 32-hex body as a live Shopify Access Token and
        # blocked the push -- a synthetic credential indistinguishable from a
        # real one is a liability in a public repo. The `shpat_` PREFIX is what
        # this entry tests (it is in `_CREDENTIAL_VALUE_PREFIXES`,
        # constants.py:790), so the prefix is preserved and only the body is
        # made obviously synthetic. Keep it that way.
        "shpat_NOTAREALTOKENnotarealtokenZZZZZZ",
        "credential",
    ),
    ("twilio_sync_token", "SK4f8c02b91ae7d365c0148bf2e69a7d31", "credential"),
    ("stripe_connect_token", "sk_live_9Kp2Rm7Vt4XqL8bYn1Wd3Gc6Za0Ue5Jh", "credential"),
    (
        "intercom_access_token",
        "dG9rOjBhNGQ3ZjE5YzYyYmUzODA1NGYxYWM3YjlkMmU2MDM1",
        "credential",
    ),
    (
        "zoom_server_token",
        "eyJhbGciOiJIUzUxMiJ9.em9vbS1zdHVi.Kp2Rm4Vt9XqL8bYn",
        "credential",
    ),
    (
        "hubspot_private_token",
        "pat-eu1-9f3a71ce-4d05-4827-96bd-94ac3e5f0187",
        "credential",
    ),
    (
        "miro_board_token",
        "eyJtaXJvLnRva2VuIjoiMWEyYjNjIn0.7Kp2Rm4Vt9XqL8bYn1Wd",
        "credential",
    ),
    (
        "figma_personal_token",
        "figd_6Yn1Wd8Fs3Gc0Za5Ue9Jh2Kp7Rm4Vt1Xq8Ld3",
        "credential",
    ),
    (
        "asana_pat_token",
        "1/1204987654321098:9f3a71ce4d0582716bd94ac3e5f01872",
        "credential",
    ),
    ("linear_api_token", "lin_api_2c9f47b1ea08d3657f04a1cb9e2d6853c1e8", "credential"),
    (
        "notion_integration_token",
        "ntn_7Kp2Rm4Vt9XqL8bYn1Wd3Gc6Za0Ue5JhFs2Rm",
        "credential",
    ),
    (
        "airtable_pat_token",
        "patQ2mZr7vXn41Ld.0a4d7f19c62be38054f1ac7b9d2e6035",
        "credential",
    ),
]

# --------------------------------------------------------------------------------------------
# TOKEN ARM -- safe
# --------------------------------------------------------------------------------------------
TOKEN_ARM_SAFE = [
    # Design-system tokens (the canonical non-credential use of the word "token")
    ("ui_badge_token", "badge.status.warning", "safe"),
    ("design_spacing_token", "space-md-16", "safe"),
    ("color_ramp_token", "brand.blue.600", "safe"),
    ("typography_scale_token", "heading-lg/1.25", "safe"),
    ("layout_grid_token", "grid.12col.gutter-24", "safe"),
    ("motion_easing_token", "ease-out-quint", "safe"),
    ("elevation_shadow_token", "shadow.level.3", "safe"),
    ("border_radius_token", "radius-pill", "safe"),
    ("icon_size_token", "icon.sm.20", "safe"),
    ("breakpoint_token", "md:768px", "safe"),
    ("opacity_scale_token", "alpha.disabled.38", "safe"),
    ("focus_ring_token", "focus.outline.2px.offset-2", "safe"),
    # Text / parsing / NLP tokens
    ("tokenizer_pad_token", "<pad>", "safe"),
    ("morpheme_split_token", "@@ation", "safe"),
    ("stopword_token", "the", "safe"),
    ("csv_delimiter_token", ";", "safe"),
    ("template_placeholder_token", "{{customer_name}}", "safe"),
    ("markdown_fence_token", "```", "safe"),
    ("regex_capture_token", "(?P<year>[0-9]{4})", "safe"),
    ("path_separator_token", "/", "safe"),
    ("document_terminator_token", "<|endofdoc|>", "safe"),
    ("lexer_error_token", "UNEXPECTED_INDENT", "safe"),
    ("prompt_token_budget", "4096", "safe"),
    ("completion_token_limit", "1536", "safe"),
    ("truncation_side_token", "left", "safe"),
    # Workflow / business-state tokens
    ("queue_position_token", "position 14 of 260", "safe"),
    ("approval_stage_token", "awaiting_finance_review", "safe"),
    ("escalation_level_token", "L2", "safe"),
    ("sla_breach_token", "breached_by_4h", "safe"),
    ("shipment_status_token", "out_for_delivery", "safe"),
    ("payment_state_token", "authorized_not_captured", "safe"),
    ("subscription_phase_token", "trial_day_7", "safe"),
    ("onboarding_step_token", "step_3_verify_email", "safe"),
    ("ticket_priority_token", "P2", "safe"),
    ("workflow_branch_token", "manual_review", "safe"),
    # i18n / content tokens
    ("icu_message_token", "{count, plural, one {# item} other {# items}}", "safe"),
    ("gender_form_token", "feminine", "safe"),
    ("honorific_token", "Dr.", "safe"),
    ("script_direction_token", "rtl", "safe"),
    ("glossary_term_token", "shopping cart", "safe"),
    # Feature-flag / config tokens
    ("flag_variant_token", "control", "safe"),
    ("release_channel_token", "canary", "safe"),
    ("maintenance_window_token", "Sun 02:00-04:00 UTC", "safe"),
    ("deprecation_notice_token", "removed in v5.0", "safe"),
    ("region_affinity_token", "eu-central-1", "safe"),
    # Non-str values on the token arm (bool/int/float KEEP per the shipped policy)
    # NOTE (disclosed, step 2): the scalar literal `True` is also present in the shipped corpus.
    # `bool` has two inhabitants, so this collision is NOT re-derivable -- re-deriving the NAME is
    # all that is possible. Reported as a structurally unavoidable scalar overlap, not trimmed,
    # because dropping it would remove the bool arm's credential/safe shape coverage entirely.
    ("push_notifications_enabled_token", True, "safe"),
    ("max_retries_token", 5, "safe"),
    ("sampling_ratio_token", 0.25, "safe"),
    ("verbose_logging_token", False, "safe"),
    ("timeout_seconds_token", 30, "safe"),
]

HOLDOUT = KEY_ARM_CREDENTIAL + KEY_ARM_SAFE + TOKEN_ARM_CREDENTIAL + TOKEN_ARM_SAFE


def arm_of(name: str) -> str:
    """Return 'key' or 'token' for a holdout name. Every name carries exactly one trigger."""
    return "token" if "token" in name else "key"


# --------------------------------------------------------------------------------------------
# DISCLOSED RESIDUAL VERDICTS -- two-sided pins, same idiom as
# `TOKEN_KNOWN_OVER_STRIPPED` in `context_key_corpus.py`.
#
# These sets are DISCLOSURE, not absolution. Each name below is a live wrong
# verdict on a corpus that had never seen the filter. They are populated FROM
# MEASUREMENT, never from intent.
# --------------------------------------------------------------------------------------------

HOLDOUT_KNOWN_FAIL_OPEN: frozenset[str] = frozenset(
    {
        # 22 characters, under the 24-character length floor, so the value layer
        # is structurally blind and the NAME layer is the only control -- and
        # `digitalocean_spaces_key` carries no name-layer trigger the filter
        # recognises. This is accepted gap G1 of the D-005 list.
        #
        # REFUSED ON MEASUREMENT, and this is the important part: step 5 swept
        # the floor at 24/22/20/18/16/14/13/12/10/8. Floor 22 catches this exact
        # value AND stays inside both bounds -- so it was available, and it was
        # still refused. Tuning a threshold to the one instance you just watched
        # leak is what `LESSONS [I:4]` forbids, and floor 22 would destroy three
        # real values (`field_key`, `natural_key`, `currency_format_key`) to buy
        # this one. Every floor low enough to also catch the shipped corpus's
        # 13-character `terminal_key` puts holdout key-arm over-strip at 18-22%,
        # past the 15% bound.
        "digitalocean_spaces_key",
        # `<label>:<pure hex>` colon composite. The filter splits on the last
        # colon and judges the tail; `robot$ci+deploy:<32 hex>` rides the
        # carve-out that exists so `cache_key: "sha256:..."` survives. Accepted
        # gap G7, disclosed by the PRIOR plan (D-021) before this corpus was
        # written -- so this is a known gap being independently re-found, not a
        # new one. No step of this plan was scoped to it.
        "harbor_robot_token",
        # `<numeric id>:<secret>` colon composite -- the Asana / Cloudinary PAT
        # shape. Same carve-out, same accepted gap G7, same disclosure. Its safe
        # twin (`routing_key` in the shipped corpus, an ordinary namespaced
        # cursor) is over-stripped by the same rule, which is why the rule
        # cannot simply be reversed: it is a genuine two-sided cost, not an
        # oversight.
        "asana_pat_token",
    }
)

HOLDOUT_KNOWN_OVER_STRIPPED: frozenset[str] = frozenset(
    {
        # `"trace-a1b2c3d4-span-0007"` -- an ordinary observability identifier
        # that clears the length floor with a mixed character class and enough
        # entropy to look generated. It IS generated; it just is not secret.
        # This is the honest, measured cost of the generic shape arm, and it is
        # the axis with headroom (S-1): over-strip 2.0% against a 15% bound.
        "span_link_key",
        # `"focus.outline.2px.offset-2"` -- a design-system token, the canonical
        # non-credential use of the word "token", struck by the same arm for the
        # same reason.
        "focus_ring_token",
    }
)
