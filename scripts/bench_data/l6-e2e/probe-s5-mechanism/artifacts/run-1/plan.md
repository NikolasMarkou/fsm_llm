# Plan

## Goal
Add exponential backoff retry logic to the uploader module so that transient upload failures are automatically retried with increasing delays, preventing repeated failures on temporary network issues.

## Problem Statement
The current uploader.py lacks any retry mechanism. When a file upload fails due to transient network conditions (e.g., timeouts, 503 errors), it fails immediately without attempting recovery. Expected behavior: Upload should succeed or fail after N retries with exponential delays. Actual behavior: Single attempt only. Invariants: Backoff must not exceed MAX_RETRY_DELAY; retry count must not exceed MAX_RETRIES. Edge cases: Network partition, rate limiting, disk full on remote.

## Context
Foundings show uploader.py exists with basic upload logic but no error handling or retry patterns. Python standard library provides time.sleep and random for backoff calculation. No existing retry decorator found in codebase.

## Files To Modify
uploader.py: Add a retry_with_backoff() wrapper function that wraps the upload operation, calculates exponential delays using 2^attempt * base_delay, and retries up to MAX_RETRIES times on transient errors.

## Steps
1. Define constants MAX_RETRIES=5, BASE_DELAY=1s, MAX_RETRY_DELAY=60s in uploader.py (risk: low, dependency: none). 2. Implement retry_with_backoff() function using time.sleep for delays and random.uniform for jitter (risk: medium, dependency: step 1). 3. Wrap existing upload logic to call retry_with_backoff() (risk: high, dependency: steps 1-2). 4. Add unit tests verifying exponential delay calculation and max cap enforcement (risk: low, dependency: step 3)

## Assumptions
Transient errors are defined as HTTP 5xx status codes or connection timeouts; permanent errors (4xx except 409) should not retry.

## Failure Modes
Step 1 fails if constants conflict with existing code (blast radius: small). Step 2 fails if time.sleep is unavailable (blast radius: medium, fallback to manual sleep). Step 3 fails if upload signature changes (blast radius: high, requires review). Step 4 fails if tests don't catch edge cases (blast radius: low).

## Pre-Mortem & Falsification Signals
Assume plan failed: Upload still fails immediately on transient errors; backoff delays are linear instead of exponential; MAX_RETRY_DELAY is exceeded causing long waits. Stop triggers: Test suite shows non-exponential delay pattern; manual test shows upload retry count > 5.

## Success Criteria
Upload retries exactly 5 times before giving up on transient errors. Delays follow formula: min(2^attempt * 1s, 60s) with optional jitter. No permanent errors are retried. Code changes limited to uploader.py only.

## Verification Strategy
Run pytest::test_upload_retry_logic to verify exponential backoff math. Manually trigger upload failure and observe retry count and delay timing. Check that MAX_RETRIES=5 is enforced in logs.

## Complexity Budget
Max 1 new file (none, modifying existing). Max 20 lines added. No new abstractions or dependencies introduced.
