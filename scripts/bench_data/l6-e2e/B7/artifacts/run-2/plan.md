# Plan

## Goal
Add exponential backoff retry logic to the uploader function so that transient network failures are handled automatically with increasing delays between attempts.

## Problem Statement
The current uploader uses a single request without any retry mechanism. Expected behavior: upload succeeds on first attempt or retries with exponential delay until success or max attempts. Actual behavior: single attempt fails permanently on any error. Invariants: max 5 attempts, base delay 1s, max delay 60s, backoff multiplier 2x. Edge cases: network timeout, HTTP 5xx errors, disk full.

## Context
Foundings show uploader.py has a simple requests.post call with no retry logic. The workspace contains only three files: README.md, config.py, and uploader.py. No existing retry infrastructure found in config.py.

## Files To Modify
uploader.py - wrap the upload function with retry decorator implementing exponential backoff

## Steps
1. Add import for time module to calculate delays (risk: medium, dependency: none). 2. Implement _calculate_delay function using formula min(base * 2^attempt, max_delay) (risk: low, dependency: step 1). 3. Create retry wrapper decorator that calls upload with exponential backoff logic (risk: medium, dependency: steps 1-2). 4. Replace original upload function call in main code with retry-wrapped version (risk: high, dependency: step 3)

## Assumptions
Network timeouts are transient and recoverable. HTTP 5xx errors indicate server-side issues that may resolve. Max attempts of 5 is sufficient for most transient failures.

## Failure Modes
If time module import fails, delays won't calculate correctly causing infinite loop or crash. If max_delay cap is missing, delays could grow unbounded. If retry decorator wraps wrong function, upload behavior changes unexpectedly.

## Pre-Mortem & Falsification Signals
Assume plan failed: uploader still makes single attempt on error (check logs for no retry attempts), delays don't increase exponentially (check timing between attempts), max attempts exceeded without success (check attempt count in logs).

## Success Criteria
Upload succeeds after transient failure with exponential delay. Max 5 attempts reached before giving up. Delays follow formula min(1s * 2^n, 60s). No new files added beyond uploader.py.

## Verification Strategy
Run test case that triggers network error and verify retry occurs with correct delays. Check logs for attempt count and timing between retries. Verify function returns success on eventual upload completion.

## Complexity Budget
Max 1 new file (uploader.py modified), no new abstractions, net lines added: ~20 (decorator + helper functions).
