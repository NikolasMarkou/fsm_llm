# Plan

## Goal
Add exponential backoff retry logic to the uploader function in uploader.py so that transient network failures are handled automatically with increasing delays between retries.

## Problem Statement
The current uploader function makes a single HTTP request and returns immediately on failure. Expected behavior: upload succeeds or fails after one attempt. Actual behavior: no resilience against transient errors, leading to data loss or repeated manual intervention. Invariants: the retry mechanism must not alter the original URL or file path; it must respect the maximum retry count (3 attempts). Edge cases: network timeouts, HTTP 5xx errors, and connection resets should all trigger retries with exponential delays.

## Context
Foundings indicate three findings were indexed regarding the uploader's current behavior. The workspace contains uploader.py which currently uses requests.post without any error handling or retry logic. No prior decisions exist for this task.

## Files To Modify
uploader.py: wrap the upload function to add retry logic with exponential backoff, including a max_retries parameter and a delay calculation based on attempt number.

## Steps
1. Modify uploader.py to import time and random modules (risk: breaking existing imports; dependency: none). 2. Implement an internal _upload_with_retry helper that performs the HTTP request with exponential backoff delays between attempts (risk: introducing new function signature complexity; dependency: step 1). 3. Update the public upload function to call _upload_with_retry and handle the return value appropriately (risk: changing function behavior unexpectedly; dependency: step 2). 4. Add type hints and docstrings for clarity (risk: minor code quality impact; dependency: steps 1-3).

## Assumptions
- The requests library is available and stable.
- Network failures are transient and will resolve on retry.
- Maximum of 3 retry attempts is acceptable.
- Base delay is 1 second, doubling each attempt.

## Failure Modes
- Import error: if time or random modules are missing, the module fails to load (blast radius: entire uploader broken).
- Unexpected HTTP status codes: if non-standard errors occur, they may not be caught by current logic (blast radius: silent failures).
- Excessive delays: if backoff calculation is wrong, retries may take too long (blast radius: degraded performance).

## Pre-Mortem & Falsification Signals
- Assume the plan failed because the uploader now hangs indefinitely or raises an exception on first import.
- Falsification signal: running `python -c 'from workspace.uploader import upload; print(upload.__doc__)'` does not show retry logic documentation.
- Falsification signal: a test that uploads to an unreachable server succeeds without retrying.

## Success Criteria
- The uploader function includes a _upload_with_retry helper.
- Retries occur on requests.exceptions.Timeout, requests.exceptions.ConnectionError, and HTTP 5xx responses.
- Delays follow exponential backoff starting at 1 second.
- Maximum of 3 total attempts (including the first).

## Verification Strategy
- Run `python -c 'from workspace.uploader import upload; print(upload.__doc__)'` to confirm documentation includes retry logic.
- Write a quick test that uploads to an unreachable URL and verifies it retries exactly 3 times before raising.
- Check that no new files were created outside uploader.py.

## Complexity Budget
- Max 1 new file (uploader.py).
- Net lines added: ~25 lines for retry logic.
- No new abstractions or dependencies introduced.
