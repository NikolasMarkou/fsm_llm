# Plan

## Goal
Add exponential backoff retry logic to the uploader function in uploader.py so that transient network failures are handled automatically with increasing delays between retries.

## Problem Statement
The current uploader function makes a single HTTP request without any retry mechanism. If the server is temporarily unavailable or returns a transient error (e.g., 503, connection timeout), the upload fails immediately. Expected behavior: uploads should succeed on transient failures after exponential backoff. Actual behavior: immediate failure on first attempt. Invariants: retries must not exceed RETRIES=0 currently but will be increased; delays must grow exponentially; max delay must cap at TIMEOUT=30s. Edge cases: consecutive failures, network partition, server-side rate limiting.

## Context
Foundings show 3 files in workspace: README.md, config.py, uploader.py. Current config has RETRIES=0 and TIMEOUT=30. The uploader function uses requests.post without any retry logic. No prior decisions exist for this task.

## Files To Modify
uploader.py: modify upload() to wrap the request in a loop with exponential backoff; config.py: increase RETRIES from 0 to 3

## Steps
1. Update config.py to set RETRIES=3 (risk: configuration drift, dependency: none); 2. Modify uploader.py's upload() function to implement retry logic with exponential backoff starting at 1s and capping at TIMEOUT (risk: infinite loop if max delay not enforced, dependency: config.py changes); 3. Add logging for each attempt and total duration (risk: verbose logs in production, dependency: step 2); 4. Update README.md to document the new retry behavior (risk: documentation mismatch, dependency: none)

## Assumptions
The requests library supports a custom retry mechanism or we can implement it manually; server errors are distinguishable from client errors via status codes; network conditions are transient and recoverable; config.py values are read at runtime, not module load time.

## Failure Modes
If RETRIES remains 0 in config.py, no retries occur. If max delay is not capped, delays grow unbounded causing long hangs. If requests library version lacks needed features, manual implementation may be incomplete. Network timeouts may not be detected correctly if socket timeout is missing.

## Pre-Mortem & Falsification Signals
Assume the upload fails after 3 attempts with increasing delays. Stop triggers: no retry loop in uploader.py code; RETRIES still equals 0 in config.py; logs show only one HTTP request attempt; tests fail on transient server errors.

## Success Criteria
The upload function retries up to RETRIES times before failing. Delays between retries follow exponential growth (1s, 2s, 4s...). Maximum delay does not exceed TIMEOUT seconds. Logs record each attempt and total elapsed time. Tests pass when server returns transient errors.

## Verification Strategy
Run a test that uploads to an endpoint returning 503 three times in succession; verify the function retries exactly RETRIES times and then fails. Check logs for exponential delay pattern. Inspect config.py to confirm RETRIES=3.

## Complexity Budget
Max 2 new files (none needed, only modifications). Add ~15 lines of code to uploader.py for retry logic. No new abstractions or modules. Net change: +10 lines in uploader.py, +1 line in config.py.
