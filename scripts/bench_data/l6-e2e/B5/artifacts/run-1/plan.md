# Plan

## Goal
Add exponential backoff retry logic to the uploader function in uploader.py so that failed upload attempts are retried with increasing delays before giving up.

## Problem Statement
The current uploader function makes a single HTTP request and returns immediately on failure. Expected behavior: uploads should be resilient to transient network errors by automatically retrying with exponential backoff. Actual behavior: any error causes immediate failure. Invariants: maximum 3 retries allowed; delay between retries doubles each time starting at 1 second; final timeout must not exceed 60 seconds. Edge cases: consecutive failures, very slow networks, server-side timeouts.

## Context
Foundings show uploader.py currently has no retry logic (RETRIES=0). The config.py defines TIMEOUT=30 and ENDPOINT. No existing retry mechanism exists in the codebase.

## Files To Modify
uploader.py: modify upload() function to wrap requests.post with retry logic using exponential backoff; config.py: update RETRIES from 0 to 3

## Steps
1. Modify uploader.py to import time and add a retry wrapper around requests.post that implements exponential backoff starting at 1 second, doubling each retry up to 3 attempts total (risk: incorrect delay calculation or infinite loop if max retries not enforced; dependency: config.py values); 2. Update config.py to set RETRIES=3 (risk: wrong retry count affects reliability; dependency: none); 3. Add a helper function calculate_delay(attempt) in uploader.py that returns the backoff delay based on attempt number and config (risk: math error causing too fast or too slow retries; dependency: config.py values)

## Assumptions
The requests library is available and supports standard HTTP POST; network errors are transient and recoverable; server responses include appropriate status codes; configuration values are read at runtime

## Failure Modes
If RETRIES is set incorrectly, the system may retry too few times or too many times causing resource exhaustion; if delay calculation is wrong, retries may be too aggressive (overloading server) or too conservative (missing transient failures); if requests.post raises unexpected exceptions not caught by retry logic, backoff will not apply

## Pre-Mortem & Falsification Signals
Assume the plan failed: most likely reason is incorrect exponential formula causing delays to be constant instead of doubling; observable signal: logs show identical delays between retries; second likely reason is RETRIES=0 still in config causing no retries; observable signal: upload fails on first attempt without retry message

## Success Criteria
Upload function retries up to 3 times before failing; Delay between retries follows exponential pattern (1s, 2s, 4s); Final timeout does not exceed 60 seconds; No new files or abstractions introduced beyond uploader.py and config.py

## Verification Strategy
Run unit test that simulates 3 consecutive failures and verifies exactly 3 attempts with delays of 1s, 2s, 4s; Check logs for retry attempt messages; Verify final exception is raised after 3rd failure; Run integration test against mock server that returns 500 on first call and 200 on subsequent calls

## Complexity Budget
Max 2 files modified; No new abstractions or helper modules; Net lines added: ~15 (retry logic + delay function); No external dependencies added
