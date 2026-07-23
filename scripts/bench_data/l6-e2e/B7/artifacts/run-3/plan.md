# Plan

## Goal
Add exponential backoff retry logic to the uploader so that transient failures are automatically retried with increasing delays before giving up.

## Problem Statement
The current uploader fails immediately on any error without retry. Expected behavior: upload attempts should be retried up to a maximum number of times with exponentially increasing delays between attempts. Invariants: The uploader must not block indefinitely; it must respect the configured max retries and backoff parameters. Edge cases: Network timeouts, 5xx server errors, and transient disk failures should all trigger retry logic.

## Context
4 findings are indexed but no findings.md file exists in plan_dir. Based on standard patterns for this task, we need to modify the uploader's error handling to implement exponential backoff. The base delay is typically 1 second, doubling each retry attempt up to a maximum of 60 seconds or max retries of 5.

## Files To Modify
uploader.py: Add retry logic with exponential backoff in the upload function, including a helper for calculating delays and a decorator or wrapper around the upload call.

## Steps
1. [HIGH] Modify uploader.py to add an exponential_backoff_retry decorator that wraps the upload function, calculates delay as min(base * 2^attempt, max_delay), and retries up to MAX_RETRIES times. Dependency: None. Risk: Incorrect backoff calculation causing infinite loops or too-fast retries.

## Assumptions
- The uploader module has access to configuration for base_delay, max_delay, and max_retries.
- Network errors and HTTP 5xx responses are considered transient failures.
- The retry decorator can be applied without significant performance impact.

## Failure Modes
- Incorrect backoff calculation: Could cause infinite retries or exhaustion of resources. Blast radius: High.
- Missing configuration values: Plan fails silently with wrong behavior. Blast radius: Medium.
- Uploader function signature changes: Decorator breaks. Blast radius: High.

## Pre-Mortem & Falsification Signals
- Uploads fail immediately without retry on transient errors.
- Logs show constant delay between attempts instead of increasing delays.
- Tests pass but manual upload fails repeatedly.

## Success Criteria
- Uploader retries failed uploads up to MAX_RETRIES times.
- Delay between retries follows exponential backoff formula.
- Max delay does not exceed configured max_delay.
- Successful uploads complete without retry.

## Verification Strategy
- Run unit tests for the retry decorator with mocked upload failures and successes.
- Verify log output shows increasing delays between retry attempts.
- Manually trigger a transient failure and observe automatic retry behavior.

## Complexity Budget
- Add 1 new function (exponential_backoff_retry).
- Modify existing uploader.py file only.
- Net lines added: < 50.
