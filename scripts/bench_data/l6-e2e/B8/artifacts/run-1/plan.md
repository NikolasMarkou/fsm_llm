# Plan

## Goal
Add exponential backoff retry logic to the uploader so that transient failures are automatically retried with increasing delays before failing.

## Problem Statement
The current uploader fails immediately on any error without retry. Expected behavior: upload attempts should be retried up to a maximum number of times with exponentially increasing delays between attempts. Invariants: The uploader must not block indefinitely; it must respect the configured max retries and backoff parameters. Edge cases: Network timeouts, server 5xx errors, and transient disk failures should all trigger retry logic.

## Context
Three findings are indexed regarding the uploader's current error handling behavior. The task is to implement exponential backoff without changing the core upload logic or introducing new abstractions.

## Files To Modify
uploader.py: Add retry decorator with exponential backoff configuration (max_retries=3, base_delay=1s).

## Steps
Step 1: Define constants for max_retries and base_delay in uploader.py [risk: config error, dependency: none]. Step 2: Implement the exponential_backoff_retry helper function using time.sleep with calculated delays [risk: timing issues, dependency: Step 1]. Step 3: Wrap the existing upload logic with the retry decorator [risk: incorrect wrapping, dependency: Step 2]. Step 4: Update tests to verify retry behavior under simulated failures [risk: test flakiness, dependency: Step 3].

## Assumptions
The uploader module is a Python file with an existing upload function. Network errors are raised as exceptions. The maximum number of retries and base delay are configurable constants.

## Failure Modes
Step 1 fails if constants are defined incorrectly, causing incorrect retry counts or delays. Step 2 fails if time.sleep is not available or if the delay calculation overflows. Step 3 fails if the decorator syntax is incompatible with existing code structure. Step 4 fails if tests do not properly simulate transient failures.

## Pre-Mortem & Falsification Signals
Assume the uploader hangs forever: check if max_retries is set to infinity or if base_delay is zero. Assume retries are skipped: verify that exceptions are caught and re-raised incorrectly. Assume delays are constant: inspect the sleep calculation logic for exponential growth.

## Success Criteria
The upload function retries up to 3 times on transient errors.

## Verification Strategy
Run uploader with a mock server that returns 500 error twice then succeeds; verify exactly 3 attempts were made. Check logs for exponential delay pattern (1s, 2s, 4s).

## Complexity Budget
Max 1 new file, no new abstractions, net lines added < 50.
