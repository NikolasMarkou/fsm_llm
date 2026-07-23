# Decisions

## Decision 1: Implement exponential backoff retry logic in uploader.py

**Rationale**: The plan requires adding retry logic with exponential backoff. The implementation uses a `retry` function that wraps the upload request, checking for transient errors (500/503 status codes) and applying delays of 1s, 2s, and 4s between retries. After 3 attempts total (initial + 2 retries), it raises the original exception.

**Back-link**: Plan step 1 - "uploader.py: modify upload() function to wrap requests.post with retry logic using exponential backoff"
