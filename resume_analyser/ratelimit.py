"""Thread-safe, in-memory sliding-window rate limiter.

Single-process by design (one HF Space container). For several replicas,
swap `SlidingWindowRateLimiter` for a Redis-backed implementation with the
same `check(key)` interface -- see docs/SYSTEM_DESIGN.md.
"""

import threading
import time
from collections import deque


class RateLimitExceeded(Exception):
    def __init__(self, retry_after):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry in {retry_after:.0f}s.")


class SlidingWindowRateLimiter:
    """Allow at most `max_requests` per `window_seconds` for each key."""

    def __init__(self, max_requests=10, window_seconds=60.0, max_keys=10_000, clock=time.monotonic):
        if max_requests < 1 or window_seconds <= 0:
            raise ValueError("max_requests must be >= 1 and window_seconds > 0")
        self.max_requests = max_requests
        self.window = window_seconds
        self.max_keys = max_keys  # bounds memory against many-distinct-client abuse
        self._clock = clock
        self._hits = {}
        self._lock = threading.Lock()

    def check(self, key):
        """Record a request for `key`; raise RateLimitExceeded if over the limit."""
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            cutoff = now - self.window
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.max_requests:
                raise RateLimitExceeded(retry_after=hits[0] + self.window - now)
            hits.append(now)
            if len(self._hits) > self.max_keys:
                self._evict(cutoff)

    def _evict(self, cutoff):
        for k in [k for k, h in self._hits.items() if not h or h[-1] <= cutoff]:
            del self._hits[k]
        # Still over capacity (all keys active): drop the oldest-inserted ones.
        while len(self._hits) > self.max_keys:
            self._hits.pop(next(iter(self._hits)))
