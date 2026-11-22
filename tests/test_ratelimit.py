import pytest

from resume_analyser.ratelimit import RateLimitExceeded, SlidingWindowRateLimiter


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_blocks_after_limit_then_recovers():
    clock = FakeClock()
    rl = SlidingWindowRateLimiter(max_requests=2, window_seconds=10, clock=clock)
    rl.check("a")
    rl.check("a")
    with pytest.raises(RateLimitExceeded) as exc:
        rl.check("a")
    assert 0 < exc.value.retry_after <= 10
    clock.t += 10.1
    rl.check("a")  # window slid


def test_keys_are_independent():
    rl = SlidingWindowRateLimiter(max_requests=1, window_seconds=60, clock=FakeClock())
    rl.check("a")
    rl.check("b")
    with pytest.raises(RateLimitExceeded):
        rl.check("a")


def test_memory_is_bounded():
    rl = SlidingWindowRateLimiter(max_requests=1, window_seconds=60, max_keys=50, clock=FakeClock())
    for i in range(500):
        rl.check(f"ip-{i}")
    assert len(rl._hits) <= 50


def test_invalid_config():
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(max_requests=0)
