import time
import threading


class RateLimiter:
    """Token bucket rate limiter. Default: 1 request per 2 seconds."""

    def __init__(self, requests_per_second: float = 0.5):
        self._interval = 1.0 / requests_per_second
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            wait_for = self._last + self._interval - now
            if wait_for > 0:
                time.sleep(wait_for)
            self._last = time.monotonic()
