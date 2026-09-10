"""FPS measurement utilities for real-time vision pipelines."""

import time


class FPSMeter:
    """Measure instantaneous and smoothed frames-per-second."""

    def __init__(self, smoothing: float = 0.9) -> None:
        if not 0.0 < smoothing < 1.0:
            raise ValueError("smoothing must be between 0 and 1.")

        self.smoothing = smoothing
        self._last_time: float | None = None
        self._fps: float = 0.0

    def update(self) -> float:
        """Record a processed frame and return the smoothed FPS."""
        current_time = time.perf_counter()

        if self._last_time is None:
            self._last_time = current_time
            return self._fps

        elapsed = current_time - self._last_time
        self._last_time = current_time

        if elapsed <= 0:
            return self._fps

        instantaneous_fps = 1.0 / elapsed

        if self._fps == 0.0:
            self._fps = instantaneous_fps
        else:
            self._fps = (
                self.smoothing * self._fps
                + (1.0 - self.smoothing) * instantaneous_fps
            )

        return self._fps

    @property
    def fps(self) -> float:
        """Return the current FPS estimate."""
        return self._fps