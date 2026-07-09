"""Rolling-throughput ETA estimation.

Feed (monotonic_time, step) observations in; ask for steps/sec and
remaining seconds out. Repeated steps are legitimate (tqdm redraws the
same step); a step LOWER than the last means a new epoch/run started, so
prior observations are discarded — a rate computed across the reset would
be negative garbage.
"""

from __future__ import annotations

from collections import deque


class EtaEstimator:
    """Throughput/ETA over a rolling window of the last `window` observations."""

    def __init__(self, window: int = 50) -> None:
        self.window = window
        self._obs: deque[tuple[float, int]] = deque(maxlen=window)

    def observe(self, t: float, step: int) -> None:
        """Record one (time, step) point; a step regression clears the window."""
        if self._obs and self._obs[-1][1] > step:
            self._obs.clear()     # new epoch reset
        self._obs.append((t, step))

    def steps_per_sec(self) -> float | None:
        """Rate across the window; None with <2 observations or no time span."""
        if len(self._obs) < 2:
            return None

        (t0, s0) = self._obs[0]
        (t1, s1) = self._obs[-1]

        dt = t1 - t0
        if dt <= 0:
            return None

        return (s1 - s0) / dt

    def eta_seconds(self, current_step: int, total_steps: int) -> float | None:
        """Seconds until total_steps at the current rate; None when the rate
        is None or <= 0 (a stalled run has no finite ETA)."""
        rate = self.steps_per_sec()

        if rate is None or rate <= 0:
            return None

        return (total_steps - current_step) / rate
        

def format_time(seconds: float | None) -> str:
    """Human Time: None → "—", 42 → "42s", 872 → "14m 32s", 3725 → "1h 2m".
    """

    if seconds is None:
        return "—"
    
    s = int(seconds)

    if s < 60:
        return f"{s}s"
    elif s < 3600:
        return f"{s // 60}m {s % 60}s"
    else:
        return f"{s // 3600}h {(s % 3600) // 60}m"
