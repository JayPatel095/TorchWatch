"""Rolling-throughput ETA estimation.

Feed the estimator (monotonic_time, step) observations as they arrive from
the metrics stream; ask it for steps/sec and remaining seconds. Contract:

- The rolling rate uses only the most recent `window` observations:
  rate = (last_step - first_step) / (last_time - first_time) over that span.
- Fewer than 2 observations, or a zero time span, yields rate None.
- Steps can repeat (tqdm redraws the same step with a new postfix); repeats
  are legitimate observations, not errors.
- A step LOWER than the previous one means a new epoch/run started (Lightning
  counts within-epoch; the demo source loops): discard prior observations and
  start over — a rate computed across the reset would be negative garbage.
- eta_seconds = (total - current) / rate; None when the rate is None or <= 0
  (a stalled run has no finite ETA).
"""

from __future__ import annotations

from collections import deque


class EtaEstimator:
    def __init__(self, window: int = 50) -> None:
        self.window = window
        self._obs: deque[tuple[float, int]] = deque(maxlen=window)

    def observe(self, t: float, step: int) -> None:
        """Record one (monotonic seconds, step) observation."""
        raise NotImplementedError

    def steps_per_sec(self) -> float | None:
        """Rolling throughput over the window; None if not yet computable."""
        raise NotImplementedError

    def eta_seconds(self, current_step: int, total_steps: int) -> float | None:
        """Estimated seconds until total_steps; None when unknowable."""
        raise NotImplementedError


def format_eta(seconds: float | None) -> str:
    """Human ETA: None → "—", 42 → "~42s", 872 → "~14m 32s", 3725 → "~1h 2m".

    Under a minute: seconds only. Under an hour: minutes + seconds. An hour
    or more: hours + minutes (seconds are noise at that horizon).
    """
    raise NotImplementedError
