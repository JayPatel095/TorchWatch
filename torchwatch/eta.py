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
        if self._obs and self._obs[-1][1] > step:
            self._obs.clear()     # new epoch reset
        self._obs.append((t, step))

    def steps_per_sec(self) -> float | None:
        if len(self._obs) < 2:
            return None
        
        (t0, s0) = self._obs[0] 
        (t1, s1) = self._obs[-1]

        dt = t1 - t0
        if dt <= 0:
            return None
        
        return (s1 - s0) / dt

    def eta_seconds(self, current_step: int, total_steps: int) -> float | None:
        rate = self.steps_per_sec()

        if rate is None or rate <= 0:
            return None           # stalled or failed, ie no finite eta
        
        return (total_steps - current_step) / rate
        

def format_eta(seconds: float | None) -> str:
    """Human ETA: None → "—", 42 → "~42s", 872 → "~14m 32s", 3725 → "~1h 2m".
    """

    if seconds == None:
        return "-"
    
    s = int(seconds)

    if s < 60:
        return f"~{s}s"
    elif s < 3600:
        return f"~{s // 60}m {s % 60}s"
    else:
        return f"~{s // 3600}h {(s % 3600) // 60}m"
