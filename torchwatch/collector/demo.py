"""Synthetic metrics source powering `torchwatch --demo`.

Emits the same TrainingUpdate stream the live readers do, so the whole
metrics pipeline can be exercised (and screen-recorded) anywhere. The run
loops on completion, which also exercises the ETA step-regression reset.
"""

from __future__ import annotations

import math
import random

from torchwatch.collector.stdout import TrainingUpdate


class DemoMetrics:
    """A plausible decaying-loss training run with no process attached."""

    label = "demo metrics (synthetic)"
    interval_s = 0.1

    def __init__(self, total_steps: int = 1000, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._total = total_steps
        self._step = 0

    def next_update(self) -> TrainingUpdate:
        """Advance one step and return its update; wraps around at the end."""
        self._step += 1
        if self._step > self._total:
            self._step = 1
        base = 2.5 * math.exp(-3.0 * self._step / self._total)
        loss = max(0.01, base + self._rng.gauss(0.0, 0.02))
        return TrainingUpdate(
            step=self._step,
            total_steps=self._total,
            loss=round(loss, 4),
            format_name="demo",
        )

    def close(self) -> None:
        """Nothing to release; exists so the CLI can close any source."""
