"""Scrolling loss sparkline.

The pure mapping lives in `spark()` (unit-tested, no Textual involved);
`LossSparkline` is the thin widget wrapper that owns the value window.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from textual.widgets import Static

BLOCKS = "▁▂▃▄▅▆▇█"


def spark(values: list[float], width: int = 40) -> str:
    """Map the last `width` values to one block char each.

    Contract:
    - Empty input → "".
    - Scale linearly within the visible window: lo→BLOCKS[0], hi→BLOCKS[7],
      level = round((v - lo) / (hi - lo) * 7).
    - A flat window (hi == lo) renders mid-blocks: BLOCKS[3] for every value.
    - Only the most recent `width` values are drawn; older ones scroll off.
    """
    if not values:
        return ""
    
    n = len(values)

    if n > width:
        latest_values = values[-width:]
    else:
        latest_values = values[:]

    lo = min(latest_values)
    hi = max(latest_values)
    
    if hi == lo:
        return len(latest_values) * BLOCKS[3]
    
    text = []
    
    for val in latest_values:
        level = round((val - lo) / (hi - lo) * 7)
        text.append(BLOCKS[level])

    return "".join(text)


class LossSparkline(Static):
    """Sparkline of recent loss values with the latest value as a label."""

    def __init__(self, window: int = 120, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.border_title = "loss"
        self._values: deque[float] = deque(maxlen=window)

    def push(self, loss: float) -> None:
        """Append one loss value and redraw."""
        self._values.append(loss)
        self.update(f"{spark(list(self._values))}  {self._values[-1]:.4f}")
