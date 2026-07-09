"""Per-GPU status panel widget.

Renders one GPU's live stats — utilization, VRAM, temperature, power — as a
two-line bordered box:

    ┌─ GPU 0 · NVIDIA A100 80GB ────────────────┐
    │ util  94% ██████████████████░░  temp 71°C │
    │ vram 74.2/80.0 GiB (93%)      power 312W  │
    └───────────────────────────────────────────┘

The app owns polling and assigns `panel.sample = <GpuSample>` each tick; the
`reactive` attribute triggers `watch_sample`, which redraws the widget.
"""

from __future__ import annotations

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

from torchwatch.alerts import TEMP_ALERT_C, TEMP_WARN_C, VRAM_ALERT_PCT, VRAM_WARN_PCT
from torchwatch.collector.nvidia import GiB, GpuSample


def pressure_color(
    value: float, warn: float = VRAM_WARN_PCT, alert: float = VRAM_ALERT_PCT
) -> str:
    """Map a value to a Rich color name: green below `warn`, yellow from
    `warn` to `alert`, red at `alert` and above. Defaults are the VRAM
    percentage bands; temperature passes its own (TEMP_WARN_C/TEMP_ALERT_C)."""
    if value < warn:
        return "green"
    elif warn <= value < alert:
        return "yellow"
    else:
        return "red"


def bar(pct: float | None, width: int = 20) -> str:
    """Render a fixed-width horizontal gauge like "██████████░░░░░░░░░░"
    for a 0-100 percentage. None (metric unsupported) gives an empty bar."""
    if pct is None:
        return "░" * width

    filled_cells = min(round(pct / 100 * width), width)
    unfilled_cells = width - filled_cells
    return "█" * filled_cells + "░" * unfilled_cells


class GpuPanel(Static):
    """Displays one GPU's live stats. The app owns polling, this only renders."""

    # Assigning `panel.sample = <GpuSample>` triggers watch_sample() below.
    sample: reactive[GpuSample | None] = reactive(None)

    def watch_sample(self, sample: GpuSample | None) -> None:
        """Redraw for a fresh sample; no-op until the first poll arrives.

        utilization_pct / temperature_c / power_w may each be None (NVML
        reports "not supported" on some GPUs) and render as "--".
        """
        if sample is None:
            return

        self.border_title = f"GPU {sample.index} · {sample.name}"

        text = Text()

        text.append("util ")

        if sample.utilization_pct is None:  # NVML: "not supported" on this GPU
            text.append("--  ")
            text.append(bar(None))
        else:
            util_pct = round(sample.utilization_pct)
            if util_pct < 10:
                text.append(f"{util_pct}%   ")
            elif util_pct < 100:
                text.append(f"{util_pct}%  ")
            else:
                text.append(f"{util_pct}% ")
            text.append(bar(util_pct), style=pressure_color(util_pct))

        text.append(" ")

        text.append("temp ")
        temp_c = sample.temperature_c

        if temp_c is None:
            text.append("--  ")
        else:
            text.append(
                f"{temp_c}°C ",
                style=pressure_color(temp_c, warn=TEMP_WARN_C, alert=TEMP_ALERT_C),
            )

        text.append("\n")

        text.append("vram ")
        vram_tot = round(sample.vram_total_bytes / GiB, 1)
        vram_use = round(sample.vram_used_bytes / GiB, 1)
        vram_pct = sample.vram_pct

        text.append(
            f"{vram_use}/{vram_tot} GiB ({round(vram_pct, 1)}%) ",
            style=pressure_color(vram_pct),
        )

        text.append("power ")
        power_w = sample.power_w

        if power_w is None:
            text.append("-- ")
        else:
            text.append(f"{power_w} W ")

        self.update(text)
