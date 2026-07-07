"""Alert logic: VRAM pressure suggestions and loss stall/spike detection.

Pure functions over plain values — no widgets, no threads — so every rule
is unit-testable. The app wires results into the UI: the GPU panel shows
the VRAM suggestion, the metrics panel flags stalls and spikes.

Thresholds live here (not in the widgets) because they ARE the alert
rules; the panel's colors follow them.
"""

from __future__ import annotations

WARN_PCT = 85.0
ALERT_PCT = 95.0


def vram_suggestion(vram_pct: float) -> str | None:
    """An actionable hint once VRAM crosses ALERT_PCT; None below it.

    The brief's three escape hatches, in rising order of effort: reduce
    batch size; mixed precision (torch.cuda.amp, bf16/fp16); gradient
    checkpointing (torch.utils.checkpoint). One short string naming them —
    it renders as a single line inside a GPU panel.
    """
    if vram_pct < ALERT_PCT:
        return None
    else:
        return f"vram usage of {vram_pct}% >= alert threshold of {ALERT_PCT}%. recommend trying a smaller batch size, mixed precision (amp/bf16), or gradient checkpointing"

def is_stalled(losses: list[float], window: int = 100, threshold: float = 0.001) -> bool:
    """True when loss has not meaningfully improved across the last `window`.

    Rule: relative range over the window — (max - min) / (max + 1e-8) —
    is below `threshold`. Fewer than `window` losses → False (not enough
    evidence to accuse a run of stalling).
    """
    if len(losses) < window:
        return False                # ignore startup noise
    
    windowed_losses = losses[-window:]
    max_loss = max(windowed_losses)
    min_loss = min(windowed_losses)

    rel_range = (max_loss - min_loss) / (max_loss + 1e-8)

    if rel_range < threshold:       # conclude insufficient loss improvement
        return True
    else:
        return False

def is_spiking(losses: list[float], window: int = 20, multiplier: float = 2.0) -> bool:
    """True when the latest loss jumped above `multiplier`x the recent mean.

    Rule: mean of the previous `window - 1` losses (excluding the latest);
    latest > multiplier x that mean → spike. Fewer than `window` losses
    → False.
    """
    if len(losses) < window:
        return False                # ignore startup noise
    
    rolling_mean = sum(losses[-window-1:-1])/(window-1)

    if losses[-1] > rolling_mean * multiplier:
        return True                 # conclude spiking currently
    else:
        return False

    