"""Alert logic: VRAM pressure suggestions and loss stall/spike detection.

Pure functions over plain values — no widgets, no threads — so every rule
is unit-testable. The app wires results into the UI: the GPU panel shows
the VRAM suggestion, the metrics panel flags stalls and spikes.

Thresholds live here (not in the widgets) because they ARE the alert
rules; the panel's colors follow them.
"""

from __future__ import annotations

from dataclasses import dataclass

WARN_PCT = 85.0
ALERT_PCT = 95.0


@dataclass(frozen=True)
class Alert:
    """One active alert as shown to the user.

    `key` identifies the condition (e.g. "vram:0", "stall"), so a condition
    that persists across ticks is one alert, not a stack of duplicates.
    `first_seen` / `last_seen` are clock readings passed in by the caller.
    """

    key: str
    message: str
    first_seen: float
    last_seen: float


class AlertLog:
    """Collects rule firings and decides what is currently worth showing.

    The rules (above) are stateless per-tick checks; this class adds time.
    An alert stays visible ("lingers") for `ttl_s` seconds after the
    condition was LAST reported, so a one-tick VRAM spike doesn't flash in
    and out faster than a human can read it.

    All methods take `now` as a parameter instead of reading the clock —
    the clock is a system boundary, same as `proc_root` in tail.py, so
    tests can drive time by hand.
    """

    def __init__(self, ttl_s: float = 10.0) -> None:
        self.ttl_s = ttl_s

    def report(self, key: str, message: str, now: float) -> None:
        """Record that `key`'s condition is true at time `now`.

        First report of a key creates the entry (first_seen = now); a
        repeat report refreshes last_seen and replaces the message (the
        numbers inside it may have changed) but keeps first_seen.
        """
        raise NotImplementedError

    def active(self, now: float) -> list[Alert]:
        """Alerts still within their lifespan at `now`, oldest first.

        An alert is active while `now - last_seen <= ttl_s`; expired
        entries are dropped. Ordering by first_seen keeps the display
        stable — entries don't jump around as newer alerts re-fire.
        """
        raise NotImplementedError


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
        return f"vram usage of {round(vram_pct, 1)}% >= alert threshold of {ALERT_PCT}%. recommend trying a smaller batch size, mixed precision (amp/bf16), or gradient checkpointing"

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
    
    rolling_mean = sum(losses[-window:-1])/(window-1)

    if losses[-1] > rolling_mean * multiplier:
        return True                 # conclude spiking currently
    else:
        return False
