"""Alert rules and the AlertLog that decides what is currently shown.

Pure logic over plain values — no widgets, no threads — so every rule is
unit-testable. Thresholds live here because they ARE the alert rules; the
widgets' colors follow them.
"""

from __future__ import annotations

from dataclasses import dataclass

VRAM_WARN_PCT = 85.0
VRAM_ALERT_PCT = 95.0

# Temperature bands sit higher than the VRAM ones: GPUs run hot by design,
# slowdown-throttling starts around the low 90s, and ~100°C is shutdown
# territory — warning at 85 would cry wolf on every healthy training run.
TEMP_WARN_C = 90.0
TEMP_ALERT_C = 100.0


@dataclass(frozen=True)
class Alert:
    """One active alert. `key` identifies the condition (e.g. "vram:0"),
    so a condition persisting across ticks is one alert, not a stack of
    duplicates."""

    key: str
    message: str
    first_seen: float
    last_seen: float


class AlertLog:
    """Collects rule firings and decides what is currently worth showing.

    An alert lingers for `ttl_s` seconds after its condition was last
    reported, so a one-tick spike doesn't flash faster than a human can
    read. Methods take `now` as a parameter — the clock is a system
    boundary — so tests can drive time by hand.
    """

    def __init__(self, ttl_s: float = 10.0) -> None:
        self.ttl_s = ttl_s
        self._alerts: dict[str, Alert] = {}

    def report(self, key: str, message: str, now: float) -> None:
        """Record that `key`'s condition is true at `now`: new (and expired)
        keys start a fresh entry; live ones refresh last_seen and the
        message but keep first_seen."""

        old = self._alerts.get(key)
        if old is not None and (now - old.last_seen) <= self.ttl_s:
            self._alerts[key] = Alert(
                key=key,
                message=message,
                first_seen=old.first_seen,
                last_seen=now,
            )
        else:
            self._alerts[key] = Alert(
                key=key,
                message=message,
                first_seen=now,
                last_seen=now,
            )

    def active(self, now: float) -> list[Alert]:
        """Alerts with `now - last_seen <= ttl_s`, oldest first — ordering
        by first_seen keeps the display stable as alerts re-fire."""
        active_alerts = []

        for alert in self._alerts.values():
            if now - alert.last_seen <= self.ttl_s:
                active_alerts.append(alert)
        
        active_alerts.sort(key=lambda x: x.first_seen)

        return active_alerts

def vram_suggestion(vram_pct: float) -> str | None:
    """An actionable hint once VRAM crosses VRAM_ALERT_PCT; None below it."""
    if vram_pct < VRAM_ALERT_PCT:
        return None
    else:
        return f"vram usage of {round(vram_pct, 1)}% >= alert threshold of {VRAM_ALERT_PCT}%. recommend trying a smaller batch size, mixed precision (amp/bf16), or gradient checkpointing"


def temp_warning(temp_c: float | None) -> str | None:
    """An actionable hint once temperature reaches TEMP_ALERT_C; None
    below it or when the sensor is unreadable (temp_c None)."""
    if temp_c is None or temp_c < TEMP_ALERT_C:
        return None
    return (
        f"temperature of {round(temp_c, 1)}°C >= alert threshold of {TEMP_ALERT_C}°C. "
        "gpu is throttling or near shutdown — check chassis airflow/dust, "
        "fan curve, and ambient temperature"
    )

def is_stalled(losses: list[float], window: int = 100, threshold: float = 0.001) -> bool:
    """True when the relative range of the last `window` losses —
    (max - min) / (max + 1e-8) — is below `threshold`; fewer than
    `window` losses → False."""
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
    """True when the latest loss exceeds `multiplier`× the mean of the
    previous `window - 1` losses; fewer than `window` losses → False."""
    if len(losses) < window:
        return False                # ignore startup noise
    
    rolling_mean = sum(losses[-window:-1])/(window-1)

    if losses[-1] > rolling_mean * multiplier:
        return True                 # conclude spiking currently
    else:
        return False
