"""The dedicated alerts area.

One bordered box listing every currently-active alert, one per line. The
app owns the AlertLog and calls `show_alerts(log.active(now))` on every
tick from either worker, so entries appear when a rule fires, linger for
the log's ttl, and drop out when they expire.

The whole panel hides itself when there is nothing to say — most of the
time training is healthy and the dashboard shouldn't reserve space for
an empty box.
"""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from torchwatch.alerts import Alert


class AlertPanel(Static):
    """Renders active alerts; collapsed entirely while there are none."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.border_title = "alerts"
        self.display = False

    def show_alerts(self, alerts: list[Alert]) -> None:
        """Replace the panel contents with `alerts`, hiding when empty."""
        self.display = bool(alerts)
        if not alerts:
            self.update("")
            return

        text = Text()
        for i, alert in enumerate(alerts):
            if i:
                text.append("\n")
            text.append("⚠ ", style="bold red")
            text.append(alert.message, style="red")
        self.update(text)
