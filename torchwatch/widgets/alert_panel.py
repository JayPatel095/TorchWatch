"""The dedicated alerts area: one line per currently-active alert.

The app owns the AlertLog and calls `show_alerts(log.active(now))` each
tick. The panel hides itself entirely when there is nothing to say —
healthy runs shouldn't reserve space for an empty box.
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.widgets import Static

from torchwatch.alerts import Alert


class AlertPanel(Static):
    """Renders active alerts; collapsed entirely while there are none."""

    def __init__(self, **kwargs: Any) -> None:
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
