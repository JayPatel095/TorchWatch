"""One-line training progress summary: step, throughput, ETA."""

from __future__ import annotations

from typing import Any

from textual.widgets import Static


class EtaBar(Static):
    """Renders "elapsed X · step A/B · R it/s · ETA Y" from pre-formatted parts."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.border_title = "progress"

    def update_metrics(
        self,
        step: int | None,
        total_steps: int | None,
        rate: float | None,
        elapsed_text: str,
        eta_text: str,
    ) -> None:
        """Rebuild the progress line; None/absent parts render as "—"."""
        if step is not None and total_steps:
            step_txt = f"step {step}/{total_steps}"
        elif step is not None:
            step_txt = f"step {step}"
        else:
            step_txt = "step —"
        rate_txt = f"{rate:.1f} it/s" if rate else "— it/s"
        self.update(f"elapsed {elapsed_text} · {step_txt} · {rate_txt} · ETA {eta_text}")
