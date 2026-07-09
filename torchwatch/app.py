"""The torchwatch Textual application.

Polls the GPU collector on a background thread and renders one GpuPanel
per device in a live grid.

Architecture — one rule matters here:

    The TUI runs in an async event loop on the main thread. pynvml calls
    BLOCK, so polling happens in a background *thread* worker. Widgets
    are NOT thread-safe, so the worker never touches them directly —
    it hands samples back to the UI thread via `self.call_from_thread`.

        [thread worker]  _poll_loop:  collector.sample() every poll_ms
               │
               │  self.call_from_thread(self._apply_samples, samples)
               ▼
        [UI thread]      _apply_samples:  mount/update GpuPanel widgets
"""

from __future__ import annotations

import time
from collections import deque
from typing import Protocol

from textual.app import App, ComposeResult
from textual.containers import Grid
from textual.css.query import NoMatches
from textual.widgets import Footer, Header
from textual.worker import get_current_worker

from torchwatch.alerts import AlertLog, is_spiking, is_stalled, temp_warning, vram_suggestion
from torchwatch.collector.nvidia import Collector, GpuSample, create_collector
from torchwatch.collector.stdout import TrainingUpdate
from torchwatch.eta import EtaEstimator, format_time
from torchwatch.widgets.alert_panel import AlertPanel
from torchwatch.widgets.eta_bar import EtaBar
from torchwatch.widgets.gpu_panel import GpuPanel
from torchwatch.widgets.sparkline import LossSparkline


class MetricsSource(Protocol):
    """Anything that yields TrainingUpdates: the demo source now, the live
    stdout reader later. `label` is shown in the header so the user always
    knows where the numbers come from."""

    label: str
    interval_s: float

    def next_update(self) -> TrainingUpdate | None: ...


class TorchwatchApp(App[None]):
    TITLE = "torchwatch"

    CSS = """
    #gpu-grid {
        layout: grid;
        grid-size: 2;
        grid-rows: auto;
        grid-gutter: 1 2;
        padding: 1 2;
        height: auto;
    }
    GpuPanel {
        height: auto;
        border: round $secondary;
        padding: 0 1;
    }
    #loss-spark, #eta-bar {
        height: auto;
        border: round $secondary;
        margin: 0 2;
        padding: 0 1;
    }
    #alert-panel {
        height: auto;
        border: round $error;
        margin: 0 2;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "quit"),
        ("p", "toggle_pause", "pause"),
    ]

    def __init__(
        self,
        poll_ms: int = 500,
        pid: int | None = None,
        collector: Collector | None = None,
        metrics_source: MetricsSource | None = None,
    ) -> None:
        super().__init__()
        # Tests inject a MockCollector; normal runs auto-detect.
        if collector is None:
            collector, fallback_reason = create_collector()
        else:
            fallback_reason = None
        self._base_sub_title = ""
        self.collector: Collector = collector
        self.fallback_reason = fallback_reason
        self.poll_ms = poll_ms
        self.watched_pid = pid
        self.paused = False
        self.metrics_source = metrics_source
        self._eta = EtaEstimator()
        self._start_time = time.time()
        self._alert_log = AlertLog()
        # Loss history for the stall/spike rules; 120 covers the stall
        # window (100) with headroom, matching the sparkline's window.
        self._losses: deque[float] = deque(maxlen=120)

    def compose(self) -> ComposeResult:
        """Header bar, GPU grid, metrics panel, keybinding footer."""
        yield Header(show_clock=True)
        yield AlertPanel(id="alert-panel")
        yield Grid(id="gpu-grid")
        yield LossSparkline(id="loss-spark")
        yield EtaBar(id="eta-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Set the header context line and start the polling worker.

        The sub_title shows the watched pid (if any) and a mock-data note
        when NVML is unavailable, so fake numbers are never mistaken for a
        real GPU.
        """
        parts = []
        if self.watched_pid is not None:
            parts.append(f"pid {self.watched_pid}")
        if self.fallback_reason is not None:
            parts.append(f"mock data (NVML unavailable: {self.fallback_reason})")
        if self.metrics_source is not None:
            parts.append(self.metrics_source.label)

        self._base_sub_title = " · ".join(parts)
        self.sub_title = self._base_sub_title

        # exclusive= is per-group; without distinct groups the second worker
        # would cancel the first.
        self.run_worker(self._poll_loop, thread=True, exclusive=True, group="gpu-poll")
        if self.metrics_source is not None:
            self.run_worker(self._metrics_loop, thread=True, exclusive=True, group="metrics")

    def _poll_loop(self) -> None:
        """Background thread: sample every poll_ms and hand off to the UI.

        Exits when the worker is cancelled (quit) or the app shuts down
        mid-handoff (call_from_thread raises RuntimeError).
        """
        worker = get_current_worker()

        while not worker.is_cancelled:
            if not self.paused:
                samples = self.collector.sample()
                try:
                    self.call_from_thread(self._apply_samples, samples)
                except RuntimeError:
                    break

            time.sleep(self.poll_ms/1000)

    def _apply_samples(self, samples: list[GpuSample]) -> None:
        """UI thread: create or update one GpuPanel per sample."""
        try:
            grid = self.query_one("#gpu-grid", Grid)
        except NoMatches:
            return

        now = time.monotonic()
        for sample in samples:
            try:
                panel = grid.query_one(f"#gpu-{sample.index}", GpuPanel)
            except NoMatches:
                panel = GpuPanel(id=f"gpu-{sample.index}")
                grid.mount(panel)

            panel.sample = sample

            suggestion = vram_suggestion(sample.vram_pct)
            if suggestion is not None:
                self._alert_log.report(
                    f"vram:{sample.index}", f"gpu {sample.index}: {suggestion}", now
                )

            warning = temp_warning(sample.temperature_c)
            if warning is not None:
                self._alert_log.report(
                    f"temp:{sample.index}", f"gpu {sample.index}: {warning}", now
                )

        # Runs every poll tick, so it also expires alerts whose rules have
        # gone quiet — the metrics loop alone couldn't clear a vram alert.
        self._refresh_alerts(now)

    def _metrics_loop(self) -> None:
        """Background thread: pull TrainingUpdates and hand off to the UI.

        Same lifecycle rules as _poll_loop: exit on worker cancellation or
        on RuntimeError from call_from_thread during shutdown.
        """
        worker = get_current_worker()

        while not worker.is_cancelled:
            if not self.paused:
                update = self.metrics_source.next_update()
                if update is not None:
                    try:
                        self.call_from_thread(self._apply_update, update)
                    except RuntimeError:
                        break
            time.sleep(self.metrics_source.interval_s)

    def _apply_update(self, update: TrainingUpdate) -> None:
        """UI thread: feed one TrainingUpdate to the sparkline and ETA bar."""
        try:
            spark_widget = self.query_one("#loss-spark", LossSparkline)
            eta_bar = self.query_one("#eta-bar", EtaBar)
        except NoMatches:  # tick landed before the widgets mounted
            return

        now = time.monotonic()
        if update.loss is not None:
            spark_widget.push(update.loss)
            self._losses.append(update.loss)
            losses = list(self._losses)
            if is_spiking(losses):
                self._alert_log.report(
                    "loss:spike",
                    f"loss spiked to {update.loss:.4g} — over 2x the recent average",
                    now,
                )
            if is_stalled(losses):
                self._alert_log.report(
                    "loss:stall",
                    "loss has stalled: no meaningful change over the last 100 updates",
                    now,
                )
            self._refresh_alerts(now)
        if update.step is not None:
            self._eta.observe(now, update.step)

        eta = None
        if update.step is not None and update.total_steps:
            eta = self._eta.eta_seconds(update.step, update.total_steps)
        f_eta = format_time(eta) if eta is None else "~" + format_time(eta)

        elapsed = time.time() - self._start_time
        f_elapsed = format_time(elapsed)

        eta_bar.update_metrics(
            update.step, update.total_steps, self._eta.steps_per_sec(), f_elapsed, f_eta
        )

    def _refresh_alerts(self, now: float) -> None:
        """UI thread: sync the alerts area with what the log says is live."""
        try:
            panel = self.query_one("#alert-panel", AlertPanel)
        except NoMatches:
            return
        panel.show_alerts(self._alert_log.active(now))

    def action_toggle_pause(self) -> None:
        """Bound to `p`: toggle polling and the PAUSED marker in the header."""
        self.paused = not self.paused
        self.sub_title = self._base_sub_title + (" · PAUSED" if self.paused else "")