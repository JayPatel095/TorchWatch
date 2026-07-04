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

from textual.app import App, ComposeResult
from textual.containers import Grid
from textual.css.query import NoMatches
from textual.widgets import Footer, Header
from textual.worker import get_current_worker

from torchwatch.collector.nvidia import Collector, GpuSample, create_collector
from torchwatch.widgets.gpu_panel import GpuPanel


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

    def compose(self) -> ComposeResult:
        """Header bar, GPU grid, keybinding footer."""
        yield Header(show_clock=True)
        yield Grid(id="gpu-grid")
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

        self._base_sub_title = " · ".join(parts)
        self.sub_title = self._base_sub_title

        self.run_worker(self._poll_loop, thread=True, exclusive=True)

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
        grid = self.query_one("#gpu-grid", Grid)

        for sample in samples:
            try:
                panel = grid.query_one(f"#gpu-{sample.index}", GpuPanel)
            except NoMatches:
                panel = GpuPanel(id=f"gpu-{sample.index}")
                grid.mount(panel)

            panel.sample = sample

    def action_toggle_pause(self) -> None:
        """Bound to `p`: toggle polling and the PAUSED marker in the header."""
        self.paused = not self.paused
        self.sub_title = self._base_sub_title + (" · PAUSED" if self.paused else "")