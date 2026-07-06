"""M2 acceptance test: panels mount and keep updating while the app runs.

Runs the real app headless via Textual's test harness (`App.run_test`).
When this passes, the milestone's "widget updates live — not just on
startup" requirement is met.
"""

import asyncio

from torchwatch.app import TorchwatchApp
from torchwatch.collector.demo import DemoMetrics
from torchwatch.collector.nvidia import MockCollector
from torchwatch.widgets.eta_bar import EtaBar
from torchwatch.widgets.gpu_panel import GpuPanel
from torchwatch.widgets.sparkline import LossSparkline


def test_panels_mount_and_update_live():
    async def scenario() -> None:
        app = TorchwatchApp(poll_ms=25, collector=MockCollector(count=3, seed=7))
        async with app.run_test(size=(100, 40)) as pilot:
            # Phase 1: all three panels appear and receive a first sample.
            for _ in range(40):
                await pilot.pause(0.05)
                panels = list(app.query(GpuPanel))
                if len(panels) == 3 and all(p.sample is not None for p in panels):
                    break
            else:
                raise AssertionError(
                    "GPU panels never mounted/populated — is the worker running "
                    "and calling _apply_samples?"
                )

            # Phase 2: samples keep changing → polling is live, not one-shot.
            first = [p.sample for p in app.query(GpuPanel)]
            for _ in range(40):
                await pilot.pause(0.05)
                if [p.sample for p in app.query(GpuPanel)] != first:
                    break
            else:
                raise AssertionError(
                    "panel samples never changed after mount — poll loop "
                    "appears to have stopped after one tick"
                )

            # Phase 3: `p` pauses polling — samples freeze until pressed again.
            await pilot.press("p")
            await pilot.pause(0.2)  # let any in-flight tick land
            frozen = [p.sample for p in app.query(GpuPanel)]
            await pilot.pause(0.3)  # ~12 poll intervals
            assert [p.sample for p in app.query(GpuPanel)] == frozen, (
                "samples kept changing while paused — does action_toggle_pause "
                "flip self.paused?"
            )

            await pilot.press("p")
            for _ in range(40):
                await pilot.pause(0.05)
                if [p.sample for p in app.query(GpuPanel)] != frozen:
                    break
            else:
                raise AssertionError("samples never resumed after unpausing")

    asyncio.run(scenario())


def test_metrics_pipeline_reaches_widgets():
    """With a metrics source attached, loss values must reach the sparkline
    and the ETA bar must render a progress line — end to end through the
    metrics worker, _apply_update, and the widgets."""

    async def scenario() -> None:
        app = TorchwatchApp(
            poll_ms=200,
            collector=MockCollector(count=1, seed=3),
            metrics_source=DemoMetrics(total_steps=200, seed=5),
        )
        async with app.run_test(size=(100, 40)) as pilot:
            for _ in range(60):
                await pilot.pause(0.05)
                spark_widget = app.query_one("#loss-spark", LossSparkline)
                if len(spark_widget._values) >= 3:
                    break
            else:
                raise AssertionError("no loss values reached the sparkline widget")
            bar_text = str(app.query_one("#eta-bar", EtaBar).render())
            assert "step " in bar_text and "ETA" in bar_text

    asyncio.run(scenario())
