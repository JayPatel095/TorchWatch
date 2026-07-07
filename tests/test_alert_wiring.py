"""Alert rules must actually reach the widgets, not just pass unit tests."""

import asyncio

from textual.app import App

from torchwatch.collector.nvidia import GiB, GpuSample
from torchwatch.widgets.gpu_panel import GpuPanel
from torchwatch.widgets.sparkline import LossSparkline


def _sample(vram_pct: float) -> GpuSample:
    total = 80 * GiB
    return GpuSample(
        index=0,
        name="Fake A100",
        utilization_pct=50,
        vram_used_bytes=int(total * vram_pct / 100),
        vram_total_bytes=total,
        temperature_c=60,
        power_w=250.0,
    )


class PanelHarness(App[None]):
    def compose(self):
        yield GpuPanel(id="gpu-0")


class SparkHarness(App[None]):
    def compose(self):
        yield LossSparkline(id="loss-spark")


def test_panel_shows_suggestion_above_alert_threshold():
    async def scenario() -> None:
        app = PanelHarness()
        async with app.run_test(size=(120, 20)) as pilot:
            panel = app.query_one(GpuPanel)
            panel.sample = _sample(97.0)
            await pilot.pause()
            rendered = str(panel.render())
            assert "batch" in rendered.lower()

    asyncio.run(scenario())


def test_panel_stays_quiet_below_alert_threshold():
    async def scenario() -> None:
        app = PanelHarness()
        async with app.run_test(size=(120, 20)) as pilot:
            panel = app.query_one(GpuPanel)
            panel.sample = _sample(60.0)
            await pilot.pause()
            rendered = str(panel.render())
            assert "batch" not in rendered.lower()

    asyncio.run(scenario())


def test_sparkline_flags_a_spike():
    async def scenario() -> None:
        app = SparkHarness()
        async with app.run_test(size=(120, 20)) as pilot:
            widget = app.query_one(LossSparkline)
            for loss in [1.0] * 19 + [2.5]:
                widget.push(loss)
            await pilot.pause()
            assert "spike" in str(widget.render())

    asyncio.run(scenario())


def test_sparkline_flags_a_stall():
    async def scenario() -> None:
        app = SparkHarness()
        async with app.run_test(size=(120, 20)) as pilot:
            widget = app.query_one(LossSparkline)
            for _ in range(100):
                widget.push(0.5)
            await pilot.pause()
            assert "stalled" in str(widget.render())

    asyncio.run(scenario())


def test_sparkline_no_flags_on_healthy_loss():
    async def scenario() -> None:
        app = SparkHarness()
        async with app.run_test(size=(120, 20)) as pilot:
            widget = app.query_one(LossSparkline)
            for i in range(100):
                widget.push(2.0 - 0.015 * i)
            await pilot.pause()
            rendered = str(widget.render())
            assert "spike" not in rendered and "stalled" not in rendered

    asyncio.run(scenario())
