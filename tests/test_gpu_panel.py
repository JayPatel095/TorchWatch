"""GpuPanel rendering, including the None-field cases MockCollector never emits."""

import asyncio

from textual.app import App

from torchwatch.collector.nvidia import GiB, GpuSample
from torchwatch.widgets.gpu_panel import GpuPanel


class PanelHarness(App[None]):
    def compose(self):
        yield GpuPanel(id="gpu-0")


def test_panel_renders_sample_with_unsupported_fields():
    """NVML reports util/temp/power as 'not supported' on some GPUs — the
    collector maps those to None, and the panel must render '--', not crash."""

    async def scenario() -> None:
        app = PanelHarness()
        async with app.run_test(size=(120, 20)) as pilot:
            panel = app.query_one(GpuPanel)
            panel.sample = GpuSample(
                index=0,
                name="Odd GPU",
                utilization_pct=None,
                vram_used_bytes=10 * GiB,
                vram_total_bytes=80 * GiB,
                temperature_c=None,
                power_w=None,
            )
            await pilot.pause()
            rendered = str(panel.render())
            assert "--" in rendered  # placeholders, not a dead panel
            assert "12" in rendered  # vram pct still rendered (10/80 → 12%)

    asyncio.run(scenario())
