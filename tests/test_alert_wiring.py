"""Alerts must flow from the rules through the AlertLog into the alerts area."""

import asyncio

from textual.app import App

from torchwatch.alerts import Alert
from torchwatch.app import TorchwatchApp
from torchwatch.collector.nvidia import GiB, GpuSample
from torchwatch.collector.stdout import TrainingUpdate
from torchwatch.alerts import TEMP_ALERT_C, TEMP_WARN_C
from torchwatch.widgets.alert_panel import AlertPanel
from torchwatch.widgets.gpu_panel import GpuPanel, pressure_color


def test_temperature_uses_its_own_color_bands():
    # regression: temp was passed as temp/100 (or vram bands) — a 96°C GPU
    # rendered green. With temperature bands, hot must actually look hot.
    assert pressure_color(72, warn=TEMP_WARN_C, alert=TEMP_ALERT_C) == "green"
    assert pressure_color(93, warn=TEMP_WARN_C, alert=TEMP_ALERT_C) == "yellow"
    assert pressure_color(101, warn=TEMP_WARN_C, alert=TEMP_ALERT_C) == "red"


def _sample(vram_pct: float, temp_c: int | None = 60) -> GpuSample:
    total = 80 * GiB
    return GpuSample(
        index=0,
        name="Fake A100",
        utilization_pct=50,
        vram_used_bytes=int(total * vram_pct / 100),
        vram_total_bytes=total,
        temperature_c=temp_c,
        power_w=250.0,
    )


class FakeCollector:
    """Always reports the same sample, so alert conditions are deterministic."""

    is_mock = True

    def __init__(self, vram_pct: float, temp_c: int | None = 60) -> None:
        self._vram_pct = vram_pct
        self._temp_c = temp_c

    def gpu_count(self) -> int:
        return 1

    def sample(self) -> list[GpuSample]:
        return [_sample(self._vram_pct, self._temp_c)]


class ScriptedSource:
    """Replays a fixed list of losses, then goes quiet."""

    label = "scripted"
    interval_s = 0.01

    def __init__(self, losses: list[float]) -> None:
        self._updates = [
            TrainingUpdate(step=i + 1, total_steps=len(losses), loss=loss, format_name="plain")
            for i, loss in enumerate(losses)
        ]

    def next_update(self) -> TrainingUpdate | None:
        return self._updates.pop(0) if self._updates else None

    def close(self) -> None:
        pass


class PanelHarness(App[None]):
    def compose(self):
        yield AlertPanel(id="alert-panel")


def test_alert_panel_hidden_when_empty():
    async def scenario() -> None:
        app = PanelHarness()
        async with app.run_test(size=(120, 20)) as pilot:
            panel = app.query_one(AlertPanel)
            assert panel.display is False  # collapsed from the start

            panel.show_alerts(
                [Alert(key="k", message="something bad", first_seen=0.0, last_seen=0.0)]
            )
            await pilot.pause()
            assert panel.display is True
            assert "something bad" in str(panel.render())

            panel.show_alerts([])
            await pilot.pause()
            assert panel.display is False

    asyncio.run(scenario())


async def _wait_for_alert(app: TorchwatchApp, pilot, needle: str) -> str:
    for _ in range(60):
        await pilot.pause(0.05)
        panel = app.query_one("#alert-panel", AlertPanel)
        rendered = str(panel.render())
        if panel.display and needle in rendered.lower():
            return rendered
    raise AssertionError(f"no active alert containing {needle!r} ever appeared")


def test_vram_alert_reaches_alert_panel():
    async def scenario() -> None:
        app = TorchwatchApp(poll_ms=25, collector=FakeCollector(vram_pct=97.0))
        async with app.run_test(size=(120, 30)) as pilot:
            rendered = await _wait_for_alert(app, pilot, "batch")
            assert "vram" in rendered.lower()
            assert "gpu 0" in rendered.lower()  # says WHICH gpu is under pressure
            # the suggestion lives in the alerts area now, not the GPU panel
            gpu_text = str(app.query_one(GpuPanel).render())
            assert "batch" not in gpu_text.lower()

    asyncio.run(scenario())


def test_temp_alert_reaches_alert_panel():
    async def scenario() -> None:
        app = TorchwatchApp(poll_ms=25, collector=FakeCollector(vram_pct=50.0, temp_c=101))
        async with app.run_test(size=(120, 30)) as pilot:
            rendered = await _wait_for_alert(app, pilot, "temperature")
            assert "gpu 0" in rendered.lower()
            assert "airflow" in rendered.lower()

    asyncio.run(scenario())


def test_healthy_vram_raises_no_alert():
    async def scenario() -> None:
        app = TorchwatchApp(poll_ms=25, collector=FakeCollector(vram_pct=60.0))
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause(0.5)
            assert app.query_one("#alert-panel", AlertPanel).display is False

    asyncio.run(scenario())


def test_loss_spike_reaches_alert_panel():
    async def scenario() -> None:
        app = TorchwatchApp(
            poll_ms=200,
            collector=FakeCollector(vram_pct=50.0),
            metrics_source=ScriptedSource([1.0] * 19 + [2.5]),
        )
        async with app.run_test(size=(120, 30)) as pilot:
            await _wait_for_alert(app, pilot, "spike")

    asyncio.run(scenario())


def test_loss_stall_reaches_alert_panel():
    async def scenario() -> None:
        app = TorchwatchApp(
            poll_ms=200,
            collector=FakeCollector(vram_pct=50.0),
            metrics_source=ScriptedSource([0.5] * 100),
        )
        async with app.run_test(size=(120, 30)) as pilot:
            await _wait_for_alert(app, pilot, "stall")

    asyncio.run(scenario())
