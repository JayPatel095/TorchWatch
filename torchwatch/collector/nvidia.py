"""pynvml wrapper with graceful fallback when NVML is unavailable.

`create_collector()` is the entry point: it returns a live `NvmlCollector`
when NVML works, otherwise a `MockCollector` plus the reason for the
fallback, so the UI can tell the user it is showing fake data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")

GiB = 1024**3


@dataclass(frozen=True)
class GpuSample:
    index: int
    name: str
    utilization_pct: int | None
    vram_used_bytes: int
    vram_total_bytes: int
    temperature_c: int | None
    power_w: float | None

    @property
    def vram_pct(self) -> float:
        if self.vram_total_bytes == 0:
            return 0.0
        return 100.0 * self.vram_used_bytes / self.vram_total_bytes


class NvmlCollector:
    """Live GPU stats via NVML. Raises on construction if NVML is unusable."""

    is_mock = False

    def __init__(self) -> None:
        import pynvml

        pynvml.nvmlInit()
        self._nv = pynvml

    def gpu_count(self) -> int:
        return int(self._nv.nvmlDeviceGetCount())

    def sample(self) -> list[GpuSample]:
        samples = []
        for i in range(self.gpu_count()):
            handle = self._nv.nvmlDeviceGetHandleByIndex(i)
            name = self._nv.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):  # pynvml < 11.5 returns bytes
                name = name.decode(errors="replace")
            mem = self._nv.nvmlDeviceGetMemoryInfo(handle)
            util = self._maybe(lambda: int(self._nv.nvmlDeviceGetUtilizationRates(handle).gpu))
            temp = self._maybe(
                lambda: int(
                    self._nv.nvmlDeviceGetTemperature(handle, self._nv.NVML_TEMPERATURE_GPU)
                )
            )
            power = self._maybe(lambda: self._nv.nvmlDeviceGetPowerUsage(handle) / 1000.0)
            samples.append(
                GpuSample(
                    index=i,
                    name=name,
                    utilization_pct=util,
                    vram_used_bytes=int(mem.used),
                    vram_total_bytes=int(mem.total),
                    temperature_c=temp,
                    power_w=power,
                )
            )
        return samples

    def _maybe(self, fn: Callable[[], T]) -> T | None:
        """Some fields (power, temp) are unsupported on some GPUs; treat as absent."""
        try:
            return fn()
        except self._nv.NVMLError:
            return None

    def close(self) -> None:
        try:
            self._nv.nvmlShutdown()
        except Exception:
            pass


class MockCollector:
    """Fake wandering GPU data so the TUI is usable on machines without NVIDIA GPUs."""

    is_mock = True

    def __init__(self, count: int = 2, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._count = count
        self._util = [self._rng.uniform(80, 95) for _ in range(count)]
        self._vram = [self._rng.uniform(0.65, 0.75) for _ in range(count)]  # fraction of total
        self._total = 80 * GiB

    def gpu_count(self) -> int:
        return self._count

    def sample(self) -> list[GpuSample]:
        samples = []
        for i in range(self._count):
            self._util[i] = min(100.0, max(55.0, self._util[i] + self._rng.uniform(-4, 4)))
            self._vram[i] = min(0.97, max(0.5, self._vram[i] + self._rng.uniform(-0.004, 0.005)))
            util = round(self._util[i])
            samples.append(
                GpuSample(
                    index=i,
                    name="Mock A100 80GB",
                    utilization_pct=util,
                    vram_used_bytes=int(self._vram[i] * self._total),
                    vram_total_bytes=self._total,
                    temperature_c=round(55 + util * 0.25 + self._rng.uniform(-1, 1)),
                    power_w=round(150 + util * 1.8 + self._rng.uniform(-10, 10), 1),
                )
            )
        return samples

    def close(self) -> None:
        pass


Collector = NvmlCollector | MockCollector


def create_collector() -> tuple[Collector, str | None]:
    """Return (collector, fallback_reason). reason is None when NVML is live."""
    try:
        return NvmlCollector(), None
    except ImportError:
        reason = "pynvml is not installed"
    except Exception as exc:  # NVMLError_DriverNotLoaded, LibraryNotFound, ...
        reason = str(exc) or type(exc).__name__
    return MockCollector(), reason
