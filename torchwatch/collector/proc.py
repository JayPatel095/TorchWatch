"""PyTorch process discovery: powers `torchwatch list` and pid auto-detection.

The psutil boundary (iter_process_info) is separate from the decision
logic so the latter is testable with fakes. Evidence: a loaded /torch/
library path (maps are empty when unreadable — Linux/permissions), or
torchrun / torch.distributed.run in the cmdline. A bare python process
with NO torch evidence is not a match — false positives cost more trust
than false negatives — and torchwatch itself must never match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from pathlib import Path

import psutil


@dataclass(frozen=True)
class ProcInfo:
    """One process's identity and detection evidence, as plain data."""

    pid: int
    name: str  # executable name, e.g. "python3.12", "torchrun"
    cmdline: tuple[str, ...]
    maps: tuple[str, ...] = ()  # loaded file paths; empty when unreadable


def iter_process_info() -> Iterator[ProcInfo]:
    """psutil adapter: yield ProcInfo for every process we can inspect."""
    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            info = proc.info
            try:
                # Linux-only API: absent from psutil on macOS (the platform-aware
                # stubs flag it here) — the AttributeError below is that case at
                # runtime.
                maps = tuple(
                    m.path for m in proc.memory_maps(grouped=True)  # type: ignore[attr-defined]
                )
            except (psutil.Error, OSError, PermissionError, AttributeError):
                maps = ()  # unreadable (other users' processes) or unsupported OS
            yield ProcInfo(
                pid=info["pid"],
                name=info["name"] or "",
                cmdline=tuple(info["cmdline"] or ()),
                maps=maps,
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue  # process vanished or is fully off-limits mid-iteration


def is_pytorch_process(info: ProcInfo) -> bool:
    """Decide whether one process is a PyTorch training candidate."""
    
    haystack = " ".join((info.name, *info.cmdline))

    if "torchwatch" in haystack:
        return False
    
    if "torchrun" in haystack or "torch.distributed.run" in haystack:
        return True
    
    exe = Path(info.cmdline[0]).name if info.cmdline else ""
    is_python = info.name.startswith("python") or exe.startswith("python")
    has_torch = any("/torch/" in path for path in info.maps)

    if is_python and has_torch:
        return True

    return False

def find_pytorch_processes(
    processes: Iterable[ProcInfo] | None = None,
) -> list[ProcInfo]:
    """All PyTorch candidates, best-evidence ordering left to callers."""
    if processes is None:
        processes = iter_process_info()
    return [info for info in processes if is_pytorch_process(info)]
