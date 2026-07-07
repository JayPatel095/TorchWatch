"""PyTorch process discovery: powers `torchwatch list` and pid auto-detection.

The system boundary (psutil iteration) is separated from the decision logic
(is this a PyTorch process?) so the latter is testable with fake data —
same trick as `proc_root` in tail.py.

Detection evidence, strongest first:
- a loaded library path containing the torch package (site-packages/torch/)
  — definitive, but reading another process's maps needs Linux and, for
  other users' processes, permissions; `maps` is empty when unreadable
- `torchrun` / `torch.distributed.run` in the command line — definitive
- a bare python process with NO torch evidence is NOT a match: false
  positives ("attached to my jupyter kernel?") cost more trust than false
  negatives, and the wrapper mode always exists as the fallback

torchwatch's own processes must never match (we are a python process too).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from pathlib import Path

import psutil


@dataclass(frozen=True)
class ProcInfo:
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
                maps = tuple(m.path for m in proc.memory_maps(grouped=True))
            except (psutil.Error, OSError, PermissionError):
                maps = ()  # macOS / other users' processes: maps unreadable
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
