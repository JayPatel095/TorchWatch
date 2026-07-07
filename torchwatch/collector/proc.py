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
    """Decide whether one process is a PyTorch training candidate.

    The rules (each covered by a test in tests/test_proc.py):
    1. torchwatch itself (or its CLI) never matches, regardless of evidence.
    2. `torchrun` or `torch.distributed.run` anywhere in name/cmdline → True.
    3. A python-ish process (name starts with "python", or argv[0] ends with
       one) whose maps include a path containing "/torch/" → True.
    4. Anything else — including python processes with no torch evidence —
       → False.
    """
    raise NotImplementedError


def find_pytorch_processes(
    processes: Iterable[ProcInfo] | None = None,
) -> list[ProcInfo]:
    """All PyTorch candidates, best-evidence ordering left to callers."""
    if processes is None:
        processes = iter_process_info()
    return [info for info in processes if is_pytorch_process(info)]
