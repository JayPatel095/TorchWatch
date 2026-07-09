"""Log-tail metrics source: attach to a running process whose stdout is a file.

The cluster case: `python train.py > train.log 2>&1 &` — tailing a regular
file steals nothing from anyone. Anything else is refused with an
actionable AttachError: a tty yields keystrokes rather than output, a pipe
read would steal data from its real consumer, and macOS has no /proc to
resolve fd 1 through at all.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import stat
from pathlib import Path

from torchwatch.collector.stdout import StdoutParser, TrainingUpdate
from torchwatch.collector.stream import FrameAssembler

# How much existing log to backfill on attach: enough for the parser to
# lock onto the format and total_steps immediately, small enough to be
# instant on a multi-GB log.
BACKFILL_BYTES = 8192


class AttachError(Exception):
    """Attaching to the pid is not possible; .args[0] says why and what to try."""


def resolve_stdout_path(pid: int, proc_root: str | Path = "/proc") -> Path:
    """Return the regular file the pid's stdout is redirected to."""

    link = Path(proc_root) / str(pid) / "fd" / "1"
    if not link.exists():
        if not Path(proc_root).exists():
            raise AttachError(
                "no /proc filesystem on this system (macOS?). pid attach is "
                "Linux-only; use `torchwatch run -- <cmd>` instead"
            )
        raise AttachError(
            f"pid {pid}: {link} not found — the process may have exited, "
            "or it belongs to another user (permission denied)"
        )
    
    resolved = link.resolve()
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise AttachError(
            f"pid {pid}: stdout ({link} → {resolved}) is not a redirected log file. "
            "reading a terminal or pipe would steal its output. restart it under "
            "`torchwatch run -- <cmd>`, or redirect next time: python train.py > train.log 2>&1"
        )
    
    return resolved


class TailSource:
    """MetricsSource that follows a redirected stdout log file."""

    interval_s = 0.1

    # Two construction paths: __init__ (pid attach) sets both; for_file
    # (direct file tail) sets neither.
    pid: int | None
    _proc_root: str | Path | None

    def __init__(self, pid: int, proc_root: str | Path = "/proc") -> None:
        self.pid = pid
        self._proc_root = proc_root
        self.label = f"tail: pid {pid}"
        self.parser = StdoutParser()
        self._assembler = FrameAssembler()
        self._updates: queue.Queue[TrainingUpdate] = queue.Queue()
        self._stop = threading.Event()
        self._path: Path | None = None

    @classmethod
    def for_file(cls, path: str | Path) -> "TailSource":
        """Tail a log file directly, no pid resolution (also used by tests)."""
        source = cls.__new__(cls)
        source.pid = None
        source._proc_root = None
        source.label = f"tail: {path}"
        source.parser = StdoutParser()
        source._assembler = FrameAssembler()
        source._updates = queue.Queue()
        source._stop = threading.Event()
        source._path = Path(path)
        return source

    def start(self) -> None:
        """Resolve (if attached by pid) and start the follow thread."""
        if self._path is None:  # pid-attach path: __init__ guaranteed these
            assert self.pid is not None and self._proc_root is not None
            self._path = resolve_stdout_path(self.pid, self._proc_root)
        threading.Thread(
            target=self._follow, daemon=True, name="torchwatch-tail"
        ).start()

    def _follow(self) -> None:
        """Read appended data forever; backfill the last BACKFILL_BYTES first."""
        assert self._path is not None
        with open(self._path, "r", errors="replace") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - BACKFILL_BYTES))
            while not self._stop.is_set():
                data = f.read(4096)
                if data:
                    for frame in self._assembler.feed(data):
                        self._parse(frame)
                else:
                    time.sleep(0.05)  # at EOF: wait for the writer to append

    def _parse(self, frame: str) -> None:
        """Queue the frame's TrainingUpdate, if it parses as one."""
        update = self.parser.parse_line(frame)
        if update is not None:
            self._updates.put(update)

    def next_update(self) -> TrainingUpdate | None:
        """Next queued update, or None when the queue is empty."""
        try:
            return self._updates.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        """Signal the follow thread to stop."""
        self._stop.set()
