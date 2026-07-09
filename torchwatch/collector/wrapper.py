"""Wrapper-mode metrics source: run a training command under a pty we own.

Being the parent is what makes the output legitimately readable — no
attaching tricks, no stolen pipes. The child sees a real tty (so tqdm
renders its usual bars); a daemon thread reads the pty master through the
assembler → parser → queue pipeline. Quitting the dashboard terminates
the wrapped run, by design.
"""

from __future__ import annotations

import codecs
import fcntl
import os
import queue
import struct
import subprocess
import termios
import threading

from torchwatch.collector.stdout import StdoutParser, TrainingUpdate
from torchwatch.collector.stream import FrameAssembler


class WrapperSource:
    """MetricsSource that spawns `command` under a pty and parses its output."""

    interval_s = 0.05

    def __init__(self, command: list[str]) -> None:
        self.command = command
        shown = " ".join(command)
        if len(shown) > 40:
            shown = shown[:39] + "…"
        self.label = f"run: {shown}"
        self.parser = StdoutParser()
        self.exit_code: int | None = None
        self._assembler = FrameAssembler()
        # pty reads can split multibyte utf-8; an incremental decoder keeps
        # the partial bytes instead of mangling them.
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._updates: queue.Queue[TrainingUpdate] = queue.Queue()
        self._proc: subprocess.Popen[bytes] | None = None
        self._master_fd: int | None = None

    @property
    def pid(self) -> int | None:
        """Child pid, or None before start()."""
        return self._proc.pid if self._proc is not None else None

    def start(self) -> None:
        """Spawn the child on a fresh pty and start the reader thread."""
        master, slave = os.openpty()
        # Give the child a plausible terminal size so tqdm picks sane ncols.
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 120, 0, 0))
        self._proc = subprocess.Popen(
            self.command,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            start_new_session=True,
            close_fds=True,
        )
        os.close(slave)  # parent keeps only the master side
        self._master_fd = master
        threading.Thread(
            target=self._read_loop, daemon=True, name="torchwatch-wrapper-read"
        ).start()

    def _read_loop(self) -> None:
        """Reader thread: drain the pty until EOF/EIO, then reap the child."""
        assert self._master_fd is not None and self._proc is not None
        try:
            while True:
                data = os.read(self._master_fd, 4096)
                if not data:
                    break
                self._ingest(self._decoder.decode(data))
        except OSError:
            pass  # EIO: child closed the slave side
        tail = self._assembler.flush()
        if tail is not None:
            self._parse(tail)
        self.exit_code = self._proc.wait()

    def _ingest(self, text: str) -> None:
        """Feed decoded text through the assembler; parse completed frames."""
        for frame in self._assembler.feed(text):
            self._parse(frame)

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
        """Terminate (then kill) the child if alive; release the pty."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None
