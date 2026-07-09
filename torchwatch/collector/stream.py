"""Frame assembly: turn raw stream chunks into parser-ready lines.

A pty delivers arbitrary chunks — a read can end mid-line or carry fifty
tqdm frames at once. \\r, \\n, and \\r\\n all end a frame (\\r\\n is ONE
boundary, not two); blank frames are dropped.
"""

from __future__ import annotations

import re


class FrameAssembler:
    """Buffers a stream and emits only complete, non-blank frames."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, chunk: str) -> list[str]:
        """Absorb one chunk; return the frames it completed, holding any
        trailing partial in the buffer."""
        self._buf += chunk

        parts = re.split(r"[\r\n]+", self._buf)
        self._buf = parts[-1]

        return [frame for frame in parts[:-1] if frame.strip()]

    def flush(self) -> str | None:
        """Return the held partial (if non-blank) and reset — call at
        end-of-stream so a final unterminated line isn't lost."""
        temp = self._buf
        self._buf = ""

        return temp if temp.strip() else None

