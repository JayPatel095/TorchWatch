"""Frame assembly: turn raw stream chunks into parser-ready lines.

A wrapped process writes to its pty in arbitrary chunks — a read can end
mid-line, or deliver fifty tqdm frames at once. The assembler buffers the
stream and emits only COMPLETE frames, ready for StdoutParser.parse_line.

Contract:
- feed(chunk) returns the frames completed by this chunk, in order. A
  trailing partial (no terminator yet) is held in the buffer, not returned.
- \\r, \\n, and \\r\\n all end a frame; \\r\\n is ONE boundary, not two.
- Blank frames (empty or whitespace-only, e.g. from consecutive
  terminators) are dropped, never emitted.
- flush() returns the held partial if it is non-blank, else None, and
  clears the buffer — call it at end-of-stream so a final unterminated
  line isn't lost.
"""

from __future__ import annotations


class FrameAssembler:
    def __init__(self) -> None:
        self._buf = ""

    def feed(self, chunk: str) -> list[str]:
        """Absorb one chunk; return the frames it completed."""
        raise NotImplementedError

    def flush(self) -> str | None:
        """Return the final unterminated frame (if non-blank) and reset."""
        raise NotImplementedError
