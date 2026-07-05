"""Training-log line parser: extracts (step, total_steps, loss) from stdout.

Format detection is registry-driven: FORMATS holds (name, pattern) pairs in
priority order, and for each line the first matching pattern wins. Every
pattern uses the same named groups — (?P<step>...), (?P<total>...),
(?P<loss>...) — omitting the ones its format doesn't carry, so extraction
via match.groupdict() is uniform across formats.

Constraints the patterns must respect (all exercised by the fixtures in
tests/fixtures/):
- tqdm frames are separated by \\r, not \\n; the reader splits before calling
  parse_line. Early frames ("0/60 [...]") carry no loss postfix — a step-only
  match is still a valid update (ETA needs it), with loss=None.
- Lightning lines are tqdm lines underneath (Lightning wraps tqdm), so
  pattern order/specificity decides who claims them.
- HF Trainer logs are python-dict reprs; they are parsed with a regex, never
  eval'd — log lines are untrusted input. The 'loss' key must be anchored so
  'eval_loss'/'train_loss' don't match.
- Loss text may be '0.4230', '2.5e-05', or 'nan'; float() accepts all three.
- Progress-bar glyphs are unicode and terminal-dependent; patterns anchor on
  the stable parts (the N/M counter, key=value postfixes), not the bar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Canonical format names, as reported to the UI and asserted by tests.
TQDM = "tqdm"
HF_TRAINER = "hf_trainer"
PLAIN = "plain"
LIGHTNING = "lightning"


@dataclass(frozen=True)
class TrainingUpdate:
    """One parsed log line; fields are None when the format doesn't carry them."""

    step: int | None
    total_steps: int | None
    loss: float | None
    format_name: str


# Ordered by priority: first match wins per line.
FORMATS: list[tuple[str, re.Pattern[str]]] = []


class StdoutParser:
    """Line-at-a-time parser.

    Tracks what it has recognized so the UI can report "parsing: tqdm" or
    "no training metrics detected":
    - `matched_format`: name of the most recent match (None until the first).
    - `formats_seen`: every format name that has matched at least once.
    """

    def __init__(self) -> None:
        self.matched_format: str | None = None
        self.formats_seen: set[str] = set()

    def parse_line(self, line: str) -> TrainingUpdate | None:
        """Parse one line/frame against FORMATS in order.

        The first match is recorded in matched_format/formats_seen and
        returned as a TrainingUpdate (absent groups become None). Lines
        matching no format return None.
        """
        raise NotImplementedError
