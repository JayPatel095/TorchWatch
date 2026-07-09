"""Training-log line parser: extracts (step, total_steps, loss) from stdout.

FORMATS holds (name, pattern) pairs in priority order; the first match per
line wins, and all patterns share the same named groups (step/total/loss)
so extraction is uniform. Traps the fixtures pin down: early tqdm frames
carry no loss (a step-only match is still a valid update); Lightning lines
are tqdm underneath, so pattern order decides; HF Trainer dict-reprs are
regexed, never eval'd (log lines are untrusted), with 'loss' quote-anchored
so eval_loss/train_loss don't match; patterns anchor on stable text, not
terminal-dependent progress-bar glyphs.
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


# Loss values in the wild: '0.4230', '2.5e-05', 'nan', 'inf'. float() takes all.
_NUM = r"(?:nan|inf|[\d.eE+-]+)"

# The tqdm signature: the `N/M [` counter right before the timing bracket.
# The loss postfix is optional (early frames have none) and may be followed by
# further postfix pairs (Lightning appends v_num=...), so no closing anchor.
_BAR = r"(?P<step>\d+)/(?P<total>\d+)\s*\[(?:[^\]]*?[,\s]loss=(?P<loss>" + _NUM + r"))?"

# Ordered by priority: first match wins per line.
FORMATS: list[tuple[str, re.Pattern[str]]] = [
    # Lightning wraps tqdm — the `Epoch N/M:` prefix is its only tell, so it
    # must outrank the generic tqdm pattern.
    (LIGHTNING, re.compile(r"Epoch \d+(?:/\d+)?:\s.*?" + _BAR)),
    (TQDM, re.compile(_BAR)),
    # Quote-anchored: 'eval_loss'/'train_loss' contain loss but not 'loss'.
    (HF_TRAINER, re.compile(r"[{,]\s*'loss':\s*(?P<loss>" + _NUM + r")")),
    (PLAIN, re.compile(
        r"(?i)\bstep\s+(?P<step>\d+)/(?P<total>\d+)"
        r".*?\bloss\s*[:=]\s*(?P<loss>" + _NUM + r")"
    )),
]


class StdoutParser:
    """Line-at-a-time parser that remembers what it has recognized:
    `matched_format` (most recent match) and `formats_seen` (all so far),
    so the UI can report "parsing: tqdm" or "no training metrics detected".
    """

    def __init__(self) -> None:
        self.matched_format: str | None = None
        self.formats_seen: set[str] = set()

    def parse_line(self, line: str) -> TrainingUpdate | None:
        """Return the first format's TrainingUpdate (absent groups → None
        fields), or None when no format matches."""
        for name, pattern in FORMATS:
            match = pattern.search(line)

            if match is None:
                continue
            
            self.matched_format = name
            self.formats_seen.add(name)
            
            gd = match.groupdict()

            step = gd.get("step")
            if step is not None:
                step = int(step)

            total = gd.get("total")
            if total is not None:
                total = int(total)

            loss = gd.get("loss")
            if loss is not None:
                loss = float(loss)

            return TrainingUpdate(step=step, total_steps=total, loss=loss, format_name=name)
        
        return None


