"""M3 acceptance tests: parse the fixture logs in tests/fixtures/.

Green here = the parser handles all four formats plus the traps
(eval_loss/train_loss, loss-less progress frames, pure noise).
"""

import re
from pathlib import Path

from torchwatch.collector.stdout import StdoutParser

FIXTURES = Path(__file__).parent / "fixtures"


def frames(name: str) -> list[str]:
    """Fixture file → clean frames. tqdm separates frames with \\r, so split
    on both; this mirrors what the live reader must do."""
    text = (FIXTURES / name).read_text()
    return [ln for ln in re.split(r"[\r\n]+", text) if ln.strip()]


def parse_fixture(name: str):
    parser = StdoutParser()
    updates = [u for u in (parser.parse_line(f) for f in frames(name)) if u is not None]
    return parser, updates


def test_tqdm_real_capture():
    parser, updates = parse_fixture("tqdm.log")
    assert parser.matched_format == "tqdm"
    # 122 frames, every one carries a step counter.
    assert len(updates) >= 100
    # Very first trange frame has no loss postfix — still a valid step update.
    assert (updates[0].step, updates[0].total_steps, updates[0].loss) == (0, 60, None)
    # Postfix frames carry the loss.
    assert any(u.step == 29 and u.loss == 1.0655 for u in updates)
    # Final frame: complete run.
    last = updates[-1]
    assert (last.step, last.total_steps, last.loss) == (60, 60, 0.4145)


def test_hf_trainer_dict_lines():
    parser, updates = parse_fixture("hf_trainer.log")
    assert "hf_trainer" in parser.formats_seen
    losses = {u.loss for u in updates if u.loss is not None}
    # The two real training losses, and ONLY those — eval_loss (1.7523...)
    # and train_loss (1.8912) must not be swallowed by the 'loss' pattern.
    assert losses == {2.4915, 1.9847}
    # The interleaved bare progress bars still give step-only updates.
    assert any(u.step == 50 and u.total_steps == 500 and u.loss is None for u in updates)


def test_plain_print_lines():
    parser, updates = parse_fixture("plain.log")
    assert "plain" in parser.formats_seen
    assert (updates[0].step, updates[0].total_steps, updates[0].loss) == (100, 1000, 0.4230)
    # The capitalized `Loss=` variant parses too.
    assert any(u.step == 350 and u.loss == 0.3312 for u in updates)
    assert len(updates) == 5


def test_lightning_lines():
    parser, updates = parse_fixture("lightning.log")
    # Lightning wraps tqdm, so claiming these lines as "tqdm" is acceptable —
    # what matters is the extracted values.
    assert parser.matched_format in ("lightning", "tqdm")
    assert any(u.step == 225 and u.total_steps == 500 and u.loss == 1.892 for u in updates)
    assert any(u.step == 300 and u.loss == 0.987 for u in updates)


def test_noise_matches_nothing():
    parser, updates = parse_fixture("noise.log")
    assert updates == []
    assert parser.matched_format is None
    assert parser.formats_seen == set()
