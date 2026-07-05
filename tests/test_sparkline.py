"""M4 acceptance tests for the pure sparkline mapping."""

from torchwatch.widgets.sparkline import BLOCKS, spark


def test_empty():
    assert spark([]) == ""


def test_single_value_is_one_block():
    out = spark([1.0])
    assert len(out) == 1 and out in BLOCKS


def test_flat_window_renders_mid_blocks():
    assert spark([2.0, 2.0, 2.0]) == BLOCKS[3] * 3


def test_full_range_maps_each_level():
    assert spark([0, 1, 2, 3, 4, 5, 6, 7]) == BLOCKS


def test_descending_loss_descends():
    assert spark([8, 7, 6, 5, 4, 3, 2, 1]) == BLOCKS[::-1]


def test_width_crops_to_most_recent():
    values = [float(v) for v in range(100)]
    out = spark(values, width=10)
    assert len(out) == 10
    assert out == spark(values[-10:], width=10)
    assert out[0] == BLOCKS[0] and out[-1] == BLOCKS[-1]
