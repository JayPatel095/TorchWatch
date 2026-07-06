"""Acceptance tests for the stream frame assembler."""

from torchwatch.collector.stream import FrameAssembler


def test_partial_line_is_held_until_terminated():
    fa = FrameAssembler()
    assert fa.feed("Step 1 | lo") == []
    assert fa.feed("ss: 0.5\n") == ["Step 1 | loss: 0.5"]


def test_multiple_lines_in_one_chunk_keep_tail():
    fa = FrameAssembler()
    assert fa.feed("a\nb\nc") == ["a", "b"]
    assert fa.flush() == "c"


def test_carriage_return_frames():
    fa = FrameAssembler()
    assert fa.feed("f1\rf2\rf3") == ["f1", "f2"]
    assert fa.flush() == "f3"


def test_crlf_is_one_boundary():
    fa = FrameAssembler()
    assert fa.feed("x\r\ny\r\n") == ["x", "y"]
    assert fa.flush() is None


def test_blank_frames_dropped():
    fa = FrameAssembler()
    assert fa.feed("a\n   \n\n\nb\n") == ["a", "b"]


def test_empty_chunk_is_noop():
    fa = FrameAssembler()
    assert fa.feed("") == []
    assert fa.flush() is None


def test_flush_after_clean_terminator_is_none():
    fa = FrameAssembler()
    fa.feed("done\n")
    assert fa.flush() is None
