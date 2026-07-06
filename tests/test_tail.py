"""Tests for the log-tail attach source.

resolve_stdout_path is exercised against a fake /proc tree (this dev
machine has no real one); the follow loop against a real growing file.
"""

import os
import time

import pytest

from torchwatch.collector.tail import AttachError, TailSource, resolve_stdout_path


def fake_proc(tmp_path, pid, target):
    """Build <tmp>/proc/<pid>/fd/1 as a symlink to `target`."""
    fd_dir = tmp_path / "proc" / str(pid) / "fd"
    fd_dir.mkdir(parents=True)
    (fd_dir / "1").symlink_to(target)
    return tmp_path / "proc"


def test_resolves_redirected_log(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("Step 1/10 | loss: 1.0\n")
    proc = fake_proc(tmp_path, 4242, log)
    assert resolve_stdout_path(4242, proc) == log


def test_pipe_stdout_refused_with_guidance(tmp_path):
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    proc = fake_proc(tmp_path, 4242, fifo)
    with pytest.raises(AttachError) as exc:
        resolve_stdout_path(4242, proc)
    # the refusal must tell the user what to do instead
    assert "torchwatch run" in str(exc.value) or "redirect" in str(exc.value)


def test_missing_pid_refused(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    with pytest.raises(AttachError):
        resolve_stdout_path(9999, proc)


def test_no_proc_at_all_refused(tmp_path):
    with pytest.raises(AttachError):
        resolve_stdout_path(4242, tmp_path / "nonexistent-proc")


def test_follow_picks_up_appended_lines(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("Step 1/100 | loss: 1.0000\n")  # backfill catches this

    source = TailSource.for_file(log)
    source.start()
    try:
        with open(log, "a") as f:
            for i in (2, 3, 4):
                f.write(f"Step {i}/100 | loss: {1.0 / i:.4f}\n")
                f.flush()

        updates = []
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and len(updates) < 4:
            update = source.next_update()
            if update is not None:
                updates.append(update)
            else:
                time.sleep(0.01)
    finally:
        source.close()

    assert [u.step for u in updates] == [1, 2, 3, 4]
    assert source.parser.matched_format == "plain"
