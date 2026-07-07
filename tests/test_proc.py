"""Tests for PyTorch process detection, on hand-built ProcInfo fakes."""

from torchwatch.collector.proc import ProcInfo, find_pytorch_processes, is_pytorch_process


def proc(pid=100, name="python3.12", cmdline=("python", "train.py"), maps=()):
    return ProcInfo(pid=pid, name=name, cmdline=tuple(cmdline), maps=tuple(maps))


TORCH_MAP = "/site-packages/torch/lib/libtorch_python.so"


def test_python_with_torch_maps_matches():
    assert is_pytorch_process(proc(maps=("/usr/lib/libc.so", TORCH_MAP)))


def test_python_without_torch_evidence_does_not_match():
    assert not is_pytorch_process(proc(maps=("/usr/lib/libc.so",)))
    assert not is_pytorch_process(proc(maps=()))  # unreadable maps: no evidence


def test_torchrun_matches_even_without_maps():
    assert is_pytorch_process(proc(name="torchrun", cmdline=("torchrun", "train.py")))
    assert is_pytorch_process(
        proc(cmdline=("python", "-m", "torch.distributed.run", "train.py"))
    )


def test_non_python_process_with_torchy_path_does_not_match():
    assert not is_pytorch_process(
        proc(name="node", cmdline=("node", "server.js"), maps=(TORCH_MAP,))
    )


def test_torchwatch_itself_never_matches():
    assert not is_pytorch_process(
        proc(cmdline=("python", "-m", "torchwatch", "--demo"), maps=(TORCH_MAP,))
    )
    assert not is_pytorch_process(
        proc(cmdline=("/repo/.venv/bin/torchwatch", "run", "--", "python", "train.py"))
    )


def test_find_filters_and_preserves_order():
    a = proc(pid=1, maps=(TORCH_MAP,))
    b = proc(pid=2)  # no evidence
    c = proc(pid=3, name="torchrun", cmdline=("torchrun", "t.py"))
    assert find_pytorch_processes([a, b, c]) == [a, c]
