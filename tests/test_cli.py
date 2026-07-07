"""CLI tests for paths that don't launch the TUI."""

from click.testing import CliRunner

from torchwatch.cli import main


def test_pid_attach_refusal_is_a_clean_error(tmp_path):
    """On a machine where attach can't work, --pid must exit with the
    AttachError message, not a traceback. (On this Mac the trigger is the
    no-/proc case; on Linux the same path fires for a vanished pid.)"""
    result = CliRunner().invoke(main, ["--pid", "99999999"])
    assert result.exit_code != 0
    assert result.exception is None or result.exception.__class__.__name__ == "SystemExit"
    assert "Error" in result.output
    assert "torchwatch run" in result.output or "exited" in result.output


def test_demo_and_pid_are_mutually_exclusive():
    result = CliRunner().invoke(main, ["--demo", "--pid", "123"])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_list_prints_detected_processes(monkeypatch):
    import torchwatch.collector.proc as proc_mod
    from torchwatch.collector.proc import ProcInfo

    fakes = [
        ProcInfo(pid=48213, name="python3.11",
                 cmdline=("python", "train.py", "--epochs", "10"), maps=()),
        ProcInfo(pid=48190, name="torchrun",
                 cmdline=("torchrun", "--nproc-per-node=4", "train.py"), maps=()),
    ]
    monkeypatch.setattr(proc_mod, "find_pytorch_processes", lambda: fakes)
    result = CliRunner().invoke(main, ["list"])
    assert result.exit_code == 0
    assert "48213" in result.output and "train.py --epochs 10" in result.output
    assert "48190" in result.output


def test_list_reports_nothing_found(monkeypatch):
    import torchwatch.collector.proc as proc_mod

    monkeypatch.setattr(proc_mod, "find_pytorch_processes", lambda: [])
    result = CliRunner().invoke(main, ["list"])
    assert result.exit_code == 0
    assert "no PyTorch processes found" in result.output
