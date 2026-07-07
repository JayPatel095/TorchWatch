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
