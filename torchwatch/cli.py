"""Click entry point for torchwatch."""

import click

from torchwatch import __version__


@click.group(invoke_without_command=True)
@click.version_option(__version__, prog_name="torchwatch")
@click.option("--pid", type=int, default=None, help="Attach to a specific PyTorch process.")
@click.option("--poll", type=int, default=500, help="Poll interval in milliseconds.")
@click.option("--demo", is_flag=True, help="Show synthetic training metrics (no process attach).")
@click.pass_context
def main(ctx: click.Context, pid: int | None, poll: int, demo: bool) -> None:
    """torchwatch: a btop for your PyTorch GPU jobs.

    With no subcommand, auto-detects the first PyTorch process and attaches.
    """
    if ctx.invoked_subcommand is not None:
        return

    if demo and pid is not None:
        raise click.UsageError("--demo and --pid are mutually exclusive")

    from torchwatch.app import MetricsSource, TorchwatchApp

    metrics_source: MetricsSource | None = None
    if demo:
        from torchwatch.collector.demo import DemoMetrics

        # 300 steps @ 0.1s → the full arc (decay, loop, spike alert) shows
        # inside ~30s, short enough for the README GIF.
        metrics_source = DemoMetrics(total_steps=300)
    elif pid is not None:
        from torchwatch.collector.tail import AttachError, TailSource

        # start() is TailSource-specific (not part of the MetricsSource
        # protocol), so call it before the variable widens to the protocol.
        tail = TailSource(pid)
        try:
            tail.start()
        except AttachError as exc:
            raise click.ClickException(str(exc)) from exc
        metrics_source = tail

    try:
        TorchwatchApp(poll_ms=poll, pid=pid, metrics_source=metrics_source).run()
    finally:
        if metrics_source is not None:
            metrics_source.close()


@main.command("run", context_settings={"ignore_unknown_options": True})
@click.argument("command", nargs=-1, required=True, type=click.UNPROCESSED)
@click.option("--poll", type=int, default=500, help="GPU poll interval in milliseconds.")
def run_training(command: tuple[str, ...], poll: int) -> None:
    """Launch a training command and watch it: torchwatch run -- python train.py

    The command runs as a child of torchwatch under a pseudo-terminal;
    quitting the dashboard terminates it.
    """
    from torchwatch.app import TorchwatchApp
    from torchwatch.collector.wrapper import WrapperSource

    source = WrapperSource(list(command))
    source.start()
    try:
        TorchwatchApp(poll_ms=poll, pid=source.pid, metrics_source=source).run()
    finally:
        source.close()


@main.command("list")
def list_processes() -> None:
    """List running PyTorch processes and exit."""
    import sys

    from torchwatch.collector.proc import find_pytorch_processes

    procs = find_pytorch_processes()
    if not procs:
        click.echo("no PyTorch processes found")
        if sys.platform == "darwin":
            click.echo(
                "note: detection reads process maps, which macOS forbids — "
                "on this machine use `torchwatch run -- <cmd>` instead",
                err=True,
            )
        return

    click.echo(f"{'PID':>7}  COMMAND")
    for proc in procs:
        cmd = " ".join(proc.cmdline) or proc.name
        if len(cmd) > 70:
            cmd = cmd[:69] + "…"
        click.echo(f"{proc.pid:>7}  {cmd}")


if __name__ == "__main__":
    main()
