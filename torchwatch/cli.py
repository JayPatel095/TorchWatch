"""Click entry point for ptwatch."""

import click

from torchwatch import __version__


@click.group(invoke_without_command=True)
@click.version_option(__version__, prog_name="ptwatch")
@click.option("--pid", type=int, default=None, help="Attach to a specific PyTorch process.")
@click.option("--poll", type=int, default=500, help="Poll interval in milliseconds.")
@click.option("--no-stdout", is_flag=True, help="Disable stdout parsing (shared memory only).")
@click.pass_context
def main(ctx: click.Context, pid: int | None, poll: int, no_stdout: bool) -> None:
    """ptwatch: a btop for your PyTorch GPU jobs.

    With no subcommand, auto-detects the first PyTorch process and attaches.
    """
    if ctx.invoked_subcommand is not None:
        return
    click.echo("hello from ptwatch")


@main.command("list")
def list_processes() -> None:
    """List running PyTorch processes and exit."""
    click.echo("hello from ptwatch (list) — process detection not implemented yet")


if __name__ == "__main__":
    main()
