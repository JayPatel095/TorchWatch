"""Click entry point for torchwatch."""

import click

from torchwatch import __version__


@click.group(invoke_without_command=True)
@click.version_option(__version__, prog_name="torchwatch")
@click.option("--pid", type=int, default=None, help="Attach to a specific PyTorch process.")
@click.option("--poll", type=int, default=500, help="Poll interval in milliseconds.")
@click.option("--no-stdout", is_flag=True, help="Disable stdout parsing (shared memory only).")
@click.pass_context
def main(ctx: click.Context, pid: int | None, poll: int, no_stdout: bool) -> None:
    """torchwatch: a btop for your PyTorch GPU jobs.

    With no subcommand, auto-detects the first PyTorch process and attaches.
    """
    if ctx.invoked_subcommand is not None:
        return
    from torchwatch.collector.nvidia import GiB, create_collector

    collector, fallback_reason = create_collector()
    if fallback_reason is not None:
        click.echo(f"NVML unavailable ({fallback_reason}) — showing mock data", err=True)
    click.echo(f"{collector.gpu_count()} GPU(s) detected")
    for gpu in collector.sample():
        util = f"{gpu.utilization_pct}%" if gpu.utilization_pct is not None else "—"
        temp = f"{gpu.temperature_c}°C" if gpu.temperature_c is not None else "—"
        power = f"{gpu.power_w:.0f}W" if gpu.power_w is not None else "—"
        click.echo(
            f"GPU {gpu.index}  {gpu.name}  util {util}  "
            f"vram {gpu.vram_used_bytes / GiB:.1f}/{gpu.vram_total_bytes / GiB:.1f} GiB  "
            f"temp {temp}  power {power}"
        )
    collector.close()


@main.command("list")
def list_processes() -> None:
    """List running PyTorch processes and exit."""
    click.echo("process detection not implemented yet")


if __name__ == "__main__":
    main()
