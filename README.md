# ptwatch

A btop for your PyTorch GPU jobs: live per-GPU utilization, VRAM pressure, training
throughput, loss curve, and ETA — all in your terminal, with zero code changes.

## Install

```
pip install ptwatch
```

## Run

```
ptwatch                # auto-detect the first PyTorch process and attach
ptwatch --pid 12345    # attach to a specific process
ptwatch list           # list running PyTorch processes
```

*Work in progress — see `ptwatch_project_brief.md` for the roadmap.*
