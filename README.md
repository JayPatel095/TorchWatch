# torchwatch

A btop for your PyTorch GPU jobs: live per-GPU utilization, VRAM pressure, training
throughput, loss curve, and ETA — all in your terminal, with zero code changes.

## Install

```
pip install torchwatch
```

## Run

```
torchwatch                # auto-detect the first PyTorch process and attach
torchwatch --pid 12345    # attach to a specific process
torchwatch list           # list running PyTorch processes
```

*Work in progress.*
