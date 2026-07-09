# torchwatch

A btop for your PyTorch GPU jobs with live per-GPU utilization, VRAM pressure, training
throughput, loss curve, and ETA, all in your terminal, with zero code changes.

![torchwatch demo](https://raw.githubusercontent.com/JayPatel095/TorchWatch/main/docs/demo.gif)

## Install

```
pip install torchwatch-tui
```

(The command and package are `torchwatch`; only the PyPI name carries the suffix.)

## Run

The easiest way is to launch your training command under torchwatch:

```
torchwatch run -- python train.py --epochs 10
```

Your script runs as a child of torchwatch under a pseudo-terminal. Its output is
parsed live, and quitting the dashboard (`q`) terminates it.

Attach to a job that's already running:

```
torchwatch list           # list running PyTorch processes
torchwatch --pid 12345    # attach to one of them
```

Attaching reads the process's stdout via `/proc` (Linux only), and only works when
stdout is redirected to a file (`python train.py > train.log`). If it isn't,
torchwatch tells you why and suggests alternatives instead of guessing.

Smoke-test the dashboard on synthetic data

```
torchwatch --demo
```

## Current Features

- **Per-GPU panels** — utilization, VRAM (used/total and a pressure-colored gauge),
  temperature, and power with one tile per device
- **Loss sparkline** — scrolling curve of recent loss values
- **Throughput + ETA** — steps/sec over a rolling window, elapsed time, and a
  time-to-completion estimate through epoch restarts
- **Alerts area** — appears only when something needs attention, and lingers long
  enough to read:
  - VRAM above 95% on a specific GPU, with concrete fixes (smaller batch size,
    mixed precision, gradient checkpointing)
  - loss stalled (no meaningful change over the last 100 updates)
  - loss spike (latest value far above the recent average)

Keys: `q` quit · `p` pause

## Supported log formats

torchwatch parses training progress straight from stdout with no imports or callbacks.
Detected automatically:

| format | example line |
|---|---|
| tqdm | ` 42/500 [00:12<02:15, 3.4it/s, loss=0.693]` |
| PyTorch Lightning | `Epoch 3/10: 42/500 [... loss=0.693 ...]` |
| HF Trainer | `{'loss': 0.693, 'learning_rate': 5e-05, 'epoch': 1.2}` |
| plain | `step 42/500 loss: 0.693` |

The header shows which format was detected. Lines that match nothing are ignored.

## Notes

- GPU stats come from NVML (NVIDIA). Without a usable NVML (e.g. macOS, no driver),
  torchwatch falls back to clearly-labeled mock data so the dashboard still runs.
- `--pid` attach and process detection use `/proc` and process memory maps, so they
  are Linux-only. `torchwatch run` works everywhere.

## Development

```
pip install -e ".[dev]"
pytest
ruff check .
```

## Planned

Roughly in order of likelihood:

- **More alert rules** — NaN/inf loss (the run is dead, say so immediately),
  GPU temperature, sustained throughput drops (often a dataloader bottleneck)
- **Alert logging** — opt-in `--alert-log alerts.log`: a timestamped record of
  every alert, so a spike at 3am is still explainable at 9am
- **Configurable thresholds** — `--vram-warn` / `--vram-alert` instead of the
  built-in 85/95
- **Generic metric sparklines** — track any `key=value` metric your logs already
  print (accuracy, val_loss, grad_norm, …) and chart the ones you pick
- **Appearance** — theme selection, optional extra GPU stats
- **Config file** — persistent defaults once the flag count justifies it
- **Raw output pane** — the wrapped process's actual stdout, scrolling under
  the dashboard
- **Multi-job dashboard** — one dashboard, several training processes
