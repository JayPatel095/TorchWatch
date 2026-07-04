# Project brief: `ptwatch` — a btop for your PyTorch GPU jobs

## Overview

A terminal dashboard that gives ML practitioners live visibility into running PyTorch training jobs: per-GPU utilization, VRAM pressure, training throughput (samples/sec), loss curve, and estimated time-to-completion — all from a single `pip install ptwatch && ptwatch`. Attaches to any running PyTorch process with zero code changes required.

The closest existing tools are `nvtop` (GPU-only, not ML-aware) and `wandb`/`tensorboard` (require instrumentation + browser). This fills the gap: a zero-config, terminal-native, ML-aware monitor.

---

## Goals

- **Primary:** A working TUI that shows GPU stats and training metrics for any running PyTorch job, no code changes needed
- **Secondary:** An optional one-liner callback for richer per-step metrics (loss, grad norm, LR)
- **Stretch:** Session recording + replay for post-hoc debugging

---

## Feature checklist

### Zero-config GPU panel
- [ ] Poll all visible CUDA devices every 500ms via `pynvml`
- [ ] Display per-GPU: utilization %, VRAM used/total (GB), temperature (°C), power draw (W)
- [ ] List top processes per GPU by VRAM consumption (via `nvidia-smi` parsed or `pynvml`)
- [ ] Graceful fallback when no NVIDIA GPU is present (show CPU + RAM only)

### PyTorch process attachment
- [ ] Scan `/proc/<pid>/cmdline` and `/proc/<pid>/maps` to detect running PyTorch processes
- [ ] Match GPU device ownership: correlate `pynvml` process list with detected PyTorch PIDs
- [ ] Expose a shared memory ring buffer (`/dev/shm/ptwatch_<pid>`) that the optional callback writes to
- [ ] If no callback is present, parse stdout for common loss patterns (e.g. `loss: 0.423`, `Loss=0.423`, tqdm-style progress bars)

### Live loss and throughput curves
- [ ] Scrolling sparkline chart for training loss (last N steps, configurable)
- [ ] Scrolling sparkline for samples/sec (computed from step timestamps or stdout parsing)
- [ ] Auto-detect loss stalls: flag if loss hasn't improved > 0.1% in last 100 steps
- [ ] Auto-detect loss spikes: flag if loss increases > 2x rolling average in a single step

### ETA estimator
- [ ] Parse current step and total steps from stdout or shared memory
- [ ] Compute ETA = (total_steps - current_step) / rolling_mean(steps_per_sec)
- [ ] Display as "~14m 32s remaining" with a ± confidence interval based on throughput variance
- [ ] Recalibrate rolling window every 30s

### VRAM pressure warnings
- [ ] Warn (yellow) when any GPU VRAM > 85%
- [ ] Alert (red) when any GPU VRAM > 95%
- [ ] On alert, surface a contextual suggestion:
  - "Try reducing batch size"
  - "Consider enabling `torch.cuda.amp` (bf16/fp16)"
  - "Consider `torch.utils.checkpoint.checkpoint_sequential` for gradient checkpointing"
- [ ] If `torch.cuda.memory_snapshot()` is accessible, surface top 5 tensors by size

### Multi-GPU layout
- [ ] When multiple GPUs detected, show a tiled grid (2-column for ≤4 GPUs, 3-column for >4)
- [ ] Global aggregate row at the top: total VRAM, mean utilization across devices
- [ ] For DDP jobs, label each panel with rank if detectable from process name or env vars (`LOCAL_RANK`, `RANK`)

### Terminal UI (Textual)
- [ ] Built with the `Textual` Python TUI framework
- [ ] Layout: header bar (job name, PID, elapsed time) → GPU grid → metrics panel (loss + throughput sparklines) → footer (keybindings)
- [ ] Keyboard shortcuts: `q` quit, `r` reset sparklines, `p` pause polling, `↑/↓` adjust poll interval
- [ ] Responsive: gracefully degrades on narrow terminals (hide sparklines below 80 cols)
- [ ] Works in any terminal that supports color (iTerm2, Windows Terminal, tmux, SSH sessions)

### CLI entry point
- [ ] `ptwatch` — auto-detect the first PyTorch process and attach
- [ ] `ptwatch --pid <PID>` — attach to a specific process
- [ ] `ptwatch --poll 250` — set poll interval in ms (default 500)
- [ ] `ptwatch --no-stdout` — disable stdout parsing (shared memory only)
- [ ] `ptwatch list` — list running PyTorch processes and exit

### Optional callback integration
- [ ] `from ptwatch import WatchdogCallback` — a HuggingFace `TrainerCallback`-compatible class
- [ ] On `on_log` event: write step, loss, grad_norm, learning_rate to shared memory ring buffer
- [ ] Ring buffer: fixed-size circular buffer in `/dev/shm/ptwatch_<pid>`, 1000 entries, msgpack-encoded
- [ ] Callback works with HuggingFace Trainer, PyTorch Lightning (via `Callback`), and raw PyTorch loops

### pip-installable package
- [ ] `pyproject.toml` with `[project.scripts] ptwatch = "ptwatch.cli:main"`
- [ ] Publish to PyPI via GitHub Actions on tag push
- [ ] Supports Python 3.10+, Linux and macOS (Windows is stretch)
- [ ] Dependencies: `textual`, `pynvml`, `psutil`, `click`, `msgpack`

### CI
- [ ] GitHub Actions: lint (ruff), type check (mypy), unit tests on CPU-only path (mock `pynvml`)
- [ ] Test that stdout parsing correctly extracts loss from common log formats (tqdm, plain print, Lightning)

### Stretch: session recording and replay
- [ ] `ptwatch record --out session.ptw` — write a compressed event log during training
- [ ] `ptwatch replay session.ptw` — replay the session in the TUI after the fact
- [ ] Format: gzipped JSONL, one event per poll tick
- [ ] Use case: debug why a run went sideways without re-running it

---

## Stack

| Layer | Choice | Notes |
|---|---|---|
| TUI framework | Textual | Handles layout, events, mouse — no curses boilerplate |
| GPU metrics | pynvml | NVIDIA Management Library Python bindings |
| Process inspection | psutil + `/proc` | Find PyTorch PIDs, parse cmdline, match GPU ownership |
| IPC | `mmap` / `/dev/shm` | Shared memory ring buffer for optional callback path |
| Serialization | msgpack | Fast binary encoding for ring buffer entries |
| CLI | Click | Entry points, subcommands, options |
| Packaging | pyproject.toml (hatchling) | pip-installable, PyPI publishable |
| CI | GitHub Actions | Lint, typecheck, mock tests, PyPI publish on tag |

---

## Repo structure (suggested)

```
ptwatch/
├── pyproject.toml
├── README.md                  # GIF demo at top, then quickstart
├── ptwatch/
│   ├── __init__.py            # exports WatchdogCallback
│   ├── cli.py                 # Click entry point
│   ├── app.py                 # Textual App class
│   ├── widgets/
│   │   ├── gpu_panel.py       # per-GPU widget
│   │   ├── sparkline.py       # scrolling loss/throughput chart
│   │   └── eta_bar.py         # ETA + progress widget
│   ├── collector/
│   │   ├── nvidia.py          # pynvml wrapper
│   │   ├── proc.py            # /proc parsing, PID detection
│   │   ├── stdout.py          # stdout log parser (regex-based)
│   │   └── shm.py             # shared memory ring buffer reader
│   ├── callback.py            # WatchdogCallback (HF + Lightning compatible)
│   └── alerts.py              # VRAM pressure, loss spike/stall detection
├── tests/
│   ├── test_stdout_parser.py
│   ├── test_eta.py
│   └── test_alerts.py
└── .github/
    └── workflows/
        ├── ci.yml
        └── publish.yml
```

---

## Implementation notes

### Stdout parsing (zero-config path)
The most common PyTorch log formats to handle:

```
# tqdm style
100%|██████████| 500/1000 [02:14<02:14,  3.72it/s, loss=0.423]

# Lightning style
Epoch 1/10: 100%|...| loss=0.4230, val_loss=0.5123

# Plain print
Step 100/1000 | loss: 0.4230 | lr: 0.001

# Transformers Trainer
{'loss': 0.423, 'learning_rate': 5e-05, 'epoch': 0.5}
```

Use a set of regexes tried in order. Log the matched format on startup so the user knows which path was taken. If none match, show "stdout parsing: no pattern matched" and suggest using `WatchdogCallback`.

### Textual layout
Textual uses a CSS-like layout system. Sketch:

```
┌─────────────────────────────────────────────────────────┐
│ ptwatch  PID: 12345  job: train.py  elapsed: 00:14:32   │
├────────────────────┬────────────────────────────────────┤
│  GPU 0 (A100 80GB) │  GPU 1 (A100 80GB)                 │
│  util: 94%  78°C   │  util: 91%  76°C                   │
│  VRAM: 74/80 GB    │  VRAM: 71/80 GB                    │
│  power: 312W       │  power: 298W                       │
├────────────────────┴────────────────────────────────────┤
│  loss  ▁▂▃▄▅▄▃▂▁▁▁▁▁▁  0.423 → 0.312                   │
│  tok/s ▄▅▅▆▆▆▆▆▆▆▆▆▆▆  3,820 samples/sec                │
│  ETA   ~28m 14s ± 2m  [step 312/1000]                  │
├─────────────────────────────────────────────────────────┤
│  q quit  r reset  p pause  ↑↓ poll interval             │
└─────────────────────────────────────────────────────────┘
```

### Shared memory ring buffer
```python
# Layout: fixed header + circular body
# Header (64 bytes): magic, version, write_idx, capacity
# Each entry (msgpack): {step, loss, grad_norm, lr, timestamp}

import mmap, struct, msgpack

SHM_PATH = f"/dev/shm/ptwatch_{pid}"
CAPACITY = 1000
HEADER_SIZE = 64
ENTRY_SIZE = 128  # max msgpack size per entry
```

Write side (callback) uses `mmap` with `MAP_SHARED`. Read side (TUI) opens the same path read-only and polls `write_idx` for changes.

### Alert logic
```python
# Loss stall: no improvement in last N steps
def is_stalled(losses, window=100, threshold=0.001):
    if len(losses) < window:
        return False
    recent = losses[-window:]
    return (max(recent) - min(recent)) / (max(recent) + 1e-8) < threshold

# Loss spike: current loss > 2x rolling mean
def is_spiking(losses, window=20, multiplier=2.0):
    if len(losses) < window:
        return False
    rolling_mean = sum(losses[-window:-1]) / (window - 1)
    return losses[-1] > multiplier * rolling_mean
```

---

## README priorities

The README leads with a GIF of the dashboard running against a real training job (record one against a small MNIST or CIFAR run). Then:

1. Install: `pip install ptwatch`
2. Run: `ptwatch` (auto-detect) or `ptwatch --pid 12345`
3. Optional richer metrics: add `WatchdogCallback`
4. What you see: annotated screenshot

Comparison table vs `nvtop`, `wandb`, `tensorboard` — makes the positioning clear.

---

## Getting started (first session)

Suggested order:

1. Set up `pyproject.toml`, package structure, and Click CLI skeleton (`ptwatch` prints "hello")
2. Build `nvidia.py` — get `pynvml` polling and printing GPU stats to stdout
3. Build the Textual app with a single GPU panel widget (hardcoded mock data first)
4. Wire `nvidia.py` into the Textual app — live updating GPU panel
5. Build `stdout.py` — regex parser for common log formats, tested against fixture logs
6. Add the loss/throughput sparkline widget
7. Add ETA computation and display
8. Add alert logic and VRAM warning display
9. Build multi-GPU grid layout
10. Build `WatchdogCallback` and shared memory path
11. Add CI, publish to PyPI

Do not start with the shared memory IPC — get the zero-config stdout path working and useful first. The callback is a power-user feature; the core value is zero-config.
