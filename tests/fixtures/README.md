# Log fixtures

Real or faithful captures of training-log formats the stdout parser must handle.

| file | provenance |
|---|---|
| `tqdm.log` | **Real capture**: `tqdm==4.x` `trange` with a `loss` postfix, stderr piped to a file (non-tty — exactly what torchwatch sees when attached). Frames are `\r`-separated. |
| `hf_trainer.log` | Hand-written to match HuggingFace `Trainer` stdout: dict-repr log lines interleaved with bare tqdm progress. The `eval_loss` / `train_loss` lines are traps — they must NOT parse as training loss. |
| `plain.log` | Hand-written `print()`-style lines, plus one `Loss=` capitalization variant. |
| `lightning.log` | Hand-written PyTorch Lightning progress lines (Lightning wraps tqdm, hence the resemblance). |
| `noise.log` | Typical non-metric stdout: warnings, wandb banner, torchvision download note, a metrics dict with no loss. Nothing here should match. |

When a real-world format breaks the parser, capture the offending lines into a
new fixture *first*, then fix the regex against it.
