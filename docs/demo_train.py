"""Stand-in training script for the README GIF (see docs/demo.tape).

Prints plain-format progress lines that torchwatch parses live. The loss
decays like a real run, with a brief mid-run jump so the spike alert makes
an appearance. No torch dependency — the point of the GIF is the dashboard,
not the training.
"""

import math
import random
import time

TOTAL = 300

for step in range(1, TOTAL + 1):
    loss = 2.5 * math.exp(-3 * step / TOTAL) + random.uniform(0.0, 0.05)
    if 200 <= step < 205:  # something goes briefly wrong
        loss *= 4
    print(f"step {step}/{TOTAL} loss: {loss:.4f}", flush=True)
    time.sleep(0.1)
