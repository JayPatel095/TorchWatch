"""End-to-end wrapper test: a real child process under the pty, parsed live."""

import sys
import time

from torchwatch.collector.wrapper import WrapperSource

SCRIPT = (
    "import time\n"
    "for i in range(5):\n"
    "    print(f'Step {i + 1}/5 | loss: {1.0 / (i + 1):.4f}', flush=True)\n"
    "    time.sleep(0.03)\n"
)


def test_wrapper_captures_child_training_lines():
    source = WrapperSource([sys.executable, "-u", "-c", SCRIPT])
    source.start()
    updates = []
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and len(updates) < 5:
        update = source.next_update()
        if update is not None:
            updates.append(update)
        else:
            time.sleep(0.01)

    assert [u.step for u in updates] == [1, 2, 3, 4, 5]
    assert updates[0].loss == 1.0
    assert updates[0].total_steps == 5
    assert source.parser.matched_format == "plain"

    # reader thread records the exit code once the child finishes
    while source.exit_code is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert source.exit_code == 0
    source.close()
