# vhs tape for the README demo GIF.
#
# Render with:  vhs docs/demo.tape   (from the repo root, venv active so
# `torchwatch` and `python` resolve to the project environment).
# Requires vhs: https://github.com/charmbracelet/vhs
#
# The story: a user looks at their training script, launches it under
# torchwatch, and watches the dashboard — loss curve, throughput/ETA, and
# a mid-run loss spike raising an alert.
#
# Pacing note: the training script runs on its own clock (spike ~20s in,
# done at 30s), so the two mid-run sleeps are anchored to that timeline;
# everything else is padded generously for first-time viewers.
#
# The GIF opens and closes on an empty prompt, so the loop is seamless.

Output docs/demo.gif

# Wide and shallow: the dashboard is only ~20 rows tall (2 GPU panels in
# one row + sparkline + eta bar + footer), so extra height is dead space.
Set FontSize 14
Set Width 1200
Set Height 420
Set Theme "Catppuccin Mocha"
Set Padding 8

# Invisible setup: give the script the filename a real user would have.
Hide
Type "cp docs/demo_train.py train.py && clear"
Enter
Show

# Open on an empty prompt for a beat before anything happens.
Sleep 2s

# Beat 1: the training script, plain and unmodified — no imports needed.
Type "cat train.py"
Sleep 1s
Enter
Sleep 5s

# Beat 2: launch it under torchwatch.
Type "clear"
Sleep 500ms
Enter
Sleep 500ms
Type "torchwatch run -- python train.py"
Sleep 1.5s
Enter

# Beat 3: watch the dashboard fill in — sparkline grows, ETA converges.
# (run-synced: the spike lands at ~20s, just into the next beat)
Sleep 15s

# Beat 4: the scripted mid-run spike raises an alert; hold while it
# lingers in the alerts area. (run-synced)
Sleep 12s

# Beat 5: the run finishes (step 300/300); linger on the completed
# dashboard. (run-synced: training is done at the 30s mark)
Sleep 7.5s

# Quit, then clear — the GIF ends exactly on the Enter, so the wiped
# terminal is the loop's opening frame and its 2s buffer plays there.
Type "q"
Sleep 1s
Type "clear"
Sleep 1s
Enter

# Invisible cleanup.
Hide
Type "rm train.py"
Enter
