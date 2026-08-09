# TCOD upstream provenance

This directory vendors the runnable TCOD/Trinity source from:

- Repository: <https://github.com/kokolerk/TCOD>
- Commit: `465eef4406ad0cff675b36bd46f37f28b1736ff9`

Only `trinity/`, `TCOD_examples/`, and the top-level packaging/license files
needed to run the ALFWorld experiment are included. Models, datasets,
checkpoints, caches, and previous runs are deliberately not copied.

## Local source patch

There is exactly one behavioral source change relative to the upstream commit:

- `trinity/common/workflows/envs/TCOD/alfworld/TCOD_f2b_workflow.py`
  uses the no-history prompt only when `history` is empty. Upstream uses it
  while `len(history) < HISTORY_LENGTH`, which incorrectly sends the second
  environment turn through the no-history prompt even though one observation
  and action are already available.

The upstream accumulated chat `memory`, action parser, F2B window, experience
queue, bounded-staleness policy, synchronization strategy, loss, and optimizer
logic are unchanged.

The original `OPD_workflow.py` and its `OPD_alfworld_workflow` registration are
also preserved. A separate local `OPD_promptfix_workflow.py` registers
`OPD_promptfix_alfworld_workflow` for the controlled Vanilla-vs-F2B comparison;
it exposes the available history from turn 2 without replacing the upstream
implementation.
