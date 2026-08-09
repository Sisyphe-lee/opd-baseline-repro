# Baseline provenance and asset boundary

## Upstream source

This repository vendors the runnable TCOD/Trinity source from:

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

## Local baseline assets

The assembled local baseline contains:

- the runnable training runtime and isolated frozen evaluator;
- Qwen2.5-3B student initialization and the GiGPO Qwen2.5-7B teacher;
- complete TCOD and Vanilla step-250 checkpoints, optimizer shards, metadata,
  final Hugging Face exports, and checkpoint buffer snapshots;
- training logs, full-274 evaluation summaries, per-task records, manifests,
  plotting inputs, regenerated figures, and validation evidence;
- the ALFWorld runtime and data, paper PDF and extracted text, and a relocated
  Python environment;
- isolated historical Qwen3 entropy-diagnostic scripts and reference figures,
  explicitly marked as non-baseline research artifacts.

Intermediate step-20 through step-240 checkpoints, stale SQLite queues from
interrupted runs, the 2.06 GB raw historical token-entropy JSONL, unrelated
Qwen3 checkpoints, failed smoke assets, caches, and unrelated benchmarks were
not copied into the assembled baseline. No source file was deleted during the
migration.

## Git publication boundary

The local directory is much larger than the Git repository. `.gitignore`
intentionally excludes:

- `models/` and `checkpoints/`;
- `.venv_tcod/` and generated ALFWorld runtime data;
- future `runs/`, large weight formats, caches, and distributed-worker state;
- credentials, Deploy Keys, `private/`, `machines.md`, and local `AGENTS.md`.

These assets remain available locally; excluding them from Git does not delete
them. Code, configs, compact datasets, logs, summaries, per-task evidence,
analysis tables, and figures remain versioned.

## Integrity evidence

The assembled folder occupies about 123 GB. Large assets are independent files,
not hard links to either source directory. Migration-time byte comparison is
recorded in `validation/bytecheck.log`; the 122-file model/checkpoint manifest is
`validation/MODEL_CHECKPOINT_SHA256SUMS`. Rechecking that manifest reads roughly
111 GB and is therefore optional unless full artifact verification is needed.
