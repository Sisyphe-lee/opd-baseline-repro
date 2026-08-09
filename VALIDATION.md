# Baseline validation report

Validation date: 2026-08-09 UTC.

## Outcome

The migrated baseline passed static integrity checks and a live GPU inference
smoke for both final checkpoints. The historical full274 result evidence was
also re-read from the migrated folder and matched the established conclusions.

## Static and provenance checks

- Asset copy and runtime relocation markers exist.
- Training/evaluation data counts are 3,553 train, 140 Seen, and 134 Unseen.
- Every referenced ALFWorld PDDL file exists below this folder.
- Both final checkpoints have `.full_checkpoint`, Hugging Face exports, FSDP
  model shards, optimizer shards, and extra state.
- Active training configs freeze 250 steps, train batch 64,
  `seq-mean-token-mean`, and 512 response tokens.
- Active evaluation configs freeze full274, horizon 30, accumulated memory,
  strict action parsing, and 512 response tokens.
- TCOD training, prompt-fixed Vanilla training, and frozen evaluation workflows
  all import dynamically from the relocated virtual environment.
- No active config, data manifest, or editable Python mapping refers to `/lcy`,
  `tcod-f2b-repro`, or `opd-alfworld-sync-repro`.
- The copied student model, teacher model, TCOD final checkpoint/buffer, and
  Vanilla final checkpoint/buffer were compared byte-for-byte with their source
  directories with no differences.
- `validation/MODEL_CHECKPOINT_SHA256SUMS` contains 122 hashes covering every
  regular file under `models/` and `checkpoints/`.

Run these checks again with:

```bash
bash scripts/validate_baseline.sh
```

To re-hash all large artifacts (this reads roughly 111 GB), run:

```bash
sha256sum -c validation/MODEL_CHECKPOINT_SHA256SUMS
```

## Historical full274 evidence

| Checkpoint | Seen | Unseen | Overall |
|---|---:|---:|---:|
| TCOD-F2B step 250 | 122/140 | 110/134 | 232/274 (84.67%) |
| Vanilla OPD step 250 | 115/140 | 103/134 | 218/274 (79.56%) |

Both summaries and all 274 per-task records are present below
`results/evaluations/`.

## Live inference smoke

The live smoke used GPU 0 and loaded each Hugging Face export from this folder,
then executed the frozen evaluator on the same one Seen and one Unseen task.

| Checkpoint | Records collected | Seen/Unseen present | Collector passed |
|---|---:|---:|---:|
| TCOD-F2B step 250 | 2 | 1 / 1 | yes |
| Vanilla OPD step 250 | 2 | 1 / 1 | yes |

The smoke success rates are not scientific estimates and do not replace the
full274 results. They only demonstrate model loading, vLLM generation, corrected
prompt construction, accumulated memory, strict parsing, ALFWorld transitions,
task-record writing, and result aggregation from the relocated baseline.

Two earlier harness attempts are retained under `validation/attempt1_*` and
`validation/attempt2_*`. The first used a 140-task manifest with
`expected-count=1`; the second supplied only one split to a collector that
summarizes both splits. Neither failure occurred in model loading or ALFWorld
inference, and the final two-split smoke resolved both harness issues.

The logs also reproduce the known accumulated-memory behavior: long later-turn
prompts can reach the 10,240-token prompt cap. This is part of the currently
frozen evaluation protocol and is not silently changed here.

## Source safety

No files were deleted from `opd-alfworld-sync-repro` or `tcod-f2b-repro`.
Deletion of either source directory remains a separate user decision.
