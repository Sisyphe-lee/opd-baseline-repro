# TCOD F2B step-250 accumulated-memory evaluation

This run evaluates the Qwen2.5-3B TCOD-F2B step-250 checkpoint on the frozen
140 seen and 134 unseen ALFWorld manifests at a 30-step horizon.

It is a single-variable comparison against the prior prompt-fixed evaluation:

- retained: corrected step-2 prompt, temperature 0.4, 512 response tokens,
  10,240 prompt tokens, seed 42, tolerant local action parser;
- changed: every turn receives the accumulated user/assistant chat memory,
  matching the public TCOD workflow's message handling.

## Results

| Protocol | Seen | Unseen | Overall |
|---|---:|---:|---:|
| Self-contained bounded history | 62/140 (44.29%) | 52/134 (38.81%) | 114/274 (41.61%) |
| Accumulated chat memory | 122/140 (87.14%) | 109/134 (81.34%) | 231/274 (84.31%) |
| Change | +42.86 pp | +42.54 pp | +42.70 pp |

Paper Table 2 reports 81.43% seen and 79.19% unseen for Qwen2.5-3B
TCOD-F2B. This local result is not an exact paper-protocol comparison because
the corrected step-2 prompt and local parser are retained and response length
remains 512 rather than the paper's reported 4096.

The result demonstrates that the earlier 41.61% score primarily measured a
train/evaluation context mismatch. Accumulated memory also reproduced prompt
truncation and malformed late-turn outputs, but most successful episodes
finished before those failures dominated.

## Artifacts

- config: `configs/eval_full274_h30_accmemory_4gpu.yaml`
- launcher: `scripts/eval_full274_h30_accmemory_4gpu.sh`
- summary: `evaluation/full274_h30/summary.json`
- ordered results: `evaluation/full274_h30/task_results.jsonl`
- per-task trajectories: `evaluation/full274_h30/task_records/`
- log: `logs/eval_full274_h30_accmemory_4gpu.log`
