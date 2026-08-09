# TCOD F2B step-250 accumulated-memory strict-parser evaluation

This run retains the corrected step-2 prompt and accumulated full chat memory,
and changes only action parsing to the exact public TCOD rule: accept content
only between case-sensitive `<action>` and `</action>` tags.

## Results

| Protocol | Seen | Unseen | Overall |
|---|---:|---:|---:|
| Accumulated memory, tolerant parser | 122/140 (87.14%) | 109/134 (81.34%) | 231/274 (84.31%) |
| Accumulated memory, strict parser | 122/140 (87.14%) | 110/134 (82.09%) | 232/274 (84.67%) |
| Paper TCOD-F2B | 81.43% | 79.19% | — |

All 274 strict-run records identify `action_parser` as
`strict_public_tcod`. The two stochastic distributed runs agree on 247 task
outcomes; 13 tolerant-run successes became failures and 14 failures became
successes. The one-task aggregate difference is therefore sampling variation,
not evidence that strict parsing improves accuracy.

The strict parser does not materially change the conclusion: accumulated chat
memory raises the local result from 41.61% to about 84.5%. The remaining known
paper-protocol differences include the corrected step-2 prompt, 512 response
tokens rather than the paper's reported 4096, hardware/topology, and the exact
unreleased teacher checkpoint.

## Artifacts

- config: `configs/eval_full274_h30_accmemory_strict_4gpu.yaml`
- launcher: `scripts/eval_full274_h30_accmemory_strict_4gpu.sh`
- summary: `evaluation/full274_h30/summary.json`
- ordered results: `evaluation/full274_h30/task_results.jsonl`
- per-task trajectories: `evaluation/full274_h30/task_records/`
- log: `logs/eval_full274_h30_accmemory_strict_4gpu.log`
