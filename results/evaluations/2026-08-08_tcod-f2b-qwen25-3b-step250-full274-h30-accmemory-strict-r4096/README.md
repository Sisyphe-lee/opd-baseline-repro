# TCOD F2B step-250 paper-length evaluation

This run evaluates the frozen Qwen2.5-3B TCOD-F2B step-250 checkpoint with:

- corrected step-2 prompt;
- accumulated full chat memory;
- exact public TCOD `<action>...</action>` parser;
- 30 environment steps;
- 10,240 prompt tokens and 4,096 response tokens;
- temperature 0.4, top-p 1.0, top-k -1, seed 42.

To improve hardware utilization, this run uses 96 workflow runners and a 0.65
vLLM GPU-memory fraction on four A100 GPUs. The prior 512-token run used 64
runners and a 0.45 fraction, so task-level differences include distributed
sampling variation in addition to the response-cap change.

## Results

| Protocol | Seen | Unseen | Overall |
|---|---:|---:|---:|
| Strict parser, response cap 512 | 122/140 (87.14%) | 110/134 (82.09%) | 232/274 (84.67%) |
| Strict parser, response cap 4096 | 122/140 (87.14%) | 108/134 (80.60%) | 230/274 (83.94%) |
| Paper TCOD-F2B | 81.43% | 79.19% | — |

All 274 records confirm `sampling.max_tokens=4096` and
`action_parser=strict_public_tcod`. Across the two distributed runs, 244 task
outcomes agree, 16 successes became failures, and 14 failures became successes.
The maximum generated response was only 528 characters, so the old 512-token
limit was not binding for these trajectories. The 0.73-point aggregate change
is therefore consistent with sampling variation rather than a response-length
effect.

## Artifacts

- config: `configs/eval_full274_h30_accmemory_strict_r4096_4gpu.yaml`
- launcher: `scripts/eval_full274_h30_accmemory_strict_r4096_4gpu.sh`
- summary: `evaluation/full274_h30/summary.json`
- ordered results: `evaluation/full274_h30/task_results.jsonl`
- per-task trajectories: `evaluation/full274_h30/task_records/`
- log: `logs/eval_full274_h30_accmemory_strict_r4096_4gpu.log`
