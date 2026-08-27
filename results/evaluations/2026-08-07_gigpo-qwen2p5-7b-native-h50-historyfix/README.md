# GiGPO full274 h50 after ALFWorld history fix

This run repeats the earlier TCOD-style full274 evaluation of
`models/GiGPO-Qwen2.5-7B-Instruct-ALFWorld` after one evaluator change:

```diff
- if len(history) < HISTORY_LENGTH:
+ if not history:
```

The no-history template is now used only for the initial prompt. Starting on
step 2, the history template explicitly restores `task_description` and includes
the available partial history, matching the GiGPO official environment manager.

All other evaluation settings are held fixed against the previous full274 h50
run: the exact 140 seen and 134 unseen manifests, horizon 50, temperature 0.4,
top-p 1.0, top-k -1, response cap 512, eight TP=1 vLLM engines, and seed 42.

## Validated result

| Split | Previous | History-fixed | Change |
|---|---:|---:|---:|
| seen | 49/140 (35.00%) | 135/140 (96.43%) | +61.43 pp |
| unseen | 53/134 (39.55%) | 125/134 (93.28%) | +53.73 pp |
| overall | 102/274 (37.23%) | 260/274 (94.89%) | +57.66 pp |
| split macro | 37.28% | 94.86% | +57.58 pp |

Additional fixed-run diagnostics:

- action parse validity: seen 100%, unseen 100%
- action admissibility: seen 99.61%, unseen 99.85%
- timeout rate: seen 3.57%, unseen 6.72%
- average rounds: seen 9.09, unseen 11.98

The collector validated exactly 274 unique records against both frozen
manifests. The run exited successfully and released all eight GPUs.

## Previously audited games

Both same-game failures used to diagnose the bug now succeed:

- box -> dresser: 4 actions (`go`, `take`, `go`, `move`)
- spraybottle -> toilet: 4 actions (`go`, `take`, `go`, `move`)

This demonstrates that the earlier local failure labels were correct for their
generated trajectories, while the missing task on step 2 was the upstream cause
of the behavioral divergence.

## Artifacts

- config: `configs/eval_full274_h50_historyfix_8gpu.yaml`
- launcher: `scripts/eval_full274_h50_historyfix_8gpu.sh`
- full log: `logs/eval_full274_h50_historyfix_8gpu.log`
- aggregate summary: `evaluation/full274_h50/summary.json`
- manifest-ordered results: `evaluation/full274_h50/task_results.jsonl`
- per-game trajectories: `evaluation/full274_h50/task_records/`
- exact evaluator change: `history_fix.patch`
