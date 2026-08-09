# ALFWorld evaluation protocol

Updated: 2026-08-04

## Purpose

Maintain two frozen evaluation tiers. Use the quick tier for routine iteration
and the full tier only for milestone checkpoints and final reporting. Teacher,
TCOD-F2B, and future methods must use identical task identities and inference
settings within a comparison.

## Shared inference contract

- Workflow: `TCOD_eval_alfworld_workflow`
- Prompt/history: TCOD/ATOD-compatible prompt, admissible actions, history length 2
- Thinking mode: disabled at the Qwen chat-template level
- Environment horizon: 50 actions
- Response cap: 512 tokens per turn
- Sampling: temperature 0.4, top-p 1.0, top-k -1
- Default execution: four TP=1 vLLM replicas on four A100 GPUs
- Report separately: seen success rate, unseen success rate, and their macro mean

Do not compare these scores directly with TCOD paper tables that use a 30-step
horizon. The longer local horizon is intentional and applies to every locally
compared model.

## Quick tier: quick-72

The routine set contains 72 unique games:

- 36 `valid_seen`
- 36 `valid_unseen`
- six games from each of the six ALFWorld task types in each split
- deterministic selection seed `20260802`

Manifests:

- `reproduction_data/alfworld/quick72_seen.jsonl`
  - SHA-256: `63794757419a82663a880d045f7c7568359ec46b59eb41a816ed4c13739f15a3`
- `reproduction_data/alfworld/quick72_unseen.jsonl`
  - SHA-256: `92aafe658b727e12d046f9ca1341644bab0f3f26ae129297342dce98426e68b8`

Generation command:

```bash
.venv_tcod/bin/python scripts/make_alfworld_quick_evalset.py \
  --per-type 6 \
  --seed 20260802 \
  --output-prefix quick72
```

Canonical example configs:

- Teacher: `reproduction_configs/alfworld_quick72_h50_teacher_step150_4gpu.yaml`
- TCOD student: `reproduction_configs/alfworld_quick72_h50_tcod_step250_4gpu.yaml`

Target runtime is approximately ten minutes or less on four A100 GPUs, including
model startup. Record the measured wall time with every training run.

## Full tier: full-274

The full set contains every unique validation game:

- 140 `valid_seen`
- 134 `valid_unseen`

Manifests:

- `reproduction_data/alfworld/full_valid_seen.jsonl`
  - SHA-256: `3f93167b4da2d68e789409785c9a328e0cf55a3b40bd60165cbd537011e460d3`
- `reproduction_data/alfworld/full_valid_unseen.jsonl`
  - SHA-256: `60c3edd7e9fba923d5befe5e88af424967b79c6d325eead4634a88a396e37662`

Generation command:

```bash
.venv_tcod/bin/python scripts/make_alfworld_full_evalset.py
```

Reference configs:

- Teacher: `reproduction_configs/alfworld_full274_h50_teacher_step150_4gpu.yaml`
- TCOD student: `reproduction_configs/alfworld_full274_h50_tcod_step250_3gpu.yaml`

The GPU count changes throughput, not the evaluation population. For final paired
comparisons, use the same GPU/engine layout where practical and always record it.

## Running the evaluator

The evaluator is Trinity bench mode; the YAML selects the checkpoint, tasksets,
workflow, decoding settings, and GPU layout. Run from the repository root.

For one four-GPU model:

```bash
tmux new-session -d -s eval_quick72_teacher \
  "cd /vepfs-mlp2/mlp-public/252302025/lcy/tcod-f2b-repro && \
   scripts/run_alfworld_eval.sh \
     reproduction_configs/alfworld_quick72_h50_teacher_step150_4gpu.yaml \
     0,1,2,3 \
     quick72_teacher_step150 \
     6388"
```

Arguments are `CONFIG`, `CUDA_VISIBLE_DEVICES`, `RUN_TAG`, and optional
`RAY_PORT`. The launcher limits BLAS/OpenMP threads, disables proxies, creates an
isolated short-path Ray session under `/dev/shm`, writes `logs/<RUN_TAG>.log`, and
cleans up only that Ray session. Never use a global `ray stop` on the shared
machine.

To evaluate a new checkpoint, copy the nearest canonical YAML and change only:

1. `name`
2. `model.model_path`
3. `cluster.gpu_per_node` and `explorer.rollout_model.engine_num` if the GPU count
   changes

Do not change task manifests, workflow, horizon, response cap, or sampling
parameters for a controlled comparison.

## Result validity gate

`finished_task_count` means evaluation jobs returned; it is not task success.
Likewise, `done`/`env_terminated` is not success: TextWorld's default 50-step
`Limit` wrapper sets `done=True` on timeout. The audited evaluator defines
`task_success` as ALFWorld `won=True` or a positive task reward and records
`env_timeout` separately. This was validated with both a forced timeout and an
expert-completed trajectory.

An aggregate result may enter `EXPERIMENT_RESULTS.md` after the success metric is
validated and its denominators match the manifests. Final publication reporting
also requires persisted per-task JSONL; the current Trinity bench aggregator does
not yet provide that file.

Every reported result must include:

- training run ID and exact checkpoint path
- quick-72 or full-274 contract
- seen/unseen numerator and denominator
- success metric source and validation method
- config and log paths
- GPU/engine layout and wall time

Legacy fixed-144 and ATOD-128 results use different populations or horizons and
must remain explicitly labelled; never combine them with quick-72/full-274.
