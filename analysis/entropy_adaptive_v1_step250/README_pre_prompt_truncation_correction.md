# Entropy-adaptive v1: step-250 results

> **重要更正（2026-08-10）：** 原报告中的 late-turn 熵恢复比例混入了 `prompt_truncated` 占位行。超过 10240 token 后，当前模型封装并不会继续生成，而会返回最后一个 prompt token 作为单 token、零 action-mask 的 Experience；workflow 仍会将它送入环境和教师诊断。全部 1,311 条失败训练轨迹最终都进入了该状态，因此原热力图后半段的极低熵主要是 runtime/诊断伪影，不能解释为教师置信度恢复。排除占位行后，84.09% 仍低于局部 crossing 峰值，但只有 14.96% 的后缀均值低于早期 baseline，只有 36.58% 的最后三个真实生成 turn 低于 baseline。

## Bottom line

Entropy-adaptive v1 is a useful negative result. It confirms strong trajectory-level heterogeneity in teacher-entropy dynamics, but it does **not** support the irreversible rule “the first sustained teacher-entropy spike marks the point after which teacher supervision is no longer useful.” On the frozen full274 evaluation, the step-250 policy is materially worse than both frozen baselines at seed 42.

The scientifically defensible conclusion is therefore:

1. the observation of heterogeneous entropy trajectories survives on the current Qwen2.5 / prompt-fixed / accumulated-memory protocol;
2. the v1 hard suffix cutoff is not a successful adaptation rule;
3. v2 should be recovery-aware and soft, and must be compared with a matched step-250 full-loss control.

## What v1 did

For each complete student rollout, v1 computed response-level teacher top-16 head entropy $H_t$. The first three turns defined the trajectory-local baseline

$$
B=\frac{1}{3}\sum_{i=0}^{2}H_i.
$$

It then found the first zero-based turn $t$ satisfying

$$
\frac{1}{3}\sum_{j=t-2}^{t}(H_j-B)\ge 0.175.
$$

The environment rollout and teacher scoring still ran to completion. Only Experiences before the detected frontier were returned for training; the crossing turn and suffix were excluded. Thus v1 tested a post-hoc hard loss cutoff, not online early stopping and not a compute-saving method.

Configuration: top-$k=16$, baseline turns $=3$, sustained window $=3$, minimum retained turns $=3$, full environment horizon $=30$, seed $=42$. Training resumed from the complete step-10 checkpoint and finished at step 250, saving every 20 steps and at step 250.

## Training and diagnostics

- Step 11 to 250 took 5.72 hours, or 41.97 optimizer updates/hour.
- Diagnostics contain 50,878 raw turn rows. Resume overlap duplicated 408 rows at Explorer step 8; selecting the last record per trajectory-turn leaves 50,470 rows.
- The cleaned panel contains 2,736 trajectories and 50,470 turns. Training-rollout success was 1,425/2,736 = 52.08%. This is an in-distribution training statistic, not full274 evaluation accuracy.
- 421/2,736 trajectories (15.39%) triggered a frontier. Overall, 42,895/50,470 turns (84.99%) were retained and 7,575 were dropped.
- Mean trajectory length was 18.45 turns; mean retained length was 15.68 turns. The median detected frontier was turn 10 in one-based display coordinates.
- Early model versions $\le 30$ had 13.39% training-rollout success; the last 30 model versions had 71.74%. Late in training the trigger rate fell to about 5% and the retained fraction rose to about 97%, so the algorithm approached full-loss behavior.
- Prompt truncation is a major confounder: 1,313 trajectories and 15,612 turns were marked truncated. There were 11,409 retained truncated turns. These rows have zero response action masks, so “retained turns” overstates the effective number of trainable rows while such rows can still enter the sequence-level batch denominator.

No fatal Trainer error, traceback, CUDA OOM, NCCL error, or Ray task failure was found in the completed training log. The diagnostics test suite passed: 16 tests.

## The frontier mechanism failed its key assumption

Among the 421 triggered trajectories:

- mean early baseline teacher entropy: 0.2024;
- mean crossing-turn entropy: 0.4699;
- mean suffix entropy: 0.2059;
- mean last-three-post-frontier entropy: 0.1124;
- 378/421 = 89.79% had suffix entropy below the crossing entropy;
- 345/421 = 81.95% had their last three post-frontier turns below their own early baseline;
- 291/421 = 69.12% had mean suffix entropy below their own early baseline.

Therefore a sustained local spike is usually a transient event, not an absorbing “teacher is now useless” boundary. The hard cutoff systematically discards many post-spike states on which the teacher becomes confident again.

The detector is also strongly outcome-associated: only 3.37% of successful trajectories triggered, versus 28.45% of failed trajectories. Triggered trajectories succeeded 11.40% of the time, versus 59.48% for non-triggered trajectories. This makes the frontier a useful failure marker, but it does not establish that removing its suffix improves learning. The retained action-valid rate (77.94%) and dropped action-valid rate (28.56%) are descriptive only because turn position, failure, trajectory length, and context truncation are all confounded.

## Frozen full274 evaluation

All three evaluations used the frozen protocol: 274 tasks, horizon 30, accumulated memory, strict lowercase action parsing, temperature 0.4, top-p 1, top-k -1, and 512 response tokens.

- seed 42: 193/274 = 70.44% (seen 110/140 = 78.57%; unseen 83/134 = 61.94%);
- seed 43: 201/274 = 73.36% (seen 109/140 = 77.86%; unseen 92/134 = 68.66%);
- seed 44: 203/274 = 74.09% (seen 109/140 = 77.86%; unseen 94/134 = 70.15%);
- mean across seeds: 72.63%; sample standard deviation: 1.93 percentage points.

The 597/822 descriptive pooled count must not be treated as 822 independent tasks because the same 274 task identities are reused across seeds.

At the shared seed 42, the frozen baselines were:

- TCOD F2B: 232/274 = 84.67%; adaptive v1 is lower by 39 tasks and 14.23 percentage points. Paired outcomes: adaptive-only 8, TCOD-only 47, both successful 185, both failed 34; exact McNemar $p=8.07\times10^{-8}$.
- Vanilla OPD: 218/274 = 79.56%; adaptive v1 is lower by 25 tasks and 9.12 percentage points. Paired outcomes: adaptive-only 17, Vanilla-only 42, both successful 176, both failed 39; exact McNemar $p=0.00155$.

The largest seed-42 task-type deficits relative to TCOD are `look_at_obj_in_light` (-29.03 pp), `pick_two_obj_and_place` (-26.83 pp), and `pick_cool_then_place_in_recep` (-19.57 pp). `pick_and_place_simple` ties TCOD and is 5.08 pp above Vanilla. The damage is therefore concentrated in tasks that require longer or more compositional recovery, exactly where an irreversible suffix cutoff is most suspect.

## Causal limitation and next experiment

This run does **not** have a matched full-loss step-250 control trained with the same current instrumentation, layout, resume path, and wall-clock conditions. Consequently, the final performance gap cannot be attributed entirely to entropy masking from this run alone. The seed-42 comparisons against frozen TCOD and Vanilla establish that the produced policy is worse; a matched full-loss run is required to isolate the causal effect of the cutoff.

The next scientifically clean experiment should:

1. replace hard irreversible truncation with a recovery-aware soft weight;
2. combine teacher entropy with teacher surprisal / sampled reverse KL and action validity rather than using entropy alone;
3. distinguish action-span entropy from whole-response top-$k$ head entropy;
4. explicitly control prompt truncation and zero-mask rows;
5. train a matched full-loss step-250 control with equal environment, token, optimizer, and evaluation budgets.

## Reproducibility assets

- Training parser and plots: `analyze_training.py`
- Evaluation parser and plots: `analyze_evaluation.py`
- Machine-readable summaries: `summary.json`, `evaluation_summary.json`
- Cleaned trajectory/turn panels: `trajectory_summary.csv`, `diagnostics_by_trajectory_turn.csv`, `frontier_aligned_rows.csv`
- Training dynamics: `training_overview.png`
- Frontier recovery: `frontier_mechanism.png`
- Latest-version heatmap: `teacher_entropy_frontier_heatmap_latest.png`
- Three-seed evaluation: `evaluation_three_seed_comparison.png`
- Frozen-baseline comparison: `evaluation_seed42_frozen_baselines.png`
- Task-type deltas: `evaluation_task_type_deltas.png`
- Pipeline timings: `pipeline_timing.png`

Training diagnostics source: `runs/experiments/entropy_adaptive_v1_step10_8gpu_s2t4_r16/diagnostics/trajectory_metrics.jsonl`.

Step-250 checkpoint source: `runs/experiments/entropy_adaptive_v1_step10_8gpu_s2t4_r16/checkpoints/ALFWORLD_ENTROPY_ADAPTIVE_V1/entropy_adaptive_v1_qwen25_step10_8gpu_s2t4_r16_seed42_20260809164432/global_step_250/actor/huggingface`.
