# Vanilla OPD diagnostics snapshot

Status: **complete**

Diagnostics steps: 1–500

Student model-version range: 1–500

Top-k head: 16
Plotted complete steps: 500; partial steps excluded from plots: none
Trajectory-turn view: 8000 trajectories; turns 0–29

## Metric contract

- `student_entropy_topk_mean` and `teacher_entropy_topk_mean` are the mean entropy of the returned top-16 logprob head. They are not full-vocabulary entropy.
- `entropy_curve.png` is the canonical hypothesis-facing chart: its horizontal axis is the **within-trajectory environment turn**, not training step. The denominator panel makes late-turn survivorship visible.
- `entropy_by_model_version.png` is the separate training-evolution chart. Each point first collapses all turns and trajectories in one Explorer batch, then the displayed line applies a 5-step trailing mean.
- `teacher_entropy_by_outcome_progress.png` linearly interpolates each trajectory onto a shared 0–1 progress grid and equal-weights trajectories. This controls the raw-turn chart's length/survivorship effect and splits by final audited `task_success`.
- `teacher_entropy_frontier_heatmap.png` keeps every rollout separate, places failures above successes, subtracts each rollout's own early baseline, and sorts within outcome by first sustained threshold crossing.
- `teacher_entropy_threshold_crossing_{all,failure,success}.png` show when pooled and outcome-specific rollouts cross several online-compatible ΔH thresholds; termination before crossing is treated as right censoring.
- `teacher_entropy_observation_boundary.png` is the original raw schema-v2 alignment. Its previous-last/new-first contrast mixes observation effects with response-token position and should not be interpreted alone.
- `fixed_panel_same_task_variability.png` uses repeated stochastic rollouts of the same fixed task and checkpoint; fixed-panel rows are evaluation-only and never enter the training buffer.
- `fixed_panel_same_task_effects.png` fits an equal-trajectory-weighted progress model with game and Student-version fixed effects and task-cluster uncertainty; `failure × progress` tests whether failed rollouts drift faster after conditioning on the repeated task/checkpoint panel.
- `fixed_panel_failure_prediction.png` reports leave-one-game-out turn-5/10/15 landmark prediction. Only trajectories still active at each cutoff enter that risk set; final-failure predictiveness does not imply that the trajectory has no distillation value.
- `fixed_panel_teacher_entropy_boundary_deconfounded.png` compares the raw previous-last/new-first jump with a first-block-to-first-block position control, then stratifies the controlled change by previous action category and new-observation length.
- `fixed_panel_teacher_entropy_action_boundary.png` uses exact token alignment to the parsed `<action>…</action>` span when `--tokenizer-path` is supplied.
- `fixed_panel_frontier_by_checkpoint.png` tracks the same fixed tasks across checkpoints using observed cumulative crossing incidence. Unlike the pooled Kaplan–Meier-style training plots, termination without crossing remains non-crossing.
- `trajectory_summary.csv` contains paired within-trajectory Teacher entropy deltas and slopes. Last-5 minus first-5 is only defined for trajectories with at least 10 turns.
- `sampled_reverse_kl_token_weighted` is the response-token-weighted mean of `log p_student(token) - log p_teacher(token)` on student-sampled tokens. Its expectation is the sampled reverse KL `D_KL(student || teacher)`, but it is not full-distribution KL. Schema-v2 writes the correct reverse-KL name and retains the historical forward-KL alias for compatibility.
- `reverse_kl_loss_curve.png` separates this direct sampled reverse-KL estimate from `actor/final_loss`, which is the Trainer's PPO surrogate loss.
- Surprisal is `-log p(token)` for the sampled response token.
- Top-k mass is the sum of probabilities represented by the returned top-k head. Its near-one value is a coverage diagnostic, not a quality score.
- Success rate is trajectory-level: rows are deduplicated by `(diagnostics_step, task_id, run_id)` and the trajectory is successful if any row reports the audited `task_success` flag.
- `training_step` in the JSONL is the Explorer `batch_id`. When `explorer.log` is available, model-version charts add the corresponding student `model_version`; trajectory-turn charts never use it as their horizontal axis.
- `rollout_wait_explore_step_sec` is Explorer collection-step wall time. It can include waiting for the step and monitor/evaluation work; it is not a pure single-environment execution time.

## Coverage and source handling

The diagnostics JSONL files are read in the order supplied on the command line. If multiple files contain the same `(diagnostics_source, training_step)`, the last file owns that whole source-step. Training and fixed-panel rows at the same step therefore cannot overwrite one another. Malformed final lines are ignored and counted in `summary.json`, which makes the script safe to run while training is still appending JSONL.

Before trajectory charts are drawn, the script fails fast on mixed top-k definitions, unexpected diagnostics kinds, duplicate/missing turns, missing entropy values, or inconsistent final outcomes. The tables contain extra guardrail metrics including valid-action rate, timeout/lost rate, mean environment rounds, response-token volume, entropy/surprisal gaps, and trainer timing when the corresponding logs are supplied.
