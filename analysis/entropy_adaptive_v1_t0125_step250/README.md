# Adaptive v1, threshold 0.125: curriculum-profile figures

This directory contains the analysis requested after the threshold-0.125 step-250
training run.  Generate or refresh the assets with:

```bash
python analysis/plot_adaptive_v1_curriculum.py \
  --target-diagnostics runs/experiments/entropy_adaptive_v1_t0125_250step_8gpu_s2t4_r16/diagnostics/trajectory_metrics.jsonl \
  --target-threshold 0.125 \
  --output-dir analysis/entropy_adaptive_v1_t0125_step250
```

## Statistical definitions

`curriculum_imposed_loss_horizons.png` deliberately removes environment
termination from the comparison.  Vanilla OPD has horizon 30 for every trajectory.
TCOD F2B uses its configured one-based batch schedule
`K_b = min(1 + floor(b / 2), 30)` for 193 batches of 16 trajectories.  Each
Adaptive panel uses the recorded zero-based `entropy_frontier_turn` as the number
of turns retained before the crossing, or 30 if no frontier was detected.

`realized_trainable_turns_chronological.png` counts a turn only when
`loss_retained` is true and `truncate_status != "prompt_truncated"`.  Thus the plot
does not mistake prompt-length placeholders, whose action masks are all zero, for
trainable responses.  The black line is a centered rolling median spanning 2% of
trajectories.  For Vanilla and TCOD, the count is reconstructed from their frozen checkpoint
`pipeline_input` buffers: trajectories are keyed by batch/task/run, and a distinct
environment step counts only when its deserialized Experience has a non-empty
`action_mask` with at least one true token.  The two Adaptive panels use their
complete diagnostic records and additionally require `loss_retained`.

`teacher_entropy_frontier_heatmap_latest.png` is the prompt-truncation-corrected
latest-policy heatmap for threshold 0.125.  It subtracts each trajectory's mean
teacher entropy over the first three recorded turns, leaves prompt placeholders
blank, and marks the detected frontier and first prompt truncation separately.

## Reproducibility

- `curriculum_imposed_loss_horizons.csv`: all four imposed-horizon profiles.
- `realized_trainable_turns.csv`: all four evidence-backed chronological profiles.
- `plot_summary.json`: headline statistics and the threshold-0.175 replay assertion.
- `provenance.json`: diagnostic paths, SHA-256 hashes, row counts, buffer reconstruction rules, schemas, versions,
  and analytical baseline definitions.

The script refuses mixed or missing threshold values and must reproduce the known
threshold-0.175 headline statistics before it will emit the new comparison.
