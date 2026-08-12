# Adaptive v1 tau=0.125: 4-GPU versus 8-GPU training layout

## Result

Both fixed step-250 checkpoints were evaluated with the same frozen full274
protocol and the same four-engine evaluation layout at inference seeds 42, 43,
and 44.

| Evaluation seed | 8-GPU-trained checkpoint | 4-GPU-trained checkpoint | 4-GPU minus 8-GPU | Exact paired McNemar p |
|---:|---:|---:|---:|---:|
| 42 | 222/274 (81.02%) | 226/274 (82.48%) | +4 (+1.46 pp) | 0.608 |
| 43 | 223/274 (81.39%) | 227/274 (82.85%) | +4 (+1.46 pp) | 0.636 |
| 44 | 219/274 (79.93%) | 227/274 (82.85%) | +8 (+2.92 pp) | 0.243 |
| Descriptive three-seed total | 664/822 (80.78%) | 680/822 (82.73%) | +16 (+1.95 pp) | not used naively |

The same 274 tasks are repeated across evaluation seeds, so 822 rows are not
independent trials. The primary uncertainty analysis clusters by `game_file`,
keeping all three seed outcomes together. A 200,000-replicate task-clustered
bootstrap gives a 95% interval of [-1.09, 4.99] percentage points for the
overall difference. An exact task-cluster sign-flip test gives p=0.237.
Seen and unseen cluster-sign-flip p-values are 0.233 and 0.734, respectively.

Therefore the current evidence does not establish that the two checkpoints
have different expected full274 accuracy. The 4-GPU-trained checkpoint is
consistently higher in these three evaluation draws, but zero remains
compatible with the task-clustered interval.

These are evaluation seeds for two fixed seed-42 training runs. They measure
inference-sampling variability, not variability across independently trained
models. A claim about the causal effect of training layout requires matched
independent training seeds for both layouts.

Machine-readable values are in `comparison.json`; per-seed values are in
`per_seed_comparison.csv`.
