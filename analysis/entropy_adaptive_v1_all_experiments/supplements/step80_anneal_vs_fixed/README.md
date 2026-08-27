# Step-80 Adaptive continuations versus TCOD F2B

This supplement plots the full model-version range 0--250 for fixed tau=0.1, linear annealing, immediate Vanilla, and TCOD F2B. The two step-80 branches reuse the original fixed-tau trajectory stream for versions 0--79 and switch to their own recorded continuation at version 80. It separates three effects:

1. the configured raw-threshold ramp;
2. the amount of loss masking actually imposed;
3. rollout/task-length variation introduced by the restarted asynchronous stream.

## Main finding

The linear threshold ramp is functionally too aggressive. Although explicit
`full` starts at model version 160, the threshold is already 0.2125 at version
90 and 0.325 at version 100. In versions 100--119, 99.1% of annealed
trajectories receive a full imposed horizon and the mean number of available
turns removed by masking is zero.

Across the complete post-step80 suffix, the annealed branch triggers a frontier
on only 31/1881 trajectories (1.65%), compared with 15.72% for fixed tau=0.1.
Its realized-turn mean (10.98) is essentially the same as immediate Vanilla
(11.00), rather than fixed tau=0.1 (9.03).

TCOD is reconstructed from the frozen step-250 buffer. Before the resume it uses the canonical pre-step80 buffer stream; from explorer batch 79 onward it is restricted to the actual resume branch by matching each batch to the model version in the resume log. TCOD reaches K=30 around model version 64. Before K=30 the buffer retains only trainable turns, so panel D deliberately leaves TCOD excluded-turn counts missing rather than incorrectly plotting zero. Its plotted rollout data end at model version 248 because the explorer stopped after producing the last batch used by the trainer; this is not a truncated training checkpoint.

The visible jump is not located exactly at the splice. It is strongest around
versions 90--99 / post-step80 trajectories 112--160. In that window the
annealed branch has 3.59 more realized turns than fixed tau=0.1. Of this gap,
2.37 turns come from reduced masking and 1.22 turns come from longer available
rollouts/task-stream variation.

Therefore the plot supports both statements: the ramp parameters release
masking too quickly, and part of the local spike is stochastic rollout
composition rather than the schedule alone.

## Reproduction

```bash
.venv_tcod/bin/python analysis/analyze_step80_anneal_vs_fixed.py
```

The exact curriculum metrics and input hashes are retained in `summary.json`, the two CSV files, and `provenance.json`.

## PPO training loss

`step80_branch_training_loss.png` compares `actor/final_loss` for fixed tau=0.1, linear annealing, and immediate Vanilla. The faint traces are raw per-step values; the solid traces are centered 11-step rolling means. Both continuation branches use the exact fixed-tau steps 1--80 as their common prefix and their own recorded steps 81--250 after the splice. The raw metrics, summary, and hashes are retained in `step80_branch_training_loss.csv`, `step80_branch_training_loss_summary.json`, and `step80_branch_training_loss_provenance.json`.

This is on-policy PPO loss over method-dependent samples and loss masks, so its absolute level is not an evaluation-accuracy metric.
