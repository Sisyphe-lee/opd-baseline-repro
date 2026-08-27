# Adaptive v1 step-80 branch experiment

## Objective

Test whether the early Adaptive v1 curriculum creates a favorable optimization
state that remains useful under later full-horizon Vanilla OPD, and whether
keeping a fixed entropy threshold after step 80 harms late-stage OPD.

## Frozen branch point

- Source run: `entropy_adaptive_v1_t0100_250step_4gpu_s1t1_r4_seed42`
- Source checkpoint: `global_step_80`
- Frozen full274 at step 80: 126/274 (45.99%)
- Full state retained: model, optimizer, LR scheduler, and per-rank RNG state
- Last recoverable Explorer state before the checkpoint: iteration 56 and task
  index 896
- The original run's in-flight iteration 57 and in-memory experience queue were
  not stored in the full checkpoint. Both new branches therefore restart from
  the same iteration-56/task-index-896 state with separate empty queues.

## Interventions

### Immediate full-horizon branch

Starting at model version 80, set `entropy_frontier_strategy: full`. This is the
strict Vanilla OPD loss-selection behavior: every non-truncated response in the
completed trajectory is returned for training. It does not approximate Vanilla
with a numerical threshold.

### Linear annealing branch

Keep the configured base threshold $t_*=0.100$ at model version 80, then raise
the effective threshold linearly until model version 160:

$$
t(s)=0.100+\frac{s-80}{80}(1-0.100), \qquad 80 < s < 160.
$$

At $s\ge160$, switch the effective strategy explicitly to `full`; the code does
not continue using $t=1$ as a proxy for Vanilla.

## Matched controls

The two branches keep the same step-80 model, optimizer, LR scheduler, RNG
state, 4-GPU layout, training seed, train batch size, rollout batch size,
student/teacher models, sampling temperature, loss aggregation, KL coefficient,
staleness limit, task stream position, and total target step 250. They differ
only in post-step-80 loss selection.

## Evaluation and interpretation

Each branch receives one seed-42 frozen full274 evaluation at step 250 using
horizon 30, accumulated memory, strict action parsing, temperature 0.4, and a
512-token response limit.

- Immediate full versus the original from-scratch Vanilla step-250 result tests
  whether the Adaptive warm-up state is associated with a better later
  full-horizon OPD endpoint.
- Annealing versus immediate full tests whether a gradual release after step 80
  is better than an abrupt release from the same checkpoint.
- Either branch versus the original fixed-$t=0.100$ endpoint is evidence about
  whether fixed masking hurts late OPD, but this comparison is not as clean as
  the direct branch-to-branch comparison because the original continuation kept
  an in-flight queue state that was not checkpointed.
- With one training/evaluation seed, conclusions are descriptive evidence for
  mechanism validation, not a population-level estimate of seed variance.
