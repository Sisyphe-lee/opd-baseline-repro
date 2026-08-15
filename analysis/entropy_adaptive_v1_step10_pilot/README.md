# Entropy-adaptive frontier v1: lightweight mechanism check

## Bottom line

This 10-update pilot does **not** provide evidence that retrospective
teacher-entropy prefix selection improves ALFWorld success. Across three paired
full274 evaluation seeds, adaptive obtains 69/822 successes and full-loss
control obtains 73/822 (8.39% versus 8.88%, -0.49 percentage points). The
per-seed differences are -2, -6, and +4 tasks, and none of the paired exact
McNemar tests is significant.

The mechanism measurements are still informative. The detector finds a real
local entropy spike, but the entropy does not stay high after the crossing.
Consequently, the strong rule "once entropy rises, all later teacher guidance is
unhelpful" is not supported by this run.

## Intervention

Both arms use the same Qwen2.5-3B student, frozen GiGPO Qwen2.5-7B teacher,
optimizer, full environment horizon 30, accumulated-memory prompt protocol,
strict action parser, batch sizes, seed, and 10 optimizer updates.

- **adaptive**: run the complete trajectory and score every student response
  with the teacher; retain only the prefix before the entropy frontier for
  training.
- **full**: run and score the same way, but retain every turn.

For each trajectory, let the first-three-turn response-level top-16 head entropy
mean be

$$
\bar H_0 = \frac{1}{3}\sum_{t=0}^{2} H_t.
$$

The first end-of-window turn $f$ satisfying

$$
\frac{1}{3}\sum_{j=f-2}^{f}(H_j-\bar H_0) \ge 0.175
$$

is the frontier. Training keeps turns $t<f$ (with a minimum retained prefix of
three turns). This is retrospective masking: it isolates data selection but
does not save rollout or teacher compute.

The entropy is not full-vocabulary entropy and is not action-only entropy. It is
the normalized entropy of the teacher's returned top-16 probability head,
averaged over all response tokens.

## Training diagnostics

| Metric | Adaptive | Full |
|---|---:|---:|
| Complete trajectories | 128 | 128 |
| Recorded turns | 3,636 | 3,737 |
| Frontier-triggered trajectories | 68 (53.1%) | 0 |
| Retained turns | 2,274 (62.5%) | 3,737 (100%) |
| Dropped turns | 1,362 (37.5%) | 0 |
| Prompt-truncated turns | 1,473 (40.5%) | 1,536 (41.1%) |
| Prompt-truncated retained turns | 702 | 1,536 |
| Mean valid-action rate, retained | 55.8% | 42.3% |
| Mean valid-action rate, dropped | 28.6% | n/a |

Among the 68 triggered trajectories, 62 later reached prompt truncation. The
frontier preceded the first truncated turn by a median of 9 turns. Of the 1,362
dropped turns, 771 (56.6%) were prompt-truncated placeholders and 591 were
non-truncated responses. Thus the rule is partly acting as an early
context-length/trajectory-degradation detector, not purely as a teacher
reliability detector.

For triggered trajectories, mean teacher entropy is 0.420 over baseline turns,
0.711 at the crossing, but only 0.324 over the complete dropped suffix. This
recovery directly warns against treating the first sustained spike as an
irreversible boundary. Suffix means are also affected by prompt-truncated
placeholder rows, so they should not be read as a clean counterfactual estimate.

## Evaluation

All evaluations use full274 (140 seen + 134 unseen), horizon 30, accumulated
memory, strict lowercase action parsing, temperature 0.4, top-p 1, top-k -1,
and a 512-token response cap. Seeds 42, 43, and 44 are paired within each row.

| Seed | Adaptive | Full | Difference | Adaptive-only | Full-only | McNemar p |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 24/274 | 26/274 | -2 | 17 | 19 | 0.868 |
| 43 | 20/274 | 26/274 | -6 | 14 | 20 | 0.392 |
| 44 | 25/274 | 21/274 | +4 | 21 | 17 | 0.627 |
| Descriptive total | 69/822 | 73/822 | -4 | 52 | 56 | n/a |

The descriptive total reuses the same 274 tasks across seeds, so its rows are
correlated and it is not assigned a naive pooled McNemar p-value. The direction
changes across seeds, and the per-seed difference ranges from -2.19 to +1.46
percentage points.

## Interpretation and next experiment

This pilot supports the weaker observation that entropy dynamics and late-turn
control degradation are heterogeneous across trajectories. It does not yet
establish stable per-task pacing: there is only one rollout per
task/model-version cell. It also does not show that suffix teacher guidance is
useless or that the current hard frontier improves downstream success.

The next version should not simply train longer with the same hard rule. First:

1. control context length (bounded structured history or an explicit
   non-truncated analysis panel);
2. replace irreversible first-crossing with a recovery-aware score, such as
   entropy AUC plus teacher surprisal/reverse-KL and action validity;
3. calibrate the threshold on Qwen2.5 rather than inheriting 0.175 from the old
   Qwen3 diagnostic;
4. test action-span entropy separately from response-level entropy;
5. use repeated rollouts for the same task and checkpoint before claiming a
   stable per-question curriculum.

Raw comparisons are under comparison/, comparison_seed43/, and
comparison_seed44/. Exact training/evaluation configs and complete diagnostic
JSONL files remain in their respective experiment run directories.
