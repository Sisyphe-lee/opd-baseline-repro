# Experiment results

Updated: 2026-08-06

## Current status

The trusted result registry was reset after evaluation contracts and metric
semantics were found to be inconsistent across earlier runs. The evaluator was
then audited and corrected: TextWorld's default 50-step wrapper returns
`done=True` on timeout, so `done` cannot be used as success. Canonical results
use `task_success = won or positive_task_reward` and record timeout separately.

The three student step-200 checkpoints below were evaluated on the same complete
274-task population and have persisted per-task records. This full-274 comparison
supersedes the earlier quick-72 TCOD step-250 result in this registry.

## Canonical results

| Training run | Checkpoint | Evaluation tier | Seen SR | Unseen SR | Macro SR | Status |
|---|---|---|---:|---:|---:|---|
| External ATOD GRPO teacher | `model_ckpt/teacher_model` (Qwen3-4B step-150) | quick-72, h50 | 19/36 = 52.78% | 18/36 = 50.00% | 51.39% | Accepted aggregate |
| `2026-08-05_vanilla-opd-diagnostics-v2-fixedpanel20` | Vanilla OPD Qwen3-1.7B step-200 | full-274, h50 | 44/140 = 31.43% | 26/134 = 19.40% | 25.42% | Accepted paired full result |
| `2026-08-04_tcod-f2b-qwen3-1.7b-step250` | TCOD-F2B Qwen3-1.7B step-200 | full-274, h50 | 41/140 = 29.29% | 35/134 = 26.12% | 27.70% | Accepted paired full result |
| `2026-08-06_entropy-frontier-opd-dh0175-resume60-fixedpanel-optin` | Entropy-frontier OPD, hard cutoff $\Delta H=0.175$, Qwen3-1.7B step-200 | full-274, h50 | 36/140 = 25.71% | 27/134 = 20.15% | 22.93% | Accepted paired full result |

The teacher row uses the quick-72 tier and is not directly compared with the
three full-274 student rows. Among the matched student evaluations, TCOD has the
highest point estimate: +2.29 macro percentage points over Vanilla OPD and +4.77
points over the entropy-frontier hard cutoff. The overall, task-weighted success
counts are 70/274 (25.55%) for Vanilla OPD, 76/274 (27.74%) for TCOD, and 63/274
(22.99%) for the entropy-frontier method.

Exact paired McNemar tests do not show a conventional $p<0.05$ difference on
this single evaluation pass: Vanilla OPD versus TCOD has $p=0.488$ (23
Vanilla-only successes and 29 TCOD-only successes), Vanilla OPD versus entropy
frontier has $p=0.401$ (29 versus 22), and TCOD versus entropy frontier has
$p=0.066$ (28 versus 15). Thus the ordering is useful evidence for method design,
but the 2--5 point gaps should not be treated as established population-level
improvements from one checkpoint and one sampling seed.

### Evidence

- Teacher config: `reproduction_configs/alfworld_quick72_h50_teacher_step150_4gpu.yaml`
- Teacher benchmark log:
  `reproduction_outputs/TCOD-F2B-REPRO/quick72-h50-512/teacher-qwen3-4b-grpo-step150-4gpu-metricfix-v2/log/benchmark.log`
- Teacher execution: four TP=1 A100 replicas; 178.9 seconds of evaluation time.
- Metric validation: a forced 50-step failure returned
  `done=True, reward=0, won=False`; an expert success returned
  `done=True, reward=1, won=True`. The evaluator counts only the latter.

Full-274 step-200 evidence:

- Evaluation ID: `full274_h50_step200_seed42`; all three evaluations use the
  frozen full-274 manifests, temperature 0.4, top-p 1.0, top-k -1, a 50-action
  horizon, and four TP=1 A100 replicas.
- Vanilla OPD checkpoint:
  `reproduction_outputs/vanilla_opd_diagnostics_v2_fixedpanel20/TCOD-F2B-REPRO/alfworld-vanilla-opd-diagnostics-v2/qwen3-1p7b-teacher-qwen3-4b-vanilla-opd-fixedpanel20/global_step_200/actor/huggingface/`
- TCOD checkpoint:
  `reproduction_outputs/tcod_f2b_qwen3_1p7b_teacher_step150_official/TCOD-F2B-REPRO/alfworld-official/qwen3-1p7b-teacher-qwen3-4b-grpo-step150_20260804070926/global_step_200/actor/huggingface/`
- Entropy-frontier checkpoint:
  `reproduction_outputs/entropy_frontier_opd_dh0175_fixedpanel20/TCOD-F2B-REPRO/alfworld-entropy-frontier-opd/qwen3-1p7b-teacher-qwen3-4b-entropy-frontier-dh0175/global_step_200/actor/huggingface/`
- Each training run stores its exact evaluator config, log, aggregate summary,
  274-row `task_results.jsonl`, and individual trajectory records under
  `runs/<run-id>/evaluation/full274_step200/`.
- Approximate end-to-end wall times from evaluator initialization to completion
  were 9m09s for Vanilla OPD, 8m55s for TCOD, and 9m12s for entropy frontier.
- The aggregate summaries and paired contingency tables are collected in
  `runtime/analysis/full274_step200_triplet/comparison.json`.

## Admission rule

A routine-iteration aggregate is added here only when all of the following are
available:

1. exact training run and checkpoint identity;
2. frozen quick-72 or full-274 manifest hashes;
3. complete inference contract and evaluator config;
4. seen/unseen `task_success` derived from ALFWorld `won` or positive reward;
5. aggregate denominators reconciled with the frozen manifests;
6. config, launcher, key log, and result summary stored in the training run.

Final publication reporting additionally requires persisted per-task JSONL so
task identities, task-type slices, and paired differences can be audited. The
three full-274 student results satisfy this persistence requirement. The external
teacher quick-72 row remains an accepted aggregate qualification result and is
not part of the paired full-274 comparison.

Historical numbers are retained under `Docs/legacy/` for diagnosis only. They are
not citable baselines.
