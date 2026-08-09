# Frozen baseline specification

## Scope

The baseline compares TCOD-F2B and prompt-fixed Vanilla OPD on ALFWorld using
the same Qwen2.5-3B student initialization, GiGPO Qwen2.5-7B teacher, data,
optimizer, batch settings, training length, and evaluator. Neither run includes
SFT. The teacher choice and prompt fix are deliberate local reproduction choices.

## Training contract

| Setting | Frozen value |
|---|---|
| Student | Qwen2.5-3B-Instruct |
| Teacher | GiGPO-Qwen2.5-7B-Instruct-ALFWorld |
| Steps | 250 |
| Explorer batch | 16 |
| Train batch | 64 |
| Repeat times | 1 |
| Student rollouts per task occurrence | 1 |
| Optimizer LR | 1e-6 |
| Loss aggregation | `seq-mean-token-mean` |
| Max staleness | 2 |
| Student response limit | 512 |
| Teacher response limit | 512 |
| Environment horizon | 30 |
| Hardware layout | 1 student GPU, 1 teacher GPU, 2 FSDP trainer GPUs |

TCOD uses `TCOD_f2b_alfworld_workflow`. Vanilla uses
`OPD_promptfix_alfworld_workflow`. Both retain accumulated chat memory. The
prompt-fixed code switches to the history-bearing template after the first
transition so the second environment step does not lose the task context.

### Future-training diagnostics retention

The frozen runs above predate token-level diagnostics, so this rule is not
retroactive. For every new training experiment derived from this baseline,
plotting diagnostics are mandatory by default: enable the instrumented
workflow, set `diagnostics_enabled: true`, a positive `diagnostics_top_k`
(normally 16), `diagnostics_required: true`, and an experiment-local
`diagnostics_path`. Retain the complete `trajectory_metrics.jsonl`, launcher
logs, TensorBoard events, exact config, checkpoint/model-version mapping, and
plotting provenance.

An experiment may omit these assets only when its config or README records the
exemption before launch, explains why, and lists the entropy/KL/trajectory
figures that will no longer be reproducible. Diagnostic data must not be
silently deleted or truncated after training.

## Evaluation contract

| Setting | Frozen value |
|---|---|
| Tasks | 140 valid-seen + 134 valid-unseen |
| Horizon | 30 |
| Temperature | 0.4 |
| top-p / top-k | 1.0 / -1 |
| Seed | 42 |
| Response limit | 512 |
| Prompt | corrected step-2 assembly |
| Chat memory | accumulated |
| Action parser | strict lowercase `<action>...</action>` |
| Success | ALFWorld `won` or positive task reward |

The evaluator uses `eval_utils.py`, while training workflows continue to use
the public training `utils.py`. This separation preserves both already-tested
runtime paths and prevents evaluation-only `won/lost` instrumentation from
silently changing training.

## Differences from a literal paper run

1. The local teacher is the downloaded GiGPO RL checkpoint; the paper reports a
   GRPO-trained Qwen2.5-7B teacher.
2. The public step-2 prompt assembly bug is fixed locally.
3. The frozen local response limit is 512 rather than the paper's stated 4096.
4. The local training config explicitly uses `seq-mean-token-mean`; the public
   YAML does not explicitly freeze that option.

Consequently, this folder is the controlled local baseline for future
development. Comparisons against the paper must continue to disclose these
differences.


