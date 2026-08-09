# TCOD baseline operating rules

This directory is the canonical baseline for all subsequent TCOD development.

## Scientific identity

- The paper being reproduced is arXiv:2604.24005: `https://arxiv.org/abs/2604.24005`.
- Do not invent or substitute another paper title, method, repository, or result.
- The local student is Qwen2.5-3B-Instruct. The local teacher is the downloaded
  GiGPO-Qwen2.5-7B-Instruct-ALFWorld RL model.
- This baseline intentionally fixes the public code's step-2 prompt assembly bug.
- The frozen evaluation is full274, horizon 30, accumulated chat memory, strict
  lowercase `<action>...</action>` parsing, and a 512-token response limit.
- Training uses `loss_agg_mode: seq-mean-token-mean`. There is no SFT stage.

Read `BASELINE_SPEC.md` and `VALIDATION.md` before changing training or evaluation.

## Execution and provenance

- Every training job must be launched in tmux. Use
  `scripts/launch_train_tmux.sh`; do not launch training in a transient shell.
- Long full evaluations should use `scripts/launch_eval_tmux.sh`.
- Do not modify frozen files under `configs/train`, `configs/eval`,
  `checkpoints`, or `results`. Copy a config into `configs/experiments/` and use
  a new output directory for development experiments.
- Never mix partial records into a completed evaluation directory.
- Keep all new code, configs, logs, checkpoints, and results under this
  `tcod-baseline` directory.
- Unless an experiment config or experiment README explicitly documents an
  exemption, every future training run must enable and retain the plotting
  diagnostics needed by `research_tools/legacy_entropy_diagnostics`: per-turn
  teacher/student entropy, surprisal/KL, token-block data, task outcome,
  model-version identity, and the complete `trajectory_metrics.jsonl`.
- The default diagnostic workflow settings are `diagnostics_enabled: true`,
  positive `diagnostics_top_k` (normally 16), `diagnostics_required: true`, and
  a `diagnostics_path` inside that experiment's output directory. Keep launcher
  logs, TensorBoard events, the exact config, and plotting provenance with the
  diagnostic JSONL. Do not delete or truncate these assets after training.
- A diagnostics exemption must be decided before launch and state both the
  reason and which downstream figures will become unavailable. Silence or
  storage pressure alone is not an implicit exemption.
- The sibling source directories are migration sources. Do not delete either
  source directory unless the user explicitly approves deletion after reviewing
  the completed manifest and validation report.

