# THUNLP/OPD Baseline Reproduction

This repository is a thin, reproducible overlay for the response-only OPD baseline from
[THUNLP/OPD](https://github.com/thunlp/OPD). It pins upstream code, installs the tested
runtime, downloads the exact student and teacher, validates multi-GPU execution, and
provides separate three-step and full-training launchers.

It intentionally does not vendor upstream source, model weights, datasets, checkpoints,
or logs.

Project authority and the exact handoff state are recorded in
[BASELINE_DECISIONS.md](BASELINE_DECISIONS.md) and [SETUP_STATUS.md](SETUP_STATUS.md).

## Frozen baseline

- Upstream: `thunlp/OPD@4532fd35ccfdde82adc918b265e4c964534e83d1`
- Student: `Qwen/Qwen3-1.7B-Base`
- Teacher: `lllyx/Qwen3-4B-Base-GRPO`
- Training data: upstream `datasets/dapo-math-17k.parquet`
- Validation data: upstream `datasets/test_data/AIME24/test.parquet`
- Algorithm: upstream `token_reward_direct` response-only OPD
- Target machine: 4 x A100 80GB (8 GPUs may be present; only four are required)
- Sequence contract: prompt 1024, generated response 7168, model length 8192
- Batch contract: 64 prompts, 4 rollouts per prompt, PPO mini-batch 64
- Precision: BF16 actor, reference, and teacher

This is a hardware-adapted reproduction rather than a byte-for-byte execution of the
upstream cluster launcher: it uses four A100s and BF16, while the checked-in upstream
launcher requests eight GPUs and defaults to FP32. The algorithm, dataset, sequence
lengths, global batch, rollouts, and optimizer settings are kept fixed.

## One-time setup

```bash
git clone https://github.com/Sisyphe-lee/opd-baseline-repro.git
cd opd-baseline-repro

CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/prepare_a100.sh
```

The scripts honor normal proxy and package-manager environment variables. For a mirror,
set variables such as `HF_ENDPOINT` or `PIP_INDEX_URL` before invoking them.
`prepare_a100.sh` is resumable: rerunning it reuses the conda prefix, upstream checkout,
and completed Hugging Face downloads.

## Three-step closure

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/run_smoke_4xa100.sh
./scripts/verify_run.py
```

`verify_run.py` selects the newest smoke run unless a run directory is supplied. It
requires all three optimizer steps, a success marker, and a complete actor checkpoint.

## Full baseline

Start the full one-epoch baseline only after the smoke validator passes:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/run_full_4xa100.sh
```

The launcher prints and records a timestamped artifact directory. To run a bounded job
instead of a full epoch, override `TOTAL_TRAINING_STEPS`, for example:

```bash
TOTAL_TRAINING_STEPS=100 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  ./scripts/run_full_4xa100.sh
```

## Useful overrides

Every machine-local path and important resource value is an environment override:

```bash
THUNLP_OPD_DIR=/path/to/OPD \
STUDENT_MODEL_PATH=/path/to/Qwen3-1.7B-Base \
TEACHER_MODEL_PATH=/path/to/Qwen3-4B-Base-GRPO \
PYTHON_BIN=/path/to/python \
ARTIFACT_ROOT=/path/to/artifacts \
CUDA_VISIBLE_DEVICES=2,3,4,5 \
./scripts/run_smoke_4xa100.sh
```

Set `CONFIG_ONLY=1` on either launcher to compose and print the Hydra configuration
without starting Ray or allocating model weights.

## Local qualification evidence

Before publication, the same overlay configuration completed three optimizer steps on
six local 48GB GPUs with finite losses and wrote a six-rank actor checkpoint at step 3.
The 48GB qualification used an 8192-token per-rank dynamic budget to avoid a full-vocab
log-softmax memory peak. The A100 launcher restores the 32768-token budget.

This proves the training path closes; it is not yet a scientific reproduction of the
paper result. The next gate is the three-step A100 smoke, followed by the full baseline.
