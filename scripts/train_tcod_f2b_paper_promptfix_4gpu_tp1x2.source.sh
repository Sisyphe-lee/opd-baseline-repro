#!/usr/bin/env bash
# Copyright 2026 OPD ALFWorld contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TCOD_SOURCE="${REPO_ROOT}/tcod_official"
VIRTUAL_ENV_ROOT=${VIRTUAL_ENV_ROOT:-/vepfs-mlp2/mlp-public/252302025/sjx/opd-workspaces/tcod-f2b-repro/.venv_tcod}
CONFIG_PATH=${CONFIG_PATH:-${REPO_ROOT}/configs/tcod_f2b_qwen25_3b_7b_paper_promptfix_4gpu_tp1x2.yaml}
GPU_LIST=${GPU_LIST:-4,5,6,7}
RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/ray_tmp}

if [[ $(awk -F',' '{print NF}' <<<"${GPU_LIST}") -ne 4 ]]; then
  echo "This half-scale recipe requires exactly 4 visible GPUs; got ${GPU_LIST}" >&2
  exit 2
fi

for required in \
  "${VIRTUAL_ENV_ROOT}/bin/python" \
  "${VIRTUAL_ENV_ROOT}/bin/ray" \
  "${VIRTUAL_ENV_ROOT}/bin/trinity" \
  "${CONFIG_PATH}" \
  "${REPO_ROOT}/data/tcod_official_alfworld/train_expert.jsonl"; do
  if [[ ! -e ${required} ]]; then
    echo "Missing required reproduction asset: ${required}" >&2
    exit 2
  fi
done

RUN_ROOT="${REPO_ROOT}/runs/2026-08-07_tcod-f2b-qwen25-3b-7b-paper-promptfix-4gpu-1s1t2train-v0"
mkdir -p "${RUN_ROOT}/logs" "${RAY_TMPDIR}" /dev/shm/tmp

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export PYTHONPATH="${TCOD_SOURCE}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=0
export VLLM_RAY_PER_WORKER_GPUS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export VLLM_USE_RAY_SPMD_WORKER=1
export VLLM_USE_RAY_COMPILED_DAG=1
export VLLM_NO_USAGE_STATS=1
export WANDB_MODE=${WANDB_MODE:-offline}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

cd "${REPO_ROOT}"
if ! "${VIRTUAL_ENV_ROOT}/bin/ray" status >/dev/null 2>&1; then
  "${VIRTUAL_ENV_ROOT}/bin/ray" start --head --num-gpus=4 --temp-dir="${RAY_TMPDIR}"
fi
export RAY_ADDRESS=auto

exec "${VIRTUAL_ENV_ROOT}/bin/trinity" run --config "${CONFIG_PATH}"
