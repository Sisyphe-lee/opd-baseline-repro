#!/usr/bin/env bash
# Run paired Vanilla OPD on physical GPUs 0-3 in an isolated Ray cluster.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TCOD_SOURCE="${REPO_ROOT}/tcod_official"
VIRTUAL_ENV_ROOT=${VIRTUAL_ENV_ROOT:-/vepfs-mlp2/mlp-public/252302025/sjx/opd-workspaces/tcod-f2b-repro/.venv_tcod}
CONFIG_PATH=${CONFIG_PATH:-${REPO_ROOT}/configs/vanilla_opd_qwen25_3b_7b_paper_promptfix_4gpu.yaml}
RUN_ROOT=${RUN_ROOT:-${REPO_ROOT}/runs/2026-08-07_vanilla-opd-qwen25-3b-7b-paper-promptfix-4gpu-1s1t2train-v0}
GPU_LIST=${GPU_LIST:-0,1,2,3}
RAY_PORT=${RAY_PORT:-6381}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8266}
RAY_CLIENT_PORT=${RAY_CLIENT_PORT:-25003}
RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT:-20000}
RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-24999}
RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/ray_opd_0_3}
TASK_TMPDIR=${TASK_TMPDIR:-/dev/shm/tmp_opd_0_3}
OPD_RAY_ADDRESS=${OPD_RAY_ADDRESS:-127.0.0.1:${RAY_PORT}}

if [[ $(awk -F',' '{print NF}' <<<"${GPU_LIST}") -ne 4 ]]; then
  echo "This recipe requires exactly 4 visible GPUs; got ${GPU_LIST}" >&2
  exit 2
fi

if [[ "${GPU_LIST}" != "0,1,2,3" ]]; then
  echo "Safety check: this launcher is restricted to physical GPUs 0,1,2,3; got ${GPU_LIST}" >&2
  exit 2
fi

# Do not co-locate with another user's job merely because its instantaneous
# utilization is low. Every target GPU must have zero compute processes.
IFS=',' read -r -a physical_gpus <<<"${GPU_LIST}"
for physical_gpu in "${physical_gpus[@]}"; do
  gpu_pids=$(nvidia-smi --id="${physical_gpu}" \
    --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d')
  if [[ -n "${gpu_pids}" ]]; then
    echo "Safety check failed: physical GPU ${physical_gpu} is occupied by PID(s): ${gpu_pids//$'\n'/, }." >&2
    nvidia-smi --id="${physical_gpu}" \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader 2>/dev/null >&2 || true
    echo "Vanilla OPD was not started; no existing process was changed." >&2
    exit 3
  fi
done

for required in \
  "${VIRTUAL_ENV_ROOT}/bin/python" \
  "${VIRTUAL_ENV_ROOT}/bin/ray" \
  "${VIRTUAL_ENV_ROOT}/bin/trinity" \
  "${CONFIG_PATH}" \
  "${REPO_ROOT}/data/tcod_official_alfworld/train_expert.jsonl"; do
  if [[ ! -e ${required} ]]; then
    echo "Missing required Vanilla OPD asset: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${RUN_ROOT}/logs" "${RAY_TMPDIR}" "${TASK_TMPDIR}"

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export PYTHONPATH="${TCOD_SOURCE}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=0
export VLLM_RAY_PER_WORKER_GPUS=1
export VLLM_USE_RAY_SPMD_WORKER=1
export VLLM_USE_RAY_COMPILED_DAG=1
export VLLM_NO_USAGE_STATS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export WANDB_MODE=${WANDB_MODE:-offline}
export RAY_TMPDIR
export TMPDIR="${TASK_TMPDIR}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

if ! "${VIRTUAL_ENV_ROOT}/bin/ray" status --address="${OPD_RAY_ADDRESS}" >/dev/null 2>&1; then
  "${VIRTUAL_ENV_ROOT}/bin/ray" start \
    --head \
    --port="${RAY_PORT}" \
    --num-gpus=4 \
    --temp-dir="${RAY_TMPDIR}" \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --ray-client-server-port="${RAY_CLIENT_PORT}" \
    --min-worker-port="${RAY_MIN_WORKER_PORT}" \
    --max-worker-port="${RAY_MAX_WORKER_PORT}" \
    --disable-usage-stats
fi
export RAY_ADDRESS="${OPD_RAY_ADDRESS}"

cd "${REPO_ROOT}"
exec "${VIRTUAL_ENV_ROOT}/bin/trinity" run --config "${CONFIG_PATH}"
