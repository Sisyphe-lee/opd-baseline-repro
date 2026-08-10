#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "Training must run inside tmux." >&2
  exit 2
fi
if (( $# != 4 )); then
  echo "Usage: $0 CONFIG GPU_IDS RAY_PORT RUN_ROOT" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$1"
GPU_IDS="$2"
RAY_PORT="$3"
RUN_ROOT="$4"

case "$(realpath -m "${CONFIG}")" in
  "${ROOT}"/configs/experiments/*) ;;
  *) echo "Config must be under configs/experiments: ${CONFIG}" >&2; exit 2 ;;
esac
case "$(realpath -m "${RUN_ROOT}")" in
  "${ROOT}"/runs/experiments/*) ;;
  *) echo "Run root must be under runs/experiments: ${RUN_ROOT}" >&2; exit 2 ;;
esac
if [[ ! "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU IDs must be a comma-separated list." >&2
  exit 2
fi
if [[ ! "${RAY_PORT}" =~ ^[0-9]+$ ]]; then
  echo "RAY_PORT must be numeric." >&2
  exit 2
fi

GPU_COUNT="$(awk -F, '{print NF}' <<<"${GPU_IDS}")"
CONFIG_GPU_COUNT="$("${ROOT}/.venv_tcod/bin/python" - "${CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    print(yaml.safe_load(handle)["cluster"]["gpu_per_node"])
PY
)"
if [[ "${GPU_COUNT}" != "${CONFIG_GPU_COUNT}" ]]; then
  echo "GPU list has ${GPU_COUNT} devices, but config requests ${CONFIG_GPU_COUNT}." >&2
  exit 2
fi

IFS=',' read -r -a gpu_array <<<"${GPU_IDS}"
for gpu_id in "${gpu_array[@]}"; do
  gpu_pids="$(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
  live_gpu_pids=()
  stale_gpu_pids=()
  while IFS= read -r gpu_pid; do
    [[ -n "${gpu_pid}" ]] || continue
    if kill -0 "${gpu_pid}" 2>/dev/null; then
      live_gpu_pids+=("${gpu_pid}")
    else
      stale_gpu_pids+=("${gpu_pid}")
    fi
  done <<<"${gpu_pids}"
  if (( ${#live_gpu_pids[@]} > 0 )); then
    echo "GPU ${gpu_id} is occupied by live PID(s): ${live_gpu_pids[*]}." >&2
    exit 3
  fi
  if (( ${#stale_gpu_pids[@]} > 0 )); then
    echo "WARNING: GPU ${gpu_id} reports stale PID(s): ${stale_gpu_pids[*]}; continuing." >&2
  fi
done

for required in "${ROOT}/.venv_tcod/bin/trinity" "${ROOT}/.venv_tcod/bin/ray" "${CONFIG}"; do
  [[ -e "${required}" ]] || { echo "Missing required asset: ${required}" >&2; exit 2; }
done

RAY_TMPDIR="/dev/shm/re_${RAY_PORT}"
TASK_TMPDIR="/dev/shm/te_${RAY_PORT}"
mkdir -p "${RUN_ROOT}/logs" "${RAY_TMPDIR}" "${TASK_TMPDIR}"
cp "${CONFIG}" "${RUN_ROOT}/config.yaml"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${ROOT}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V1=0
export VLLM_RAY_PER_WORKER_GPUS=1 VLLM_USE_RAY_SPMD_WORKER=1
export VLLM_USE_RAY_COMPILED_DAG=1 VLLM_NO_USAGE_STATS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TMPDIR="${TASK_TMPDIR}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

"${ROOT}/.venv_tcod/bin/ray" start --head --port="${RAY_PORT}" --num-gpus="${GPU_COUNT}" \
  --temp-dir="${RAY_TMPDIR}" --include-dashboard=false --disable-usage-stats
export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
cd "${ROOT}"
"${ROOT}/.venv_tcod/bin/trinity" run --config "${CONFIG}" \
  2>&1 | tee "${RUN_ROOT}/logs/launcher.log"
