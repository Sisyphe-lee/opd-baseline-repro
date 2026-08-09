#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "Training must run inside tmux. Use scripts/launch_train_tmux.sh." >&2
  exit 2
fi
if (( $# != 2 )); then
  echo "Usage: $0 {tcod|vanilla} GPU_IDS" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="$1"
GPU_IDS="$2"
case "${MODE}" in
  tcod)
    CONFIG="${ROOT}/configs/train/tcod_f2b.yaml"
    RUN_ROOT="${ROOT}/runs/training/tcod_f2b"
    RAY_PORT=6380
    ;;
  vanilla)
    CONFIG="${ROOT}/configs/train/vanilla_opd.yaml"
    RUN_ROOT="${ROOT}/runs/training/vanilla_opd"
    RAY_PORT=6381
    ;;
  *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac

if [[ ! "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+){3}$ ]]; then
  echo "Exactly four comma-separated GPU IDs are required: ${GPU_IDS}" >&2
  exit 2
fi
IFS=',' read -r -a gpu_array <<<"${GPU_IDS}"
for gpu_id in "${gpu_array[@]}"; do
  gpu_pids="$(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
  if [[ -n "${gpu_pids}" ]]; then
    echo "GPU ${gpu_id} is occupied by PID(s): ${gpu_pids//$'\n'/, }." >&2
    exit 3
  fi
done

for required in "${ROOT}/.venv_tcod/bin/trinity" "${ROOT}/.venv_tcod/bin/ray" "${CONFIG}" "${ROOT}/data/tcod_official_alfworld/train_expert.jsonl"; do
  [[ -e "${required}" ]] || { echo "Missing baseline asset: ${required}" >&2; exit 2; }
done

RAY_TMPDIR="/dev/shm/ray_tcod_baseline_${MODE}"
TASK_TMPDIR="/dev/shm/tmp_tcod_baseline_${MODE}"
mkdir -p "${RUN_ROOT}/logs" "${RAY_TMPDIR}" "${TASK_TMPDIR}"

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

"${ROOT}/.venv_tcod/bin/ray" start --head --port="${RAY_PORT}" --num-gpus=4 \
  --temp-dir="${RAY_TMPDIR}" --include-dashboard=false --disable-usage-stats
export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
cd "${ROOT}"
"${ROOT}/.venv_tcod/bin/trinity" run --config "${CONFIG}" 2>&1 | tee "${RUN_ROOT}/logs/launcher.log"

