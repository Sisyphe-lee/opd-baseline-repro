#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "Two-stage training must run inside tmux." >&2
  exit 2
fi
if (( $# != 4 )); then
  echo "Usage: $0 CONFIG GPU_IDS RUN_TAG RAY_PORT" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$1"
GPU_IDS="$2"
RUN_TAG="$3"
RAY_PORT="$4"
[[ "${CONFIG}" == /* ]] || CONFIG="${ROOT}/${CONFIG}"
[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 2; }
[[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Invalid GPU list" >&2; exit 2; }
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid run tag" >&2; exit 2; }
[[ "${RAY_PORT}" =~ ^[0-9]+$ ]] || { echo "Invalid Ray port" >&2; exit 2; }

GPU_COUNT="$(awk -F, '{print NF}' <<<"${GPU_IDS}")"
CONFIG_GPU_COUNT="$("${ROOT}/.venv_tcod/bin/python" - "${CONFIG}" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["cluster"]["gpu_per_node"])
PY
)"
[[ "${GPU_COUNT}" == "${CONFIG_GPU_COUNT}" ]] || {
  echo "GPU list has ${GPU_COUNT}, config requests ${CONFIG_GPU_COUNT}." >&2
  exit 2
}

TAG_HASH="$(printf '%s' "${RUN_TAG}" | sha256sum | cut -c1-8)"
RAY_TMP="/dev/shm/ray_tsd_${TAG_HASH}"
CLIENT_PORT="$((RAY_PORT + 3615))"
DASHBOARD_PORT="$((RAY_PORT + 1878))"
WORKER_PORT_SLOT="$((RAY_PORT % 8))"
MIN_WORKER_PORT="$((36000 + WORKER_PORT_SLOT * 3000))"
MAX_WORKER_PORT="$((MIN_WORKER_PORT + 2999))"
RAY_SYSTEM_PORT_BASE="$((43000 + (RAY_PORT % 1000) * 10))"
LOG_PATH="${ROOT}/runs/experiments/two_stage_distillation/logs/${RUN_TAG}.log"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${ROOT}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V1=0
export VLLM_RAY_PER_WORKER_GPUS=1 VLLM_USE_RAY_SPMD_WORKER=1
export VLLM_USE_RAY_COMPILED_DAG=1 VLLM_NO_USAGE_STATS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
mkdir -p "${RAY_TMP}" "$(dirname "${LOG_PATH}")"

cleanup() {
  local ray_pids
  ray_pids="$(pgrep -f "${RAY_TMP}" || true)"
  if [[ -n "${ray_pids}" ]]; then
    kill ${ray_pids} 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"${ROOT}/.venv_tcod/bin/ray" start --head \
  --port="${RAY_PORT}" \
  --ray-client-server-port="${CLIENT_PORT}" \
  --dashboard-port="${DASHBOARD_PORT}" \
  --object-manager-port="$((RAY_SYSTEM_PORT_BASE + 0))" \
  --node-manager-port="$((RAY_SYSTEM_PORT_BASE + 1))" \
  --dashboard-agent-listen-port="$((RAY_SYSTEM_PORT_BASE + 2))" \
  --dashboard-agent-grpc-port="$((RAY_SYSTEM_PORT_BASE + 3))" \
  --runtime-env-agent-port="$((RAY_SYSTEM_PORT_BASE + 4))" \
  --metrics-export-port="$((RAY_SYSTEM_PORT_BASE + 5))" \
  --min-worker-port="${MIN_WORKER_PORT}" --max-worker-port="${MAX_WORKER_PORT}" \
  --num-cpus="$((GPU_COUNT * 8))" --num-gpus="${GPU_COUNT}" \
  --temp-dir="${RAY_TMP}" --include-dashboard=false --disable-usage-stats
export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
cd "${ROOT}"
"${ROOT}/.venv_tcod/bin/trinity" run --config "${CONFIG}" 2>&1 | tee "${LOG_PATH}"
