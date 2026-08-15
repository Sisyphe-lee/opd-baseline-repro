#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 4 )); then
  echo "Usage: $0 CONFIG CUDA_VISIBLE_DEVICES RUN_TAG [RAY_PORT]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$1"
GPU_IDS="$2"
RUN_TAG="$3"
RAY_PORT="${4:-6388}"

if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${ROOT}/${CONFIG}"
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Config does not exist: ${CONFIG}" >&2
  exit 2
fi
if [[ ! "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU list must look like 0,1,2,3: ${GPU_IDS}" >&2
  exit 2
fi
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_TAG may contain only letters, digits, dot, underscore, and dash." >&2
  exit 2
fi
if [[ ! "${RAY_PORT}" =~ ^[0-9]+$ ]]; then
  echo "RAY_PORT must be numeric: ${RAY_PORT}" >&2
  exit 2
fi

GPU_COUNT="$(awk -F, '{print NF}' <<<"${GPU_IDS}")"
CONFIG_GPU_COUNT="$("${ROOT}/.venv_tcod/bin/python" - "${CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
print(config["cluster"]["gpu_per_node"])
PY
)"
if [[ "${GPU_COUNT}" != "${CONFIG_GPU_COUNT}" ]]; then
  echo "GPU list has ${GPU_COUNT} devices, but config requests ${CONFIG_GPU_COUNT}." >&2
  exit 2
fi
TAG_HASH="$(printf '%s' "${RUN_TAG}" | sha256sum | cut -c1-8)"
# Ray places long session and socket names below --temp-dir. Keep this path short
# enough for Linux's 107-byte AF_UNIX socket limit; logs and results remain in
# the workspace, while Ray's transient runtime lives in shared memory.
RAY_TMP="/dev/shm/ray_tcod_${TAG_HASH}"
LOG_PATH="${ROOT}/logs/${RUN_TAG}.log"
CLIENT_PORT="$((RAY_PORT + 3615))"
DASHBOARD_PORT="$((RAY_PORT + 1878))"
# Keep the worker range separate from the client port and from the training
# launcher (which uses 36000-38999). Ray's default 10002-19999 range includes
# the client port for the 640x evaluation ports used by this repository.
WORKER_PORT_SLOT="$((RAY_PORT % 8))"
MIN_WORKER_PORT="$((39000 + WORKER_PORT_SLOT * 3000))"
MAX_WORKER_PORT="$((MIN_WORKER_PORT + 2999))"
RAY_SYSTEM_PORT_BASE="$((43000 + (RAY_PORT % 1000) * 10))"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_RAY_PER_WORKER_GPUS=1
export VLLM_USE_RAY_SPMD_WORKER=1
export VLLM_USE_RAY_COMPILED_DAG=1
export VLLM_NO_USAGE_STATS=1
VERL_OVERRIDE="${ROOT}/.runtime_overrides/verl"
if [[ -d "${VERL_OVERRIDE}/verl" ]]; then
  export PYTHONPATH="${VERL_OVERRIDE}:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
else
  export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

mkdir -p "${RAY_TMP}" "${ROOT}/logs"

cleanup() {
  local ray_pids
  ray_pids="$(pgrep -f "${RAY_TMP}" || true)"
  if [[ -n "${ray_pids}" ]]; then
    kill ${ray_pids} 2>/dev/null || true
    for _ in {1..20}; do
      sleep 0.1
      ray_pids="$(pgrep -f "${RAY_TMP}" || true)"
      [[ -z "${ray_pids}" ]] && return 0
    done
    # Ray's GCS server can ignore SIGTERM after a completed benchmark. Keep
    # cleanup scoped to this run's unique temp path, then force only survivors.
    kill -9 ${ray_pids} 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"${ROOT}/.venv_tcod/bin/ray" start \
  --head \
  --port="${RAY_PORT}" \
  --ray-client-server-port="${CLIENT_PORT}" \
  --dashboard-port="${DASHBOARD_PORT}" \
  --object-manager-port="$((RAY_SYSTEM_PORT_BASE + 0))" \
  --node-manager-port="$((RAY_SYSTEM_PORT_BASE + 1))" \
  --dashboard-agent-listen-port="$((RAY_SYSTEM_PORT_BASE + 2))" \
  --dashboard-agent-grpc-port="$((RAY_SYSTEM_PORT_BASE + 3))" \
  --runtime-env-agent-port="$((RAY_SYSTEM_PORT_BASE + 4))" \
  --metrics-export-port="$((RAY_SYSTEM_PORT_BASE + 5))" \
  --min-worker-port="${MIN_WORKER_PORT}" \
  --max-worker-port="${MAX_WORKER_PORT}" \
  --num-cpus="$((GPU_COUNT * 8))" \
  --num-gpus="${GPU_COUNT}" \
  --temp-dir="${RAY_TMP}" \
  --include-dashboard=false \
  --disable-usage-stats

export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
cd "${ROOT}"
"${ROOT}/.venv_tcod/bin/trinity" run --config "${CONFIG}" 2>&1 | tee "${LOG_PATH}"
