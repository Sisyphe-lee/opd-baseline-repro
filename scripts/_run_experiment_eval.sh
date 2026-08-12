#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "Full evaluation must run inside tmux." >&2
  exit 2
fi
if (( $# != 5 )); then
  echo "Usage: $0 CONFIG GPU_IDS RUN_TAG RAY_PORT RUN_ROOT" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$1"
GPU_IDS="$2"
RUN_TAG="$3"
RAY_PORT="$4"
RUN_ROOT="$5"

case "$(realpath -m "${CONFIG}")" in
  "${ROOT}"/configs/experiments/*) ;;
  *) echo "Config must be under configs/experiments: ${CONFIG}" >&2; exit 2 ;;
esac
case "$(realpath -m "${RUN_ROOT}")" in
  "${ROOT}"/runs/experiments/*) ;;
  *) echo "Run root must be under runs/experiments: ${RUN_ROOT}" >&2; exit 2 ;;
esac
if [[ ! "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+){3}$ ]]; then
  echo "Exactly four comma-separated GPU IDs are required: ${GPU_IDS}" >&2
  exit 2
fi

IFS=',' read -r -a gpu_array <<<"${GPU_IDS}"
for gpu_id in "${gpu_array[@]}"; do
  gpu_pids="$(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
  while IFS= read -r gpu_pid; do
    [[ -n "${gpu_pid}" ]] || continue
    if kill -0 "${gpu_pid}" 2>/dev/null; then
      if [[ "${ALLOW_OCCUPIED_EVAL_GPUS:-0}" == "1" ]]; then
        echo "WARNING: GPU ${gpu_id} is occupied by live PID ${gpu_pid}; continuing because ALLOW_OCCUPIED_EVAL_GPUS=1." >&2
        continue
      fi
      echo "GPU ${gpu_id} is occupied by live PID ${gpu_pid}." >&2
      exit 3
    fi
    echo "WARNING: GPU ${gpu_id} reports stale PID ${gpu_pid}; continuing." >&2
  done <<<"${gpu_pids}"
done

RECORD_DIR="${RUN_ROOT}/task_records"
RESULT_JSONL="${RUN_ROOT}/task_results.jsonl"
SUMMARY_JSON="${RUN_ROOT}/summary.json"
if [[ -e "${RECORD_DIR}" || -e "${RESULT_JSONL}" || -e "${SUMMARY_JSON}" ]]; then
  echo "Refusing to mix with existing output under ${RUN_ROOT}" >&2
  exit 2
fi

mkdir -p "${RECORD_DIR}" "${RUN_ROOT}/logs"
cp "${CONFIG}" "${RUN_ROOT}/eval_config.yaml"
cd "${ROOT}"
bash scripts/run_alfworld_eval.sh "${CONFIG}" "${GPU_IDS}" "${RUN_TAG}" "${RAY_PORT}"
"${ROOT}/.venv_tcod/bin/python" scripts/collect_alfworld_eval_results.py \
  --manifest data/eval_manifests/full_valid_seen.jsonl \
  --manifest data/eval_manifests/full_valid_unseen.jsonl \
  --record-dir "${RECORD_DIR}" \
  --output-jsonl "${RESULT_JSONL}" \
  --summary-json "${SUMMARY_JSON}" \
  --expected-count 274
echo "Completed experiment evaluation: ${SUMMARY_JSON}"
