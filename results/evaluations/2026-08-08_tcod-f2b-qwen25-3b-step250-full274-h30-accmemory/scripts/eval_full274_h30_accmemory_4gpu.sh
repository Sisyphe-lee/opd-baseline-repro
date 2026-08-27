#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TCOD_RUNTIME_ROOT=${TCOD_RUNTIME_ROOT:-/vepfs-mlp2/mlp-public/252302025/sjx/opd-workspaces/tcod-f2b-repro}
VIRTUAL_ENV_ROOT=${VIRTUAL_ENV_ROOT:-${TCOD_RUNTIME_ROOT}/.venv_tcod}
CONFIG=${RUN_ROOT}/configs/eval_full274_h30_accmemory_4gpu.yaml
EVAL_ROOT=${RUN_ROOT}/evaluation/full274_h30
RECORD_DIR=${EVAL_ROOT}/task_records
RESULT_JSONL=${EVAL_ROOT}/task_results.jsonl
SUMMARY_JSON=${EVAL_ROOT}/summary.json
LOG_PATH=${RUN_ROOT}/logs/eval_full274_h30_accmemory_4gpu.log
MANIFEST_ROOT=/vepfs-mlp2/mlp-public/252302025/sjx/opd-workspaces/opd-alfworld-sync-repro/runs/2026-08-08_tcod-f2b-qwen25-3b-step250-full274-h30-paper/manifests
GPU_IDS=${1:-0,1,2,3}
RAY_PORT=${2:-6695}
RUN_TAG=f2b_qwen25_3b_step250_full274_h30_accmemory_4gpu_20260808

if [[ ${GPU_IDS} != "0,1,2,3" ]]; then
  echo "Safety check: this evaluation is restricted to physical GPUs 0,1,2,3." >&2
  exit 2
fi

for gpu_id in 0 1 2 3; do
  gpu_pids=$(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')
  if [[ -n ${gpu_pids} ]]; then
    echo "GPU ${gpu_id} already has compute process(es): ${gpu_pids}" >&2
    exit 3
  fi
done

if [[ -e ${RECORD_DIR} || -e ${RESULT_JSONL} || -e ${SUMMARY_JSON} ]]; then
  echo "Refusing to mix with existing or partial results under ${EVAL_ROOT}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/logs" "${RECORD_DIR}"
"${TCOD_RUNTIME_ROOT}/scripts/run_alfworld_eval.sh" "${CONFIG}" "${GPU_IDS}" "${RUN_TAG}" "${RAY_PORT}" 2>&1 | tee "${LOG_PATH}"

"${VIRTUAL_ENV_ROOT}/bin/python" "${TCOD_RUNTIME_ROOT}/scripts/collect_alfworld_eval_results.py" \
  --manifest "${MANIFEST_ROOT}/full_valid_seen.jsonl" \
  --manifest "${MANIFEST_ROOT}/full_valid_unseen.jsonl" \
  --record-dir "${RECORD_DIR}" \
  --output-jsonl "${RESULT_JSONL}" \
  --summary-json "${SUMMARY_JSON}" \
  --expected-count 274

echo "Completed accumulated-memory evaluation: ${SUMMARY_JSON}"
