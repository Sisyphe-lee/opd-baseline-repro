#!/usr/bin/env bash
set -euo pipefail

TCOD_ROOT=/vepfs-mlp2/mlp-public/252302025/sjx/opd-workspaces/tcod-f2b-repro
RUN_ROOT=/vepfs-mlp2/mlp-public/252302025/sjx/opd-workspaces/opd-alfworld-sync-repro/runs/2026-08-07_gigpo-qwen2p5-7b-native-h50-historyfix
CONFIG="${RUN_ROOT}/configs/eval_full274_h50_historyfix_8gpu.yaml"
EVAL_ROOT="${RUN_ROOT}/evaluation/full274_h50"
RECORD_DIR="${EVAL_ROOT}/task_records"
RESULT_JSONL="${EVAL_ROOT}/task_results.jsonl"
SUMMARY_JSON="${EVAL_ROOT}/summary.json"
LOG_PATH="${RUN_ROOT}/logs/eval_full274_h50_historyfix_8gpu.log"
GPU_IDS=${1:-0,1,2,3,4,5,6,7}
RAY_PORT=${2:-6710}
RUN_TAG=full274_h50_gigpo_qwen2p5_7b_historyfix_8gpu_20260807

if [[ "$(sha256sum "${TCOD_ROOT}/reproduction_data/alfworld/full_valid_seen.jsonl" | awk '{print $1}')" != "3f93167b4da2d68e789409785c9a328e0cf55a3b40bd60165cbd537011e460d3" ]]; then
  echo "Full-274 seen manifest checksum mismatch" >&2
  exit 2
fi
if [[ "$(sha256sum "${TCOD_ROOT}/reproduction_data/alfworld/full_valid_unseen.jsonl" | awk '{print $1}')" != "60c3edd7e9fba923d5befe5e88af424967b79c6d325eead4634a88a396e37662" ]]; then
  echo "Full-274 unseen manifest checksum mismatch" >&2
  exit 2
fi

IFS=',' read -r -a gpu_array <<<"${GPU_IDS}"
if [[ "${#gpu_array[@]}" -ne 8 ]]; then
  echo "Expected exactly eight GPU IDs, got ${GPU_IDS}" >&2
  exit 2
fi
for gpu_id in "${gpu_array[@]}"; do
  gpu_pids="$(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
  if [[ -n "${gpu_pids}" ]]; then
    echo "GPU ${gpu_id} already has compute process(es): ${gpu_pids}" >&2
    exit 2
  fi
done

if [[ -e "${RECORD_DIR}" || -e "${RESULT_JSONL}" || -e "${SUMMARY_JSON}" ]]; then
  echo "Refusing to mix with existing or partial results under ${EVAL_ROOT}" >&2
  exit 2
fi

mkdir -p "${RECORD_DIR}" "${RUN_ROOT}/logs"
cd "${TCOD_ROOT}"
"${TCOD_ROOT}/scripts/run_alfworld_eval.sh" \
  "${CONFIG}" "${GPU_IDS}" "${RUN_TAG}" "${RAY_PORT}" 2>&1 | tee "${LOG_PATH}"

"${TCOD_ROOT}/.venv_tcod/bin/python" "${TCOD_ROOT}/scripts/collect_alfworld_eval_results.py" \
  --manifest "${TCOD_ROOT}/reproduction_data/alfworld/full_valid_seen.jsonl" \
  --manifest "${TCOD_ROOT}/reproduction_data/alfworld/full_valid_unseen.jsonl" \
  --record-dir "${RECORD_DIR}" \
  --output-jsonl "${RESULT_JSONL}" \
  --summary-json "${SUMMARY_JSON}" \
  --expected-count 274

echo "Completed and validated history-fixed GiGPO full-274 evaluation: ${SUMMARY_JSON}"
