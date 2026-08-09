#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "Full evaluation must run inside tmux. Use scripts/launch_eval_tmux.sh." >&2
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
    CONFIG="${ROOT}/configs/eval/tcod_f2b_step250_full274.yaml"
    RUN_ROOT="${ROOT}/runs/evaluation/tcod_f2b_step250_full274"
    RAY_PORT=6696
    ;;
  vanilla)
    CONFIG="${ROOT}/configs/eval/vanilla_opd_step250_full274.yaml"
    RUN_ROOT="${ROOT}/runs/evaluation/vanilla_opd_step250_full274"
    RAY_PORT=6706
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
  [[ -z "${gpu_pids}" ]] || { echo "GPU ${gpu_id} is occupied: ${gpu_pids}" >&2; exit 3; }
done

RECORD_DIR="${RUN_ROOT}/task_records"
RESULT_JSONL="${RUN_ROOT}/task_results.jsonl"
SUMMARY_JSON="${RUN_ROOT}/summary.json"
if [[ -e "${RECORD_DIR}" || -e "${RESULT_JSONL}" || -e "${SUMMARY_JSON}" ]]; then
  echo "Refusing to mix with existing output under ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RECORD_DIR}" "${RUN_ROOT}/logs"
cd "${ROOT}"
bash scripts/run_alfworld_eval.sh "${CONFIG}" "${GPU_IDS}" "baseline_${MODE}_full274" "${RAY_PORT}"
"${ROOT}/.venv_tcod/bin/python" scripts/collect_alfworld_eval_results.py \
  --manifest data/eval_manifests/full_valid_seen.jsonl \
  --manifest data/eval_manifests/full_valid_unseen.jsonl \
  --record-dir "${RECORD_DIR}" --output-jsonl "${RESULT_JSONL}" \
  --summary-json "${SUMMARY_JSON}" --expected-count 274
echo "Completed frozen ${MODE} evaluation: ${SUMMARY_JSON}"

