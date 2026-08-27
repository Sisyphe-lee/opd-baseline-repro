#!/usr/bin/env bash
set -euo pipefail

TCOD_ROOT=/vepfs-mlp2/mlp-public/252302025/sjx/opd-workspaces/tcod-f2b-repro
RUN_ROOT=/vepfs-mlp2/mlp-public/252302025/sjx/opd-workspaces/opd-alfworld-sync-repro/runs/2026-08-07_qwen2p5-3b-instruct-init-full274-h50-historyfix
CONFIG=${RUN_ROOT}/configs/eval_full274_h50_historyfix_7gpu.yaml
EVAL_ROOT=${RUN_ROOT}/evaluation/full274_h50
GPU_IDS=${1:-0,2,3,4,5,6,7}
RUN_TAG=qwen2p5_3b_instruct_init_full274_h50_historyfix_7gpu_20260807

if [[ "$(sha256sum "${TCOD_ROOT}/reproduction_data/alfworld/full_valid_seen.jsonl" | awk '{print $1}')" != "3f93167b4da2d68e789409785c9a328e0cf55a3b40bd60165cbd537011e460d3" ]]; then
  echo "Full-274 seen manifest checksum mismatch" >&2
  exit 2
fi
if [[ "$(sha256sum "${TCOD_ROOT}/reproduction_data/alfworld/full_valid_unseen.jsonl" | awk '{print $1}')" != "60c3edd7e9fba923d5befe5e88af424967b79c6d325eead4634a88a396e37662" ]]; then
  echo "Full-274 unseen manifest checksum mismatch" >&2
  exit 2
fi

IFS=',' read -r -a gpu_array <<<"${GPU_IDS}"
if [[ "${#gpu_array[@]}" -ne 7 ]]; then
  echo "Expected exactly seven GPU IDs, got ${GPU_IDS}" >&2
  exit 2
fi
for gpu_id in "${gpu_array[@]}"; do
  gpu_pids="$(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
  if [[ -n "${gpu_pids}" ]]; then
    echo "GPU ${gpu_id} already has compute process(es): ${gpu_pids}" >&2
    exit 2
  fi
done

if [[ -e "${EVAL_ROOT}/task_records" || -e "${EVAL_ROOT}/task_results.jsonl" || -e "${EVAL_ROOT}/summary.json" ]]; then
  echo "Refusing to mix with existing or partial results under ${EVAL_ROOT}" >&2
  exit 2
fi
mkdir -p "${EVAL_ROOT}/task_records" "${RUN_ROOT}/logs"

cd "${TCOD_ROOT}"
"${TCOD_ROOT}/scripts/run_alfworld_eval.sh" \
  "${CONFIG}" "${GPU_IDS}" "${RUN_TAG}" 6770 2>&1 | tee "${RUN_ROOT}/logs/eval_full274_h50_historyfix_7gpu.log"

"${TCOD_ROOT}/.venv_tcod/bin/python" "${TCOD_ROOT}/scripts/collect_alfworld_eval_results.py" \
  --manifest "${TCOD_ROOT}/reproduction_data/alfworld/full_valid_seen.jsonl" \
  --manifest "${TCOD_ROOT}/reproduction_data/alfworld/full_valid_unseen.jsonl" \
  --record-dir "${EVAL_ROOT}/task_records" \
  --output-jsonl "${EVAL_ROOT}/task_results.jsonl" \
  --summary-json "${EVAL_ROOT}/summary.json" \
  --expected-count 274

echo "Completed and validated Qwen2.5-3B-Instruct initialization full274/h50 evaluation."
