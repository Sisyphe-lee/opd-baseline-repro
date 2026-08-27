#!/usr/bin/env bash
set -euo pipefail

[[ -n "${TMUX:-}" ]] || { echo "Live validation must run in tmux." >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-0}"
cd "${ROOT}"

gpu_pids="$(nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
[[ -z "${gpu_pids}" ]] || { echo "GPU ${GPU_ID} is occupied: ${gpu_pids}" >&2; exit 3; }

for mode in tcod vanilla; do
  record_dir="${ROOT}/validation/live/${mode}/task_records"
  result_jsonl="${ROOT}/validation/live/${mode}/task_results.jsonl"
  summary_json="${ROOT}/validation/live/${mode}/summary.json"
  [[ -d "${record_dir}" && -z "$(find "${record_dir}" -mindepth 1 -print -quit)" ]] || { echo "Record directory is not empty: ${record_dir}" >&2; exit 2; }
  [[ ! -e "${result_jsonl}" && ! -e "${summary_json}" ]] || { echo "Smoke result already exists for ${mode}" >&2; exit 2; }
  if [[ "${mode}" == tcod ]]; then
    config=configs/validation/tcod_step250_one_task.yaml
    port=6720
  else
    config=configs/validation/vanilla_step250_one_task.yaml
    port=6730
  fi
  bash scripts/run_alfworld_eval.sh "${config}" "${GPU_ID}" "baseline_live_smoke_${mode}" "${port}"
  .venv_tcod/bin/python scripts/collect_alfworld_eval_results.py \
    --manifest data/eval_manifests/live_smoke_seen_unseen.jsonl \
    --record-dir "${record_dir}" --output-jsonl "${result_jsonl}" \
    --summary-json "${summary_json}" --expected-count 2
done
date -u '+%Y-%m-%dT%H:%M:%SZ' > validation/live/LIVE_SMOKE_COMPLETED
echo "TCOD and Vanilla live smoke validation completed."
