#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "Zero diagnostic must run inside tmux." >&2
  exit 2
fi
if (( $# != 1 )); then
  echo "Usage: $0 GPU_IDS" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="$1"
[[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+){7}$ ]] || {
  echo "Exactly eight comma-separated GPU IDs are required." >&2
  exit 2
}

IFS=',' read -r -a GPU_ARRAY <<<"${GPU_IDS}"
for gpu_id in "${GPU_ARRAY[@]}"; do
  gpu_pids="$(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
  if [[ -n "${gpu_pids}" && "${gpu_id}" != "0" ]]; then
    echo "GPU ${gpu_id} is occupied by PID(s): ${gpu_pids//$'\n'/, }." >&2
    exit 3
  fi
  if [[ -n "${gpu_pids}" && "${gpu_id}" == "0" ]]; then
    echo "WARNING: proceeding with explicitly allowed light occupancy on GPU 0: ${gpu_pids//$'\n'/, }."
  fi
done

RUN_ROOT="${ROOT}/runs/experiments/two_stage_distillation"
EVAL_ROOT="${RUN_ROOT}/evaluation"
ANALYSIS_DIR="${ROOT}/analysis/two_stage_zero_diagnostic_offline30_online220_seed42"
DATA="${ROOT}/data/two_stage_distillation/generated/teacher_success_prefix_seqkd_seed42.jsonl"
MANIFEST="${ROOT}/data/two_stage_distillation/generated/teacher_success_prefix_seqkd_seed42.manifest.json"
TEACHER_RECORDS="${RUN_ROOT}/teacher_collection/seed42/task_records"
OFFLINE_CKPT="${RUN_ROOT}/checkpoints/ALFWORLD_TWO_STAGE_DISTILLATION/zero-diagnostic/teacher_success_offline30_seed42/global_step_30/actor/huggingface"
ONLINE_JOB="${RUN_ROOT}/checkpoints/ALFWORLD_TWO_STAGE_DISTILLATION/zero-diagnostic/teacher_success_offline30_online220_seed42"
ONLINE_CKPT="${ONLINE_JOB}/global_step_220/actor/huggingface"
DIAGNOSTICS="${RUN_ROOT}/diagnostics/offline30_online220_seed42/trajectory_metrics.jsonl"

verify_eval() {
  "${ROOT}/.venv_tcod/bin/python" - "$1" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
rows = [line for line in (root / "task_results.jsonl").read_text().splitlines() if line.strip()]
assert summary["task_count"] == 274 and len(rows) == 274
PY
}

run_eval() {
  local label="$1" config="$2" port="$3"
  local out="${EVAL_ROOT}/${label}"
  if [[ -f "${out}/summary.json" && -f "${out}/task_results.jsonl" ]]; then
    verify_eval "${out}"
    echo "Skipping verified evaluation: ${label}"
    return
  fi
  for target in "${out}/task_records" "${out}/task_results.jsonl" "${out}/summary.json"; do
    [[ ! -e "${target}" ]] || { echo "Refusing partial evaluation output: ${target}" >&2; exit 2; }
  done
  mkdir -p "${out}/task_records" "${out}/logs"
  bash "${ROOT}/scripts/run_alfworld_eval.sh" "${config}" "${GPU_IDS}" "${label}_full274" "${port}"
  "${ROOT}/.venv_tcod/bin/python" "${ROOT}/scripts/collect_alfworld_eval_results.py" \
    --manifest "${ROOT}/data/eval_manifests/full_valid_seen.jsonl" \
    --manifest "${ROOT}/data/eval_manifests/full_valid_unseen.jsonl" \
    --record-dir "${out}/task_records" \
    --output-jsonl "${out}/task_results.jsonl" \
    --summary-json "${out}/summary.json" --expected-count 274
  verify_eval "${out}"
}

plot_comparison() {
  "${ROOT}/.venv_tcod/bin/python" "${ROOT}/scripts/analyze_two_stage_zero_diagnostic.py" \
    --evaluation-root "${EVAL_ROOT}" --output-dir "${ANALYSIS_DIR}"
}

echo "[1/8] Frozen student-init full274 (required to define warm)."
run_eval student_init_seed42 "${ROOT}/configs/experiments/two_stage_init_full274_seed42.yaml" 6420

echo "[2/8] Teacher trajectory collection and offline dataset construction."
if [[ ! -f "${DATA}" || ! -f "${MANIFEST}" ]]; then
  [[ ! -e "${TEACHER_RECORDS}" ]] || { echo "Refusing partial teacher records: ${TEACHER_RECORDS}" >&2; exit 2; }
  mkdir -p "${TEACHER_RECORDS}" "${RUN_ROOT}/teacher_collection/seed42/logs"
  bash "${ROOT}/scripts/run_alfworld_eval.sh" \
    "${ROOT}/configs/experiments/two_stage_teacher_collection_seed42.yaml" \
    "${GPU_IDS}" two_stage_teacher_collection_seed42 6421
  "${ROOT}/.venv_tcod/bin/python" "${ROOT}/scripts/build_teacher_success_sft.py" \
    --record-dir "${TEACHER_RECORDS}" --output-jsonl "${DATA}" --manifest-json "${MANIFEST}" \
    --tokenizer "${ROOT}/models/Qwen2.5-3B-Instruct" \
    --expected-records 3553 --min-samples 1920
fi
"${ROOT}/.venv_tcod/bin/python" - "${DATA}" "${MANIFEST}" <<'PY'
import hashlib, json, sys
data, manifest_path = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
actual = hashlib.sha256(open(data, "rb").read()).hexdigest()
assert actual == manifest["output_sha256"]
assert manifest["output_samples"] >= 1920
PY

echo "[3/8] Offline distillation: 30 updates on eight trainer GPUs."
if [[ ! -d "${OFFLINE_CKPT}" ]]; then
  bash "${ROOT}/scripts/run_two_stage_train_config.sh" \
    "${ROOT}/configs/experiments/two_stage_offline30_seed42.yaml" \
    "${GPU_IDS}" two_stage_offline30_seed42 6422
fi
[[ -d "${OFFLINE_CKPT}" ]] || { echo "Offline HF checkpoint missing: ${OFFLINE_CKPT}" >&2; exit 2; }

echo "[4/8] Offline-30 frozen full274."
run_eval offline30_seed42 "${ROOT}/configs/experiments/two_stage_offline30_full274_seed42.yaml" 6423

echo "[5/8] Warm-start comparison figures."
plot_comparison

echo "[6/8] Vanilla online distillation: 220 updates, save every 20."
mkdir -p "${RUN_ROOT}/buffers"
if [[ ! -d "${ONLINE_CKPT}" ]]; then
  bash "${ROOT}/scripts/run_two_stage_train_config.sh" \
    "${ROOT}/configs/experiments/two_stage_online220_from_offline30_seed42.yaml" \
    "${GPU_IDS}" two_stage_offline30_online220_seed42 6424
fi
for step in $(seq 20 20 220); do
  [[ -d "${ONLINE_JOB}/global_step_${step}/actor/huggingface" ]] || {
    echo "Missing required online HF checkpoint at step ${step}." >&2
    exit 2
  }
done

echo "[7/8] Final frozen full274."
run_eval offline30_online220_seed42 \
  "${ROOT}/configs/experiments/two_stage_offline30_online220_full274_seed42.yaml" 6425

echo "[8/8] Final comparison and online entropy/KL plots."
plot_comparison
"${ROOT}/.venv_tcod/bin/python" \
  "${ROOT}/research_tools/legacy_entropy_diagnostics/scripts/plot_vanilla_opd_diagnostics.py" \
  --diagnostics "${DIAGNOSTICS}" \
  --output-dir "${ANALYSIS_DIR}/online_diagnostics" \
  --checkpoint-job-dir "${ONLINE_JOB}" --final-trainer-step 220 \
  --expected-trajectories 16 --tokenizer-path "${ROOT}/models/Qwen2.5-3B-Instruct" \
  --status complete
echo "ZERO DIAGNOSTIC COMPLETE: ${ANALYSIS_DIR}"
