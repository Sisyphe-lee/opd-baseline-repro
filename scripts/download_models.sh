#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_prefix="${ENV_PREFIX:-$project_root/.venv_opd_official}"
python_bin="${PYTHON_BIN:-$env_prefix/bin/python}"
model_root="${MODEL_ROOT:-$project_root/models}"
student_repo="${STUDENT_MODEL_REPO:-Qwen/Qwen3-1.7B-Base}"
teacher_repo="${TEACHER_MODEL_REPO:-lllyx/Qwen3-4B-Base-GRPO}"
student_dir="${STUDENT_MODEL_PATH:-$model_root/Qwen3-1.7B-Base}"
teacher_dir="${TEACHER_MODEL_PATH:-$model_root/Qwen3-4B-Base-GRPO}"
max_workers="${HF_MAX_WORKERS:-8}"

if [[ ! -x "$python_bin" ]]; then
    echo "Missing Python environment: $python_bin. Run scripts/bootstrap_a100.sh first." >&2
    exit 1
fi

mkdir -p "$student_dir" "$teacher_dir"
"$python_bin" -m huggingface_hub.commands.huggingface_cli download \
    "$student_repo" --local-dir "$student_dir" --max-workers "$max_workers"
"$python_bin" -m huggingface_hub.commands.huggingface_cli download \
    "$teacher_repo" --local-dir "$teacher_dir" --max-workers "$max_workers"

for required_file in "$student_dir/config.json" "$teacher_dir/config.json"; do
    if [[ ! -s "$required_file" ]]; then
        echo "Model download is incomplete: $required_file" >&2
        exit 1
    fi
done

echo "Model downloads completed."
echo "student_model=$student_dir"
echo "teacher_model=$teacher_dir"
