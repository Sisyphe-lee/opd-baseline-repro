#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_prefix="${ENV_PREFIX:-$project_root/.venv_opd_official}"
opd_dir="${THUNLP_OPD_DIR:-$project_root/third_party/THUNLP_OPD}"
opd_commit="4532fd35ccfdde82adc918b265e4c964534e83d1"
opd_url="https://github.com/thunlp/OPD.git"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but was not found in PATH." >&2
    exit 1
fi

mkdir -p "$(dirname "$opd_dir")"
if [[ ! -d "$opd_dir/.git" ]]; then
    git clone "$opd_url" "$opd_dir"
fi
git -C "$opd_dir" fetch origin "$opd_commit"
git -C "$opd_dir" checkout --detach "$opd_commit"
if [[ "$(git -C "$opd_dir" rev-parse HEAD)" != "$opd_commit" ]]; then
    echo "Failed to pin THUNLP/OPD to $opd_commit" >&2
    exit 1
fi

if [[ ! -x "$env_prefix/bin/python" ]]; then
    conda create -y -p "$env_prefix" python=3.12 pip
fi

python_bin="$env_prefix/bin/python"
pip_cmd=("$python_bin" -m pip)
"${pip_cmd[@]}" install --upgrade pip setuptools wheel
"${pip_cmd[@]}" install "vllm==0.11.0"
"${pip_cmd[@]}" install -r "$project_root/requirements-core.txt"
"${pip_cmd[@]}" install --no-build-isolation "flashinfer-python==0.3.1"
"${pip_cmd[@]}" install --no-deps \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1%2Bcu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

"${pip_cmd[@]}" check
"$python_bin" - <<'PY'
import cupy
import datasets
import flash_attn
import flashinfer
import numpy
import ray
import scipy
import tensordict
import torch
import transformers
import vllm

print(f"torch={torch.__version__}")
print(f"vllm={vllm.__version__}")
print(f"transformers={transformers.__version__}")
print(f"flash_attn={flash_attn.__version__}")
print(f"flashinfer={flashinfer.__version__}")
print(f"ray={ray.__version__}")
print(f"numpy={numpy.__version__}, scipy={scipy.__version__}")
print(f"datasets={datasets.__version__}, tensordict={tensordict.__version__}, cupy={cupy.__version__}")
PY

echo "Bootstrap completed."
echo "python=$python_bin"
echo "THUNLP_OPD_DIR=$opd_dir"
