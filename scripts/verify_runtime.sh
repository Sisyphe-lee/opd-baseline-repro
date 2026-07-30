#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_prefix="${ENV_PREFIX:-$project_root/.venv_opd_official}"
python_bin="${PYTHON_BIN:-$env_prefix/bin/python}"
num_gpus="${NUM_GPUS:-4}"

if [[ ! -x "$python_bin" ]]; then
    echo "Missing Python environment: $python_bin" >&2
    exit 1
fi

"$python_bin" -m pip check
"$python_bin" - <<'PY'
import flash_attn
import ray
import torch
import transformers
import vllm
from flash_attn import flash_attn_func

assert torch.cuda.is_available(), "CUDA is unavailable"
q = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q, q, q)
torch.cuda.synchronize()
assert out.shape == q.shape
print(f"PASS: FlashAttention BF16 kernel on {torch.cuda.get_device_name(0)}")
print(
    f"torch={torch.__version__}, vllm={vllm.__version__}, "
    f"transformers={transformers.__version__}, ray={ray.__version__}, "
    f"flash_attn={flash_attn.__version__}"
)
PY

"$python_bin" -m torch.distributed.run \
    --standalone --nproc_per_node="$num_gpus" \
    "$project_root/scripts/check_nccl_cuda_graph.py"

echo "Runtime verification completed."
