#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$script_dir/bootstrap_a100.sh"
"$script_dir/download_models.sh"
"$script_dir/verify_runtime.sh"

echo "A100 machine preparation completed successfully."
