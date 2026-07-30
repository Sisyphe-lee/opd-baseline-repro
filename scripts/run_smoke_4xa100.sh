#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-response_only_smoke}"
export NUM_GPUS="${NUM_GPUS:-4}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
export SAVE_FREQ="${SAVE_FREQ:-3}"

exec "$script_dir/train_response_only_opd.sh"
