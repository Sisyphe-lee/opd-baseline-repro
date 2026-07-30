#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-response_only_full}"
export NUM_GPUS="${NUM_GPUS:-4}"
# Hydra null means that upstream derives the step count from one full data epoch.
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
export SAVE_FREQ="${SAVE_FREQ:-20}"

exec "$script_dir/train_response_only_opd.sh"
