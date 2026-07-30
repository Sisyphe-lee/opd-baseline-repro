#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="${THUNLP_OPD_DIR:-$project_root/third_party/THUNLP_OPD}"
env_prefix="${ENV_PREFIX:-$project_root/.venv_opd_official}"
python_bin="${PYTHON_BIN:-$env_prefix/bin/python}"

run_stamp="$(date +%Y%m%d_%H%M%S)"
run_name="${RUN_NAME:-response_only_opd}"
artifact_root="${ARTIFACT_ROOT:-$project_root/artifacts}"
artifact_dir="${ARTIFACT_DIR:-$artifact_root/$run_name/$run_stamp}"
checkpoint_dir="$artifact_dir/checkpoint"
log_file="$artifact_dir/train.log"

student_model="${STUDENT_MODEL_PATH:-$project_root/models/Qwen3-1.7B-Base}"
teacher_model="${TEACHER_MODEL_PATH:-$project_root/models/Qwen3-4B-Base-GRPO}"
train_data="${TRAIN_DATA_PATH:-$repo_root/datasets/dapo-math-17k.parquet}"
val_data="${VAL_DATA_PATH:-$repo_root/datasets/test_data/AIME24/test.parquet}"
reward_fn="${REWARD_FUNCTION_PATH:-$repo_root/verl/verl/utils/reward_score/ttrl_math/__init__.py}"

train_batch_size="${TRAIN_BATCH_SIZE:-64}"
mini_batch_size="${MINI_BATCH_SIZE:-64}"
max_prompt_length="${MAX_PROMPT_LENGTH:-1024}"
max_response_length="${MAX_RESPONSE_LENGTH:-7168}"
max_model_len="${MAX_MODEL_LEN:-8192}"
max_tokens_per_gpu="${MAX_TOKENS_PER_GPU:-32768}"
num_rollouts="${NUM_ROLLOUTS:-4}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.8}"
cudagraph_capture_sizes="${CUDAGRAPH_CAPTURE_SIZES:-[1,2,4,8,16,32]}"
forward_prefetch="${FORWARD_PREFETCH:-True}"
reward_micro_batch_size="${REWARD_MICRO_BATCH_SIZE:-24}"
reward_forward_max_tokens_per_gpu="${REWARD_FORWARD_MAX_TOKENS_PER_GPU:-$max_tokens_per_gpu}"
num_gpus="${NUM_GPUS:-4}"
save_freq="${SAVE_FREQ:-20}"
total_training_steps="${TOTAL_TRAINING_STEPS:-null}"
rollout_data_dir="${ROLLOUT_DATA_DIR:-null}"

for required_path in "$python_bin" "$repo_root/.git" "$student_model" "$teacher_model" \
    "$train_data" "$val_data" "$reward_fn"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Missing required path: $required_path" >&2
        exit 1
    fi
done

expected_commit="4532fd35ccfdde82adc918b265e4c964534e83d1"
actual_commit="$(git -C "$repo_root" rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
    echo "THUNLP/OPD commit mismatch: expected $expected_commit, got $actual_commit" >&2
    exit 1
fi

mkdir -p "$artifact_dir"
exec > >(tee -a "$log_file") 2>&1

python_env_root="$(cd "$(dirname "$python_bin")/.." && pwd)"
nvidia_libs="$(find "$python_env_root/lib"/python*/site-packages/nvidia -type d -name lib 2>/dev/null | paste -sd: -)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [[ -n "$nvidia_libs" ]]; then
    export LD_LIBRARY_PATH="$nvidia_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH="$repo_root/verl${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=disabled
export SWANLAB_MODE=disabled
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"

echo "artifact_dir=$artifact_dir"
echo "python_bin=$python_bin"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "student_model=$student_model"
echo "teacher_model=$teacher_model"
echo "train_data=$train_data"
echo "run_name=$run_name"
echo "train_batch_size=$train_batch_size, num_rollouts=$num_rollouts"
echo "max_response_length=$max_response_length, max_tokens_per_gpu=$max_tokens_per_gpu"
echo "total_training_steps=$total_training_steps"

cd "$repo_root"

hydra_args=()
if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
    hydra_args+=(--cfg job)
fi

"$python_bin" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=token_reward_direct \
    algorithm.grpo_outcome_weight=1.0 \
    data.shuffle=False \
    data.train_files="$train_data" \
    data.val_files="$val_data" \
    data.train_batch_size="$train_batch_size" \
    data.max_prompt_length="$max_prompt_length" \
    data.max_response_length="$max_response_length" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$student_model" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$mini_batch_size" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$max_tokens_per_gpu" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch="$forward_prefetch" \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$max_tokens_per_gpu" \
    +actor_rollout_ref.rollout.log_prob_top_k=16 \
    +actor_rollout_ref.rollout.top_k_strategy=only_stu \
    +actor_rollout_ref.rollout.reward_weight_mode=student_p \
    +actor_rollout_ref.rollout.teacher_temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$gpu_memory_utilization" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$max_tokens_per_gpu" \
    actor_rollout_ref.rollout.max_model_len="$max_model_len" \
    actor_rollout_ref.rollout.cudagraph_capture_sizes="$cudagraph_capture_sizes" \
    actor_rollout_ref.rollout.n="$num_rollouts" \
    actor_rollout_ref.rollout.repetition_penalty=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$max_tokens_per_gpu" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.enable=True \
    +reward_model.reward_kwargs.enable_format_reward=False \
    reward_model.model.path="$teacher_model" \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=False \
    +reward_model.model.dtype=bfloat16 \
    reward_model.micro_batch_size_per_gpu="$reward_micro_batch_size" \
    reward_model.forward_max_token_len_per_gpu="$reward_forward_max_tokens_per_gpu" \
    custom_reward_function.path="$reward_fn" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=False \
    trainer.logger='[console]' \
    trainer.project_name=opd_prefill \
    trainer.experiment_name="$run_name" \
    trainer.n_gpus_per_node="$num_gpus" \
    trainer.nnodes=1 \
    trainer.save_freq="$save_freq" \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$total_training_steps" \
    trainer.resume_mode=disable \
    trainer.default_local_dir="$checkpoint_dir" \
    trainer.rollout_data_dir="$rollout_data_dir" \
    trainer.is_plot=False \
    "${hydra_args[@]}"

if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
    echo "Hydra configuration composed successfully."
else
    echo "Training completed successfully."
    echo "checkpoint_dir=$checkpoint_dir"
fi
