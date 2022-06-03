#!/bin/bash

GPUS=$1

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/..")

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export PYTHONPATH=$PROJECT_DIR

declare -a seeds=(0 1 2)

for seed in "${seeds[@]}"; do
  export CUDA_VISIBLE_DEVICES=$GPUS
  mkdir -p ~/offline_c_learning/c_learning_logs/sawyer_window/$seed
  nohup \
  python $PROJECT_DIR/train_eval.py \
    --gin_bindings="train_eval.env_name='sawyer_window'" \
    --gin_bindings="train_eval.random_seed=${seed}" \
    --gin_bindings="train_eval.num_iterations=3000000" \
    --gin_bindings="train_eval.log_subset=(3, None)" \
    --gin_bindings="SawyerWindow.reset.arm_goal_type='goal'" \
    --gin_bindings="goal_fn.relabel_next_prob=0.5" \
    --gin_bindings="goal_fn.relabel_future_prob=0.0" \
    --root_dir ~/offline_c_learning/c_learning_logs/sawyer_window/$seed \
  > ~/offline_c_learning/c_learning_logs/sawyer_window/$seed/stream.log 2>&1 &
done
