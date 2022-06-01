#!/bin/bash

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/..")

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export PYTHONPATH=$PROJECT_DIR

declare -a seeds=(0 1 2)

for seed in "${seeds[@]}"; do
  export CUDA_VISIBLE_DEVICES=6
  nohup \
  python $PROJECT_DIR/train_eval.py \
    --gin_bindings="train_eval.env_name='sawyer_drawer'" \
    --gin_bindings="train_eval.random_seed=${seed}" \
    --gin_bindings="train_eval.num_iterations=3000000" \
    --gin_bindings="train_eval.log_subset=(3, None)" \
    --gin_bindings="goal_fn.relabel_next_prob=0.3" \
    --gin_bindings="goal_fn.relabel_future_prob=0.2" \
    --gin_bindings="SawyerDrawer.reset.arm_goal_type='goal'" \
    --root_dir ~/offline_c_learning/c_learning_logs/sawyer_drawer/$seed \
  > ~/offline_c_learning/terminal_logs/sawyer_drawer-"$seed".log 2>&1 &
done
