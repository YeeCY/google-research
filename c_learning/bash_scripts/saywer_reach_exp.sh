#!/bin/bash

DATE=$1

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/..")

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export PYTHONPATH=$PROJECT_DIR

declare -a seeds=(1)

for seed in "${seeds[@]}"; do
  mkdir -p ~/offline_c_learning/c_learning_logs/"${DATE}"/sawyer_reach_600k/$seed
  nohup \
  python $PROJECT_DIR/train_eval.py \
    --gin_bindings="train_eval.env_name='sawyer_reach'" \
    --gin_bindings="train_eval.random_seed=${seed}" \
    --gin_bindings="train_eval.num_iterations=600000" \
    --gin_bindings="obs_to_goal.start_index=0" \
    --gin_bindings="obs_to_goal.end_index=3" \
    --gin_bindings="goal_fn.relabel_next_prob=0.5" \
    --gin_bindings="goal_fn.relabel_future_prob=0.0" \
    --root_dir ~/offline_c_learning/c_learning_logs/"${DATE}"/sawyer_reach_600k/$seed \
  > ~/offline_c_learning/c_learning_logs/"${DATE}"/sawyer_reach_600k/$seed/stream.log 2>&1 &
done
