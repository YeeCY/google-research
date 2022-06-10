#!/bin/bash

EXP_LABEL=$1

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/..")

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export PYTHONPATH=$PROJECT_DIR
export HDF5_USE_FILE_LOCKING=FALSE

declare -a seeds=(0 1 2)

for seed in "${seeds[@]}"; do
  mkdir -p ~/offline_c_learning/c_learning_logs/"${EXP_LABEL}"_no_w_clipping/maze2d_umaze_v1/$seed
  nohup \
  python $PROJECT_DIR/train_eval.py \
    --gin_bindings="train_eval.env_name='maze2d-umaze-v1'" \
    --gin_bindings="train_eval.random_seed=${seed}" \
    --gin_bindings="train_eval.num_iterations=1000000" \
    --gin_bindings="obs_to_goal.start_index=0" \
    --gin_bindings="obs_to_goal.end_index=2" \
    --gin_bindings="goal_fn.relabel_next_prob=0.5" \
    --gin_bindings="goal_fn.relabel_future_prob=0.0" \
    --gin_bindings="critic_loss.w_clipping=1000000" \
    --root_dir ~/offline_c_learning/c_learning_logs/"${EXP_LABEL}"_no_w_clipping/maze2d_umaze_v1/$seed \
  > ~/offline_c_learning/c_learning_logs/"${EXP_LABEL}"_no_w_clipping/maze2d_umaze_v1/$seed/stream.log 2>&1 &
done
