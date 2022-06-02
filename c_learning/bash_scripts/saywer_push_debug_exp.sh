#!/bin/bash

GPUS=$1

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/..")

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export PYTHONPATH=$PROJECT_DIR

declare -a seeds=(0)

for seed in "${seeds[@]}"; do
  export CUDA_VISIBLE_DEVICES=$GPUS
  nohup \
  python $PROJECT_DIR/train_eval.py \
    --gin_bindings="train_eval.env_name='sawyer_push'" \
    --gin_bindings="train_eval.random_seed=${seed}" \
    --gin_bindings="train_eval.num_iterations=3000000" \
    --gin_bindings="train_eval.log_subset=(3, 6)" \
    --gin_bindings="goal_fn.relabel_next_prob=0.3" \
    --gin_bindings="goal_fn.relabel_future_prob=0.2" \
    --gin_bindings="SawyerPush.reset.arm_goal_type='goal'" \
    --gin_bindings="SawyerPush.reset.fix_z=True" \
    --gin_bindings="load_sawyer_push.random_init=True" \
    --gin_bindings="load_sawyer_push.wide_goals=True" \
    --root_dir ~/offline_c_learning/c_learning_logs_debug/sawyer_push/$seed \
  > ~/offline_c_learning/c_learning_logs_debug/sawyer_push/$seed/stream.log 2>&1 &
done
