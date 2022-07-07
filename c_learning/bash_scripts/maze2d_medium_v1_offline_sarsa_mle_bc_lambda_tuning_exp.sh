#!/bin/bash

EXP_LABEL=$1

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/..")

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export PYTHONPATH=$PROJECT_DIR
export HDF5_USE_FILE_LOCKING=FALSE

declare -a all_bc_lambdas=(0.05 0.1 0.2 0.3 0.5)
declare -a seeds=(0)

for bc_lambda in "${all_bc_lambdas[@]}"; do
  for seed in "${seeds[@]}"; do
    rm $CONDA_PREFIX/lib/python*/site-packages/mujoco_py/generated/mujocopy-buildlock
    mkdir -p ~/offline_c_learning/c_learning_offline_logs/"${EXP_LABEL}"/maze2d_medium_v1_bc_lambda="${bc_lambda}"/$seed
    nohup \
    python $PROJECT_DIR/train_eval_offline_d4rl.py \
      --gin_bindings="train_eval_offline.agent='c_learning_agent'" \
      --gin_bindings="train_eval_offline.env_name='maze2d-medium-v1'" \
      --gin_bindings="train_eval_offline.random_seed=${seed}" \
      --gin_bindings="train_eval_offline.num_iterations=1000000" \
      --gin_bindings="train_eval_offline.max_future_steps=200" \
      --gin_bindings="obs_to_goal.start_index=(0,)" \
      --gin_bindings="obs_to_goal.end_index=(2,)" \
      --gin_bindings="goal_fn.relabel_next_prob=0.5" \
      --gin_bindings="goal_fn.relabel_future_prob=0.0" \
      --gin_bindings="c_learning_agent.critic_loss.sarsa_q=True" \
      --gin_bindings="c_learning_agent.actor_loss.log_ratio_loss=True" \
      --gin_bindings="c_learning_agent.actor_loss.mle_bc_loss=True" \
      --gin_bindings="c_learning_agent.actor_loss.bc_lambda=${bc_lambda}" \
      --root_dir ~/offline_c_learning/c_learning_offline_logs/"${EXP_LABEL}"/maze2d_medium_v1_bc_lambda="${bc_lambda}"/$seed \
    > ~/offline_c_learning/c_learning_offline_logs/"${EXP_LABEL}"/maze2d_medium_v1_bc_lambda="${bc_lambc_lambdabda}"/$seed/stream.log 2>&1 & \
    sleep 5
  done
done
