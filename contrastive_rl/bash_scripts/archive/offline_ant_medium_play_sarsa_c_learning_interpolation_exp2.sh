#!/bin/bash

EXP_LABEL=$1

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/..")

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export PYTHONPATH=$PROJECT_DIR
export HDF5_USE_FILE_LOCKING=FALSE
export D4RL_SUPPRESS_IMPORT_ERROR=1
export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

declare -a c_learning_probs=(0.6 0.8 1.0)
declare -a seeds=(0)

for c_learning_prob_idx in "${!c_learning_probs[@]}"; do
  for seed in "${seeds[@]}"; do
#    if (($c_learning_prob_idx < 2)); then
#      export CUDA_VISIBLE_DEVICES="$c_learning_prob_idx"
#    else
#      export CUDA_VISIBLE_DEVICES="$(($c_learning_prob_idx + 1))"
#    fi
    export CUDA_VISIBLE_DEVICES=$c_learning_prob_idx
    rm $CONDA_PREFIX/lib/python*/site-packages/mujoco_py/generated/mujocopy-buildlock
    rm -r ~/offline_c_learning/contrastive_rl_logs/offline/"${EXP_LABEL}"/c_learning_prob_"${c_learning_probs[$c_learning_prob_idx]}"/offline_ant_medium_play/$seed
    mkdir -p ~/offline_c_learning/contrastive_rl_logs/offline/"${EXP_LABEL}"/c_learning_prob_"${c_learning_probs[$c_learning_prob_idx]}"/offline_ant_medium_play/$seed
    nohup \
    python $PROJECT_DIR/lp_contrastive.py \
      --env_name=offline_ant_medium_play \
      --alg=sarsa_c_learning \
      --seed="$seed" \
      --c_learning_prob="${c_learning_probs[$c_learning_prob_idx]}" \
      --lp_launch_type=local_mt \
      --root_dir=~/offline_c_learning/contrastive_rl_logs/offline/"${EXP_LABEL}"/c_learning_prob_"${c_learning_probs[$c_learning_prob_idx]}"/offline_ant_medium_play/$seed \
    > ~/offline_c_learning/contrastive_rl_logs/offline/"${EXP_LABEL}"/c_learning_prob_"${c_learning_probs[$c_learning_prob_idx]}"/offline_ant_medium_play/$seed/stream.log 2>&1 & \
    sleep 5
  done
done
