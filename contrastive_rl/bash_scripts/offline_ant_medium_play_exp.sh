#!/bin/bash

EXP_LABEL=$1

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/..")

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/chongyiz/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export PYTHONPATH=$PROJECT_DIR
export HDF5_USE_FILE_LOCKING=FALSE
export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=.50

declare -a seeds=(0 1)

for seed in "${seeds[@]}"; do
  # shellcheck disable=SC2004
  export CUDA_VISIBLE_DEVICES="$(($seed * 2 + 1))"
  rm $CONDA_PREFIX/lib/python*/site-packages/mujoco_py/generated/mujocopy-buildlock
  rm -r ~/offline_c_learning/contrastive_rl_logs/offline/"${EXP_LABEL}"/offline_ant_medium_play/$seed
  mkdir -p ~/offline_c_learning/contrastive_rl_logs/offline/"${EXP_LABEL}"/offline_ant_medium_play/$seed
  nohup \
  python $PROJECT_DIR/lp_contrastive.py \
    --env_name=offline_ant_medium_play \
    --alg=c_learning \
    --seed="$seed" \
    --lp_launch_type=local_mt \
    --root_dir=~/offline_c_learning/contrastive_rl_logs/offline/"${EXP_LABEL}"/offline_ant_medium_play/$seed \
  > ~/offline_c_learning/contrastive_rl_logs/offline/"${EXP_LABEL}"/offline_ant_medium_play/$seed/stream.log 2>&1 & \
  sleep 5
done
