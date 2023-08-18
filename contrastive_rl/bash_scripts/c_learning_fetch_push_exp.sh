#!/bin/bash

EXP_LABEL=$1

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/..")

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export PYTHONPATH=$PROJECT_DIR
export HDF5_USE_FILE_LOCKING=FALSE
export TF_FORCE_GPU_ALLOW_GROWTH=true
export D4RL_SUPPRESS_IMPORT_ERROR=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LOG_DIR="/projects/rsalakhugroup/chongyiz/contrastive_rl_logs"

declare -a seeds=(0)

for seed in "${seeds[@]}"; do
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export GPUS=3
  rm -rf $CONDA_PREFIX/lib/python*/site-packages/mujoco_py/generated/mujocopy-buildlock
  # shellcheck disable=SC2115
  rm -rf "$LOG_DIR"/"${EXP_LABEL}"/$seed
  mkdir -p "$LOG_DIR"/"${EXP_LABEL}"/$seed
  nohup \
  python $PROJECT_DIR/lp_contrastive.py \
    --env_name=fetch_push \
    --alg=c_learning \
    --max_number_of_steps=3_000_000 \
    --seed="$seed" \
    --lp_launch_type=local_mp \
    --root_dir="$LOG_DIR"/"${EXP_LABEL}"/$seed \
  > "$LOG_DIR"/"${EXP_LABEL}"/$seed/stream.log 2>&1 & \
  sleep 3
done
