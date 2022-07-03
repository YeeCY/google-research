#!/bin/bash

EXP_LABEL=$1

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/..")

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/data/chongyiz/anaconda3/envs/contrastive_rl/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/chongyiz/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export CUDA_VISBLE_DEVICES=7
export PYTHONPATH=$PROJECT_DIR
export HDF5_USE_FILE_LOCKING=FALSE

declare -a seeds=(0 1 2)

for seed in "${seeds[@]}"; do
  rm $CONDA_PREFIX/lib/python*/site-packages/mujoco_py/generated/mujocopy-buildlock
  mkdir -p ~/offline_c_learning/contrastive_rl_logs/"${EXP_LABEL}"/offline_ant_umaze/$seed
  nohup \
  python $PROJECT_DIR/lp_contrastive.py \
    --env_name=offline_ant_umaze \
    --alg=c_learning \
    --seed="$seed" \
    --lp_launch_type=local_mt \
    --log_dir=~/offline_c_learning/contrastive_rl_logs/"${EXP_LABEL}"/offline_ant_umaze/$seed \
  > ~/offline_c_learning/contrastive_rl_logs/"${EXP_LABEL}"/offline_ant_umaze/$seed/stream.log 2>&1 &
done
