# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1

python lp_contrastive.py \
  --debug=True \
  --lp_launch_type=local_mt \
  --root_dir=~/contrastive_rl_logs_debug/offline/ant_umaze_c_learning_debug/0 \
  --env_name=offline_ant_umaze \
  --alg=c_learning \
  --seed=0
