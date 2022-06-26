# coding=utf-8
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

r"""Example running contrastive RL in JAX.

Run using multi-processing (required for image-based experiments):
  python lp_contrastive.py --lp_launch_type=local_mp

Run using multi-threading
  python lp_contrastive.py --lp_launch_type=local_mt


"""
import functools
from typing import Any, Dict

from absl import app
from absl import flags
import contrastive
from contrastive import utils as contrastive_utils
import launchpad as lp

FLAGS = flags.FLAGS
flags.DEFINE_bool('debug', True, 'Runs training for just a few steps.')


@functools.lru_cache()
def get_env(env_name, start_index, end_index):
  return contrastive_utils.make_environment(env_name, start_index, end_index,
                                            seed=0)


def get_program(params):
  """Constructs the program."""

  env_name = params['env_name']
  seed = params.pop('seed')

  if params.get('use_image_obs', False) and not params.get('local', False):
    print('WARNING: overwriting parameters for image-based tasks.')
    params['num_sgd_steps_per_step'] = 16
    params['prefetch_size'] = 16
    params['num_actors'] = 10

  config = contrastive.ContrastiveConfig(**params)

  env_factory = lambda seed: contrastive_utils.make_environment(  # pylint: disable=g-long-lambda
      env_name, config.start_index, config.end_index, seed)

  env_factory_no_extra = lambda seed: env_factory(seed)[0]  # Remove obs_dim.
  environment, obs_dim = get_env(env_name, config.start_index,
                                 config.end_index)
  assert (environment.action_spec().minimum == -1).all()
  assert (environment.action_spec().maximum == 1).all()
  config.obs_dim = obs_dim
  config.max_episode_steps = getattr(environment, '_step_limit') + 1
  network_factory = functools.partial(
      contrastive.make_networks, obs_dim=obs_dim, repr_dim=config.repr_dim,
      repr_norm=config.repr_norm, twin_q=config.twin_q,
      use_image_obs=config.use_image_obs,
      hidden_layer_sizes=config.hidden_layer_sizes)

  agent = contrastive.DistributedContrastive(
      seed=seed,
      environment_factory=env_factory_no_extra,
      network_factory=network_factory,
      config=config,
      num_actors=config.num_actors,
      log_to_bigtable=True,
      max_number_of_steps=config.max_number_of_steps)
  return agent.build()


def main(_):
  # Create experiment description.

  # 1. Select an environment.
  # Supported environments:
  #   Metaworld: sawyer_{push,drawer,bin,window}
  #   OpenAI Gym Fetch: fetch_{reach,push}
  #   D4RL AntMaze: ant_{umaze,,medium,large},
  #   2D nav: point_{Small,Cross,FourRooms,U,Spiral11x11,Maze11x11}
  # Image observation environments:
  #   Metaworld: sawyer_image_{push,drawer,bin,window}
  #   OpenAI Gym Fetch: fetch_{reach,push}_image
  #   2D nav: point_image_{Small,Cross,FourRooms,U,Spiral11x11,Maze11x11}
  # env_name = 'fetch_reach'
  env_name = 'ant_umaze'
  params = {
      'seed': 0,
      'use_random_actor': True,
      'entropy_coefficient': None if 'image' in env_name else 0.0,
      'env_name': env_name,
      'max_number_of_steps': 1_000_000,
      'use_image_obs': 'image' in env_name,
  }

  # 2. Select an algorithm. The currently-supported algorithms are:
  # contrastive_nce, contrastive_cpc, c_learning, nce+c_learning, gcbc.
  # Many other algorithms can be implemented by passing other parameters
  # or adding a few lines of code.
  # alg = 'contrastive_nce'
  alg = 'c_learning'
  if alg == 'contrastive_nce':
    pass  # Just use the default hyperparameters
  elif alg == 'contrastive_cpc':
    params['use_cpc'] = True
  elif alg == 'c_learning':
    params['use_td'] = True
    params['twin_q'] = True
  elif alg == 'nce+c_learning':
    params['use_td'] = True
    params['twin_q'] = True
    params['add_mc_to_td'] = True
  elif alg == 'gcbc':
    params['use_gcbc'] = True
  else:
    raise NotImplementedError('Unknown method: %s' % alg)

  # 3. Select compute parameters. The default parameters are already tuned, so
  # use this mainly for debugging.
  if FLAGS.debug:
    params.update({
        'min_replay_size': 2_000,
        'local': True,
        'num_sgd_steps_per_step': 1,
        'prefetch_size': 1,
        'num_actors': 1,
        'batch_size': 32,
        'max_number_of_steps': 10_000,
        'samples_per_insert_tolerance_rate': 1.0,
    })

  program = get_program(params)
  # Set terminal='tmux' if you want different components in different windows.
  lp.launch(program, terminal='current_terminal')

if __name__ == '__main__':
  app.run(main)
