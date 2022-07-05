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

"""Train and Eval C-learning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time
import imageio

from absl import app
from absl import flags
from absl import logging
import c_learning_agent
import c_learning_envs
import c_learning_utils
import gin
import numpy as np
from six.moves import range
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common

# limit gpu memory usage
import offline_c_learning_agent

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

flags.DEFINE_string('load_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Directory for restoring the learned agent')
flags.DEFINE_string('save_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Directory for saving videos')
flags.DEFINE_bool('run_eagerly', False, 'Enables / disables eager execution of tf.functions.')
flags.DEFINE_multi_string('gin_file', None, 'Path to the trainer config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding to pass through.')

FLAGS = flags.FLAGS


@gin.configurable
def vis_policy(
        load_dir,
        save_dir,
        agent='c_learning_agent',
        video_filename='video.mp4',
        env_name='sawyer_reach',
        actor_fc_layers=(256, 256),
        critic_obs_fc_layers=None,
        critic_action_fc_layers=None,
        critic_joint_fc_layers=(256, 256),
        num_episodes=5,
        random_seed=0,
        actor_std=None,
):
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    load_dir = os.path.expanduser(load_dir)
    load_dir = os.path.join(load_dir, 'train')
    save_dir = os.path.expanduser(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if env_name.startswith('metaworld'):
        env_name = env_name.replace('-', '.', 1)

    _, eval_tf_env, obs_dim = c_learning_envs.load(env_name, seed=random_seed)
    eval_py_env = eval_tf_env.pyenv.envs[0]

    time_step_spec = eval_tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = eval_tf_env.action_spec()

    if actor_std is None:
        proj_net = tanh_normal_projection_network.TanhNormalProjectionNetwork
    else:
        proj_net = functools.partial(
            tanh_normal_projection_network.TanhNormalProjectionNetwork,
            std_transform=lambda t: actor_std * tf.ones_like(t))

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=actor_fc_layers,
        continuous_projection_net=proj_net)

    critic_net = c_learning_utils.ClassifierCriticNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=critic_obs_fc_layers,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers)

    if agent == 'c_learning_agent':
        tf_agent = c_learning_agent.CLearningAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=None,
            critic_optimizer=None,
            obs_dim=obs_dim,
        )
    elif agent == 'offline_c_learning_agent':
        tf_agent = offline_c_learning_agent.OfflineCLearningAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            behavioral_cloning_network=None,
            actor_optimizer=None,
            critic_optimizer=None,
            behavioral_cloning_optimizer=None,
            obs_dim=obs_dim,
        )
    else:
        raise NotImplementedError

    # load learned agent
    train_checkpointer = common.Checkpointer(
        ckpt_dir=load_dir,
        agent=tf_agent)

    train_checkpointer.initialize_or_restore()
    eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
    # eval_policy = random_tf_policy.RandomTFPolicy(
    #     eval_tf_env.time_step_spec(), eval_tf_env.action_spec())

    video_path = os.path.join(save_dir, video_filename)
    with imageio.get_writer(video_path, fps=60) as video:
        for _ in range(num_episodes):
            time_step = eval_tf_env.reset()
            video.append_data(eval_py_env.render())
            while not time_step.is_last():
                action_step = eval_policy.action(time_step)
                time_step = eval_tf_env.step(action_step.action)
                video.append_data(eval_py_env.render())
    logging.info("Save video to: {}".format(os.path.abspath(video_path)))


def main(_):
    tf.compat.v1.enable_v2_behavior()
    if FLAGS.run_eagerly:
        tf.config.run_functions_eagerly(True)
    logging.set_verbosity(logging.INFO)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)
    load_dir = FLAGS.load_dir
    save_dir = FLAGS.save_dir
    vis_policy(load_dir, save_dir)


if __name__ == '__main__':
    flags.mark_flag_as_required('load_dir')
    flags.mark_flag_as_required('save_dir')
    app.run(main)
