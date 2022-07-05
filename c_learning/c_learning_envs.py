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

"""Load and wrap the d4rl environments.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import logging
import gin
import numpy as np
import h5py
from tqdm import tqdm

import d4rl
from d4rl.offline_env import download_dataset_from_url, get_keys
from d4rl import pointmaze
from d4rl import locomotion

import gym
import mujoco_py
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
from metaworld.envs.mujoco import sawyer_xyz

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers

import antmaze_env
import c_learning_utils


os.environ['SDL_VIDEODRIVER'] = 'dummy'


# When collecting trajectory snippets for training, we use discount = 0 to
# decide when to break a trajectory; we don't use the step_type. For data
# collection, we therefore should set done=True only when the environment truly
# terminates, not when we've reached the goal.
# Eventually, we want to create the train_env by taking any gym_env or py_env,
# putting a learned goal-sampling wrapper around it, and then using that.


def load_sawyer_reach():
    gym_env = SawyerReach()
    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=51,
    )
    return tf_py_environment.TFPyEnvironment(env)


@gin.configurable
def load_sawyer_push(random_init=False, wide_goals=False,
                     include_gripper=False):
    """Load the sawyer pushing (and picking) environment.

    Args:
      random_init: (bool) Whether to randomize the initial arm position.
      wide_goals: (bool) Whether to use a wider range of Y positions for goals.
        The Y axis parallels the ground, pointing from the robot to the table.
      include_gripper: (bool) Whether to include the gripper open/close state in
        the observation.
    Returns:
      tf_env: An environment.
    """
    if wide_goals:
        goal_low = (-0.1, 0.5, 0.05)
    else:
        goal_low = (-0.1, 0.8, 0.05)
    if include_gripper:
        gym_env = SawyerPushGripper(random_init=random_init, goal_low=goal_low)
    else:
        gym_env = SawyerPush(random_init=random_init, goal_low=goal_low)
    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=151,
    )
    return tf_py_environment.TFPyEnvironment(env)


@gin.configurable
def load_sawyer_drawer(random_init=False):
    gym_env = SawyerDrawer(random_init=random_init)
    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=151,
    )
    return tf_py_environment.TFPyEnvironment(env)


def load_sawyer_drawer_v2():
    gym_env = SawyerDrawerV2()
    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=501,
    )
    return tf_py_environment.TFPyEnvironment(env)


@gin.configurable
def load_sawyer_window(rotMode='fixed'):  # pylint: disable=invalid-name
    gym_env = SawyerWindow(rotMode=rotMode)
    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=151,
    )
    return tf_py_environment.TFPyEnvironment(env)


def load_sawyer_faucet():
    gym_env = SawyerFaucet()
    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=151,
    )
    return tf_py_environment.TFPyEnvironment(env)


def load_maze2d_open_v0():
    gym_env = Maze2DOpenV0()
    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=151,
    )

    return tf_py_environment.TFPyEnvironment(env)


def load_maze2d_umaze_v1():
    gym_env = Maze2DUMazeV1()
    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=301,
    )

    return tf_py_environment.TFPyEnvironment(env)


def load_maze2d_medium_v1():
    gym_env = Maze2DMediumV1()
    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=601,
    )

    return tf_py_environment.TFPyEnvironment(env)


def load_maze2d_large_v1():
    gym_env = Maze2DLargeV1()
    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=801,
    )

    return tf_py_environment.TFPyEnvironment(env)


def load_halfcheetah_medium_v2():
    gym_env = gym.make("halfcheetah-medium-v2")

    # obs = gym_env.reset()
    # obs, reward, done, info = gym_env.step(gym_env.action_space.sample())

    env = suite_gym.wrap_env(
        gym_env.env,
        max_episode_steps=1001,
    )

    return tf_py_environment.TFPyEnvironment(env)

    # register(
    #     id=env_name,
    #     entry_point='d4rl.gym_mujoco.gym_envs:get_%s_env' % agent.replace('halfcheetah', 'cheetah').replace('walker2d',
    #                                                                                                         'walker'),
    #     max_episode_steps=1000,
    #     kwargs={
    #         'deprecated': version != 'v2',
    #         'ref_min_score': infos.REF_MIN_SCORE[env_name],
    #         'ref_max_score': infos.REF_MAX_SCORE[env_name],
    #         'dataset_url': infos.DATASET_URLS[env_name]
    #     }
    # )


def load_pen_human_v1():
    gym_env = gym.make("pen-human-v1")
    gym_env = c_learning_utils.OfflineAdroitWrapper(gym_env.env)

    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=200,
    )

    return tf_py_environment.TFPyEnvironment(env)


def load_hammer_human_v1():
    gym_env = gym.make("hammer-human-v1")
    gym_env = c_learning_utils.OfflineAdroitWrapper(gym_env.env)

    gym_env.reset()
    gym_env.step(gym_env.action_space.sample())

    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=200,
    )

    return tf_py_environment.TFPyEnvironment(env)


def load_door_human_v1():
    gym_env = gym.make("door-human-v1")
    gym_env = c_learning_utils.OfflineAdroitWrapper(gym_env.env)

    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=200,
    )

    return tf_py_environment.TFPyEnvironment(env)


def load_relocate_human_v1():
    gym_env = gym.make("relocate-human-v1")
    gym_env = c_learning_utils.OfflineAdroitWrapper(gym_env.env)

    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=200,
    )

    return tf_py_environment.TFPyEnvironment(env)


def load_antmaze_umaze_v0():
    gym_env = gym.make("antmaze-umaze-v0")
    gym_env = c_learning_utils.OfflineAntMazeWrapper(gym_env.env)

    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=701,
    )

    return tf_py_environment.TFPyEnvironment(env)


def load_antmaze_umaze_v2():
    # gym_env = AntMazeUmazeV2()
    # gym_env = antmaze_env.AntMaze(
    #     'umaze',
    #     non_zero_reset=True,
    #     dataset_url='http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5')
    gym_env = gym.make("antmaze-umaze-v2")
    # gym_env = c_learning_utils.AntMazeWrapper(gym_env.env)
    gym_env = c_learning_utils.OfflineAntMazeWrapper(gym_env.env)

    # env = suite_gym.wrap_env(
    #     gym_env,
    #     max_episode_steps=701,
    # )
    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=701,
    )

    return tf_py_environment.TFPyEnvironment(env)


def load_antmaze_umaze_diverse_v2():
    # gym_env = AntMazeUmazeDiverseV2()
    # gym_env = antmaze_env.AntMaze(
    #     'umaze',
    #     non_zero_reset=True,
    #     dataset_url='http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_u-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5')
    gym_env = gym.make("antmaze-umaze-diverse-v2")
    gym_env = c_learning_utils.OfflineAntMazeWrapper(gym_env.env)
    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=1001,
    )

    return tf_py_environment.TFPyEnvironment(env)


def load_antmaze_medium_play_v2():
    gym_env = gym.make("antmaze-medium-play-v2")
    gym_env = c_learning_utils.OfflineAntMazeWrapper(gym_env.env)
    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=1001,
    )

    return tf_py_environment.TFPyEnvironment(env)


def load_antmaze_medium_diverse_v2():
    # gym_env = AntMazeMediumDiverseV2()
    # gym_env = antmaze_env.AntMaze('medium', non_zero_reset=True)
    gym_env = gym.make("antmaze-medium-diverse-v2")
    gym_env = c_learning_utils.OfflineAntMazeWrapper(gym_env.env)
    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=1001,
    )

    return tf_py_environment.TFPyEnvironment(env)


def load_antmaze_large_play_v2():
    gym_env = gym.make("antmaze-large-play-v2")
    gym_env = c_learning_utils.OfflineAntMazeWrapper(gym_env.env)
    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=1001,
    )

    return tf_py_environment.TFPyEnvironment(env)


def load_antmaze_large_diverse_v2():
    # gym_env = AntMazeLargeDiverseV2()
    # gym_env = antmaze_env.AntMaze('large', non_zero_reset=True)
    gym_env = gym.make("antmaze-large-diverse-v2")
    gym_env = c_learning_utils.OfflineAntMazeWrapper(gym_env.env)
    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=1001,
    )

    return tf_py_environment.TFPyEnvironment(env)


def load_metaworld(env_name, seed=None):
    goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + "-goal-observable"]
    gym_env = goal_observable_cls(seed)
    gym_env = c_learning_utils.MetaWorldWrapper(gym_env)

    env = suite_gym.wrap_env(
        gym_env,
        max_episode_steps=gym_env.max_path_length + 1,
    )

    return tf_py_environment.TFPyEnvironment(env)


def load(env_name, seed=None):
    """Creates the training and evaluation environment.

    This method automatically detects whether we are using a subset of the
    observation for the goal and modifies the observation space to include the
    full state + partial goal.

    Args:
      env_name: (str) Name of the environment.
    Returns:
      tf_env, eval_tf_env, obs_dim: The training and evaluation environments.
    """
    if env_name == 'sawyer_reach':
        tf_env = load_sawyer_reach()
        eval_tf_env = load_sawyer_reach()
    elif env_name == 'sawyer_push':
        tf_env = load_sawyer_push()
        eval_tf_env = load_sawyer_push()
        eval_tf_env.envs[0]._env.gym.MODE = 'eval'  # pylint: disable=protected-access
    elif env_name == 'sawyer_drawer':
        tf_env = load_sawyer_drawer()
        eval_tf_env = load_sawyer_drawer()
    elif env_name == 'sawyer_drawer_v2':
        tf_env = load_sawyer_drawer_v2()
        eval_tf_env = load_sawyer_drawer_v2()
    elif env_name == 'sawyer_window':
        tf_env = load_sawyer_window()
        eval_tf_env = load_sawyer_window()
    elif env_name == 'sawyer_faucet':
        tf_env = load_sawyer_faucet()
        eval_tf_env = load_sawyer_faucet()
    # elif env_name == 'maze2d-open-v0':
    #     tf_env = load_maze2d_open_v0()
    #     eval_tf_env = load_maze2d_open_v0()
    elif env_name == 'maze2d-umaze-v1':
        tf_env = load_maze2d_umaze_v1()
        eval_tf_env = load_maze2d_umaze_v1()
    elif env_name == 'maze2d-medium-v1':
        tf_env = load_maze2d_medium_v1()
        eval_tf_env = load_maze2d_medium_v1()
    elif env_name == 'maze2d-large-v1':
        tf_env = load_maze2d_large_v1()
        eval_tf_env = load_maze2d_large_v1()
    elif env_name == 'halfcheetah-medium-v2':
        tf_env = load_halfcheetah_medium_v2()
        eval_tf_env = load_halfcheetah_medium_v2()
    elif env_name == 'pen-human-v1':
        tf_env = load_pen_human_v1()
        eval_tf_env = load_pen_human_v1()
    elif env_name == 'hammer-human-v1':
        tf_env = load_hammer_human_v1()
        eval_tf_env = load_hammer_human_v1()
    elif env_name == 'door-human-v1':
        tf_env = load_door_human_v1()
        eval_tf_env = load_door_human_v1()
    elif env_name == 'relocate-human-v1':
        tf_env = load_relocate_human_v1()
        eval_tf_env = load_relocate_human_v1()
    elif env_name == 'antmaze-umaze-v0':
        tf_env = load_antmaze_umaze_v0()
        eval_tf_env = load_antmaze_umaze_v0()
    elif env_name == 'antmaze-umaze-v2':
        tf_env = load_antmaze_umaze_v2()
        eval_tf_env = load_antmaze_umaze_v2()
    elif env_name == 'antmaze-umaze-diverse-v2':
        tf_env = load_antmaze_umaze_diverse_v2()
        eval_tf_env = load_antmaze_umaze_diverse_v2()
    elif env_name == 'antmaze-medium-play-v2':
        tf_env = load_antmaze_medium_play_v2()
        eval_tf_env = load_antmaze_medium_play_v2()
    elif env_name == 'antmaze-medium-diverse-v2':
        tf_env = load_antmaze_medium_diverse_v2()
        eval_tf_env = load_antmaze_medium_diverse_v2()
    elif env_name == 'antmaze-large-play-v2':
        tf_env = load_antmaze_large_play_v2()
        eval_tf_env = load_antmaze_large_play_v2()
    elif env_name == 'antmaze-large-diverse-v2':
        tf_env = load_antmaze_large_diverse_v2()
        eval_tf_env = load_antmaze_large_diverse_v2()
    elif env_name.startswith('metaworld'):
        env_name_ = env_name.split('.')[-1]
        tf_env = load_metaworld(env_name_, seed=seed)
        eval_tf_env = load_metaworld(env_name_, seed=seed)
    else:
        raise NotImplementedError('Unsupported environment: %s' % env_name)
    assert len(tf_env.envs) == 1
    assert len(eval_tf_env.envs) == 1

    # By default, the environment observation contains the current state and goal
    # state. By setting the obs_to_goal parameters, the use can specify that the
    # agent should only look at certain subsets of the goal state. The following
    # code modifies the environment observation to include the full state but only
    # the user-specified dimensions of the goal state.
    if env_name.startswith('metaworld'):
        obs_dim = tf_env.observation_spec().shape[0] - 6
    else:
        obs_dim = tf_env.observation_spec().shape[0] // 2

    # if env_name.startswith('antmaze'):
    #     tf_env = tf_py_environment.TFPyEnvironment(
    #         c_learning_utils.CanonicalActionSpaceWrapper(tf_env.envs[0]))
    #     eval_tf_env = tf_py_environment.TFPyEnvironment(
    #         c_learning_utils.CanonicalActionSpaceWrapper(eval_tf_env.envs[0]))

    try:
        start_index = gin.query_parameter('obs_to_goal.start_index')
        assert len(start_index) == 1
        start_index = start_index[0]
    except ValueError:
        start_index = 0
    try:
        end_index = gin.query_parameter('obs_to_goal.end_index')
        assert len(end_index) == 1
        end_index = end_index[0]
    except ValueError:
        end_index = None
    if end_index is None:
        end_index = obs_dim

    # (chongyiz): hardcoded index
    if env_name.startswith('metaworld'):
        start_index = 0
        end_index = 6

    indices = np.concatenate([
        np.arange(obs_dim),
        np.arange(obs_dim + start_index, obs_dim + end_index)
    ])
    tf_env = tf_py_environment.TFPyEnvironment(
        wrappers.ObservationFilterWrapper(tf_env.envs[0], indices))
    eval_tf_env = tf_py_environment.TFPyEnvironment(
        wrappers.ObservationFilterWrapper(eval_tf_env.envs[0], indices))

    return (tf_env, eval_tf_env, obs_dim)


# class SawyerReach(sawyer_xyz.SawyerReachPushPickPlaceEnv):
#     """Wrapper for the sawyer_reach task."""
#
#     def __init__(self):
#         super(SawyerReach, self).__init__(task_type='reach')
#         self.observation_space = gym.spaces.Box(
#             low=np.full(12, -np.inf),
#             high=np.full(12, np.inf),
#             dtype=np.float32)
#
#     def reset(self):
#         goal = self.sample_goals(1)['state_desired_goal'][0]
#         self.goal = goal
#         self._state_goal = goal
#         return self.reset_model()
#
#     def step(self, action):
#         s, r, done, info = super(SawyerReach, self).step(action)
#         r = 0.0
#         done = False
#         return s, r, done, info
#
#     def _get_obs(self):
#         obs = super(SawyerReach, self)._get_obs()
#         return np.concatenate([obs, self.goal, np.zeros(3)])
#
#
# class SawyerPush(sawyer_xyz.SawyerReachPushPickPlaceEnv):
#     """Wrapper for the sawyer_push task."""
#
#     def __init__(self, random_init=False, goal_low=None):
#         assert goal_low is not None
#
#         super(SawyerPush, self).__init__(
#             task_type='push', random_init=random_init, goal_low=goal_low)
#         self.observation_space = gym.spaces.Box(
#             low=np.full(12, -np.inf),
#             high=np.full(12, np.inf),
#             dtype=np.float32)
#
#     @gin.configurable(module='SawyerPush')
#     def reset(self,
#               arm_goal_type='random',
#               fix_z=False,
#               fix_xy=False,
#               fix_g=False,
#               reset_puck=False,
#               in_hand_prob=0,
#               custom_eval=False,
#               reset_to_puck_prob=0.0):
#         assert arm_goal_type in ['random', 'puck', 'goal']
#         if custom_eval and self.MODE == 'eval':
#             arm_goal_type = 'goal'
#             in_hand_prob = 0
#             reset_to_puck_prob = 0.0
#         self._arm_goal_type = arm_goal_type
#         # The arm_goal seems to be set to some (dummy) value before we can reset
#         # the environment.
#         self._arm_goal = np.zeros(3)
#         if fix_g:
#             self._gripper_goal = np.array([0.016])
#         else:
#             self._gripper_goal = np.random.uniform(0, 0.04, (1,))
#         obs = super(SawyerPush, self).reset()
#         if reset_puck:
#             puck_pos = self.sample_goals(1)['state_desired_goal'][0]
#             puck_pos[2] = 0.015
#         else:
#             puck_pos = obs[3:6]
#         # The following line ensures that the puck starts face-up, not on edge.
#         self._set_obj_xyz_quat(puck_pos, 0.0)
#         if np.random.random() < reset_to_puck_prob:
#             obs = self._get_obs()
#             self.data.set_mocap_pos('mocap', obs[3:6])
#             self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
#             for _ in range(10):
#                 self.do_simulation([-1, 1], self.frame_skip)
#
#         if np.random.random() < in_hand_prob:
#             for _ in range(10):
#                 obs, _, _, _ = self.step(np.array([0, 0, 0, 1]))
#             self._set_obj_xyz_quat(obs[:3], 0.0)
#         obs = self._get_obs()
#
#         self.goal = self.sample_goals(1)['state_desired_goal'][0]
#         if fix_z:
#             self.goal[2] = 0.015
#         if fix_xy:
#             self.goal[:2] = obs[3:5]
#
#         self._set_goal_marker(self.goal)
#         self._state_goal = self.goal.copy()
#
#         if arm_goal_type == 'random':
#             self._arm_goal = self.sample_goals(1)['state_desired_goal'][0]
#             if fix_z:
#                 self._arm_goal[2] = 0.015
#         elif arm_goal_type == 'puck':
#             self._arm_goal = obs[3:6]
#         elif arm_goal_type == 'goal':
#             self._arm_goal = self.goal.copy()
#         else:
#             raise NotImplementedError
#         return self._get_obs()
#
#     def step(self, action):
#         try:
#             s, r, done, info = super(SawyerPush, self).step(action)
#         except mujoco_py.MujocoException as me:
#             logging.info('MujocoException: %s', me)
#             s = self.reset()
#             info = {}
#
#         r = 0.0
#         done = False
#         return s, r, done, info
#
#     def _get_obs(self):
#         obs = super(SawyerPush, self)._get_obs()
#         obs = np.concatenate([obs, self._arm_goal, self.goal])
#         return obs
#
#
# class SawyerPushGripper(SawyerPush):
#     """Wrapper for the sawyer_push task, including the gripper in the state."""
#
#     MODE = 'train'
#
#     def __init__(self, random_init=False, goal_low=None):
#         assert goal_low is not None
#
#         super(SawyerPushGripper, self).__init__(
#             random_init=random_init, goal_low=goal_low)
#         self.observation_space = gym.spaces.Box(
#             low=np.full(14, -np.inf), high=np.full(14, np.inf), dtype=np.float32)
#
#     def _get_obs(self):
#         obs = super(SawyerPushGripper, self)._get_obs()
#         gripper = self.get_gripper_pos()
#         obs = np.concatenate(
#             [obs, gripper, self._arm_goal, self.goal, self._gripper_goal])
#         return obs
#
#
# class SawyerWindow(sawyer_xyz.SawyerWindowCloseEnv):
#     """Wrapper for the sawyer_window task."""
#
#     def __init__(self, rotMode='fixed'):  # pylint: disable=invalid-name
#         super(SawyerWindow, self).__init__(random_init=False, rotMode=rotMode)
#         self.observation_space = gym.spaces.Box(
#             low=np.full(12, -np.inf), high=np.full(12, np.inf), dtype=np.float32)
#
#     def sample_goal(self):
#         low = np.array([-0.09, 0.73, 0.15])
#         high = np.array([0.09, 0.73, 0.15])
#         return np.random.uniform(low, high)
#
#     @gin.configurable(module='SawyerWindow')
#     def reset(self, arm_goal_type='random', reset_puck=True):
#         assert arm_goal_type in ['random', 'puck', 'goal']
#         self.goal = self.sample_goal()
#         self._state_goal = self.goal.copy()
#         self._arm_goal = np.zeros(3)
#         super(SawyerWindow, self).reset()
#         # Randomize the window position
#         pos = self.sim.model.body_pos[self.model.body_name2id('window')]
#         if reset_puck:
#             pos[0] = self.sample_goal()[0]
#         else:
#             pos[0] = 0.0
#         self.sim.model.body_pos[self.model.body_name2id('window')] = pos
#         another_pos = pos.copy()
#         another_pos[1] += 0.03
#         self.sim.model.body_pos[self.model.body_name2id(
#             'window_another')] = another_pos
#
#         # We have set the desired state of the window above. We have to step the
#         # environment once (using a null-op action) for these changes to take
#         # effect.
#         obs, _, _, _ = self.step(np.zeros(4))
#         if arm_goal_type == 'random':
#             self._arm_goal = self.sample_goal()
#         elif arm_goal_type == 'puck':
#             self._arm_goal = obs[3:6]
#         elif arm_goal_type == 'goal':
#             self._arm_goal = self.goal.copy()
#         else:
#             raise NotImplementedError
#         return self._get_obs()
#
#     def step(self, action):
#         try:
#             s, r, done, info = super(SawyerWindow, self).step(action)
#         except mujoco_py.MujocoException as me:
#             logging.info('MujocoException: %s', me)
#             s = self.reset()
#             info = {}
#         r = 0.0
#         done = False
#         return s, r, done, info
#
#     def _get_obs(self):
#         obs = super(SawyerWindow, self)._get_obs()
#         return np.concatenate([obs, self._arm_goal, self.goal])
#
#
# class SawyerDrawer(sawyer_xyz.SawyerDrawerOpenEnv):
#     """Wrapper for the sawyer_drawer task."""
#
#     def __init__(self, random_init=False):
#         super(SawyerDrawer, self).__init__(random_init=random_init)
#         self.observation_space = gym.spaces.Box(
#             low=np.full(12, -np.inf), high=np.full(12, np.inf), dtype=np.float32)
#
#     @gin.configurable(module='SawyerDrawer')
#     def reset(self, arm_goal_type='puck'):
#         assert arm_goal_type in ['puck', 'goal']
#         self._arm_goal = np.zeros(3)
#         self.goal = np.zeros(3)
#         self._state_goal = np.zeros(3)
#         obs = super(SawyerDrawer, self).reset()
#         offset = np.random.uniform(-0.2, 0)
#         self._set_obj_xyz(offset)
#
#         self.goal = obs[3:6]
#         self.goal[1] = np.random.uniform(0.5, 0.7)
#         if arm_goal_type == 'puck':
#             self._arm_goal = obs[3:6]
#         elif arm_goal_type == 'goal':
#             self._arm_goal = self.goal.copy()
#         else:
#             raise NotImplementedError
#         return self._get_obs()
#
#     def step(self, action):
#         s, r, done, info = super(SawyerDrawer, self).step(action)
#         r = 0.0
#         done = False
#         return s, r, done, info
#
#     def _get_obs(self):
#         obs = super(SawyerDrawer, self)._get_obs()
#         return np.concatenate([obs, self._arm_goal, self.goal])
#
#
class SawyerDrawerV2(ALL_V2_ENVIRONMENTS['drawer-close-v2']):
    """Wrapper for the SawyerDrawer environment."""

    def __init__(self):
        super(SawyerDrawerV2, self).__init__()
        # self._random_reset_space.low[0] = 0
        # self._random_reset_space.high[0] = 0
        self._partially_observable = False
        self._freeze_rand_vec = False
        self._set_task_called = True
        self._target_pos = np.zeros(0)  # We will overwrite this later.
        self.reset()
        self._freeze_rand_vec = False  # Set False to randomize the goal position.

    # def _get_pos_objects(self):
    #     return self.get_body_com('drawer_link') + np.array([.0, -.16, 0.0])

    def reset_model(self):
        super(SawyerDrawerV2, self).reset_model()
        self._set_obj_xyz(np.random.uniform(-0.15, 0.0))
        self._target_pos = self._get_pos_objects().copy()
        self.data.site_xpos[self.model.site_name2id('goal')] = self._target_pos

        self._set_obj_xyz(np.random.uniform(-0.15, 0.0))
        return self._get_obs()

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=np.full(8, -np.inf),
            high=np.full(8, np.inf),
            dtype=np.float32)

    def _get_obs(self):
        # finger_right, finger_left = (self._get_site_pos('rightEndEffector'),
        #                              self._get_site_pos('leftEndEffector'))
        # tcp_center = (finger_right + finger_left) / 2.0
        # obj = self._get_pos_objects()
        # # Arm position is same as drawer position. We only provide the drawer
        # # Y coordinate.
        # return np.concatenate([tcp_center, [obj[1]],
        #                        self._target_pos, [self._target_pos[1]]])
        pos_hand = self.get_endeff_pos()
        obj = self._get_pos_objects()

        return np.concatenate([pos_hand, [obj[1]],
                               self._target_pos, [self._target_pos[1]]])

    def step(self, action):
        obs, _, done, info = super(SawyerDrawerV2, self).step(action)
        reward = info['success']

        return obs, reward, done, info


# class SawyerFaucet(sawyer_xyz.SawyerFaucetOpenEnv):
#     """Wrapper for the sawyer_faucet task."""
#
#     def __init__(self):
#         super(SawyerFaucet, self).__init__()
#         self.observation_space = gym.spaces.Box(
#             low=np.full(12, -np.inf), high=np.full(12, np.inf), dtype=np.float32)
#
#     @gin.configurable(module='SawyerFaucet')
#     def reset(self, arm_goal_type='goal', init_width=np.pi / 2,
#               goal_width=np.pi / 2):
#         assert arm_goal_type in ['puck', 'goal']
#         self._arm_goal = np.zeros(3)
#         self.goal = np.zeros(3)
#         self._state_goal = np.zeros(3)
#         obs = super(SawyerFaucet, self).reset()
#
#         offset = np.random.uniform(-goal_width, goal_width)
#         self._set_obj_xyz(offset)
#         self.goal = self._get_obs()[3:6]
#
#         offset = np.random.uniform(-init_width, init_width)
#         self._set_obj_xyz(offset)
#         obs = self._get_obs()
#
#         if arm_goal_type == 'puck':
#             self._arm_goal = obs[3:6]
#         elif arm_goal_type == 'goal':
#             self._arm_goal = self.goal.copy()
#         else:
#             raise NotImplementedError
#         return self._get_obs()
#
#     def step(self, action):
#         s, r, done, info = super(SawyerFaucet, self).step(action)
#         r = 0.0
#         done = False
#         return s, r, done, info
#
#     def _get_obs(self):
#         obs = super(SawyerFaucet, self)._get_obs()
#         return np.concatenate([obs, self._arm_goal, self.goal])


class Maze2DBase(pointmaze.MazeEnv):
    def __init__(self, **kwargs):
        super(Maze2DBase, self).__init__(**kwargs)

    def get_dataset(self, h5path=None):
        if h5path is None:
            if self._dataset_url is None:
                raise ValueError("Offline env not configured with a dataset URL.")
            h5path = download_dataset_from_url(self.dataset_url)

        data_dict = {}
        with h5py.File(h5path, 'r') as dataset_file:
            for k in tqdm(get_keys(dataset_file), desc="load datafile"):
                try:  # first try loading as an array
                    data_dict[k] = dataset_file[k][:]
                except ValueError as e:  # try loading as a scalar
                    data_dict[k] = dataset_file[k][()]

        # Run a few quick sanity checks
        for key in ['observations', 'actions', 'rewards', 'terminals']:
            assert key in data_dict, 'Dataset is missing key %s' % key
        N_samples = data_dict['observations'].shape[0]
        # (chongyiz): concatenate goals to observations
        assert 'infos/goal' in data_dict
        data_dict['observations'] = np.concatenate([
            data_dict['observations'], data_dict['infos/goal'], np.zeros([N_samples, 2])], axis=-1)
        if self.observation_space.shape is not None:
            assert data_dict['observations'].shape[1:] == self.observation_space.shape, \
                'Observation shape does not match env: %s vs %s' % (
                    str(data_dict['observations'].shape[1:]), str(self.observation_space.shape))
        assert data_dict['actions'].shape[1:] == self.action_space.shape, \
            'Action shape does not match env: %s vs %s' % (
                str(data_dict['actions'].shape[1:]), str(self.action_space.shape))
        if data_dict['rewards'].shape == (N_samples, 1):
            data_dict['rewards'] = data_dict['rewards'][:, 0]
        assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (
            str(data_dict['rewards'].shape))
        if data_dict['terminals'].shape == (N_samples, 1):
            data_dict['terminals'] = data_dict['terminals'][:, 0]
        assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (
            str(data_dict['rewards'].shape))
        return data_dict

    def step(self, action):
        obs, reward, done, info = super(Maze2DBase, self).step(action)
        done = False
        return obs, reward, done, info

    def _get_obs(self):
        obs = super(Maze2DBase, self)._get_obs()
        return np.concatenate([obs, self._target, np.zeros(2)], dtype=np.float32)


class Maze2DOpenV0(Maze2DBase):
    """Wrapper for the D4RL maze2d-open-v0 task."""

    def __init__(self,
                 maze_spec=pointmaze.OPEN,
                 reward_type='sparse',
                 reset_target=True,
                 ref_min_score=0.01,
                 ref_max_score=20.66,
                 dataset_url='http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-open-sparse.hdf5',
                 **kwargs):
        super(Maze2DOpenV0, self).__init__(
            maze_spec=maze_spec,
            reward_type=reward_type,
            reset_target=reset_target,
            ref_min_score=ref_min_score,
            ref_max_score=ref_max_score,
            dataset_url=dataset_url,
            **kwargs
        )


class Maze2DUMazeV1(Maze2DBase):
    """Wrapper for the D4RL maze2d-umaze-v1 task."""

    def __init__(self,
                 maze_spec=pointmaze.U_MAZE,
                 reward_type='sparse',
                 reset_target=True,
                 ref_min_score=23.85,
                 ref_max_score=161.86,
                 dataset_url='http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5',
                 **kwargs):
        super(Maze2DUMazeV1, self).__init__(
            maze_spec=maze_spec,
            reward_type=reward_type,
            reset_target=reset_target,
            ref_min_score=ref_min_score,
            ref_max_score=ref_max_score,
            dataset_url=dataset_url,
            **kwargs
        )


class Maze2DMediumV1(Maze2DBase):
    """Wrapper for the D4RL maze2d-medium-v1 task."""

    def __init__(self,
                 maze_spec=pointmaze.MEDIUM_MAZE,
                 reward_type='sparse',
                 reset_target=True,
                 ref_min_score=13.13,
                 ref_max_score=277.39,
                 dataset_url='http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-medium-sparse-v1.hdf5',
                 **kwargs):
        super(Maze2DMediumV1, self).__init__(
            maze_spec=maze_spec,
            reward_type=reward_type,
            reset_target=reset_target,
            ref_min_score=ref_min_score,
            ref_max_score=ref_max_score,
            dataset_url=dataset_url,
            **kwargs
        )


class Maze2DLargeV1(Maze2DBase):
    """Wrapper for the D4RL maze2d-large-v1 task."""

    def __init__(self,
                 maze_spec=pointmaze.LARGE_MAZE,
                 reward_type='sparse',
                 reset_target=True,
                 ref_min_score=6.7,
                 ref_max_score=273.99,
                 dataset_url='http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5',
                 **kwargs):
        super(Maze2DLargeV1, self).__init__(
            maze_spec=maze_spec,
            reward_type=reward_type,
            reset_target=reset_target,
            ref_min_score=ref_min_score,
            ref_max_score=ref_max_score,
            dataset_url=dataset_url,
            **kwargs
        )


class AntMazeBase(locomotion.ant.AntMazeEnv):
    def __init__(self, **kwargs):
        super(AntMazeBase, self).__init__(**kwargs)

        # TODO (chongyiz): convert action_space of antmaze to [-1, 1]
        self.orig_action_space = self.action_space
        self.action_space = gym.spaces.Box(
            low=-np.ones(self.orig_action_space.shape),
            high=np.ones(self.orig_action_space.shape),
            dtype=np.float32
        )

    def get_dataset(self, h5path=None):
        if h5path is None:
            if self._dataset_url is None:
                raise ValueError("Offline env not configured with a dataset URL.")
            h5path = download_dataset_from_url(self.dataset_url)

        data_dict = {}
        with h5py.File(h5path, 'r') as dataset_file:
            for k in tqdm(get_keys(dataset_file), desc="load datafile"):
                try:  # first try loading as an array
                    data_dict[k] = dataset_file[k][:]
                except ValueError as e:  # try loading as a scalar
                    data_dict[k] = dataset_file[k][()]

        # Run a few quick sanity checks
        for key in ['observations', 'actions', 'rewards', 'terminals']:
            assert key in data_dict, 'Dataset is missing key %s' % key
        N_samples = data_dict['observations'].shape[0]
        # (chongyiz): concatenate goals to observations
        assert 'infos/goal' in data_dict
        data_dict['observations'] = np.concatenate([
            data_dict['observations'], data_dict['infos/goal'], np.zeros([N_samples, 27])], axis=-1)
        if self.observation_space.shape is not None:
            assert data_dict['observations'].shape[1:] == self.observation_space.shape, \
                'Observation shape does not match env: %s vs %s' % (
                    str(data_dict['observations'].shape[1:]), str(self.observation_space.shape))
        assert data_dict['actions'].shape[1:] == self.action_space.shape, \
            'Action shape does not match env: %s vs %s' % (
                str(data_dict['actions'].shape[1:]), str(self.action_space.shape))
        if data_dict['rewards'].shape == (N_samples, 1):
            data_dict['rewards'] = data_dict['rewards'][:, 0]
        assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (
            str(data_dict['rewards'].shape))
        if data_dict['terminals'].shape == (N_samples, 1):
            data_dict['terminals'] = data_dict['terminals'][:, 0]
        assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (
            str(data_dict['rewards'].shape))
        return data_dict

    def step(self, action):
        if hasattr(self, 'orig_action_space'):
            scale = self.orig_action_space.high - self.orig_action_space.low
            offset = self.orig_action_space.low

            # Map action to [0, 1].
            action = 0.5 * (action + 1.0)

            # Map action to [spec.minimum, spec.maximum].
            action *= scale
            action += offset

        obs, reward, done, info = super(AntMazeBase, self).step(action)
        return obs, reward, done, info

    def _get_obs(self):
        obs = super(AntMazeBase, self)._get_obs()
        return np.concatenate([obs, self.target_goal, np.zeros(27)], dtype=np.float32)


class AntMazeUmazeV2(AntMazeBase):
    def __init__(self,
                 maze_map=locomotion.maze_env.U_MAZE_TEST,
                 reward_type='sparse',
                 dataset_url='http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5',
                 non_zero_reset=True,
                 eval=True,
                 maze_size_scaling=4.0,
                 ref_min_score=0.0,
                 ref_max_score=1.0,
                 v2_resets=True):
        super(AntMazeUmazeV2, self).__init__(
            maze_map=maze_map,
            reward_type=reward_type,
            dataset_url=dataset_url,
            non_zero_reset=non_zero_reset,
            eval=eval,
            maze_size_scaling=maze_size_scaling,
            ref_min_score=ref_min_score,
            ref_max_score=ref_max_score,
            v2_resets=v2_resets)


class AntMazeUmazeDiverseV2(AntMazeBase):
    def __init__(self,
                 maze_map=locomotion.maze_env.U_MAZE_TEST,
                 reward_type='sparse',
                 dataset_url='http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_u-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5',
                 non_zero_reset=False,
                 eval=True,
                 maze_size_scaling=4.0,
                 ref_min_score=0.0,
                 ref_max_score=1.0,
                 v2_resets=True):
        super(AntMazeUmazeDiverseV2, self).__init__(
            maze_map=maze_map,
            reward_type=reward_type,
            dataset_url=dataset_url,
            non_zero_reset=non_zero_reset,
            eval=eval,
            maze_size_scaling=maze_size_scaling,
            ref_min_score=ref_min_score,
            ref_max_score=ref_max_score,
            v2_resets=v2_resets)


class AntMazeMediumDiverseV2(AntMazeBase):
    def __init__(self,
                 maze_map=locomotion.maze_env.BIG_MAZE_TEST,
                 reward_type='sparse',
                 dataset_url='http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5',
                 non_zero_reset=False,
                 eval=True,
                 maze_size_scaling=4.0,
                 ref_min_score=0.0,
                 ref_max_score=1.0,
                 v2_resets=True):
        super(AntMazeMediumDiverseV2, self).__init__(
            maze_map=maze_map,
            reward_type=reward_type,
            dataset_url=dataset_url,
            non_zero_reset=non_zero_reset,
            eval=eval,
            maze_size_scaling=maze_size_scaling,
            ref_min_score=ref_min_score,
            ref_max_score=ref_max_score,
            v2_resets=v2_resets)


class AntMazeLargeDiverseV2(AntMazeBase):
    def __init__(self,
                 maze_map=locomotion.maze_env.HARDEST_MAZE_TEST,
                 reward_type='sparse',
                 dataset_url='http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5',
                 non_zero_reset=False,
                 eval=True,
                 maze_size_scaling=4.0,
                 ref_min_score=0.0,
                 ref_max_score=1.0,
                 v2_resets=True):
        super(AntMazeLargeDiverseV2, self).__init__(
            maze_map=maze_map,
            reward_type=reward_type,
            dataset_url=dataset_url,
            non_zero_reset=non_zero_reset,
            eval=eval,
            maze_size_scaling=maze_size_scaling,
            ref_min_score=ref_min_score,
            ref_max_score=ref_max_score,
            v2_resets=v2_resets)
