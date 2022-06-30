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

"""Utility for loading the AntMaze environments."""
import d4rl
from d4rl.offline_env import download_dataset_from_url, get_keys

import gym
import numpy as np
import h5py
from tqdm import tqdm

R = 'r'
G = 'g'
U_MAZE = [[1, 1, 1, 1, 1],
          [1, R, G, G, 1],
          [1, 1, 1, G, 1],
          [1, G, G, G, 1],
          [1, 1, 1, 1, 1]]

BIG_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, G, 1, 1, G, G, 1],
            [1, G, G, 1, G, G, G, 1],
            [1, 1, G, G, G, 1, 1, 1],
            [1, G, G, 1, G, G, G, 1],
            [1, G, 1, G, G, 1, G, 1],
            [1, G, G, G, 1, G, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]]

HARDEST_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, R, G, G, G, 1, G, G, G, G, G, 1],
                [1, G, 1, 1, G, 1, G, 1, G, 1, G, 1],
                [1, G, G, G, G, G, G, 1, G, G, G, 1],
                [1, G, 1, 1, 1, 1, G, 1, 1, 1, G, 1],
                [1, G, G, 1, G, 1, G, G, G, G, G, 1],
                [1, 1, G, 1, G, 1, G, 1, G, 1, 1, 1],
                [1, G, G, 1, G, G, G, 1, G, G, G, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]


class AntMaze(d4rl.locomotion.ant.AntMazeEnv):
    """Utility wrapper for the AntMaze environments.

    For comparisons in the offline RL setting, we used unmodified AntMaze tasks,
    without this wrapper.
    """

    # register(
    #     id='antmaze-umaze-v2',
    #     entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    #     max_episode_steps=700,
    #     kwargs={
    #         'maze_map': maze_env.U_MAZE_TEST,
    #         'reward_type': 'sparse',
    #         'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5',
    #         'non_zero_reset': False,
    #         'eval': True,
    #         'maze_size_scaling': 4.0,
    #         'ref_min_score': 0.0,
    #         'ref_max_score': 1.0,
    #         'v2_resets': True,
    #     }
    # )

    def __init__(self, map_name, non_zero_reset=False, dataset_url=None):
        self._goal_obs = np.zeros(29)
        if map_name == 'umaze':
            maze_map = U_MAZE
        elif map_name == 'medium':
            maze_map = BIG_MAZE
        elif map_name == 'large':
            maze_map = HARDEST_MAZE
        else:
            raise NotImplementedError
        super(AntMaze, self).__init__(maze_map=maze_map,
                                      reward_type='sparse',
                                      non_zero_reset=non_zero_reset,
                                      eval=True,
                                      maze_size_scaling=4.0,
                                      ref_max_score=1.0,
                                      ref_min_score=0.0,
                                      high=np.full((58,), np.inf),
                                      dtype=np.float32,
                                      dataset_url=dataset_url)

        # TODO (chongyiz): convert action_space of antmaze to [-1, 1]
        self.orig_action_space = self.action_space
        self.action_space = gym.spaces.Box(
            low=-np.ones(self.orig_action_space.shape),
            high=np.ones(self.orig_action_space.shape),
            dtype=np.float32
        )

    # def get_dataset(self, h5path=None):
    #     data_dict = super().get_dataset(h5path=h5path)
    #     assert 'infos/goal' in data_dict
    #     N_samples = len(data_dict['observations'])
    #     data_dict['observations'] = np.concatenate([
    #         data_dict['observations'], data_dict['infos/goal'],
    #         np.zeros([N_samples, 27])], axis=-1)
    #
    #     return data_dict

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

    def reset(self):
        super(AntMaze, self).reset()
        goal_xy = self._goal_sampler(np.random)
        state = self.sim.get_state()
        state = state._replace(
            qpos=np.concatenate([goal_xy, state.qpos[2:]]))
        self.sim.set_state(state)
        for _ in range(50):
            self.do_simulation(np.zeros(8), self.frame_skip)
        self._goal_obs = self.BASE_ENV._get_obs(self).copy()  # pylint: disable=protected-access
        super(AntMaze, self).reset()
        return self._get_obs()

    def step(self, action):
        if hasattr(self, 'orig_action_space'):
            scale = self.orig_action_space.high - self.orig_action_space.low
            offset = self.orig_action_space.low

            # Map action to [0, 1].
            action = 0.5 * (action + 1.0)

            # Map action to [spec.minimum, spec.maximum].
            action *= scale
            action += offset

        super(AntMaze, self).step(action)
        s = self._get_obs()
        dist = np.linalg.norm(self._goal_obs[:2] - s[:2])
        # Distance threshold from [RIS, Chane-Sane '21] and [UPN, Srinivas '18].
        r = (dist <= 0.5)
        done = False
        info = {}
        return s, r, done, info

    def _get_obs(self):
        assert self._expose_all_qpos  # pylint: disable=protected-access
        s = self.BASE_ENV._get_obs(self)  # pylint: disable=protected-access
        return np.concatenate([s, self._goal_obs]).astype(np.float32)

    def _get_reset_location(self):
        if np.random.random() < 0.5:
            return super(AntMaze, self)._get_reset_location()
        else:
            return self._goal_sampler(np.random)
