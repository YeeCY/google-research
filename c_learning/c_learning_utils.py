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

"""Helper functions for C-learning."""

import enum
import collections

import numpy as np
import gym

import gin
import tensorflow as tf
from tf_agents.agents.ddpg import critic_network
from tf_agents.metrics import tf_metric
from tf_agents.metrics import tf_metrics
from tf_agents.utils import common
from tf_agents.utils import nest_utils
from tf_agents.replay_buffers import tf_uniform_replay_buffer
# from tf_agents.environments import PyEnvironmentBaseWrapper
# from tf_agents.specs import BoundedArraySpec

from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv


class CLearningGoalLabel(enum.Enum):
    NEXT = 1
    NEXT_FUTURE = 2
    FUTURE = 3
    RANDOM = 4


def truncated_geometric(horizon, gamma):
    """Generates sampled from a truncated geometric distribution.

    Args:
      horizon: A 1-d tensor of horizon lengths for each element in the batch.
        The returned samples will be less than the corresponding horizon.
      gamma: The discount factor. Importantly, we sample from a Geom(1 - gamma)
        distribution.
    Returns:
      indices: A 1-d tensor of integers, one for each element of the batch.
    """
    max_horizon = tf.reduce_max(horizon)
    batch_size = tf.shape(horizon)[0]
    indices = tf.tile(
        tf.range(max_horizon, dtype=tf.float32)[None], (batch_size, 1))
    # probs = tf.where(indices < horizon[:, None], gamma ** indices,
    #                  tf.zeros_like(indices))
    # (chongyiz): update future sampling probs to start from next_step
    probs = tf.where(tf.math.logical_and(tf.zeros_like(indices) < indices, indices < horizon[:, None]),
                     gamma ** (indices - 1), tf.zeros_like(indices))
    probs = probs / tf.reduce_sum(probs, axis=1)[:, None]
    indices = tf.random.categorical(tf.math.log(probs), 1, dtype=tf.int32)
    return indices[:, 0]  # Remove the extra dimension.


def get_future_goals(observation, discount, gamma):
    """Samples future goals according to a geometric distribution."""
    num_obs = observation.shape[0]
    traj_len = observation.shape[1]
    first_terminal_or_zero = tf.argmax(
        discount == 0, axis=1, output_type=tf.int32)
    any_terminal = tf.reduce_any(discount == 0, axis=1)
    first_terminal = tf.where(any_terminal, first_terminal_or_zero, traj_len)
    first_terminal = tf.cast(first_terminal, tf.float32)
    if num_obs == 0:
        # The truncated_geometric function breaks if called on an empty list.
        # In that case, we manually create an empty list of future goals.
        indices = tf.zeros((0,), dtype=tf.int32)
    else:
        indices = truncated_geometric(first_terminal, gamma)
    stacked_indices = tf.stack([tf.range(num_obs), indices], axis=1)
    return tf.gather_nd(observation, stacked_indices)


def get_last_goals(observation, discount):
    """Extracts that final observation before termination.

    Args:
      observation: a B x T x D tensor storing the next T time steps. These time
        steps may be part of a new trajectory. This function will only consider
        observations that occur before the first terminal.
      discount: a B x T tensor indicating whether the episode has terminated.
    Returns:
      last_obs: a B x D tensor storing the last observation in each trajectory
        that occurs before the first terminal.
    """
    num_obs = observation.shape[0]
    traj_len = observation.shape[1]
    first_terminal_or_zero = tf.argmax(
        discount == 0, axis=1, output_type=tf.int32)
    any_terminal = tf.reduce_any(discount == 0, axis=1)
    first_terminal = tf.where(any_terminal, first_terminal_or_zero, traj_len)
    # If the first state is terminal then first_terminal - 1 = -1. In this case we
    # use the state itself as the goal.
    last_nonterminal = tf.clip_by_value(first_terminal - 1, 0, traj_len)
    stacked_indices = tf.stack([tf.range(num_obs), last_nonterminal], axis=1)
    last_obs = tf.gather_nd(observation, stacked_indices)
    return last_obs


@gin.configurable
def obs_to_goal(obs, start_index=(0,), end_index=(None,)):
    goal = []
    for start_idx, end_idx in zip(start_index, end_index):
        if end_index is None:
            goal.append(obs[:, start_idx:])
        else:
            goal.append(obs[:, start_idx:end_idx])
    return tf.concat(goal, axis=-1)


@gin.configurable
def goal_fn(experience,
            buffer_info,
            relabel_orig_prob=0.0,
            relabel_next_prob=0.5,
            relabel_future_prob=0.0,
            relabel_last_prob=0.0,
            batch_size=None,
            obs_dim=None,
            gamma=None):
    """Given experience, sample goals in three ways.

    The three ways are using the next state, an arbitrary future state, or a
    random state. For the future state relabeling, care must be taken to ensure
    that we don't sample experience across the episode boundary. We automatically
    set relabel_random_prob = (1 - relabel_next_prob - relabel_future_prob).

    Args:
      experience: The experience that we aim to relabel.
      buffer_info: Information about the replay buffer. We will not change this.
      relabel_orig_prob: (float) Fraction of experience to not relabel.
      relabel_next_prob: (float) Fraction of experience to relabel with the next
        state.
      relabel_future_prob: (float) Fraction of experience to relabel with a future
        state.
      relabel_last_prob: (float) Fraction of experience to relabel with the
        final state.
      batch_size: (int) The size of the batch.
      obs_dim: (int) The dimension of the observation.
      gamma: (float) The discount factor. Future states are sampled according to
        a Geom(1 - gamma) distribution.
    Returns:
      experience: A modified version of the input experience where the goals
        have been changed and the rewards and terminal flags are recomputed.
      buffer_info: Information about the replay buffer.

    """
    assert batch_size is not None
    assert obs_dim is not None
    assert gamma is not None
    relabel_orig_num = int(relabel_orig_prob * batch_size)
    relabel_next_num = int(relabel_next_prob * batch_size)
    relabel_future_num = int(relabel_future_prob * batch_size)
    relabel_last_num = int(relabel_last_prob * batch_size)
    relabel_random_num = batch_size - (
            relabel_orig_num + relabel_next_num + relabel_future_num +
            relabel_last_num)
    assert relabel_random_num >= 0

    orig_goals = experience.observation[:relabel_orig_num, 0, obs_dim:]

    index = relabel_orig_num
    next_goals = experience.observation[index:index + relabel_next_num,
                 1, :obs_dim]

    index = relabel_orig_num + relabel_next_num
    future_goals = get_future_goals(
        experience.observation[index:index + relabel_future_num, :, :obs_dim],
        experience.discount[index:index + relabel_future_num], gamma)

    index = relabel_orig_num + relabel_next_num + relabel_future_num
    last_goals = get_last_goals(
        experience.observation[index:index + relabel_last_num, :, :obs_dim],
        experience.discount[index:index + relabel_last_num])

    # For random goals we take other states from the same batch.
    random_goals = tf.random.shuffle(experience.observation[:relabel_random_num,
                                     0, :obs_dim])
    new_goals = obs_to_goal(tf.concat([next_goals, future_goals,
                                       last_goals, random_goals], axis=0))
    goals = tf.concat([orig_goals, new_goals], axis=0)

    obs = experience.observation[:, :2, :obs_dim]
    reward = tf.reduce_all(obs_to_goal(obs[:, 1]) == goals, axis=-1)
    reward = tf.cast(reward, tf.float32)
    reward = tf.tile(reward[:, None], [1, 2])
    new_obs = tf.concat([obs, tf.tile(goals[:, None, :], [1, 2, 1])], axis=2)

    next_goals = obs_to_goal(experience.observation[:, 1, :obs_dim])
    next_future_goals = obs_to_goal(get_future_goals(
        experience.observation[:, 1:, :obs_dim],
        experience.discount[:, 1:], gamma))
    future_goals = obs_to_goal(get_future_goals(
        experience.observation[:, :, :obs_dim],
        experience.discount[:], gamma))
    random_goals = obs_to_goal(
        tf.random.shuffle(experience.observation[:, 0, :obs_dim]))
    full_batch_new_goals = tf.concat(
        [next_goals, next_future_goals, future_goals, random_goals],
        axis=1)
    new_obs = tf.concat([new_obs, tf.tile(full_batch_new_goals[:, None, :], [1, 2, 1])], axis=2)

    experience = experience.replace(
        observation=new_obs,
        action=experience.action[:, :2],
        step_type=experience.step_type[:, :2],
        next_step_type=experience.next_step_type[:, :2],
        discount=experience.discount[:, :2],
        reward=reward,
    )
    return experience, buffer_info


@gin.configurable
def offline_goal_fn(experience,
                    buffer_info,
                    relabel_orig_prob=0.0,
                    relabel_next_prob=0.5,
                    # relabel_future_prob=0.0,
                    relabel_next_future_prob=0.0,
                    relabel_last_prob=0.0,
                    batch_size=None,
                    obs_dim=None,
                    gamma=None,
                    setting='b'):
    """Given experience, sample goals in three ways.

    The three ways are using the next state, an arbitrary future state, or a
    random state. For the future state relabeling, care must be taken to ensure
    that we don't sample experience across the episode boundary. We automatically
    set relabel_random_prob = (1 - relabel_next_prob - relabel_future_prob).

    Args:
      experience: The experience that we aim to relabel.
      buffer_info: Information about the replay buffer. We will not change this.
      relabel_orig_prob: (float) Fraction of experience to not relabel.
      relabel_next_prob: (float) Fraction of experience to relabel with the next
        state.
      relabel_future_prob: (float) Fraction of experience to relabel with a future
        state.
      relabel_last_prob: (float) Fraction of experience to relabel with the
        final state.
      batch_size: (int) The size of the batch.
      obs_dim: (int) The dimension of the observation.
      gamma: (float) The discount factor. Future states are sampled according to
        a Geom(1 - gamma) distribution.
    Returns:
      experience: A modified version of the input experience where the goals
        have been changed and the rewards and terminal flags are recomputed.
      buffer_info: Information about the replay buffer.

    """
    assert batch_size is not None
    assert obs_dim is not None
    assert gamma is not None

    # TODO (chongyiz): Currently, we only implement off-policy version,
    #  and we are gonna implement TD(\lambda) version later
    if setting == 'b':
        assert relabel_next_prob + relabel_next_future_prob == 0.5
    elif setting == 'c':
        assert relabel_next_prob == 0.5
    else:
        raise NotImplementedError

    relabel_orig_num = int(relabel_orig_prob * batch_size)
    relabel_next_num = int(relabel_next_prob * batch_size)
    relabel_last_num = int(relabel_last_prob * batch_size)
    if setting == 'b':
        relabel_next_future_num = int(relabel_next_future_prob * batch_size)
        relabel_future_num = batch_size - (
                relabel_orig_num + relabel_next_num + relabel_next_future_num + relabel_last_num)
        relabel_random_num = 0
        assert relabel_future_num >= 0
    elif setting == 'c':
        # relabel_future_num = int(relabel_future_prob * batch_size)
        relabel_future_num = 0
        relabel_next_future_num = 0
        relabel_random_num = batch_size - (
                relabel_orig_num + relabel_next_num + relabel_last_num)
        assert relabel_random_num >= 0

    orig_goals = experience.observation[:relabel_orig_num, 0, obs_dim:]

    index = relabel_orig_num
    next_goals = experience.observation[index:index + relabel_next_num,
                 1, :obs_dim]

    index = relabel_orig_num + relabel_next_num
    next_future_goals = get_future_goals(
        experience.observation[index:index + relabel_next_future_num, 1:, :obs_dim],
        experience.discount[index:index + relabel_next_future_num, 1:], gamma)

    index = relabel_orig_num + relabel_next_num + relabel_next_future_num
    future_goals = get_future_goals(
        experience.observation[index:index + relabel_future_num, :, :obs_dim],
        experience.discount[index:index + relabel_future_num], gamma)

    index = relabel_orig_num + relabel_next_num + relabel_future_num + relabel_next_future_num
    last_goals = get_last_goals(
        experience.observation[index:index + relabel_last_num, :, :obs_dim],
        experience.discount[index:index + relabel_last_num])

    # For random goals we take other states from the same batch.
    random_goals = tf.random.shuffle(experience.observation[:relabel_random_num,
                                     0, :obs_dim])
    new_goals = obs_to_goal(tf.concat([next_goals, next_future_goals, future_goals,
                                       last_goals, random_goals], axis=0))
    goals = tf.concat([orig_goals, new_goals], axis=0)

    obs = experience.observation[:, :2, :obs_dim]

    # (chongyiz): construct rewards
    if setting == 'b':
        next_mask = tf.reduce_all(obs_to_goal(obs[:, 1]) == goals, axis=-1)
        next_label = tf.cast(next_mask, tf.float32) * CLearningGoalLabel.NEXT.value

        next_future_mask = np.zeros(batch_size, dtype=np.float32)
        next_future_mask[
        relabel_orig_num + relabel_next_num:relabel_orig_num + relabel_next_num + relabel_next_future_num] = 1.0
        next_future_mask = tf.convert_to_tensor(next_future_mask)
        next_future_label = next_future_mask * CLearningGoalLabel.NEXT_FUTURE.value

        future_mask = tf.math.logical_and(
            tf.reduce_all(obs_to_goal(obs[:, 1]) != goals, axis=-1), ~tf.cast(next_future_mask, tf.bool))
        future_label = tf.cast(future_mask, tf.float32) * CLearningGoalLabel.FUTURE.value
        goal_label = next_label + next_future_label + future_label
    elif setting == 'c':
        next_mask = tf.reduce_all(obs_to_goal(obs[:, 1]) == goals, axis=-1)
        next_label = tf.cast(next_mask, tf.float32) * CLearningGoalLabel.NEXT.value

        random_mask = tf.reduce_all(obs_to_goal(obs[:, 1]) != goals, axis=-1)
        random_label = tf.cast(random_mask, tf.float32) * CLearningGoalLabel.RANDOM.value
        goal_label = next_label + random_label
    else:
        raise NotImplementedError

    goal_label = tf.tile(goal_label[:, None], [1, 2])
    new_obs = tf.concat([obs, tf.tile(goals[:, None, :], [1, 2, 1])], axis=2)

    next_goals = obs_to_goal(experience.observation[:, 1, :obs_dim])
    next_future_goals = obs_to_goal(get_future_goals(
        experience.observation[:, 1:, :obs_dim],
        experience.discount[:, 1:], gamma))
    future_goals = obs_to_goal(get_future_goals(
        experience.observation[:, :, :obs_dim],
        experience.discount[:], gamma))
    random_goals = obs_to_goal(
        tf.random.shuffle(experience.observation[:, 0, :obs_dim]))
    full_batch_new_goals = tf.concat(
        [next_goals, next_future_goals, future_goals, random_goals],
        axis=1)

    new_obs = tf.concat([new_obs, tf.tile(full_batch_new_goals[:, None, :], [1, 2, 1])], axis=2)
    new_reward = tf.stack([experience.reward[:, :2], goal_label], axis=2)

    experience = experience.replace(
        observation=new_obs,
        action=experience.action[:, :2],
        step_type=experience.step_type[:, :2],
        next_step_type=experience.next_step_type[:, :2],
        discount=experience.discount[:, :2],
        reward=new_reward,
    )
    return experience, buffer_info


def sequence_dataset(path_buffer, max_path_length):
    dataset = path_buffer._buffer
    N = path_buffer.n_transitions_stored
    data_ = collections.defaultdict(list)

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminal'][i])
        final_timestep = (episode_step == max_path_length - 1)

        for k in dataset:
            data_[k].append(np.squeeze(dataset[k][i]))

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


@gin.configurable
class ClassifierCriticNetwork(critic_network.CriticNetwork):
    """Creates a critic network."""

    def __init__(self,
                 input_tensor_spec,
                 observation_fc_layer_params=None,
                 action_fc_layer_params=None,
                 joint_fc_layer_params=None,
                 kernel_initializer=None,
                 last_kernel_initializer=None,
                 name='ClassifierCriticNetwork'):
        super(ClassifierCriticNetwork, self).__init__(
            input_tensor_spec,
            observation_fc_layer_params=observation_fc_layer_params,
            action_fc_layer_params=action_fc_layer_params,
            joint_fc_layer_params=joint_fc_layer_params,
            kernel_initializer=kernel_initializer,
            last_kernel_initializer=last_kernel_initializer,
            name=name,
        )

        last_layers = [
            tf.keras.layers.Dense(
                1,
                activation=tf.math.sigmoid,
                kernel_initializer=last_kernel_initializer,
                name='value')
        ]
        self._joint_layers = self._joint_layers[:-1] + last_layers


class BaseDistanceMetric(tf_metric.TFStepMetric):
    """Computes the initial distance to the goal."""

    def __init__(self,
                 prefix='Metrics',
                 dtype=tf.float32,
                 batch_size=1,
                 buffer_size=10,
                 obs_dim=None,
                 start_index=(0,),
                 end_index=(None,),
                 name=None):
        assert obs_dim is not None
        self._start_index = start_index
        self._end_index = end_index
        self._obs_dim = obs_dim
        name = self.NAME if name is None else name
        super(BaseDistanceMetric, self).__init__(name=name, prefix=prefix)
        self._buffer = tf_metrics.TFDeque(buffer_size, dtype)
        self._dist_buffer = tf_metrics.TFDeque(
            2000, dtype)  # Episodes should have length less than 2k
        self.dtype = dtype

    @common.function(autograph=True)
    def call(self, trajectory):
        obs = trajectory.observation
        s = obs[:, :self._obs_dim]
        g = obs[:, self._obs_dim:]
        dist_to_goal = tf.norm(
            obs_to_goal(obs_to_goal(s), self._start_index, self._end_index) -
            obs_to_goal(g, self._start_index, self._end_index),
            axis=1)
        tf.assert_equal(tf.shape(obs)[0], 1)
        if trajectory.is_mid():
            self._dist_buffer.extend(dist_to_goal)
        if trajectory.is_last()[0] and self._dist_buffer.length > 0:
            self._update_buffer()
            self._dist_buffer.clear()
        return trajectory

    def result(self):
        return self._buffer.mean()

    @common.function
    def reset(self):
        self._buffer.clear()

    def _update_buffer(self):
        raise NotImplementedError


class InitialDistance(BaseDistanceMetric):
    """Computes the initial distance to the goal."""
    NAME = 'InitialDistance'

    def _update_buffer(self):
        initial_dist = self._dist_buffer.data[0]
        self._buffer.add(initial_dist)


class FinalDistance(BaseDistanceMetric):
    """Computes the final distance to the goal."""
    NAME = 'FinalDistance'

    def _update_buffer(self):
        final_dist = self._dist_buffer.data[-1]
        self._buffer.add(final_dist)


class AverageDistance(BaseDistanceMetric):
    """Computes the average distance to the goal."""
    NAME = 'AverageDistance'

    def _update_buffer(self):
        avg_dist = self._dist_buffer.mean()
        self._buffer.add(avg_dist)


class MinimumDistance(BaseDistanceMetric):
    """Computes the minimum distance to the goal."""
    NAME = 'MinimumDistance'

    def _update_buffer(self):
        min_dist = self._dist_buffer.min()
        tf.Assert(
            tf.math.is_finite(min_dist), [
                min_dist, self._dist_buffer.length, self._dist_buffer._head,  # pylint: disable=protected-access
                self._dist_buffer.data
            ],
            summarize=1000)
        self._buffer.add(min_dist)


class DeltaDistance(BaseDistanceMetric):
    """Computes the net distance traveled towards the goal. Positive is good."""
    NAME = 'DeltaDistance'

    def _update_buffer(self):
        delta_dist = self._dist_buffer.data[0] - self._dist_buffer.data[-1]
        self._buffer.add(delta_dist)


class TFUniformReplayBuffer(tf_uniform_replay_buffer.TFUniformReplayBuffer):
    def add_episode(self, episode_items):
        """Adds an episode of items to the replay buffer.

        Args:
          episode_items: An item or list/tuple/nest of episodic items to be added to the replay
            buffer. `episode_items` must match the data_spec of this class, with a
            batch_size dimension added to the beginning of each tensor/array.

        Returns:
          Adds `items` to the replay buffer.
        """
        nest_utils.assert_same_structure(episode_items, self._data_spec)
        num_timesteps = nest_utils.get_outer_shape(
            tf.nest.map_structure(tf.convert_to_tensor, episode_items),
            self._data_spec)[0]

        with tf.device(self._device), tf.name_scope(self._scope):
            min_id = tf.identity(self._last_id + 1)
            max_id = self._increment_last_id(tf.cast(num_timesteps, tf.int64))
            ids = tf.range(min_id, max_id + 1)
            write_rows = self._get_rows_for_id(ids)
            write_id_op = self._id_table.write(write_rows, ids)
            write_data_op = self._data_table.write(write_rows, episode_items)
            return tf.group(write_id_op, write_data_op)


class SuccessRate(tf_metric.TFStepMetric):
    NAME = 'SuccessRate'

    def __init__(self,
                 prefix='Metrics',
                 dtype=tf.float32,
                 # batch_size=1,
                 buffer_size=10,
                 # obs_dim=None,
                 # start_index=(0,),
                 # end_index=(None,),
                 name=None):
        # assert obs_dim is not None
        # self._start_index = start_index
        # self._end_index = end_index
        # self._obs_dim = obs_dim
        name = self.NAME if name is None else name
        super(SuccessRate, self).__init__(name=name, prefix=prefix)
        self._buffer = tf_metrics.TFDeque(buffer_size, dtype)
        self._reward_buffer = tf_metrics.TFDeque(
            2000, dtype)  # Episodes should have length less than 2k
        self.dtype = dtype

    @common.function(autograph=True)
    def call(self, trajectory):
        # obs = trajectory.observation
        # s = obs[:, :self._obs_dim]
        # g = obs[:, self._obs_dim:]
        # dist_to_goal = tf.norm(
        #     obs_to_goal(obs_to_goal(s), self._start_index, self._end_index) -
        #     obs_to_goal(g, self._start_index, self._end_index),
        #     axis=1)
        reward = trajectory.reward
        tf.assert_equal(tf.shape(reward)[0], 1)
        if trajectory.is_mid():
            self._reward_buffer.extend(reward)
        if trajectory.is_last()[0] and self._reward_buffer.length > 0:
            self._update_buffer()
            self._reward_buffer.clear()
        return trajectory

    def result(self):
        return self._buffer.mean()

    @common.function
    def reset(self):
        self._buffer.clear()

    def _update_buffer(self):
        # raise NotImplementedError
        success = tf.cast(
            tf.reduce_sum(self._reward_buffer.data) >= 1, float)
        print("success: {}".format(success))
        self._buffer.add(success * 100)


# class FinalSuccessRate(BaseSuccessRateMetric):
#     """Computes the final success rate."""
#     NAME = 'FinalSuccessRate'
#
#     def _update_buffer(self):
#         final_success_rate = self._success_buffer.data[-1]
#         self._buffer.add(final_success_rate)
#
#
# class AverageSuccessRate(BaseSuccessRateMetric):
#     """Computes the average success rate."""
#     NAME = 'AverageSuccessRate'
#
#     def _update_buffer(self):
#         avg_success_rate = self._success_buffer.mean()
#         self._buffer.add(avg_success_rate)


class AverageNormalizedScore(tf_metric.TFStepMetric):
    """Computes the average normalized score."""
    NAME = 'AverageNormalizedScore'

    def __init__(self,
                 prefix='Metrics',
                 dtype=tf.float32,
                 buffer_size=10,
                 # obs_dim=None,
                 # start_index=(0,),
                 # end_index=(None,),
                 ref_max_score=None,
                 ref_min_score=None,
                 name=None):
        # assert obs_dim is not None
        assert ref_max_score is not None
        assert ref_min_score is not None
        self._ref_max_score = ref_max_score
        self._ref_min_score = ref_min_score
        # self._start_index = start_index
        # self._end_index = end_index
        # self._obs_dim = obs_dim
        name = self.NAME if name is None else name
        super(AverageNormalizedScore, self).__init__(name=name, prefix=prefix)
        self._buffer = tf_metrics.TFDeque(buffer_size, dtype)
        self._reward_buffer = tf_metrics.TFDeque(
            2000, dtype)  # Episodes should have length less than 2k
        self.dtype = dtype

    @common.function(autograph=True)
    def call(self, trajectory):
        reward = trajectory.reward
        if len(tf.shape(reward)) == 2:
            reward = reward[0]
        else:
            assert len(tf.shape(reward)) == 1

        self._reward_buffer.extend(reward)
        if trajectory.is_last()[0] and self._reward_buffer.length > 0:
            self._update_buffer()
            self._reward_buffer.clear()
        return trajectory

    def result(self):
        return self._buffer.mean()

    @common.function
    def reset(self):
        self._reward_buffer.clear()

    @common.function
    def _update_buffer(self):
        ret = tf.reduce_sum(self._reward_buffer.data)
        score = 100 * (ret - self._ref_min_score) / \
                (self._ref_max_score - self._ref_min_score)
        self._buffer.add(score)


# @gin.configurable
# class CanonicalActionSpaceWrapper(PyEnvironmentBaseWrapper):
#     def __init__(self, env, clip=False):
#         super().__init__(env)
#         self._action_spec = env.action_spec()
#         self._clip = clip
#
#     def _step(self, action):
#         # Get scale and offset of output action spec.
#         if isinstance(self._action_spec, BoundedArraySpec):
#             # Get scale and offset of output action spec.
#             scale = self._action_spec.maximum - self._action_spec.minimum
#             offset = self._action_spec.minimum
#
#             # Maybe clip the action.
#             if self._clip:
#                 action = np.clip(action, -1.0, 1.0)
#
#             # Map action to [0, 1].
#             action = 0.5 * (action + 1.0)
#
#             # Map action to [spec.minimum, spec.maximum].
#             action *= scale
#             action += offset
#
#         return self._env.step(action)
#
#     def action_spec(self):
#         if isinstance(self._action_spec, BoundedArraySpec):
#             return self._action_spec.replace(
#                 minimum=-np.ones(self._action_spec.shape),
#                 maximum=np.ones(self._action_spec.shape))
#         else:
#             return self._action_spec


# def _convert_spec(nested_spec: types.NestedSpec) -> types.NestedSpec:
#     """Converts all bounded specs in nested spec to the canonical scale."""
#
#     def _convert_single_spec(spec: specs.Array) -> specs.Array:
#         """Converts a single spec to canonical if bounded."""
#         if isinstance(spec, specs.BoundedArray):
#             return spec.replace(
#                 minimum=-np.ones(spec.shape), maximum=np.ones(spec.shape))
#         else:
#             return spec
#
#     return tree.map_structure(_convert_single_spec, nested_spec)


# def _scale_nested_action(
#         nested_action,
#         nested_spec,
#         clip,
# ):
#     """Converts a canonical nested action back to the given nested action spec."""
#
#     def _scale_action(action: np.ndarray, spec):
#         """Converts a single canonical action back to the given action spec."""
#         if isinstance(spec, specs.BoundedArray):
#             # Get scale and offset of output action spec.
#             scale = spec.maximum - spec.minimum
#             offset = spec.minimum
#
#             # Maybe clip the action.
#             if clip:
#                 action = np.clip(action, -1.0, 1.0)
#
#             # Map action to [0, 1].
#             action = 0.5 * (action + 1.0)
#
#             # Map action to [spec.minimum, spec.maximum].
#             action *= scale
#             action += offset
#
#         return action
#
#     return tree.map_structure(_scale_action, nested_action, nested_spec)


class MetaWorldWrapper():
    def __init__(self, env):
        assert isinstance(env, SawyerXYZEnv), f"Invalid environment type: {type(env)}"
        super().__init__(env)

        unwrapped_obs_dim = self.env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=np.full(12, -np.inf),
            high=np.full(12, np.inf),
            dtype=np.float32
        )

    def _augment_obs(self, obs):
        if 'AssemblyV2' in type(self.env).__name__:
            obs[-3] += 0.13
            obs = np.concatenate([
                obs, self.env._target_pos + np.array([0.13, 0, 0])], dtype=np.float32)
        else:
            pos_hand = self.env.get_endeff_pos()
            obj_pos = self.env._get_pos_objects()

            obs = np.concatenate([pos_hand, obj_pos, self.env._target_pos, self.env._target_pos])

        return obs

    def reset(self):
        # return self.env.reset()
        obs = self.env.reset()

        return self._augment_obs(obs)

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        reward = info['success']

        # drawer-open-v2
        # if 'AssemblyV2' in type(self.env).__name__:
        #     return np.zero()
        #     obs[-3] += 0.13
        #     obs = np.concatenate([
        #         obs, self.env._target_pos + np.array([0.13, 0, 0])], dtype=np.float32)
        # else:
        #     obs = np.concatenate([obs, self.env._target_pos])

        return self._augment_obs(obs), reward, done, info


class AntMazeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.unwrapped_observation_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            low=-np.ones(58),
            high=np.ones(58),
            dtype=np.float32
        )

    def _augment_goal(self, obs):
        return np.concatenate([obs, self.target_goal, np.zeros(27)], dtype=np.float32)

    def get_dataset(self, **kwargs):
        dataset = self.env.get_dataset(**kwargs)

        N_samples = dataset['observations'].shape[0]
        dataset['observations'] = np.concatenate([
            dataset['observations'], dataset['infos/goal'],
            np.zeros([N_samples, 27])], axis=-1)

        return dataset

    def reset(self):
        obs = self.env.reset()

        return self._augment_goal(obs)

    def step(self, action):
        obs, reward, _, info = self.env.step(action)
        done = False

        return self._augment_goal(obs), reward, done, info


class OfflineAntMazeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        env.observation_space = gym.spaces.Box(
            low=np.full(58, -np.inf),
            high=np.full(58, np.inf),
            dtype=np.float32,
        )
        super(OfflineAntMazeWrapper, self).__init__(env)

    def get_dataset(self, **kwargs):
        dataset = self.env.get_dataset(**kwargs)

        N_samples = dataset['observations'].shape[0]
        dataset['observations'] = np.concatenate([
            dataset['observations'], dataset['infos/goal'],
            np.zeros([N_samples, 27])], axis=-1)

        return dataset

    def observation(self, observation):
        goal_obs = np.zeros_like(observation)
        goal_obs[:2] = self.env.target_goal
        return np.concatenate([observation, goal_obs])

    def reset(self):
        obs = self.env.reset()

        goal_obs = np.zeros_like(obs)
        goal_obs[:2] = self.env.target_goal

        return np.concatenate([obs, goal_obs])

    def step(self, action):
        obs, reward, _, info = self.env.step(action)

        goal_obs = np.zeros_like(obs)
        goal_obs[:2] = self.env.target_goal
        done = False

        return np.concatenate([obs, goal_obs]), reward, done, info


class OfflineAdroitWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.unwrapped_observation_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.full(90, -np.inf),
            high=np.full(90, np.inf),
            dtype=np.float32
        )

    def _augment_goal(self, obs):
        return np.concatenate([obs, np.zeros(45)], dtype=np.float32)

    def get_dataset(self, **kwargs):
        dataset = self.env.get_dataset(**kwargs)

        N_samples = dataset['observations'].shape[0]
        dataset['observations'] = np.concatenate([
            dataset['observations'], np.zeros([N_samples, 45])], axis=-1)

        return dataset

    def reset(self):
        obs = self.env.reset()

        return self._augment_goal(obs)

    def step(self, action):
        obs, reward, _, info = self.env.step(action)
        done = False

        return self._augment_goal(obs), reward, done, info
