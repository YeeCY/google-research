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

"""C-learning.

Implements the off-policy goal-conditioned C-learning algorithm from
"C-Learning: Learning to Achieve Goals via Recursive Classification" by
Eysenbach et al (2020): https://arxiv.org/abs/2011.08909
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from typing import Callable, Optional, Text

import gin
from six.moves import zip
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.policies import actor_policy
from tf_agents.policies import tf_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.utils import object_identity

EPSILON = 1e-7
CLearningLossInfo = collections.namedtuple(
    'LossInfo', ('critic_loss', 'actor_loss'))


@gin.configurable
class CLearningAgent(tf_agent.TFAgent):
    """A C-learning Agent."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 critic_network,
                 actor_network,
                 actor_optimizer,
                 critic_optimizer,
                 obs_dim,
                 actor_loss_weight=1.0,
                 critic_loss_weight=0.5,
                 actor_policy_ctor=actor_policy.ActorPolicy,
                 critic_network_2=None,
                 target_critic_network=None,
                 target_critic_network_2=None,
                 target_update_tau=1.0,
                 target_update_period=1,
                 td_errors_loss_fn=tf.math.squared_difference,
                 gamma=1.0,
                 gradient_clipping=None,
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 train_step_counter=None,
                 name=None):
        """Creates a C-learning Agent.

        By default, the environment observation contains the current state and goal
        state. By setting the obs_to_goal gin config in c_learning_utils, the user
        can specify that the agent should only look at certain subsets of the goal
        state.

        Args:
          time_step_spec: A `TimeStep` spec of the expected time_steps.
          action_spec: A nest of BoundedTensorSpec representing the actions.
          critic_network: A function critic_network((observations, actions)) that
            returns the q_values for each observation and action.
          actor_network: A function actor_network(observation, action_spec) that
            returns action distribution.
          actor_optimizer: The optimizer to use for the actor network.
          critic_optimizer: The default optimizer to use for the critic network.
          actor_loss_weight: The weight on actor loss.
          critic_loss_weight: The weight on critic loss.
          actor_policy_ctor: The policy class to use.
          critic_network_2: (Optional.)  A `tf_agents.network.Network` to be used as
            the second critic network during Q learning.  The weights from
            `critic_network` are copied if this is not provided.
          target_critic_network: (Optional.)  A `tf_agents.network.Network` to be
            used as the target critic network during Q learning. Every
            `target_update_period` train steps, the weights from `critic_network`
            are copied (possibly withsmoothing via `target_update_tau`) to `
            target_critic_network`.  If `target_critic_network` is not provided, it
            is created by making a copy of `critic_network`, which initializes a new
            network with the same structure and its own layers and weights.
            Performing a `Network.copy` does not work when the network instance
            already has trainable parameters (e.g., has already been built, or when
            the network is sharing layers with another).  In these cases, it is up
            to you to build a copy having weights that are not shared with the
            original `critic_network`, so that this can be used as a target network.
            If you provide a `target_critic_network` that shares any weights with
            `critic_network`, a warning will be logged but no exception is thrown.
          target_critic_network_2: (Optional.) Similar network as
            target_critic_network but for the critic_network_2. See documentation
            for target_critic_network. Will only be used if 'critic_network_2' is
            also specified.
          target_update_tau: Factor for soft update of the target networks.
          target_update_period: Period for soft update of the target networks.
          td_errors_loss_fn:  A function for computing the elementwise TD errors
            loss.
          gamma: A discount factor for future rewards.
          gradient_clipping: Norm length to clip gradients.
          debug_summaries: A bool to gather debug summaries.
          summarize_grads_and_vars: If True, gradient and network variable summaries
            will be written during training.
          train_step_counter: An optional counter to increment every time the train
            op is run.  Defaults to the global_step.
          name: The name of this agent. All variables in this module will fall under
            that name. Defaults to the class name.
        """
        tf.Module.__init__(self, name=name)

        self._check_action_spec(action_spec)

        self._critic_network_1 = critic_network
        self._critic_network_1.create_variables()
        if target_critic_network:
            target_critic_network.create_variables()
        self._target_critic_network_1 = (
            common.maybe_copy_target_network_with_checks(self._critic_network_1,
                                                         target_critic_network,
                                                         'TargetCriticNetwork1'))

        if critic_network_2 is not None:
            self._critic_network_2 = critic_network_2
        else:
            self._critic_network_2 = critic_network.copy(name='CriticNetwork2')
            # Do not use target_critic_network_2 if critic_network_2 is None.
            target_critic_network_2 = None
        self._critic_network_2.create_variables()
        if target_critic_network_2:
            target_critic_network_2.create_variables()
        self._target_critic_network_2 = (
            common.maybe_copy_target_network_with_checks(self._critic_network_2,
                                                         target_critic_network_2,
                                                         'TargetCriticNetwork2'))

        if actor_network:
            actor_network.create_variables()
        self._actor_network = actor_network

        policy = actor_policy_ctor(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=self._actor_network,
            training=False)

        self._train_policy = actor_policy_ctor(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=self._actor_network,
            training=True)

        self._target_update_tau = target_update_tau
        self._target_update_period = target_update_period
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._actor_loss_weight = actor_loss_weight
        self._critic_loss_weight = critic_loss_weight
        self._td_errors_loss_fn = td_errors_loss_fn
        self._gamma = gamma
        self._gradient_clipping = gradient_clipping
        self._debug_summaries = debug_summaries
        self._summarize_grads_and_vars = summarize_grads_and_vars
        self._update_target = self._get_target_updater(
            tau=self._target_update_tau, period=self._target_update_period)

        train_sequence_length = 2 if not critic_network.state_spec else None

        super(CLearningAgent, self).__init__(
            time_step_spec,
            action_spec,
            policy=policy,
            collect_policy=policy,
            train_sequence_length=train_sequence_length,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter)

    def _check_action_spec(self, action_spec):
        flat_action_spec = tf.nest.flatten(action_spec)
        for spec in flat_action_spec:
            if spec.dtype.is_integer:
                raise NotImplementedError(
                    'CLearningAgent does not currently support discrete actions. '
                    'Action spec: {}'.format(action_spec))

    def _initialize(self):
        """Returns an op to initialize the agent.

        Copies weights from the Q networks to the target Q network.
        """
        common.soft_variables_update(
            self._critic_network_1.variables,
            self._target_critic_network_1.variables,
            tau=1.0)
        common.soft_variables_update(
            self._critic_network_2.variables,
            self._target_critic_network_2.variables,
            tau=1.0)

    def _train(self, experience, weights):
        """Returns a train op to update the agent's networks.

        This method trains with the provided batched experience.

        Args:
          experience: A time-stacked trajectory object.
          weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.

        Returns:
          A train_op.

        Raises:
          ValueError: If optimizers are None and no default value was provided to
            the constructor.
        """
        squeeze_time_dim = not self._critic_network_1.state_spec
        time_steps, policy_steps, next_time_steps = (
            trajectory.experience_to_transitions(experience, squeeze_time_dim))
        actions = policy_steps.action

        trainable_critic_variables = list(object_identity.ObjectIdentitySet(
            self._critic_network_1.trainable_variables +
            self._critic_network_2.trainable_variables))

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert trainable_critic_variables, ('No trainable critic variables to '
                                                'optimize.')
            tape.watch(trainable_critic_variables)
            critic_loss = self._critic_loss_weight * self.critic_loss(
                time_steps,
                actions,
                next_time_steps,
                td_errors_loss_fn=self._td_errors_loss_fn,
                gamma=self._gamma,
                weights=weights,
                training=True)

        tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
        critic_grads = tape.gradient(critic_loss, trainable_critic_variables)
        self._apply_gradients(critic_grads, trainable_critic_variables,
                              self._critic_optimizer)

        trainable_actor_variables = self._actor_network.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert trainable_actor_variables, ('No trainable actor variables to '
                                               'optimize.')
            tape.watch(trainable_actor_variables)
            actor_loss = self._actor_loss_weight * self.actor_loss(
                time_steps, actions, weights=weights)
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
        actor_grads = tape.gradient(actor_loss, trainable_actor_variables)
        self._apply_gradients(actor_grads, trainable_actor_variables,
                              self._actor_optimizer)

        with tf.name_scope('Losses'):
            tf.compat.v2.summary.scalar(
                name='critic_loss', data=critic_loss, step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='actor_loss', data=actor_loss, step=self.train_step_counter)

        self.train_step_counter.assign_add(1)
        self._update_target()

        total_loss = critic_loss + actor_loss

        extra = CLearningLossInfo(
            critic_loss=critic_loss, actor_loss=actor_loss)

        return tf_agent.LossInfo(loss=total_loss, extra=extra)

    def _apply_gradients(self, gradients, variables, optimizer):
        # list(...) is required for Python3.
        grads_and_vars = list(zip(gradients, variables))
        if self._gradient_clipping is not None:
            grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                             self._gradient_clipping)

        if self._summarize_grads_and_vars:
            eager_utils.add_variables_summaries(grads_and_vars,
                                                self.train_step_counter)
            eager_utils.add_gradients_summaries(grads_and_vars,
                                                self.train_step_counter)

        optimizer.apply_gradients(grads_and_vars)

    def _get_target_updater(self, tau=1.0, period=1):
        """Performs a soft update of the target network parameters.

        For each weight w_s in the original network, and its corresponding
        weight w_t in the target network, a soft update is:
        w_t = (1- tau) x w_t + tau x ws

        Args:
          tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
          period: Step interval at which the target network is updated.

        Returns:
          A callable that performs a soft update of the target network parameters.
        """
        with tf.name_scope('update_target'):
            def update():
                """Update target network."""
                critic_update_1 = common.soft_variables_update(
                    self._critic_network_1.variables,
                    self._target_critic_network_1.variables,
                    tau,
                    tau_non_trainable=1.0)

                critic_2_update_vars = common.deduped_network_variables(
                    self._critic_network_2, self._critic_network_1)

                target_critic_2_update_vars = common.deduped_network_variables(
                    self._target_critic_network_2, self._target_critic_network_1)

                critic_update_2 = common.soft_variables_update(
                    critic_2_update_vars,
                    target_critic_2_update_vars,
                    tau,
                    tau_non_trainable=1.0)

                return tf.group(critic_update_1, critic_update_2)

            return common.Periodically(update, period, 'update_targets')

    def _actions_log_probs_and_entropy(self, time_steps, future_goal=False):
        """Get actions and corresponding log probabilities from policy."""
        # Get raw action distribution from policy, and initialize bijectors list.
        # batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
        batch_size = time_steps.observation.shape[0]
        policy_state = self._train_policy.get_initial_state(batch_size)
        if future_goal:
            # TODO (chongyiz): check this.
            new_time_steps = ts.TimeStep(
                step_type=time_steps.step_type,
                reward=time_steps.reward,
                discount=time_steps.discount,
                observation=tf.concat([
                    time_steps.observation[:, :self._obs_dim],
                    time_steps.observation[
                        :, self._obs_dim + 3 * self._goal_dim:self._obs_dim + 4 * self._goal_dim]
                ], axis=-1)
            )
        else:
            new_time_steps = ts.TimeStep(
                step_type=time_steps.step_type,
                reward=time_steps.reward,
                discount=time_steps.discount,
                observation=time_steps.observation[:, :self._obs_dim + self._goal_dim])
        action_distribution = self._train_policy.distribution(
            new_time_steps, policy_state=policy_state).action

        # Sample actions and log_pis from transformed distribution.
        actions = tf.nest.map_structure(lambda d: d.sample(), action_distribution)
        log_pi = common.log_probability(action_distribution, actions,
                                        self.action_spec)
        entropy = common.entropy(action_distribution.input_distribution, self.action_spec)

        return actions, log_pi, entropy

    def _log_probs(self, time_steps, actions, future_goal=False):
        """Get log probabilities of actions from policy."""
        # Get raw action distribution from policy, and initialize bijectors list.
        # batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
        batch_size = time_steps.observation.shape[0]
        policy_state = self._train_policy.get_initial_state(batch_size)
        if future_goal:
            new_time_steps = ts.TimeStep(
                step_type=time_steps.step_type,
                reward=time_steps.reward,
                discount=time_steps.discount,
                observation=tf.concat([
                    time_steps.observation[:, :self._obs_dim],
                    time_steps.observation[
                    :, self._obs_dim + 3 * self._goal_dim:self._obs_dim + 4 * self._goal_dim]
                ], axis=-1)
            )
        else:
            new_time_steps = ts.TimeStep(
                step_type=time_steps.step_type,
                reward=time_steps.reward,
                discount=time_steps.discount,
                observation=time_steps.observation[:, :self._obs_dim + self._goal_dim])
        action_distribution = self._train_policy.distribution(
            new_time_steps, policy_state=policy_state).action

        # Get log_pis from transformed distribution.
        # actions = tf.nest.map_structure(lambda d: d.sample(), action_distribution)
        log_pi = common.log_probability(action_distribution, actions,
                                        self.action_spec)

        return log_pi

    @gin.configurable
    def critic_loss(self,
                    time_steps,
                    actions,
                    next_time_steps,
                    td_errors_loss_fn,
                    gamma=1.0,
                    weights=None,
                    training=False,
                    w_clipping=20.0,
                    self_normalized=False,
                    lambda_fix=False,
                    ):
        """Computes the critic loss for C-learning training.

        Args:
          time_steps: A batch of timesteps.
          actions: A batch of actions.
          next_time_steps: A batch of next timesteps.
          td_errors_loss_fn: A function(td_targets, predictions) to compute
            elementwise (per-batch-entry) loss.
          gamma: Discount for future rewards.
          weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.
          training: Whether this loss is being used for training.
          w_clipping: Maximum value used for clipping the weights. Use -1 to do no
            clipping; use None to use the recommended value of 1 / (1 - gamma).
          self_normalized: Whether to normalize the weights to the average is 1.
            Empirically this usually hurts performance.
          lambda_fix: Whether to include the adjustment when using future positives.
            Empirically this has little effect.

        Returns:
          critic_loss: A scalar critic loss.
        """
        del weights
        if w_clipping is None:
            w_clipping = 1 / (1 - gamma)
        rfp = gin.query_parameter('goal_fn.relabel_future_prob')
        rnp = gin.query_parameter('goal_fn.relabel_next_prob')
        assert rfp + rnp == 0.5
        with tf.name_scope('critic_loss'):
            # nest_utils.assert_same_structure(actions, self.action_spec)
            # nest_utils.assert_same_structure(time_steps, self.time_step_spec)
            # nest_utils.assert_same_structure(next_time_steps, self.time_step_spec)

            next_actions, _ = self._actions_and_log_probs(next_time_steps)
            target_input = (next_time_steps.observation, next_actions)
            target_q_values1, _ = self._target_critic_network_1(
                target_input, next_time_steps.step_type, training=False)
            target_q_values2, _ = self._target_critic_network_2(
                target_input, next_time_steps.step_type, training=False)
            target_q_values = tf.minimum(target_q_values1, target_q_values2)
            # target_q_values1, _ = self._critic_network_1(
            #     target_input, next_time_steps.step_type, training=False)
            # target_q_values2, _ = self._critic_network_2(
            #     target_input, next_time_steps.step_type, training=False)
            # target_q_values = tf.minimum(target_q_values1, target_q_values2)

            w = tf.stop_gradient(target_q_values / (1 - target_q_values))
            if w_clipping >= 0:
                w = tf.clip_by_value(w, 0, w_clipping)
            else:
                # w = tf.clip_by_value(w, 0, 1e30)
                w = tf.clip_by_value(w, 0, 1e25)
                # w = tf.clip_by_value(w, 0, tf.float32.max)
            tf.debugging.assert_all_finite(w, 'Not all elements of w are finite')
            if self_normalized:
                w = w / tf.reduce_mean(w)

            batch_size = nest_utils.get_outer_shape(time_steps,
                                                    self._time_step_spec)[0]
            half_batch = batch_size // 2
            float_batch_size = tf.cast(batch_size, float)
            num_next = tf.cast(tf.round(float_batch_size * rnp), tf.int32)
            num_future = tf.cast(tf.round(float_batch_size * rfp), tf.int32)
            if lambda_fix:
                lambda_coef = 2 * rnp
                weights = tf.concat([tf.fill((num_next,), (1 - gamma)),
                                     tf.fill((num_future,), 1.0),
                                     (1 + lambda_coef * gamma * w)[half_batch:]],
                                    axis=0)
            else:
                weights = tf.concat([tf.fill((num_next,), (1 - gamma)),
                                     tf.fill((num_future,), 1.0),
                                     (1 + gamma * w)[half_batch:]],
                                    axis=0)

            # Note that we assume that episodes never terminate. If they do, then
            # we need to include next_time_steps.discount in the (negative) TD target.
            # We exclude the termination here so that we can use termination to
            # indicate task success during evaluation. In the evaluation setting,
            # task success depends on the task, but we don't want the termination
            # here to depend on the task. Hence, we ignored it.
            if lambda_fix:
                lambda_coef = 2 * rnp
                y = lambda_coef * gamma * w / (1 + lambda_coef * gamma * w)
            else:
                y = gamma * w / (1 + gamma * w)
            td_targets = tf.stop_gradient(next_time_steps.reward +
                                          (1 - next_time_steps.reward) * y)
            if rfp > 0:
                td_targets = tf.concat([tf.ones(half_batch),
                                        td_targets[half_batch:]], axis=0)

            observation = time_steps.observation
            pred_input = (observation, actions)
            pred_td_targets1, _ = self._critic_network_1(
                pred_input, time_steps.step_type, training=training)
            pred_td_targets2, _ = self._critic_network_2(
                pred_input, time_steps.step_type, training=training)

            # (chongyiz): cross-entropy implementation of the classifier loss
            # ce_critic_loss1 = td_errors_loss_fn(td_targets, pred_td_targets1)
            # ce_critic_loss2 = td_errors_loss_fn(td_targets, pred_td_targets2)
            # critic_loss = ce_critic_loss1 + ce_critic_loss2

            # (chongyiz): three term implementation of the classifier loss,
            # this implementation is similar to https://github.com/keras-team/keras/blob/v2.9.0/keras/losses.py#L496-L596,
            # which will be more numerical stable.
            pred_td_targets1 = tf.clip_by_value(pred_td_targets1, EPSILON, 1. - EPSILON)
            pred_td_targets2 = tf.clip_by_value(pred_td_targets2, EPSILON, 1. - EPSILON)
            # [1, ..., 1, 0, ..., 0]
            first_half_batch_mask = next_time_steps.reward
            first_half_batch_mask = tf.concat([tf.ones(half_batch),
                                               first_half_batch_mask[half_batch:]], axis=0)
            # [0, ..., 0, 1, ..., 1]
            second_half_batch_mask = 1 - next_time_steps.reward
            second_half_batch_mask = tf.concat([tf.zeros(half_batch),
                                                second_half_batch_mask[half_batch:]], axis=0)
            critic_loss1 = \
                -first_half_batch_mask * tf.math.log(pred_td_targets1 + EPSILON) - \
                second_half_batch_mask * tf.stop_gradient(1 - y) * tf.math.log(1 - pred_td_targets1 + EPSILON) - \
                second_half_batch_mask * tf.stop_gradient(y) * tf.math.log(pred_td_targets1 + EPSILON)
            # tf.debugging.assert_near(ce_critic_loss1, critic_loss1,
            #                          rtol=tf.constant(EPSILON), atol=tf.constant(EPSILON))
            critic_loss2 = \
                -first_half_batch_mask * tf.math.log(pred_td_targets2 + EPSILON) - \
                second_half_batch_mask * tf.stop_gradient(1 - y) * tf.math.log(1 - pred_td_targets2 + EPSILON) - \
                second_half_batch_mask * tf.stop_gradient(y) * tf.math.log(pred_td_targets2 + EPSILON)
            # tf.debugging.assert_near(ce_critic_loss2, critic_loss2,
            #                          rtol=tf.constant(EPSILON), atol=tf.constant(EPSILON))

            critic_loss = critic_loss1 + critic_loss2

            if critic_loss.shape.rank > 1:
                # Sum over the time dimension.
                critic_loss = tf.reduce_sum(
                    critic_loss, axis=range(1, critic_loss.shape.rank))

            agg_loss = common.aggregate_losses(
                per_example_loss=critic_loss,
                sample_weight=weights,
                regularization_loss=(self._critic_network_1.losses +
                                     self._critic_network_2.losses))
            critic_loss = agg_loss.total_loss
            self._critic_loss_debug_summaries(td_targets, pred_td_targets1,
                                              pred_td_targets2, weights)

            # log classifier prediction
            tf.compat.v2.summary.scalar(
                name='C1 / (1 - C1)',
                data=tf.reduce_mean(pred_td_targets1 / (1 - pred_td_targets1)),
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='C2 / (1 - C2)',
                data=tf.reduce_mean(pred_td_targets2 / (1 - pred_td_targets2)),
                step=self.train_step_counter)
            # tf.compat.v2.summary.scalar(
            #     name='next C1 / (1 - C1)',
            #     data=tf.reduce_mean(pred_td_targets1[:num_next] / (1 - pred_td_targets1[:num_next])),
            #     step=self.train_step_counter)
            # tf.compat.v2.summary.scalar(
            #     name='next C2 / (1 - C2)',
            #     data=tf.reduce_mean(pred_td_targets2[:num_next] / (1 - pred_td_targets2[:num_next])),
            #     step=self.train_step_counter)
            # tf.compat.v2.summary.scalar(
            #     name='future C1 / (1 - C1)',
            #     data=tf.reduce_mean(
            #         pred_td_targets1[num_next:num_next + num_future] /
            #         (1 - pred_td_targets1[num_next:num_next + num_future])),
            #     step=self.train_step_counter)
            # tf.compat.v2.summary.scalar(
            #     name='future C2 / (1 - C2)',
            #     data=tf.reduce_mean(
            #         pred_td_targets2[num_next:num_next + num_future] /
            #         (1 - pred_td_targets2[num_next:num_next + num_future])),
            #     step=self.train_step_counter)
            # tf.compat.v2.summary.scalar(
            #     name='random C1 / (1 - C1)',
            #     data=tf.reduce_mean(pred_td_targets1[half_batch:] / (1 - pred_td_targets1[half_batch:])),
            #     step=self.train_step_counter)
            # tf.compat.v2.summary.scalar(
            #     name='random C2 / (1 - C2)',
            #     data=tf.reduce_mean(pred_td_targets2[half_batch:] / (1 - pred_td_targets2[half_batch:])),
            #     step=self.train_step_counter)
            # tf.compat.v2.summary.scalar(
            #     name='C1 / (1 - C1) is NAN',
            #     data=tf.math.is_nan(tf.reduce_mean(pred_td_targets1 / (1 - pred_td_targets1))),
            #     step=self.train_step_counter)
            # tf.compat.v2.summary.scalar(
            #     name='C2 / (1 - C2) is NAN',
            #     data=tf.math.is_nan(tf.reduce_mean(pred_td_targets2 / (1 - pred_td_targets2))),
            #     step=self.train_step_counter)
            # tf.compat.v2.summary.scalar(
            #     name='C1 / (1 - C1) is inf',
            #     data=tf.math.is_inf(tf.reduce_mean(pred_td_targets1 / (1 - pred_td_targets1))),
            #     step=self.train_step_counter)
            # tf.compat.v2.summary.scalar(
            #     name='C2 / (1 - C2) is inf',
            #     data=tf.math.is_inf(tf.reduce_mean(pred_td_targets2 / (1 - pred_td_targets2))),
            #     step=self.train_step_counter)

            return critic_loss

    @gin.configurable
    def actor_loss(self,
                   time_steps,
                   actions,
                   weights=None,
                   ce_loss=False,
                   mse_bc_loss=False,
                   mle_bc_loss=False,
                   bc_lambda=0.25,
                   aw_loss=False,
                   aw_lambda=0.5):
        """Computes the actor_loss for C-learning training.

        Args:
          time_steps: A batch of timesteps.
          actions: A batch of actions.
          weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.
          ce_loss: (bool) Whether to update the actor using the cross entropy loss,
            which corresponds to using the log C-value. The default actor loss
            differs by not including the log. Empirically we observed no difference.

        Returns:
          actor_loss: A scalar actor loss.
        """
        with tf.name_scope('actor_loss'):
            # nest_utils.assert_same_structure(time_steps, self.time_step_spec)

            # try future goal for both setting b and c
            sampled_actions, sampled_log_pi, sampled_entropy = \
                self._actions_log_probs_and_entropy(time_steps, future_goal=True)
            log_pi = self._log_probs(time_steps, actions, future_goal=True)

            sampled_target_input = (
                time_steps.observation[:, :self._obs_dim + self._goal_dim],
                sampled_actions
            )
            sampled_q_values1, _ = self._critic_network_1(
                sampled_target_input, time_steps.step_type, training=False)
            sampled_q_values2, _ = self._critic_network_2(
                sampled_target_input, time_steps.step_type, training=False)
            sampled_q_values = tf.minimum(sampled_q_values1, sampled_q_values2)
            if ce_loss:
                actor_loss = tf.keras.losses.binary_crossentropy(
                    tf.ones_like(sampled_q_values), sampled_q_values)
            else:
                actor_loss = -1.0 * sampled_q_values

            if mse_bc_loss:
                actor_loss += bc_lambda * tf.losses.mse(actions, sampled_actions)

            if mle_bc_loss:
                actor_loss -= bc_lambda * log_pi

            if aw_loss:
                target_input = (
                    time_steps.observation[:, :self._obs_dim + self._goal_dim],
                    actions
                )
                q_values1, _ = self._critic_network_1(
                    target_input, time_steps.step_type, training=False)
                q_values2, _ = self._critic_network_2(
                    target_input, time_steps.step_type, training=False)
                q_values = tf.minimum(q_values1, q_values2)

                actor_loss = -tf.reduce_mean(
                    log_pi *
                    tf.math.exp((tf.math.log(q_values) - tf.math.log(1 - q_values)) / aw_lambda)
                )

            if actor_loss.shape.rank > 1:
                # Sum over the time dimension.
                actor_loss = tf.reduce_sum(
                    actor_loss, axis=range(1, actor_loss.shape.rank))
            reg_loss = self._actor_network.losses if self._actor_network else None
            agg_loss = common.aggregate_losses(
                per_example_loss=actor_loss,
                sample_weight=weights,
                regularization_loss=reg_loss)
            actor_loss = agg_loss.total_loss
            self._actor_loss_debug_summaries(actor_loss, actions, sampled_log_pi,
                                             sampled_q_values, time_steps)

            # log entropy
            tf.compat.v2.summary.scalar(
                name='policy_entropy',
                data=tf.reduce_mean(sampled_entropy),
                step=self.train_step_counter)

            return actor_loss

    def _critic_loss_debug_summaries(self, td_targets, pred_td_targets1,
                                     pred_td_targets2, weights):
        if self._debug_summaries:
            td_errors1 = td_targets - pred_td_targets1
            td_errors2 = td_targets - pred_td_targets2
            td_errors = tf.concat([td_errors1, td_errors2], axis=0)
            common.generate_tensor_summaries('td_errors', td_errors,
                                             self.train_step_counter)
            common.generate_tensor_summaries('td_targets', td_targets,
                                             self.train_step_counter)
            common.generate_tensor_summaries('pred_td_targets1', pred_td_targets1,
                                             self.train_step_counter)
            common.generate_tensor_summaries('pred_td_targets2', pred_td_targets2,
                                             self.train_step_counter)
            common.generate_tensor_summaries('weights', weights,
                                             self.train_step_counter)

    def _actor_loss_debug_summaries(self, actor_loss, actions, log_pi,
                                    target_q_values, time_steps):
        if self._debug_summaries:
            common.generate_tensor_summaries('actor_loss', actor_loss,
                                             self.train_step_counter)
            common.generate_tensor_summaries('actions', actions,
                                             self.train_step_counter)
            common.generate_tensor_summaries('log_pi', log_pi,
                                             self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='entropy_avg',
                data=-tf.reduce_mean(input_tensor=log_pi),
                step=self.train_step_counter)
            common.generate_tensor_summaries('target_q_values', target_q_values,
                                             self.train_step_counter)
            batch_size = nest_utils.get_outer_shape(time_steps,
                                                    self._time_step_spec)[0]
            policy_state = self._train_policy.get_initial_state(batch_size)
            action_distribution = self._train_policy.distribution(
                time_steps, policy_state).action
            if isinstance(action_distribution, tfp.distributions.Normal):
                common.generate_tensor_summaries('act_mean', action_distribution.loc,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('act_stddev',
                                                 action_distribution.scale,
                                                 self.train_step_counter)
            elif isinstance(action_distribution, tfp.distributions.Categorical):
                common.generate_tensor_summaries('act_mode', action_distribution.mode(),
                                                 self.train_step_counter)
            common.generate_tensor_summaries('entropy_action',
                                             action_distribution.entropy(),
                                             self.train_step_counter)
