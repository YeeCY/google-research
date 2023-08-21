# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Contrastive RL networks definition."""
import dataclasses
from typing import Optional, Callable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from acme.jax import networks as networks_lib
from acme.jax import utils

import tensorflow_probability
tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions


@dataclasses.dataclass
class ContrastiveNetworks:
    """Network and pure functions for the Contrastive RL agent."""
    policy_network: networks_lib.FeedForwardNetwork
    q_network: networks_lib.FeedForwardNetwork
    log_prob: networks_lib.LogProbFn
    repr_fn: Callable[Ellipsis, networks_lib.NetworkOutput]
    sample: networks_lib.SampleFn
    sample_eval: Optional[networks_lib.SampleFn] = None


class RelaxedOnehotCategoricalHead(networks_lib.CategoricalHead):
    def __call__(self, inputs: jnp.ndarray) -> tfd.Distribution:
        logits = self._linear(inputs)
        if not isinstance(self._logit_shape, int):
            logits = hk.Reshape(self._logit_shape)(logits)
        return tfd.RelaxedOneHotCategorical(temperature=1.0, logits=logits)


def apply_policy_and_sample(
        networks,
        eval_mode=False):
    """Returns a function that computes actions."""
    sample_fn = networks.sample if not eval_mode else networks.sample_eval
    if not sample_fn:
        raise ValueError('sample function is not provided')

    def apply_and_sample(params, key, obs):
        return sample_fn(networks.policy_network.apply(params, obs), key)

    return apply_and_sample


def make_networks(
        spec,
        obs_dim,
        repr_dim=64,
        repr_norm=False,
        repr_norm_temp=True,
        hidden_layer_sizes=(256, 256),
        actor_min_std=1e-6,
        twin_q=False,
        use_image_obs=False):
    """Creates networks used by the agent."""

    num_dimensions = int(np.prod(spec.actions.shape))
    TORSO = networks_lib.AtariTorso  # pylint: disable=invalid-name

    def _unflatten_obs(obs):
        state = jnp.reshape(obs[:, :obs_dim], (-1, 64, 64, 3)) / 255.0
        goal = jnp.reshape(obs[:, obs_dim:], (-1, 64, 64, 3)) / 255.0
        return state, goal

    def _unflatten_img(img):
        img = jnp.reshape(img, (-1, 64, 64, 3)) / 255.0
        return img

    # def _repr_fn(obs, action, hidden=None):
    #     # The optional input hidden is the image representations. We include this
    #     # as an input for the second Q value when twin_q = True, so that the two Q
    #     # values use the same underlying image representation.
    #     if hidden is None:
    #         if use_image_obs:
    #             state, goal = _unflatten_obs(obs)
    #             img_encoder = TORSO()
    #             state = img_encoder(state)
    #             goal = img_encoder(goal)
    #         else:
    #             state = obs[:, :obs_dim]
    #             goal = obs[:, obs_dim:]
    #     else:
    #         state, goal = hidden
    #
    #     sa_encoder = hk.nets.MLP(
    #         list(hidden_layer_sizes) + [repr_dim],
    #         w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
    #         activation=jax.nn.relu,
    #         name='sa_encoder')
    #     sa_repr = sa_encoder(jnp.concatenate([state, action], axis=-1))
    #
    #     g_encoder = hk.nets.MLP(
    #         list(hidden_layer_sizes) + [repr_dim],
    #         w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
    #         activation=jax.nn.relu,
    #         name='g_encoder')
    #     g_repr = g_encoder(goal)
    #
    #     if repr_norm:
    #         sa_repr = sa_repr / jnp.linalg.norm(sa_repr, axis=1, keepdims=True)
    #         g_repr = g_repr / jnp.linalg.norm(g_repr, axis=1, keepdims=True)
    #
    #         if repr_norm_temp:
    #             log_scale = hk.get_parameter('repr_log_scale', [], dtype=sa_repr.dtype,
    #                                          init=jnp.zeros)
    #             sa_repr = sa_repr / jnp.exp(log_scale)
    #     return sa_repr, g_repr, (state, goal)

    def _repr_fn(obs, goal, future_obs, hidden=None):
        # The optional input hidden is the image representations. We include this
        # as an input for the second Q value when twin_q = True, so that the two Q
        # values use the same underlying image representation.
        if hidden is None:
            if use_image_obs:
                obs = _unflatten_img(obs)
                goal = _unflatten_img(goal)
                future_obs = _unflatten_img(future_obs)
                img_encoder = TORSO()
                state = img_encoder(obs)
                goal = img_encoder(goal)
                future_state = img_encoder(future_obs)
            else:
                state = obs
                goal = goal
                future_state = future_obs
        else:
            # this line of code should match the return values!
            state, goal, future_state = hidden

        # sa_encoder = hk.nets.MLP(
        #     list(hidden_layer_sizes) + [repr_dim * num_dimensions],
        #     w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
        #     activation=jax.nn.relu,
        #     name='sa_encoder')
        # sa_repr = sa_encoder(state).reshape([-1, repr_dim, num_dimensions])
        #
        # g_encoder = hk.nets.MLP(
        #     list(hidden_layer_sizes) + [repr_dim * repr_dim * num_dimensions],
        #     w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
        #     activation=jax.nn.relu,
        #     name='g_encoder')
        # g_repr = g_encoder(goal).reshape([-1, repr_dim, repr_dim, num_dimensions])
        #
        # fs_encoder = hk.nets.MLP(
        #     list(hidden_layer_sizes) + [repr_dim * num_dimensions],
        #     w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
        #     activation=jax.nn.relu,
        #     name='fs_encoder')
        # fs_repr = fs_encoder(future_state).reshape([-1, repr_dim, num_dimensions])

        logit_encoder = hk.nets.MLP(
            list(hidden_layer_sizes) + [num_dimensions],
            # w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
            activation=jax.nn.relu,
            name='logit_encoder')
        logits = logit_encoder(jnp.concatenate([state, goal, future_state], axis=-1))

        if repr_norm:
            sag_repr = sag_repr / jnp.linalg.norm(sag_repr, axis=1, keepdims=True)
            fs_repr = fs_repr / jnp.linalg.norm(fs_repr, axis=1, keepdims=True)

            if repr_norm_temp:
                log_scale = hk.get_parameter('repr_log_scale', [], dtype=sag_repr.dtype,
                                             init=jnp.zeros)
                sag_repr = sag_repr / jnp.exp(log_scale)

        return logits, (state, goal, future_state)

    def _combine_repr(sa_repr, g_repr, fs_repr):
        # gfs_repr = jnp.einsum('ijkl,ikl->ijl', g_repr, fs_repr)
        # we should use the goal representation together with the sa_repr
        sag_repr = jnp.einsum('ijkl,ikl->ijl', g_repr, sa_repr)

        # return jax.numpy.einsum('ikl,jkl->ijl', sa_repr, gfs_repr)
        return jax.numpy.einsum('ikl,jkl->ijl', sag_repr, fs_repr)

    # def _critic_fn(obs, action):
    #     sa_repr, g_repr, hidden = _repr_fn(obs, action)
    #     outer = _combine_repr(sa_repr, g_repr)
    #     if twin_q:
    #         sa_repr2, g_repr2, _ = _repr_fn(obs, action, hidden=hidden)
    #         outer2 = _combine_repr(sa_repr2, g_repr2)
    #         # outer.shape = [batch_size, batch_size, 2]
    #         outer = jnp.stack([outer, outer2], axis=-1)
    #     return outer

    def _critic_fn(obs, goal, future_obs):
        logits, hidden = _repr_fn(obs, goal, future_obs)
        # outer = _combine_repr(sa_repr, g_repr, fs_repr)
        if twin_q:
            logits2, _ = _repr_fn(obs, goal, future_obs, hidden=hidden)
            # outer2 = _combine_repr(sa_repr2, g_repr2, fs_repr2)
            # outer.shape = [batch_size, batch_size, 2]
            logits = jnp.stack([logits, logits2], axis=-1)
        else:
            logits = logits[:, :, None]
        return logits

    def _actor_fn(obs):
        if use_image_obs:
            state, goal = _unflatten_obs(obs)
            obs = jnp.concatenate([state, goal], axis=-1)
            obs = TORSO(obs)
        # network = hk.Sequential([
        #     hk.nets.MLP(
        #         list(hidden_layer_sizes),
        #         w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
        #         activation=jax.nn.relu,
        #         activate_final=True),
        #     networks_lib.NormalTanhDistribution(num_dimensions,
        #                                         min_scale=actor_min_std),
        # ])
        network = hk.Sequential([
            hk.nets.MLP(
                list(hidden_layer_sizes) + [num_dimensions],
                # w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
                activation=jax.nn.relu,
                activate_final=False),
            # networks_lib.NormalTanhDistribution(num_dimensions,
            #                                     min_scale=actor_min_std),
            # jax.nn.softmax,
            # networks_lib.DiscreteValued
            # RelaxedOnehotCategoricalHead(num_dimensions)
            jax.nn.softmax,
        ])

        return network(obs)

    policy = hk.without_apply_rng(hk.transform(_actor_fn))
    critic = hk.without_apply_rng(hk.transform(_critic_fn))
    repr_fn = hk.without_apply_rng(hk.transform(_repr_fn))

    # Create dummy observations and actions to create network parameters.
    # dummy_action = utils.zeros_like(spec.actions)
    # dummy_obs = utils.zeros_like(spec.observations)
    # dummy_action = utils.add_batch_dim(dummy_action)
    # dummy_obs = utils.add_batch_dim(dummy_obs)

    dummy_action = utils.zeros_like(spec.actions)
    dummy_obs = utils.zeros_like(spec.observations)[:obs_dim]
    dummy_future_obs = utils.zeros_like(spec.observations)[:obs_dim]
    dummy_goal = utils.zeros_like(spec.observations)[obs_dim:]
    dummy_obs_and_goal = utils.zeros_like(spec.observations)
    dummy_action = utils.add_batch_dim(dummy_action)
    dummy_obs = utils.add_batch_dim(dummy_obs)
    dummy_future_obs = utils.add_batch_dim(dummy_future_obs)
    dummy_goal = utils.add_batch_dim(dummy_goal)
    dummy_obs_and_goal = utils.add_batch_dim(dummy_obs_and_goal)

    # return ContrastiveNetworks(
    #     policy_network=networks_lib.FeedForwardNetwork(
    #         lambda key: policy.init(key, dummy_obs), policy.apply),
    #     q_network=networks_lib.FeedForwardNetwork(
    #         lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply),
    #     repr_fn=repr_fn.apply,
    #     log_prob=lambda params, actions: params.log_prob(actions),
    #     sample=lambda params, key: params.sample(seed=key),
    #     sample_eval=lambda params, key: params.mode(),
    # )

    def sample(params, key):
        c = params.cumsum(axis=1)
        u = jax.random.uniform(key, shape=(len(c), 1))
        action = (u < c).argmax(axis=1)
        action = jax.nn.one_hot(action, num_dimensions)

        return action

    def sample_eval(params, key):
        # logits = params.logits
        action = params.argmax(axis=-1)
        action = jax.nn.one_hot(action, num_dimensions)

        return action

    return ContrastiveNetworks(
        policy_network=networks_lib.FeedForwardNetwork(
            lambda key: policy.init(key, dummy_obs_and_goal), policy.apply
        ),
        q_network=networks_lib.FeedForwardNetwork(
            lambda key: critic.init(key, dummy_obs, dummy_goal, dummy_future_obs),
            critic.apply
        ),
        repr_fn=repr_fn.apply,
        log_prob=lambda params, actions: params.log_prob(actions),
        # sample=lambda params, key: params.sample(seed=key),
        sample=sample,
        # sample_eval=lambda params, key: params.mode(),
        sample_eval=sample_eval,
    )
