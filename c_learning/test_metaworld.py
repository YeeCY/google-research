from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
# these are ordered dicts where the key : value
# is env_name : env_constructor

import numpy as np


def main():
    # door_open_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["door-open-v2-goal-observable"]
    # door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN["door-open-v2-goal-hidden"]
    reach_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["reach-v2-goal-observable"]
    reach_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN["reach-v2-goal-hidden"]

    env = reach_goal_hidden_cls()
    env.reset()  # Reset environment
    a = env.action_space.sample()  # Sample an action
    obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
    assert (obs[-3:] == np.zeros(3)).all()  # goal will be zeroed out because env is HiddenGoal

    # You can choose to initialize the random seed of the environment.
    # The state of your rng will remain unaffected after the environment is constructed.
    env1 = reach_goal_observable_cls(seed=5)
    env2 = reach_goal_observable_cls(seed=5)

    env1.reset()  # Reset environment
    env2.reset()
    a1 = env1.action_space.sample()  # Sample an action
    a2 = env2.action_space.sample()
    next_obs1, _, _, _ = env1.step(a1)  # Step the environoment with the sampled random action
    next_obs2, _, _, _ = env2.step(a2)
    assert (next_obs1[-3:] == next_obs2[-3:]).all()  # 2 envs initialized with the same seed will have the same goal
    assert not (next_obs2[-3:] == np.zeros(
        3)).all()  # The env's are goal observable, meaning the goal is not zero'd out

    env3 = reach_goal_observable_cls(seed=10)  # Construct an environment with a different seed
    env1.reset()  # Reset environment
    env3.reset()
    a1 = env1.action_space.sample()  # Sample an action
    a3 = env3.action_space.sample()
    next_obs1, _, _, _ = env1.step(a1)  # Step the environoment with the sampled random action
    next_obs3, _, _, _ = env3.step(a3)

    assert not (next_obs1[-3:] == next_obs3[
                                  -3:]).all()  # 2 envs initialized with different seeds will have different goals
    assert not (next_obs1[-3:] == np.zeros(
        3)).all()  # The env's are goal observable, meaning the goal is not zero'd out


if __name__ == "__main__":
    main()
