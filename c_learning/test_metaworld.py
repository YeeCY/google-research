from absl import app
from absl import flags

import gym
import numpy as np

from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

from PIL import Image


flags.DEFINE_string('env_name', 'reach-v2', 'Name of metaworld task.')

FLAGS = flags.FLAGS


class SawyerDrawer(ALL_V2_ENVIRONMENTS['drawer-close-v2']):
    """Wrapper for the SawyerDrawer environment."""

    def __init__(self):
        super(SawyerDrawer, self).__init__()
        self._random_reset_space.low[0] = 0
        self._random_reset_space.high[0] = 0
        self._partially_observable = False
        self._freeze_rand_vec = False
        self._set_task_called = True
        self._target_pos = np.zeros(0)  # We will overwrite this later.
        self.reset()
        self._freeze_rand_vec = False  # Set False to randomize the goal position.

    # def _get_pos_objects(self):
    #     return self.get_body_com('drawer_link') + np.array([.0, -.16, 0.0])

    def reset_model(self):
        super(SawyerDrawer, self).reset_model()
        # self._set_obj_xyz(np.random.uniform(-0.15, 0.0))  # open, close
        self._set_obj_xyz(-0.1)  # open, close

        img = self.render(offscreen=True)
        img = Image.fromarray(img)
        img_path = "/data/chongyiz/offline_c_learning/debug/img0.png"
        img.save(img_path)
        print("Image saved to: {}".format(img_path))

        self._target_pos = self._get_pos_objects().copy()

        self.data.site_xpos[self.model.site_name2id('goal')] = self._target_pos
        img = self.render(offscreen=True)
        img = Image.fromarray(img)
        img_path = "/data/chongyiz/offline_c_learning/debug/img1.png"
        img.save(img_path)
        print("Image saved to: {}".format(img_path))

        self._set_obj_xyz(np.random.uniform(-0.15, 0.0))

        self.data.site_xpos[self.model.site_name2id('goal')] = self.get_endeff_pos()
        img = self.render(offscreen=True)
        img = Image.fromarray(img)
        img_path = "/data/chongyiz/offline_c_learning/debug/img2.png"
        img.save(img_path)
        print("Image saved to: {}".format(img_path))

        self.data.site_xpos[self.model.site_name2id('goal')] = self.tcp_center
        img = self.render(offscreen=True)
        img = Image.fromarray(img)
        img_path = "/data/chongyiz/offline_c_learning/debug/img3.png"
        img.save(img_path)
        print("Image saved to: {}".format(img_path))

        self.data.site_xpos[self.model.site_name2id('goal')] = self._target_pos
        img = self.render(offscreen=True)
        img = Image.fromarray(img)
        img_path = "/data/chongyiz/offline_c_learning/debug/img4.png"
        img.save(img_path)
        print("Image saved to: {}".format(img_path))

        exit()
        return self._get_obs()

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=np.full(8, -np.inf),
            high=np.full(8, np.inf),
            dtype=np.float32)

    def _get_obs(self):
        finger_right, finger_left = (self._get_site_pos('rightEndEffector'),
                                     self._get_site_pos('leftEndEffector'))
        tcp_center = (finger_right + finger_left) / 2.0
        obj = self._get_pos_objects()
        # Arm position is same as drawer position. We only provide the drawer
        # Y coordinate.
        return np.concatenate([tcp_center, [obj[1]],
                               self._target_pos, [self._target_pos[1]]])

    def step(self, action):
        obs = super(SawyerDrawer, self).step(action)
        return obs, 0.0, False, {}


def main_sawyer(_):
    env = SawyerDrawer()
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    print()


def main(_):
    # door_open_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["door-open-v2-goal-observable"]
    # door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN["door-open-v2-goal-hidden"]
    # reach_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["reach-v2-goal-observable"]
    # reach_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN["reach-v2-goal-hidden"]
    goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[FLAGS.env_name + "-goal-observable"]
    # reach_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[FLAGS.env_name + "-goal-hidden"]

    # env = reach_goal_hidden_cls()
    # env.reset()  # Reset environment
    # a = env.action_space.sample()  # Sample an action
    # obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
    # assert (obs[-3:] == np.zeros(3)).all()  # goal will be zeroed out because env is HiddenGoal

    # You can choose to initialize the random seed of the environment.
    # The state of your rng will remain unaffected after the environment is constructed.
    env1 = goal_observable_cls(seed=5)
    env2 = goal_observable_cls(seed=5)

    env1.reset()  # Reset environment
    env2.reset()
    a1 = env1.action_space.sample()  # Sample an action
    a2 = env2.action_space.sample()
    next_obs1, _, _, _ = env1.step(a1)  # Step the environoment with the sampled random action
    next_obs2, _, _, _ = env2.step(a2)
    assert (next_obs1[-3:] == next_obs2[-3:]).all()  # 2 envs initialized with the same seed will have the same goal
    assert not (next_obs2[-3:] == np.zeros(
        3)).all()  # The env's are goal observable, meaning the goal is not zero'd out

    env3 = goal_observable_cls(seed=10)  # Construct an environment with a different seed
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
    # app.run(main)
    app.run(main_sawyer)
