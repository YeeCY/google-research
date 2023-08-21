import gym
import gym.utils as gym_utils
import numpy as np

SIZE = 5
NUM_STATES = SIZE ** 2
NUM_ACTIONS = 5


class GridWorldEnv(gym.Env, gym_utils.EzPickle):
    def __init__(self):
        super().__init__()

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(NUM_ACTIONS,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.full((NUM_STATES * 2,), -np.inf),
            high=np.full((NUM_STATES * 2,), np.inf),
            dtype=np.float32)

        self._current_s = None
        self._goal = None

    def int2onehot(self, state):
        onehot_state = np.zeros(NUM_STATES, dtype=np.float32)
        onehot_state[state] = 1.0

        return onehot_state

    def reset(self):
        s = np.random.randint(NUM_STATES)
        g = np.random.randint(NUM_STATES)
        self._current_s = s
        self._goal = g

        return np.concatenate([self.int2onehot(s), self.int2onehot(g)])

    def step(self, action):
        assert self._current_s is not None
        ij = np.array(np.unravel_index(self._current_s, (SIZE, SIZE)))
        a = np.argmax(action)

        if a == 0:
            ij[0] += 1
        elif a == 1:
            ij[1] += 1
        elif a == 2:
            ij[0] -= 1
        elif a == 3:
            ij[1] -= 1
        ij = ij.clip(0, SIZE - 1)

        s = np.ravel_multi_index(ij, (SIZE, SIZE))
        self._current_s = s

        r = self.compute_reward(s)

        return np.concatenate([self.int2onehot(s), self.int2onehot(self._goal)]), r, False, {}

    def compute_reward(self, state):
        s_ij = np.array(np.unravel_index(state, (SIZE, SIZE)))
        g_ij = np.array(np.unravel_index(self._goal, (SIZE, SIZE)))
        if np.all(s_ij == g_ij):
            r = 1.0
        else:
            r = 0.0

        return r

    def render(self, mode="human"):
        raise NotImplementedError