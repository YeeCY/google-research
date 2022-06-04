import gym
import d4rl


def main():
    # Create the environment
    env = gym.make('maze2d-umaze-v1')

    # d4rl abides by the OpenAI gym interface
    env.reset()
    env.step(env.action_space.sample())

    # Each task is associated with a dataset
    # dataset contains observations, actions, rewards, terminals, and infos
    dataset = env.get_dataset()
    print(dataset['observations'])  # An N x dim_observation Numpy array of observations

    # Alternatively, use d4rl.qlearning_dataset which
    # also adds next_observations.
    q_learning_dataset = d4rl.qlearning_dataset(env)

    # Get full rollouts
    trajectories = list(d4rl.sequence_dataset(env))
    print()


if __name__ == "__main__":
    main()
