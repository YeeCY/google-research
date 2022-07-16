import os
# import glob
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle


def filter_outliers(x, y, m=100.0, eps=1e-6):
    """
    Reference: https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    """
    # indices = np.all(
    #     abs(y - np.mean(y, axis=-1, keepdims=True)) < m * np.std(y, axis=-1, keepdims=True),
    #     axis=-1)
    # filtered_x, filtered_y = x[indices], y[indices]
    d = np.abs(y - np.median(y, axis=0, keepdims=True))
    mdev = np.median(d, axis=0, keepdims=True)
    s = d / (mdev + eps)
    indices = np.all(s < m, axis=-1)
    # data = y[..., 0]
    # d = np.abs(data - np.median(data))
    # mdev = np.median(d)
    # s = d / mdev if mdev else 0.
    # indices = s < m

    filtered_x, filtered_y = x[indices], y[indices]

    return filtered_x, filtered_y


def window_smooth(y, window_width=20, smooth_coef=0.05):
    window_size = int(window_width / 2)
    y_padding = np.concatenate([y[:1] for _ in range(window_size)], axis=0).flatten()
    y = np.concatenate([y_padding, y], axis=0)
    y_padding = np.concatenate([y[-1:] for _ in range(window_size)], axis=0).flatten()
    y = np.concatenate([y, y_padding], axis=0)

    coef = list()
    for i in range(window_width + 1):
        coef.append(np.exp(- smooth_coef * abs(i - window_size)))
    coef = np.array(coef)

    yw = list()
    for t in range(len(y) - window_width):
        yw.append(np.sum(y[t:t + window_width + 1] * coef) / np.sum(coef))

    return np.array(yw).flatten()


def collect_data(root_exp_data_dir, stats, algos, env_name,
                 timestep_field, max_steps, num_sgd_steps_per_step):
    data = {}
    for module_subdir, stats_field, stats_name in stats:
        stats_data = {}
        for algo, algo_dir in algos:
            exp_dir = os.path.join(root_exp_data_dir, algo_dir, env_name)
            # algo_data = []
            seed_dirs = [os.path.join(exp_dir, exp_subdir)
                         for exp_subdir in os.listdir(exp_dir)
                         if re.search(r'^\d+$', exp_subdir)]

            csv_paths = [os.path.join(seed_dir, 'logs', module_subdir, 'logs.csv')
                         for seed_dir in seed_dirs]

            try:
                # df = pd.read_csv(csv_path)
                algo_data = []
                for idx, csv_path in enumerate(csv_paths):
                    df = pd.read_csv(csv_path)
                    df = df.drop_duplicates(timestep_field, keep='last')
                    df = df[df[timestep_field] <= max_steps // num_sgd_steps_per_step]
                    if idx == 0:
                        algo_data.append(
                            df[timestep_field].values * num_sgd_steps_per_step)
                    if len(df[stats_field]) < len(algo_data[0]):
                        algo_data = [data[:len(df[stats_field])] for data in algo_data]
                        algo_data.append(df[stats_field].values)
                    else:
                        algo_data.append(df[stats_field].values[:len(algo_data[0])])
                algo_data = np.asarray(algo_data).T

                # df = pd.concat((pd.read_csv(f) for f in all_csv_paths), ignore_index=True)
            except:
                print(f"CSV path not found: {csv_path}")
                continue

            stats_data[algo] = algo_data
        data[stats_field] = stats_data

    return data


def main(args):
    root_exp_data_dir = os.path.expanduser(args.root_exp_data_dir)
    assert os.path.exists(root_exp_data_dir), \
        "Cannot find root_data_exp_dir: {}".format(root_exp_data_dir)
    fig_save_dir = os.path.expanduser(args.fig_save_dir)
    os.makedirs(fig_save_dir, exist_ok=True)

    f, axes = plt.subplots(1, len(args.stats))
    if len(args.stats) == 1:
        axes = [axes]
    f.set_figheight(6)
    f.set_figwidth(6 * len(args.stats))

    # read all data
    data = collect_data(root_exp_data_dir, args.stats, args.algos,
                        args.env_name, args.timestep_field, args.max_steps,
                        args.num_sgd_steps_per_step)

    # plot
    # num_curves = len(args.algos)
    # cmap = plt.cm.get_cmap('hsv', num_curves)
    cycol = cycle('bgrcmk')
    for algo_idx, (algo, _) in enumerate(args.algos):
        c = next(cycol)
        for stat_idx, (_, stats_field, stats_name) in enumerate(args.stats):
            x = data[stats_field][algo][..., 0]
            y = data[stats_field][algo][..., 1:]

            # x, y = filter_outliers(x, y)

            mean = window_smooth(np.mean(y, axis=-1))
            std = window_smooth(np.std(y, axis=-1))

            axes[stat_idx].plot(x * 1e-6, mean, label=algo, color=c)
            axes[stat_idx].fill_between(x * 1e-6, mean - 0.5 * std, mean + 0.5 * std,
                                        facecolor=c, alpha=0.35)
            axes[stat_idx].set_xlabel('Iterations (M)')
            axes[stat_idx].set_ylabel(stats_name)
            axes[stat_idx].legend(framealpha=0.)

    f_path = os.path.abspath(os.path.join(fig_save_dir, args.fig_filename + '.png'))
    f.suptitle(args.fig_title)
    plt.tight_layout()
    plt.savefig(fname=f_path)
    print(f"Save figure to: {f_path}")
    # plt.show()


if __name__ == "__main__":
    # custom argument type
    def str_pair(s):
        splited_s = s.split(',')
        assert splited_s, 'invalid string pair'
        return splited_s[0], splited_s[1]

    def str_triplet(s):
        splited_s = s.split(',')
        assert splited_s, 'invalid string pair'
        return splited_s[0], splited_s[1], splited_s[2]

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_exp_data_dir', type=str, default='~/offline_c_learning/contrastive_rl_logs/offline')
    parser.add_argument('--fig_title', type=str, default='offline_ant_medium_play')
    parser.add_argument('--fig_save_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'figures'))
    parser.add_argument('--fig_filename', type=str,
                        default='offline_ant_medium_play')
    parser.add_argument('--algos', type=str_pair, nargs='+', default=[
        ('c-learning', 'jul_14_c_learning'),
        ('sarsa c-learning', 'jul_15_sarsa_c_learning'),
        ('fitted sarsa c-learning', 'jul_15_fitted_sarsa_c_learning'),
        ('contrastive nce', 'jul_15_contrastive_nce'),
        ('contrastive nce random goal negative action sampling',
         'jul_15_contrastive_nce_random_goal_neg_action_sampling'),
        ('contrastive nce future goal negative action sampling',
         'jul_15_contrastive_nce_future_goal_neg_action_sampling'),
    ])
    parser.add_argument('--env_name', type=str, default='offline_ant_medium_play')
    parser.add_argument('--stats', type=str_triplet, nargs='+', default=[
        # ('evaluator', 'success', 'Success'),
        ('evaluator', 'success_10', 'Success Rate 10'),
        ('evaluator', 'success_100', 'Success Rat 100'),
        ('evaluator', 'success_1000', 'Success Rate 1000'),
        ('evaluator', 'final_dist', 'Final Distance'),
        # ('evaluator', 'episode_return', 'Episode Return'),
        ('learner', 'actor_loss', 'Actor Loss'),
        ('learner', 'critic_loss', 'Critic Loss'),
        ('learner', 'behavioral_cloning_loss', '(Standalone) Behavioral Cloning Loss')
    ])
    parser.add_argument('--timestep_field', type=str, default='learner_steps')
    parser.add_argument('--max_steps', type=int, default=np.iinfo(np.int64).max)
    parser.add_argument('--num_sgd_steps_per_step', type=int, default=64)
    args = parser.parse_args()

    main(args)
