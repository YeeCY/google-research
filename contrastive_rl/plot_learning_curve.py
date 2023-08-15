import os
# import glob
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interp
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


def collect_data(root_exp_log_dir, stats, algos,
                 timestep_field, max_steps):
    data = {}
    for module_subdir, stats_field, stats_name in stats:
        stats_data = {}
        for algo, algo_dir in algos:
            try:
                exp_dir = os.path.join(root_exp_log_dir, algo_dir)
                # algo_data = []
                seed_dirs = [os.path.join(exp_dir, exp_subdir)
                             for exp_subdir in os.listdir(exp_dir)
                             if re.search(r'^\d+$', exp_subdir)]

                csv_paths = [os.path.join(seed_dir, 'logs', module_subdir, 'logs.csv')
                             for seed_dir in seed_dirs]

                # df = pd.read_csv(csv_path)
                algo_data = []
                algo_data_timesteps = []
                algo_data_values = []
                for idx, csv_path in enumerate(csv_paths):
                    df = pd.read_csv(csv_path)
                    df = df.drop_duplicates(timestep_field, keep='last')
                    df = df[df[timestep_field] <= max_steps]

                    values = df[stats_field].values
                    timesteps = df[timestep_field].values
                    invalid_mask = np.isinf(values) | np.isnan(values)
                    if np.any(invalid_mask):
                        values = values[~invalid_mask]
                        timesteps = timesteps[~invalid_mask]

                    algo_data_timesteps.append(timesteps)
                    algo_data_values.append(values)

                # Interpolate to the max timesteps
                ref_idx, max_timestep = 0, 0
                ref_timesteps = None
                for idx, timesteps in enumerate(algo_data_timesteps):
                    if timesteps[-1] >= max_timestep:
                        ref_idx = idx
                        ref_timesteps = timesteps
                        max_timestep = timesteps[-1]

                algo_data.append(ref_timesteps)

                for idx, (timesteps, values) in enumerate(zip(
                        algo_data_timesteps, algo_data_values)):
                    if idx != ref_idx:
                        interpolation = interp.interp1d(
                            timesteps, values,
                            bounds_error=False,
                            fill_value=(values[0], values[-1]))
                        interp_values = interpolation(ref_timesteps)
                        algo_data.append(interp_values)
                    else:
                        algo_data.append(values)
                algo_data = np.asarray(algo_data).T

                # df = pd.concat((pd.read_csv(f) for f in all_csv_paths), ignore_index=True)
            except FileNotFoundError:
                print(f"CSV path not found: {csv_path}")
                continue

            stats_data[algo] = algo_data
        data[stats_field] = stats_data

    return data


def main(args):
    root_exp_log_dir = os.path.expanduser(args.root_exp_log_dir)
    assert os.path.exists(root_exp_log_dir), \
        "Cannot find root_data_exp_dir: {}".format(root_exp_log_dir)
    fig_save_dir = os.path.expanduser(args.fig_save_dir)
    os.makedirs(fig_save_dir, exist_ok=True)

    f, axes = plt.subplots(1, len(args.stats))
    if len(args.stats) == 1:
        axes = [axes]
    f.set_figheight(4)
    f.set_figwidth(4 * len(args.stats))

    # read all data
    data = collect_data(root_exp_log_dir, args.stats, args.algos,
                        args.timestep_field, args.max_steps)

    # plot
    num_curves = len(args.algos)
    cmap = plt.cm.get_cmap('tab20', num_curves)
    cycol = cycle(cmap.colors)
    for algo_idx, (algo, _) in enumerate(args.algos):
        c = next(cycol)
        for stat_idx, (_, stats_field, stats_name) in enumerate(args.stats):
            try:
                x = data[stats_field][algo][..., 0]
                y = data[stats_field][algo][..., 1:]

                # if stats_field in ['actor_loss',
                #                    'behavioral_cloning_loss',
                #                    'q_ratio',
                #                    'q_pos_ratio',
                #                    'q_neg_ratio']:
                #     x, y = filter_outliers(x, y)

                # mean = window_smooth(np.mean(y, axis=-1))
                # std = window_smooth(np.std(y, axis=-1))
                # we used success_rate_1000 and don't need smoothing
                mean = np.mean(y, axis=-1)
                std = np.std(y, axis=-1)

                axes[stat_idx].plot(x, mean, label=algo, color=c)
                axes[stat_idx].fill_between(x, mean - 0.5 * std, mean + 0.5 * std,
                                            facecolor=c, alpha=0.35)
                axes[stat_idx].set_xlabel(args.timestep_field.replace('_', ' '),
                                          fontsize=14)
                axes[stat_idx].set_ylabel(stats_name, fontsize=14)
                axes[stat_idx].ticklabel_format(style='sci', scilimits=(-3, 4), axis='x')
                axes[stat_idx].legend(loc="upper right", fontsize=12)
            except KeyError:
                print("Algorithm {} data not found".format(algo))

    f_path = os.path.abspath(os.path.join(fig_save_dir, args.fig_filename + '.pdf'))
    f.suptitle(args.fig_title, fontsize=16)
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
    parser.add_argument('--root_exp_log_dir', type=str,
                        default='/projects/rsalakhugroup/chongyiz/contrastive_rl_logs/')
    parser.add_argument('--fig_title', type=str, default='Fetch Reach Image')
    parser.add_argument('--fig_save_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'figures'))
    parser.add_argument('--fig_filename', type=str,
                        default='fetch_reach_image')
    parser.add_argument('--algos', type=str_pair, nargs='+', default=[
        ('C-Learning', '20230814_c_learning_fetch_push'),
    ])
    parser.add_argument('--stats', type=str_triplet, nargs='+', default=[
        ('evaluator', 'success', 'Success Rate'),
        ('evaluator', 'success_1000', 'Success Rate 1000'),
        ('evaluator', 'final_dist', 'Final Distance'),
        ('learner', 'actor_loss', 'Actor Loss'),
        ('learner', 'critic_loss', 'Critic Loss'),
    ])
    parser.add_argument('--timestep_field', type=str, default='actor_steps')
    parser.add_argument('--max_steps', type=int, default=np.iinfo(np.int64).max)
    args = parser.parse_args()

    main(args)
