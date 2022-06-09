import os
import glob
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


def collect_data(root_exp_data_dir, stats, algos, env_name, seeds, max_steps):
    data = {}
    for stat, _ in stats:
        stat_data = {}
        for algo, algo_dir in algos:
            exp_dir = os.path.join(root_exp_data_dir, algo_dir, env_name)
            # algo_data = []
            all_csv_paths = glob.glob(os.path.join(exp_dir, "*_{}.csv".format(stat)))

            # csv_path = os.path.join(exp_dir, 'seed_' + ''.join([str(seed) for seed in seeds]) + '.csv')

            try:
                # df = pd.read_csv(csv_path)
                algo_data = []
                for idx, csv_path in enumerate(all_csv_paths):
                    df = pd.read_csv(csv_path)
                    df = df[df['Step'] <= max_steps]
                    if idx == 0:
                        algo_data.append(df['Step'].values)
                    if len(df['Value']) < len(algo_data[0]):
                        algo_data = [data[:len(df['Value'])] in algo_data]
                    algo_data.append(df['Value'].values)
                algo_data = np.asarray(algo_data).T

                # df = pd.concat((pd.read_csv(f) for f in all_csv_paths), ignore_index=True)
            except:
                print(f"One of CSV paths not found: {all_csv_paths}")
                continue

            # df = df[df['Step'] <= max_steps]
            # reference: https://stackoverflow.com/questions/19071199/drop-columns-whose-name-contains-a-specific-string-from-pandas-dataframe
            # df.drop(list(df.filter(regex='MIN')), axis=1, inplace=True)
            # df.drop(list(df.filter(regex='MAX')), axis=1, inplace=True)

            # for seed in seeds:
            #     # csv_path = os.path.join(exp_dir, 'seed_' + str(seed) + '.csv')
            #
            #     # try:
            #     #     df = pd.read_csv(csv_path)
            #     # except:
            #     #     print(f"CSV path not found: {csv_path}")
            #     #     continue
            #     #
            #     # df = df[df['Step'] <= max_steps]
            #     seed_data = df[df.columns[1]].values
            #     algo_data.append(seed_data)
            # algo_data = df.values
            stat_data[algo] = algo_data
        data[stat] = stat_data

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
                        args.env_name, args.seeds, args.max_steps)

    # plot
    # num_curves = len(args.algos)
    # cmap = plt.cm.get_cmap('hsv', num_curves)
    cycol = cycle('bgrcmk')
    for algo_idx, (algo, _) in enumerate(args.algos):
        c = next(cycol)
        for stat_idx, (stat, stat_name) in enumerate(args.stats):
            x = data[stat][algo][..., 0]
            y = data[stat][algo][..., 1:]

            x, y = filter_outliers(x, y)

            mean = window_smooth(np.mean(y, axis=-1))
            std = window_smooth(np.std(y, axis=-1))

            axes[stat_idx].plot(x * 1e-6, mean, label=algo, color=c)
            axes[stat_idx].fill_between(x * 1e-6, mean - 0.5 * std, mean + 0.5 * std,
                                        facecolor=c, alpha=0.35)
            axes[stat_idx].set_xlabel('Iterations (M)')
            axes[stat_idx].set_ylabel(stat_name)
            axes[stat_idx].legend(framealpha=0.)
            # axes[stat_idx].title.set_text(stat_name)

    f_path = os.path.abspath(os.path.join(fig_save_dir, args.fig_title + '.png'))
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
        return (splited_s[0], splited_s[1])


    parser = argparse.ArgumentParser()
    parser.add_argument('--root_exp_data_dir', type=str, default='~/offline_c_learning/exp_data')
    parser.add_argument('--fig_title', type=str, default='Off-policy and Offline C-Learning Comparison')
    parser.add_argument('--fig_save_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../figures'))
    parser.add_argument('--algos', type=str_pair, nargs='+', default=[
        ('off-policy', 'off-policy/jun9'),
        ('offline', 'offline/jun9'),
    ])
    parser.add_argument('--env_name', type=str, default='maze2d_open_v0')
    parser.add_argument('--stats', type=str_pair, nargs='+', default=[
        ('FinalDistance', 'Evaluation Final Distance'),
        ('pred_td_targets1 _ (1 - pred_td_targets1)', 'C1 / (1 - C1)'),
        ('pred_td_targets2 _ (1 - pred_td_targets2)', 'C2 / (1 - C2)')
    ])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--max_steps', type=int, default=np.iinfo(np.int64).max)
    args = parser.parse_args()

    main(args)
