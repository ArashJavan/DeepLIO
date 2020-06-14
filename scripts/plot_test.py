import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

dname = os.path.dirname(os.path.realpath(__file__))
module_dir = os.path.abspath("{}/../deeplio".format(dname))
content_dir = os.path.abspath("{}/..".format(dname))
sys.path.append(dname)
sys.path.append(module_dir)
sys.path.append(content_dir)

loss_exclude = [] # [1, 2, 3, 6, 7, 8]
map_exclude = [] # [1, 3, 6, 7, 8]
colors = ['red',
          'green',
          'blue',
          'fuchsia',
          'darkgreen',
          'navy',
          'orange',
          'violet',
          'yellow',
          'brown'
          ]

# colors = ['red',
#           'green',
#           'blue',
#           'fuchsia',
#           'darkgreen',
#           'navy',
#           'orange',
#           'violet',
#           'yellow',
#           'brown'
#           ]

colors = ['#1f77b4',
            '#ff7f0e',
            '#2ca02c',
            '#d62728',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf',
]

tests_idx = [1, 5, 8, 10]

def plot_from_csvs(csvs, suffix='pred_x'):
    map_name = csvs[0].split('/')[-1].split('.')[0]
    root_dir = Path(csvs[0]).parent.parent.parent
    plot_dir = "{}/{}".format(root_dir, map_name)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # plot map
    fig_map, ax_map = plt.subplots(1, 1)
    ax_map.set_title('Map: {}-{}'. format(map_name, suffix))
    ax_map.grid('on')
    ax_map.set_xlabel('x [m]')
    ax_map.set_ylabel('y [m]')


    # plot losses
    fig_loss, ax_loss = plt.subplots(1, 1)
    ax_loss.set_title('Loss: {}-{}'. format(map_name, suffix))
    ax_loss.grid('on')
    ax_loss.set_xlabel('iters')
    ax_loss.set_ylabel('loss')

    for i, csv in enumerate(csvs):
        test_nr = tests_idx[i] # int(csv.split('/')[-2].split('_')[0])
        traj = np.loadtxt(csv, delimiter=',')

        T_glob_pred = traj[:, 1:17].reshape(-1, 4, 4)
        T_glob_gt = traj[:, 17:33].reshape(-1, 4, 4)
        losses = traj[:, 33]

        if i == 0:
            ax_map.plot(T_glob_gt[:, 0, 3], T_glob_gt[:, 1, 3], alpha=0.8, linewidth=2, label='Ground Truth', color='black')
            #ax_map.scatter(T_glob_gt[:, 0, 3], T_glob_gt[:, 1, 3], alpha=0.5, s=0.1)
        if test_nr not in map_exclude:
            ax_map.plot(T_glob_pred[:, 0, 3], T_glob_pred[:, 1, 3], colors[test_nr-1], alpha=0.8, linewidth=1,
                        label='Test {}'.format(test_nr))
            #ax_map.scatter(T_glob_pred[:, 0, 3], T_glob_pred[:, 1, 3], alpha=0.5, s=0.1)

        if test_nr not in loss_exclude:
            ax_loss.plot(range(len(losses)), losses, colors[test_nr-1], alpha=1, linewidth=1, label='Test {}'.format(test_nr),
                       )
            #ax_loss.scatter(range(len(losses)), losses, alpha=0.5, s=0.1)

    ax_map.legend()
    ax_loss.legend()

    fig_map.savefig("{}/map_{}_{}.png".format(plot_dir, map_name, suffix), dpi=400)
    fig_loss.savefig("{}/loss_{}_{}.png".format(plot_dir, map_name, suffix), dpi=400)
    plt.close(fig_map)
    plt.close(fig_loss)

maps = ['2011_09_26_0023',
        '2011_09_30_0016',
        '2011_09_30_0027']


csv_pathes = list(Path("{}/outputs".format(content_dir)).rglob('*.csv'))
csv_pathes = np.array([str(csv) for csv in csv_pathes])

for map in maps:
    csvs_map = csv_pathes[[map in csv and 'best' not in csv for csv in csv_pathes]]

    if len(csvs_map) == 0:
        continue
    csvs_map = np.array(sorted(csvs_map, key=lambda csv: csv.split('/')[-3]))
    csv_map_xq = csvs_map[['xq' in csv for  csv in csvs_map]]
    csv_map_x = csvs_map[['x' in csv and 'xq' not in csv for csv in csvs_map]]
    plot_from_csvs(csv_map_x, suffix='pred_x')
    plot_from_csvs(csv_map_xq, suffix='pred_xq')
