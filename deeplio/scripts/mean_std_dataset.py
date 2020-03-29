import os
import yaml
import argparse

import tqdm

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from deeplio.datasets import KittiRawData

CHANNEL_NAMES = ['x', 'y', 'z', 'remission', 'rang', 'depth']


def main(args):

    with open(args['config']) as f:
        config = yaml.safe_load(f)

    print("mean-and-std-dataset: Mean and std-deviation of "
          "cylindrical projected KITTI Lidar Frames.\n args: {}".format(args))

    ds_config = config['datasets']['kitti']
    root_path = ds_config['root-path']

    ds_type = "train"
    num_channels = len(config['channels'])

    # Since we are intrested in sequence of lidar frame - e.g. multiple frame at each iteration,
    # depending on the sequence size and the current wanted index coming from pytorch dataloader
    # we must switch between each drive if not enough frames exists in that specific drive wanted from dataloader,
    # therefor we separate valid indices in each drive in bins.
    last_bin_end = -1

    pixel_num = 0
    channel_sum = np.zeros(num_channels)
    channel_sum_squared = np.zeros(num_channels)

    counter = 0
    pivot = 200
    channel_hist =  [[] for i in range(num_channels)]

    for date, drives in ds_config[ds_type].items():
        for drive in drives:
            ds = KittiRawData(root_path, str(date), str(drive), ds_config)
            print("calculating {}".format(ds.data_path))

            length = len(ds)

            for i in tqdm.tqdm(range(length)):
                im = ds.get_velo_image(i)
                pixel_num += (im.size / num_channels)
                channel_sum += np.sum(im, axis=(0, 1))
                channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

                if (counter % pivot) == 0:
                    im_flat = im.reshape(-1, 6).T
                    for c in range(num_channels):
                        channel_hist[c].extend(im_flat[c])
                counter += 1

    if args['plot']:
        plt.ioff()
        fig , ax = plt.subplots(2, 3, figsize=(10, 7))
        ax = ax.flatten()
        for j in range(num_channels):
            h = ax[j].hist(channel_hist[j], bins=50, density=True)
            ax[j].grid(True)
            ax[j].set_title("channel {}".format(CHANNEL_NAMES[j]))
        fig.tight_layout()
        fig.savefig("../histogram_mean_and_std.png")

    channel_mean = channel_sum / pixel_num
    channel_std = np.sqrt(channel_sum_squared / pixel_num - np.square(channel_mean))

    with open("../mean_and_std.txt", 'a') as f:
        f.write("mean: {}, std: {}".format(channel_mean, channel_std))

    print("mean: {}".format(channel_mean))
    print("std: {}".format(channel_std))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepLIO Training')
    parser.add_argument('-c', '--config', default="../config.yaml", help='configuration file')
    parser.add_argument('--plot', default=False, help='Plot and save histogram statistic', action='store_true')

    args = vars(parser.parse_args())
    main(args)
    print("Done!")




