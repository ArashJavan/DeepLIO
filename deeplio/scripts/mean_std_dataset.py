import os
import yaml
import argparse

import tqdm

import numpy as np

from deeplio.datasets import KittiRawData


def main(config):
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

    channel_mean = channel_sum / pixel_num
    channel_std = np.sqrt(channel_sum_squared / pixel_num - np.square(channel_mean))

    with open("../mean_and_std.txt", 'a') as f:
        f.write("mean: {}, std: {}".format(channel_mean, channel_std))

    print("mean: {}".format(channel_mean))
    print("std: {}".format(channel_std))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepLIO Training')
    parser.add_argument('-c', '--config', default="../config.yaml", help='configuration file')

    args = parser.parse_args()

    with open("../config.yaml") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
    print("Done!")




