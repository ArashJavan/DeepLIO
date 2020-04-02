import os
import yaml
import argparse
from multiprocessing import Pool, Lock
import sys
import os
dname = os.path.dirname(__file__)
module_dir = os.path.abspath("{}/..".format(dname))
content_dir = os.path.abspath("{}/..".format(module_dir))
sys.path.append(dname)
sys.path.append(module_dir)
sys.path.append(content_dir)

import tqdm

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from deeplio.datasets import KittiRawData

CHANNEL_NAMES = ['x', 'y', 'z', 'remission', 'rang', 'depth']
PIVOT = 200

lock = Lock()


def run_job(args):
    def fetch_images(i):
        pass

    pos = args[0]
    ds = args[1]

    length = len(ds)
    num_channels = len(CHANNEL_NAMES)

    pixel_num = 0
    channel_sum = np.zeros(num_channels)
    channel_sum_squared = np.zeros(num_channels)
    channel_hist = [[] for i in range(num_channels)]

    counter = 0

    for i in tqdm.trange(length, desc=ds.data_path, position=pos):
        im = ds.get_velo_image(i)
        im_channels = im.reshape(-1, 6).T
        pixel_num += (im.size / num_channels)
        channel_sum += np.sum(im_channels, axis=1)
        channel_sum_squared += np.sum(np.square(im_channels), axis=1)

        if (counter % PIVOT) == 0:
            for c in range(num_channels):
                channel_hist[c].extend(im_channels[c][im_channels[c] != 0])
        counter += 1
    return (pixel_num, channel_sum, channel_sum_squared, channel_hist)


def main(args):

    with open(args['config']) as f:
        config = yaml.safe_load(f)

    print("mean-and-std-dataset: Mean and std-deviation of "
          "cylindrical projected KITTI Lidar Frames.\n args: {}".format(args))

    ds_config = config['datasets']['kitti']
    root_path = ds_config['root-path']

    ds_type = "train"
    num_channels = len(CHANNEL_NAMES)

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
    channel_hist = [[] for i in range(num_channels)]

    datasets = [KittiRawData(root_path, str(date), str(drive), ds_config) for date, drives in ds_config[ds_type].items()
                for drive in drives]
    n_worker = len(datasets)
    counters = list(range(n_worker))

    procs = Pool(processes=n_worker, initializer=tqdm.tqdm.set_lock, initargs=(tqdm.tqdm.get_lock(),))
    data = procs.map(run_job, zip(counters, datasets))
    procs.close()

    for d in data:
        pixel_num += d[0]
        channel_sum += d[1]
        channel_sum_squared += d[2]
        for c in range(num_channels):
            channel_hist[c].extend(d[3][c])

    channel_mean = channel_sum / pixel_num
    channel_std = np.sqrt(channel_sum_squared / pixel_num - np.square(channel_mean))

    if args['plot']:
        plt.ioff()
        fig , ax = plt.subplots(2, 3, figsize=(10, 7))
        ax = ax.flatten()
        for j in range(num_channels):
            ch_hist = np.array(channel_hist[j])
            h = ax[j].hist(ch_hist, bins=100, density=True)
            ax[j].grid(True)
            ax[j].set_title("Ch: {}, Min: {:.2f}, Max:{:.2f} \n Mean: {:.2f}, Std: {:.2f}".format(CHANNEL_NAMES[j],
                                                                                          ch_hist.min(),
                                                                                          ch_hist.max(),
                                                                                          channel_mean[j],
                                                                                          channel_std[j]))
        fig.tight_layout()
        fig.savefig("../histogram_mean_and_std_old.png")

    with open("../mean_and_std.txt", 'w') as f:
        f.write("mean: {}\nstd: {}".format(channel_mean, channel_std))

    print("mean: {}".format(channel_mean))
    print("std: {}".format(channel_std))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepLIO Training')
    parser.add_argument('-c', '--config', default="../config.yaml", help='configuration file')
    parser.add_argument('--plot', default=False, help='Plot and save histogram statistic', action='store_true')

    args = vars(parser.parse_args())
    main(args)
    print("Done!")




