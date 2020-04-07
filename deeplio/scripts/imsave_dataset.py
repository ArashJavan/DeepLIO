import os
import sys
import shutil
import yaml
import argparse
from multiprocessing import Pool, Lock


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
CHANNEL_CMAP = ['jet', 'jet', 'jet', 'jet', 'nipy_spectral', 'jet']
PIVOT = 100
images_path = "./images"

def run_job(args):
    bar_pos = args[0]
    dataset = args[1]

    length = len(dataset)
    num_channels = len(CHANNEL_NAMES)

    counter = 0

    # creatin folder for saveing images
    folder_path = "{}/{}_{}".format(images_path, dataset.date, dataset.drive)
    try:
        shutil.rmtree(folder_path)
    except:
        pass

    os.makedirs(folder_path)

    for i in tqdm.trange(length, desc=dataset.data_path, position=bar_pos):
        im = dataset.get_velo_image(i)
        if i % PIVOT == 0:
            for c in range(num_channels):
                plt.imsave("{}/{}_{}.png".format(folder_path, CHANNEL_NAMES[c], i), np.absolute(im[:, :, c]), cmap=CHANNEL_CMAP[c])


def main(args):
    global images_path

    with open(args['config']) as f:
        config = yaml.safe_load(f)

    images_path = args['path']

    print("Saveing Projected Lidar Frames as Images.\n args: {}".format(args))

    ds_config = config['datasets']['kitti']
    root_path = ds_config['root-path']

    ds_type = "train"
    num_channels = len(CHANNEL_NAMES)

    datasets = [KittiRawData(root_path, str(date), str(drive), ds_config) for date, drives in ds_config[ds_type].items()
                for drive in drives]
    n_worker = len(datasets)
    counters = list(range(n_worker))

    procs = Pool(processes=n_worker, initializer=tqdm.tqdm.set_lock, initargs=(tqdm.tqdm.get_lock(),))
    procs.map(run_job, zip(counters, datasets))
    procs.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepLIO Training')
    parser.add_argument('-c', '--config', default="../config.yaml", help='configuration file')
    parser.add_argument('-p', '--path', default="../images", help='Images Path')

    args = vars(parser.parse_args())
    main(args)

    print("Done!")




