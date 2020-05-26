import os
import sys
import time
import argparse
import glob
import threading
import multiprocessing
from multiprocessing import Pool
from pathlib import Path
import tqdm
import numpy as np
import pickle

dname = os.path.dirname(__file__)
module_dir = os.path.abspath("{}/deeplio".format(dname))
content_dir = os.path.abspath("{}/..".format(dname))
sys.path.append(dname)
sys.path.append(module_dir)
sys.path.append(content_dir)

from deeplio.common import utils
from deeplio.datasets import KittiRawData


def convert_oxts2bin(args):
    bar_pos = args[0]

    ds_data = args[1]
    base_sync = ds_data[0]
    base_unsync = ds_data[1]
    drive = ds_data[2]
    date = ds_data[3]

    print("Processing {}_{}".format(date, drive))
    dataset = KittiRawData(base_sync, base_unsync, date, drive, oxts_txt=True)
    # now saving the tranformed data as a pickle binary
    oxts_bin_path_sync = os.path.join(dataset.data_path_sync, "oxts", "data.pkl")
    oxts_bin_path_unsync = os.path.join(dataset.data_path_unsync, "oxts", "data.pkl")

    with open(oxts_bin_path_sync, 'wb') as f:
        pickle.dump(dataset.oxts_sync, f)

    with open(oxts_bin_path_unsync, 'wb') as f:
        pickle.dump(dataset.oxts_unsync, f)

    print('done!')

def convert(args):
    n_worker = len(args['path'])
    counters = list(range(n_worker))
    ds_data = []
    for path in args['path']:
        # we assume path has following structure:
        # /some/folder/KITTI/2011_10_03/2011_10_03_drive_0027_extract/oxts/data
        p = Path(path)
        parents = list(p.parents)
        base = str(parents[4].absolute())
        base_sync = os.path.join(base, 'sync')
        base_unsync = os.path.join(base, 'extract')
        drive = parents[2].stem
        date = parents[1].stem.split('_')[-2]
        ds_data.append([base_sync, base_unsync, date, drive])

    procs = Pool(processes=n_worker, initializer=tqdm.tqdm.set_lock, initargs=(tqdm.tqdm.get_lock(),))
    procs.map(convert_oxts2bin, zip(counters, ds_data))
    procs.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepLIO Training')

    # Device Option
    parser.add_argument('-p',  '--path', nargs="+", help='path or a list paths to velodyne text files', required=True)
    args = vars(parser.parse_args())
    convert(args)
    print("done!")





