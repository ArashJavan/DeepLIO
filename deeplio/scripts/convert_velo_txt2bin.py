import os
import time
import argparse
import glob
import threading
import multiprocessing

import tqdm
import numpy as np

from deeplio.common import utils


def convert_txt_to_bin(velo_file):
    velo_bin = velo_file.replace('.txt', '.npy')
    frame = utils.load_velo_scan(velo_file)
    np.save(velo_bin, frame)


def convert(args):
    velo_files = np.array(glob.glob("{}/*.txt".format(args.path)))
    num_files = len(velo_files)

    processes = [] * num_files
    for i in tqdm.tqdm(range(0, num_files)):
        p = multiprocessing.Process(target=convert_txt_to_bin, args=([velo_files[i]]))
        processes.append(p)
        p.start()

        if i % 15 == 0:
            time.sleep(0.5)

    for process in processes:
        process.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepLIO Training')

    # Device Option
    parser.add_argument('-p',  '--path', dest='path', help='path to velodyne text files', required=True)
    args = parser.parse_args()
    convert(args)
    print("done!")





