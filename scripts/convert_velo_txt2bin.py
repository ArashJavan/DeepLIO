import os
import sys
import time
import argparse
import glob
import threading
import multiprocessing

import tqdm
import numpy as np

dname = os.path.dirname(__file__)
module_dir = os.path.abspath("{}/deeplio".format(dname))
content_dir = os.path.abspath("{}/..".format(dname))
sys.path.append(dname)
sys.path.append(module_dir)
sys.path.append(content_dir)

from deeplio.common import utils


def convert_txt_to_bin(velo_file):
    velo_bin = velo_file.replace('.txt', '.npy')
    frame = utils.load_velo_scan(velo_file)
    np.save(velo_bin, frame)


def convert(args):

    for p in args['path']:
        print("Converting {}".format(p))
        velo_files = np.array(glob.glob("{}/*.txt".format(p)))
        num_files = len(velo_files)

        processes = [] * num_files
        for i in tqdm.tqdm(range(0, num_files)):
            p = multiprocessing.Process(target=convert_txt_to_bin, args=([velo_files[i]]))
            processes.append(p)
            p.start()

            # give the cpu some times to finish some of the converting processes
            if i % 15 == 0:
                time.sleep(0.5)

        for process in processes:
            process.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepLIO Training')

    # Device Option
    parser.add_argument('-p',  '--path', nargs="+", help='path or a list paths to velodyne text files', required=True)
    args = vars(parser.parse_args())
    convert(args)
    print("done!")





