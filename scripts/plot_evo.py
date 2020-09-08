import os
import sys
import subprocess
import argparse
import glob

import numpy as np

sequences = ['2011_10_03_0027',
             '2011_10_03_0042',
             '2011_10_03_0034',
             '2011_09_30_0016',
             '2011_09_30_0018',
             '2011_09_30_0020',
             '2011_09_30_0027',
             '2011_09_30_0028',
             '2011_09_30_0033',
             '2011_09_30_0034',
             ]

def run_evo_traj(args):
    cmd = ["evo_traj", "kitti"] + args
    subprocess.run(cmd)

def multi_plot(pathes):
    for seq in sequences:
        gt_file = "{}/gt_kitti_{}.txt".format(pathes[0], seq)
        arg_gt = "--ref={}".format(gt_file)

        arg_files = []
        for path in pathes:
            arg_file = "{}/pred_kitti_{}.txt".format(path, seq)
            arg_files.append(arg_file)
        save_plot = "--save_plot={}/pred_kitti_{}.pdf".format(pathes[0], seq)
        args = [*arg_files, arg_gt, save_plot, "--plot_mode=xy", "--no_warnings"]
        run_evo_traj(args)

def single_plots(pathes):
    for path in pathes:
        for seq in sequences:
            gt_file = "{}/gt_kitti_{}.txt".format(path, seq)
            arg_gt = "--ref={}".format(gt_file)

            pred_file = "{}/pred_kitti_{}.txt".format(path, seq)
            arg_file = pred_file
            save_plot = "--save_plot={}/pred_kitti_{}.pdf".format(path, seq)
            args = [arg_file, arg_gt, save_plot, "--plot_mode=xy", "--no_warnings"]
            run_evo_traj(args)


def main(args):
    pathes = args.path
    pathes_xq = []
    pathes_x = []
    pathes_q = []
    for path in pathes:
        pathes_xq.append(glob.glob("{}/test_*_xq".format(path))[0])
        pathes_x.append(glob.glob("{}/test_*_x".format(path))[0])
        pathes_q.append(glob.glob("{}/test_*_q".format(path))[0])
    pathes = [pathes_xq, pathes_x, pathes_q]
    for path in pathes:
        if args.mode == "single":
            single_plots(path)
        else:
            multi_plot(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, nargs='+',
                        help='A list of directories with the test resutls.')
    parser.add_argument('-m', '--mode', type=str, default="multi", help='Plot mode, seprate plots or all combined in one plot. [single, multi]')


    args = parser.parse_args()
    main(args)