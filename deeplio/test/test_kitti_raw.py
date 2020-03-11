import os
import argparse
import yaml

import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt
from deeplio.datasets.kitti import KittiRawData
from deeplio.common.laserscan import LaserScan


def test_kittiraw(args, cfg):
    img_axis = ["x", "y", "z", "remission", "range", "range_xy"]

    output_dir = args['output']
    try:
        os.mkdir(output_dir)
    except Exception as ex:
        print(ex)
        return

    ds_config = cfg['datasets']['kitti']
    root_path = ds_config['root-path']
    ds_type = "train"
    seq_size = cfg['sequence-size']

    image_width = ds_config['image-width']
    image_height = ds_config['image-height']
    fov_up = ds_config['fov-up']
    fov_down = ds_config['fov-down']

    for date, drives in ds_config[ds_type].items():
        for drive in drives:
            ds = KittiRawData(root_path, str(date), str(drive), ds_config)
            length = len(ds)

            try:
                output_path = os.path.join(output_dir, date, drive)
                os.makedirs(output_path)
            except Exception as ex:
                print(ex)
                return

            for axis_name in img_axis:
                try:
                    axis_path = os.path.join(output_dir, date, drive, axis_name)
                    os.makedirs(axis_path)
                except Exception as ex:
                    print(ex)
                    return

            for i in range(length):
                data = ds.get_data(i, seq_size)
                images = data['images']
                imu = data['imu']
                gt = data['ground-truth']
                print("-------------------------------------------------")
                for j, imgs in enumerate(images):
                    if j == 0:
                        for k in range(len(img_axis)):
                            img = imgs[:, :, k]
                            print("{}.{}.{}: min={}, max={}".format(i, j, img_axis[k], img.min(), img.max()))
                            axis_path = os.path.join(output_dir, date, drive, img_axis[k])
                            img_path = os.path.join(axis_path, "{}_{}_{}.png".format(img_axis[k], i, j))
                            plt.imsave(img_path, img)


if __name__ == "__main__":
    with open("../config.yaml") as f:
        cfg = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="Output path", required=True)
    args =vars(parser.parse_args())
    test_kittiraw(args ,cfg)