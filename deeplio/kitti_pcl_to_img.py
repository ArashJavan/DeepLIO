import os
import glob
from os import listdir
from os.path import join, isdir
import shutil
import yaml
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

from deeplio.common import utils


class KittiPclToImage:
    def __init__(self, config, delta_theta=0.2, delta_phi=0.4, f_h=360., f_vu=3., f_vl=-25., channels=3):
        ds_infos = config['datasets']['kitti']
        self.root_path = ds_infos['root-path']

        self.date_pathes = [join(self.root_path, f) for f in listdir(self.root_path) if isdir(join(self.root_path, f))]
        self.drive_pathes = [join(f, d) for f in self.date_pathes for d in listdir(f) if isdir(join(f, d))]

        self.delta_theta = delta_theta
        self.delta_phi = delta_phi
        self.f_h = f_h
        self.f_vu = f_vu
        self.f_vl = f_vl
        self.f_v = abs(f_vu) + abs(f_vl)

        self.W = int(self.f_h / delta_theta)
        self.H = int(self.f_v / delta_phi)
        self.C = channels

    def convert(self):
        for drive_path in self.drive_pathes:
            img_dir = join(drive_path, "velodyne_images")
            if os.path.exists(img_dir) and os.path.isdir(img_dir):
                shutil.rmtree(img_dir)
            os.makedirs(img_dir)

            velo_path = join(drive_path, "velodyne_points/data")
            velo_point_pathes = glob.glob("{}/*.txt".format(velo_path))
            for i, velo_path in enumerate(velo_point_pathes):
                img_path = join(img_dir, os.path.basename(velo_path))
                print("[{}] Processsing {}".format(float(i/len(velo_point_pathes)), img_path))

                # point_cloud_raw = utils.load_velo_scan_raw(velo_path)
                # point_cloud_raw = point_cloud_raw[point_cloud_raw[:, 3] > 0]  # removing unvalid reflectance values
                #
                # N = len(point_cloud_raw)
                # zeros = np.zeros((N, 2))
                # intensities = np.hstack((point_cloud_raw[:, 3].reshape((N, 1)), zeros))
                #
                # pcl = o3d.geometry.PointCloud()
                # pcl.points = o3d.utility.Vector3dVector(point_cloud_raw[:, 0:3])
                # pcl.colors = o3d.utility.Vector3dVector(intensities)
                # pcl = pcl.voxel_down_sample(voxel_size=0.04)
                #
                # point_cloud = np.asarray(pcl.points)
                # intensities = np.asarray(pcl.colors)[:, 0]
                # point_cloud = np.hstack((point_cloud, intensities.reshape(-1, 1)))

                scan = LaserScan()
                scan.open_scan(velo_path)
                scan.do_range_projection()
                proj_xyz = scan.proj_xyz
                proj_intensities = scan.proj_remission
                proj_depth = scan.proj_range
                plt.imshow(proj_xyz)
                plt.show()
                plt.imshow(proj_intensities)
                plt.show()
                plt.imshow(proj_depth)
                plt.show()

if __name__ == "__main__":
    with open("../config.yaml") as f:
        cfg = yaml.safe_load(f)

    kitti2img = KittiPclToImage(cfg)
    kitti2img.convert()
