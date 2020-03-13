import os
import torch
import glob
import yaml
import datetime as dt
import random

import numpy as np

import torch.utils.data as data

from deeplio.common import utils
from deeplio.common.laserscan import LaserScan
from deeplio.common.logger import PyLogger


class KittiRawData:
    """ KiitiRawData
    more or less same as pykitti with some application specific changes
    """
    MAX_DIST_HDL64 = 120.
    IMU_LENGTH = 10.25

    def __init__(self, base_path, date, drive, cfg, **kwargs):
        self.drive = drive
        self.date = date
        self.calib_path = os.path.join(base_path, date)
        self.data_path = os.path.join(base_path, date, drive)
        self.frames = kwargs.get('frames', None)

        self.image_width = cfg['image-width']
        self.image_height = cfg['image-height']
        self.fov_up = cfg['fov-up']
        self.fov_down = cfg['fov-down']
        self.seq_size = cfg['sequence-size']

        # Find all the data files
        self._get_file_lists()

        # Pre-load data that isn't returned as a generator
        # Pre-load data that isn't returned as a generator
        #self._load_calib()
        self._load_timestamps()
        #self._load_oxts()

        self.imu_get_counter = 0

    def __len__(self):
        return len(self.velo_files)

    def get_velo(self, idx):
        """Read velodyne [x,y,z,reflectance] scan at the specified index."""
        return utils.load_velo_scan(self.velo_files[idx])

    def get_velo_image(self, idx):
        scan = LaserScan(H=self.image_height, W=self.image_width, fov_up=self.fov_up, fov_down=self.fov_down)
        scan.open_scan(self.velo_files[idx])
        scan.do_range_projection()
        # collect projected data and adapt ranges

        proj_xyz = scan.proj_xyz / self.MAX_DIST_HDL64
        proj_remission = scan.proj_remission
        proj_range = 1 / scan.proj_range
        proj_range_xy = 1 / scan.proj_range_xy
        image = np.dstack((proj_xyz, proj_remission, proj_range, proj_range_xy))
        return image

    def _get_file_lists(self):
        """Find and list data files for each sensor."""
        self.oxts_files = sorted(glob.glob(
            os.path.join(self.data_path, 'oxts', 'data', '*.txt')))
        self.velo_files = sorted(glob.glob(
            os.path.join(self.data_path, 'velodyne_points',
                         'data', '*.txt')))

        # Subselect the chosen range of frames, if any
        if self.frames is not None:
            self.oxts_files = utils.subselect_files(
                self.oxts_files, self.frames)
            self.velo_files = utils.subselect_files(
                self.velo_files, self.frames)

        self.oxts_files = np.asarray(self.oxts_files)
        self.velo_files = np.asarray(self.velo_files)

    def _load_calib_rigid(self, filename):
        """Read a rigid transform calibration file as a numpy.array."""
        filepath = os.path.join(self.calib_path, filename)
        data = utils.read_calib_file(filepath)
        return utils.transform_from_rot_trans(data['R'], data['T'])

    def _load_calib(self):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}

        # Load the rigid transformation from IMU to velodyne
        data['T_velo_imu'] = self._load_calib_rigid('calib_imu_to_velo.txt')

    def _load_timestamps(self):
        """Load timestamps from file."""
        timestamp_file_imu = os.path.join(self.data_path, 'oxts', 'timestamps.txt')
        timestamp_file_velo = os.path.join(self.data_path, 'velodyne_points', 'timestamps.txt')

        # Read and parse the timestamps
        self.timestamps_imu = []
        with open(timestamp_file_imu, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                self.timestamps_imu.append(t)
        self.timestamps_imu = np.array(self.timestamps_imu)

        # Read and parse the timestamps
        self.timestamps_velo = []
        with open(timestamp_file_velo, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                self.timestamps_velo.append(t)
        self.timestamps_velo = np.array(self.timestamps_velo)

    def _load_oxts(self):
        """Load OXTS data from file."""
        self.oxts = np.array(utils.load_oxts_packets_and_poses(self.oxts_files))

    def _load_oxts_lazy(self, indices):
        oxts = utils.load_oxts_packets_and_poses(self.oxts_files[indices])
        return oxts

    def get_data(self, start_index, length):
        """
        Get a sequence of velodyne and imu data
        :param start_index: start index
        :param length: length of sequence
        :return:
        """

        velo_timespamps =  [self.timestamps_velo[idx] for idx in range(start_index, start_index + length)]
        images = [self.get_velo_image(idx) for idx in range(start_index, start_index + length)]

        # difference combination of a sequence length
        # e.g. for sequence-length = 3, we have following combinations
        # [0, 1], [0, 2], [1, 2]
        combinations = [[x, y] for y in range(length) for x in range(y)]

        # we do not want that the network memorizes an specific combination pattern
        random.shuffle(combinations)

        data = []
        for combi in combinations:
            idx_0 = combi[0]
            idx_1 = combi[1]

            image_0 = images[idx_0]
            image_1 = images[idx_1]

            velo_start_ts = velo_timespamps[idx_0]
            velo_stop_ts = velo_timespamps[idx_1]

            mask = ((self.timestamps_imu >= velo_start_ts) & (self.timestamps_imu < velo_stop_ts))
            indices = np.argwhere(mask).flatten();
            #indices = range(self.imu_get_counter, np.ceil(self.imu_get_counter + (self.IMU_LENGTH * (self.seq_size - 1))).astype(np.int))
            #self.imu_get_counter += np.ceil(self.IMU_LENGTH).astype(np.int)

            if len(indices) == 0:
                print("Warning: No imu data found for index {}, velo-timestamps: [{} - {}]".format(start_index, velo_start_ts, velo_stop_ts))
                imu_values = [0.]
                gt = [None]
            else:
                oxts = self._load_oxts_lazy(indices)
                #otxs = [self.oxts[index] for index in indices]
                imu_values = [[oxt.packet.ax, oxt.packet.ay, oxt.packet.az, oxt.packet.wx, oxt.packet.wy, oxt.packet.wz] for oxt in oxts]
                gt = [otx.T_w_imu.flatten() for otx in oxts]

                # print("V: {} I:{}\n   {}   {} \n **********".format(velo_start_ts, imu_ts[0], velo_stop_ts, imu_ts[-1]))

            data_dict = {'images': [image_0, image_1], 'imu': imu_values, 'ground-truth': gt}
            data.append(data_dict)
        return data


class Kitti(data.Dataset):
    def __init__(self, config, ds_type='train', transform=None):
        """
        :param root_path:
        :param config: Configuration file including split settings
        :param transform:
        """
        ds_config = cfg['datasets']['kitti']
        root_path = ds_config['root-path']

        self.transform = transform

        self.ds_type = ds_type
        self.seq_size = ds_config['sequence-size']

        self.dataset = []
        self.length_each_drive = []
        self.bins = []

        # Since we are intrested in sequence of lidar frame - e.g. multiple frame at each iteration,
        # depending on the sequence size and the current wanted index coming from pytorch dataloader
        # we must switch between each drive if not enough frames exists in that specific drive wanted from dataloader,
        # therefor we separate valid indices in each drive in bins.
        last_bin_end = -1
        for date, drives in ds_config[self.ds_type].items():
            for drive in drives:
                ds = KittiRawData(root_path, str(date), str(drive), ds_config)

                length = len(ds)

                bin_start = last_bin_end + 1
                bin_end = bin_start + length - self.seq_size
                self.bins.append([bin_start, bin_end])
                last_bin_end = bin_end

                self.length_each_drive.append(length)
                self.dataset.append(ds)

        self.bins = np.asarray(self.bins)
        self.length_each_drive = np.array(self.length_each_drive)

        self.length = self.bins.flatten()[-1] + 1

        self.logger = PyLogger(name="KittiDataset")

        # printing dataset informations
        self.logger.info("Kitti-Dataset Informations")
        self.logger.info("DS-Type: {}, Length: {}, Seq.length: {}".format(ds_type, self.length, self.seq_size))
        for i in range(len(self.length_each_drive)):
            date = self.dataset[i].date
            drive = self.dataset[i].drive
            length = self.length_each_drive[i]
            bins = self.bins[i]
            self.logger.info("Date: {}, Drive: {}, length: {}, bins: {}".format(date, drive, length, bins))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        idx = -1
        num_drive = -1
        for i, bin in enumerate(self.bins):
            bin_start = bin[0]
            bin_end = bin[1]
            if bin_start <= index <= bin_end:
                idx = index - bin_start
                num_drive = i
                break

        if idx < 0 or num_drive < 0:
            print("Error: No bins and no drive number found!")
            return None

        start = time.time()
        dataset = self.dataset[num_drive]
        data = dataset.get_data(idx, self.seq_size)
        end = time.time()
        print("idx:{}, Delta-Time: {}".format(index, end - start))

        return data


if __name__ == "__main__":
    from matplotlib import  pyplot as plt
    import time
    import open3d as o3d
    import copy

    def draw_registration_result(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    with open("../../config.yaml") as f:
        cfg = yaml.safe_load(f)
    dataset = Kitti(config=cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    #data = dataset.__getitem__(334)
    #imu = data['imu']
    #gt = data['ground-truth']

    for i, datas in enumerate(dataloader):
        data = datas[0]
        img_0 = data['images'][0]
        img_1 = data['images'][1]

        xyz_0 = img_0[0, :, :, :3].numpy()
        xyz_0 = xyz_0.reshape(-1, 3) * KittiRawData.MAX_DIST_HDL64
        indices = np.where(np.all(xyz_0 == [0., 0., 0.], axis=1))[0]
        xyz_0 = np.delete(xyz_0, indices, axis=0)

        xyz_1 = img_1[0, :, :, :3].numpy()
        xyz_1 = xyz_1.reshape(-1, 3) * KittiRawData.MAX_DIST_HDL64
        indices = np.where(np.all(xyz_1 == [0., 0., 0.], axis=1))[0]
        xyz_1 = np.delete(xyz_1, indices, axis=0)

        T_gt = data['ground-truth'][-1].numpy().reshape(4, 4)

        pcd_0 = o3d.geometry.PointCloud()
        pcd_0.points = o3d.utility.Vector3dVector(xyz_0)

        pcd_1 = o3d.geometry.PointCloud()
        pcd_1.points = o3d.utility.Vector3dVector(xyz_1)

        draw_registration_result(pcd_0, pcd_1, np.identity(4))
        draw_registration_result(pcd_0, pcd_1, T_gt)

        #
        # im1 = img1[0].numpy()
        # im2 = img2[0].numpy()
        #
        # plt.imshow(im1[:, :, 0])
        # plt.show()
    print(dataset)