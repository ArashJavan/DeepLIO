import os
import torch
import glob
import yaml
import datetime as dt
import numpy as np

import torch.utils.data as data

from deeplio.common import utils
from deeplio.common.laserscan import LaserScan

class KittiRawData:
    """ KiitiRawData
    more or less same as pykitti with some application specific changes
    """
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

        # Find all the data files
        self._get_file_lists()

        # Pre-load data that isn't returned as a generator
        # Pre-load data that isn't returned as a generator
        #self._load_calib()
        self._load_timestamps()
        self._load_oxts()

    @property
    def velo(self):
        """Generator to read velodyne [x,y,z,reflectance] scan data from binary files."""
        # Return a generator yielding Velodyne scans.
        # Each scan is a Nx4 array of [x,y,z,reflectance]
        return utils.yield_velo_scans(self.velo_files)

    def __len__(self):
        return len(self.velo_files)

    def get_velo(self, idx):
        """Read velodyne [x,y,z,reflectance] scan at the specified index."""
        return utils.load_velo_scan(self.velo_files[idx])

    def get_velo_image(self, idx):
        scan = LaserScan(H=self.image_height, W=self.image_width, fov_up=self.fov_up, fov_down=self.fov_down)
        scan.open_scan(self.velo_files[idx])
        scan.do_range_projection()
        xyz = scan.proj_xyz
        remissions = scan.proj_remission
        range = scan.proj_range
        image = np.dstack((xyz, remissions, range))
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

    def get_data(self, start, length):
        """
        Get a sequence of velodyne and imu data
        :param start: start index
        :param length: length of sequence
        :return:
        """
        velo_start_ts = self.timestamps_velo[start]
        velo_stop_ts = self.timestamps_velo[start + length - 1]

        images = [self.get_velo_image(idx) for idx in range(start, start+length)]

        mask = ((self.timestamps_imu >= velo_start_ts) & (self.timestamps_imu < velo_stop_ts))
        imu_ts = self.timestamps_imu[mask]
        imu_values = [[otx[0].ax, otx[0].ay, otx[0].az, otx[0].wx, otx[0].wy, otx[0].wz] for otx in self.oxts[mask]]

        ground_truth = [otx[1].flatten() for otx in self.oxts[mask]]

        data = {}
        data['images'] = images
        data['imu'] = imu_values
        data['ground-truth'] = ground_truth
        return data


class Kitti(data.Dataset):
    def __init__(self, config, ds_type='train', transform=None):
        """
        :param root_path:
        :param config: Configuration file including split settings
        :param transform:
        """
        ds_config = cfg['datasets']['kitti']
        self.root_path = ds_config['root-path']
        self.transform = transform

        self.ds_type = ds_type
        self.seq_size = cfg['sequence-size']

        self.dataset = []
        self.length_each_drive = []
        self.length = 0
        for date, drives in ds_config[self.ds_type].items():
            for drive in drives:
                ds = KittiRawData(self.root_path, str(date), str(drive), ds_config)
                length = len(ds)
                self.length_each_drive.append(length)
                self.length += length
                self.dataset.append(ds)

        self.length_each_drive = np.array(self.length_each_drive)
        self.length -= self.seq_size
        self.num_drives = len(self.length_each_drive)

    def __len__(self):
        return self.length

    def __getitem__(self, index, test=False):
        if torch.is_tensor(index):
            index = index.tolist()

        total_length = 0
        for num_drive in range(self.num_drives):
            total_length += self.length_each_drive[num_drive] - self.seq_size
            if index <= total_length:
                break

        if num_drive > 0:
            total_len_prev_drive = np.cumsum(self.length_each_drive - self.seq_size)
            idx = index - total_len_prev_drive[num_drive - 1] - 1
        else:
            idx = index

        if not test:
            data = self.dataset[num_drive].get_data(idx, self.seq_size)
        else:
            data = idx
        return data


if __name__ == "__main__":
    from matplotlib import  pyplot as plt

    with open("../config.yaml") as f:
        cfg = yaml.safe_load(f)
    dataset = Kitti(config=cfg)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    #for i in range(len(dataset)):
    #    if i > 1107:
    #        pass
    # print(i, dataset.__getitem__(i, test=True))

    for i, data in enumerate(dataloader):
        img1, img2 = data['images']
        imu = data['imu']
        gt = data['ground-truth']

        plt.imshow(img1[:, :, 0])
        plt.show()
    print(dataset)