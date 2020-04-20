import os
import torch
import glob
import datetime as dt
import pickle
from threading import Thread

import time # for start stop calc

import numpy as np

import torch.utils.data as data

from deeplio.common import utils
from deeplio.common.laserscan import LaserScan
from deeplio.common import logger


class KittiRawData:
    """ KiitiRawData
    more or less same as pykitti with some application specific changes
    """
    MAX_DIST_HDL64 = 120.
    IMU_LENGTH = 10.25

    def __init__(self, base_path, date, drive, cfg=None, oxts_bin=False, oxts_txt=False, **kwargs):
        self.drive = drive
        self.date = date
        self.dataset = kwargs.get('dataset', 'extract')
        self.drive_full = date + '_drive_' + drive + '_' + self.dataset
        self.calib_path = os.path.join(base_path, date)
        self.data_path = os.path.join(base_path, date, self.drive_full)
        self.frames = kwargs.get('frames', None)

        if cfg is not None:
            ds_config = cfg['kitti']
            self.image_width = ds_config.get('image-width', 1024)
            self.image_height = ds_config.get('image-height', 64)
            self.fov_up = ds_config.get('fov-up', 3)
            self.fov_down = ds_config.get('fov-down', -25)
            self.seq_size = cfg.get('sequence-size', 2)
            self.max_depth = ds_config.get('max-depth', 80)
            self.min_depth = ds_config.get('min-depth', 2)
            self.inv_depth = ds_config.get('inverse-depth', False)

        # Find all the data files
        self._get_velo_files()

        #self._load_calib()
        self._load_timestamps()

        # Give priority to binary files, sicne they are laoded much faster
        if oxts_bin:
            self._load_oxts_bin()
        elif oxts_txt:
            self._get_oxt_files()
            self._load_oxts()

        self.imu_get_counter = 0

    def __len__(self):
        return len(self.velo_files)

    def get_velo(self, idx):
        """Read velodyne [x,y,z,reflectance] scan at the specified index."""
        return utils.load_velo_scan(self.velo_files[idx])

    def get_velo_image(self, idx):
        scan = LaserScan(H=self.image_height, W=self.image_width, fov_up=self.fov_up, fov_down=self.fov_down,
                         min_depth=self.min_depth, max_depth=self.max_depth, inverse_depth=self.inv_depth)
        scan.open_scan(self.velo_files[idx])
        scan.do_range_projection()
        # collect projected data and adapt ranges

        proj_xyz = scan.proj_xyz
        proj_remission = scan.proj_remission
        proj_range = scan.proj_range
        proj_range_xy = scan.proj_range_xy

        image = np.dstack((proj_xyz, proj_remission, proj_range, proj_range_xy))
        return image

    def _get_velo_files(self):
        # first try to get binary files
        self.velo_files = sorted(glob.glob(
            os.path.join(self.data_path, 'velodyne_points',
                         'data', '*.npy')))
        # if there is no bin files for velo, so the velo file are in text format
        if self.velo_files is None:
            self.velo_files = sorted(glob.glob(
                os.path.join(self.data_path, 'velodyne_points',
                             'data', '*.txt')))

        # Subselect the chosen range of frames, if any
        if self.frames is not None:
            self.velo_files = utils.subselect_files(
                self.velo_files, self.frames)
        self.velo_files = np.asarray(self.velo_files)

    def _get_oxt_files(self):
        """Find and list data files for each sensor."""
        self.oxts_files = sorted(glob.glob(
            os.path.join(self.data_path, 'oxts', 'data', '*.txt')))

        if self.frames is not None:
            self.oxts_files = utils.subselect_files(
                self.oxts_files, self.frames)
        self.oxts_files = np.asarray(self.oxts_files)

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

    def _load_oxts_bin(self):
        oxts_file = os.path.join(self.data_path, 'oxts', 'data.pkl')
        with open(oxts_file, 'rb') as f:
            self.oxts = pickle.load(f)

    def _load_oxts_lazy(self, indices):
        oxts = utils.load_oxts_packets_and_poses(self.oxts_files[indices])
        return oxts

    def calc_gt_from_oxts(self, oxts):
        transformations = [oxt.T_w_imu for oxt in oxts]

        T_w0 = transformations[0]
        R_w0 = T_w0[:3, :3]
        t_w0 = T_w0[:3, 3]
        T_w0_inv = np.identity(4)
        T_w0_inv[:3, :3] = R_w0.T
        T_w0_inv[:3, 3] = -np.matmul(R_w0.T, t_w0)

        gt_s = [np.matmul(T_w0_inv, T_0i) for T_0i in transformations]
        return gt_s


class Kitti(data.Dataset):
    # In unsynced KITTI raw dataset are some timestamp holes - i.g. 2011_10_03_27
    # e.g. there is no corresponding IMU/GPS measurment to some velodyne frames,
    # We set the min. no. so we can check and ignore these holes.
    MIN_NUM_OXT_SAMPLES = 8

    def __init__(self, config, ds_type='train', transform=None):
        """
        :param root_path:
        :param config: Configuration file including split settings
        :param transform:
        """
        ds_config_common = config['datasets']
        ds_config = ds_config_common['kitti']
        root_path = ds_config['root-path']

        self.seq_size = ds_config_common['sequence-size']
        self.channels = config['channels']

        self.ds_type = ds_type
        self.transform = transform

        self.datasets = []
        self.length_each_drive = []
        self.bins = []
        self.images = [None] * self.seq_size

        # Since we are intrested in sequence of lidar frame - e.g. multiple frame at each iteration,
        # depending on the sequence size and the current wanted index coming from pytorch dataloader
        # we must switch between each drive if not enough frames exists in that specific drive wanted from dataloader,
        # therefor we separate valid indices in each drive in bins.
        last_bin_end = -1
        for date, drives in ds_config[self.ds_type].items():
            for drive in drives:
                date = str(date).replace('-', '_')
                drive = '{0:04d}'.format(drive)
                ds = KittiRawData(root_path, date, drive, ds_config_common, oxts_bin=True)

                length = len(ds)

                bin_start = last_bin_end + 1
                bin_end = bin_start + length - 1
                self.bins.append([bin_start, bin_end])
                last_bin_end = bin_end

                self.length_each_drive.append(length)
                self.datasets.append(ds)

        self.bins = np.asarray(self.bins)
        self.length_each_drive = np.array(self.length_each_drive)

        self.length = self.bins.flatten()[-1] + 1

        self.logger = logger.global_logger

    def load_images(self, dataset, indices):
        threads = [None] * self.seq_size

        for i in range(self.seq_size):
            idx = indices[i]

            threads[i] = Thread(target=self.load_image, args=(dataset, indices[i], i))
            threads[i].start()

        for i in range(self.seq_size):
            threads[i].join()

    def load_image(self, dataset, ds_index, img_index):
        img = dataset.get_velo_image(ds_index)
        img = img[:, :, self.channels]
        self.images[img_index] = torch.from_numpy(img)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        start = time.time()

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
            self.logger.error("Error: No bins and no drive number found!")
            return None

        dataset = self.datasets[num_drive]

        # get frame indices
        len_ds = len(dataset)
        if idx <= len_ds - self.seq_size:
            indices = list(range(idx, idx + self.seq_size))
        elif (len_ds - self.seq_size) < idx < len_ds:
            indices = list(range(len_ds - self.seq_size, len_ds))
        else:
            self.logger.error("Wrong index ({}) in {}_{}".format(idx, dataset.date, dataset.drive))
            raise Exception("Wrong index ({}) in {}_{}".format(idx, dataset.date, dataset.drive))

        # Get frame timestamps
        velo_timespamps = [dataset.timestamps_velo[idx] for idx in indices]

        self.load_images(dataset, indices)
        images = self.images

        imus = []
        gts = []
        for i in range(self.seq_size - 1):
            velo_start_ts = velo_timespamps[i]
            velo_stop_ts = velo_timespamps[i+1]

            mask = ((dataset.timestamps_imu >= velo_start_ts) & (dataset.timestamps_imu < velo_stop_ts))
            oxt_indices = np.argwhere(mask).flatten()
            len_oxt = len(oxt_indices)

            if (len_oxt== 0) or (len_oxt < self.MIN_NUM_OXT_SAMPLES):
                self.logger.debug("Not enough OXT-samples: Index: {}, DS: {}_{}, len:{}, velo-timestamps: {}-{}".format(index, dataset.date, dataset.drive, len_oxt, velo_start_ts, velo_stop_ts))
                tmp_imu = np.zeros((self.seq_size - 1, self.MIN_NUM_OXT_SAMPLES, 6))
                tmp_gt = np.zeros((self.seq_size - 1, self.MIN_NUM_OXT_SAMPLES, 4, 4))
                items = [images, tmp_imu, tmp_gt]
                if self.transform:
                    items = self.transform(items)
                data = {'images': items[0], 'imus': items[1], 'gts': items[2], 'valid': False, 'meta': [0]}
                return data
            else:
                oxts_timestamps = dataset.timestamps_imu[oxt_indices]
                oxts = dataset.oxts[oxt_indices]
                imu_values = np.array([[oxt[0].ax, oxt[0].ay, oxt[0].az, oxt[0].wx, oxt[0].wy, oxt[0].wz] for oxt in oxts])
                gt = np.array([oxt[1] for oxt in oxts])

                imus.append(imu_values)
                gts.append(gt)

        items = [images, imus, gts]

        meta_data = {'index': [index], 'date': [dataset.date], 'drive': [dataset.drive], 'velo-index': [indices],
                     'velo-timestamps': [ts.timestamp() for ts in velo_timespamps],
                     'oxts-timestamps': [ts.timestamp() for ts in oxts_timestamps]}

        if self.transform:
             items = self.transform(items)

        data = {'images': items[0], 'imus': items[1], 'gts': items[2], 'valid': True, 'meta': meta_data}

        end = time.time()
        #self.logger.debug("Idx:{}, dt: {}".format(index, end - start))

        return data

    def __repr__(self):
        # printing dataset informations
        rep = "Kitti-Dataset" \
              "Type: {}, Length: {}, Seq.length: {}\n" \
              "Date\tDrive\tlength\tstart-end\n".format(self.ds_type, self.length, self.seq_size)
        seqs = ""
        for i in range(len(self.length_each_drive)):
            date = self.datasets[i].date
            drive = self.datasets[i].drive
            length = self.length_each_drive[i]
            bins = self.bins[i]
            seqs = "".join("{}{}\t{}\t{}\t{}\n".format(seqs, date, drive, length, bins))
        rep = "{}{}".format(rep,seqs)
        return rep
