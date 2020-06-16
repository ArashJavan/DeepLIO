import datetime as dt
import glob
import os
import pickle
import time  # for start stop calc
from threading import Thread

import numpy as np
import torch
import torch.utils.data as data

from deeplio.common import utils, logger
from deeplio.common.laserscan import LaserScan


class KittiRawData:
    """ KiitiRawData
    more or less same as pykitti with some application specific changes
    """
    def __init__(self, base_path_sync, base_path_unsync, date, drive,
                 cfg=None, oxts_bin=False, oxts_txt=False, max_points=150000, **kwargs):
        self.drive = drive
        self.date = date

        self.dataset_sync = 'sync'
        self.dataset_unsync = 'extract'
        self.drive_full_sync = date + '_drive_' + drive + '_' + self.dataset_sync
        self.drive_full_unsync = date + '_drive_' + drive + '_' + self.dataset_unsync

        self.calib_path = os.path.join(base_path_sync, date)

        self.data_path_sync = os.path.join(base_path_sync, date, self.drive_full_sync)
        self.data_path_unsync = os.path.join(base_path_unsync, date, self.drive_full_unsync)

        self.frames = kwargs.get('frames', None)
        self.max_points = max_points

        if cfg is not None:
            ds_config = cfg['kitti']
            self.image_width = ds_config.get('image-width', 1024)
            self.image_height = ds_config.get('image-height', 64)
            self.fov_up = ds_config.get('fov-up', 3)
            self.fov_down = ds_config.get('fov-down', -25)
            self.max_depth = ds_config.get('max-depth', 80)
            self.min_depth = ds_config.get('min-depth', 2)

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
        scan = LaserScan(project=True, H=self.image_height, W=self.image_width, fov_up=self.fov_up, fov_down=self.fov_down,
                         min_depth=self.min_depth, max_depth=self.max_depth)
        scan.open_scan(self.velo_files[idx])

        # get projected data
        proj_xyz = scan.proj_xyz
        proj_remission = scan.proj_remission
        proj_range = scan.proj_range

        image = np.dstack((proj_xyz, proj_range, proj_remission))
        return image

    def get_imu_values(self, idx):
        oxt = self.oxts_unsync[idx]
        imu_values = np.array([[oxt[0].ax, oxt[0].ay, oxt[0].az,
                                oxt[0].wx, oxt[0].wy, oxt[0].wz]
                               for oxt in oxt], dtype=np.float)
        return imu_values

    def _get_velo_files(self):
        # first try to get binary files
        self.velo_files = sorted(glob.glob(
            os.path.join(self.data_path_sync, 'velodyne_points',
                         'data', '*.bin')))
        # if there is no bin files for velo, so the velo file are in text format
        if self.velo_files is None:
            self.velo_files = sorted(glob.glob(
                os.path.join(self.data_path_unsync, 'velodyne_points',
                             'data', '*.txt')))

        # Subselect the chosen range of frames, if any
        if self.frames is not None:
            self.velo_files = utils.subselect_files(
                self.velo_files, self.frames)
        self.velo_files = np.asarray(self.velo_files)

    def _get_oxt_files(self):
        """Find and list data files for each sensor."""
        self.oxts_files_sync = sorted(glob.glob(
            os.path.join(self.data_path_sync, 'oxts', 'data', '*.txt')))

        if self.frames is not None:
            self.oxts_files_sync = utils.subselect_files(
                self.oxts_files_sync, self.frames)
        self.oxts_files_sync = np.asarray(self.oxts_files_sync)

        self.oxts_files_unsync = sorted(glob.glob(
            os.path.join(self.data_path_unsync, 'oxts', 'data', '*.txt')))

        if self.frames is not None:
            self.oxts_files_unsync = utils.subselect_files(
                self.oxts_files_unsync, self.frames)
        self.oxts_files_unsync = np.asarray(self.oxts_files_unsync)

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
        timestamp_file_unsync = os.path.join(self.data_path_unsync, 'oxts', 'timestamps.txt')
        timestamp_file_velo = os.path.join(self.data_path_sync, 'velodyne_points', 'timestamps.txt')
        timestamp_file_sync = os.path.join(self.data_path_sync, 'oxts', 'timestamps.txt')

        # Read and parse the timestamps
        self.timestamps_unsync = []
        with open(timestamp_file_unsync, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                self.timestamps_unsync.append(t)
        self.timestamps_unsync = np.array(self.timestamps_unsync)

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

        # Read and parse the timestamps
        self.timestamps_sync = []
        with open(timestamp_file_sync, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                self.timestamps_sync.append(t)
        self.timestamps_sync = np.array(self.timestamps_sync)

    def _load_oxts(self):
        """Load OXTS data from file."""
        self.oxts_sync = np.array(utils.load_oxts_packets_and_poses(self.oxts_files_sync))
        self.oxts_unsync = np.array(utils.load_oxts_packets_and_poses(self.oxts_files_unsync))

    def _load_oxts_bin(self):
        oxts_file_sync = os.path.join(self.data_path_sync, 'oxts', 'data.pkl')
        with open(oxts_file_sync, 'rb') as f:
            self.oxts_sync = pickle.load(f)

        oxts_file_unsync = os.path.join(self.data_path_unsync, 'oxts', 'data.pkl')
        with open(oxts_file_unsync, 'rb') as f:
            self.oxts_unsync = pickle.load(f)

    def _load_oxts_lazy(self, indices):
        oxts = utils.load_oxts_packets_and_poses(self.oxts_files_sync[indices])
        return oxts


class Kitti(data.Dataset):
    # In unsynced KITTI raw dataset are some timestamp holes - i.g. 2011_10_03_27
    # e.g. there is no corresponding IMU/GPS measurment to some velodyne frames,
    # We set the min. no. so we can check and ignore these holes.
    MIN_NUM_OXT_SAMPLES = 8

    def __init__(self, config, ds_type='train', transform=None, has_imu=True, has_lidar=True):
        """
        :param root_path:
        :param config: Configuration file including split settings
        :param transform:
        """
        ds_config_common = config['datasets']
        ds_config = ds_config_common['kitti']
        self._seq_size = ds_config_common['sequence-size'] # Increment because we need always one sample more
        self.internal_seq_size = self.seq_size + 1
        self.inv_depth = ds_config.get('inverse-depth', False)
        self.mean_img = ds_config['mean-image']
        self.std_img = ds_config['std-image']
        self.mean_imu = ds_config['mean-imu']
        self.std_imu = ds_config['std-imu']
        self.channels = config['channels']

        self.has_imu = has_imu
        self.has_lidar = has_lidar

        crop_factors = ds_config.get('crop-factors', [0, 0])
        self.crop_top = crop_factors[0]
        self.crop_left = crop_factors[1]

        self.ds_type = ds_type
        self.transform = transform

        self.datasets = []
        self.length_each_drive = []
        self.bins = []
        self.images = [None] * self.internal_seq_size

        root_path_sync = ds_config['root-path-sync']
        root_path_unsync = ds_config['root-path-unsync']

        # Since we are intrested in sequence of lidar frame - e.g. multiple frame at each iteration,
        # depending on the sequence size and the current wanted index coming from pytorch dataloader
        # we must switch between each drive if not enough frames exists in that specific drive wanted from dataloader,
        # therefor we separate valid indices in each drive in bins.
        last_bin_end = -1
        for date, drives in ds_config[self.ds_type].items():
            for drive in drives:
                date = str(date).replace('-', '_')
                drive = '{0:04d}'.format(drive)
                ds = KittiRawData(root_path_sync, root_path_unsync, date, drive, ds_config_common, oxts_bin=True)

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

        self.logger = logger.get_app_logger()

    @property
    def seq_size(self):
        return self._seq_size

    @seq_size.setter
    def seq_size(self, size):
        self.seq_size = size
        self.internal_seq_size = size + 1

    def load_ground_truth(self, dataset, indices):
        gts_alls = dataset.oxts_sync[indices]
        gts = []
        for gt in gts_alls:
            T = gt[1]
            x = T[:3, 3].flatten()
            R = T[:3, :3].flatten()
            v = np.array([gt[0].vf, gt[0].vl, gt[0].vu])
            gts.append(np.hstack((x, R, v)))
        return np.array(gts)

    def load_images(self, dataset, indices):
        threads = [None] * self.internal_seq_size

        for i in range(self.internal_seq_size):
            threads[i] = Thread(target=self.load_image, args=(dataset, indices[i], i))
            threads[i].start()

        for i in range(self.internal_seq_size):
            threads[i].join()

    def load_image(self, dataset, ds_index, img_index):
        img = dataset.get_velo_image(ds_index)
        self.images[img_index] = img

    def load_imus(self, dataset, velo_timestamps):
        imus = []
        valids = []
        for i in range(self.internal_seq_size - 1):
            velo_start_ts = velo_timestamps[i]
            velo_stop_ts = velo_timestamps[i+1]

            mask = ((dataset.timestamps_unsync >= velo_start_ts) & (dataset.timestamps_unsync < velo_stop_ts))
            oxt_indices = np.argwhere(mask).flatten()
            len_oxt = len(oxt_indices)

            if (len_oxt== 0) or (len_oxt < self.MIN_NUM_OXT_SAMPLES):
                self.logger.debug("Not enough OXT-samples: DS: {}_{}, len:{}, velo-timestamps: {}-{}".
                                  format(dataset.date, dataset.drive, len_oxt, velo_start_ts, velo_stop_ts))
                imu_values = np.zeros((self.MIN_NUM_OXT_SAMPLES, 6), dtype=np.float)
                valids.append(False)
            else:
                oxts = dataset.oxts_unsync[oxt_indices]
                imu_values = np.array([[oxt[0].ax, oxt[0].ay, oxt[0].az,
                                        oxt[0].wx, oxt[0].wy, oxt[0].wz]
                                       for oxt in oxts], dtype=np.float)
                valids.append(True)
            imus.append(imu_values)
        return imus, valids

    def transform_images(self):
        imgs_org = torch.stack([torch.from_numpy(im.transpose(2, 0, 1)) for im in self.images])

        ct, cl = self.crop_top, self.crop_left
        mean = torch.as_tensor(self.mean_img)
        std = torch.as_tensor(self.std_img)
        imgs_normalized = [torch.from_numpy(img[ct:-ct, cl:-cl, :].transpose(2, 0, 1)) for img in self.images]
        imgs_normalized = torch.stack(imgs_normalized)
        if self.inv_depth:
            im_depth = imgs_normalized[:, 3]
            im_depth[im_depth > 0.] = 1 / im_depth[im_depth > 0.]
        imgs_normalized.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        imgs_normalized = imgs_normalized[:, self.channels]
        return imgs_org, imgs_normalized

    def transform_imus(self, imus):
        imus_norm = [torch.from_numpy((imu - self.mean_imu) / self.std_imu).type(torch.float32) for imu in imus]
        return imus_norm

    def get_dataset_and_index(self, index):
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
        if idx <= len_ds - self.internal_seq_size:
            indices = list(range(idx, idx + self.internal_seq_size))
        elif (len_ds - self.internal_seq_size) < idx < len_ds:
            indices = list(range(len_ds - self.internal_seq_size, len_ds))
        else:
            self.logger.error("Wrong index ({}) in {}_{}".format(idx, dataset.date, dataset.drive))
            raise Exception("Wrong index ({}) in {}_{}".format(idx, dataset.date, dataset.drive))
        return dataset, indices

    def create_imu_data(self, dataset, indices, velo_timespamps):
        # load and transform imus
        imus, valids = self.load_imus(dataset, velo_timespamps)
        imus = self.transform_imus(imus)
        data = {'imus': imus, 'valids': valids}
        return data

    def create_lidar_data(self, dataset, indices, velo_timespamps):
        # load and transform images
        self.load_images(dataset, indices)
        org_images, proc_images = self.transform_images()
        data = {'images': proc_images, 'untrans-images': org_images}
        return data

    def create_data_deeplio(self, dataset, indices, velo_timespamps):
        imu_data = self.create_imu_data(dataset, indices, velo_timespamps)
        img_data = self.create_lidar_data(dataset, indices, velo_timespamps)
        data = {**imu_data, **img_data}
        return data

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        start = time.time()
        dataset, indices = self.get_dataset_and_index(index)

        # Get frame timestamps
        velo_timespamps = [dataset.timestamps_velo[idx] for idx in indices]

        lidar_data = {}
        if self.has_lidar:
            lidar_data = self.create_lidar_data(dataset, indices, velo_timespamps)

        imu_data = {}
        if self.has_imu:
            imu_data = self.create_imu_data(dataset, indices, velo_timespamps)

        arch_data = {**imu_data, **lidar_data}

        # load and transform ground truth
        gts = torch.from_numpy(self.load_ground_truth(dataset, indices)).type(torch.float32)

        meta_data = {'index': [index], 'date': [dataset.date], 'drive': [dataset.drive], 'velo-index': [indices],
                     'velo-timestamps': [ts.timestamp() for ts in velo_timespamps]}

        data = {'data': arch_data, 'gts': gts, 'meta': meta_data}

        end = time.time()

        #print(index, dataset.data_path, indices)
        #self.logger.debug("Idx:{}, dt: {}".format(index, end - start))
        return data

    def __repr__(self):
        # printing dataset informations
        rep = "Kitti-Dataset" \
              "Type: {}, Length: {}, Seq.length: {}\n" \
              "Date\tDrive\tlength\tstart-end\n".format(self.ds_type, self.length, self.internal_seq_size)
        seqs = ""
        for i in range(len(self.length_each_drive)):
            date = self.datasets[i].date
            drive = self.datasets[i].drive
            length = self.length_each_drive[i]
            bins = self.bins[i]
            seqs = "".join("{}{}\t{}\t{}\t{}\n".format(seqs, date, drive, length, bins))
        rep = "{}{}".format(rep,seqs)
        return rep
