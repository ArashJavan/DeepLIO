import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler

from liegroups.torch import SO3, utils

from deeplio.common.spatial import *

class DataCombiCreater(object):
    def __init__(self, combinations, device='cpu'):
        self.combinations = combinations
        self.device = device
        self.seq_size = get_config_container().seq_size

        self.res_imgs = None
        self.res_img_org = None
        self.res_normals = None
        self.res_normals_org = None
        self.res_imu = None
        self.res_gt_f2f = None
        self.res_gt_f2g = None
        self.res_gt_global = None

    def process(self, data):
        imgs = []
        imgs_org = []
        normals = []
        normals_org = []
        imus = []
        gt_f2f = []
        gt_f2g = []

        has_imgs = 'images' in data
        has_imu = 'imus' in data

        if has_imgs:
            imgs, normals = self.process_images(data['images'].to(self.device))
            imgs_org, normals_org = self.process_images(data['untrans-images'].to(self.device))

        # only in deeplio and deepio we have imus
        if has_imu:
            imus = data['imus'].to(self.device)

        n_batches = len(data['gts'])
        for b in range(n_batches):
            gts = data['gts'][b]
            gts_local, gts_glob = self.process_ground_turth(gts)
            gt_f2f.append(gts_local)
            gt_f2g.append(gts_glob)


        gt_global = data['gts'].to(self.device)
        gt_f2f = torch.stack(gt_f2f).to(self.device, non_blocking=True)
        gt_f2g = torch.stack(gt_f2g).to(self.device, non_blocking=True)

        self.res_imgs = imgs
        self.res_img_org = imgs_org
        self.res_normals = normals
        self.res_normals_org = normals_org
        self.res_imu = imus
        self.res_gt_f2f = gt_f2f
        self.res_gt_f2g = gt_f2g
        self.res_gt_global = gt_global

    def process_images(self, imgs):
        imgs = imgs[:, self.combinations] # dim=[BxSxTxCxHxW]
        xyz_imgs = imgs[:, :, :, 0:3]
        normal_imgs = imgs[:, :, :, 3:].contiguous()
        return xyz_imgs, normal_imgs

    def process_imus(self, imus):
        imu_seq = []
        for j, combi in enumerate(self.combinations):
            # Determining IMU measurment btw. each combination
            max_idx = max(combi)
            min_idx = min(combi)
            imu_tmp = []
            for k in range(min_idx, max_idx):
                imu_tmp.extend(imus[k])
            imu_seq.append(torch.stack(imu_tmp).to(self.device))
        return imu_seq

    def process_ground_turth(self, gts):
        T_global = []
        v_global = []

        for gt in gts:
            t = gt[0:3]
            R = gt[3:12].reshape(3, 3)
            T = torch.eye(4)
            T[:3, 3] = t
            T[:3, :3] = R
            T_global.append(T)
            v = gt[12:]
            v_global.append(v)

        state_f2f = []
        for combi in self.combinations:
            T_i = T_global[combi[0]]
            T_i_inv = inv_SE3(T_i)
            T_ip1 = T_global[combi[1]]
            T_i_ip1 = torch.matmul(T_i_inv, T_ip1)
            dx = T_i_ip1[:3, 3].contiguous()
            dq = SO3.from_matrix(T_i_ip1[:3, :3], normalize=False).log() # rotation_matrix_exp_to_log(T_i_ip1[:3, :3].unsqueeze(0).contiguous()).squeeze()

            if torch.isnan(dq).any() or torch.isinf(dq).any():
                raise ValueError("gt-f2f:\n{}".format(dq))

            #dq = quaternion_exp_to_log(dq).squeeze()
            state_f2f.append(torch.cat([dx, dq]))

        T_0 = T_global[0]
        T_0_inv = inv_SE3(T_0)
        state_f2g = []
        for combi in self.combinations:
            T_ip1 = T_global[combi[1]]
            T_i_ip1 = torch.matmul(T_0_inv, T_ip1)
            dx = T_i_ip1[:3, 3].contiguous()
            dq = SO3.from_matrix(T_i_ip1[:3, :3]).to_quaternion()
            state_f2g.append(torch.cat([dx, dq]))

        gt_f2f = torch.stack(state_f2f).to(self.device, non_blocking=True)
        gt_f2g = torch.stack(state_f2g).to(self.device, non_blocking=True)

        return gt_f2f, gt_f2g

    def __call__(self, args):
        return self.process(args)


class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """
    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0, last_epoch=-1):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        lr = [(base_lr - self.end_learning_rate) *
              ((1 - self.last_epoch / self.max_decay_steps) ** (self.power)) +
              self.end_learning_rate for base_lr in self.base_lrs]

        return lr

    def _get_closed_form_lr(self):
        if self.last_epoch > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) *
                ((1 - self.last_epoch / self.max_decay_steps) ** (self.power)) +
                self.end_learning_rate for base_lr in self.base_lrs]


class ConfigContainer:
    """Class for holding config informations which can be used by NN-models
    """
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.ds_cfg = self.cfg['datasets']
        self.curr_dataset_cfg = self.cfg['datasets'][self.cfg['current-dataset']]
        self.combinations = np.array(self.ds_cfg['combinations'])
        self.seq_size = len(self.combinations) # self.ds_cfg['sequence-size']
        self.timestamps = len(self.combinations[0])
        self.device = self.args.device
        self.batch_size = self.args.batch_size
        self.seq_size_data = self.ds_cfg['sequence-size']


config_container = None


def get_config_container():
    if config_container is None:
        raise ValueError("Config container must be created by Worker first!")
    return config_container


def build_config_container(cfg, args):
    global config_container
    config_container = ConfigContainer(cfg, args)
    return config_container

