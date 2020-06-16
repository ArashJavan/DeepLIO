import numpy as np
import torch

from deeplio.common.spatial import inv_SE3, rotation_matrix_to_quaternion


class DataCombiCreater(object):
    def __init__(self, combinations, device='cpu'):
        self.combinations = combinations
        self.device = device

    def process(self, data):
        res_imgs = []
        res_imu = []
        res_gt_local = []

        has_imgs = 'images' in data
        has_imu = 'imus' in data

        if has_imgs:
            n_batches = len(data['images'])
            res_imgs = data['images'].to(self.device)
        else:
            n_batches = len(data['imus'])

        for b in range(n_batches):
            gts = data['gts'][b]
            gts_local = self.process_ground_turth(gts)
            res_gt_local.append(gts_local)

            # only in deeplio and deepio we have imus
            if has_imu:
                imus = data['imus'][b]
                res = self.process_imus(imus)
                res_imu.append(res)
                #res_imu = [torch.stack(imu).to(self.device, non_blocking=True) for imu_seq in res_imu for imu in imu_seq]

        res_gt_local = torch.stack(res_gt_local).to(self.device, non_blocking=True)
        return res_imgs, res_imu, res_gt_local

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

        state_local = []
        for combi in self.combinations:
            T_i = T_global[combi[0]]
            T_i_inv = inv_SE3(T_i)
            T_ip1 = T_global[combi[1]]
            T_i_ip1 = torch.matmul(T_i_inv, T_ip1)
            dx = T_i_ip1[:3, 3].contiguous()
            dq = rotation_matrix_to_quaternion(T_i_ip1[:3, :3].contiguous())
            dv = v_global[combi[1]] - v_global[combi[0]]
            state_local.append(torch.cat([dx, dq, dv]))

        gt_local = torch.stack(state_local).to(self.device, non_blocking=True)
        return gt_local

    def __call__(self, args):
        return self.process(args)


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
        self.device = self.args.device
        self.batch_size = self.args.batch_size


config_container = None


def get_config_container():
    if config_container is None:
        raise ValueError("Config container must be created by Worker first!")
    return config_container


def build_config_container(cfg, args):
    global config_container
    config_container = ConfigContainer(cfg, args)
    return config_container


