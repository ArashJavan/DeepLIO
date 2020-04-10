import random
import torch

class PostProcessSiameseData(object):
    def __init__(self, seq_size=2, batch_size=1):
        self.seq_size = seq_size
        self.batch_size = batch_size
        self.combinations = []

    def process(self, data):
        images = data['images']
        oxts = [data['imus'],data['gts']]

        res_im_0 = []
        res_im_1 = []
        res_imu = []
        res_gt = []

        for i in range(self.batch_size):
            imgs = images[i]
            imus = oxts[0][i]
            gts = oxts[1][i]

            combinations = [[0, x] for x in range(self.seq_size) if x > 0]
            # we do not want that the network memorizes an specific combination pattern
            random.shuffle(combinations)
            self.combinations.extend(combinations)

            T_gt = self.calc_trans_mat_combis(gts, combinations)
            res_gt.extend(T_gt)

            for j, combi in enumerate(combinations):
                idx_0 = combi[0]
                idx_1 = combi[1]

                res_im_0.append(imgs[idx_0])
                res_im_1.append(imgs[idx_1])

                # Determining IMU measurment btw. each combination
                max_idx = max(combi)
                min_idx = min(combi)
                imu_tmp = []
                for k in range(min_idx, max_idx):
                    imu_tmp.extend(imus[k])
                res_imu.append(imu_tmp)

        res_im_0 = torch.stack(res_im_0)
        res_im_1 = torch.stack(res_im_1)
        res_gt = torch.stack(res_gt)
        res_imu = [torch.stack(imu) for imu in res_imu]
        return res_im_0, res_im_1, res_gt, res_imu

    def calc_trans_mat_combis(self, transformations, combinations):
        T_global = []
        for i in range(self.seq_size):
            if i == 0:
                T_global.append(transformations[i, 0])
            else:
                T_global.append(transformations[i-1, -1])
        T = []
        for combi in combinations:
            T_i = T_global[combi[0]]
            T_i_inv = self.inv_SE3(T_i)
            T_ip1 = T_global[combi[1]]
            T_i_ip1 = torch.matmul(T_i_inv, T_ip1)
            T.append(T_i_ip1)
        return T

    def inv_SE3(self, T):
        R = T[:3, :3]
        t = T[:3, 3]
        T_inv = torch.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -torch.matmul(R.T, t)
        return  T_inv

    def __call__(self, args):
        return self.process(args)
