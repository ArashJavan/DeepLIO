import time

import yaml
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from torch.utils import tensorboard

from liegroups.torch import SO3

from deeplio import datasets as ds
from deeplio.common import spatial
from deeplio.models import nets
from deeplio.models.misc import DataCombiCreater
from deeplio.models.worker import Worker, AverageMeter, ProgressMeter, worker_init_fn
from deeplio.losses import get_loss_function


class Tester(Worker):
    ACTION = "test"

    def __init__(self, args, cfg):
        super(Tester, self).__init__(args, cfg)

        args = self.args

        if self.batch_size != 1:
            self.logger.info("batch size in the testing mode should be set to one.")
            self.logger.info("setting batch size (batch-size = 1).")
            self.batch_size = 1

        if self.seq_size != 1:
            self.logger.info("setting sequence size (s=1)")
            raise ValueError("Sequence size mus tbe equal 1 in test mode.")

        # create the folder for saving training checkpoints
        self.checkpoint_dir = self.out_dir
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # preapre dataset and dataloaders
        transform = None

        self.test_dataset = ds.Kitti(config=self.cfg, transform=transform, ds_type='test')

        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                                           num_workers=self.num_workers,
                                                           shuffle=False,
                                                           worker_init_fn = worker_init_fn,
                                                           collate_fn = ds.deeplio_collate)

        self.data_permuter = DataCombiCreater(combinations=self.combinations,
                                              device=self.device)
        self.model = nets.get_model(input_shape=(self.n_channels, self.im_height_model, self.im_width_model),
                                    cfg=self.cfg, device=self.device)
        self.criterion = get_loss_function(self.cfg, args.device)

        self.has_lidar = True if self.model.lidar_feat_net is not None else False
        self.has_imu = True if self.model.imu_feat_net is not None else False

        self.tensor_writer = tensorboard.SummaryWriter(log_dir=self.runs_dir)

        # debugging and visualizing
        self.logger.print("System Training Configurations:")
        self.logger.print("args: {}".
                          format(self.args))

        self.logger.print(yaml.dump(self.cfg))
        self.logger.print(self.test_dataset)

    def run(self):
        self.is_running = True

        self.test()

        self.logger.info("Testing done!")
        self.close()

    def test(self):
        writer = self.tensor_writer
        model = self.model

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            self.logger,
            len(self.test_dataloader),
            [batch_time, losses],
            prefix='Test: ')

        seq_names = []
        last_seq = None
        curr_seq = None

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            end = time.time()

            for idx, data in enumerate(self.test_dataloader):

                # check if we can run or are we stopped
                if not self.is_running:
                    return 0

                # prepare data
                self.data_permuter(data)
                imgs = self.data_permuter.res_imgs
                normals = self.data_permuter.res_normals
                imus = self.data_permuter.res_imu
                imgs_org = self.data_permuter.res_img_org
                normals_org = self.data_permuter.res_normals_org
                gts_f2f = self.data_permuter.res_gt_f2f
                gts_f2g = self.data_permuter.res_gt_f2g
                gts_global = self.data_permuter.res_gt_global

                if torch.isnan(gts_f2f).any() or torch.isinf(gts_f2f).any():
                    raise ValueError("gt-f2f:\n{}".format(gts_f2f))
                if torch.isnan(gts_f2g).any() or torch.isinf(gts_f2g).any():
                    raise ValueError("gt-f2g:\n{}".format(gts_f2g))

                # prepare ground truth tranlational and rotational part
                gt_f2f_x = gts_f2f[:, :, 0:3]
                gt_f2f_q = gts_f2f[:, :, 3:]
                gt_f2g_x = gts_f2g[:, :, 0:3]
                gt_f2g_q = gts_f2g[:, :, 3:7]

                # compute model predictions and loss
                pred_f2f_x, pred_f2f_r = self.model([[imgs, normals], imus])
                #pred_f2f_r = spatial.normalize_quaternion(pred_f2f_r)
                pred_f2g_x, pred_f2g_r = self.se3_to_SE3(pred_f2f_x, pred_f2f_r)

                loss = self.criterion(pred_f2f_x, pred_f2f_r,
                                      pred_f2g_x, pred_f2g_r,
                                      gt_f2f_x, gt_f2f_q,
                                      gt_f2g_x, gt_f2g_q)

                # measure accuracy and record loss
                losses.update(loss.detach().item(), len(pred_f2f_x))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # get meta information for saving the odom. results
                meta = data['metas'][0]
                date, drive = meta['date'][0], meta['drive'][0]
                idx = meta['index'][0]
                velo_ts = meta['velo-timestamps']

                gt_global = data['gts'][0].cpu().numpy() # gts_global[0].cpu().numpy()
                seq_name = "{}_{}".format(date, drive)
                if seq_name not in seq_names:
                    if last_seq is not None:
                        last_seq.write_to_file()

                    curr_seq = OdomSeqRes(date, drive, output_dir=self.out_dir)
                    T_glob = np.identity(4)
                    T_glob[:3, 3] = gt_global[0, 0:3]  # t
                    T_glob[:3, :3] = gt_global[0, 3:12].reshape(3, 3) # R
                    curr_seq.add_local_prediction(velo_ts[0], 0., T_glob, T_glob)

                    # add the file name and file-pointer to the list
                    seq_names.append(seq_name)
                    losses.reset()

                # global ground truth pose
                T_glob = np.identity(4)
                T_glob[:3, 3] = gt_global[1, 0:3]  # t
                T_glob[:3, :3] = gt_global[1, 3:12].reshape(3, 3) # R

                gt_x = gt_f2f_x.detach().cpu().squeeze()
                gt_q = gt_f2f_q.detach().cpu().squeeze()
                pred_f2f_x = pred_f2f_x.detach().cpu().squeeze()
                pred_f2f_r = pred_f2f_r.detach().cpu().squeeze()

                if not np.all(data['valids']):
                    pred_f2f_x = gt_x
                    pred_f2f_r = gt_q

                T_local = np.identity(4)

                # tranlation
                T_local[:3, 3] = pred_f2f_x.numpy() #  gt_x.numpy()
                #T_local[:3, 3] = gt_x.numpy()

                if self.args.param == 'xq':
                    T_local[:3, 3] = pred_f2f_x.numpy()
                    T_local[:3, :3] = SO3.exp(pred_f2f_r).as_matrix().numpy() # spatial.quaternion_to_rotation_matrix(pred_f2f_r).numpy()
                elif self.args.param == 'x':
                    T_local[:3, 3] = pred_f2f_x.numpy()
                    T_local[:3, :3] = SO3.exp(gt_q).as_matrix().numpy() # spatial.quaternion_to_rotation_matrix(gt_q).numpy()
                elif self.args.param == 'q':
                    T_local[:3, 3] = gt_x.numpy()
                    T_local[:3, :3] = SO3.exp(pred_f2f_r).as_matrix().numpy()
                else:
                    T_local[:3, 3] = gt_x.numpy()
                    T_local[:3, :3] = spatial.quaternion_to_rotation_matrix(gt_q).numpy()

                curr_seq.add_local_prediction(velo_ts[1], losses.avg, T_local, T_glob)

                last_seq = curr_seq
                if idx % self.args.print_freq == 0:
                    progress.display(idx)
                    # update tensorboard
                    step_val = idx
                    self.tensor_writer.add_scalar\
                        ("Loss test", losses.avg, step_val)
                    self.tensor_writer.flush()

        if curr_seq is not None:
            curr_seq.write_to_file()

    def se3_to_SE3(self, f2f_x, f2f_r):
        batch_size, seq_size, _ = f2f_x.shape

        f2g_q = torch.zeros((batch_size, seq_size, 4), dtype=f2f_x.dtype, device=f2f_x.device)
        f2g_x = torch.zeros((batch_size, seq_size, 3), dtype=f2f_x.dtype, device=f2f_x.device)

        for b in range(batch_size):
            R_prev = torch.zeros((3, 3), dtype=f2f_x.dtype, device=f2f_x.device)
            R_prev[:] = torch.eye(3, dtype=f2f_x.dtype, device=f2f_x.device)
            t_prev = torch.zeros((3), dtype=f2f_x.dtype, device=f2f_x.device)

            for s in range(seq_size):
                t_cur = f2f_x[b, s]
                #q_cur = spatial.euler_to_rotation_matrix (f2f_r[b, s])
                w_cur = f2f_r[b, s]
                R_cur = SO3.exp(w_cur).as_matrix() # spatial.quaternion_to_rotation_matrix(q_cur)

                if not torch.isclose(torch.det(R_cur), torch.FloatTensor([1.]).to(self.device)).all():
                    raise ValueError("Det error:\nR\n{}\nq:\n{}".format(R_cur, w_cur))

                t_prev = torch.matmul(R_prev, t_cur) + t_prev
                R_prev = torch.matmul(R_prev, R_cur)

                if not torch.isclose(torch.det(R_prev), torch.FloatTensor([1.]).to(self.device)).all():
                    raise ValueError("Det error:\nR\n{}".format(R_prev))

                f2g_q[b, s] = spatial.rotation_matrix_to_quaternion(R_prev)
                f2g_x[b, s] = t_prev
        return f2g_x, f2g_q

    def eval_model_and_loss(self, imgs,imus, gt_local_x, gt_local_q):
        pred_x = None
        pred_q = None
        loss = None
        return pred_x, pred_q, loss


class TesterDeepLIO(Tester):
    ACTION = "test_deeplio"

    def eval_model_and_loss(self, imgs, imus, gt_local_x, gt_local_q):
        # compute model predictions and loss
        pred_x, pred_q = self.model([imgs, imus])

        # rotation
        if self.args.param == 'xq':
            loss = self.criterion(pred_x, pred_q, gt_local_x, gt_local_q)
        elif self.args.param == 'x':
            loss = self.criterion(pred_x, gt_local_q, gt_local_x, gt_local_q)
        elif self.args.param == 'q':
            loss = self.criterion(gt_local_x, pred_q, gt_local_x, gt_local_q)
        else:
            loss = self.criterion(gt_local_x, gt_local_q, gt_local_x, gt_local_q)
        return pred_x, pred_q, loss


class OdomSeqRes:
    def __init__(self, date, drive, output_dir="."):
        self.date = date
        self.drive = drive
        self.T_local_pred = []
        self.T_global = []
        self.timestamps = []
        self.loss = []
        self.out_dir = output_dir

    def add_local_prediction(self, timestamp, loss, T_local, T_gt_global):
        self.timestamps.append(timestamp)
        self.loss.append(loss)
        self.T_local_pred.append(T_local)
        self.T_global.append(T_gt_global)

    def write_to_file(self):
        T_glob_pred = []
        T_0i = self.T_local_pred[0]
        T_glob_pred.append(T_0i)
        for i in range(1, len(self.T_local_pred)):
            T_i = self.T_local_pred[i]
            T = np.matmul(T_0i, T_i)
            T_glob_pred.append(T)
            T_0i = np.copy(T)

        self.T_global = np.array(self.T_global)
        self.T_local_pred = np.array(self.T_local_pred)
        T_glob_pred = np.array(T_glob_pred)

        self.timestamps = np.asarray(self.timestamps).reshape(-1, 1)
        self.loss = np.asarray(self.loss).reshape(-1, 1)

        res = np.hstack((self.timestamps,
                         T_glob_pred.reshape(len(T_glob_pred), -1),
                         self.T_global.reshape(len(self.T_global), -1),
                         self.loss))

        fname = "{}/{}_{}.csv".format(self.out_dir, self.date, self.drive)
        np.savetxt(fname, res, fmt='%.5f', delimiter=',')

        fname = "{}/{}_{}.png".format(self.out_dir, self.date, self.drive)
        plt.figure()

        plt.plot(self.T_global[:, 0, 3], self.T_global[:, 1, 3], alpha=0.5, linewidth=1, label="GT")
        plt.scatter(self.T_global[:, 0, 3], self.T_global[:, 1, 3], alpha=0.7, s=0.5)

        plt.plot(T_glob_pred[:, 0, 3], T_glob_pred[:, 1, 3], alpha=0.5, linewidth=1, label="DeepLIO")
        plt.scatter(T_glob_pred[:, 0, 3], T_glob_pred[:, 1, 3], alpha=0.7, s=0.5)

        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.grid()
        plt.legend()
        plt.savefig(fname, figsize=(50, 50), dpi=600)





