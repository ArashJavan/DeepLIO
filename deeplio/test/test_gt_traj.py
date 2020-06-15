import argparse
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import yaml
# matplotlib.use('Agg')
from matplotlib import pyplot as plt

dname = os.path.abspath(os.path.dirname(__file__))
content_dir = os.path.abspath("{}/..".format(dname))
sys.path.append(dname)
sys.path.append(content_dir)


from deeplio import datasets as ds
from deeplio.common import spatial
from deeplio.models.misc import DataCombiCreater
from deeplio.models.worker import Worker, AverageMeter, ProgressMeter, worker_init_fn


class TestTraj(Worker):
    ACTION = "test_traj"

    def __init__(self, args, cfg):
        super(TestTraj, self).__init__(args, cfg)
        args = self.args

        if self.seq_size > 1:
            self.logger.info("sequence size in the testing mode should be set to two.")
            #raise ValueError("sequence size in the testing mode should be set to two.")

        if self.batch_size != 1:
            self.logger.info("batch size in the testing mode should be set to one.")
            self.logger.info("setting batch size (batch-size = 1).")
            self.batch_size = 1

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
        batch_time = AverageMeter('Time', ':6.3f')
        progress = ProgressMeter(
            self.logger,
            len(self.test_dataloader),
            [batch_time],
            prefix='Test: ')

        end = time.time()

        global_traj = []
        f2f_traj = []
        for idx, data in enumerate(self.test_dataloader):
            # check if we can run or are we stopped
            if not self.is_running:
                return 0

            imgs, imus, gts_local = self.data_permuter(data)
            gts_local.squeeze_()  # remove empty dimensions
            x_local = gts_local[0:3]
            q_local = gts_local[3:7]
            R_local = spatial.quaternion_to_rotation_matrix(q_local)
            T_local = torch.eye(4)
            T_local[:3, 3] = x_local
            T_local[:3, :3] = R_local
            f2f_traj.append(T_local)

            gt_first = data['gts'][0, 0]
            x_global0 = gt_first[:3]
            R_global0 = gt_first[3:12].reshape(3, 3)
            T_global0 = torch.eye(4)
            T_global0[:3, 3] = x_global0
            T_global0[:3, :3] = R_global0
            global_traj.append(T_global0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % self.args.print_freq == 0:
                progress.display(idx)
                # update tensorboard

            #if idx > 50:
            #    break

        #glob_traj = [g for gts in global_traj for g in gts]
        glob_traj = torch.stack(global_traj)

        local_traj = []
        local_traj.append(glob_traj[0])
        for i in range(1, len(glob_traj)):
            T_0i =  glob_traj[i-1]
            T_0i_inv = spatial.inv_SE3(T_0i)
            T_0ip1 = glob_traj[i]
            T_i_ip1 = torch.matmul(T_0i_inv, T_0ip1)
            local_traj.append(T_i_ip1)

        glob_traj_pred = []
        diffs = []
        T_0i = local_traj[0]
        for i in range(1, len(local_traj)):
            T_ip1 = local_traj[i]
            T = torch.matmul(T_0i, T_ip1)
            T_0i = T
            glob_traj_pred.append(T)
            diffs.append(torch.max(torch.abs(T - glob_traj[i])))
        print(torch.stack(diffs).max())

        #for i in range(len(local_traj)):
        #    print(torch.max(f2f_traj[i] - local_traj[i+1]))

        glob_traj_pred_2 = []
        diffs = []
        T_0i = glob_traj[0]
        for i in range(0, len(f2f_traj)):
            T_ip1 = f2f_traj[i]
            T = torch.matmul(T_0i, T_ip1)
            T_0i = T
            glob_traj_pred_2.append(T)
            diffs.append(torch.max(torch.abs(T - glob_traj[i])))

        print(torch.stack(diffs).max())

        glob_traj_pred = torch.stack(glob_traj_pred)
        glob_traj_pred_2 = torch.stack(glob_traj_pred_2)

        plt.plot(glob_traj[:, 0, 3], glob_traj[:, 1, 3], alpha=0.5, label="GT")
        plt.scatter(glob_traj[:, 0, 3], glob_traj[:, 1, 3], s=10, alpha=0.5)

        plt.plot(glob_traj_pred[:, 0, 3], glob_traj_pred[:, 1, 3], alpha=0.5, label="pred")
        plt.scatter(glob_traj_pred[:, 0, 3], glob_traj_pred[:, 1, 3], s=10, alpha=0.5)

        plt.plot(glob_traj_pred_2[:, 0, 3], glob_traj_pred_2[:, 1, 3], alpha=0.5, label="pred")
        plt.scatter(glob_traj_pred_2[:, 0, 3], glob_traj_pred_2[:, 1, 3], s=10, alpha=0.5)

        plt.legend()
        plt.grid('true')
        plt.show()


            # prepare data
        #     imgs_0, imgs_1, imgs_untrans_0, imgs_untrans_1, gts, imus = self.post_processor(data)
        #
        #     # send data to device
        #     imgs_0 = imgs_0.to(self.device, non_blocking=True)
        #     imgs_1 = imgs_1.to(self.device, non_blocking=True)
        #     gts = gts.to(self.device, non_blocking=True)
        #     imus = [imu.to(self.device, non_blocking=True) for imu in imus]
        #
        #     # prepare ground truth tranlational and rotational part
        #     gt_pos = gts[:, :3, 3].contiguous()
        #     gt_rot = spatial.rotation_matrix_to_quaternion(gts[:, :3, :3].contiguous())
        #
        #     # measure elapsed time
        #     batch_time.update(time.time() - end)
        #     end = time.time()
        #
        #     # get meta information for saving the odom. results
        #     meta = data['metas'][0]
        #     date, drive = meta['date'][0], meta['drive'][0]
        #     idx = meta['index'][0]
        #     velo_idx = meta['velo-index'][0]
        #     oxts_ts = meta['oxts-timestamps']
        #     velo_ts = meta['velo-timestamps']
        #
        #     gt_global = data['gts'][0][0].detach().cpu().numpy()
        #     seq_name = "{}_{}".format(date, drive)
        #     if seq_name not in seq_names:
        #         if last_seq is not None:
        #             last_seq.write_to_file()
        #
        #         curr_seq = OdomSeqRes(date, drive, output_dir=self.out_dir)
        #         curr_seq.add_local_prediction(velo_ts[0], 0., gt_global[0], gt_global[0])
        #
        #         # add the file name and file-pointer to the list
        #         seq_names.append(seq_name)
        #
        #     gt_local = gt_pos.detach().cpu().squeeze()
        #     gt_rot = gt_rot.detach().cpu().squeeze()
        #
        #     T_local = np.identity(4)
        #     T_local[:3, 3] = gt_local.numpy() # pred_x.detach().cpu().squeeze().numpy()
        #     T_local[:3, :3] = spatial.quaternion_to_rotation_matrix(gt_rot).numpy() # spatial.quaternion_to_rotation_matrix(pred_q.detach().cpu().squeeze()).numpy()
        #     T_local = gts.detach().cpu().squeeze().numpy()
        #     curr_seq.add_local_prediction(velo_ts[1], 0., T_local, gt_global[-1])
        #
        #     last_seq = curr_seq
        #     f idx % self.args.print_freq == 0:i
        #         progress.display(idx)
        #         # update tensorboard
        #         step_val = idx
        #
        #     if idx > 120:
        #         break
        #
        # if curr_seq is not None:
        #     curr_seq.write_to_file()


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
                         self.T_local_pred.reshape(len(self.T_local_pred), -1),
                         T_glob_pred.reshape(len(T_glob_pred), -1),
                         self.T_global.reshape(len(self.T_global), -1),
                         self.loss))

        fname = "{}/{}_{}.csv".format(self.out_dir, self.date, self.drive)
        np.savetxt(fname, res, fmt='%.5f', delimiter=',')

        fname = "{}/{}_{}.png".format(self.out_dir, self.date, self.drive)
        plt.plot(self.T_global[:, 0, 3], self.T_global[:, 1, 3], alpha=0.5, label="GT")
        plt.scatter(self.T_global[:, 0, 3], self.T_global[:, 1, 3], alpha=0.5, s=0.5)
        plt.plot(T_glob_pred[:, 0, 3], T_glob_pred[:, 1, 3], alpha=0.5, label="Predicted")
        plt.scatter(T_glob_pred[:, 0, 3], T_glob_pred[:, 1, 3], alpha=0.5, s=0.5)

        plt.grid()
        plt.legend()
        plt.savefig(fname, figsize=(50, 50), dpi=300)


def signal_handler(signum, frame):
    if trainer is not None:
        trainer.close()
    sys.stdout.flush()
    sys.exit(1)

trainer = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepLIO Training')

    # Hyper Params
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N',
                        help='mini-batch size (default: 1), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--model', default='', type=str, metavar='PATH',
                        help='path to model checkpoint')
    parser.add_argument('-c', '--config', default="./config.yaml", help='Path to configuration file')
    parser.add_argument('-d', '--debug', default=False, help='debug logging', action='store_true', dest='debug')
    parser.add_argument('--plot', default=False, help='plot the results', action='store_true', dest='plot')
    parser.add_argument('--device', default='cpu', type=str, metavar='DEVICE',
                        help='Device to use [cpu, cuda].')

    signal.signal(signal.SIGINT, signal_handler)

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tester = TestTraj(args, cfg)
    tester.run()

    print("Done!")


