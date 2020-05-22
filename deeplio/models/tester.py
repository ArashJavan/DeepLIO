import os
import yaml
import datetime
import time
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from torchvision import transforms
from torch.utils import tensorboard

from deeplio import datasets as ds
from deeplio.losses.losses import LWSLoss, HWSLoss
from deeplio.common import spatial, utils
from deeplio.models import nets
from deeplio.models.misc import PostProcessSiameseData
from deeplio.models.worker import Worker, AverageMeter, ProgressMeter, worker_init_fn
from .transforms import ToTensor, Normalize, CenterCrop


class Tester(Worker):
    ACTION = "test"

    def __init__(self, parser):
        super(Tester, self).__init__(parser)
        args = self.args

        if self.seq_size != 2:
            self.logger.info("sequence size in the testing mode should be set to two.")
            raise ValueError("sequence size in the testing mode should be set to two.")

        if self.batch_size != 1:
            self.logger.info("batch size in the testing mode should be set to one.")
            self.logger.info("setting batch size (batch-size = 1).")
            self.batch_size = 1

        # create the folder for saving training checkpoints
        self.checkpoint_dir = self.out_dir
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # preapre dataset and dataloaders
        transform = transforms.Compose([ToTensor(),
                                        Normalize(mean=self.mean, std=self.std),
                                        CenterCrop(size=(self.im_height_model, self.im_width_model))])

        self.test_dataset = ds.Kitti(config=self.cfg, transform=transform, ds_type='test')
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                                           num_workers=self.num_workers,
                                                           shuffle=False,
                                                           worker_init_fn = worker_init_fn,
                                                           collate_fn = ds.deeplio_collate)

        self.post_processor = PostProcessSiameseData(seq_size=self.seq_size, batch_size=self.batch_size, shuffle=False)
        self.model = nets.DeepLIOS0(input_shape=(self.im_height_model, self.im_width_model,
                                                 self.n_channels), cfg=self.cfg['arch'])
        self.model.to(self.device)

        self.criterion = LWSLoss()

        self.tensor_writer = tensorboard.SummaryWriter(log_dir=self.runs_dir)

        # debugging and visualizing
        self.logger.print("DeepLIO Testing Configurations:")
        self.logger.print("batch-size:{}, workers: {}".
                          format(self.batch_size, self.num_workers))

        # load trained model checkpoint
        if args.model:
            if not os.path.isfile(args.model):
                self.logger.error("no model checkpoint found at '{}'".format(args.model))
                parser.print_help()
                parser.exit(1)
            ckp_path = args.model

            self.logger.info("loading checkpoint '{}".format(ckp_path))
            checkpoint = torch.load(ckp_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.logger.info("loaded checkpoint '{}' (epoch {})".format(ckp_path, checkpoint['epoch']))
        else:
            self.logger.error("no model checkpoint provided.")
            parser.print_help()
            parser.exit(1)

        self.logger.print(yaml.dump(self.cfg))
        self.logger.print(self.test_dataset)

    def run(self):
        self.is_running = True

        self.test()

        self.logger.info("Testing done!")
        self.close()

    def test(self):
        writer = self.tensor_writer
        criterion = self.criterion
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

                    # skip invalid data without ground-truth
                if not torch.all(data['valid']):
                    continue

                # prepare data
                imgs_0, imgs_1, imgs_untrans_0, imgs_untrans_1, gts, imus = self.post_processor(data)

                # send data to device
                imgs_0 = imgs_0.to(self.device, non_blocking=True)
                imgs_1 = imgs_1.to(self.device, non_blocking=True)
                gts = gts.to(self.device, non_blocking=True)
                imus = [imu.to(self.device, non_blocking=True) for imu in imus]

                # prepare ground truth tranlational and rotational part
                gt_x = gts[:, :3, 3].contiguous()
                gt_q = spatial.rotation_matrix_to_quaternion(gts[:, :3, :3].contiguous())

                # compute model predictions and loss
                pred_x, pred_q, _, _ = model([imgs_0, imgs_1])

                pred_q_norm = spatial.normalize_quaternion(pred_q)
                pred_axis_angle = spatial.quaternion_to_angle_axis(pred_q_norm)
                gt_axis_angle = spatial.quaternion_to_angle_axis(gt_q)

                #self.logger.print("px: {}\ngx: {}".format(pred_x.detach().cpu().numpy(), gt_x.detach().cpu().numpy()))
                #self.logger.print("pq: {}\ngq: {}".format(pred_axis_angle.detach().cpu().tolist(), gt_axis_angle.detach().cpu().tolist()))
                #self.logger.print("pq: {}\ngq: {}".format(pred_q.detach().cpu().tolist(), gt_q.detach().cpu().tolist()))

                loss = criterion(pred_x, pred_q, gt_x, gt_q)

                # measure accuracy and record loss
                losses.update(loss.detach().item(), len(pred_x))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # get meta information for saving the odom. results
                meta = data['metas'][0]
                date, drive = meta['date'][0], meta['drive'][0]
                idx = meta['index'][0]
                velo_idx = meta['velo-index'][0]
                oxts_ts = meta['oxts-timestamps']
                velo_ts = meta['velo-timestamps']

                gt_global = data['gts'][0][0].detach().cpu().numpy()
                seq_name = "{}_{}".format(date, drive)
                if seq_name not in seq_names:
                    if last_seq is not None:
                        last_seq.write_to_file()

                    curr_seq = OdomSeqRes(date, drive, output_dir=self.out_dir)
                    curr_seq.add_local_prediction(velo_ts[0], 0., gt_global[0], gt_global[0])

                    # add the file name and file-pointer to the list
                    seq_names.append(seq_name)

                gt_local = gt_x.detach().cpu().squeeze()
                gt_q = gt_q.detach().cpu().squeeze()

                T_local = np.identity(4)
                T_local[:3, 3] = pred_x.detach().cpu().squeeze().numpy() #  gt_local.numpy()
                T_local[:3, :3] = spatial.quaternion_to_rotation_matrix(pred_q.detach().cpu().squeeze()).numpy() #  spatial.quaternion_to_rotation_matrix(gt_q).numpy()
                # T_local = gts.detach().cpu().squeeze().numpy()
                curr_seq.add_local_prediction(velo_ts[1], losses.avg, T_local, gt_global[-1])

                last_seq = curr_seq
                if idx % self.args.print_freq == 0:
                    progress.display(idx)
                    # update tensorboard
                    step_val = idx
                    self.tensor_writer.add_scalar\
                        ("Loss test", losses.avg, step_val)
                    self.tensor_writer.flush()

                #if idx > 20:
                #    break

        if curr_seq is not None:
            curr_seq.write_to_file()


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
        plt.plot(self.T_global[:, 0, 3], self.T_global[:, 1, 3], alpha=0.5, label="GT-global")
        plt.scatter(self.T_global[:, 0, 3], self.T_global[:, 1, 3], alpha=0.5, s=0.5)
        plt.plot(T_glob_pred[:, 0, 3], T_glob_pred[:, 1, 3], alpha=0.5, label="GT-local")
        plt.scatter(T_glob_pred[:, 0, 3], T_glob_pred[:, 1, 3], alpha=0.5, s=0.5)

        plt.grid()
        plt.legend()
        plt.savefig(fname, figsize=(50, 50), dpi=600)





