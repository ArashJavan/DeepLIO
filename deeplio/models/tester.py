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

from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils import tensorboard


from deeplio import datasets as ds
from deeplio.losses.losses import *
from deeplio.common import spatial, utils
from deeplio.models import nets
from deeplio.models.misc import PostProcessSiameseData
from deeplio.models.worker import Worker, AverageMeter, ProgressMeter, worker_init_fn


class Tester(Worker):
    ACTION = "test"

    def __init__(self, parser):
        super(Tester, self).__init__(parser)
        args = self.args

        if self.seq_size != 2:
            self.logger.info("sequence size in the testing mode should be set to two.")
            self.logger.info("setting sequence size (seq-size = 2).")
            self.seq_size = 2

        if self.batch_size != 1:
            self.logger.info("batch size in the testing mode should be set to one.")
            self.logger.info("setting sequence size (batch-size = 1).")
            self.batch_size = 1

        # create the folder for saving training checkpoints
        self.checkpoint_dir = self.out_dir
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # preapre dataset and dataloaders
        transform = transforms.Compose([ds.ToTensor(),
                                        ds.Normalize(mean=self.mean, std=self.std)])

        self.test_dataset = ds.Kitti(config=self.cfg, transform=transform, ds_type='test')
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                                           num_workers=self.num_workers,
                                                           shuffle=False,
                                                           worker_init_fn = worker_init_fn,
                                                           collate_fn = ds.deeplio_collate)

        self.post_processor = PostProcessSiameseData(seq_size=self.seq_size, batch_size=self.batch_size, shuffle=False)
        self.model = nets.DeepLIOS0(input_shape=(self.n_channels, self.im_height, self.im_width), p=0)
        self.model.to(self.device)

        self.criterion = GeoConstLoss()

        self.tensor_writer = tensorboard.SummaryWriter(log_dir=self.runs_dir)

        # debugging and visualizing
        self.logger.print("DeepLIO Testing Configurations:")
        self.logger.print("batch-size:{}, workers: {}".
                          format(self.batch_size, self.num_workers))

        # optionally resume from a checkpoint
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
        self.logger.print(self.model.repr())


    def run(self):
        self.is_running = True

        self.test()

        self.logger.info("Training done!")
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
                imgs_0, imgs_1, gts, imus = self.post_processor(data)

                # send data to device
                imgs_0 = imgs_0.to(self.device, non_blocking=True)
                imgs_1 = imgs_1.to(self.device, non_blocking=True)
                gts = gts.to(self.device, non_blocking=True)
                imus = [imu.to(self.device, non_blocking=True) for imu in imus]

                # prepare ground truth tranlational and rotational part
                gt_pos = gts[:, :3, 3].contiguous()
                gt_rot = spatial.rotation_matrix_to_quaternion(gts[:, :3, :3].contiguous())
                gts = [gt_pos, gt_rot]

                # compute model predictions and loss
                preds = model([imgs_0, imgs_1])
                loss = criterion(preds, gts)

                # measure accuracy and record loss
                losses.update(loss.detach().item(), len(preds))

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

                seq_name = "{}_{}".format(date, drive)
                if seq_name not in seq_names:
                    if last_seq is not None:
                        last_seq.write_to_file()

                        if self.args.plot:
                            last_seq.save_plot()

                    curr_seq = OdomSeqRes(date, drive, output_dir=self.out_dir)
                    curr_seq.start_pose = data['gts'][0][0][0].detach().cpu().numpy()
                    curr_seq.start_time = velo_ts[0]
                    # add the file name and file-pointer to the list
                    seq_names.append(seq_name)
                gt_local = gt_pos[0].detach().cpu().numpy()
                gt_global = data['gts'][0][0][-1][0:3, 3].detach().cpu().numpy()
                curr_seq.add_res(preds[0].detach().cpu().numpy(), velo_ts[1], losses.avg, gt_local=gt_local, gt_global=gt_global)

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
            if self.args.plot:
                curr_seq.save_plot()


class OdomSeqRes:
    def __init__(self, date, drive, output_dir="."):
        self.date = date
        self.drive = drive
        self.odom_res = []
        self.timestamps = []
        self.gt_local = []
        self.gt_global = []

        self.start_pose = np.identity(4)
        self.start_time = 0.
        self.loss = []
        self.out_dir = output_dir

    def add_res(self, odom, timestamp, loss, gt_local, gt_global):
        self.timestamps.append(timestamp)
        self.loss.append(loss)
        self.odom_res.append(odom)
        self.gt_local.append(gt_local)
        self.gt_global.append(gt_global)

    def write_to_file(self):
        start_pos = self.start_pose[:3, 2]

        odom_local = np.asarray(self.odom_res)
        odom_global = np.cumsum(odom_local, axis=0) + start_pos

        self.timestamps = np.asarray(self.timestamps).reshape(-1, 1)
        self.loss = np.asarray(self.loss).reshape(-1, 1)
        if self.gt_global and self.gt_local:
            gt_local = np.asarray(self.gt_local)
            gt_global = np.asarray(self.gt_global)

            res = np.hstack((self.timestamps, odom_local, gt_local, odom_global, gt_global, self.loss))
        else:
            res = np.hstack((self.timestamps, odom_local, odom_global, self.loss))

        fname = "{}/{}_{}.csv".format(self.out_dir, self.date, self.drive)
        np.savetxt(fname, res, fmt='%.5f', delimiter=',')

    def save_plot(self):
        if self.gt_global:
            gt_global = np.asarray(self.gt_global)

        start_pos = self.start_pose[:3, 2]

        odom_local = np.asarray(self.odom_res)
        odom_global = np.cumsum(odom_local, axis=0) + start_pos

        fname = "{}/{}_{}.png".format(self.out_dir, self.date, self.drive)
        fig, axs = plt.subplots(1, 1)
        axs.plot(odom_global[:, 0], odom_global[:, 1], label="preds")

        if self.gt_global and self.gt_local:
            axs.plot(gt_global[:, 0], gt_global[:, 1], label="gt")

        fig.savefig(fname)




