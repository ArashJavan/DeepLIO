import os
import sys
import yaml
import time
import datetime
import shutil
import argparse
from pathlib import Path

import numpy as np

import torch
from torch.backends import cudnn
from torchvision import transforms
from torch import optim
from torch.utils import tensorboard
from torchvision.utils import make_grid

from pytorch_model_summary import summary

dname = os.path.abspath(os.path.dirname(__file__))
content_dir = os.path.abspath("{}/..".format(dname))
sys.path.append(dname)
sys.path.append(content_dir)

from deeplio.datasets import kitti, transfromers
from deeplio.models import deeplio_nets as net
from deeplio.models.misc import *
from deeplio.losses.losses import *
from deeplio.common.spatial import *
from deeplio.common.utils import colorize
from deeplio.common import logger
from deeplio.common.logger import PyLogger


SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def worker_init_fn(worker_id):
    set_seed(seed=SEED)


class Trainer:
    def __init__(self, args):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        self.dname = os.path.dirname(__file__)
        self.content_dir = os.path.abspath("{}/..".format(dname))

        self.args = args
        self.cfg = cfg
        self.ds_cfg = self.cfg['datasets']
        self.curr_dataset_cfg = self.cfg['datasets'][self.cfg['current-dataset']]
        self.seq_size = self.ds_cfg['sequence-size']

        self.batch_size = self.args.batch_size
        self.epochs = self.args.epochs
        num_workers = self.args.workers

        mean = np.array(self.curr_dataset_cfg['mean'])
        std = np.array(self.curr_dataset_cfg['std'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # get input images shape and channels
        self.im_height, self.im_width = self.curr_dataset_cfg['image-height'], self.curr_dataset_cfg['image-width']
        self.n_channels = len(self.cfg['channels'])

        # create output folder structure
        self.out_dir = "{}/outputs/train_{}".format(self.content_dir,  datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.checkpoint_dir = self.out_dir
        runs_dir = "{}/outputs/runs/{}".format(self.content_dir,  datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        log_dir = self.checkpoint_dir
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(runs_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        self.best_acc = float('inf')

        cudnn.benchmark = True

        set_seed(seed=SEED)

        flog_name = "{}/train_{}.log".format(log_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.logger = PyLogger(filename=flog_name)
        logger.global_logger = self.logger

        # preapre dataset and dataloaders
        transform = transforms.Compose([transfromers.ToTensor(),
                                        transfromers.Normalize(mean=mean, std=std)])

        self.train_dataset = kitti.Kitti(config=cfg, transform=transform)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                            num_workers=num_workers,
                                                            shuffle=True,
                                                            worker_init_fn = worker_init_fn)

        self.val_dataset = kitti.Kitti(config=cfg, transform=transform, ds_type='validation')
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size,
                                                            num_workers=num_workers,
                                                            shuffle=False,
                                                            worker_init_fn = worker_init_fn)


        self.post_processor = PostProcessSiameseData(seq_size=self.seq_size, batch_size=self.batch_size)
        self.model = net.DeepLIOS0(input_shape=(self.n_channels, self.im_height, self.im_width), p=0)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = GeoConstLoss()

        self.tensor_writer = tensorboard.SummaryWriter(log_dir=runs_dir)

        # debugging and visualizing
        self.logger.print("DeepLIO Training Configurations:")
        self.logger.print("lr: {}, batch-size:{}, workers: {}, epochs:{}".
                          format(self.args.lr, self.batch_size, num_workers, self.epochs))
        self.logger.print(yaml.dump(self.cfg))
        self.logger.print(self.train_dataset)
        self.logger.print(self.val_dataset)
        self.logger.print(self.model.repr())

        imgs = torch.randn((2, 3, self.n_channels, self.im_height, self.im_width)).to(self.device)
        self.model.eval()
        self.logger.print(summary(self.model, imgs))
        self.tensor_writer.add_graph(self.model, imgs)

    def train(self):
        for epoch in range(self.epochs):
            lr = self.adjust_learning_rate(epoch)
            self.logger.info("Starting epoch:{}, lr:{}".format(epoch, lr))

            # train for one epoch
            self.train_data(epoch)

            acc = self.validate(epoch)

            # remember best acc and save checkpoint
            is_best = acc < self.best_acc
            self.best_acc = min(acc, self.best_acc)

            self.save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'best_acc1': self.best_acc,
                'optimizer' : self.optimizer.state_dict(),
            }, is_best)

        self.tensor_writer.close()
        self.logger.info("Training done!")

    def train_data(self, epoch):
        writer = self.tensor_writer
        optimizer = self.optimizer
        criterion = self.criterion
        model = self.model

        # switch to train mode
        model.train()

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        #pred_disp = PredDisplay()
        progress = ProgressMeter(
            self.logger,
            len(self.train_dataloader),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(epoch))

        end = time.time()
        for idx, data in enumerate(self.train_dataloader):

            # measure data loading time
            data_time.update(time.time() - end)

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
            gt_rot = rotation_matrix_to_quaternion(gts[:, :3, :3].contiguous())
            gts = [gt_pos, gt_rot]

            # compute model predictions and loss
            preds = model([imgs_0, imgs_1])
            loss = criterion(preds, gts)

            # measure accuracy and record loss
            losses.update(loss.detach().item(), len(preds))
            #pred_disp.update(preds.detach().cpu().numpy(), gts[0].cpu().numpy())

            # zero the parameter gradients, compute gradient and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % args.print_freq == 0:
                progress.display(idx)

                # update tensorboard
                step_val = epoch * len(self.train_dataloader) + idx
                self.tensor_writer.add_scalar("Loss train", losses.avg, step_val)
                imgs = data['images'].reshape(self.batch_size * self.seq_size,
                                              self.n_channels, self.im_height, self.im_width)
                imgs_remossion = imgs[:, 0:1, :, :]
                imgs_remossion = [torch.from_numpy(colorize(img)).permute(2, 0, 1) for img in imgs_remossion]
                imgs_remossion = torch.stack(imgs_remossion)
                imgs_remossion = make_grid(imgs_remossion, nrow=2)
                self.tensor_writer.add_image("Image remissions", imgs_remossion, global_step=step_val)

                imgs_depth = imgs[:, 1:2, :, :]
                imgs_depth = [torch.from_numpy(colorize(img, cmap='viridis')).permute(2, 0, 1) for img in imgs_depth]
                imgs_depth = torch.stack(imgs_depth)
                imgs_depth = make_grid(imgs_depth, nrow=2)
                self.tensor_writer.add_image("Image depth", imgs_depth, global_step=step_val)

                for tag, param in self.model.named_parameters():
                    self.tensor_writer.add_histogram(tag, param.grad.detach().cpu().numpy(), step_val)
                self.tensor_writer.flush()

    def validate(self, epoch):
        writer = self.tensor_writer
        optimizer = self.optimizer
        criterion = self.criterion
        model = self.model

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            self.logger,
            len(self.val_dataloader),
            [batch_time, losses],
            prefix='Test: ')

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            end = time.time()
            for idx, data in enumerate(self.val_dataloader):
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
                gt_rot = rotation_matrix_to_quaternion(gts[:, :3, :3].contiguous())
                gts = [gt_pos, gt_rot]

                # compute model predictions and loss
                preds = model([imgs_0, imgs_1])
                loss = criterion(preds, gts)

                # measure accuracy and record loss
                losses.update(loss.detach().item(), len(preds))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if idx % args.print_freq == 0:
                    progress.display(idx)

                    # update tensorboard
                    step_val = epoch * len(self.train_dataloader) + idx
                    self.tensor_writer.add_scalars("Loss val", losses.avg, step_val)
                    imgs = data['images'].reshape(self.batch_size * self.seq_size,
                                                  self.n_channels, self.im_height, self.im_width)
                    imgs_remossion = imgs[:, 0:1, :, :]
                    imgs_remossion = [torch.from_numpy(colorize(img)).permute(2, 0, 1) for img in imgs_remossion]
                    imgs_remossion = torch.stack(imgs_remossion)
                    imgs_remossion = make_grid(imgs_remossion, nrow=2)
                    self.tensor_writer.add_image("Image remissions", imgs_remossion, global_step=step_val)

                    imgs_depth = imgs[:, 1:2, :, :]
                    imgs_depth = [torch.from_numpy(colorize(img, cmap='viridis')).permute(2, 0, 1) for img in
                                  imgs_depth]
                    imgs_depth = torch.stack(imgs_depth)
                    imgs_depth = make_grid(imgs_depth, nrow=2)
                    self.tensor_writer.add_image("Image depth", imgs_depth, global_step=step_val)

                    for tag, param in self.model.named_parameters():
                        self.tensor_writer.add_histogram(tag, param.grad.detach().cpu().numpy(), step_val)
                    self.tensor_writer.flush()
        return losses.avg

    def save_checkpoint(self, state, is_best):
        epoch = state['epoch']
        filename = '{}/cpkt_{}_{}.tar'.format(self.checkpoint_dir, epoch, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        torch.save(state, filename)
        if is_best:
            filename_best = '{}/cpkt_best_{}_{}.tar'.format(self.checkpoint_dir, epoch, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
            shutil.copyfile(filename, filename_best)

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the niital LR decayed by 10 every 10 epochs """
        lr = self.args.lr * (0.1 ** (epoch // 10))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class PredDisplay(object):
    def __init__(self, name="Preds-vs-GT", fmt=':f'):
        self.name = name
        self.fmt = fmt
        self. pred = None
        self.gt = None

    def update(self, pred, gt):
        self.pred = pred
        self.gt = gt

    def __str__(self):
        fmtstr = '{name} {pred} -> {gt}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, logger, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepLIO Training')

    # Hyper Params
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N',
                        help='mini-batch size (default: 1), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('-c', '--config', default="./config.yaml", help='Path to configuration file')

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()




