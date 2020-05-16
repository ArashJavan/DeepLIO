import os
import yaml
import datetime
import time
import shutil
from pathlib import Path

import numpy as np

import torch
import torch.utils.data
from torchvision import transforms
from torch import optim
from torchvision.utils import make_grid

import tensorboardX
from tensorboardX import SummaryWriter

from pytorch_model_summary import summary

from deeplio import datasets as ds
from deeplio.losses import get_loss_function
from deeplio.common import spatial, utils
from deeplio.models import nets
from deeplio.models.misc import PostProcessSiameseData
from deeplio.models.worker import Worker, AverageMeter, ProgressMeter, worker_init_fn
from .transforms import ToTensor, Normalize, CenterCrop
from .optimizer import create_optimizer


class Trainer(Worker):
    ACTION = "train"

    def __init__(self, parser):
        super(Trainer, self).__init__(parser)

        if self.args.resume and self.args.evaluate:
            print("Error: can either resume training or evaluate, not both at the same time!")
            parser.print_help()
            parser.exit(1)

        args = self.args

        self.start_epoch = args.start_epoch
        self.epochs = args.epochs
        self.best_acc = float('inf')

        # create the folder for saving training checkpoints
        self.checkpoint_dir = self.out_dir
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # preapre dataset and dataloaders
        transform = transforms.Compose([ToTensor(),
                                        Normalize(mean=self.mean, std=self.std),
                                        CenterCrop(size=(self.im_height_model, self.im_width_model))])

        self.train_dataset = ds.Kitti(config=self.cfg, transform=transform)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                            num_workers=self.num_workers,
                                                            shuffle=False,
                                                            worker_init_fn=worker_init_fn,
                                                            collate_fn=ds.deeplio_collate)

        self.val_dataset = ds.Kitti(config=self.cfg, transform=transform, ds_type='validation')
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size,
                                                          num_workers=self.num_workers,
                                                          shuffle=False,
                                                          worker_init_fn=worker_init_fn,
                                                          collate_fn = ds.deeplio_collate)

        self.post_processor = PostProcessSiameseData(seq_size=self.seq_size, batch_size=self.batch_size, shuffle=True)
        self.model = nets.DeepLIOS0(input_shape=(self.im_height_model, self.im_width_model,
                                                 self.n_channels), cfg=self.cfg['arch'])
        self.model.to(self.device) #should be before creating optimizer

        self.optimizer = create_optimizer(self.model.parameters(), self.cfg, args)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        self.criterion = get_loss_function(self.cfg, args.device)

        self.tensor_writer = SummaryWriter(log_dir=self.runs_dir)

        # debugging and visualizing
        self.logger.print("DeepLIO Training Configurations:")
        self.logger.print("lr: {}, batch-size:{}, workers: {}, epochs:{}".
                          format(args.lr, self.batch_size, self.num_workers, self.epochs))

        # optionally resume from a checkpoint
        if args.resume or args.evaluate:
            ckp_path = ""
            if os.path.isfile(args.resume):
                ckp_path = args.resume
            elif os.path.isfile(args.evaluate):
                ckp_path = args.evaluate
            else:
                self.logger.error("no checkpoint found at '{}'".format(ckp_path))
                parser.print_help()
                parser.exit(1)

            self.logger.info("loading checkpoint '{}'".format(ckp_path))
            checkpoint = torch.load(ckp_path, map_location=self.device)
            self.start_epoch = checkpoint['epoch']
            self.best_acc = checkpoint['best_acc']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info("loaded checkpoint '{}' (epoch {})".format(ckp_path, checkpoint['epoch']))

        self.logger.print(yaml.dump(self.cfg))
        self.logger.print(self.train_dataset)
        self.logger.print(self.val_dataset)

        # log the network structure and number of params
        imgs = torch.randn((2, 3, self.n_channels, self.im_height_model, self.im_width_model)).to(self.device)
        self.model.eval()
        self.logger.print(summary(self.model, imgs))
        #self.tensor_writer.add_graph(self.model, imgs)

    def run(self):
        self.is_running = True

        if self.args.evaluate:
            self.validate(epoch=0)
            return

        for epoch in range(self.start_epoch, self.epochs):

            # check if we can run or are we stopped
            if not self.is_running:
                break

            self.logger.info("Starting epoch:{}, lr:{}".format(epoch, self.lr_scheduler.get_last_lr()[0]))

            # train for one epoch
            self.train_data(epoch)

            acc = self.validate(epoch)

            # remember best acc and save checkpoint
            is_best = acc < self.best_acc
            self.best_acc = min(acc, self.best_acc)

            self.save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'best_acc': self.best_acc,
                'optimizer' : self.optimizer.state_dict(),
            }, is_best)

            self.lr_scheduler.step()
            self.tensor_writer.add_scalar("LR", self.lr_scheduler.get_last_lr()[0], epoch)

        self.logger.info("Training done!")
        self.close()

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
        data_last = None
        step_val = 0.
        for idx, data in enumerate(self.train_dataloader):

            # check if we can run or are we stopped
            if not self.is_running:
                break

            # measure data loading time
            data_time.update(time.time() - end)

            # skip invalid data without ground-truth
            if not torch.all(data['valid']):
                continue

            # prepare data
            imgs_0, imgs_1,  imgs_untrans_0, imgs_untrans_1, gts, imus = self.post_processor(data)

            # send data to device
            imgs_0 = imgs_0.to(self.device, non_blocking=True)
            imgs_1 = imgs_1.to(self.device, non_blocking=True)

            imgs_untrans_0 = imgs_untrans_0.to(self.device, non_blocking=True)
            imgs_untrans_1 = imgs_untrans_1.to(self.device, non_blocking=True)

            gts = gts.to(self.device, non_blocking=True)
            imus = [imu.to(self.device, non_blocking=True) for imu in imus]

            # prepare ground truth tranlational and rotational part
            gt_pos = gts[:, :3, 3].contiguous()
            gt_rot = spatial.rotation_matrix_to_quaternion(gts[:, :3, :3].contiguous())

            # compute model predictions and loss
            pred_x, pred_q, _, _ = model([imgs_0, imgs_1])
            loss = criterion(pred_x, pred_q, gt_pos, gt_rot)

            # measure accuracy and record loss
            losses.update(loss.detach().item(), len(pred_x))
            #pred_disp.update(preds.detach().cpu().numpy(), gts[0].cpu().numpy())

            # zero the parameter gradients, compute gradient and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % self.args.print_freq == 0:
                progress.display(idx)

                ### update tensorboard
                step_val = epoch * len(self.train_dataloader) + idx
                self.tensor_writer.add_scalar("Train/Loss", losses.avg, step_val)
                self.tensor_writer.add_scalar("Train/Gradnorm", calc_grad_norm(self.model.parameters()), step_val)
                self.tensor_writer.flush()
            data_last = data

        # save infos to -e.g. gradient hists and images to tensorbaord and the end of training
        b, s, c, h, w = np.asarray(data_last['images'].shape)
        imgs = data_last['images'].reshape(b*s, c, h, w)
        imgs_remossion = imgs[:, 1, :, :]
        imgs_remossion = [torch.from_numpy(utils.colorize(img)).permute(2, 0, 1) for img in imgs_remossion]
        imgs_remossion = torch.stack(imgs_remossion)
        imgs_remossion = make_grid(imgs_remossion, nrow=2)
        self.tensor_writer.add_image("Images/Remissions", imgs_remossion, global_step=step_val)

        imgs_range = imgs[:, 0, :, :]
        imgs_range = [torch.from_numpy(utils.colorize(img, cmap='viridis')).permute(2, 0, 1) for img in imgs_range]
        imgs_range = torch.stack(imgs_range)
        imgs_range = make_grid(imgs_range, nrow=2)
        self.tensor_writer.add_image("Images/Range", imgs_range, global_step=step_val)

        for tag, param in self.model.named_parameters():
            self.tensor_writer.add_histogram(tag, param.data.detach().cpu().numpy(), step_val)

    def validate(self, epoch):
        writer = self.tensor_writer
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
                gt_pos = gts[:, :3, 3].contiguous()
                gt_rot = spatial.rotation_matrix_to_quaternion(gts[:, :3, :3].contiguous())

                # compute model predictions and loss
                pred_x, pred_q = model([imgs_0, imgs_1])
                loss = criterion(pred_x, pred_q, gt_pos, gt_rot)

                # measure accuracy and record loss
                losses.update(loss.detach().item(), len(pred_x))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if idx % self.args.print_freq == 0:
                    progress.display(idx)
                    # update tensorboard
                    step_val = epoch * len(self.val_dataloader) + idx
                    self.tensor_writer.add_scalar\
                        ("Val/Loss", losses.avg, step_val)
                    self.tensor_writer.flush()
        return losses.avg

    def save_checkpoint(self, state, is_best):
        epoch = state['epoch']
        filename = '{}/cpkt_{}_{}_{}.tar'.format(self.checkpoint_dir, self.model.name, epoch, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        torch.save(state, filename)
        if is_best:
            filename_best = '{}/cpkt_{}_best.tar'.format(self.checkpoint_dir, self.model.name)
            shutil.copyfile(filename, filename_best)


def calc_grad_norm(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
    return total_norm
