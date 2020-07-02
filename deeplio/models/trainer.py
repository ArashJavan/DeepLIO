import os
import shutil
import time
import yaml
from pathlib import Path

import numpy as np

import torch
import torch.utils.data
from torchvision.utils import make_grid

from pytorch_model_summary import summary

from deeplio import datasets as ds
from deeplio.common import spatial, utils
from deeplio.losses import get_loss_function
from deeplio.models import nets
from deeplio.models.misc import DataCombiCreater, PolynomialLRDecay
from .optimizer import create_optimizer
from .worker import Worker, AverageMeter, ProgressMeter, worker_init_fn


class Trainer(Worker):
    ACTION = "train"

    def __init__(self, args, cfg):
        super(Trainer, self).__init__(args, cfg)

        if self.args.resume and self.args.evaluate:
            raise ValueError("Error: can either resume training or evaluate, not both at the same time!")

        args = self.args

        self.start_epoch = args.start_epoch
        self.epochs = args.epochs
        self.best_acc = float('inf')
        self.step_val = 0.

        # create the folder for saving training checkpoints
        self.checkpoint_dir = self.out_dir
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # preapre dataset and dataloaders
        transform = None
        self.data_last = None

        self.model = nets.get_model(input_shape=(self.n_channels, self.im_height_model, self.im_width_model),
                                    cfg=self.cfg, device=self.device)

        self.criterion = get_loss_function(self.cfg, args.device)
        self.optimizer = create_optimizer([{'params': self.model.parameters()},
                                           {'params': self.criterion.parameters()}]
                                          , self.cfg, args)
        self.lr_scheduler = PolynomialLRDecay(self.optimizer, max_decay_steps=self.epochs, end_learning_rate=0.00005,
                                              power=2.0)

        self.has_lidar = True if self.model.lidar_feat_net is not None else False
        self.has_imu = True if self.model.imu_feat_net is not None else False

        self.train_dataset = ds.Kitti(config=self.cfg, transform=transform,
                                      has_imu=self.has_imu, has_lidar=self.has_lidar)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                            num_workers=self.num_workers,
                                                            shuffle=True,
                                                            worker_init_fn=worker_init_fn,
                                                            collate_fn=ds.deeplio_collate,
                                                            drop_last=True)

        self.val_dataset = ds.Kitti(config=self.cfg, transform=transform, ds_type='validation',
                                    has_imu=self.has_imu, has_lidar=self.has_lidar)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size,
                                                          num_workers=self.num_workers,
                                                          shuffle=True,
                                                          worker_init_fn=worker_init_fn,
                                                          collate_fn = ds.deeplio_collate,
                                                          drop_last=True)

        self.data_permuter = DataCombiCreater(combinations=self.combinations,
                                              device=self.device)

        # debugging and visualizing
        self.logger.print("System Training Configurations:")
        self.logger.print("args: {}".
                          format(self.args))

        # optionally resume from a checkpoint
        if args.resume or args.evaluate:
            model_cfg = self.cfg[self.cfg['arch']]
            pretrained = self.model.pretrained
            if not pretrained:
                self.logger.error("no model checkpoint loaded!")
                raise ValueError("no model checkpoint loaded!")

            ckp_path = model_cfg['model-path']
            if not os.path.isfile(args.resume):
                self.logger.error("no checkpoint found at '{}'".format(ckp_path))
                raise ValueError("no checkpoint found at '{}'".format(ckp_path))

            self.logger.info("loading from checkpoint '{}'".format(ckp_path))
            checkpoint = torch.load(ckp_path, map_location=self.device)
            self.start_epoch = checkpoint['epoch']
            self.best_acc = checkpoint['best_acc']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info("loaded checkpoint '{}' (epoch {})".format(ckp_path, checkpoint['epoch']))

        self.logger.print(yaml.dump(self.cfg))
        self.logger.print(self.train_dataset)
        self.logger.print(self.val_dataset)

        self.post_init()

    def post_init(self):
        raise NotImplementedError()

    def post_train_iter(self):
        raise NotImplementedError()

    def post_valiate(self):
        raise NotImplementedError()

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
            self.train(epoch)

            acc = self.validate(epoch)

            # remember best acc and save checkpoint
            is_best = acc < self.best_acc
            self.best_acc = min(acc, self.best_acc)

            filename = 'cpkt_{}'.format(self.model.name)
            self.save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'best_acc': self.best_acc,
                'optimizer' : self.optimizer.state_dict(),
                'criterion': self.criterion.state_dict(),
            }, is_best, filename)

            feat_nets = self.model.get_feat_networks()
            for feat_net in feat_nets:
                filename = 'cpkt_{}'.format(feat_net.name)
                self.save_checkpoint({
                    'state_dict': feat_net.state_dict(),
                }, is_best, filename)

            self.lr_scheduler.step()
            self.tensor_writer.add_scalar("LR", self.lr_scheduler.get_last_lr()[0], epoch)

        self.logger.info("Training done!")
        self.close()

    def train(self, epoch):
        optimizer = self.optimizer

        # switch to train mode
        self.model.train()
        self.criterion.train()

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
        #torch.autograd.set_detect_anomaly(True)
        #with torch.autograd.detect_anomaly():
        for idx, data in enumerate(self.train_dataloader):

            # check if we can run or are we stopped
            if not self.is_running:
                break

            if self.data_last is None:
                self.data_last = data

            # if self.has_imu:
            #     idx_invalid = np.where(np.sum(np.array(data["valids"]), axis=1) != self.seq_size_data)[0]
            #     if len(idx_invalid) > 0:
            #         try:
            #             data['imus'] = [data['imus'][x] for x in range(len(data['gts'])) if x not in idx_invalid]
            #             data['valids'] = [data['valids'][x] for x in range(len(data['gts'])) if x not in idx_invalid]
            #             data['metas'] = [data['metas'][x] for x in range(len(data['gts'])) if x not in idx_invalid]
            #             data['gts'] = data['gts'][[x for x in range(len(data['gts'])) if x not in idx_invalid]]
            #         except IndexError as ex:
            #             self.logger.warning("{} (idx-inv={}, len={})".format(str(ex), idx_invalid, len(data['gts'])))
            #             continue

            # measure data loading time
            data_time.update(time.time() - end)

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
            if torch.isnan(gts_f2g).any() or torch.isinf(gts_f2g).any() :
                raise ValueError("gt-f2g:\n{}".format(gts_f2g))

            if torch.isnan(normals).any() or torch.isinf(normals).any():
                raise ValueError("normals:\n{}".format(normals))
            if torch.isnan(imgs).any() or torch.isinf(imgs).any():
                raise ValueError("imgs:\n{}".format(imgs))

            # prepare ground truth tranlational and rotational part
            gt_f2f_x = gts_f2f[:, :, 0:3]
            gt_f2f_q = gts_f2f[:, :, 3:]
            gt_f2g_x = gts_f2g[:, :, 0:3]
            gt_f2g_q = gts_f2g[:, :, 3:7]

            # compute model predictions and loss
            pred_f2f_x, pred_f2f_r = self.model([[imgs, normals], imus])
            #pred_f2f_r = spatial.normalize_quaternion(pred_f2f_r)

            if torch.isnan(pred_f2f_x).any() or torch.isinf(pred_f2f_x).any():
                raise ValueError("pred_f2f_x:\n{}".format(pred_f2f_x))
            if torch.isnan(pred_f2f_r).any() or torch.isinf(pred_f2f_r).any():
                raise ValueError("pred_f2f_r:\n{}".format(pred_f2f_r))

            pred_f2g_x, pred_f2g_r = self.se3_to_SE3(pred_f2f_x, pred_f2f_r)

            # gt_f2g_xx, pred_f2g_rr = self.se3_to_SE3(gt_f2f_x, gt_f2f_q)
            # print(gt_f2g_xx - gt_f2g_x)
            # print(pred_f2g_rr - gt_f2g_q)

            loss = self.criterion(pred_f2f_x, pred_f2f_r,
                                  pred_f2g_x, pred_f2g_r,
                                  gt_f2f_x, gt_f2f_q,
                                  gt_f2g_x, gt_f2g_q)

            # measure accuracy and record loss
            losses.update(loss.detach().item(), self.batch_size*self.seq_size)
            #pred_disp.update(preds.detach().cpu().numpy(), gts[0].cpu().numpy())

            # zero the parameter gradients, compute gradient and optimizer step
            self.optimizer.zero_grad()
            loss.backward()

            param_means = torch.FloatTensor([param.mean().detach() for param in self.model.parameters() if param is not None])
            if torch.isnan(param_means).any():
                raise ValueError("param_means:\n{}".format(param_means))

            param_grad_means = torch.FloatTensor([param.grad.mean().detach() for param in self.model.parameters() if param.grad is not None])
            if torch.isnan(param_grad_means).any():
                raise ValueError("param_grad_means:\n{}".format(param_grad_means))

            if idx % 10 == 0:
                pass # print("param-mean: {}, grad-mean: {}".format(param_means.mean().data, param_grad_means.mean().data))

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % self.args.print_freq == 0:

                if idx % (5 * self.args.print_freq) == 0:
                    # print some prediction results
                    x = pred_f2g_x[0, 0:2].detach().cpu().flatten()
                    q = pred_f2g_r[0, 0:2].detach().cpu().flatten()
                    x_gt = gt_f2g_x[0:2].detach().cpu().flatten()
                    q_gt = gt_f2g_q[0:2].detach().cpu().flatten()

                    self.logger.print("x-hat: [{:.4f},{:.4f},{:.4f}], [{:.4f},{:.4f},{:.4f}]"
                                      "\nx-gt:  [{:.4f},{:.4f},{:.4f}], [{:.4f},{:.4f},{:.4f}]".
                                      format(x[0], x[1], x[2], x[3], x[4], x[5],
                                             x_gt[0], x_gt[2], x_gt[2], x_gt[3], x_gt[4], x_gt[5]))

                    self.logger.print("q-hat: [{:.4f},{:.4f},{:.4f},{:.4f}], [{:.4f},{:.4f},{:.4f},{:.4f}]"
                                      "\nq-gt:  [{:.4f},{:.4f},{:.4f},{:.4f}], [{:.4f},{:.4f},{:.4f},{:.4f}]".
                                      format(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7],
                                             q_gt[0], q_gt[1], q_gt[2], q_gt[3], q_gt[4], q_gt[5], q_gt[6], q_gt[7]))

                progress.display(idx)
                grad_norm = calc_grad_norm(self.model.parameters())

                ### update tensorboard
                self.step_val = epoch * len(self.train_dataloader) + idx
                self.tensor_writer.add_scalar("Train/Loss", losses.avg, self.step_val)
                self.tensor_writer.add_scalar("Train/Gradnorm", grad_norm, self.step_val)
                self.tensor_writer.add_scalar("Train/sx", self.criterion.sx.data, self.step_val)
                self.tensor_writer.add_scalar("Train/sq", self.criterion.sq.data, self.step_val)
                self.tensor_writer.flush()

            self.data_last = data

        self.post_train_iter()

        for tag, param in self.model.named_parameters():
            self.tensor_writer.add_histogram(tag, param.data.detach().cpu().numpy(), self.step_val)

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
                euler_cur = f2f_r[b, s]
                R_cur = spatial.euler_to_rotation_matrix(euler_cur.unsqueeze(0)).squeeze() # spatial.quaternion_to_rotation_matrix(q_cur)

                if not torch.isclose(torch.det(R_cur), torch.FloatTensor([1.]).to(self.device)).all():
                    raise ValueError("Det error:\nR\n{}\nq:\n{}".format(R_cur, euler_cur))

                t_prev = torch.matmul(R_prev, t_cur) + t_prev
                R_prev = torch.matmul(R_prev, R_cur)

                if not torch.isclose(torch.det(R_prev), torch.FloatTensor([1.]).to(self.device)).all():
                    raise ValueError("Det error:\nR\n{}".format(R_prev))

                f2g_q[b, s] = spatial.rotation_matrix_to_quaternion(R_prev)
                f2g_x[b, s] = t_prev
        return f2g_x, f2g_q

    def validate(self, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            self.logger,
            len(self.val_dataloader),
            [batch_time, losses],
            prefix='Test: ')

        # switch to evaluate mode
        self.model.eval()
        self.criterion.eval()

        with torch.no_grad():
            end = time.time()
            for idx, data in enumerate(self.val_dataloader):

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

                if torch.isnan(normals).any() or torch.isinf(normals).any():
                    raise ValueError("normals:\n{}".format(normals))
                if torch.isnan(imgs).any() or torch.isinf(imgs).any():
                    raise ValueError("imgs:\n{}".format(imgs))

                # prepare ground truth tranlational and rotational part
                gt_f2f_x = gts_f2f[:, :, 0:3]
                gt_f2f_q = gts_f2f[:, :, 3:]
                gt_f2g_x = gts_f2g[:, :, 0:3]
                gt_f2g_q = gts_f2g[:, :, 3:7]

                # compute model predictions and loss
                pred_f2f_x, pred_f2f_r = self.model([[imgs, normals], imus])
                # pred_f2f_r = spatial.normalize_quaternion(pred_f2f_r)

                if torch.isnan(pred_f2f_x).any() or torch.isinf(pred_f2f_x).any():
                    raise ValueError("pred_f2f_x:\n{}".format(pred_f2f_x))
                if torch.isnan(pred_f2f_r).any() or torch.isinf(pred_f2f_r).any():
                    raise ValueError("pred_f2f_r:\n{}".format(pred_f2f_r))

                pred_f2g_x, pred_f2g_r = self.se3_to_SE3(pred_f2f_x, pred_f2f_r)
                loss = self.criterion(pred_f2f_x, pred_f2f_r,
                                      pred_f2g_x, pred_f2g_r,
                                      gt_f2f_x, gt_f2f_q,
                                      gt_f2g_x, gt_f2g_q)

                # measure accuracy and record loss
                losses.update(loss.detach().item(), self.batch_size*self.seq_size)

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

        self.post_valiate()
        return losses.avg

    def save_checkpoint(self, state, is_best, filename="checkpoint"):
        file_path = '{}/{}.tar'.format(self.checkpoint_dir, filename)
        torch.save(state, file_path)
        if is_best:
            file_path_best = '{}/{}_best.tar'.format(self.checkpoint_dir, filename)
            shutil.copyfile(file_path, file_path_best)


class TrainerDeepLIO(Trainer):
    ACTION = "train_deeplio"

    def post_init(self):
        # log the network structure and number of params
        # log the network structure and number of params
        self.logger.info("{}: Network architecture:".format(self.model.name))
        xyz_data = torch.randn((self.batch_size, self.seq_size, self.timestamps, self.n_channels, self.im_height_model, self.im_width_model)).to(self.device)
        normal_data =  xyz_data
        imu_data = torch.rand((self.batch_size,  self.seq_size, 10, 6)).to(self.device)
        self.model.eval()
        self.logger.print(summary(self.model, [[xyz_data, normal_data], imu_data]))

    def post_train_iter(self):
        if self.has_lidar:
            # save infos to -e.g. gradient hists and images to tensorbaord and the end of training
            b, s, c, h, w = np.asarray(self.data_last['images'].shape)
            imgs = self.data_last['images'].reshape(b*s, c, h, w)
            imgs_remossion = imgs[:, 1, :, :]
            imgs_remossion = [torch.from_numpy(utils.colorize(img)).permute(2, 0, 1) for img in imgs_remossion]
            imgs_remossion = torch.stack(imgs_remossion)
            imgs_remossion = make_grid(imgs_remossion, nrow=2)
            self.tensor_writer.add_image("Images/Remissions", imgs_remossion, global_step=self.step_val)

            imgs_range = imgs[:, 0, :, :]
            imgs_range = [torch.from_numpy(utils.colorize(img, cmap='viridis')).permute(2, 0, 1) for img in imgs_range]
            imgs_range = torch.stack(imgs_range)
            imgs_range = make_grid(imgs_range, nrow=2)
            self.tensor_writer.add_image("Images/Range", imgs_range, global_step=self.step_val)

    def post_valiate(self):
        pass


def calc_grad_norm(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
    return total_norm
