import os
import random
import yaml
import argparse
import multiprocessing # getting number of cpus

import torch
from torchvision import transforms
from torch import optim
from torch.utils import tensorboard

from deeplio.datasets import kitti
from deeplio.datasets import transfromers
from deeplio.models import deeplio_nets as net
from deeplio.models.misc import *
from deeplio.losses.losses import *
from deeplio.common.spatial import *
from deeplio.common.utils import set_seed
from deeplio.visualization.utilities import *

from pytorch_model_summary import summary

SEED = 42

def worker_init_fn(worker_id):
    set_seed(seed=SEED)


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_cfg = self.cfg['datasets'][self.cfg['current-dataset']]
        self.batch_size = self.cfg['batch-size']
        self.seq_size = self.dataset_cfg['sequence-size']

        self.epoch = self.cfg['epoch']
        num_workers = int(multiprocessing.cpu_count()/2)

        mean = np.array(self.dataset_cfg['mean'])
        std = np.array(self.dataset_cfg['std'])

        set_seed(seed=SEED)

        transform = transforms.Compose([transfromers.ToTensor(),
                                        transfromers.Normalize(mean=mean, std=std)])
        dataset = kitti.Kitti(config=cfg, transform=transform)
        self.train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                            num_workers=num_workers,
                                                            shuffle=True,
                                                            worker_init_fn = worker_init_fn)

        self.post_processor = PostProcessSiameseData(seq_size=self.seq_size, batch_size=self.batch_size)

        self.tensor_writer = tensorboard.SummaryWriter()

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in args.gpu_ids)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        channels = len(self.cfg['channels'])
        self.model = net.DeepLIOS0(input_shape=(channels, None, None), p=0)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = GeoConstLoss()

    def train(self):
        writer = self.tensor_writer
        optimizer = self.optimizer
        criterion = self.criterion
        model = self.model
        model.train()

        for epoch in range(self.epoch):
            running_loss = 0.
            for idx, data in enumerate(self.train_dataloader):

                # skip invalid data without ground-truth
                if not torch.all(data['valid']):
                    continue

                imgs_0, imgs_1, gts, imus = self.post_processor(data)
                imgs_0 = imgs_0.to(self.device)
                imgs_1 = imgs_1.to(self.device)
                gts = gts.to(self.device)
                imus = [imu.to(self.device) for imu in imus]

                # zero the parameter gradients
                optimizer.zero_grad()

                preds = model([imgs_0, imgs_1])

                gt_pos = gts[:, :3, 3].contiguous()
                gt_rot = rotation_matrix_to_quaternion(gts[:, :3, :3].contiguous())
                gts = [gt_pos, gt_rot]

                loss = criterion(preds, gts)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if idx % 10 == 0:
                    writer.add_scalar("Loss/train", running_loss / 10, len(self.train_dataloader) + idx)
                    running_loss = 0.
                    print("[{}] loss: {}".format(idx, loss.data))
                    preds_ = [preds[i].cpu().detach().numpy() for i in range(2)]
                    gts_ = [gts[i].cpu().detach().numpy() for i in range(2)]
                    print("{}\n{}".format(preds_[0], gts_[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepLIO Training')

    parser.add_argument('--model_path', default='./model', type=str, help='path to where model')

    # Hyper Params
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optim')
    parser.add_argument('--weight_decay', default=0.0001, type=float, help='weight decay for optim')
    parser.add_argument('--lr_step', default=1000, type=int, help='number of lr step')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='gamma for lr scheduler')

    parser.add_argument('--epochs', default=5, type=int, help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='number of epoch to start learning')
    parser.add_argument('--pretrain', default=False, type=bool, help='Whether or not to pretrain')
    parser.add_argument('--resume', default=False, type=bool, help='Whether or not to resume')

    # Device Option
    parser.add_argument('--gpu_ids', dest='gpu_ids', default=[0, 1], nargs="+", type=int, help='which gpu you use')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='mini-batch size')

    args = parser.parse_args()

    with open("../config.yaml") as f:
        cfg = yaml.safe_load(f)

    trainer = Trainer(cfg)
    trainer.train()




