import os
import sys
import yaml
import argparse

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
from deeplio.common.logger import PyLogger


SEED = 42


def worker_init_fn(worker_id):
    set_seed(seed=SEED)


class Trainer:
    def __init__(self, args):
        with open(args['config']) as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg
        self.ds_cfg = self.cfg['datasets']
        self.curr_dataset_cfg = self.cfg['datasets'][self.cfg['current-dataset']]
        self.batch_size = self.cfg['batch-size']
        self.seq_size = self.ds_cfg['sequence-size']

        self.n_epoch = self.cfg['epoch']
        num_workers = self.cfg.get('num-workers', 0)

        mean = np.array(self.curr_dataset_cfg['mean'])
        std = np.array(self.curr_dataset_cfg['std'])

        set_seed(seed=SEED)

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

        self.tensor_writer = tensorboard.SummaryWriter(log_dir=self.cfg['log-dir'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        H, W = self.curr_dataset_cfg['image-height'], self.curr_dataset_cfg['image-width']
        C = len(self.cfg['channels'])
        self.model = net.DeepLIOS0(input_shape=(C, H, W), p=0)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = GeoConstLoss()

        self.logger = PyLogger("deeplio_training")

        # debugging and visualizing
        print("DeepLIO Training")
        print("n-workers: {}, batch-size: {}, seq-size:{}, lr:{}, in-shape: ({},{},{})".format(
            num_workers, self.batch_size, self.seq_size, args.lr, C, H, W))
        print(self.train_dataset)
        print(self.val_dataset)

        self.tensor_writer.add_graph(self.model, torch.randn((2, 3, C, H, W)).to(self.device))

    def train(self):
        for epoch in range(self.n_epoch):
            self.logger.info("Starting epoch {}".format(epoch))
            self.train_internal()
        self.logger.info("Training done!")

    def train_internal(self):
        writer = self.tensor_writer
        optimizer = self.optimizer
        criterion = self.criterion
        model = self.model
        model.train()

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
                preds_ = preds.detach().cpu().numpy()
                gts_ = [gts[i].cpu().detach().numpy() for i in range(2)]
                print("{}\n{}".format(preds_, gts_[0]))


if __name__ == '__main__':
    dname = os.path.dirname(__file__)
    content_dir = os.path.abspath("{}/..".format(dname))
    sys.path.append(dname)
    sys.path.append(content_dir)

    parser = argparse.ArgumentParser(description='DeepLIO Training')

    # Hyper Params
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
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

    args = vars(parser.parse_args())

    trainer = Trainer(args)
    trainer.train()




