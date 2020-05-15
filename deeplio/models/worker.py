import os
import yaml
import datetime
import time
import logging
import random
from pathlib import Path

import numpy as np

import torch
from torch.backends import cudnn

from deeplio.common import *

SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def worker_init_fn(worker_id):
    # set_seed(seed=SEED)
    pass


class Worker:
    ACTION = "worker"

    def __init__(self, parser):
        args = parser.parse_args()
        
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        self.dname = os.path.dirname(__file__)
        self.content_dir = os.path.abspath("{}/../..".format(self.dname))

        self.args = args
        self.cfg = cfg
        self.ds_cfg = self.cfg['datasets']
        self.curr_dataset_cfg = self.cfg['datasets'][self.cfg['current-dataset']]
        self.seq_size = self.ds_cfg['sequence-size']

        self.batch_size = self.args.batch_size
        self.num_workers = self.args.workers

        self.mean = np.array(self.curr_dataset_cfg['mean'])
        self.std = np.array(self.curr_dataset_cfg['std'])

        self.device = args.device

        # get input images shape and channels
        crop_height, crop_width = self.curr_dataset_cfg.get('crop-factors', [0, 0])
        self.im_height, self.im_width = self.curr_dataset_cfg['image-height'], self.curr_dataset_cfg['image-width']
        self.im_height_model = self.im_height - (2 * crop_height)
        self.im_width_model = self.im_width - (2 * crop_width)

        self.n_channels = len(self.cfg['channels'])
        
        # create output folder structure
        self.out_dir = "{}/outputs/{}_{}".format(self.content_dir, self.ACTION, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

        self.runs_dir = "{}/outputs/{}_runs/{}".format(self.content_dir, self.ACTION, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        log_dir = self.out_dir
        Path(self.runs_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        torch.cuda.empty_cache()
        cudnn.benchmark = True

        #set_seed(seed=SEED)

        flog_name = "{}/{}_{}.log".format(log_dir, self.ACTION, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        if args.debug:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO
        self.logger = logger.PyLogger(filename=flog_name, level=log_level)
        logger.global_logger = self.logger
        self.is_running = False

    def run(self):
        raise NotImplementedError()

    def close(self):
        self.logger.info("Stopping training ({})!".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        self.is_running = False

        # give some times to porcesses and loops to finish
        time.sleep(0.5)

        self.tensor_writer.close()
        self.logger.close()


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






