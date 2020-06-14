import argparse
import os
import signal
import sys

import yaml

dname = os.path.abspath(os.path.dirname(__file__))
content_dir = os.path.abspath("{}/..".format(dname))
sys.path.append(dname)
sys.path.append(content_dir)

from deeplio.models.trainer import TrainerDeepIO, TrainerDeepLO, TrainerDeepLIO


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
    parser.add_argument('--lr-decay', '--learning-rate-decay', default=30, type=int,
                        metavar='LR-DECAY-STEP', help='learning rate decay step', dest='lr_decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', default='', type=str,  metavar='PATH',
                        help='evaluate model with given checkpoint on validation set (default: none)')
    parser.add_argument('-c', '--config', default="./config.yaml", help='Path to configuration file')
    parser.add_argument('-d', '--debug', default=False, help='debug logging', action='store_true', dest='debug')
    parser.add_argument('--device', default='cpu', type=str, metavar='DEVICE',
                        help='Device to use [cpu, cuda].')

    signal.signal(signal.SIGINT, signal_handler)

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    arch = cfg['arch'].lower()

    if arch == 'deepio':
        trainer = TrainerDeepIO(args, cfg)
    elif arch == 'deeplo':
        trainer = TrainerDeepLO(args, cfg)
    elif arch == 'deeplio':
        trainer = TrainerDeepLIO(args, cfg)
    trainer.run()

    print("Done!")
