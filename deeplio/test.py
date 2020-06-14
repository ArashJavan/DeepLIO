import argparse
import os
import signal
import sys

import yaml

dname = os.path.abspath(os.path.dirname(__file__))
content_dir = os.path.abspath("{}/..".format(dname))
sys.path.append(dname)
sys.path.append(content_dir)

from deeplio.models.tester import TesterDeepIO, TesterDeepLO, TesterDeepLIO


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
    parser.add_argument('-c', '--config', default="./config.yaml", help='Path to configuration file')
    parser.add_argument('-d', '--debug', default=False, help='debug logging', action='store_true', dest='debug')
    parser.add_argument('--plot', default=False, help='plot the results', action='store_true', dest='plot')
    parser.add_argument('--device', default='cpu', type=str, metavar='DEVICE',
                        help='Device to use [cpu, cuda].')
    parser.add_argument('--param', default='xq', type=str, help='Which parameter to predict (default: xq) [x, xq]', dest="param")

    signal.signal(signal.SIGINT, signal_handler)

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    arch = cfg['arch'].lower()

    if arch == 'deepio':
        tester = TesterDeepIO(args, cfg)
    elif arch == 'deeplo':
        tester = TesterDeepLO(args, cfg)
    elif arch == 'deeplio':
        tester = TesterDeepLIO(args, cfg)

    tester.run()

    print("Done!")
