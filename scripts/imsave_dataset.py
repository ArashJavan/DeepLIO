import argparse
import os
import sys

import matplotlib
import yaml
from tqdm import tqdm

matplotlib.use('agg')
import matplotlib.pyplot as plt

dname = os.path.dirname(os.path.realpath(__file__))
module_dir = os.path.abspath("{}/../deeplio".format(dname))
content_dir = os.path.abspath("{}/..".format(dname))
sys.path.append(dname)
sys.path.append(module_dir)
sys.path.append(content_dir)


import logging
import datetime
from pathlib import Path

import torch.utils.data
import torch.nn.functional as F

from deeplio.common.logger import get_app_logger
from deeplio.datasets import Kitti, deeplio_collate


def main(args):
    with open(args['config']) as f:
        cfg = yaml.safe_load(f)

    ds_type = "train"

    batch_size = 3
    num_workers = 8

    OUTPUT_PATH = "{}/outputs/images".format(content_dir)

    # create directoy to save images
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    # max. number of image swe want to save
    max_num_it = -1

    # Dataset class needs a global logger, so creat it here
    # TODO: Remove dependecy of dataset to global logger, so it can have its own
    flog_name = "{}/{}_{}.log".format(OUTPUT_PATH, "Dataset-Visualization",
                                      datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    logger = get_app_logger(filename=flog_name, level=logging.INFO)

    # create dataset andataloader
    kitti_dataset = Kitti(config=cfg, transform=None, ds_type=ds_type)
    dataloader = torch.utils.data.DataLoader(kitti_dataset, batch_size=batch_size,
                                             num_workers=num_workers,
                                             shuffle=False,
                                             collate_fn=deeplio_collate)
    print("Length of dataset is {}".format(len(dataloader)))

    pbar = tqdm(total=len(dataloader))

    # Iterate through datset and save images
    for idx, data in enumerate(dataloader):
        ims = data['images'].detach().cpu()
        for b in range(len(ims)):
            metas = data['metas'][b]
            index = metas['index'][0]
            date = metas['date'][0]
            drive = metas['drive'][0]

            ims_batch = ims[b]
            ims_depth = [F.pad(im, (0, 0, 10, 0)) for im in ims_batch[:, 0, :, :]]
            ims_depth = torch.cat(ims_depth, dim=0)

            ims_remission = [F.pad(im, (0, 0, 10, 0)) for im in ims_batch[:, 1, :, :]]
            ims_remission = torch.cat(ims_remission, dim=0)

            fig, ax = plt.subplots(3, 1, figsize=(15, 7))
            ax[0].set_title("Depth, mean:{:.4f}, std:{:.4f}".format(ims_depth.mean(), ims_depth.std()), fontsize=5)
            ax[0].axis('off')
            ax[0].imshow(ims_depth)

            ax[1].set_title("Remission, mean:{:.4f}, std:{:.4f}".format(ims_remission.mean(), ims_remission.std()),
                            fontsize=5)
            ax[1].imshow(ims_remission)
            ax[1].axis('off')

            ax[2].hist(ims_depth.flatten(), bins=50, alpha=0.5, label="depth", density=True)
            ax[2].hist(ims_remission.flatten(), bins=50, alpha=0.5, label="remission", density=True)
            ax[2].legend()

            fname = "{}/{}_{}_{}_{}.png".format(OUTPUT_PATH, idx, index, date, drive)
            #logger.info("saving {}.".format(fname))
            fig.savefig(fname, dpi=300)
            plt.close(fig)

        pbar.update(1)
        if max_num_it > 0 and idx > max_num_it:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepLIO Training')
    parser.add_argument('-c', '--config', default="../config.yaml", help='configuration file')

    args = vars(parser.parse_args())
    main(args)

    print("Done!")




