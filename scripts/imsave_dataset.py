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
        ims = data['untrans-images'].detach().cpu()
        for b in range(len(ims)):
            metas = data['metas'][b]
            index = metas['index'][0]
            date = metas['date'][0]
            drive = metas['drive'][0]

            imgs_batch = ims[b, :, 0:3, :, :]
            #imgs_depth = torch.norm(imgs_batch, p=2, dim=1)
            #ims_depth = [F.pad(im, (0, 0, 10, 0)) for im in imgs_depth]
            #ims_depth = torch.cat(ims_depth, dim=0)
            #imgs_xyz = torch.norm(imgs_batch, p=2, dim=1)
            ims_xyz = [F.pad(im, (0, 0, 10, 0)) for im in imgs_batch]
            ims_xyz = torch.cat(ims_xyz, dim=1)
            ims_xyz = torch.abs(ims_xyz.permute(1,2, 0))

            imgs_normals = ims[b, :, 3:, :, :]
            #imgs_normals = torch.norm(imgs_normals, p=1, dim=1)
            imgs_normals = [F.pad(im, (0, 0, 10, 0)) for im in imgs_normals]
            imgs_normals = torch.cat(imgs_normals, dim=1)
            imgs_normals = imgs_normals.permute(1,2, 0)

            fig, ax = plt.subplots(3, 1, figsize=(15, 7))
            ax[0].set_title("XYZ, mean:{:.4f}, std:{:.4f}".format(ims_xyz.mean(), ims_xyz.std()), fontsize=5)
            ax[0].axis('off')
            imd = ax[0].imshow(torch.abs(ims_xyz) / ims_xyz.max())
            #fig.colorbar(imd, ax=ax[0])

            ax[1].set_title("Normal, mean:{:.4f}, std:{:.4f}".format(imgs_normals.mean(), imgs_normals.std()), fontsize=5)
            imn = ax[1].imshow((imgs_normals + 1.) / 2.)
            ax[1].axis('off')
            #fig.colorbar(imn, ax=ax[1])

            ax[2].hist(ims_xyz.flatten(), bins=50, alpha=0.5, label="xyz", density=True)
            ax[2].hist(imgs_normals.flatten(), bins=50, alpha=0.5, label="normals", density=True)
            ax[2].legend()

            fname = "{}/{}_{}_{}_{}.png".format(OUTPUT_PATH, idx, index, date, drive)
            #logger.info("saving {}.".format(fname))

            fig.tight_layout()
            fig.savefig(fname, dpi=600)
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




