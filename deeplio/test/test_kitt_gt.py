import os
import random
import yaml
import argparse
import multiprocessing

import torch
from torchvision import transforms
from torch.utils import tensorboard

from deeplio.datasets import kitti
from deeplio.datasets import transfromers
from deeplio.common.utils import *
from deeplio.models.misc import *
from deeplio.visualization.utilities import *


import matplotlib.pyplot as plt

SEED = 42

def worker_init_fn(worker_id):
    set_seed(seed=42)


def plot_images(images):
    img1, img2 = images[0], images[1]
    fig, ax = plt.subplots(3, 2)
    for i in range(1):
        for j in range(2):
            ax[i, j].imshow(images[j][i][0, :, :].cpu().numpy())
    fig.show()


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


class TestKittiGt:
    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_size = self.cfg['batch-size']

        self.ds_cfg = self.cfg['datasets']
        self.seq_size = self.ds_cfg['sequence-size']

        self.curr_dataset_cfg = self.cfg['datasets'][self.cfg['current-dataset']]

        # For this test wen need all channels, no mather what is configured originally
        cfg['channels'] = [0, 1, 2, 3, 4, 5]

        set_seed(SEED)

        num_workers = multiprocessing.cpu_count()
        transform = transforms.Compose([transfromers.ToTensor()])
        dataset = kitti.Kitti(config=cfg, transform=transform)
        self.train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                            num_workers=num_workers,
                                                            shuffle=True,
                                                            worker_init_fn = worker_init_fn)

        self.post_processor = PostProcessSiameseData(seq_size=self.seq_size, batch_size=self.batch_size)

    def run(self):
        for idx, data in enumerate(self.train_dataloader):

            # skip unvalid data without ground-truth
            if not torch.all(data['valid']):
                continue

            print("****** Fetching new Data {} ********".format(idx))

            imgs_0, imgs_1, gts, imus = self.post_processor(data)
            imgs_0 = imgs_0.numpy()
            imgs_1 = imgs_1.numpy()
            gts = gts.numpy()
            imus = [imu.numpy() for imu in imus]

            for i in range(len(imgs_0)):
                im_trgt = imgs_0[i].transpose(1, 2, 0)
                im_src = imgs_1[i].transpose(1, 2, 0)
                gt = gts[i]

                pcd_src = convert_velo_img_to_o3d(im_src)
                pcd_trgt = convert_velo_img_to_o3d(im_trgt)

                print("****** Starting New Itration {} - {} ********".format(i, self.post_processor.combinations[i]))
                print("Initial alignment")
                T_init = np.identity(4)
                evaluation = o3d.registration.evaluate_registration(pcd_src, pcd_trgt,
                                                                    0.05, T_init)
                print(evaluation)
                draw_registration_result(pcd_src, pcd_trgt, T_init)

                print("Ground-Truth alignment")
                T_init = gt
                evaluation = o3d.registration.evaluate_registration(pcd_src, pcd_trgt,
                                                                    0.05, T_init)
                print(evaluation)
                print(T_init)

                draw_registration_result(pcd_src, pcd_trgt, T_init)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepLIO Training')
    parser.add_argument('-c', '--config', default="../config.yaml", help='Path to configuration file')

    args = vars(parser.parse_args())

    with open(args['config']) as f:
        cfg = yaml.safe_load(f)

    np.set_printoptions(precision=3, suppress=True)
    kitti_gt_test = TestKittiGt(cfg)
    kitti_gt_test.run()
    print("Done!")


