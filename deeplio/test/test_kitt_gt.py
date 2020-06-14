import argparse
import os
import sys

from torch.utils import tensorboard

dname = os.path.abspath(os.path.dirname(__file__))
content_dir = os.path.abspath("{}/..".format(dname))
sys.path.append(dname)
sys.path.append(content_dir)

from deeplio import datasets as ds
from deeplio.common.utils import *
from deeplio.common.spatial import quaternion_to_rotation_matrix
from deeplio.models.misc import *
from deeplio.models.worker import Worker, worker_init_fn
from deeplio.visualization.utilities import *


import matplotlib.pyplot as plt


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


class TestKittiGt(Worker):
    ACTION="kitti-ds-tester"
    def __init__(self, parser):
        super(TestKittiGt, self).__init__(parser)
        dataset = ds.Kitti(config=self.cfg, transform=None)
        self.train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                            num_workers=self.num_workers,
                                                            shuffle=False,
                                                            worker_init_fn = worker_init_fn,
                                                            collate_fn = ds.deeplio_collate)

        self.post_processor = DataCombiCreater(seq_size=self.seq_size, batch_size=self.batch_size)

    def run(self):
        for idx, data in enumerate(self.train_dataloader):

            print("****** Fetching new Data {} ********".format(idx))

            # prepare data
            imgs_0, imgs_1, imgs_untrans_0, imgs_untrans_1, imus, gts_local, gts_global = self.post_processor(data)

            # prepare ground truth tranlational and rotational part
            gt_local_x = gts_local[:, :, 0:3].view(-1, 3)
            gt_local_q = gts_local[:, :, 3:7].view(-1, 4)

            imgs_untrans_0 = imgs_untrans_0.numpy()
            imgs_untrans_1 = imgs_untrans_1.numpy()

            for i in range(len(imgs_0)):
                im_trgt = imgs_untrans_0[i].transpose(1, 2, 0)
                im_src = imgs_untrans_1[i].transpose(1, 2, 0)
                T_gt = torch.eye(4)
                T_gt[:3, 3] = gt_local_x[i]
                T_gt[:3, :3] = quaternion_to_rotation_matrix(gt_local_q[i])
                T_gt = T_gt.numpy()

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
                T_init = T_gt
                evaluation = o3d.registration.evaluate_registration(pcd_src, pcd_trgt,
                                                                    0.05, T_init)
                print(evaluation)
                print(T_init)

                draw_registration_result(pcd_src, pcd_trgt, T_init)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepLIO Training')
    parser.add_argument('-c', '--config', default="./config.yaml", help='Path to configuration file')

    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N',
                        help='mini-batch size (default: 1), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-d', '--debug', default=False, help='debug logging', action='store_true', dest='debug')
    parser.add_argument('--device', default='cpu', type=str, metavar='DEVICE',
                        help='Device to use [cpu, cuda].')


    np.set_printoptions(precision=3, suppress=True)
    kitti_gt_test = TestKittiGt(parser)
    kitti_gt_test.run()
    print("Done!")


