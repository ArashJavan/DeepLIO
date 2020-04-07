import copy
import numpy as np

import open3d as o3d

from deeplio.datasets.kitti import KittiRawData


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def proj_torch_imge_to_3d(image):
    xyz = image[:3, :, :].numpy()
    xyz = xyz.transpose(1, 2, 0).reshape(-1, 3) * KittiRawData.MAX_DIST_HDL64
    indices = np.where(np.all(xyz == [0., 0., 0.], axis=1))[0]
    xyz = np.delete(xyz, indices, axis=0)
    return xyz


def draw_image_3d(image):
    xyz = proj_torch_imge_to_3d(image)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])


def show_image(image):
    plt.imshow(image, cmap="jet")
    plt.show()

