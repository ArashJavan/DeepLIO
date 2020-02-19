import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import open3d as o3d
import pykitti

def get_quadrant(point):
    res = 0
    x = point[0]
    y = point[1]
    if x > 0 and y >= 0:
        res = 1
    elif x <= 0 and y > 0:
        res = 2
    elif x < 0 and y <= 0:
        res = 3
    elif x >= 0 and y < 0:
        res = 4
    return res


def add_ring_info(scan_points):
    num_of_points = scan_points.shape[0]
    scan_points = np.hstack([scan_points,
                             np.zeros((num_of_points, 1))])
    velodyne_rings_count = 64
    previous_quadrant = 0
    ring = 0
    for num in range(num_of_points - 1, -1, -1):
        quadrant = get_quadrant(scan_points[num])
        if quadrant == 4 and previous_quadrant == 1 and ring < velodyne_rings_count - 1:
            ring += 1
        scan_points[num, 4] = ring
        previous_quadrant = quadrant
    return scan_points


def arctan2_deg(x1, x2):
    return np.arctan2(x1, x2) * 180. / np.pi


basedir = r'C:\Users\ajava\Datasets\KITTI'
date = '2011_09_26'
drive = '0035'

data = pykitti.raw(basedir, date, drive, frames=range(0, 50))
pcd = data.get_velo(0)
pcd = add_ring_info(pcd)

delta_theta = 0.1
delta_phi = 0.41875
f_h = 360.
f_vu = 2.
f_vl = 24.8
f_v = f_vu + f_vl

W = int(f_h / delta_theta) + 1
H = int(f_v / delta_phi)
C = 1

depth = np.linalg.norm(pcd[:, 0:2], axis=1)
distances = np.linalg.norm(pcd[:, 0:3], axis=1)
intensities = pcd[:, 3]

theta = ((f_h / 2) - arctan2_deg(pcd[:, 1], pcd[:, 0])) / delta_theta
theta_int = np.round(theta).astype(np.int)
phi = pcd[:, 4].astype(np.int)
#phi = (f_vu - arctan2_deg(pcd[:, 2], depth))
#phi[phi < 0] = 0.
#phi /= delta_phi # delta_phi # pcd[:, 4].astype(np.int)
#phi = np.round(phi).astype(np.int)

uv = np.vstack((theta_int, phi)).T

img = np.zeros((H, W))
for i, point in enumerate(pcd[:, 0:3]):
    idx = i
    u = theta_int[idx]
    v = np.abs(phi[idx] - H) - 1

    indices = np.where((uv == (u, phi[i])).all(axis=1))[0]
    if len(indices) > 1:
        r = np.min(distances[indices])
        print("[{}] error: (u, v) : ({}, {}), depth: {}, intensitiy: {}, len-idx: {}".format(i, u, v, r, intensities[idx], len(indices)))
    else:
        r = distances[idx]

    try:
        img[v, u] = r
    except IndexError as ex:
        print(ex)

pcd_o3d = o3d.geometry.PointCloud()
pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, 0:3])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(img, cmap=plt.get_cmap("jet"))
ax.grid(False)
plt.show()

#o3d.visualization.draw_geometries([pcd_o3d])

print(data)