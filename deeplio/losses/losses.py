import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplio.common.spatial import normalize_quaternion, quaternion_to_rotation_matrix, \
    convert_points_to_homogeneous, convert_points_from_homogeneous


class LWSLoss(nn.Module):
    """Linear weighted sum loss
    """
    def __init__(self, beta=1125., gamma=1., loss_Types=[True, True]):
        super(LWSLoss, self).__init__()
        self.loss_fn = nn.MSELoss()
        self.beta = beta
        self.gamma = gamma
        self.loss_Types = loss_Types

    def forward(self, pred_f2f_x, pred_f2f_r, pred_f2g_x, pred_f2g_r,
                gt_f2f_x, gt_f2f_r, gt_f2g_x, gt_f2g_q):

        if self.loss_Types[0]:
            L_t = F.mse_loss(pred_f2f_x, gt_f2f_x)
            L_w = F.mse_loss(pred_f2f_r, gt_f2f_r)
        else:
            L_t = 0.
            L_w = 0.

        if self.loss_Types[1]:
            L_p = F.mse_loss(pred_f2g_x, gt_f2g_x)
            L_q = F.mse_loss(pred_f2g_r, gt_f2g_q)
        else:
            L_p = 0.
            L_q = 0.

        loss = (L_p + L_t) + self.beta * (L_q + L_w)
        return loss

    def __repr__(self):
        if self.loss_Types[0] and self.loss_Types[1]:
            return "HWSLoss with f2f and f2g loss."
        elif self.loss_Types[0]:
            return "HWSLoss with only f2f loss."
        elif self.loss_Types[1]:
            return "HWSLoss with only f2g loss."
        else:
            return "Wrong loss combination!"

class HWSLoss(nn.Module):
    """Homoscedastic weighted Loss
    """
    def __init__(self, sx=0., sq=-2.5, learn_hyper_params=True, device="cpu", loss_Types=[True, True]):
        """
        :param sx:
        :param sq:
        :param learn_hyper_params: learning the smoothnes terms during training
        """
        super(HWSLoss, self).__init__()
        self.learn_hyper_params = learn_hyper_params
        self.loss_Types = loss_Types

        self.sx = torch.nn.Parameter(torch.tensor(sx, device=device, requires_grad=learn_hyper_params))
        self.sq = torch.nn.Parameter(torch.tensor(sq, device=device, requires_grad=learn_hyper_params))
        self.loss_fn = nn.MSELoss()

    def forward(self, pred_f2f_x, pred_f2f_r, pred_f2g_x, pred_f2g_r,
                gt_f2f_x, gt_f2f_r, gt_f2g_x, gt_f2g_q):

        if self.loss_Types[0]:
            L_t = F.mse_loss(pred_f2f_x, gt_f2f_x)
            L_w = F.mse_loss(pred_f2f_r, gt_f2f_r)
        else:
            L_t = 0.
            L_w = 0.

        if self.loss_Types[1]:
            L_p = F.mse_loss(pred_f2g_x, gt_f2g_x)
            L_q = F.mse_loss(pred_f2g_r, gt_f2g_q)
        else:
            L_p = 0.
            L_q = 0.

        loss = (L_p + L_t) * torch.exp(-self.sx) + self.sx + (L_q + L_w) * torch.exp(-self.sq) + self.sq
        return loss

    def __repr__(self):
        if self.loss_Types[0] and self.loss_Types[1]:
            return "HWSLoss with f2f and f2g loss."
        elif self.loss_Types[0]:
            return "HWSLoss with only f2f loss."
        elif self.loss_Types[1]:
            return "HWSLoss with only f2g loss."
        else:
            return "Wrong loss combination!"

class GeometricConsistencyLoss(nn.Module):
    def __init__(self, H=64, W=1800, fov_up=3.0, fov_down=-25.0, min_depth=1, max_depth=80):
        super(GeometricConsistencyLoss, self).__init__()
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down

    def forward(self, pred_x, pred_q, pred_mask0, pred_mask1, imgs_0, imgs_1, gt_x, gt_q):
        n_batches = len(pred_x)
        for b in range(n_batches):
            T = torch.eye(4)
            T[:3, :3] = quaternion_to_rotation_matrix(pred_q[b])
            T[:3, 3] = pred_x[b]

            img0 = imgs_0[b, :4]

            xyz_1 = imgs_1[b, :3].view(3, -1).T
            xyz_1 = xyz_1[torch.all(xyz_1 != 0, dim=1)]
            xyz_1 = convert_points_to_homogeneous(xyz_1)
            xyz_1_transferd = torch.matmul(xyz_1, T.T)
            xyz_1_transferd = convert_points_from_homogeneous(xyz_1_transferd)

            img1_xyz_trans, img1_range, _ = self.do_spherical_projection(xyz_1_transferd)
            img1_xyz_trans = img1_xyz_trans.permute(2, 0, 1)
            img1_trans = torch.cat((img1_xyz_trans, img1_range[None, :, :]))

            img0_normals = self.calc_normal(img0)
            img1_normals = self.calc_normal(img1_trans)
            delta_normals = img1_normals - img0_normals

            # derivative w.r.t. alpha and beta
            img1_range_dalpha = img1_range[:, :-1] - img1_range[:, 1:]
            img1_range_dalpha = img1_range_dalpha[1:-1, 1:]
            img1_range_dbeta = img1_range[-1:, :] - img1_range[1:, :]
            img1_range_dbeta = img1_range_dbeta[1:, 1:-1]

            weights = torch.exp(torch.abs(img1_range_dalpha) + torch.abs(img1_range_dbeta))
            l_n = weights * torch.norm(delta_normals, p=1) # * pred_mask1
            l_n = torch.sum(l_n)

            return l_n

    def do_spherical_projection(self, x):
        # projected range image - [H,W] range (-1 is no data)
        proj_range = torch.full((self.proj_H, self.proj_W), 0., dtype=torch.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        proj_xyz = torch.full((self.proj_H, self.proj_W, 3), 0,
                                   dtype=torch.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        proj_idx = torch.full((self.proj_H, self.proj_W), 0,
                                   dtype=torch.int32)

        # mask containing for each pixel, if it contains a point or not
        proj_mask = torch.zeros((self.proj_H, self.proj_W),
                                     dtype=torch.int32)  # [H,W] mask

        points = x

        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * math.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * math.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = torch.norm(points, 2, dim=1)
        # depth_xy = np.linalg.norm(self.points[:, 0:2], 2, axis=1)

        # get scan components
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]

        # get angles of all points
        yaw = -torch.atan2(scan_y, scan_x)
        pitch = torch.asin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / math.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = torch.floor(proj_x)
        proj_x = torch.clamp(proj_x, min=0, max=self.proj_W - 1) # in [0,W-1]

        proj_y = torch.floor(proj_y)
        proj_y = torch.clamp(proj_y, min=0, max=self.proj_H - 1)  # in [0,W-1]  # in [0,H-1]

        # order in decreasing depth
        indices = torch.arange(depth.shape[0])
        order = torch.argsort(depth, descending=True)
        depth = depth[order]
        indices = indices[order]
        points = points[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        proj_y_detached = proj_y.type(torch.long)
        proj_x_detached = proj_x.type(torch.long)

        proj_range[proj_y_detached, proj_x_detached] = depth
        proj_xyz[proj_y_detached, proj_x_detached] = points
        proj_idx[proj_y_detached, proj_x_detached] = indices.type(torch.int32)
        proj_mask = (proj_idx > 0).type(torch.int32)
        return proj_xyz, proj_range, proj_mask

    def calc_weights(self, x, alpha=-0.2):
        return torch.exp(alpha * torch.abs(x))

    def calc_normal(self, x):
        # x = torch.FloatTensor([[[7, 8, 1, 4],
        #                         [7, 7, 6, 6],
        #                         [4, 8, 2, 6],
        #                         [5, 2, 2, 3]],
        #
        #                        [[1, 7, 6, 6],
        #                         [4, 4, 3, 8],
        #                         [3, 1, 7, 2],
        #                         [3, 2, 3, 0]],
        #
        #                        [[1, 7, 3, 0],
        #                         [5, 2, 6, 3],
        #                         [2, 9, 2, 7],
        #                         [1, 7, 6, 9]]])
        # x_range = torch.norm(x, dim=0)
        # x = torch.cat((x, x_range[None, :, :]), dim=0)

        diff_vertical = x[:, :-1, :] - x[:, 1:, :]
        diff_horizontal = x[:, :, :-1] - x[:, :, 1:]

        x_diff_top = diff_vertical[:, :-1, 1:-1]
        x_diff_bottom = -diff_vertical[:, 1:, 1:-1]

        x_diff_left = diff_horizontal[:, 1:-1, :-1]
        x_diff_right = -diff_horizontal[:, 1:-1, 1:]

        x_range_diffs = torch.stack((x_diff_top[3], x_diff_left[3], x_diff_bottom[3], x_diff_right[3]))
        weights = self.calc_weights(x_range_diffs)

        x_norm_tl = torch.cross(weights[0] * x_diff_top[:3], weights[1] * x_diff_left[:3])
        x_norm_lb = torch.cross(weights[1] * x_diff_left[:3], weights[2] * x_diff_bottom[:3])
        x_norm_br = torch.cross(weights[2] * x_diff_bottom[:3], weights[3] * x_diff_right[:3])
        x_norm_rt = torch.cross(weights[3] * x_diff_right[:3], weights[0] * x_diff_top[:3])

        x_normals = torch.stack((x_norm_tl, x_norm_lb, x_norm_br, x_norm_rt))
        x_normals = torch.sum(x_normals, dim=0)
        return x_normals
