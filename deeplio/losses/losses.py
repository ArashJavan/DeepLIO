import torch


def simple_geo_const_loss(predictions, ground_truth):
    pred_pos = predictions[0]
    pred_ori = predictions[1]
    pred_ori = (pred_ori.T / torch.norm(pred_ori, dim=1)).T

    gt_pos = ground_truth[0]
    gt_ori = ground_truth[1]

    loss_pos = torch.mean(torch.norm(pred_pos - gt_pos))
    loss_ori = torch.mean(torch.norm(pred_ori - gt_ori))
    loss = loss_pos + loss_ori
    return loss

