import torch


def deeplio_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    # merging images in all batches
    data_tmp = batch[0]['data']
    has_imgs = 'images' in data_tmp
    has_imus = 'imus' in data_tmp

    res ={}
    if has_imgs:
        images = torch.stack([b['data']['images'] for b in batch])
        untrans_images = torch.stack([b['data']['untrans-images'] for b in batch])
        res['images'] = images
        res['untrans-images'] = untrans_images

        #images = torch.stack([b['data']['camera-images'] for b in batch])
        #res['camera-images'] = images

    if has_imus:
        # IMU and ground-truth measurments can have different length btw. each pair of lidar frames,
        # so we do not change their size and let them as their are
        imus = torch.stack([b['data']['imus'] for b in batch])
        valids = [b['data']['valids'] for b in batch]
        res['imus'] = imus
        res['valids'] = valids

    gts = torch.stack([b['gts'] for b in batch])
    metas = [b['meta'] for b in batch]

    res['gts'] = gts
    res['metas'] = metas
    return res

