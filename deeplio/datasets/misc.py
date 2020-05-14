import torch


def deeplio_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    # merging images in all batches
    try:
        images = torch.stack([b['images'] for b in batch])
    except TypeError as ex:
        print(ex)

    untrans_images = torch.stack([b['untrans-images'] for b in batch])

    # IMU and ground-truth measurments can have different length btw. each pair of lidar frames,
    # so we do not change their size and let them as their are
    imus = [b['imus'] for b in batch]
    gts = [b['gts'] for b in batch]

    # also vladiation field and meta-datas need not to be changed
    valids = torch.tensor([b['valid'] for b in batch])
    metas = [b['meta'] for b in batch]

    res ={}
    res['images'] = images
    res['untrans-images'] = untrans_images
    res['imus'] = imus
    res['gts'] = gts
    res['valid'] = valids
    res['metas'] = metas
    return res

