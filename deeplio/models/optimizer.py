import torch
from torch import optim


def create_optimizer(params, cfg, args, **kwargs):
    optim_type = cfg['optimizer'].lower()

    if optim_type == 'sgd':
        return optim.SGD(params, lr=args.lr,  weight_decay=args.weight_decay, **kwargs)
    elif optim_type == 'adam':
        return optim.SGD(params, lr=args.lr,  weight_decay=args.weight_decay, momentum=args.momentum, **kwargs)
    elif optim_type == 'rmsprop':
        return optim.RMSprop(params, lr=args.lr, weight_decay=args.weight_decay, **kwargs)
    else:
        raise ValueError("Optimizer {} not supported!".format(optim_type))
