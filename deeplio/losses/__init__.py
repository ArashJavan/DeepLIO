from .losses import HWSLoss, LWSLoss


def get_loss_function(cfg, device):
    loss_cfg = cfg['losses']
    loss_name = loss_cfg['active'].lower()
    loss_type = loss_cfg[loss_name]
    params = loss_type.get('params', {})

    if loss_name == 'hwsloss':
        learn_smooth = params.get('learn', False)
        sx = params.get('sx', 0.)
        sq = params.get('sq', -2.5)
        return HWSLoss(sx=sx, sq=sq, learn=learn_smooth, device=device)
    elif loss_name == 'lwsloss':
        beta = params.get('beta', 1125.)
        return LWSLoss(beta=beta)
    else:
        raise ValueError("Loss {} is not supported!".format(loss_type))
