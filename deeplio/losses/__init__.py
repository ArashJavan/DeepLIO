from .losses import HWSLoss, LWSLoss, GeometricConsistencyLoss


def get_loss_function(cfg, device):
    loss_cfg = cfg['losses']
    loss_name = loss_cfg['active'].lower()
    loss_type = loss_cfg.get(loss_name, {})
    params = loss_type.get('params', {})

    loss_type = loss_cfg['loss-type'].lower()
    # check if have both local and global loss
    if "+" in loss_type:
        loss_types = [True, True]
    elif loss_type == "global":
        loss_types = [False, True]
    elif loss_type == "local":
        loss_types = [True, False]
    else:
        raise ValueError("Wrong loss type selected!")

    if loss_name == 'hwsloss':
        learn_smooth = params.get('learn', False)
        sx = params.get('sx', 0.)
        sq = params.get('sq', -2.5)
        return HWSLoss(sx=sx, sq=sq, learn_hyper_params=learn_smooth, device=device, loss_Types=loss_types)
    elif loss_name == 'lwsloss':
        beta = params.get('beta', 1125.)
        return LWSLoss(beta=beta, loss_Types=loss_types)
    else:
        raise ValueError("Loss {} is not supported!".format(loss_name))
