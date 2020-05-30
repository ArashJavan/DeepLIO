from .deeplio_nets import DeepLIOS0, DeepLIO0, DeepLIOS1


def get_model(input_shape, cfg):
    net_name = cfg.get('name', 'deeplios0').lower()

    if net_name == 'deeplios0':
        return DeepLIOS0(input_shape, cfg)
    elif net_name == 'deeplios1':
        return DeepLIOS1(input_shape, cfg)
    elif net_name == 'deeplio0':
        return DeepLIO0(input_shape, cfg)
    else:
        raise ValueError("Netwrok {} is not supported!".format(net_name))
