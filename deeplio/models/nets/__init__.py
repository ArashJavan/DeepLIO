import os
import torch

from deeplio.common.logger import get_app_logger
from .deepio_nets import DeepIO, DeepIOFeat0, DeepIOFeat1, DeepIOFeat11
from .deeplo_nets import DeepLO, DeepLOPointSegFeat, DeepLOOdomFeatFC
from .deeplio_nets import DeepLIONBase


net_logger = get_app_logger()

def get_model(input_shape, cfg, device):
    arch_name = cfg['arch'].lower()

    if arch_name == 'deepio':
        return create_deepio_arch(cfg, device)
    elif arch_name == 'deeplo':
        return create_deeplo_arch(input_shape, cfg, device)
    else:
        raise ValueError("Netwrok {} is not supported!".format(arch_name))


def create_deepio_arch(cfg, device):
    net_logger.info("creating deepio.")

    # get deepio config
    deepio_cfg = cfg['deepio']

    # get deepio feature config
    deepio_feat_cfg = deepio_cfg['feature-net']
    feat_name = deepio_feat_cfg['name'].lower()

    if deepio_cfg['pretrained'] and deepio_feat_cfg['pretrained']:
        raise ValueError("Either the whole network or the features can be set pretrained not both at the same time!")

    # create deepio net
    deepio_net = DeepIO(cfg)

    # create feature net
    if feat_name == 'deepio0':
        feat_net = DeepIOFeat0(cfg['deepio0'])
    elif feat_name == 'deepio1':
        feat_net = DeepIOFeat1(cfg['deepio1'])

    # assigning feature net to deepio
    deepio_net.feat_net = feat_net
    deepio_net.initialize()
    deepio_net.to(device=device)

    # loading params
    if deepio_cfg['pretrained']:
        model_path = deepio_cfg['model-path']
        load_state_dict(deepio_net, model_path)
        deepio_net.pretrained = True
    elif deepio_feat_cfg['pretrained']:
        model_path = deepio_feat_cfg['model-path']
        load_state_dict(deepio_net.feat_net, model_path)
        deepio_net.feat_net.pretrained = True
        deepio_net.pretrained = True
    return deepio_net


def create_deeplo_arch(input_shape, cfg, device):
    net_logger.info("creating deeplo.")

    deeplo_cfg = cfg['deeplo']

    # get deeplo feature config
    feat_cfg = deeplo_cfg['feature-net']
    feat_name = feat_cfg['name'].lower()

    # get deeplo odometry feature config
    odom_feat_cfg = deeplo_cfg['odom-feat-net']
    odom_feat_name = odom_feat_cfg['name'].lower()

    pretrained = [deeplo_cfg['pretrained'], feat_cfg['pretrained'], odom_feat_cfg['pretrained']]
    if deeplo_cfg['pretrained'] and (sum(pretrained * 1) > 1):
        raise ValueError("Either the whole network or the features can be set pretrained not both at the same time!")

    # create deepio net
    deeplo_net = DeepLO(input_shape, cfg)

    net_logger.info("creating deeplo feature net ({}).".format(feat_name))
    # create feature net
    if feat_name == 'deeplo-feat-pointseg':
        feat_net = DeepLOPointSegFeat(input_shape, cfg['deeplo-feat-pointseg'])
    else:
        raise ValueError("Wrong feature network {}".format(feat_name))


    net_logger.info("creating deeplo odom feature net ({}).".format(odom_feat_name))
    input_shape = feat_net.get_output_shape()
    # create odometry feature net
    if odom_feat_name == 'deeplo-odom-feat-fc':
        odom_feat_net = DeepLOOdomFeatFC(input_shape, cfg['deeplo-feat-pointseg'])
    else:
        raise ValueError("Wrong odometry feature network {}".format(odom_feat_name))

    # assigning feature net to deepio
    deeplo_net.feat_net = feat_net
    deeplo_net.odom_feat_net = odom_feat_net
    deeplo_net.initialize()
    deeplo_net.to(device=device)

    # loading params
    if deeplo_cfg['pretrained']:
        model_path = deeplo_cfg['model-path']
        load_state_dict(deeplo_net, model_path)
        deeplo_net.pretrained = True

    if feat_cfg['pretrained']:
        model_path = feat_cfg['model-path']
        load_state_dict(deeplo_net.feat_net, model_path)
        deeplo_net.feat_net.pretrained = True
        deeplo_net.pretrained = True

    if odom_feat_cfg['pretrained']:
        model_path = odom_feat_cfg['model-path']
        load_state_dict(deeplo_net.odom_feat_net, model_path)
        deeplo_net.odom_feat_net.pretrained = True
        deeplo_net.pretrained = True

    return deeplo_net

def load_state_dict(module, model_path):
    net_logger.info("loading {}'s state dict ({}).".format(module.name, model_path))

    if not os.path.isfile(model_path):
        net_logger.error("{}: No model found ({})!".format(module.name, model_path))

    state_dcit = torch.load(model_path, map_location=module.device)
    module.load_state_dict(state_dcit['state_dict'])


