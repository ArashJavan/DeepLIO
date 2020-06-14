import os

import torch

from deeplio.common.logger import get_app_logger
from .deeplio_nets import DeepIO, DeepLO
from .deeplio_nets import DeepIO, DeepLO, DeepLIO, DeepLIOFusionLayer
from .imu_feat_nets import ImuFeatFC, ImufeatRNN0, ImuFeatRnn1
from .lidar_feat_nets import LidarPointSegFeat, LidarSimpleFeat0, LidarSimpleFeat1
from .odom_feat_nets import OdomFeatFC, OdomFeatRNN

net_logger = get_app_logger()


def get_model(input_shape, cfg, device):
    arch_name = cfg['arch'].lower()

    if arch_name == 'deepio':
        return create_deepio_arch(cfg, device)
    elif arch_name == 'deeplo':
        return create_deeplo_arch(input_shape, cfg, device)
    elif arch_name == 'deeplio':
        return create_deeplio_arch(input_shape, cfg, device)
    else:
        raise ValueError("Netwrok {} is not supported!".format(arch_name))


def create_deepio_arch(cfg, device):
    net_logger.info("creating deepio.")

    # get deepio config
    arch_cfg = cfg['deepio']

    # get deepio feature config
    imu_feat_cfg = arch_cfg['imu-feat-net']
    imu_feat_name = imu_feat_cfg['name'].lower()

    if arch_cfg['pretrained'] and imu_feat_cfg['pretrained']:
        raise ValueError("Either the whole network or the features can be set pretrained not both at the same time!")

    # create deepio net
    net = DeepIO(cfg)

    net_logger.info("creating deepio imu feature net ({}).".format(imu_feat_name))
    # create feature net
    if imu_feat_name == 'imu-feat-fc':
        imu_feat_net = ImuFeatFC(cfg[imu_feat_name])
    elif imu_feat_name == 'imu-feat-rnn':
        imu_feat_net = ImufeatRNN0(cfg[imu_feat_name])
    else:
        raise ValueError("Wrong feature network {}".format(imu_feat_name))

    # assigning feature net to deepio
    net.imu_feat_net = imu_feat_net
    net.initialize()
    net.to(device=device)

    # loading params
    if arch_cfg['pretrained']:
        model_path = arch_cfg['model-path']
        load_state_dict(net, model_path)
        net.pretrained = True
    elif imu_feat_cfg['pretrained']:
        model_path = imu_feat_cfg['model-path']
        load_state_dict(net.imu_feat_net, model_path)
        net.imu_feat_net.pretrained = True
        net.pretrained = True
    return net


def create_deeplo_arch(input_shape, cfg, device):
    net_logger.info("creating deeplo.")

    arch_cfg = cfg['deeplo']

    # get deeplo feature config
    lidar_feat_cfg = arch_cfg['lidar-feat-net']
    lidar_feat_name = lidar_feat_cfg['name'].lower()

    # get deeplo odometry feature config
    odom_feat_cfg = arch_cfg['odom-feat-net']
    odom_feat_name = odom_feat_cfg['name'].lower()

    pretrained = [arch_cfg['pretrained'], lidar_feat_cfg['pretrained'], odom_feat_cfg['pretrained']]
    if arch_cfg['pretrained'] and (sum(pretrained * 1) > 1):
        raise ValueError("Either the whole network or the features can be set pretrained not both at the same time!")

    # create deepio net
    net = DeepLO(input_shape, cfg)

    net_logger.info("creating deeplo feature net ({}).".format(lidar_feat_name))
    # create feature net
    if lidar_feat_name == 'lidar-feat-pointseg':
        feat_net = LidarPointSegFeat(input_shape, cfg[lidar_feat_name])
    elif lidar_feat_name == 'lidar-feat-simple-0':
        feat_net = LidarSimpleFeat0(input_shape, cfg[lidar_feat_name])
    elif lidar_feat_name == 'lidar-feat-simple-1':
        feat_net = LidarSimpleFeat1(input_shape, cfg[lidar_feat_name])
    else:
        raise ValueError("Wrong feature network {}".format(lidar_feat_name))
    # get the ouput shape of the feat-net (BxTxN)
    input_shape = feat_net.get_output_shape()

    net_logger.info("creating deeplo odom feature net ({}).".format(odom_feat_name))
    # create odometry feature net
    if odom_feat_name == 'odom-feat-fc':
        odom_feat_net = OdomFeatFC(input_shape[2], cfg[odom_feat_name])
    elif odom_feat_name == 'odom-feat-rnn':
        odom_feat_net = OdomFeatRNN(input_shape[2], cfg[odom_feat_name])
    else:
        raise ValueError("Wrong odometry feature network {}".format(odom_feat_name))

    # assigning feature net to deepio
    net.lidar_feat_net = feat_net
    net.odom_feat_net = odom_feat_net
    net.initialize()
    net.to(device=device)

    # loading params
    if arch_cfg['pretrained']:
        model_path = arch_cfg['model-path']
        load_state_dict(net, model_path)
        net.pretrained = True

    if lidar_feat_cfg['pretrained']:
        model_path = lidar_feat_cfg['model-path']
        load_state_dict(net.lidar_feat_net, model_path)
        net.lidar_feat_net.pretrained = True
        net.pretrained = True

    if odom_feat_cfg['pretrained']:
        model_path = odom_feat_cfg['model-path']
        load_state_dict(net.odom_feat_net, model_path)
        net.odom_feat_net.pretrained = True
        net.pretrained = True
    return net


def create_deeplio_arch(input_shape, cfg, device):
    net_logger.info("creating deeplio.")

    arch_cfg = cfg['deeplio']

    # get lidar feature config
    lidar_feat_cfg = arch_cfg['lidar-feat-net']
    lidar_feat_name = lidar_feat_cfg['name'].lower()

    # get imu feat config
    imu_feat_cfg = arch_cfg['imu-feat-net']
    imu_feat_name = imu_feat_cfg['name'].lower()

    # get fusion layer
    fusion_feat_name = arch_cfg['fusion-net'].lower()

    # get odometry feature config
    odom_feat_cfg = arch_cfg['odom-feat-net']
    odom_feat_name = odom_feat_cfg['name'].lower()

    pretrained = [arch_cfg['pretrained'], lidar_feat_cfg['pretrained'],
                  imu_feat_cfg['pretrained'], odom_feat_cfg['pretrained']]
    if arch_cfg['pretrained'] and (sum(pretrained * 1) > 1):
        raise ValueError("Either the whole network or the features can be set pretrained not both at the same time!")

    # create deepio net
    net = DeepLIO(input_shape, cfg)

    net_logger.info("creating deeplio lidar feature net ({}).".format(lidar_feat_name))
    # create feature net
    if lidar_feat_name == 'lidar-feat-pointseg':
        lidar_feat_net = LidarPointSegFeat(input_shape, cfg[lidar_feat_name])
    elif lidar_feat_name == 'lidar-feat-simple-0':
        lidar_feat_net = LidarSimpleFeat0(input_shape, cfg[lidar_feat_name])
    elif lidar_feat_name == 'lidar-feat-simple-1':
        lidar_feat_net = LidarSimpleFeat1(input_shape, cfg[lidar_feat_name])
    else:
        raise ValueError("Wrong feature network {}".format(lidar_feat_name))
    lidar_outshape = lidar_feat_net.get_output_shape()

    net_logger.info("creating deeplio imu feature net ({}).".format(imu_feat_name))
    # create feature net
    if imu_feat_name == 'imu-feat-fc':
        imu_feat_net = ImuFeatFC(cfg[imu_feat_name])
    elif imu_feat_name == 'imu-feat-rnn':
        imu_feat_net = ImufeatRNN0(cfg[imu_feat_name])
    else:
        raise ValueError("Wrong feature network {}".format(imu_feat_name))
    imu_outshape = imu_feat_net.get_output_shape()

    net_logger.info("creating deeplio fusion layer ({}).".format(fusion_feat_name))
    # create feature net
    fusion_inshapes = [lidar_outshape, imu_outshape]
    fusion_feat_net = DeepLIOFusionLayer(fusion_inshapes, cfg[fusion_feat_name])
    # get the ouput shape of the feat-net (BxTxN)
    fusion_outshape = fusion_feat_net.get_output_shape()

    net_logger.info("creating deeplio odom feature net ({}).".format(odom_feat_name))
    # create odometry feature net
    if odom_feat_name == 'odom-feat-fc':
        odom_feat_net = OdomFeatFC(fusion_outshape[2], cfg[odom_feat_name])
    elif odom_feat_name == 'odom-feat-rnn':
        odom_feat_net = OdomFeatRNN(fusion_outshape[2], cfg[odom_feat_name])
    else:
        raise ValueError("Wrong odometry feature network {}".format(odom_feat_name))

    # assigning feature net to deepio
    net.lidar_feat_net = lidar_feat_net
    net.imu_feat_net = imu_feat_net
    net.fusion_net = fusion_feat_net
    net.odom_feat_net = odom_feat_net
    net.initialize()
    net.to(device=device)

    # loading params
    if arch_cfg['pretrained']:
        model_path = arch_cfg['model-path']
        load_state_dict(net, model_path)
        net.pretrained = True

    if lidar_feat_cfg['pretrained']:
        model_path = lidar_feat_cfg['model-path']
        load_state_dict(net.lidar_feat_net, model_path)
        net.lidar_feat_net.pretrained = True
        net.pretrained = True

    if imu_feat_cfg['pretrained']:
        model_path = imu_feat_cfg['model-path']
        load_state_dict(net.imu_feat_net, model_path)
        net.imu_feat_net.pretrained = True
        net.pretrained = True

    if odom_feat_cfg['pretrained']:
        model_path = odom_feat_cfg['model-path']
        load_state_dict(net.odom_feat_net, model_path)
        net.odom_feat_net.pretrained = True
        net.pretrained = True
    return net


def load_state_dict(module, model_path):
    net_logger.info("loading {}'s state dict ({}).".format(module.name, model_path))

    if not os.path.isfile(model_path):
        net_logger.error("{}: No model found ({})!".format(module.name, model_path))

    state_dcit = torch.load(model_path, map_location=module.device)
    module.load_state_dict(state_dcit['state_dict'])


