import os

import torch

from deeplio.common.logger import get_app_logger
from .deeplio_nets import DeepLIO
from .imu_feat_nets import ImuFeatFC, ImufeatRNN0, ImuFeatRnn1
from .lidar_feat_nets import LidarPointSegFeat, LidarSimpleFeat0, LidarSimpleFeat1
from .odom_feat_nets import OdomFeatFC, OdomFeatRNN
from .fusion_nets import DeepLIOFusionCat, DeepLIOFusionSoft


net_logger = None


def get_model(input_shape, cfg, device):
    global net_logger

    net_logger = get_app_logger()
    return create_deeplio_arch(input_shape, cfg, device)


def create_deeplio_arch(input_shape, cfg, device):
    net_logger.info("creating deeplio.")

    arch_cfg = cfg['deeplio']

    # create deepio net
    net = DeepLIO(input_shape, cfg)

    # create feature net
    lidar_feat_net = create_lidar_feat_net(input_shape, cfg, arch_cfg, device)
    lidar_outshape = lidar_feat_net.get_output_shape() if lidar_feat_net is not None else None
    if lidar_feat_net is not None:
        net_logger.print("\tInput-shape: {}, Output-shape: {}".format(input_shape, lidar_outshape))

    imu_feat_net = create_imu_feat_net(cfg, arch_cfg, device)
    imu_outshape = imu_feat_net.get_output_shape() if imu_feat_net is not None else None
    if imu_feat_net is not None:
        net_logger.print("\tInput-shape: {}, Output-shape: {}".format(6, imu_outshape))

    if lidar_feat_net is not None and imu_feat_net is not None:
        fusion_inshapes = [lidar_outshape, imu_outshape]
        fusion_feat_net = create_fusion_net(fusion_inshapes, cfg, arch_cfg, device)
        fusion_outshape = fusion_feat_net.get_output_shape()
        net_logger.print("\tInput-shape: {}, Output-shape: {}".format(fusion_inshapes, fusion_outshape))
    else:
        fusion_feat_net = None

    if fusion_feat_net is not None:
        odom_inshape = fusion_outshape
    elif lidar_feat_net is not None:
        odom_inshape = lidar_outshape
    elif imu_feat_net is not None:
        odom_inshape = imu_outshape
    else:
        raise ValueError("No input-shape for odometry network is defined, please check you configuration!")

    odom_feat_net = create_odometry_feat_net((odom_inshape[0], odom_inshape[1]+7), cfg, arch_cfg, device)
    odom_outshape = odom_feat_net.get_output_shape() if odom_feat_net is not None else None
    if odom_feat_net is not None:
        net_logger.print("\tInput-shape: {}, Output-shape: {}".format(odom_inshape, odom_outshape))

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
    return net


def create_lidar_feat_net(input_shape, cfg, arch_cfg, device):
    # get lidar feature config
    feat_cfg = arch_cfg['lidar-feat-net']
    feat_name = feat_cfg.get('name', None)

    if feat_name is None:
        return None

    feat_name = feat_name.lower()

    net_logger.info("creating deeplio lidar feature net ({}).".format(feat_name))

    # create feature net
    if feat_name == 'lidar-feat-pointseg':
        feat_net = LidarPointSegFeat(input_shape, cfg[feat_name])
    elif feat_name == 'lidar-feat-simple-0':
        feat_net = LidarSimpleFeat0(input_shape, cfg[feat_name])
    elif feat_name == 'lidar-feat-simple-1':
        feat_net = LidarSimpleFeat1(input_shape, cfg[feat_name])
    else:
        raise ValueError("Wrong feature network {}".format(feat_name))

    feat_net.to(device)

    if feat_cfg['pretrained']:
        model_path = feat_cfg['model-path']
        # TODO: PointSeg is mad eof Encoder and decoder,
        # Load them in a more generic way.
        if feat_name == 'lidar-feat-pointseg' and 'encoder' in model_path:
            net = feat_net.encoder1
            load_state_dict(net, model_path)
        elif feat_name == 'lidar-feat-pointseg' and 'decoder' in model_path:
            net = feat_net.decoder
            load_state_dict(net, model_path)
        else:
            load_state_dict(feat_net, model_path)
        feat_net.pretrained = True
    return feat_net


def create_imu_feat_net(cfg, arch_cfg, device):
    # get imu feat config
    feat_cfg = arch_cfg['imu-feat-net']
    feat_name = feat_cfg.get('name', None)

    if feat_name is None:
        return None

    feat_name = feat_name.lower()

    net_logger.info("creating deeplio imu feature net ({}).".format(feat_name))

    # create feature net
    if feat_name == 'imu-feat-fc':
        feat_net = ImuFeatFC(cfg[feat_name])
    elif feat_name == 'imu-feat-rnn':
        feat_net = ImufeatRNN0(cfg[feat_name])
    else:
        raise ValueError("Wrong feature network {}".format(feat_name))

    feat_net.to(device)

    if feat_cfg['pretrained']:
        model_path = feat_cfg['model-path']
        load_state_dict(feat_net, model_path)
        feat_net.pretrained = True
    return feat_net


def create_fusion_net(input_shape, cfg, arch_cfg, device):
    # get fusion layer
    feat_cfg = arch_cfg.get('fusion-net', None)
    feat_name = feat_cfg.get('name', None)

    if feat_name is None:
        return None

    feat_name = feat_name.lower()

    net_logger.info("creating deeplio fusion layer ({}).".format(feat_name))

    if feat_name == 'fusion-layer-cat':
        feat_net = DeepLIOFusionCat(input_shape, cfg[feat_name])
    elif feat_name == 'fusion-layer-soft':
        feat_net = DeepLIOFusionSoft(input_shape, cfg[feat_name])
    else:
        raise ValueError("Wrong feature network {}".format(feat_name))
    return feat_net


def create_odometry_feat_net(input_shape, cfg, arch_cfg, device):
    # get lidar feature config
    feat_cfg = arch_cfg['odom-feat-net']
    feat_name = feat_cfg.get('name', None)

    if feat_name is None:
        return None

    feat_name = feat_name.lower()

    net_logger.info("creating deeplio odom feature net ({}).".format(feat_name))

    # create feature net
    if feat_name == 'odom-feat-fc':
        feat_net = OdomFeatFC(input_shape[2], cfg[feat_name])
    elif feat_name == 'odom-feat-rnn':
        feat_net = OdomFeatRNN(input_shape[-1], cfg[feat_name])
    else:
        raise ValueError("Wrong odometry feature network {}".format(feat_name))

    feat_net.to(device)

    if feat_cfg['pretrained']:
        model_path = feat_cfg['model-path']
        load_state_dict(feat_net, model_path)
        feat_net.pretrained = True
    return feat_net


def load_state_dict(module, model_path):
    net_logger.info("loading {}'s state dict ({}).".format(module.name, model_path))

    if not os.path.isfile(model_path):
        net_logger.error("{}: No model found ({})!".format(module.name, model_path))

    state_dcit = torch.load(model_path, map_location=module.device)
    module.load_state_dict(state_dcit['state_dict'])


