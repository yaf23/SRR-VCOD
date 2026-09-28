"""Model registries and construction helpers for SRRNet."""

import warnings

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
SEGMENTORS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'Passing train_cfg or test_cfg to build_segmentor is deprecated; '
            'specify them in the model configuration instead.',
            UserWarning,
            stacklevel=2)
    if cfg.get('train_cfg') is not None and train_cfg is not None:
        raise ValueError(
            'train_cfg cannot be specified in both the model configuration '
            'and the build_segmentor argument.')
    if cfg.get('test_cfg') is not None and test_cfg is not None:
        raise ValueError(
            'test_cfg cannot be specified in both the model configuration '
            'and the build_segmentor argument.')
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
