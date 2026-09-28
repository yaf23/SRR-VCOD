"""Decoder base class used by the SRRNet segmentation and scoring head."""

from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.runner import auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize

from ..builder import build_loss


class BaseDecodeHead(nn.Module, metaclass=ABCMeta):
    """Minimal MMSeg-compatible decoder with SRR score supervision."""

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 decoder_params=None,
                 ignore_index=255,
                 sampler=None,
                 align_corners=False):
        super().__init__()
        del decoder_params
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.sampler = (build_pixel_sampler(sampler, context=self)
                        if sampler is not None else None)

        # These standard MMSeg layers are kept for checkpoint compatibility.
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.dropout = (nn.Dropout2d(dropout_ratio)
                        if dropout_ratio > 0 else None)
        self.fp16_enabled = False

    def extra_repr(self):
        return (f'input_transform={self.input_transform}, '
                f'ignore_index={self.ignore_index}, '
                f'align_corners={self.align_corners}')

    def _init_inputs(self, in_channels, in_index, input_transform):
        if input_transform is not None:
            if input_transform not in {'resize_concat', 'multiple_select'}:
                raise ValueError(f'Unsupported input transform: {input_transform}')
            if not isinstance(in_channels, (list, tuple)):
                raise TypeError('in_channels must be a sequence')
            if not isinstance(in_index, (list, tuple)):
                raise TypeError('in_index must be a sequence')
            if len(in_channels) != len(in_index):
                raise ValueError('in_channels and in_index must have equal length')

        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform == 'resize_concat':
            self.in_channels = sum(in_channels)
        elif input_transform == 'multiple_select':
            self.in_channels = in_channels
        else:
            if not isinstance(in_channels, int) or not isinstance(in_index, int):
                raise TypeError(
                    'Single-input decoders require integer in_channels and in_index')
            self.in_channels = in_channels

    def init_weights(self):
        normal_init(self.conv_seg, mean=0, std=0.01)

    def _transform_inputs(self, inputs):
        if self.input_transform == 'resize_concat':
            selected = [inputs[index] for index in self.in_index]
            selected = [
                resize(
                    input=value,
                    size=selected[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
                for value in selected
            ]
            return torch.cat(selected, dim=1)
        if self.input_transform == 'multiple_select':
            return [inputs[index] for index in self.in_index]
        return inputs[self.in_index]

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Return segmentation logits and pixel-wise error logits."""
        raise NotImplementedError

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        del img_metas, train_cfg
        segmentation_logits, error_logits = self.forward(inputs)
        losses = self.losses(segmentation_logits, gt_semantic_seg)

        foreground_probability = F.softmax(
            segmentation_logits.detach(), dim=1)[:, 1:2]
        target = resize(
            input=gt_semantic_seg.float(),
            size=foreground_probability.shape[2:],
            mode='nearest')
        predicted_error = torch.sigmoid(error_logits)
        losses['loss_conf'] = F.mse_loss(
            predicted_error, torch.abs(foreground_probability - target)) * 5.0
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        del img_metas, test_cfg
        return self.forward(inputs)

    def cls_seg(self, features):
        if self.dropout is not None:
            features = self.dropout(features)
        return self.conv_seg(features)

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        weight = (self.sampler.sample(seg_logit, seg_label)
                  if self.sampler is not None else None)
        seg_label = seg_label.squeeze(1)
        return {
            'loss_seg': self.loss_decode(
                seg_logit,
                seg_label,
                weight=weight,
                ignore_index=self.ignore_index)
        }
