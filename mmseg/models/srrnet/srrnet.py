# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
"""SRRNet segmentor, encoder wrapper and confidence-aware decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.core import add_prefix
from mmseg.ops import resize

from .. import builder
from ..builder import BACKBONES, HEADS, SEGMENTORS
from ..decode_heads.decode_head import BaseDecodeHead
from ..segmentors.base import BaseSegmentor


@SEGMENTORS.register_module()
class SRRNet(BaseSegmentor):
    """Segmentor implementing Scoring, Remember and Reference."""

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)
        if not self.with_decode_head:
            raise RuntimeError('SRRNet requires a decode head.')

    def _init_decode_head(self, decode_head):
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        if auxiliary_head is None:
            return
        if isinstance(auxiliary_head, list):
            self.auxiliary_head = nn.ModuleList([
                builder.build_head(head_config)
                for head_config in auxiliary_head
            ])
        else:
            self.auxiliary_head = builder.build_head(auxiliary_head)

    def init_weights(self, pretrained=None):
        super().init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            auxiliary_heads = (
                self.auxiliary_head
                if isinstance(self.auxiliary_head, nn.ModuleList)
                else [self.auxiliary_head])
            for auxiliary_head in auxiliary_heads:
                auxiliary_head.init_weights()

    def extract_feat(self, img):
        features = self.backbone(img)
        if self.with_neck:
            features = self.neck(features)
        return features

    def _decode_head_forward_train(self, features, img_metas,
                                   gt_semantic_seg):
        losses = self.decode_head.forward_train(
            features, img_metas, gt_semantic_seg, self.train_cfg)
        return add_prefix(losses, 'decode')

    def _decode_head_forward_test(self, features, img_metas):
        return self.decode_head.forward_test(
            features, img_metas, self.test_cfg)

    def _auxiliary_head_forward_train(self, features, img_metas,
                                      gt_semantic_seg):
        losses = {}
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for index, auxiliary_head in enumerate(self.auxiliary_head):
                auxiliary_losses = auxiliary_head.forward_train(
                    features, img_metas, gt_semantic_seg, self.train_cfg)
                losses.update(add_prefix(auxiliary_losses, f'aux_{index}'))
        else:
            auxiliary_losses = self.auxiliary_head.forward_train(
                features, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(auxiliary_losses, 'aux'))
        return losses

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Compute segmentation and confidence losses."""
        features = self.extract_feat(img)
        losses = self._decode_head_forward_train(
            features, img_metas, gt_semantic_seg)
        if self.with_auxiliary_head:
            losses.update(self._auxiliary_head_forward_train(
                features, img_metas, gt_semantic_seg))
        return losses

    def encode_decode(self, img, img_metas):
        """Encode an 11-channel SRR input and decode its two outputs."""
        features = self.extract_feat(img)
        logits, confidence = self._decode_head_forward_test(
            features, img_metas)
        return logits, torch.sigmoid(confidence)

    def whole_inference(self, img, img_meta, rescale):
        """Run whole-image inference and retain unscaled logits for memory."""
        seg_logit, confidence = self.encode_decode(img, img_meta)
        raw_logit = seg_logit.detach()
        if rescale:
            size = (img.shape[2:] if torch.onnx.is_in_onnx_export()
                    else img_meta[0]['ori_shape'][:2])
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return seg_logit, confidence, raw_logit

    def inference(self, img, img_meta, rescale):
        """Return probabilities, confidence and memory logits."""
        if self.test_cfg['mode'] != 'whole':
            raise NotImplementedError(
                'SRRNet supports whole-image inference only.')
        seg_logit, confidence, raw_logit = self.whole_inference(
            img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)

        if img_meta[0]['flip']:
            flip_direction = img_meta[0]['flip_direction']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))
            else:
                raise ValueError(
                    f'Unsupported flip direction: {flip_direction}')
        return output, confidence, raw_logit

    def simple_test(self, img, img_meta, rescale=True):
        """Run ordered single-frame inference for the temporal state machine."""
        probabilities, confidence, raw_logit = self.inference(
            img, img_meta, rescale)
        prediction = probabilities.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            return prediction.unsqueeze(0)
        prediction = list(prediction.detach().cpu().numpy())

        raw_mask = resize(
            raw_logit,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        raw_mask = F.softmax(raw_mask, dim=1).argmax(dim=1)
        raw_mask = raw_mask.unsqueeze(0).cpu()
        return prediction, confidence, raw_mask

    def aug_test(self, imgs, img_metas, rescale=True):
        raise NotImplementedError(
            'Temporal SRRNet inference requires one ordered image stream and '
            'does not support test-time augmentation.')


@BACKBONES.register_module()
class SRRBackbone(nn.Module):
    """Registry wrapper that preserves the ``backbone.backbone`` key path."""

    def __init__(self, architecture='RMATransformerLarge', **kwargs):
        super().__init__()
        kwargs['type'] = architecture
        # Keep this attribute name for released checkpoint compatibility.
        self.backbone = builder.build_backbone(kwargs)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained)

    def forward(self, x):
        return self.backbone(x)


@HEADS.register_module()
class SRRHead(BaseDecodeHead):
    """Fuse multi-level RMA features and predict mask and confidence maps."""

    def __init__(self, feature_strides, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        if len(feature_strides) != len(self.in_channels):
            raise ValueError(
                'feature_strides and in_channels must have equal lengths.')
        if min(feature_strides) != feature_strides[0]:
            raise ValueError('The first feature level must have minimum stride.')
        self.feature_strides = feature_strides

        c1_channels, c2_channels, c3_channels, c4_channels = self.in_channels
        embedding_dim = kwargs['decoder_params']['embed_dim']

        # These names are part of the released checkpoint state dictionary.
        self.linear_c4 = LinearEmbedding(c4_channels * 3, embedding_dim)
        self.linear_c3 = LinearEmbedding(c3_channels * 3, embedding_dim)
        self.linear_c2 = LinearEmbedding(c2_channels * 3, embedding_dim)
        self.linear_c1 = LinearEmbedding(c1_channels * 3, embedding_dim)
        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=kwargs['norm_cfg'])
        self.conv1 = nn.Conv2d(
            embedding_dim, 256, kernel_size=3, stride=1, padding=1)
        self.linear_pred = nn.Conv2d(256, self.num_classes, kernel_size=1)
        self.mae_pred = nn.Conv2d(
            256 + self.num_classes, 1, kernel_size=1)

    @staticmethod
    def _split_stream_features(features):
        channels = features.shape[1]
        if channels % 3:
            raise ValueError(
                'RMA features must contain three equal channel groups.')
        stream_channels = channels // 3
        reference = features[:, :stream_channels]
        current = features[:, stream_channels:2 * stream_channels]
        previous = features[:, 2 * stream_channels:]
        return reference, current, previous

    @staticmethod
    def _project(embedding, features, output_size=None):
        batch_size = features.shape[0]
        projected = embedding(features).permute(0, 2, 1).reshape(
            batch_size, -1, features.shape[2], features.shape[3])
        if output_size is not None:
            projected = resize(
                projected,
                size=output_size,
                mode='bilinear',
                align_corners=False)
        return projected

    def forward(self, inputs):
        c1, c2, c3, c4 = self._transform_inputs(inputs)
        streams = [self._split_stream_features(level)
                   for level in (c1, c2, c3, c4)]

        # The released decoder concatenates current, previous and reference
        # features at each level in this exact order.
        fused_levels = [
            torch.cat([current, previous, reference], dim=1)
            for reference, current, previous in streams
        ]
        output_size = fused_levels[0].shape[2:]
        projected_c4 = self._project(
            self.linear_c4, fused_levels[3], output_size)
        projected_c3 = self._project(
            self.linear_c3, fused_levels[2], output_size)
        projected_c2 = self._project(
            self.linear_c2, fused_levels[1], output_size)
        projected_c1 = self._project(self.linear_c1, fused_levels[0])

        fused = self.linear_fuse(torch.cat([
            projected_c4, projected_c3, projected_c2, projected_c1
        ], dim=1))
        fused = F.relu(fused)
        if self.dropout is not None:
            fused = self.dropout(fused)
        fused = self.conv1(fused)

        logits = self.linear_pred(fused)
        confidence_input = torch.cat([fused, logits.detach()], dim=1)
        confidence = self.mae_pred(confidence_input)
        return logits, confidence


class LinearEmbedding(nn.Module):
    """Project a feature map to decoder tokens."""

    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.proj(x.flatten(2).transpose(1, 2))


__all__ = ['SRRNet', 'SRRBackbone', 'SRRHead']
