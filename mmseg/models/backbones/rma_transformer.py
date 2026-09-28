# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
"""Reference-memory attention transformer used by SRRNet."""

import math
from functools import partial

import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger, split_temporal_input


class DepthwiseStreamConv(nn.Module):
    """Apply one shared depthwise convolution to all three token streams."""

    def __init__(self, dim=768):
        super().__init__()
        # Keep this attribute name for compatibility with released checkpoints.
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, bias=True,
            groups=dim)

    def forward(self, x, height, width):
        batch_size, token_count, channels = x.shape
        if token_count % 3:
            raise ValueError(
                'Reference, previous-frame and current-frame token streams '
                'must have equal lengths.')

        stream_length = token_count // 3
        streams = x.split(stream_length, dim=1)
        outputs = []
        for stream in streams:
            stream = stream.transpose(1, 2).reshape(
                batch_size, channels, height, width)
            stream = self.dwconv(stream)
            outputs.append(stream.flatten(2).transpose(1, 2))
        return torch.cat(outputs, dim=1)


class MixFeedForward(nn.Module):
    """Transformer feed-forward layer with spatial depthwise convolution."""

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # The nested ``dwconv.dwconv`` path is part of checkpoint state keys.
        self.dwconv = DepthwiseStreamConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = (module.kernel_size[0] * module.kernel_size[1]
                       * module.out_channels)
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, height, width):
        x = self.fc1(x)
        x = self.dwconv(x, height, width)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class ReferenceMemoryAttention(nn.Module):
    """RMA attention over reference, previous and current frame streams.

    The reference stream attends to itself, the previous-frame stream attends
    to itself and the reference, and the current-frame stream attends to all
    three streams.
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 sr_ratio=1):
        super().__init__()
        if dim % num_heads:
            raise ValueError(
                f'Embedding dimension {dim} is not divisible by '
                f'{num_heads} attention heads.')

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = (module.kernel_size[0] * module.kernel_size[1]
                       * module.out_channels)
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def _reduced_kv(self, stream, batch_size, channels, height, width):
        stream = stream.permute(0, 2, 1).reshape(
            batch_size, channels, height, width)
        stream = self.sr(stream).reshape(
            batch_size, channels, -1).permute(0, 2, 1)
        stream = self.norm(stream)
        return self.kv(stream).reshape(
            batch_size, -1, 2, self.num_heads,
            channels // self.num_heads).permute(2, 0, 3, 1, 4)

    def forward(self, x, height, width):
        batch_size, token_count, channels = x.shape
        if token_count % 3:
            raise ValueError(
                'Reference, previous-frame and current-frame token streams '
                'must have equal lengths.')

        stream_length = token_count // 3
        query = self.q(x).reshape(
            batch_size, token_count, self.num_heads,
            channels // self.num_heads).permute(0, 2, 1, 3)
        reference_query, previous_query, current_query = query.split(
            stream_length, dim=2)

        reference, previous, current = x.split(stream_length, dim=1)
        if self.sr_ratio > 1:
            reference_kv = self._reduced_kv(
                reference, batch_size, channels, height, width)
            previous_kv = self._reduced_kv(
                previous, batch_size, channels, height, width)
            current_kv = self._reduced_kv(
                current, batch_size, channels, height, width)
            key_value = torch.cat(
                [reference_kv, previous_kv, current_kv], dim=3)
            reduced_lengths = (
                reference_kv.shape[3], previous_kv.shape[3],
                current_kv.shape[3])
        else:
            key_value = self.kv(x).reshape(
                batch_size, -1, 2, self.num_heads,
                channels // self.num_heads).permute(2, 0, 3, 1, 4)
            reduced_length = key_value.shape[3] // 3
            reduced_lengths = (reduced_length,) * 3

        key, value = key_value[0], key_value[1]
        reference_key, previous_key, _ = key.split(
            reduced_lengths, dim=2)
        reference_value, previous_value, _ = value.split(
            reduced_lengths, dim=2)

        reference_attention = (
            reference_query @ reference_key.transpose(-2, -1)) * self.scale
        reference_attention = self.attn_drop(
            reference_attention.softmax(dim=-1))
        reference_output = (
            reference_attention @ reference_value).transpose(1, 2).reshape(
                batch_size, stream_length, channels)

        memory_key = torch.cat([reference_key, previous_key], dim=2)
        memory_value = torch.cat([reference_value, previous_value], dim=2)
        previous_attention = (
            previous_query @ memory_key.transpose(-2, -1)) * self.scale
        previous_attention = self.attn_drop(
            previous_attention.softmax(dim=-1))
        previous_output = (
            previous_attention @ memory_value).transpose(1, 2).reshape(
                batch_size, stream_length, channels)

        current_attention = (
            current_query @ key.transpose(-2, -1)) * self.scale
        current_attention = self.attn_drop(current_attention.softmax(dim=-1))
        current_output = (
            current_attention @ value).transpose(1, 2).reshape(
                batch_size, stream_length, channels)

        output = torch.cat(
            [reference_output, previous_output, current_output], dim=1)
        output = self.proj(output)
        return self.proj_drop(output)


class RMABlock(nn.Module):
    """Transformer block containing reference-memory attention."""

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ReferenceMemoryAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio)
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity())
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixFeedForward(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = (module.kernel_size[0] * module.kernel_size[1]
                       * module.out_channels)
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, height, width):
        x = x + self.drop_path(
            self.attn(self.norm1(x), height, width))
        return x + self.drop_path(self.mlp(self.norm2(x), height, width))


class OverlapPatchEmbed(nn.Module):
    """Convert an image-like feature map to overlapping patch tokens."""

    def __init__(self,
                 img_size=224,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H = img_size[0] // patch_size[0]
        self.W = img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = (module.kernel_size[0] * module.kernel_size[1]
                       * module.out_channels)
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), height, width


class RMATransformer(nn.Module):
    """Four-stage transformer for SRRNet's three visual streams."""

    def __init__(self,
                 img_size=512,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=(64, 128, 256, 512),
                 num_heads=(1, 2, 4, 8),
                 mlp_ratios=(4, 4, 4, 4),
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm,
                 depths=(3, 4, 6, 3),
                 sr_ratios=(8, 4, 2, 1)):
        super().__init__()
        del patch_size
        self.num_classes = num_classes
        self.depths = depths

        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0])
        self.patch_embed_ref1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=4,
            embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3])

        drop_path_rates = [
            value.item()
            for value in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        offset = 0
        self.block1 = self._make_stage(
            embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias,
            qk_scale, drop_rate, attn_drop_rate, drop_path_rates, offset,
            norm_layer, depths[0], sr_ratios[0])
        self.norm1 = norm_layer(embed_dims[0])

        offset += depths[0]
        self.block2 = self._make_stage(
            embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias,
            qk_scale, drop_rate, attn_drop_rate, drop_path_rates, offset,
            norm_layer, depths[1], sr_ratios[1])
        self.norm2 = norm_layer(embed_dims[1])

        offset += depths[1]
        self.block3 = self._make_stage(
            embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias,
            qk_scale, drop_rate, attn_drop_rate, drop_path_rates, offset,
            norm_layer, depths[2], sr_ratios[2])
        self.norm3 = norm_layer(embed_dims[2])

        offset += depths[2]
        self.block4 = self._make_stage(
            embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias,
            qk_scale, drop_rate, attn_drop_rate, drop_path_rates, offset,
            norm_layer, depths[3], sr_ratios[3])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    @staticmethod
    def _make_stage(dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    drop_rate,
                    attn_drop_rate,
                    drop_path_rates,
                    offset,
                    norm_layer,
                    depth,
                    sr_ratio):
        return nn.ModuleList([
            RMABlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rates[offset + index],
                norm_layer=norm_layer,
                sr_ratio=sr_ratio)
            for index in range(depth)
        ])

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = (module.kernel_size[0] * module.kernel_size[1]
                       * module.out_channels)
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_checkpoint(
                self,
                pretrained,
                map_location='cpu',
                strict=False,
                logger=get_root_logger())

    @staticmethod
    def _tokens_to_feature(tokens, batch_size, height, width):
        return tokens.reshape(
            batch_size, height, width, -1).permute(
                0, 3, 1, 2).contiguous()

    def _run_stage(self,
                   reference,
                   current,
                   previous,
                   patch_embed,
                   blocks,
                   norm):
        batch_size = current.shape[0]
        reference_tokens, reference_height, reference_width = patch_embed(
            reference)
        current_tokens, current_height, current_width = patch_embed(current)
        previous_tokens, previous_height, previous_width = patch_embed(
            previous)

        reference_length = reference_height * reference_width
        previous_length = previous_height * previous_width
        current_length = current_height * current_width
        tokens = torch.cat(
            [reference_tokens, previous_tokens, current_tokens], dim=1)
        for block in blocks:
            tokens = block(tokens, current_height, current_width)
        tokens = norm(tokens)

        reference_tokens, previous_tokens, current_tokens = tokens.split(
            [reference_length, previous_length, current_length], dim=1)
        reference = self._tokens_to_feature(
            reference_tokens, batch_size, reference_height, reference_width)
        current = self._tokens_to_feature(
            current_tokens, batch_size, current_height, current_width)
        previous = self._tokens_to_feature(
            previous_tokens, batch_size, previous_height, previous_width)
        output = torch.cat([reference, current, previous], dim=1)
        return reference, current, previous, output

    def forward_features(self, x):
        (current, previous_image, previous_mask, reference_image,
         reference_mask) = split_temporal_input(x)
        previous = torch.cat([previous_image, previous_mask], dim=1)
        reference = torch.cat([reference_image, reference_mask], dim=1)
        batch_size = current.shape[0]

        current_tokens, current_height, current_width = self.patch_embed1(
            current)
        previous_tokens, previous_height, previous_width = (
            self.patch_embed_ref1(previous))
        reference_tokens, reference_height, reference_width = (
            self.patch_embed_ref1(reference))
        reference_length = reference_height * reference_width
        previous_length = previous_height * previous_width
        current_length = current_height * current_width

        tokens = torch.cat(
            [reference_tokens, previous_tokens, current_tokens], dim=1)
        for block in self.block1:
            tokens = block(tokens, current_height, current_width)
        tokens = self.norm1(tokens)
        reference_tokens, previous_tokens, current_tokens = tokens.split(
            [reference_length, previous_length, current_length], dim=1)
        reference = self._tokens_to_feature(
            reference_tokens, batch_size, reference_height, reference_width)
        current = self._tokens_to_feature(
            current_tokens, batch_size, current_height, current_width)
        previous = self._tokens_to_feature(
            previous_tokens, batch_size, previous_height, previous_width)
        outputs = [torch.cat([reference, current, previous], dim=1)]

        for patch_embed, blocks, norm in (
                (self.patch_embed2, self.block2, self.norm2),
                (self.patch_embed3, self.block3, self.norm3),
                (self.patch_embed4, self.block4, self.norm4)):
            reference, current, previous, output = self._run_stage(
                reference, current, previous, patch_embed, blocks, norm)
            outputs.append(output)
        return outputs

    def forward(self, x):
        return self.forward_features(x)


@BACKBONES.register_module()
class RMATransformerLarge(RMATransformer):
    """Large RMA encoder used by the released SRRNet model."""

    def __init__(self):
        super().__init__(
            patch_size=4,
            embed_dims=(64, 128, 320, 512),
            num_heads=(1, 2, 5, 8),
            mlp_ratios=(4, 4, 4, 4),
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=(3, 4, 18, 3),
            sr_ratios=(8, 4, 2, 1),
            drop_rate=0.0,
            drop_path_rate=0.1)


__all__ = ['RMATransformerLarge']
