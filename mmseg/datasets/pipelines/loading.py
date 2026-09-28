"""Segmentation-mask loading."""

import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES
from .mask_utils import normalize_binary_mask


@PIPELINES.register_module()
class LoadAnnotations:
    """Load a binary segmentation mask for training."""

    def __init__(self,
                 reduce_zero_label=False,
                 binary_mask=True,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.binary_mask = binary_mask
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        filename = results['ann_info']['seg_map']
        if results.get('mask_prefix') is not None:
            filename = osp.join(results['mask_prefix'], filename)
        content = self.file_client.get(filename)
        target = mmcv.imfrombytes(
            content, flag='unchanged', backend=self.imdecode_backend)
        if target is None:
            raise FileNotFoundError(f'Unable to decode mask: {filename}')
        target = target.squeeze().astype(np.uint8)
        if self.binary_mask:
            target = normalize_binary_mask(target)
        if self.reduce_zero_label:
            target[target == 0] = 255
            target = target - 1
            target[target == 254] = 255
        results['gt_semantic_seg'] = target
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'reduce_zero_label={self.reduce_zero_label}, '
                f'binary_mask={self.binary_mask}, '
                f"imdecode_backend='{self.imdecode_backend}')")


__all__ = ['LoadAnnotations']
