"""Convert processed arrays to model inputs and metadata containers."""

from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer

from ..builder import PIPELINES


def to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    if isinstance(data, int):
        return torch.LongTensor([data])
    if isinstance(data, float):
        return torch.FloatTensor([data])
    raise TypeError(f'Cannot convert {type(data)} to a tensor')


@PIPELINES.register_module()
class ImageToTensor:
    """Convert HWC image arrays to CHW tensors."""

    def __init__(self, keys):
        self.keys = tuple(keys)

    def __call__(self, results):
        for key in self.keys:
            image = results[key]
            if image.ndim < 3:
                image = image[..., None]
            results[key] = to_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1)))
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys})'


@PIPELINES.register_module()
class DefaultFormatBundle:
    """Format training images and segmentation targets for MMCV collation."""

    def __call__(self, results):
        if 'img' in results:
            image = results['img']
            if image.ndim < 3:
                image = image[..., None]
            image = np.ascontiguousarray(image.transpose(2, 0, 1))
            results['img'] = DataContainer(to_tensor(image), stack=True)
        if 'gt_semantic_seg' in results:
            target = results['gt_semantic_seg'][None, ...].astype(np.int64)
            results['gt_semantic_seg'] = DataContainer(
                to_tensor(target), stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class Collect:
    """Collect model inputs and CPU-only frame metadata."""

    DEFAULT_META_KEYS = (
        'filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
        'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
        'sequence_id', 'frame_index', 'is_first', 'ann_filename')

    def __init__(self, keys, meta_keys=DEFAULT_META_KEYS):
        self.keys = tuple(keys)
        self.meta_keys = tuple(meta_keys)

    def __call__(self, results):
        metadata = {key: results.get(key) for key in self.meta_keys}
        data = {
            'img_metas': DataContainer(metadata, cpu_only=True),
        }
        data.update({key: results[key] for key in self.keys})
        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}(keys={self.keys}, '
                f'meta_keys={self.meta_keys})')


__all__ = ['Collect', 'DefaultFormatBundle', 'ImageToTensor', 'to_tensor']
