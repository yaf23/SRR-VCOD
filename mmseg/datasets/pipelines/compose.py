"""Composable data transforms."""

from collections.abc import Sequence

from mmcv.utils import build_from_cfg

from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose:
    """Apply configured transforms in order."""

    def __init__(self, transforms):
        if not isinstance(transforms, Sequence):
            raise TypeError('transforms must be a sequence')
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                built_transform = build_from_cfg(transform, PIPELINES)
            elif not callable(transform):
                raise TypeError('Each transform must be callable or a config')
            else:
                built_transform = transform
            self.transforms.append(built_transform)

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        transforms = '\n'.join(f'    {transform}'
                               for transform in self.transforms)
        return f'{self.__class__.__name__}(\n{transforms}\n)'
