"""Single-scale wrapper required by the MMSeg test call convention."""

from ..builder import PIPELINES
from .compose import Compose


@PIPELINES.register_module()
class SingleScaleTestPipeline:
    """Apply one deterministic test pipeline and wrap outputs as lists."""

    def __init__(self, transforms, img_scale):
        if not (isinstance(img_scale, tuple) and len(img_scale) == 2):
            raise TypeError('img_scale must be a (width, height) tuple')
        self.transforms = Compose(transforms)
        self.img_scale = img_scale

    def __call__(self, results):
        results = dict(results)
        results.update(
            scale=self.img_scale,
            flip=False,
            flip_direction='horizontal')
        data = self.transforms(results)
        return {key: [value] for key, value in data.items()}

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'transforms={self.transforms}, img_scale={self.img_scale})')


__all__ = ['SingleScaleTestPipeline']
