"""Augmentation and normalization for SRRNet frame tuples."""

import numpy as np

from ..builder import PIPELINES
from .transforms import Normalize, PhotoMetricDistortion


def _split_training_tuple(image):
    if image.ndim != 3 or image.shape[-1] != 11:
        raise ValueError(
            f'Expected an HxWx11 training tuple, got {image.shape}')
    return (
        image[..., 0:3],
        image[..., 3:6],
        image[..., 6:7],
        image[..., 7:10],
        image[..., 10:11],
    )


@PIPELINES.register_module()
class PhotoMetricDistortionVCODFrames(PhotoMetricDistortion):
    """Apply photometric augmentation independently to the three RGB frames."""

    def __call__(self, results):
        current, previous, previous_mask, reference, reference_mask = (
            _split_training_tuple(results['img']))
        current = super().__call__(dict(img=current))['img']
        previous = super().__call__(dict(img=previous))['img']
        reference = super().__call__(dict(img=reference))['img']
        results['img'] = np.concatenate(
            [current, previous, previous_mask, reference, reference_mask],
            axis=-1)
        return results


@PIPELINES.register_module()
class NormalizeVCODFrames(Normalize):
    """Normalize RGB streams while leaving binary masks unchanged."""

    def _normalize(self, image):
        result = super().__call__(dict(img=image))
        return result['img'], result['img_norm_cfg']

    def __call__(self, results):
        image = results['img']
        if results['test_mode']:
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f'Inference expects one RGB frame, got {image.shape}')
            results['img'], results['img_norm_cfg'] = self._normalize(image)
            return results

        current, previous, previous_mask, reference, reference_mask = (
            _split_training_tuple(image))
        current, norm_config = self._normalize(current)
        previous, _ = self._normalize(previous)
        reference, _ = self._normalize(reference)
        results['img'] = np.concatenate(
            [current, previous, previous_mask, reference, reference_mask],
            axis=-1)
        results['img_norm_cfg'] = norm_config
        return results


__all__ = ['NormalizeVCODFrames', 'PhotoMetricDistortionVCODFrames']
