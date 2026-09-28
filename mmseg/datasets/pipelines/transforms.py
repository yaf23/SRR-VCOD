"""Image transforms used by SRRNet training and inference."""

import random

import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class ResizeToMultiple:
    """Resize image and masks so both spatial dimensions divide evenly."""

    def __init__(self, size_divisor=32, interpolation='bilinear'):
        self.size_divisor = size_divisor
        self.interpolation = interpolation

    def __call__(self, results):
        image = mmcv.imresize_to_multiple(
            results['img'],
            divisor=self.size_divisor,
            scale_factor=1,
            interpolation=self.interpolation)
        results['img'] = image
        results['img_shape'] = image.shape
        results['pad_shape'] = image.shape
        for key in results.get('seg_fields', []):
            results[key] = mmcv.imresize(
                results[key], image.shape[:2][::-1], interpolation='nearest')
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'size_divisor={self.size_divisor}, '
                f"interpolation='{self.interpolation}')")


@PIPELINES.register_module()
class Resize:
    """Resize image and segmentation fields to a configured scale."""

    def __init__(self, img_scale=None, keep_ratio=True):
        if img_scale is not None:
            if not (isinstance(img_scale, tuple) and len(img_scale) == 2):
                raise TypeError('img_scale must be a (width, height) tuple')
        self.img_scale = img_scale
        self.keep_ratio = keep_ratio

    def __call__(self, results):
        scale = results.get('scale', self.img_scale)
        if scale is None:
            raise ValueError('Resize requires img_scale or results["scale"]')
        original_height, original_width = results['img'].shape[:2]
        if self.keep_ratio:
            image, scale_factor = mmcv.imrescale(
                results['img'], scale, return_scale=True)
            width_scale = height_scale = scale_factor
        else:
            image, width_scale, height_scale = mmcv.imresize(
                results['img'], scale, return_scale=True)

        results['img'] = image
        results['img_shape'] = image.shape
        results['pad_shape'] = image.shape
        results['scale'] = scale
        results['scale_factor'] = np.array(
            [width_scale, height_scale, width_scale, height_scale],
            dtype=np.float32)
        results['keep_ratio'] = self.keep_ratio
        target_size = (image.shape[1], image.shape[0])
        for key in results.get('seg_fields', []):
            results[key] = mmcv.imresize(
                results[key], target_size, interpolation='nearest')
        if original_height <= 0 or original_width <= 0:
            raise ValueError('Input image has invalid spatial dimensions')
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'img_scale={self.img_scale}, keep_ratio={self.keep_ratio})')


@PIPELINES.register_module()
class RandomFlip:
    """Flip image and masks with a fixed probability."""

    def __init__(self, prob=0.5, direction='horizontal'):
        if not 0 <= prob <= 1:
            raise ValueError('Flip probability must be in [0, 1]')
        if direction not in {'horizontal', 'vertical', 'diagonal'}:
            raise ValueError(f'Unsupported flip direction: {direction}')
        self.prob = prob
        self.direction = direction

    def __call__(self, results):
        if 'flip' not in results:
            results['flip'] = random.random() < self.prob
        results.setdefault('flip_direction', self.direction)
        if results['flip']:
            direction = results['flip_direction']
            results['img'] = mmcv.imflip(results['img'], direction=direction)
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imflip(
                    results[key], direction=direction).copy()
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'prob={self.prob}, direction={self.direction!r})')


@PIPELINES.register_module()
class Pad:
    """Pad image and masks to a fixed size or size divisor."""

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=255):
        if (size is None) == (size_divisor is None):
            raise ValueError('Specify exactly one of size and size_divisor')
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def __call__(self, results):
        if self.size is not None:
            image = mmcv.impad(
                results['img'], shape=self.size, pad_val=self.pad_val)
        else:
            image = mmcv.impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = image
        results['pad_shape'] = image.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key], shape=image.shape[:2], pad_val=self.seg_pad_val)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(size={self.size}, '
                f'size_divisor={self.size_divisor}, pad_val={self.pad_val}, '
                f'seg_pad_val={self.seg_pad_val})')


@PIPELINES.register_module()
class Normalize:
    """Normalize one RGB image with channel-wise mean and standard deviation."""

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        results['img'] = mmcv.imnormalize(
            results['img'], self.mean, self.std, self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(mean={self.mean.tolist()}, '
                f'std={self.std.tolist()}, to_rgb={self.to_rgb})')


@PIPELINES.register_module()
class RandomCrop:
    """Randomly crop image and masks, avoiding single-class dominance."""

    def __init__(self, crop_size, cat_max_ratio=1.0, ignore_index=255):
        self.crop_size = tuple(crop_size)
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def _crop_bbox(self, image):
        margin_height = max(image.shape[0] - self.crop_size[0], 0)
        margin_width = max(image.shape[1] - self.crop_size[1], 0)
        offset_height = random.randint(0, margin_height)
        offset_width = random.randint(0, margin_width)
        return (offset_height, offset_height + self.crop_size[0],
                offset_width, offset_width + self.crop_size[1])

    @staticmethod
    def _crop(value, box):
        y1, y2, x1, x2 = box
        return value[y1:y2, x1:x2, ...]

    def __call__(self, results):
        crop_box = self._crop_bbox(results['img'])
        segmentation_fields = results.get('seg_fields', [])
        if self.cat_max_ratio < 1.0 and segmentation_fields:
            target = results[segmentation_fields[0]]
            for _ in range(10):
                cropped = self._crop(target, crop_box)
                labels, counts = np.unique(cropped, return_counts=True)
                counts = counts[labels != self.ignore_index]
                if len(counts) > 1 and counts.max() / counts.sum() < self.cat_max_ratio:
                    break
                crop_box = self._crop_bbox(results['img'])

        results['img'] = self._crop(results['img'], crop_box)
        results['img_shape'] = results['img'].shape
        for key in segmentation_fields:
            results[key] = self._crop(results[key], crop_box)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(crop_size={self.crop_size}, '
                f'cat_max_ratio={self.cat_max_ratio})')


@PIPELINES.register_module()
class PhotoMetricDistortion:
    """Apply standard brightness, contrast, saturation and hue jitter."""

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    @staticmethod
    def _coin_flip():
        return random.randint(0, 1)

    def _contrast(self, image):
        if self._coin_flip():
            image *= random.uniform(self.contrast_lower, self.contrast_upper)
        return image

    def __call__(self, results):
        image = results['img'].astype(np.float32)
        if self._coin_flip():
            image += random.uniform(
                -self.brightness_delta, self.brightness_delta)

        contrast_first = self._coin_flip()
        if contrast_first:
            image = self._contrast(image)
        image = mmcv.bgr2hsv(image)
        if self._coin_flip():
            image[..., 1] *= random.uniform(
                self.saturation_lower, self.saturation_upper)
        if self._coin_flip():
            image[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            image[..., 0][image[..., 0] > 360] -= 360
            image[..., 0][image[..., 0] < 0] += 360
        image = mmcv.hsv2bgr(image)
        if not contrast_first:
            image = self._contrast(image)
        if self._coin_flip():
            image = image[..., np.random.permutation(3)]
        results['img'] = image
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'brightness_delta={self.brightness_delta}, '
                f'contrast_range=({self.contrast_lower}, {self.contrast_upper}), '
                f'saturation_range=({self.saturation_lower}, '
                f'{self.saturation_upper}), hue_delta={self.hue_delta})')


__all__ = [
    'Normalize', 'Pad', 'PhotoMetricDistortion', 'RandomCrop', 'RandomFlip',
    'Resize', 'ResizeToMultiple'
]
