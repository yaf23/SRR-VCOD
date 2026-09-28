"""Load SRRNet frame tuples directly from downloaded datasets."""

import os.path as osp
import random

import cv2
import mmcv
import numpy as np

from ..builder import PIPELINES
from .mask_utils import normalize_binary_mask


@PIPELINES.register_module()
class LoadVCODFrames:
    """Load one RGB frame for inference or an 11-channel training tuple."""

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    @staticmethod
    def _resolve(prefix, filename):
        if filename is None:
            return None
        if prefix is None or osp.isabs(filename):
            return filename
        return osp.join(prefix, filename)

    def _read(self, filename, flag):
        content = self.file_client.get(filename)
        image = mmcv.imfrombytes(
            content, flag=flag, backend=self.imdecode_backend)
        if image is None:
            raise FileNotFoundError(f'Unable to decode image: {filename}')
        return image

    def _read_mask(self, prefix, filename, shape):
        if filename is None:
            return np.zeros(shape + (1,), dtype=np.uint8)
        mask = self._read(self._resolve(prefix, filename), 'grayscale')
        mask = normalize_binary_mask(mask, foreground_value=255)
        if mask.shape != shape:
            mask = cv2.resize(
                mask, dsize=shape[::-1], interpolation=cv2.INTER_NEAREST)
        return mask[..., None]

    @staticmethod
    def _resize_image(image, shape):
        if image.shape[:2] == shape:
            return image
        return cv2.resize(
            image, dsize=shape[::-1], interpolation=cv2.INTER_LINEAR)

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        record = results['img_info']
        image_prefix = results.get('image_prefix')
        mask_prefix = results.get('mask_prefix')
        current_path = self._resolve(image_prefix, record['filename'])
        current = self._read(current_path, self.color_type)
        shape = current.shape[:2]
        annotation_path = self._resolve(
            mask_prefix, record.get('ann', {}).get('seg_map'))
        results.update(
            filename=current_path,
            ori_filename=record.get('ori_filename', record['filename']),
            sequence_id=record['sequence_id'],
            frame_index=record['frame_index'],
            is_first=record['is_first'],
            ann_filename=annotation_path)

        if results['test_mode']:
            image = current
        else:
            previous_path = self._resolve(
                image_prefix, record.get('prev_filename') or record['filename'])
            previous = self._resize_image(
                self._read(previous_path, self.color_type), shape)
            previous_mask = self._read_mask(
                mask_prefix, record.get('prev_seg_map'), shape)

            reference_names = record.get('ref_filenames') or [record['filename']]
            reference_masks = record.get('ref_seg_maps') or [None]
            reference_index = random.randrange(len(reference_names))
            while (len(reference_names) > 1
                   and reference_names[reference_index] == record['filename']):
                reference_index = random.randrange(len(reference_names))
            reference_path = self._resolve(
                image_prefix, reference_names[reference_index])
            reference = self._resize_image(
                self._read(reference_path, self.color_type), shape)
            reference_mask_name = (
                reference_masks[reference_index]
                if reference_index < len(reference_masks) else None)
            reference_mask = self._read_mask(
                mask_prefix, reference_mask_name, shape)
            image = np.concatenate(
                [current, previous, previous_mask, reference, reference_mask],
                axis=-1)

        if self.to_float32:
            image = image.astype(np.float32)
        results.update(
            img=image,
            img_shape=current.shape,
            ori_shape=current.shape,
            pad_shape=current.shape,
            scale_factor=1.0,
            img_norm_cfg=dict(
                mean=np.zeros(3, dtype=np.float32),
                std=np.ones(3, dtype=np.float32),
                to_rgb=False))
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'to_float32={self.to_float32}, '
                f"color_type='{self.color_type}', "
                f"imdecode_backend='{self.imdecode_backend}')")
