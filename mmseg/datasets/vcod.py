"""Direct adapters for COD10K, MoCA-Mask and CAD.

The adapters index the original downloaded directories and build temporal
relationships in memory.  No optical-flow or duplicated previous-frame
directory is required.
"""

import os
import os.path as osp
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from mmcv.utils import print_log
from torch.utils.data import Dataset

from mmseg.core import average_sequence_metrics, evaluate_sequence
from mmseg.utils import get_root_logger

from .builder import DATASETS
from .cad_protocol import (load_cad_paper_protocol,
                           map_processed_cad_frames)
from .pipelines import Compose
from .pipelines.mask_utils import normalize_binary_mask


IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')
MASK_SUFFIXES = ('.png', '.jpg', '.jpeg')


def natural_sort_key(value):
    """Sort path components containing numbers in human order."""
    text = os.fspath(value).replace('\\', '/').lower()
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in re.split(r'(\d+)', text)
        if part)


def _list_files(directory, suffixes):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    suffixes = {suffix.lower() for suffix in suffixes}
    return sorted(
        (path for path in directory.iterdir()
         if path.is_file() and path.suffix.lower() in suffixes),
        key=natural_sort_key)


def pair_files(image_dir,
               mask_dir,
               image_suffixes=IMAGE_SUFFIXES,
               mask_suffixes=MASK_SUFFIXES,
               key=None):
    """Pair image and mask files by a unique, shared key."""
    key = key or (lambda path: path.stem.lower())

    def index_unique(paths, kind):
        index = {}
        for path in paths:
            file_key = key(path)
            if file_key in index:
                raise ValueError(
                    f'Duplicate {kind} key {file_key!r}: '
                    f'{index[file_key]} and {path}')
            index[file_key] = path
        return index

    images = index_unique(_list_files(image_dir, image_suffixes), 'image')
    masks = index_unique(_list_files(mask_dir, mask_suffixes), 'mask')
    image_only = sorted(
        images.keys() - masks.keys(),
        key=lambda value: natural_sort_key(str(value)))
    mask_only = sorted(
        masks.keys() - images.keys(),
        key=lambda value: natural_sort_key(str(value)))
    if image_only or mask_only:
        raise ValueError(
            f'Image/mask key mismatch: image-only={image_only[:10]}, '
            f'mask-only={mask_only[:10]}')
    shared_keys = sorted(
        images,
        key=lambda value: natural_sort_key(str(value)))
    return [(images[file_key], masks[file_key]) for file_key in shared_keys]


def read_binary_mask(path):
    """Decode common 0/255 and CAD 1/2 masks as class IDs 0/1."""
    with Image.open(path) as image:
        mask = np.asarray(image)
    return normalize_binary_mask(mask, foreground_value=1)


def _relative(path, root):
    return osp.relpath(os.fspath(path), os.fspath(root))


class BaseVCODDataset(Dataset):
    """Base dataset for directly indexed VCOD samples."""

    CLASSES = ('Background', 'Foreground')
    PALETTE = [[0, 0, 0], [255, 255, 255]]
    requires_sequence_inference = True

    def _initialize(self,
                    records,
                    pipeline,
                    data_root,
                    image_root,
                    mask_root,
                    split,
                    test_mode=False,
                    ignore_index=255,
                    reduce_zero_label=False,
                    **kwargs):
        if kwargs:
            unknown = ', '.join(sorted(kwargs))
            raise TypeError(f'Unsupported dataset options: {unknown}')
        if not records:
            raise RuntimeError(f'No paired image/mask samples under {data_root}')

        self.img_infos = records
        self._direct_img_infos = records
        self.pipeline = Compose(pipeline)
        self.data_root = osp.abspath(os.fspath(data_root))
        self.image_root = osp.abspath(os.fspath(image_root))
        self.mask_root = osp.abspath(os.fspath(mask_root))
        self.split = split
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.custom_classes = False
        print_log(
            f'Loaded {len(records)} {self.__class__.__name__} samples',
            logger=get_root_logger())

    def __len__(self):
        return len(self.img_infos)

    def get_ann_info(self, index):
        return self.img_infos[index]['ann']

    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['image_prefix'] = self.image_root
        results['mask_prefix'] = self.mask_root

    def _prepare(self, index, include_annotation):
        record = self.img_infos[index]
        results = dict(img_info=record, test_mode=self.test_mode)
        if include_annotation:
            results['ann_info'] = record['ann']
        self.pre_pipeline(results)
        return self.pipeline(results)

    def __getitem__(self, index):
        return self._prepare(index, include_annotation=not self.test_mode)

    def get_gt_seg_maps(self, efficient_test=False):
        del efficient_test
        return [
            read_binary_mask(osp.join(self.mask_root, record['ann']['seg_map']))
            for record in self.img_infos
        ]

    @staticmethod
    def _unwrap_prediction(prediction):
        if isinstance(prediction, (list, tuple)):
            if len(prediction) != 1:
                raise ValueError('Expected one prediction per dataset sample')
            prediction = prediction[0]
        return np.asarray(prediction)

    def evaluate(self, results, metric='VCOD', logger=None, **kwargs):
        del kwargs
        requested = {metric} if isinstance(metric, str) else set(metric)
        if requested not in ({'VCOD'}, {'Fwbeta'}):
            raise KeyError(
                'Direct VCOD datasets support metric="VCOD" or "Fwbeta"')
        if len(results) != len(self):
            raise ValueError(
                f'Expected {len(self)} predictions, received {len(results)}')

        grouped_predictions = defaultdict(list)
        grouped_ground_truths = defaultdict(list)
        ground_truths = self.get_gt_seg_maps()
        for record, prediction, ground_truth in zip(
                self.img_infos, results, ground_truths):
            sequence_id = record['sequence_id']
            grouped_predictions[sequence_id].append(
                self._unwrap_prediction(prediction))
            grouped_ground_truths[sequence_id].append(ground_truth)

        sequence_metrics = [
            evaluate_sequence(
                grouped_predictions[sequence_id],
                grouped_ground_truths[sequence_id])
            for sequence_id in grouped_predictions
        ]
        metrics = average_sequence_metrics(sequence_metrics)
        if requested == {'Fwbeta'}:
            metrics = {'Fwbeta': metrics['Fwbeta']}
        print_log(
            'VCOD metrics: ' + ', '.join(
                f'{name}={value:.6f}' for name, value in metrics.items()),
            logger=logger)
        return dict(metrics)


@DATASETS.register_module()
class COD10KDataset(BaseVCODDataset):
    """COD10K-v3 adapter using the official Train/Test layout."""

    def __init__(self,
                 pipeline,
                 data_root,
                 split='train',
                 cam_only=True,
                 sample_stride=1,
                 test_mode=False,
                 **kwargs):
        split = str(split).lower()
        if split not in {'train', 'test'}:
            raise ValueError("COD10K split must be 'train' or 'test'")
        if sample_stride < 1:
            raise ValueError('COD10K sample_stride must be at least 1')

        data_root = osp.abspath(os.fspath(data_root))
        split_root = Path(data_root) / split.capitalize()
        image_root = split_root / 'Image'
        mask_root = split_root / 'GT_Object'
        pairs = pair_files(image_root, mask_root)
        if cam_only:
            pairs = [pair for pair in pairs
                     if '-cam-' in pair[0].stem.lower()]

        categories = defaultdict(list)
        for image_path, mask_path in pairs:
            tokens = image_path.stem.split('-')
            category = tokens[-2] if len(tokens) >= 2 else image_path.stem
            categories[category].append((image_path, mask_path))

        records = []
        for category in sorted(categories, key=natural_sort_key):
            category_pairs = sorted(
                categories[category], key=lambda pair: pair[0].name)
            category_pairs = category_pairs[::sample_stride]
            for frame_index, (image_path, mask_path) in enumerate(category_pairs):
                if frame_index:
                    previous_path, previous_mask = category_pairs[frame_index - 1]
                elif len(category_pairs) > 1:
                    previous_path, previous_mask = category_pairs[1]
                else:
                    previous_path, previous_mask = image_path, mask_path
                references = category_pairs[1:] or [(image_path, mask_path)]
                records.append(dict(
                    filename=_relative(image_path, image_root),
                    ann=dict(seg_map=_relative(mask_path, mask_root)),
                    prev_filename=_relative(previous_path, image_root),
                    prev_seg_map=_relative(previous_mask, mask_root),
                    ref_filenames=[
                        _relative(path, image_root) for path, _ in references
                    ],
                    ref_seg_maps=[
                        _relative(path, mask_root) for _, path in references
                    ],
                    sequence_id=category,
                    frame_index=frame_index,
                    is_first=frame_index == 0,
                    ori_filename=image_path.name))

        self._initialize(
            records=records,
            pipeline=pipeline,
            data_root=data_root,
            image_root=image_root,
            mask_root=mask_root,
            split=split,
            test_mode=test_mode,
            **kwargs)


def _resolve_moca_split(data_root, split):
    aliases = {
        'train': ('Train', 'TrainDataset_per_sq'),
        'test': ('Test', 'TestDataset_per_sq'),
    }
    split = str(split).lower()
    if split not in aliases:
        raise ValueError("MoCA-Mask split must be 'train' or 'test'")

    data_root = Path(data_root).resolve()
    accepted = {name.lower() for name in aliases[split]}
    if data_root.name.lower() in accepted:
        return data_root, split
    children = ({child.name.lower(): child for child in data_root.iterdir()
                 if child.is_dir()} if data_root.is_dir() else {})
    for alias in aliases[split]:
        if alias.lower() in children:
            return children[alias.lower()], split
    raise FileNotFoundError(
        f'Cannot find MoCA-Mask {split} split under {data_root}; '
        f'expected one of {aliases[split]}')


def _video_records(split_root, sequence_dir, pairs):
    records = []
    for frame_index, (image_path, mask_path) in enumerate(pairs):
        if frame_index == 0:
            previous_path = image_path
            previous_mask = None
            references = [(image_path, None)]
        else:
            previous_path, previous_mask = pairs[frame_index - 1]
            references = list(pairs[:frame_index])
        sequence_id = sequence_dir.name
        records.append(dict(
            filename=_relative(image_path, split_root),
            ann=dict(seg_map=_relative(mask_path, split_root)),
            prev_filename=_relative(previous_path, split_root),
            prev_seg_map=(None if previous_mask is None
                          else _relative(previous_mask, split_root)),
            ref_filenames=[
                _relative(path, split_root) for path, _ in references
            ],
            ref_seg_maps=[
                None if path is None else _relative(path, split_root)
                for _, path in references
            ],
            sequence_id=sequence_id,
            frame_index=frame_index,
            is_first=frame_index == 0,
            ori_filename=f'{sequence_id}/{image_path.name}'))
    return records


def _apply_order_manifest(records, manifest_path):
    if manifest_path is None:
        return records

    order = [
        line.strip()
        for line in Path(manifest_path).read_text(
            encoding='utf-8').splitlines()
        if line.strip()
    ]
    records_by_name = {
        record['ori_filename']: record for record in records
    }
    if len(records_by_name) != len(records):
        raise ValueError('MoCA records contain duplicate ori_filename values')

    provided = set(order)
    expected = set(records_by_name)
    if len(order) != len(provided) or provided != expected:
        raise ValueError(
            'MoCA order manifest must contain every sample exactly once; '
            f'missing={len(expected - provided)}, '
            f'extra={len(provided - expected)}, '
            f'duplicates={len(order) - len(provided)}')
    return [records_by_name[name] for name in order]


@DATASETS.register_module()
class MoCAMaskDataset(BaseVCODDataset):
    """MoCA-Mask adapter for both commonly distributed split names."""

    def __init__(self,
                 pipeline,
                 data_root,
                 split='train',
                 skip_empty_gt=None,
                 test_mode=False,
                 order_manifest=None,
                 **kwargs):
        del skip_empty_gt
        data_root = osp.abspath(os.fspath(data_root))
        split_root, split = _resolve_moca_split(data_root, split)
        records = []
        for sequence_dir in sorted(
                (path for path in split_root.iterdir() if path.is_dir()),
                key=natural_sort_key):
            pairs = pair_files(sequence_dir / 'Imgs', sequence_dir / 'GT')
            pairs = sorted(pairs, key=lambda pair: natural_sort_key(pair[0]))
            records.extend(_video_records(split_root, sequence_dir, pairs))
        records = _apply_order_manifest(records, order_manifest)

        self._initialize(
            records=records,
            pipeline=pipeline,
            data_root=data_root,
            image_root=split_root,
            mask_root=split_root,
            split=split,
            test_mode=test_mode,
            **kwargs)


def _cad_pair_key(path):
    numbers = re.findall(r'\d+', path.stem)
    return int(numbers[-1]) if numbers else path.stem.lower().replace('_gt', '')


def _resolve_cad_layout(data_root):
    data_root = Path(data_root).resolve()
    for candidate in (data_root / 'original_data', data_root):
        if not candidate.is_dir():
            continue
        if any(path.is_dir()
               and (path / 'frames').is_dir()
               and (path / 'groundtruth').is_dir()
               for path in candidate.iterdir()):
            return candidate, 'canonical'

    image_root = data_root / 'Imgs'
    mask_root = data_root / 'GT'
    if image_root.is_dir() and mask_root.is_dir():
        image_sequences = {
            path.name.lower() for path in image_root.iterdir()
            if path.is_dir()
        }
        mask_sequences = {
            path.name.lower() for path in mask_root.iterdir()
            if path.is_dir()
        }
        if image_sequences & mask_sequences:
            return data_root, 'processed'

    raise FileNotFoundError(
        f'Cannot find CAD <scene>/frames plus <scene>/groundtruth, or '
        f'Imgs/<scene> plus GT/<scene>, below {data_root}')


def _cad_sequence_locations(cad_root, layout):
    if layout == 'canonical':
        locations = []
        for sequence_dir in cad_root.iterdir():
            if (sequence_dir.is_dir()
                    and (sequence_dir / 'frames').is_dir()
                    and (sequence_dir / 'groundtruth').is_dir()):
                locations.append((
                    sequence_dir,
                    sequence_dir / 'frames',
                    sequence_dir / 'groundtruth'))
        return locations

    def index_sequences(root):
        sequences = {}
        for path in root.iterdir():
            if not path.is_dir():
                continue
            key = path.name.lower()
            if key in sequences:
                raise ValueError(
                    f'Duplicate CAD sequence name ignoring case: '
                    f'{sequences[key]} and {path}')
            sequences[key] = path
        return sequences

    image_sequences = index_sequences(cad_root / 'Imgs')
    mask_sequences = index_sequences(cad_root / 'GT')
    return [
        (image_sequences[key], image_sequences[key], mask_sequences[key])
        for key in image_sequences.keys() & mask_sequences.keys()
    ]


def _cad_pairs(image_dir, mask_dir, layout):
    image_suffixes = ('.jpg', ) if layout == 'processed' else IMAGE_SUFFIXES
    return pair_files(
        image_dir,
        mask_dir,
        image_suffixes=image_suffixes,
        key=_cad_pair_key)


def _cad_paper_sequence_pairs(cad_root, layout):
    sequence_locations = {}
    for sequence_dir, image_dir, mask_dir in _cad_sequence_locations(
            cad_root, layout):
        key = sequence_dir.name.lower()
        if key in sequence_locations:
            raise ValueError(
                f'Duplicate CAD sequence name ignoring case: '
                f'{sequence_locations[key][0]} and {sequence_dir}')
        sequence_locations[key] = (sequence_dir, image_dir, mask_dir)

    protocol_entries = defaultdict(list)
    sequence_order = []
    for entry in load_cad_paper_protocol():
        sequence_name, _ = entry.parts
        sequence_key = sequence_name.lower()
        if sequence_key not in protocol_entries:
            sequence_order.append(sequence_key)
        protocol_entries[sequence_key].append(entry)

    selected = []
    missing = []
    for sequence_key in sequence_order:
        entries = protocol_entries[sequence_key]
        location = sequence_locations.get(sequence_key)
        if location is None:
            missing.extend(entry.as_posix() for entry in entries)
            continue
        sequence_dir, image_dir, mask_dir = location
        if layout == 'processed':
            pairs = pair_files(
                image_dir, mask_dir, image_suffixes=('.jpg', ))
            pair_index = {pair[0]: pair for pair in pairs}
            mapped = map_processed_cad_frames(image_dir, entries)
            selected.append((
                sequence_dir,
                [pair_index[image_path] for _, image_path in mapped],
                entries))
            continue

        pairs = _cad_pairs(image_dir, mask_dir, layout)
        pair_index = {
            image_path.stem.lower(): (image_path, mask_path)
            for image_path, mask_path in pairs
        }
        sequence_pairs = []
        for entry in entries:
            pair = pair_index.get(Path(entry.name).stem.lower())
            if pair is None:
                missing.append(entry.as_posix())
            else:
                sequence_pairs.append(pair)
        selected.append((sequence_dir, sequence_pairs, entries))

    if missing:
        preview = ', '.join(missing[:10])
        raise FileNotFoundError(
            f'{len(missing)} CAD protocol image/mask pairs are missing '
            f'below {cad_root} (first: {preview})')
    return selected


@DATASETS.register_module()
class CADDataset(BaseVCODDataset):
    """CAD adapter for canonical downloads and legacy processed layouts."""

    def __init__(self,
                 pipeline,
                 data_root,
                 skip_empty_gt=True,
                 split=None,
                 test_mode=False,
                 paper_protocol=True,
                 **kwargs):
        if split is not None and str(split).lower() != 'test':
            raise ValueError("CAD split, when provided, must be 'test'")
        data_root = osp.abspath(os.fspath(data_root))
        cad_root, layout = _resolve_cad_layout(data_root)
        if paper_protocol:
            sequence_pairs = _cad_paper_sequence_pairs(cad_root, layout)
        else:
            sequence_pairs = [
                (sequence_dir, _cad_pairs(image_dir, mask_dir, layout), None)
                for sequence_dir, image_dir, mask_dir in sorted(
                    _cad_sequence_locations(cad_root, layout),
                    key=lambda location: natural_sort_key(location[0]))
            ]

        records = []
        for sequence_dir, pairs, entries in sequence_pairs:
            if not paper_protocol:
                pairs = sorted(
                    pairs, key=lambda pair: natural_sort_key(pair[0]))
            if skip_empty_gt:
                empty = [pair for pair in pairs
                         if not read_binary_mask(pair[1]).any()]
                if paper_protocol and empty:
                    preview = ', '.join(path.name for _, path in empty[:10])
                    raise ValueError(
                        f'CAD paper protocol includes {len(empty)} empty '
                        f'ground-truth masks (first: {preview})')
                pairs = [pair for pair in pairs if pair not in empty]
            sequence_records = _video_records(cad_root, sequence_dir, pairs)
            if entries is not None:
                for record, entry in zip(sequence_records, entries):
                    record['ori_filename'] = entry.as_posix()
            records.extend(sequence_records)

        self._initialize(
            records=records,
            pipeline=pipeline,
            data_root=data_root,
            image_root=cad_root,
            mask_root=cad_root,
            split='test',
            test_mode=test_mode,
            **kwargs)


__all__ = [
    'BaseVCODDataset', 'CADDataset', 'COD10KDataset', 'MoCAMaskDataset',
    'natural_sort_key', 'pair_files', 'read_binary_mask'
]
