"""Public model loading and downloaded-dataset sequence inference."""

import copy
import json
import os
from pathlib import Path

import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from torch.utils.data import Dataset

from mmseg.datasets import build_dataloader
from mmseg.datasets.cad_protocol import (load_cad_paper_protocol,
                                         map_processed_cad_frames)
from mmseg.datasets.pipelines import Compose
from mmseg.datasets.vcod import natural_sort_key
from mmseg.models import build_segmentor

from .sequence_inference import single_gpu_sequence_inference


_IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}


def _image_files(directory, suffixes=_IMAGE_SUFFIXES):
    suffixes = {suffix.lower() for suffix in suffixes}
    return sorted(
        (path for path in Path(directory).iterdir()
         if path.is_file() and path.suffix.lower() in suffixes),
        key=natural_sort_key)


def _contains_sequence(root, image_directory_name):
    return root.is_dir() and any(
        child.is_dir() and (child / image_directory_name).is_dir()
        for child in root.iterdir())


def _resolve_sequence_root(data_root, dataset_name):
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset root does not exist: {root}')

    if dataset_name == 'moca':
        candidates = (root, root / 'Test', root / 'TestDataset_per_sq')
        image_directory_name = 'Imgs'
    elif dataset_name == 'cad':
        candidates = (root / 'original_data', root)
        for candidate in candidates:
            if _contains_sequence(candidate, 'frames'):
                return candidate, 'frames'
        processed_root = root / 'Imgs'
        if processed_root.is_dir() and any(
                child.is_dir() for child in processed_root.iterdir()):
            return processed_root, None
        expected = ('<sequence>/frames, optionally under original_data, or '
                    'Imgs/<sequence>')
        raise FileNotFoundError(f'Cannot find {expected} below {root}')
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')

    for candidate in candidates:
        if _contains_sequence(candidate, image_directory_name):
            return candidate, image_directory_name

    if dataset_name == 'moca':
        expected = '<sequence>/Imgs under TestDataset_per_sq or Test'
    raise FileNotFoundError(f'Cannot find {expected} below {root}')


def _index_cad_frames(sequence_root, image_directory_name='frames'):
    frames = []
    sequence_directories = {}
    for path in sequence_root.iterdir():
        if not path.is_dir():
            continue
        key = path.name.lower()
        if key in sequence_directories:
            raise ValueError(
                f'Duplicate CAD sequence name ignoring case: '
                f'{sequence_directories[key]} and {path}')
        sequence_directories[key] = path

    entries_by_sequence = {}
    for entry in load_cad_paper_protocol():
        sequence_id, _ = entry.parts
        entries_by_sequence.setdefault(sequence_id, []).append(entry)

    missing = []
    for sequence_id, entries in entries_by_sequence.items():
        sequence_directory = sequence_directories.get(sequence_id.lower())
        if sequence_directory is None:
            missing.extend(entry.as_posix() for entry in entries)
            continue

        if image_directory_name is None:
            indexed_entries = map_processed_cad_frames(
                sequence_directory, entries)
        else:
            image_directory = sequence_directory / image_directory_name
            image_index = {}
            if image_directory.is_dir():
                for path in _image_files(image_directory):
                    key = path.stem.lower()
                    if key in image_index:
                        raise ValueError(
                            f'Duplicate CAD RGB frame stem in '
                            f'{image_directory}: {path.stem}')
                    image_index[key] = path
            indexed_entries = (
                (entry, image_index.get(Path(entry.name).stem.lower()))
                for entry in entries)

        for frame_index, (entry, image_path) in enumerate(indexed_entries):
            if image_path is None:
                missing.append(entry.as_posix())
                continue
            frames.append({
                'filename': os.fspath(image_path.resolve()),
                'ori_filename': entry.as_posix(),
                'sequence_id': sequence_id,
                'frame_index': frame_index,
                'is_first': frame_index == 0,
            })

    if missing:
        preview = ', '.join(missing[:10])
        raise FileNotFoundError(
            f'{len(missing)} CAD protocol frames are missing below '
            f'{sequence_root} (first: {preview})')
    return frames


def _index_frames(data_root, dataset_name):
    """Index RGB frames without requiring or inspecting annotations."""
    sequence_root, image_directory_name = _resolve_sequence_root(
        data_root, dataset_name)
    if dataset_name == 'cad':
        return _index_cad_frames(sequence_root, image_directory_name)

    frames = []
    sequence_directories = sorted(
        (path for path in sequence_root.iterdir() if path.is_dir()),
        key=natural_sort_key)
    for sequence_directory in sequence_directories:
        image_directory = sequence_directory / image_directory_name
        if not image_directory.is_dir():
            continue
        for frame_index, image_path in enumerate(_image_files(image_directory)):
            frames.append({
                'filename': os.fspath(image_path.resolve()),
                'ori_filename': f'{sequence_directory.name}/{image_path.name}',
                'sequence_id': sequence_directory.name,
                'frame_index': frame_index,
                'is_first': frame_index == 0,
            })

    if not frames:
        raise RuntimeError(f'No RGB frames found below {sequence_root}')
    return frames


def _walk_transform_configs(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_transform_configs(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_transform_configs(child)


def _test_dataset_config(cfg):
    if not hasattr(cfg, 'data') or not hasattr(cfg.data, 'test'):
        raise KeyError('The config must define data.test.pipeline')
    dataset_config = cfg.data.test
    while isinstance(dataset_config, dict) and 'dataset' in dataset_config:
        dataset_config = dataset_config['dataset']
    if not isinstance(dataset_config, dict) or 'pipeline' not in dataset_config:
        raise KeyError('The config must define data.test.pipeline')
    return dataset_config


def _inference_pipeline(cfg):
    """Build an RGB-only deterministic pipeline from a training config."""
    pipeline = copy.deepcopy(_test_dataset_config(cfg)['pipeline'])
    if not pipeline:
        raise ValueError('The test pipeline cannot be empty')
    if not isinstance(pipeline[0], dict):
        raise TypeError('The first test transform must be a config dictionary')

    pipeline[0] = dict(pipeline[0])
    pipeline[0]['type'] = 'LoadVCODFrames'

    for transform in pipeline[1:]:
        if isinstance(transform, dict) and 'transforms' in transform:
            transform['type'] = 'SingleScaleTestPipeline'
            transform.pop('flip', None)
            transform.pop('img_ratios', None)

    for transform in _walk_transform_configs(pipeline):
        transform_type = transform.get('type')
        if (isinstance(transform_type, str)
                and 'annotation' in transform_type.lower()):
            raise ValueError(
                'Inference pipelines must not load annotations')
    return pipeline


class SequenceFrameDataset(Dataset):
    """Ordered RGB-only view of a downloaded MoCA-Mask or CAD dataset."""

    CLASSES = ('Background', 'Foreground')
    PALETTE = [[0, 0, 0], [255, 255, 255]]
    requires_sequence_inference = True

    def __init__(self, frames, pipeline):
        self.img_infos = list(frames)
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, index):
        results = {
            'img_info': dict(self.img_infos[index]),
            'image_prefix': None,
            'mask_prefix': None,
            'seg_fields': [],
            'test_mode': True,
        }
        return self.pipeline(results)


def _parse_device(device_name):
    try:
        device = torch.device(device_name)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f'Invalid device {device_name!r}') from error
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only cpu and cuda devices are supported')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available')
        index = (torch.cuda.current_device()
                 if device.index is None else device.index)
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device index is out of range: {index}')
        device = torch.device('cuda', index)
    return device


def _build_model(cfg, checkpoint_path, device):
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))

    if device.type == 'cuda' and cfg.get('inference_fp16'):
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model,
        os.fspath(checkpoint_path),
        map_location='cpu',
        strict=True)
    metadata = checkpoint.get('meta') or {}
    model.CLASSES = metadata.get('CLASSES', SequenceFrameDataset.CLASSES)
    model.PALETTE = metadata.get('PALETTE', SequenceFrameDataset.PALETTE)
    model.cfg = cfg

    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model.to(device)
    if device.type == 'cuda':
        return MMDataParallel(model, device_ids=[device.index])
    return model


def run_inference(args):
    """Load a checkpoint and write ordered MoCA-Mask or CAD predictions."""
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'Config does not exist: {config_path}')
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f'Checkpoint does not exist: {checkpoint_path}')

    cfg = mmcv.Config.fromfile(os.fspath(config_path))
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    dataset_name = str(args.dataset).lower()
    frames = _index_frames(args.data_root, dataset_name)
    dataset = SequenceFrameDataset(frames, _inference_pipeline(cfg))
    device = _parse_device(args.device)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=False,
        pin_memory=device.type == 'cuda',
        dataloader_type='DataLoader')

    model = _build_model(cfg, checkpoint_path, device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_json = getattr(args, 'scores_json', None)
    score_records = [] if scores_json else None
    predictions = single_gpu_sequence_inference(
        model,
        data_loader,
        out_dir=os.fspath(output_dir),
        score_records=score_records)

    if scores_json:
        score_path = Path(scores_json).expanduser().resolve()
        score_path.parent.mkdir(parents=True, exist_ok=True)
        with score_path.open('w', encoding='utf-8') as stream:
            json.dump(score_records, stream, indent=2, ensure_ascii=False)
            stream.write('\n')

    print(
        f'Wrote {len(predictions)} binary masks from {dataset_name} to '
        f'{output_dir}')
    return predictions


__all__ = ['run_inference']
