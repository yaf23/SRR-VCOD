"""Evaluate SRRNet predictions on MoCA-Mask or CAD."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

from mmseg.core import average_sequence_metrics, evaluate_sequence
from mmseg.datasets.vcod import (CADDataset, MoCAMaskDataset,
                                 natural_sort_key, read_binary_mask)


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg'}


def _dataset_records(dataset_name, data_root, cad_paper_protocol=True):
    common = dict(pipeline=[], data_root=str(data_root), test_mode=True)
    if dataset_name == 'cad':
        dataset = CADDataset(
            paper_protocol=cad_paper_protocol,
            skip_empty_gt=True,
            **common)
    elif dataset_name == 'moca':
        dataset = MoCAMaskDataset(split='test', **common)
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')

    return [
        dict(
            sequence_id=record['sequence_id'],
            frame_name=Path(record['ori_filename']).stem,
            ground_truth=Path(dataset.mask_root) / record['ann']['seg_map'])
        for record in dataset.img_infos
    ]


def _prediction_index(prediction_root):
    prediction_root = Path(prediction_root).resolve()
    if not prediction_root.is_dir():
        raise FileNotFoundError(
            f'Prediction directory does not exist: {prediction_root}')

    index = {}
    image_paths = sorted(
        (path for path in prediction_root.rglob('*')
         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_sort_key)
    for path in image_paths:
        relative = path.relative_to(prediction_root)
        if len(relative.parts) < 2:
            raise ValueError(
                'Predictions must use <sequence>/<frame>.png layout; '
                f'found {relative}')
        sequence_id = relative.parts[-2]
        key = (sequence_id.lower(), path.stem.lower())
        if key in index:
            raise ValueError(
                f'Duplicate prediction for {sequence_id}/{path.stem}: '
                f'{index[key]} and {path}')
        index[key] = path
    return index


def discover_pairs(dataset_name,
                   data_root,
                   prediction_root,
                   cad_paper_protocol=True):
    """Match each benchmark frame to exactly one prediction."""
    records = _dataset_records(
        dataset_name, data_root, cad_paper_protocol=cad_paper_protocol)
    predictions = _prediction_index(prediction_root)
    expected_keys = {
        (record['sequence_id'].lower(), record['frame_name'].lower())
        for record in records
    }

    missing = sorted(expected_keys - predictions.keys())
    extra = sorted(predictions.keys() - expected_keys)
    if missing or extra:
        messages = []
        if missing:
            preview = ', '.join(f'{sequence}/{frame}'
                                for sequence, frame in missing[:10])
            messages.append(
                f'{len(missing)} missing predictions (first: {preview})')
        if extra:
            preview = ', '.join(f'{sequence}/{frame}'
                                for sequence, frame in extra[:10])
            messages.append(
                f'{len(extra)} unexpected predictions (first: {preview})')
        raise ValueError('; '.join(messages))

    return [
        dict(
            sequence_id=record['sequence_id'],
            frame_name=record['frame_name'],
            ground_truth=record['ground_truth'],
            prediction=predictions[(
                record['sequence_id'].lower(),
                record['frame_name'].lower())])
        for record in records
    ]


def _read_prediction(path):
    prediction = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if prediction is None:
        raise ValueError(f'Cannot decode prediction: {path}')
    return prediction


def evaluate_predictions(dataset_name, data_root, prediction_root):
    """Evaluate predictions with equal weighting across video sequences."""
    pairs = discover_pairs(dataset_name, data_root, prediction_root)
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair['sequence_id']].append(pair)

    per_sequence = {}
    for sequence_id, sequence_pairs in grouped.items():
        predictions = [_read_prediction(pair['prediction'])
                       for pair in sequence_pairs]
        ground_truths = [read_binary_mask(pair['ground_truth'])
                         for pair in sequence_pairs]
        per_sequence[sequence_id] = dict(
            evaluate_sequence(predictions, ground_truths))

    metrics = dict(average_sequence_metrics(per_sequence.values()))
    return dict(
        dataset=dataset_name.upper(),
        sequences=len(per_sequence),
        frames=len(pairs),
        metrics=metrics,
        per_sequence=per_sequence)


def _format_report(report):
    metric_labels = {
        'Salpha': 'S-alpha',
        'Fwbeta': 'Fw-beta',
        'MAE': 'MAE',
        'mDice': 'mDice',
        'mIoU': 'mIoU',
    }
    lines = [
        f"Dataset: {report['dataset']}",
        f"Sequences: {report['sequences']}",
        f"Frames: {report['frames']}",
    ]
    lines.extend(
        f"{metric_labels[name]}: {value:.9f}"
        for name, value in report['metrics'].items())
    return '\n'.join(lines) + '\n'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate SRRNet predictions with the paper metrics')
    parser.add_argument(
        '--dataset', default='moca', choices=['cad', 'moca'],
        help='Evaluation benchmark (default: moca)')
    parser.add_argument(
        '--data-root', required=True,
        help='Original downloaded dataset directory')
    parser.add_argument(
        '--pred-dir', required=True,
        help='Prediction directory using <sequence>/<frame>.png')
    parser.add_argument(
        '--output-dir',
        help='Directory for metrics.json and metrics.txt '
             '(default: <pred-dir>/evaluation)')
    return parser.parse_args()


def main():
    args = parse_args()
    report = evaluate_predictions(args.dataset, args.data_root, args.pred_dir)
    output_dir = Path(args.output_dir or Path(args.pred_dir) / 'evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)
    text_report = _format_report(report)
    (output_dir / 'metrics.txt').write_text(text_report, encoding='utf-8')
    (output_dir / 'metrics.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    print(text_report, end='')


if __name__ == '__main__':
    main()
