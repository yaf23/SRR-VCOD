"""The five VCOD metrics reported by the SRRNet paper."""

from collections import OrderedDict

import numpy as np
from py_sod_metrics import (DICEHandler, FmeasureV2, IOUHandler, MAE,
                            Smeasure, WeightedFmeasure)


METRIC_NAMES = ('Salpha', 'Fwbeta', 'MAE', 'mDice', 'mIoU')


def _as_prediction(value):
    value = np.asarray(value).squeeze()
    if value.ndim != 2:
        raise ValueError(f'Prediction must be two-dimensional, got {value.shape}')
    value = value.astype(np.float32)
    if value.size and value.max() > 1.0:
        value /= 255.0
    return np.clip(value, 0.0, 1.0)


def _as_ground_truth(value):
    value = np.asarray(value).squeeze()
    if value.ndim != 2:
        raise ValueError(
            f'Ground-truth mask must be two-dimensional, got {value.shape}')
    return value > 0


class _PreparedFmeasureV2(FmeasureV2):
    """FmeasureV2 variant accepting normalized arrays without rescaling."""

    def step(self, pred, gt, normalize=False):
        if not self._metric_handlers:
            raise ValueError('At least one metric handler is required')
        if normalize:
            return super().step(pred=pred, gt=gt, normalize=True)
        if pred.dtype not in (np.float32, np.float64):
            raise ValueError('Prediction must use a floating-point dtype')
        if pred.size and not (0 <= pred.min() <= pred.max() <= 1):
            raise ValueError('Prediction values must be in [0, 1]')
        if gt.dtype != bool:
            raise ValueError('Ground truth must be boolean')

        foreground = np.count_nonzero(gt)
        background = gt.size - foreground
        dynamic_stats = None
        adaptive_stats = None
        binary_stats = None
        for handler in self._metric_handlers.values():
            if handler.dynamic_results is not None:
                if dynamic_stats is None:
                    dynamic_stats = self.dynamically_binarizing(
                        pred=pred, gt=gt, FG=foreground, BG=background)
                handler.dynamic_results.append(handler(**dynamic_stats))
            if handler.adaptive_results is not None:
                if adaptive_stats is None:
                    adaptive_stats = self.adaptively_binarizing(
                        pred=pred, gt=gt, FG=foreground, BG=background)
                handler.adaptive_results.append(handler(**adaptive_stats))
            if handler.binary_results is not None:
                if binary_stats is None:
                    binary_stats = self.get_statistics(
                        binary=pred > 0.5,
                        gt=gt,
                        FG=foreground,
                        BG=background)
                if handler.sample_based:
                    handler.binary_results.append(handler(**binary_stats))
                else:
                    for key in ('tp', 'fp', 'tn', 'fn'):
                        handler.binary_results[key] += binary_stats[key]


def evaluate_sequence(predictions, ground_truths):
    """Evaluate one video and return its five paper metrics."""
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f'Prediction/GT count mismatch: {len(predictions)} and '
            f'{len(ground_truths)}')
    if not predictions:
        raise ValueError('A sequence must contain at least one frame')

    mae_metric = MAE()
    s_metric = Smeasure()
    weighted_f_metric = WeightedFmeasure()
    overlap_metric = _PreparedFmeasureV2(metric_handlers={
        'iou': IOUHandler(with_adaptive=True, with_dynamic=True),
        'dice': DICEHandler(with_adaptive=True, with_dynamic=True),
    })

    mae_values = []
    s_values = []
    weighted_f_values = []
    for prediction, ground_truth in zip(predictions, ground_truths):
        prediction = _as_prediction(prediction)
        ground_truth = _as_ground_truth(ground_truth)
        if prediction.shape != ground_truth.shape:
            raise ValueError(
                f'Prediction shape {prediction.shape} does not match '
                f'ground truth {ground_truth.shape}')

        mae_values.append(mae_metric.cal_mae(prediction, ground_truth))
        s_values.append(s_metric.cal_sm(prediction, ground_truth))
        if not ground_truth.any() and not (prediction >= 0.5).any():
            weighted_f = 1.0
        else:
            weighted_f = weighted_f_metric.cal_wfm(
                prediction, ground_truth)
            if np.isnan(weighted_f):
                weighted_f = 0.0
        weighted_f_values.append(weighted_f)
        overlap_metric.step(prediction, ground_truth, normalize=False)

    overlap_results = overlap_metric.get_results()
    return OrderedDict(
        Salpha=float(np.mean(s_values)),
        Fwbeta=float(np.mean(weighted_f_values)),
        MAE=float(np.mean(mae_values)),
        mDice=float(overlap_results['dice']['dynamic'].mean()),
        mIoU=float(overlap_results['iou']['dynamic'].mean()))


def average_sequence_metrics(sequence_metrics):
    """Average per-sequence metrics with equal weight per video."""
    sequence_metrics = list(sequence_metrics)
    if not sequence_metrics:
        raise ValueError('No sequence metrics were provided')
    return OrderedDict(
        (name, float(np.mean([metrics[name] for metrics in sequence_metrics])))
        for name in METRIC_NAMES)


__all__ = ['METRIC_NAMES', 'average_sequence_metrics', 'evaluate_sequence']
