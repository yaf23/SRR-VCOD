# Copyright (c) OpenMMLab. All rights reserved.
"""Evaluation hooks for SRRNet sequence inference.

The direct dataset adapters yield the current RGB frame only. Evaluation uses
:func:`single_gpu_sequence_inference` to keep the previous image, previous
prediction and best reference in memory and assemble the 11-channel model
input.

Predictions are evaluated in memory by the configured dataset. In particular,
ground-truth paths and mask normalisation come from ``dataset.img_infos`` and
``dataset.get_gt_seg_maps`` instead of dataset-name switches or prediction/GT
basename guesses.
"""

import warnings

import torch.distributed as dist
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm


def _is_sequence_dataset(dataset):
    """Return whether a dataset, possibly wrapped, is sequence ordered."""
    if (getattr(dataset, 'requires_sequence_inference', False)
            or hasattr(dataset, '_direct_img_infos')):
        return True

    wrapped = getattr(dataset, 'dataset', None)
    if wrapped is not None and wrapped is not dataset:
        return _is_sequence_dataset(wrapped)

    children = getattr(dataset, 'datasets', None)
    if children:
        return all(_is_sequence_dataset(child) for child in children)
    return False


def _run_sequence_inference(model, dataloader):
    """Run SRR sequential inference without writing intermediate masks."""
    from mmseg.apis.sequence_inference import single_gpu_sequence_inference
    return single_gpu_sequence_inference(model, dataloader, out_dir=None)


def _validate_result_count(results, dataset):
    """Fail early if predictions cannot be aligned with annotations."""
    if len(results) != len(dataset):
        raise RuntimeError(
            'Evaluation produced a different number of predictions and '
            f'annotations: {len(results)} predictions for {len(dataset)} '
            'dataset samples. Stateful evaluation requires an ordered, '
            'non-shuffled validation loader with samples_per_gpu=1.')


class EvalHook(_EvalHook):
    """Single-GPU evaluation with stateful inference for direct SRR data."""

    greater_keys = ['Salpha', 'Fwbeta', 'mDice', 'mIoU']
    less_keys = ['MAE']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 efficient_test=False,
                 pre_eval=False,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval
        self.latest_results = None

        if efficient_test:
            warnings.warn(
                '``efficient_test`` is deprecated; evaluation results are '
                'already CPU-memory friendly.',
                DeprecationWarning,
                stacklevel=2)

    def _do_evaluate(self, runner):
        if not self._should_evaluate(runner):
            return

        dataset = self.dataloader.dataset
        if not _is_sequence_dataset(dataset):
            raise TypeError(
                'SRRNet evaluation requires an ordered sequence dataset')
        results = _run_sequence_inference(runner.model, self.dataloader)

        _validate_result_count(results, dataset)
        self.latest_results = results
        runner.log_buffer.clear()
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best and key_score is not None:
            self._save_ckpt(runner, key_score)


class DistEvalHook(_DistEvalHook):
    """Distributed-training hook with rank-zero sequential evaluation.

    The validation loader in :mod:`mmseg.apis.train` is intentionally built
    with ``dist=False`` and ``shuffle=False`` so rank zero sees each complete
    sequence in order. All ranks synchronise before training resumes.
    """

    greater_keys = ['Salpha', 'Fwbeta', 'mDice', 'mIoU']
    less_keys = ['MAE']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 efficient_test=False,
                 pre_eval=False,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval
        self.latest_results = None

        if efficient_test:
            warnings.warn(
                '``efficient_test`` is deprecated; evaluation results are '
                'already CPU-memory friendly.',
                DeprecationWarning,
                stacklevel=2)

    def _broadcast_bn_buffers(self, runner):
        """Use rank-zero BatchNorm statistics consistently on every rank."""
        if not self.broadcast_bn_buffer:
            return
        for module in runner.model.modules():
            if isinstance(module, _BatchNorm) and module.track_running_stats:
                dist.broadcast(module.running_var, 0)
                dist.broadcast(module.running_mean, 0)

    def _do_evaluate(self, runner):
        if not self._should_evaluate(runner):
            return

        dataset = self.dataloader.dataset
        if not _is_sequence_dataset(dataset):
            raise TypeError(
                'SRRNet evaluation requires an ordered sequence dataset')

        self._broadcast_bn_buffers(runner)

        if runner.rank == 0:
            results = _run_sequence_inference(runner.model, self.dataloader)
            _validate_result_count(results, dataset)

            self.latest_results = results
            runner.log_buffer.clear()
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)
            if self.save_best and key_score is not None:
                self._save_ckpt(runner, key_score)

        # Non-zero ranks wait here while rank zero traverses the complete
        # non-distributed validation loader.
        dist.barrier()
