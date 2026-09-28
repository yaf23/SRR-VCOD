from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from torch.utils.data import DataLoader, Dataset

from mmseg.core.evaluation import eval_hooks


class _SequenceDataset(Dataset):

    requires_sequence_inference = True

    def __init__(self, length=2):
        self.length = length
        self.evaluated_results = None
        self.evaluated_kwargs = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return index

    def evaluate(self, results, logger=None, **kwargs):
        del logger
        self.evaluated_results = results
        self.evaluated_kwargs = kwargs
        return {'mIoU': 0.75}


class _LogBuffer:

    def __init__(self):
        self.output = {}
        self.ready = False

    def clear(self):
        self.output = {}
        self.ready = False


def _runner(rank=0):
    return SimpleNamespace(
        rank=rank,
        model=object(),
        logger=None,
        log_buffer=_LogBuffer(),
    )


class EvalHooksTest(unittest.TestCase):

    def test_single_gpu_sequence_dataset_uses_stateful_inference(self):
        dataset = _SequenceDataset()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        predictions = [
            np.zeros((2, 2), dtype=np.uint8)
            for _ in range(len(dataset))
        ]
        calls = []

        def fake_sequence_inference(model, loader):
            calls.append((model, loader))
            return predictions

        hook = eval_hooks.EvalHook(
            dataloader, interval=1, metric='VCOD')
        runner = _runner()
        with patch.object(
                eval_hooks, '_run_sequence_inference',
                fake_sequence_inference), \
                patch.object(hook, '_should_evaluate', return_value=True):
            hook._do_evaluate(runner)

        self.assertEqual(calls, [(runner.model, dataloader)])
        self.assertIs(hook.latest_results, predictions)
        self.assertIs(dataset.evaluated_results, predictions)
        self.assertEqual(dataset.evaluated_kwargs, {'metric': 'VCOD'})
        self.assertEqual(runner.log_buffer.output['mIoU'], 0.75)

    def test_distributed_hook_evaluates_results_via_dataset(self):
        dataset = _SequenceDataset()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        predictions = [
            np.ones((2, 2), dtype=np.uint8)
            for _ in range(len(dataset))
        ]
        barriers = []

        hook = eval_hooks.DistEvalHook(
            dataloader,
            interval=1,
            metric='VCOD',
            broadcast_bn_buffer=False)
        runner = _runner(rank=0)
        with patch.object(
                eval_hooks, '_run_sequence_inference',
                return_value=predictions), \
                patch.object(
                    eval_hooks.dist, 'barrier',
                    side_effect=lambda: barriers.append(1)), \
                patch.object(hook, '_should_evaluate', return_value=True):
            hook._do_evaluate(runner)

        self.assertIs(hook.latest_results, predictions)
        self.assertIs(dataset.evaluated_results, predictions)
        self.assertEqual(dataset.evaluated_kwargs, {'metric': 'VCOD'})
        self.assertEqual(barriers, [1])

    def test_result_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'different number'):
            eval_hooks._validate_result_count(
                [np.zeros((1, 1))], _SequenceDataset(2))


if __name__ == '__main__':
    unittest.main()
