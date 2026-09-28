import unittest
from unittest import mock

import numpy as np
import torch
from mmcv import Config

import mmseg.apis.inference as inference_api
from mmseg.apis.sequence_inference import (
    SequenceInferenceState, assemble_sequence_input,
    single_gpu_sequence_inference)


class _MetaContainer:

    def __init__(self, meta):
        self.data = [[meta]]


class _FakeLoader:

    def __init__(self, frames):
        self.frames = frames
        self.dataset = frames

    def __iter__(self):
        return iter(self.frames)


class _FakeModel:

    def __init__(self):
        self.inputs = []

    def eval(self):
        return self

    def __call__(self, return_loss=False, **data):
        del return_loss
        self.inputs.append(data['img'][0].clone())
        prediction = [np.zeros((2, 2), dtype=np.uint8)]
        predicted_error = torch.tensor(
            [0.1, 0.2], dtype=torch.float16)
        raw_mask = torch.ones((1, 1, 2, 2), dtype=torch.long)
        return prediction, predicted_error, raw_mask


class SequenceInferenceStateTest(unittest.TestCase):

    def test_cuda_model_uses_inference_fp16(self):
        cfg = Config(dict(
            model=dict(pretrained='mit_b3.pth', train_cfg=dict()),
            inference_fp16=True))
        model = mock.Mock()
        wrapped_model = object()

        with mock.patch.object(
                inference_api, 'build_segmentor', return_value=model), \
                mock.patch.object(inference_api, 'wrap_fp16_model') as wrap, \
                mock.patch.object(
                    inference_api, 'load_checkpoint',
                    return_value={}) as load_checkpoint, \
                mock.patch.object(
                    inference_api, 'MMDataParallel',
                    return_value=wrapped_model), \
                mock.patch.object(torch.cuda, 'set_device'):
            result = inference_api._build_model(
                cfg, 'model.pth', torch.device('cuda', 0))

        wrap.assert_called_once_with(model)
        load_checkpoint.assert_called_once_with(
            model, 'model.pth', map_location='cpu', strict=True)
        self.assertIs(result, wrapped_model)

    def test_first_and_second_frame_channel_order(self):
        state = SequenceInferenceState()
        first = torch.full((1, 3, 2, 2), 1.0)

        model_input, current, new_sequence = assemble_sequence_input(
            first, 'video-a', state)

        self.assertTrue(new_sequence)
        self.assertEqual(tuple(model_input.shape), (1, 11, 2, 2))
        torch.testing.assert_close(model_input[:, 0:3], first)
        torch.testing.assert_close(model_input[:, 3:6], first)
        torch.testing.assert_close(
            model_input[:, 6:7], torch.zeros_like(model_input[:, 6:7]))
        torch.testing.assert_close(model_input[:, 7:10], first)
        torch.testing.assert_close(
            model_input[:, 10:11], torch.zeros_like(model_input[:, 10:11]))

        first_raw_mask = torch.ones((1, 1, 2, 2), dtype=torch.long)
        state.update(current, first_raw_mask, update_reference=True)
        second = torch.full((1, 3, 2, 2), 2.0)

        model_input, _, new_sequence = assemble_sequence_input(
            second, 'video-a', state)

        self.assertFalse(new_sequence)
        torch.testing.assert_close(model_input[:, 0:3], second)
        torch.testing.assert_close(model_input[:, 3:6], first)
        torch.testing.assert_close(
            model_input[:, 6:7],
            torch.full((1, 1, 2, 2), 255.0))
        torch.testing.assert_close(model_input[:, 7:10], first)
        torch.testing.assert_close(
            model_input[:, 10:11],
            torch.full((1, 1, 2, 2), 255.0))

    def test_sequence_change_resets_all_temporal_channels(self):
        state = SequenceInferenceState()
        first = torch.full((1, 3, 2, 2), 1.0)
        _, current, _ = assemble_sequence_input(first, 'video-a', state)
        state.update(
            current, torch.ones((1, 1, 2, 2), dtype=torch.long),
            update_reference=True)

        next_sequence = torch.full((1, 3, 2, 2), 3.0)
        model_input, _, new_sequence = assemble_sequence_input(
            next_sequence, 'video-b', state)

        self.assertTrue(new_sequence)
        self.assertEqual(state.sequence_id, 'video-b')
        torch.testing.assert_close(model_input[:, 0:3], next_sequence)
        torch.testing.assert_close(model_input[:, 3:6], next_sequence)
        torch.testing.assert_close(
            model_input[:, 6:7], torch.zeros_like(model_input[:, 6:7]))
        torch.testing.assert_close(model_input[:, 7:10], next_sequence)
        torch.testing.assert_close(
            model_input[:, 10:11], torch.zeros_like(model_input[:, 10:11]))

    def test_equal_score_does_not_replace_reference(self):
        state = SequenceInferenceState()
        first = torch.full((1, 3, 2, 2), 1.0)
        _, current, _ = assemble_sequence_input(first, 'video-a', state)
        self.assertTrue(state.advance(current, torch.zeros(1, 1, 2, 2), 0.2))

        second = torch.full((1, 3, 2, 2), 2.0)
        self.assertFalse(
            state.advance(second, torch.ones(1, 1, 2, 2), 0.2))
        torch.testing.assert_close(state.reference_image, first)

    def test_inference_without_annotations_or_output_directory(self):
        first = torch.full((1, 3, 2, 2), 1.0)
        second = torch.full((1, 3, 2, 2), 2.0)
        frames = [
            {
                'img': [first],
                'img_metas': [_MetaContainer({
                    'ori_filename': 'video-a/frame-alpha.jpg',
                    'sequence_id': 'video-a',
                })],
            },
            {
                'img': [second],
                'img_metas': [_MetaContainer({
                    'ori_filename': 'video-a/frame-beta.jpg',
                    'sequence_id': 'video-a',
                })],
            },
        ]
        model = _FakeModel()
        score_records = []

        results = single_gpu_sequence_inference(
            model, _FakeLoader(frames), out_dir=None,
            score_records=score_records)

        self.assertEqual(len(results), 2)
        self.assertEqual(len(model.inputs), 2)
        self.assertEqual(tuple(model.inputs[0].shape), (1, 11, 2, 2))
        torch.testing.assert_close(model.inputs[1][:, 0:3], second)
        torch.testing.assert_close(model.inputs[1][:, 3:6], first)
        torch.testing.assert_close(
            model.inputs[1][:, 6:7],
            torch.full((1, 1, 2, 2), 255.0))
        self.assertTrue(score_records[0]['reference_updated'])
        self.assertFalse(score_records[1]['reference_updated'])
        expected_score = float(torch.tensor(
            [0.1, 0.2], dtype=torch.float16).mean().item())
        self.assertEqual(
            score_records[0]['predicted_error'], expected_score)


if __name__ == '__main__':
    unittest.main()
