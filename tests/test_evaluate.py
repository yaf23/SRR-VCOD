import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from evaluate import discover_pairs, parse_args as parse_evaluation_args
from infer import parse_args as parse_inference_args
from mmseg.apis.inference import _index_frames
from mmseg.datasets.cad_protocol import load_cad_paper_protocol
from mmseg.datasets.vcod import read_binary_mask


def _save_rgb(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (3, 2), color=(20, 40, 60)).save(path)


def _save_mask(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(values, dtype=np.uint8)).save(path)


class EvaluationPairTest(unittest.TestCase):

    def test_cli_defaults_to_moca(self):
        inference_args = parse_inference_args([
            '--config', 'model.py', '--checkpoint', 'model.pth',
            '--data-root', 'MoCA_Video', '--output-dir', 'predictions'])
        with patch('sys.argv', [
                'evaluate.py', '--data-root', 'MoCA_Video',
                '--pred-dir', 'predictions']):
            evaluation_args = parse_evaluation_args()

        self.assertEqual(inference_args.dataset, 'moca')
        self.assertEqual(evaluation_args.dataset, 'moca')

    def test_cad_evaluation_uses_paper_protocol_by_default(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / 'CAD-wrapper'
            prediction_root = Path(temporary_directory) / 'predictions'
            expected = []
            for entry in load_cad_paper_protocol():
                sequence_name, frame_name = entry.parts
                sequence = root / 'original_data' / sequence_name
                _save_rgb(sequence / 'frames' / frame_name)
                _save_mask(
                    sequence / 'groundtruth'
                    / f'{Path(frame_name).stem}_gt.png',
                    [[1, 2, 1], [1, 1, 1]])
                _save_mask(
                    prediction_root / sequence_name / frame_name,
                    [[0, 255, 0], [0, 0, 0]])
                expected.append((sequence_name, Path(frame_name).stem))

            extra_sequence = root / 'original_data' / 'chameleon'
            _save_rgb(extra_sequence / 'frames' / 'chameleon_999.png')
            _save_mask(
                extra_sequence / 'groundtruth' / 'chameleon_999_gt.png',
                [[1, 2, 1], [1, 1, 1]])

            records = discover_pairs('cad', root, prediction_root)

            self.assertEqual(
                [(record['sequence_id'], record['frame_name'])
                 for record in records],
                expected)

    def test_processed_cad_inference_and_evaluation_keep_protocol_names(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / 'CAD'
            prediction_root = Path(temporary_directory) / 'predictions'
            indexed_files = []
            sequence_counts = {}
            protocol = [
                Path('chameleon/chameleon_001.png'),
                Path('chameleon/chameleon_006.png'),
                Path('frog/frog_001.png'),
            ]

            for entry in protocol:
                sequence_name, _ = entry.parts
                frame_index = sequence_counts.get(sequence_name, 0)
                actual_name = f'{frame_index:05d}'
                image_directory = root / 'Imgs' / sequence_name
                _save_rgb(image_directory / f'{actual_name}.jpg')
                _save_rgb(image_directory / f'{actual_name}.png')
                indexed_files.append((entry, actual_name))
                sequence_counts[sequence_name] = frame_index + 1

            # RGB-only inference must work before a GT directory exists.
            with patch(
                    'mmseg.apis.inference.load_cad_paper_protocol',
                    return_value=protocol):
                frames = _index_frames(root, 'cad')
            self.assertEqual(
                [frame['ori_filename'] for frame in frames],
                [entry.as_posix() for entry in protocol])
            self.assertEqual(
                [Path(frame['filename']).stem for frame in frames],
                [actual_name for _, actual_name in indexed_files])
            self.assertTrue(all(
                Path(frame['filename']).suffix.lower() == '.jpg'
                for frame in frames))

            for entry, actual_name in indexed_files:
                sequence_name, frame_name = entry.parts
                _save_mask(
                    root / 'GT' / sequence_name / f'{actual_name}.png',
                    [[0, 255, 0], [0, 0, 0]])
                _save_mask(
                    prediction_root / sequence_name / frame_name,
                    [[0, 255, 0], [0, 0, 0]])

            with patch(
                    'mmseg.datasets.vcod.load_cad_paper_protocol',
                    return_value=protocol):
                records = discover_pairs('cad', root, prediction_root)

            self.assertEqual(
                [(record['sequence_id'], record['frame_name'])
                 for record in records],
                [(entry.parts[0], Path(entry.name).stem)
                 for entry in protocol])
            self.assertEqual(
                [record['ground_truth'].stem for record in records],
                [actual_name for _, actual_name in indexed_files])

    def test_processed_cad_inference_rejects_noncontiguous_stems(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / 'CAD'
            protocol = [
                Path('frog/frog_001.png'),
                Path('frog/frog_006.png'),
            ]
            for stem in ('00000', '00002'):
                _save_rgb(root / 'Imgs' / 'frog' / f'{stem}.jpg')

            with patch(
                    'mmseg.apis.inference.load_cad_paper_protocol',
                    return_value=protocol), self.assertRaisesRegex(
                        ValueError, 'must be numbered'):
                _index_frames(root, 'cad')

    def test_sparse_moca_names_in_downloaded_test_layouts(self):
        for split_name in ('Test', 'TestDataset_per_sq'):
            with self.subTest(split_name=split_name), \
                    tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory) / 'MoCA_Video'
                sequence = root / split_name / 'crab'
                prediction_root = Path(temporary_directory) / 'predictions'
                predictions = prediction_root / 'crab'

                for frame_name in ('00000', '00005'):
                    _save_rgb(sequence / 'Imgs' / f'{frame_name}.jpg')
                    _save_mask(
                        sequence / 'GT' / f'{frame_name}.png',
                        [[0, 255, 0], [0, 0, 0]])
                    _save_mask(
                        predictions / f'{frame_name}.png',
                        [[0, 255, 0], [0, 0, 0]])

                records = discover_pairs('moca', root, prediction_root)

                self.assertEqual(
                    [record['frame_name'] for record in records],
                    ['00000', '00005'])

    def test_cad_pairing_and_mask_normalization(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / 'CAD-wrapper'
            sequence = root / 'original_data' / 'lizard'
            prediction_root = Path(temporary_directory) / 'predictions'
            predictions = prediction_root / 'lizard'

            _save_rgb(sequence / 'frames' / 'lizard_001.png')
            foreground_gt = sequence / 'groundtruth' / '001_gt.png'
            _save_mask(foreground_gt, [[1, 2, 1], [1, 1, 1]])
            _save_mask(
                predictions / 'lizard_001.png',
                [[0, 255, 0], [0, 0, 0]])

            _save_rgb(sequence / 'frames' / 'lizard_010.png')
            _save_mask(
                sequence / 'groundtruth' / '010_gt.png',
                [[1, 1, 1], [1, 1, 1]])

            records = discover_pairs(
                'cad', root, prediction_root, cad_paper_protocol=False)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['prediction'].name, 'lizard_001.png')
            np.testing.assert_array_equal(
                read_binary_mask(foreground_gt),
                np.asarray([[0, 1, 0], [0, 0, 0]], dtype=np.uint8))

    def test_missing_predictions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / 'MoCA_Video'
            sequence = root / 'Test' / 'crab'
            predictions = Path(temporary_directory) / 'predictions'
            _save_rgb(sequence / 'Imgs' / '00000.jpg')
            _save_mask(sequence / 'GT' / '00000.png', [[0, 255]])
            predictions.mkdir()

            with self.assertRaisesRegex(ValueError, 'missing predictions'):
                discover_pairs('moca', root, predictions)


if __name__ == '__main__':
    unittest.main()
