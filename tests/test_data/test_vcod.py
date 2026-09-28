import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mmseg.datasets.vcod import (CADDataset, COD10KDataset,
                                 MoCAMaskDataset, natural_sort_key)


def _save_rgb(path, value=32):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 5, 3), value, dtype=np.uint8)).save(path)


def _save_mask(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(values, dtype=np.uint8)).save(path)


def test_natural_sort_key_orders_frame_numbers():
    names = ['frame10.png', 'frame2.png', 'frame1.png']
    assert sorted(names, key=natural_sort_key) == [
        'frame1.png', 'frame2.png', 'frame10.png'
    ]


def test_cod10k_pairs_cam_files_and_builds_category_metadata(tmp_path):
    image_dir = tmp_path / 'Train' / 'Image'
    mask_dir = tmp_path / 'Train' / 'GT_Object'
    stems = [
        'COD10K-CAM-1-Aquatic-1-BatFish-10',
        'COD10K-CAM-1-Aquatic-1-BatFish-2',
        'COD10K-CAM-1-Aquatic-1-BatFish-1',
    ]
    for index, stem in enumerate(stems):
        _save_rgb(image_dir / f'{stem}.jpg', index)
        _save_mask(mask_dir / f'{stem}.png', np.full((4, 5), 255))

    noncam = 'COD10K-NonCAM-1-Aquatic-1-BatFish-11'
    _save_rgb(image_dir / f'{noncam}.jpg')
    _save_mask(mask_dir / f'{noncam}.png', np.zeros((4, 5)))

    dataset = COD10KDataset(pipeline=[], data_root=tmp_path)

    assert len(dataset) == 3
    infos = dataset.img_infos
    assert [Path(info['filename']).stem.rsplit('-', 1)[-1]
            for info in infos] == ['1', '10', '2']
    assert all(info['sequence_id'] == 'BatFish' for info in infos)
    assert infos[0]['is_first']
    assert infos[0]['prev_filename'] == infos[1]['filename']
    assert infos[1]['prev_filename'] == infos[0]['filename']
    assert infos[0]['filename'] not in infos[0]['ref_filenames']
    assert infos[0]['filename'] not in infos[1]['ref_filenames']
    assert infos[1]['filename'] in infos[1]['ref_filenames']
    assert infos[0]['ori_filename'] == Path(infos[0]['filename']).name
    assert set(np.unique(dataset.get_gt_seg_maps()[0])) == {1}


def test_cod10k_rejects_unpaired_files(tmp_path):
    image_dir = tmp_path / 'Train' / 'Image'
    mask_dir = tmp_path / 'Train' / 'GT_Object'
    _save_rgb(image_dir / 'COD10K-CAM-1-Aquatic-1-Fish-1.jpg')
    _save_mask(
        mask_dir / 'COD10K-CAM-1-Aquatic-1-Fish-1.png',
        np.full((4, 5), 255))
    _save_rgb(image_dir / 'COD10K-CAM-1-Aquatic-1-Fish-2.jpg')

    with pytest.raises(ValueError, match='Image/mask key mismatch'):
        COD10KDataset([], tmp_path)


def test_cod10k_sample_stride_is_applied_per_category(tmp_path):
    image_dir = tmp_path / 'Test' / 'Image'
    mask_dir = tmp_path / 'Test' / 'GT_Object'
    for frame in range(5):
        stem = f'COD10K-CAM-1-Aquatic-1-Fish-{frame:05d}'
        _save_rgb(image_dir / f'{stem}.jpg')
        _save_mask(mask_dir / f'{stem}.png', np.full((4, 5), 255))

    dataset = COD10KDataset(
        [], tmp_path, split='test', sample_stride=2)

    assert [Path(info['filename']).stem.rsplit('-', 1)[-1]
            for info in dataset.img_infos] == ['00000', '00002', '00004']


def test_dataset_evaluate_can_return_only_fwbeta(tmp_path):
    image_dir = tmp_path / 'Test' / 'Image'
    mask_dir = tmp_path / 'Test' / 'GT_Object'
    stem = 'COD10K-CAM-1-Aquatic-1-Fish-00000'
    ground_truth = np.asarray(
        [[0, 255, 0, 255, 0]] * 4, dtype=np.uint8)
    _save_rgb(image_dir / f'{stem}.jpg')
    _save_mask(mask_dir / f'{stem}.png', ground_truth)
    dataset = COD10KDataset([], tmp_path, split='test')

    metrics = dataset.evaluate(
        [(ground_truth > 0).astype(np.uint8)], metric='Fwbeta')

    assert set(metrics) == {'Fwbeta'}


def test_moca_alias_pairs_by_stem_and_uses_only_earlier_references(tmp_path):
    sequence_root = tmp_path / 'TrainDataset_per_sq' / 'seqA'
    for frame in ('00010', '00002', '00001'):
        _save_rgb(sequence_root / 'Imgs' / f'{frame}.jpg')
        _save_mask(sequence_root / 'GT' / f'{frame}.png',
                   np.full((4, 5), 255))

    dataset = MoCAMaskDataset(
        pipeline=[], data_root=tmp_path, split='train', skip_empty_gt=False)

    assert [Path(info['filename']).stem for info in dataset.img_infos] == [
        '00001', '00002', '00010'
    ]
    first, second, third = dataset.img_infos
    assert first['prev_filename'] == first['filename']
    assert first['prev_seg_map'] is None
    assert first['ref_filenames'] == [first['filename']]
    assert first['ref_seg_maps'] == [None]
    assert second['prev_filename'] == first['filename']
    assert third['ref_filenames'] == [first['filename'], second['filename']]
    assert third['ori_filename'] == 'seqA/00010.jpg'


def test_moca_rejects_unpaired_files(tmp_path):
    sequence_root = tmp_path / 'TrainDataset_per_sq' / 'seqA'
    _save_rgb(sequence_root / 'Imgs' / '00000.jpg')
    _save_mask(
        sequence_root / 'GT' / '00001.png', np.full((4, 5), 255))

    with pytest.raises(ValueError, match='Image/mask key mismatch'):
        MoCAMaskDataset([], tmp_path)


def test_moca_order_manifest_preserves_temporal_links(tmp_path):
    split_root = tmp_path / 'TrainDataset_per_sq'
    for sequence, frames in {'seqA': ('00001', '00002'),
                             'seqB': ('00001', )}.items():
        for frame in frames:
            _save_rgb(split_root / sequence / 'Imgs' / f'{frame}.jpg')
            _save_mask(
                split_root / sequence / 'GT' / f'{frame}.png',
                np.full((4, 5), 255))

    default = MoCAMaskDataset([], tmp_path)
    temporal_links = {
        record['ori_filename']: (
            record['prev_filename'], tuple(record['ref_filenames']))
        for record in default.img_infos
    }
    order = ['seqB/00001.jpg', 'seqA/00002.jpg', 'seqA/00001.jpg']
    manifest = tmp_path / 'order.txt'
    manifest.write_text('\n'.join(order), encoding='utf-8')

    reordered = MoCAMaskDataset(
        [], tmp_path, order_manifest=manifest)

    assert [record['ori_filename'] for record in reordered.img_infos] == order
    assert {
        record['ori_filename']: (
            record['prev_filename'], tuple(record['ref_filenames']))
        for record in reordered.img_infos
    } == temporal_links


def test_moca_order_manifest_requires_exact_unique_sample_set(tmp_path):
    split_root = tmp_path / 'TrainDataset_per_sq' / 'seqA'
    for frame in ('00001', '00002'):
        _save_rgb(split_root / 'Imgs' / f'{frame}.jpg')
        _save_mask(
            split_root / 'GT' / f'{frame}.png', np.full((4, 5), 255))
    manifest = tmp_path / 'order.txt'
    manifest.write_text(
        'seqA/00001.jpg\nseqA/00001.jpg\nunknown/00001.jpg\n',
        encoding='utf-8')

    with pytest.raises(
            ValueError, match=r'missing=1, extra=1, duplicates=1'):
        MoCAMaskDataset([], tmp_path, order_manifest=manifest)


def test_cad_pairs_different_names_and_skips_background_masks(tmp_path):
    sequence_root = tmp_path / 'original_data' / 'scorpion1'
    for frame in ('010', '002', '001'):
        _save_rgb(sequence_root / 'frames' / f'scorpion1_{frame}.png')

    _save_mask(sequence_root / 'groundtruth' / '001_gt.png',
               np.ones((4, 5)))
    foreground = np.ones((4, 5))
    foreground[1:3, 1:4] = 2
    _save_mask(sequence_root / 'groundtruth' / '002_gt.png', foreground)
    _save_mask(sequence_root / 'groundtruth' / '010_gt.png', foreground)

    # Exercise the original positional order through ``test_mode``.
    dataset = CADDataset(
        [], tmp_path, True, 'test', False, paper_protocol=False)

    assert len(dataset) == 2
    assert dataset.requires_sequence_inference
    first, second = dataset.img_infos
    assert Path(first['filename']).stem.endswith('002')
    assert Path(second['filename']).stem.endswith('010')
    assert first['prev_filename'] == first['filename']
    assert first['prev_seg_map'] is None
    assert second['prev_filename'] == first['filename']
    assert second['ori_filename'] == 'scorpion1/scorpion1_010.png'
    assert set(np.unique(dataset.get_gt_seg_maps()[0])) == {0, 1}


def test_cad_processed_layout_maps_sequential_jpegs_to_protocol_names(
        tmp_path, monkeypatch):
    protocol = [
        Path('chameleon/chameleon_001.png'),
        Path('chameleon/chameleon_006.png'),
        Path('frog/frog_001.png'),
    ]
    monkeypatch.setattr(
        'mmseg.datasets.vcod.load_cad_paper_protocol', lambda: protocol)

    for sequence_name, actual_names in {
            'chameleon': ('00001', '00000'),
            'frog': ('00000', ),
    }.items():
        for actual_name in actual_names:
            _save_rgb(
                tmp_path / 'Imgs' / sequence_name / f'{actual_name}.jpg')
            # Processed CAD mirrors may contain a same-stem PNG copy. It is
            # not an additional RGB frame.
            _save_rgb(
                tmp_path / 'Imgs' / sequence_name / f'{actual_name}.png')
            _save_mask(
                tmp_path / 'GT' / sequence_name / f'{actual_name}.png',
                np.full((4, 5), 255))

    dataset = CADDataset(
        pipeline=[], data_root=tmp_path, paper_protocol=True)

    assert [info['ori_filename'] for info in dataset.img_infos] == [
        entry.as_posix() for entry in protocol
    ]
    assert [Path(info['filename']).stem for info in dataset.img_infos] == [
        '00000', '00001', '00000'
    ]
    assert all(
        Path(info['filename']).suffix.lower() == '.jpg'
        for info in dataset.img_infos)
    assert all(
        Path(info['filename']).parts[0] == 'Imgs'
        for info in dataset.img_infos)
    assert all(
        Path(info['ann']['seg_map']).parts[0] == 'GT'
        for info in dataset.img_infos)
    assert dataset.img_infos[1]['prev_filename'] == \
        dataset.img_infos[0]['filename']


@pytest.mark.parametrize('actual_names', [
    ('00000', ),
    ('00001', '00002'),
    ('00000', '00002'),
])
def test_cad_processed_layout_requires_contiguous_stems(
        tmp_path, monkeypatch, actual_names):
    protocol = [
        Path('frog/frog_001.png'),
        Path('frog/frog_006.png'),
    ]
    monkeypatch.setattr(
        'mmseg.datasets.vcod.load_cad_paper_protocol', lambda: protocol)
    for stem in actual_names:
        _save_rgb(tmp_path / 'Imgs' / 'frog' / f'{stem}.jpg')
        _save_mask(
            tmp_path / 'GT' / 'frog' / f'{stem}.png',
            np.full((4, 5), 255))

    with pytest.raises(ValueError, match='must be numbered'):
        CADDataset([], tmp_path)


def test_cad_processed_layout_requires_matching_rgb_gt_stems(
        tmp_path, monkeypatch):
    protocol = [
        Path('frog/frog_001.png'),
        Path('frog/frog_006.png'),
    ]
    monkeypatch.setattr(
        'mmseg.datasets.vcod.load_cad_paper_protocol', lambda: protocol)
    for stem in ('00000', '00001'):
        _save_rgb(tmp_path / 'Imgs' / 'frog' / f'{stem}.jpg')
    for stem in ('00000', '00002'):
        _save_mask(
            tmp_path / 'GT' / 'frog' / f'{stem}.png',
            np.full((4, 5), 255))

    with pytest.raises(ValueError, match='Image/mask key mismatch'):
        CADDataset([], tmp_path)


@pytest.mark.skipif(
    os.path.normcase('frog') == os.path.normcase('FROG'),
    reason='requires a case-sensitive filesystem')
def test_cad_processed_layout_rejects_case_duplicate_sequences(
        tmp_path, monkeypatch):
    protocol = [Path('frog/frog_001.png')]
    monkeypatch.setattr(
        'mmseg.datasets.vcod.load_cad_paper_protocol', lambda: protocol)
    for sequence in ('frog', 'FROG'):
        _save_rgb(tmp_path / 'Imgs' / sequence / '00000.jpg')
        _save_mask(
            tmp_path / 'GT' / sequence / '00000.png',
            np.full((4, 5), 255))

    with pytest.raises(
            ValueError, match='Duplicate CAD sequence name ignoring case'):
        CADDataset([], tmp_path)
