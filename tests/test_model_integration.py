import copy
import hashlib
import runpy
from pathlib import Path

import pytest
import torch

from mmseg.models import build_segmentor


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / 'configs' / 'srrnet' / 'moca_finetune.py'
EXPECTED_SCHEMA_SHA256 = (
    '494bada44814e5f13e65384a7aff9070a8a9e71ec9450bcfbe520aec4ac5f072')


@pytest.fixture(scope='module')
def public_model():
    config = runpy.run_path(str(CONFIG_PATH))
    model_config = copy.deepcopy(config['model'])
    model_config['pretrained'] = None
    return build_segmentor(model_config)


def _state_schema_sha256(state_dict):
    lines = []
    for name, tensor in state_dict.items():
        shape = ','.join(str(dimension) for dimension in tensor.shape)
        dtype = str(tensor.dtype).replace('torch.', '', 1)
        lines.append(f'{name}\t{shape}\t{dtype}\n')
    return hashlib.sha256(''.join(lines).encode('utf-8')).hexdigest()


def test_public_model_preserves_checkpoint_schema(public_model):
    state_dict = public_model.state_dict()

    assert len(state_dict) == 598
    assert tuple(
        state_dict['backbone.backbone.patch_embed1.proj.weight'].shape
    ) == (64, 3, 7, 7)
    assert tuple(state_dict['decode_head.linear_pred.bias'].shape) == (2, )
    assert tuple(
        state_dict['decode_head.mae_pred.weight'].shape
    ) == (1, 258, 1, 1)
    assert _state_schema_sha256(state_dict) == EXPECTED_SCHEMA_SHA256


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is required for model smoke')
def test_public_model_forward_contract_on_cuda(public_model):
    device = torch.device('cuda', 0)
    model = public_model.to(device).eval()
    image = torch.randn(1, 11, 64, 64, device=device)
    ground_truth = torch.zeros(
        1, 1, 64, 64, dtype=torch.long, device=device)
    ground_truth[:, :, 16:48, 16:48] = 1
    metadata = [dict(
        ori_shape=(64, 64, 3),
        img_shape=(64, 64, 3),
        pad_shape=(64, 64, 3),
        flip=False,
        flip_direction=None)]

    with torch.no_grad():
        losses = model.forward_train(image, metadata, ground_truth)
        prediction, confidence, raw_mask = model.simple_test(
            image, metadata, rescale=True)

    assert set(losses) == {'decode.loss_seg', 'decode.loss_conf'}
    assert all(
        loss.ndim == 0 and torch.isfinite(loss)
        for loss in losses.values())
    assert len(prediction) == 1 and prediction[0].shape == (64, 64)
    assert confidence.shape[:2] == (1, 1)
    assert torch.isfinite(confidence).all()
    assert tuple(raw_mask.shape) == (1, 1, 64, 64)
