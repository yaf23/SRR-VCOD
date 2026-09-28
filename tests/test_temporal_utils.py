import pytest
import torch

from mmseg.utils.temporal import split_temporal_input


def test_split_temporal_input_preserves_channel_order():
    tensor = torch.arange(11, dtype=torch.float32).reshape(1, 11, 1, 1)

    current, previous, previous_mask, reference, reference_mask = (
        split_temporal_input(tensor))

    torch.testing.assert_close(current.flatten(), torch.tensor([0., 1., 2.]))
    torch.testing.assert_close(previous.flatten(), torch.tensor([3., 4., 5.]))
    torch.testing.assert_close(previous_mask.flatten(), torch.tensor([6.]))
    torch.testing.assert_close(reference.flatten(), torch.tensor([7., 8., 9.]))
    torch.testing.assert_close(reference_mask.flatten(), torch.tensor([10.]))


@pytest.mark.parametrize('shape', [(11, 2, 2), (1, 10, 2, 2)])
def test_split_temporal_input_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError):
        split_temporal_input(torch.zeros(shape))
