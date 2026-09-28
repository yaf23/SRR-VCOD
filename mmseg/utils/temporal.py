"""Utilities for the temporal input consumed by SRRNet."""

from torch import Tensor


def split_temporal_input(x: Tensor):
    """Split an SRRNet input tensor into its five ordered components.

    The channel layout is fixed to current RGB, previous-frame RGB,
    previous-frame mask, reference-frame RGB and reference-frame mask.

    Args:
        x: A tensor with shape ``(N, 11, H, W)``.

    Returns:
        A tuple containing the current frame, previous frame, previous mask,
        reference frame and reference mask. Slicing returns views and therefore
        does not alter values or allocate copies.
    """
    if x.ndim != 4:
        raise ValueError(
            f'Temporal input must have shape (N, 11, H, W), got {x.ndim} '
            'dimensions.')
    if x.shape[1] != 11:
        raise ValueError(
            f'Temporal input must contain exactly 11 channels, got '
            f'{x.shape[1]}.')

    current_frame = x[:, 0:3, :, :]
    previous_frame = x[:, 3:6, :, :]
    previous_mask = x[:, 6:7, :, :]
    reference_frame = x[:, 7:10, :, :]
    reference_mask = x[:, 10:11, :, :]
    return (current_frame, previous_frame, previous_mask, reference_frame,
            reference_mask)


__all__ = ['split_temporal_input']
