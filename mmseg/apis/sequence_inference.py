"""Ordered single-GPU sequence inference for SRRNet."""

import math
import os

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer
from PIL import Image


class SequenceInferenceState:
    """Previous-frame and best-reference state for one ordered sequence."""

    def __init__(self):
        self.sequence_id = None
        self.previous_image = None
        self.previous_mask = None
        self.reference_image = None
        self.reference_mask = None
        self.best_error = math.inf

    def reset(self, current_image, sequence_id):
        """Initialize a sequence from its first RGB frame and zero masks."""
        self.sequence_id = str(sequence_id)
        current_image = current_image.detach().clone()
        zero_mask = current_image.new_zeros(
            (current_image.shape[0], 1, *current_image.shape[-2:]))
        self.previous_image = current_image
        self.previous_mask = zero_mask.clone()
        self.reference_image = current_image.clone()
        self.reference_mask = zero_mask.clone()
        self.best_error = math.inf

    def update(self, current_image, raw_mask, update_reference=False):
        """Advance the previous frame and optionally replace the reference."""
        mask = _normalise_state_mask(raw_mask, current_image)
        current_image = current_image.detach().clone()
        self.previous_image = current_image
        self.previous_mask = mask
        if update_reference:
            self.reference_image = current_image.clone()
            self.reference_mask = mask.clone()

    def advance(self, current_image, raw_mask, predicted_error):
        """Advance state and keep the frame with the lowest predicted error."""
        predicted_error = float(predicted_error)
        update_reference = predicted_error < self.best_error
        if update_reference:
            self.best_error = predicted_error
        self.update(
            current_image, raw_mask, update_reference=update_reference)
        return update_reference


def _normalise_state_mask(raw_mask, current_image):
    """Convert the model's binary BCHW mask to the cached 0-255 scale."""
    if not torch.is_tensor(raw_mask):
        raise TypeError('The model must return the state mask as a tensor')
    expected_shape = (
        current_image.shape[0], 1, *current_image.shape[-2:])
    if tuple(raw_mask.shape) != expected_shape:
        raise ValueError(
            f'Expected state mask shape {expected_shape}, '
            f'received {tuple(raw_mask.shape)}')
    if not torch.all((raw_mask == 0) | (raw_mask == 1)):
        raise ValueError('The model state mask must contain only 0 and 1')
    return raw_mask.detach().to(
        device=current_image.device,
        dtype=current_image.dtype).mul(255).clone()


def assemble_sequence_input(current_rgb, sequence_id, state):
    """Append cached sequence channels to one current RGB tensor."""
    if not torch.is_tensor(current_rgb) or current_rgb.ndim != 4:
        raise ValueError('The current frame must be a BCHW tensor')
    if current_rgb.shape[0] != 1 or current_rgb.shape[1] != 3:
        raise ValueError(
            'Sequence inference accepts exactly one current RGB frame')

    sequence_id = str(sequence_id)
    new_sequence = state.sequence_id != sequence_id
    if new_sequence:
        state.reset(current_rgb, sequence_id)

    model_input = torch.cat((
        current_rgb,
        state.previous_image,
        state.previous_mask,
        state.reference_image,
        state.reference_mask,
    ), dim=1)
    return model_input, current_rgb, new_sequence


def _unwrap_container(value):
    if isinstance(value, DataContainer):
        value = value.data
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        return _unwrap_container(value)
    if isinstance(value, list):
        return [_unwrap_container(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_unwrap_container(item) for item in value)
    if isinstance(value, dict):
        return {key: _unwrap_container(item) for key, item in value.items()}
    return value


def _metadata_from_batch(data):
    metadata = data['img_metas']
    if (isinstance(metadata, (list, tuple)) and len(metadata) == 1
            and hasattr(metadata[0], 'data')):
        metadata = metadata[0].data
    metadata = _unwrap_container(metadata)
    while (isinstance(metadata, (list, tuple)) and len(metadata) == 1
           and not isinstance(metadata[0], dict)):
        metadata = metadata[0]
    if isinstance(metadata, dict):
        metadata = [metadata]
    if (not isinstance(metadata, (list, tuple)) or len(metadata) != 1
            or not isinstance(metadata[0], dict)):
        raise ValueError('Inference requires metadata for exactly one frame')
    return metadata[0]


def _current_rgb_from_batch(data):
    images = data.get('img')
    if isinstance(images, (list, tuple)):
        if len(images) != 1:
            raise ValueError('Test-time augmentation is not supported')
        images = images[0]
    current_rgb = _unwrap_container(images)
    if not torch.is_tensor(current_rgb):
        raise TypeError('The inference pipeline must produce a tensor')
    return current_rgb


def _sequence_id_from_meta(metadata):
    sequence_id = str(metadata['sequence_id'])
    if not sequence_id:
        raise ValueError('Frame metadata contains an empty sequence identifier')
    return sequence_id


def _frame_name_from_meta(metadata):
    source = metadata['ori_filename']
    basename = os.path.basename(str(source).replace('\\', '/'))
    if not basename:
        raise ValueError('Frame metadata contains an empty original filename')
    return os.path.splitext(basename)[0] + '.png'


def _binary_prediction(prediction):
    prediction = np.asarray(prediction)
    prediction = np.squeeze(prediction)
    if prediction.ndim != 2:
        raise ValueError(
            'Each predicted segmentation mask must be two-dimensional')
    if np.issubdtype(prediction.dtype, np.floating):
        foreground = prediction >= 0.5
    else:
        foreground = prediction > 0
    return foreground.astype(np.uint8) * 255


def _save_binary_prediction(metadata, prediction, output_dir):
    sequence_id = os.path.basename(
        _sequence_id_from_meta(metadata).replace('\\', '/').rstrip('/'))
    if sequence_id in ('', '.', '..'):
        raise ValueError('Invalid sequence identifier in frame metadata')
    sequence_dir = os.path.join(output_dir, sequence_id)
    mmcv.mkdir_or_exist(sequence_dir)
    output_path = os.path.join(
        sequence_dir, _frame_name_from_meta(metadata))
    Image.fromarray(_binary_prediction(prediction), mode='L').save(output_path)
    return output_path


def _call_model(model, data):
    if hasattr(model, 'module'):
        return model(return_loss=False, rescale=True, **data)
    plain_data = _unwrap_container(data)
    return model(return_loss=False, rescale=True, **plain_data)


def single_gpu_sequence_inference(model,
                                  data_loader,
                                  out_dir=None,
                                  score_records=None):
    """Run ordered SRRNet inference without reading annotations."""
    model.eval()
    state = SequenceInferenceState()
    predictions = []
    progress = mmcv.ProgressBar(len(data_loader.dataset))

    if out_dir is not None:
        mmcv.mkdir_or_exist(out_dir)

    for data in data_loader:
        metadata = _metadata_from_batch(data)
        sequence_id = _sequence_id_from_meta(metadata)
        current_rgb = _current_rgb_from_batch(data)
        model_input, current_rgb, _ = assemble_sequence_input(
            current_rgb, sequence_id, state)

        model_data = dict(data)
        model_data['img'] = [model_input]
        with torch.no_grad():
            model_output = _call_model(model, model_data)
        if not isinstance(model_output, (list, tuple)) or len(model_output) != 3:
            raise RuntimeError(
                'The model must return masks, predicted error, and a raw mask')
        batch_predictions, predicted_error, raw_mask = model_output
        if len(batch_predictions) != 1:
            raise RuntimeError(
                'Sequence inference requires a batch size of one')

        error_tensor = torch.as_tensor(predicted_error)
        if not error_tensor.is_floating_point():
            raise TypeError('The predicted error map must be floating point')
        # Preserve the checkpoint's inference precision. In the published
        # FP16 recipe, rounding occurs while reducing the error map and can
        # affect which frame becomes the long-term reference.
        score = float(error_tensor.mean().item())
        if not math.isfinite(score):
            raise RuntimeError('The predicted error score is not finite')
        reference_updated = state.advance(current_rgb, raw_mask, score)

        prediction = batch_predictions[0]
        predictions.append(prediction)
        output_path = None
        if out_dir is not None:
            saved_path = _save_binary_prediction(
                metadata, prediction, out_dir)
            output_path = os.path.relpath(saved_path, out_dir).replace(
                os.sep, '/')
        if score_records is not None:
            score_records.append({
                'sequence': sequence_id,
                'frame': _frame_name_from_meta(metadata),
                'predicted_error': score,
                'reference_updated': reference_updated,
                'output': output_path,
            })
        progress.update()

    return predictions


__all__ = [
    'SequenceInferenceState', 'assemble_sequence_input',
    'single_gpu_sequence_inference'
]
