import numpy as np
import pytest

from mmseg.core.evaluation.paper_metrics import (
    METRIC_NAMES, average_sequence_metrics, evaluate_sequence)


def test_perfect_predictions_score_near_one_on_paper_metrics():
    first = np.zeros((16, 16), dtype=np.uint8)
    first[3:13, 4:12] = 1
    second = np.zeros((16, 16), dtype=np.uint8)
    second[2:10, 6:15] = 1
    ground_truths = [first, second]
    predictions = [mask * 255 for mask in ground_truths]

    metrics = evaluate_sequence(predictions, ground_truths)

    assert tuple(metrics) == METRIC_NAMES
    assert metrics['MAE'] == pytest.approx(0.0, abs=1e-12)
    for name in ('Salpha', 'Fwbeta', 'mDice', 'mIoU'):
        assert 0.99 < metrics[name] <= 1.0


def test_sequence_average_weights_each_video_equally():
    first = dict.fromkeys(METRIC_NAMES, 0.25)
    second = dict.fromkeys(METRIC_NAMES, 0.75)

    averaged = average_sequence_metrics([first, second])

    assert tuple(averaged) == METRIC_NAMES
    assert all(value == pytest.approx(0.5) for value in averaged.values())


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match='does not match'):
        evaluate_sequence(
            [np.zeros((2, 2), dtype=np.uint8)],
            [np.zeros((3, 3), dtype=np.uint8)])
