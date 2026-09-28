import numpy as np


def normalize_binary_mask(mask, foreground_value=1):
    """Normalize common binary-mask encodings without rewriting the dataset.

    MoCA-Mask commonly uses 0/255, while the original CAD release uses 1/2
    (background/foreground).  The model training target uses 0/1 and the
    temporal mask input uses 0/255, so callers choose the desired foreground
    value.
    """
    mask = np.asarray(mask)
    values = np.unique(mask)
    if values.size and np.all(np.isin(values, (1, 2))):
        foreground = mask == 2
    else:
        foreground = mask > 0
    return np.where(foreground, foreground_value, 0).astype(np.uint8)
