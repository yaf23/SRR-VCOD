"""Dataset wrappers used by the released pre-training recipe."""

from .builder import DATASETS


@DATASETS.register_module()
class RepeatDataset:
    """Repeat a dataset without duplicating its index in memory."""

    def __init__(self, dataset, times):
        if times < 1:
            raise ValueError('Repeat count must be positive')
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        self.PALETTE = dataset.PALETTE
        self._original_length = len(dataset)

    def __getitem__(self, index):
        return self.dataset[index % self._original_length]

    def __len__(self):
        return self.times * self._original_length
