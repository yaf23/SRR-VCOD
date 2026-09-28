"""Dataset and pipeline registries."""

import random
from functools import partial

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from mmcv.utils.parrots_wrapper import DataLoader, PoolDataLoader
from torch.utils.data import DistributedSampler


DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def build_dataset(cfg, default_args=None):
    """Build a direct VCOD dataset, optionally wrapped for repetition."""
    if cfg['type'] == 'RepeatDataset':
        from .dataset_wrappers import RepeatDataset
        return RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    return build_from_cfg(cfg, DATASETS, default_args)


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     dataloader_type='PoolDataLoader',
                     **kwargs):
    """Build a deterministic MMCV-compatible PyTorch data loader."""
    rank, world_size = get_dist_info()
    if dist:
        sampler = DistributedSampler(
            dataset,
            world_size,
            rank,
            shuffle=shuffle,
            seed=0 if seed is None else seed)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    worker_init = (partial(
        worker_init_fn,
        num_workers=num_workers,
        rank=rank,
        seed=seed) if seed is not None else None)
    loader_types = {
        'DataLoader': DataLoader,
        'PoolDataLoader': PoolDataLoader,
    }
    if dataloader_type not in loader_types:
        raise ValueError(f'Unsupported dataloader: {dataloader_type}')

    return loader_types[dataloader_type](
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=worker_init,
        drop_last=drop_last,
        **kwargs)


def worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
