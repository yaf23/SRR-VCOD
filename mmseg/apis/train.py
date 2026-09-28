"""Training helpers for the vendored MMCV runner."""

import os
import random

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import build_runner, get_dist_info

from mmseg import digit_version
from mmseg.core import DistEvalHook, EvalHook, build_optimizer
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import (build_ddp, build_dp, find_latest_checkpoint,
                         get_root_logger)


def init_random_seed(seed=None, device='cuda'):
    """Return one random seed shared by every distributed worker."""
    if seed is not None:
        return seed

    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    value = torch.tensor(
        seed if rank == 0 else 0, dtype=torch.int32, device=device)
    dist.broadcast(value, src=0)
    return value.item()


def set_random_seed(seed, deterministic=False):
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _build_train_loaders(datasets, cfg, distributed):
    loader_cfg = dict(
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        drop_last=True)
    loader_cfg.update({
        key: value
        for key, value in cfg.data.items()
        if key not in {
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        }
    })
    loader_cfg.update(cfg.data.get('train_dataloader', {}))
    return [build_dataloader(dataset, **loader_cfg) for dataset in datasets]


def _wrap_model(model, cfg, distributed):
    if distributed:
        return build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=cfg.get('find_unused_parameters', False))

    if not torch.cuda.is_available():
        minimum = digit_version('1.4.4')
        assert digit_version(mmcv.__version__) >= minimum, (
            'CPU training requires MMCV >= 1.4.4')
    return build_dp(model, cfg.device, device_ids=cfg.gpu_ids)


def _register_validation(runner, cfg, distributed):
    validation_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    loader_cfg = dict(
        samples_per_gpu=1,
        workers_per_gpu=1,
        num_gpus=1,
        dist=False,
        shuffle=False,
        drop_last=False)
    loader_cfg.update(cfg.data.get('val_dataloader', {}))
    validation_loader = build_dataloader(validation_dataset, **loader_cfg)

    evaluation_cfg = dict(cfg.get('evaluation', {}))
    evaluation_cfg['by_epoch'] = False
    hook_class = DistEvalHook if distributed else EvalHook
    runner.register_hook(
        hook_class(validation_loader, **evaluation_cfg), priority='LOW')


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Train SRRNet from an MMCV config."""
    logger = get_root_logger(log_level=cfg.log_level)
    datasets = list(dataset) if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = _build_train_loaders(datasets, cfg, distributed)
    model = _wrap_model(model, cfg, distributed)

    if cfg.get('runner') is None:
        raise KeyError('The training config must define runner')
    if cfg.runner.get('type') != 'IterBasedRunner':
        raise ValueError('SRRNet recipes require an IterBasedRunner')

    optimizer = build_optimizer(model, cfg.optimizer)
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    runner.register_training_hooks(
        cfg.lr_config,
        cfg.optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config'))
    runner.timestamp = timestamp

    if validate:
        _register_validation(runner, cfg, distributed)

    if cfg.resume_from is None and cfg.get('auto_resume'):
        cfg.resume_from = find_latest_checkpoint(cfg.work_dir)
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from, strict=True)

    runner.run(data_loaders, cfg.workflow)
