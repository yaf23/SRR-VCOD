"""Train SRRNet with an MMCV configuration file."""

import argparse
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train SRRNet')
    parser.add_argument('config', help='Path to a training configuration')
    parser.add_argument('--work-dir', help='Directory for logs and checkpoints')
    parser.add_argument('--load-from', help='Checkpoint used to initialize weights')
    parser.add_argument('--resume-from', help='Checkpoint used to resume a run')
    parser.add_argument(
        '--no-validate', action='store_true', help='Disable validation')

    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        '--gpus', type=int, default=1,
        help='Number of GPUs for non-distributed training')
    gpu_group.add_argument(
        '--gpu-ids', type=int, nargs='+',
        help='GPU IDs for non-distributed training')

    parser.add_argument('--seed', type=int, help='Random seed')
    parser.set_defaults(deterministic=True)
    parser.add_argument(
        '--deterministic', dest='deterministic', action='store_true',
        help='Use deterministic CuDNN operations (default)')
    parser.add_argument(
        '--non-deterministic', dest='deterministic', action='store_false',
        help='Allow non-deterministic CuDNN operations')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='Override config values, for example key=value')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none', help='Distributed job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    return args


def _resolve_work_dir(cfg, config_path, cli_work_dir):
    if cli_work_dir:
        return cli_work_dir
    if cfg.get('work_dir'):
        return cfg.work_dir
    config_name = osp.splitext(osp.basename(config_path))[0]
    return osp.join('work_dirs', config_name)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = _resolve_work_dir(cfg, args.config, args.work_dir)
    if args.load_from:
        cfg.load_from = args.load_from
    if args.resume_from:
        cfg.resume_from = args.resume_from
    cfg.gpu_ids = (args.gpu_ids if args.gpu_ids is not None
                   else range(args.gpus))

    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    environment = collect_env()
    environment_text = '\n'.join(f'{key}: {value}'
                                 for key, value in environment.items())
    logger.info('Environment info:\n%s', environment_text)
    logger.info('Distributed training: %s', distributed)
    logger.info('Config:\n%s', cfg.pretty_text)

    seed = init_random_seed(args.seed, device=cfg.get('device', 'cuda'))
    cfg.seed = seed
    set_random_seed(seed, deterministic=args.deterministic)
    logger.info('Random seed: %d, deterministic: %s',
                seed, args.deterministic)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    model.CLASSES = datasets[0].CLASSES

    metadata = dict(
        env_info=environment_text,
        seed=seed,
        exp_name=osp.basename(args.config))
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=not args.no_validate,
        timestamp=timestamp,
        meta=metadata)


if __name__ == '__main__':
    main()
