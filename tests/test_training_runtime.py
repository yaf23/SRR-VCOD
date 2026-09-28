from unittest import mock

from mmcv import Config
from torch.utils.data import Dataset

import mmseg.apis.train as train_api
import mmseg.datasets.builder as dataset_builder


class _Dataset(Dataset):

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return index


def test_training_uses_fp32_hook_and_strict_checkpoint_loading():
    cfg = Config(dict(
        log_level='INFO',
        runner=dict(type='IterBasedRunner', max_iters=1),
        optimizer=dict(type='SGD', lr=0.1),
        optimizer_config=dict(
            grad_clip=dict(max_norm=5.0, norm_type=2)),
        inference_fp16=True,
        lr_config=dict(policy='fixed'),
        checkpoint_config=None,
        log_config=dict(interval=1, hooks=[]),
        momentum_config=None,
        work_dir='.',
        resume_from=None,
        load_from='full_model.pth',
        auto_resume=False,
        workflow=[('train', 1)]))
    runner = mock.Mock()

    with mock.patch.object(
            train_api, 'get_root_logger', return_value=mock.Mock()), \
            mock.patch.object(
                train_api, '_build_train_loaders', return_value=[object()]), \
            mock.patch.object(
                train_api, '_wrap_model', return_value=object()), \
            mock.patch.object(
                train_api, 'build_optimizer', return_value=object()), \
            mock.patch.object(
                train_api, 'build_runner', return_value=runner):
        train_api.train_segmentor(
            object(), object(), cfg, distributed=True, validate=False)

    optimizer_hook = runner.register_training_hooks.call_args.args[1]
    assert optimizer_hook == cfg.optimizer_config
    runner.load_checkpoint.assert_called_once_with(
        'full_model.pth', strict=True)


def test_distributed_sampler_uses_dataloader_seed():
    with mock.patch.object(
            dataset_builder, 'get_dist_info', return_value=(1, 2)):
        data_loader = dataset_builder.build_dataloader(
            _Dataset(),
            samples_per_gpu=1,
            workers_per_gpu=0,
            dist=True,
            seed=1379519528,
            dataloader_type='DataLoader')

    assert data_loader.sampler.seed == 1379519528
