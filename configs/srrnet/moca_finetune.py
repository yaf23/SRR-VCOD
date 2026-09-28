"""MoCA-Mask fine-tuning recipe for SRRNet."""

import os


data_root = os.getenv('SRR_DATA_ROOT', 'data')
moca_root = os.getenv(
    'MOCA_MASK_ROOT', os.path.join(data_root, 'MoCA_Video'))
cad_root = os.getenv(
    'CAD_ROOT', os.path.join(data_root, 'CamouflagedAnimalDataset'))
pretrain_checkpoint = os.getenv(
    'SRR_COD10K_CHECKPOINT',
    'checkpoints/srrnet_cod10k_iter_50000.pth')

image_norm = dict(
    mean=[112.442, 119.932, 107.157],
    std=[44.471, 43.250, 43.346],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadVCODFrames'),
    dict(type='LoadAnnotations', reduce_zero_label=False, binary_mask=True),
    dict(type='Resize', img_scale=(2048, 512)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortionVCODFrames'),
    dict(type='NormalizeVCODFrames', **image_norm),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadVCODFrames'),
    dict(
        type='SingleScaleTestPipeline',
        img_scale=(2048, 512),
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='NormalizeVCODFrames', **image_norm),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='MoCAMaskDataset',
        data_root=moca_root,
        split='train',
        pipeline=train_pipeline),
    val=dict(
        type='CADDataset',
        data_root=cad_root,
        paper_protocol=True,
        skip_empty_gt=True,
        pipeline=test_pipeline),
    test=dict(
        type='MoCAMaskDataset',
        data_root=moca_root,
        split='test',
        pipeline=test_pipeline))

norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='SRRNet',
    pretrained=None,
    backbone=dict(type='SRRBackbone', architecture='RMATransformerLarge'),
    decode_head=dict(
        type='SRRHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=1024),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optimizer = dict(
    type='AdamW',
    lr=1e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
            'head': dict(lr_mult=10.0),
        }))
optimizer_config = dict(grad_clip=dict(max_norm=5.0, norm_type=2))
lr_config = dict(policy='fixed')

runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=2000, metric='Fwbeta')
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook', by_epoch=False)])

device = 'cuda'
dist_params = dict(backend='nccl')
inference_fp16 = True
log_level = 'INFO'
load_from = pretrain_checkpoint
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
