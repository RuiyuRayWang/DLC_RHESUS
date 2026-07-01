_base_ = [
    'mmyolo::_base_/default_runtime.py',
]

# model settings
model = dict(
    type='YOLODetector',
    backbone=dict(
        type='EfficientRep',
        depth_multiple=0.67,
        width_multiple=0.75,
        channels_list=[64, 128, 256, 512, 1024],
        num_csp_blocks=2,
    ),
    neck=dict(
        type='RepPANNeck',
        depth_multiple=0.67,
        width_multiple=0.75,
        in_channels=[256, 512, 1024],
        out_channels=[128, 256, 512],
        num_csp_blocks=2,
    ),
    head=dict(
        type='RTMDetHead',
        num_classes=1,
        in_channels=[128, 256, 512],
        feat_channels=256,
        stacked_convs=2,
        share_conv=True,
        pred_kernel_size=1,
        weight_init='gaussian',
        act_cfg=dict(type='SiLU'),
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_dfl=dict(
            type='DistributionFocalLoss',
            loss_weight=0.25),
    ),
)

# data settings
data_root = 'data/rtmdet_dataset/'  # Modify this path
dataset_type = 'CocoDataset'
classes = ('monkey',)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(640, 640),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(
        type='Resize',
        scale=(640, 640),
        pad_val=dict(img=114),
        keep_ratio=True),
    dict(
        type='Pad',
        pad_val=dict(img=(114, 114, 114))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='Resize',
        scale=(640, 640),
        pad_val=dict(img=114),
        keep_ratio=True),
    dict(
        type='Pad',
        pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape'))
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations_train.json',
        img_prefix='images',
        pipeline=train_pipeline,
        metainfo=dict(classes=classes),
        cache_on_disk=False))

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations_val.json',
        img_prefix='images',
        pipeline=val_pipeline,
        metainfo=dict(classes=classes),
        cache_on_disk=False))

test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.937, weight_decay=0.0005),
    clip_grad=dict(max_norm=35, norm_type=2))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        T_max=450,
        eta_min_ratio=0.05,
        by_epoch=False,
        begin=500,
        end=950)
]

# training config
train_cfg = dict(max_epochs=100, val_interval=5)
val_cfg = dict()
test_cfg = dict()

# hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# evaluation
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations_val.json',
    metric=['bbox'],
    format_only=False)

test_evaluator = val_evaluator

# Others
default_scope = 'mmyolo'
