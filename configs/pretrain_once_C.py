dataset_type = 'OnceImageDataset'
data_root = './data/once'
img_scale = (1920, 1020)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GenerateMask', input_size=14, mask_ratio=0.75),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'mask'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GenerateMask', input_size=224, mask_ratio=0.75),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'mask'])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/pretrain_img_infos.pkl',
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/pretrain_img_infos.pkl',
        pipeline=test_pipeline,
        test_mode=False,),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/pretrain_img_infos.pkl',
        pipeline=test_pipeline,
        test_mode=True,))

model = dict(
    type='PretrainVisionTransformer',
    img_size=224, 
    patch_size=16, 
    encoder_in_chans=3, 
    encoder_num_classes=0, 
    encoder_embed_dim=768, 
    encoder_depth=12,
    encoder_num_heads=12, 
    decoder_num_classes=768, 
    decoder_embed_dim=512, 
    decoder_depth=8,
    decoder_num_heads=8, 
    mlp_ratio=4., 
    qkv_bias=False, 
    qk_scale=None, 
    drop_rate=0., 
    attn_drop_rate=0.,
    drop_path_rate=0., 
    init_values=0.,
    use_learnable_pos_emb=False,
    mask_ratio=0.75,
    normalize_target=True,
    train_cfg={},
    test_cfg={}
    )
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)  # for 8gpu * 2sample_per_gpu
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
total_epochs = 15
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None #'./work_dirs/stransfusion_once_voxel_LC/latest.pth'
workflow = [('train', 1)]
gpu_ids = range(3, 5)
