_base_ = [
    '../_base_/models/retinanet_emo_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

model = dict(
    backbone=dict(
        type='EMO',
        depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 288], exp_ratios=[2., 3., 4., 4.],
        norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
        dw_kss=[3, 3, 5, 5], dim_heads=[24, 24, 32, 32], window_sizes=[10, 10, 10, 10], attn_ss=[False, False, True, True],
        qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False,
        attn_pre=True, pre_dim=0, sync_bn=True, out_indices=(1, 2, 3, 4),
        pretrained='../../resources/EMO_5M/net.pth',
        frozen_stages=1, norm_eval=True),
    neck=dict(in_channels=[48, 72, 160, 288],)
)

bs_ratio = 1
optimizer = dict(type='AdamW', lr=0.0002 * bs_ratio, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr=0)

data = dict(
    samples_per_gpu=4 * bs_ratio,
    workers_per_gpu=4 * min(bs_ratio, 2)
)

