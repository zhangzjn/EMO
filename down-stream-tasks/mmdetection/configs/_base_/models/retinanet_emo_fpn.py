# model settings
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='MetaMobile',
        depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 288], exp_ratios=[2., 3., 4., 4.],
        norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
        dw_kss=[3, 3, 5, 5], dim_heads=[24, 24, 32, 32], window_sizes=[10, 10, 10, 10], attn_ss=[False, False, True, True],
        qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False,
        attn_pre=True, pre_dim=0, sync_bn=True, out_indices=(1, 2, 3, 4),
        pretrained=None,
        frozen_stages=1, norm_eval=True),
    neck=dict(
        type='FPN',
        in_channels=[48, 72, 160, 288],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
