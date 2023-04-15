# Copyright (c) Phigent Robotics. All rights reserved.
# circle NMS
# mAP: 0.2439                                                                                                             
# mATE: 0.8052
# mASE: 0.2915
# mAOE: 0.7103
# mAVE: 1.0257
# mAAE: 0.2811
# NDS: 0.3132
# Eval time: 75.8s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.428   0.659   0.172   0.131   1.209   0.270
# truck   0.196   0.838   0.244   0.161   1.053   0.285
# bus     0.258   0.885   0.226   0.220   2.172   0.416
# trailer 0.062   1.157   0.251   0.755   0.645   0.076
# construction_vehicle    0.055   0.983   0.491   1.244   0.102   0.386
# pedestrian      0.245   0.780   0.304   1.280   0.926   0.514
# motorcycle      0.236   0.731   0.277   1.024   1.610   0.247
# bicycle 0.205   0.761   0.299   1.425   0.488   0.056
# traffic_cone    0.373   0.603   0.354   nan     nan     nan
# barrier 0.381   0.657   0.296   0.152   nan     nan

# rotate NMS
# mAP: 0.2265
# mATE: 0.8286
# mASE: 0.2911
# mAOE: 0.6957
# mAVE: 0.9823
# mAAE: 0.2731
# NDS: 0.3062
# Eval time: 78.7s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.410   0.649   0.172   0.123   1.203   0.273
# truck   0.183   0.824   0.242   0.148   1.053   0.288
# bus     0.231   0.881   0.222   0.205   2.102   0.403
# trailer 0.054   1.162   0.260   0.776   0.664   0.066
# construction_vehicle    0.043   0.971   0.487   1.202   0.101   0.358
# pedestrian      0.252   0.871   0.304   1.249   0.809   0.512
# motorcycle      0.204   0.766   0.274   1.031   1.476   0.234
# bicycle 0.174   0.791   0.294   1.361   0.450   0.051
# traffic_cone    0.362   0.670   0.357   nan     nan     nan
# barrier 0.352   0.700   0.300   0.166   nan     nan
_base_ = ['mmdet3d::_base_/datasets/nus-3d.py', 
          'mmdet3d::_base_/default_runtime.py',]
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
custom_imports = dict(imports=['projects.bevdet'])
default_scope = 'mmdet3d'
# Model
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 1.0],
}

voxel_size = [0.1, 0.1, 0.2]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    bgr_to_rgb=True)
debug_vis_cfg = dict(debug_dir='debug/visualization',
                     gt_range=[0, 105],
                     pc_range=point_cloud_range,
                     vis_count=2000,
                     debug_name='bevdet')
model = dict(
    type='BEVDet',
    # debug_vis_cfg=debug_vis_cfg, 
    data_preprocessor=dict(type='Det3DDataPreprocessor',
                        **img_norm_cfg,
                        pad_size_divisor=32),
    img_backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(#differs
        type='LSSFPN',
        in_channels=1024 + 2048,
        out_channels=256,
        upsampling_scale_output=None,
        input_feat_indexes=(0, 1),
        upsampling_scale=2,
        use_input_conv=True),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=(256, 704),
        downsample=16,
        in_channels=256,
        out_channels=64,
        accelerate=False),
    img_bev_encoder_backbone=dict(# differs
        type='CustomResNet',
        depth=18,
        num_stages=3,
        stem_channels=64,
        base_channels=128,
        out_indices=(0, 1, 2),
        strides=(2, 2, 2),
        dilations=(1, 1, 1),
        frozen_stages=-1,
        with_cp=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    img_bev_encoder_neck=dict(
        type='LSSFPN', in_channels=64 * 8 + 64 * 2, out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=10, class_names=['car', 'truck',
                                            'construction_vehicle',
                                            'bus', 'trailer',
                                            'barrier',
                                            'motorcycle', 'bicycle',
                                            'pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            # pc_range=[-75.2, -75.2],
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    test_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            # pc_range=[-75.2, -75.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)))

# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nus_v2/'
nusc_scale = (704, 396)
crop_wh=(704, 256)
crop_hw=(256, 704)

train_pipeline = [
    # To avoid 'flip' information conflict between RandomFlip and RandomFlip3D,
    # 3D space augmentation should be conducted before loading images and
    # conducting image-view space augmentation.
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),#update_img2lidar
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),#update_img2lidar
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='filename2img_path'),
    # The order of image-view augmentation should be
    # resize -> crop -> pad -> flip -> rotate
    dict(
        type='MultiViewWrapper',
        transforms=[
            dict(
                type='RandomResize', ratio_range=(0.864, 1.25), #just did not change cam2img
                scale=nusc_scale),
            dict(
                type='RangeLimitedRandomCrop',
                relative_x_offset_range=(0.0, 1.0),
                relative_y_offset_range=(1.0, 1.0),
                crop_size=crop_hw),
            dict(type='Pad', size=crop_wh),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='RandomRotate',
                range=(-5.4, 5.4),
                img_border_value=0,
                level=1,
                prob=1.0),
            # dict(type='Normalize', **img_norm_cfg)
        ],
        collected_keys=['scale_factor', 'crop', 'pad_shape', 'flip',
                        'rotate']),
    dict(type='GetBEVDetInputs'),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='Pack3DDetInputsBEVDet', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='filename2img_path'),
    # The order of image-view augmentation should be
    # resize -> crop -> pad -> flip -> rotate
    dict(
        type='MultiViewWrapper',
        transforms=[
            dict(
                type='RandomResize',
                ratio_range=(1.091, 1.091),
                scale=nusc_scale),
            dict(
                type='RangeLimitedRandomCrop',
                relative_x_offset_range=(0.5, 0.5),
                relative_y_offset_range=(1.0, 1.0),
                crop_size=crop_hw),
            dict(type='Pad', size=crop_wh),
            dict(type='RandomFlip', prob=0.0),
            dict(
                type='RandomRotate',
                range=(-0.0, 0.0),
                img_border_value=0,
                level=1,
                prob=0.0),
            # dict(type='Normalize', **img_norm_cfg)
        ],
        collected_keys=['scale_factor', 'crop', 'pad_shape', 'flip',
                        'rotate']),
    dict(type='GetBEVDetInputs'),
    dict(type='Pack3DDetInputsBEVDet', keys=['img'])
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

metainfo = dict(classes=class_names)
data_prefix = dict(pts='samples/LIDAR_TOP',
                   sweeps='sweeps/LIDAR_TOP',
                   CAM_FRONT='samples/CAM_FRONT',
                   CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
                   CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
                   CAM_BACK='samples/CAM_BACK',
                   CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
                   CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',)
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        load_type='frame_based',
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        data_prefix=data_prefix,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'))

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type,
                data_root=data_root,
                ann_file='nuscenes_infos_val.pkl',
                load_type='frame_based',
                pipeline=test_pipeline,
                metainfo=metainfo,
                modality=input_modality,
                test_mode=True,
                data_prefix=data_prefix,
                box_type_3d='LiDAR'))
test_dataloader = val_dataloader

val_evaluator = dict(type='NuScenesMetric',
                     data_root=data_root,
                     ann_file=data_root + 'nuscenes_infos_val.pkl',
                     metric='bbox')
test_evaluator = val_evaluator
# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=1e-07),
    # paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
    clip_grad=dict(max_norm=5, norm_type=2),
)

total_epochs = 24

train_cfg = dict(type='EpochBasedTrainLoop',
                 max_epochs=total_epochs,
                 val_interval=12)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(checkpoint=dict(
    type='CheckpointHook', interval=1, max_keep_ckpts=1, save_last=True))

param_scheduler = [
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=200),
    dict(type='MultiStepLR',
         by_epoch=True,
         milestones=[19, 23],
         begin=0,
         end=24)
]
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')
