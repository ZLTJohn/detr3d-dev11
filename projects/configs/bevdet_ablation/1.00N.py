_base_ = [
          'mmdet3d::_base_/default_runtime.py',]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 4.0]
# For nuScenes we usually do 10-class detection
waymo_class_names = ['Car', 'Pedestrian', 'Cyclist']
nusc_class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
custom_imports = dict(imports=['projects.bevdet'])
default_scope = 'mmdet3d'
waymo_type = 'CustomWaymo'
waymo_root = 'data/waymo_dev1x/kitti_format'
waymo_train_pkl = 'waymo_infos_train_2Hz_mono_front.pkl'
waymo_train_interval = 1    # 2Hz_part means interval = 5x3
waymo_val_pkl = 'waymo_infos_val_2Hz_part_mono_front.pkl'
waymo_val_interval = 1
waymo_h = 640
waymo_w = 960
waymo_h_crop = 436
waymo_w_crop = 960
pad_h = 512
pad_w = 1312
nusc_type = 'CustomNusc'
nusc_root = 'data/nus_v2/'
nusc_train_pkl = 'nuscenes_infos_train_mono_front.pkl'
nusc_val_pkl = 'nuscenes_infos_val_part_mono_front.pkl'
nusc_h = 450
nusc_w = 800
nusc_h_crop = 307
nusc_w_crop = 800
work_dir = 'work_dirs_bevdet_mono_ablate/1.00N' 
# Model
# lss: h w
# resize: w h
# crop: h w
# pad: h w
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 4, 9],
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
                     vis_count=5000,
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
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(#differs
        type='LSSFPN',
        in_channels=1024 + 2048,
        out_channels=256,
        upsampling_scale_output=None,
        input_feat_indexes=(0, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        upsampling_scale=2,
        use_input_conv=True),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=(pad_h,pad_w),
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
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    img_bev_encoder_neck=dict(
        type='LSSFPN', 
        in_channels=64 * 8 + 64 * 2, 
        out_channels=256,
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    pts_bbox_head=dict(
        type='CustomCenterHead',
        in_channels=256,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        tasks=[
            dict(num_class=3, class_names=waymo_class_names),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=7),
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
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
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
# bev_aug = [dict(type='GlobalRotScaleTrans',
#                 rot_range=[0, 0.0],
#                 scale_ratio_range=[1., 1.0],
#                 translation_std=[0, 0, 0]),
#            dict(type='RandomFlip3D',
#                 sync_2d=False,
#                 flip_ratio_bev_horizontal=0.,
#                 flip_ratio_bev_vertical=0.)]
bevdet_input_default = [
    dict(type='filename2img_path'),
    dict(type='GetBEVDetInputs'),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputsBEVDet', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]

# The order of image-view augmentation should be
# resize -> crop -> pad -> flip -> rotate
waymo_img_no_aug = [dict(type='MultiViewWrapper',
    transforms=[
        dict(type='RandomResize', ratio_range=(1.0, 1.0), scale=(waymo_w,waymo_h)),
        dict(type='RangeLimitedRandomCrop',
            relative_x_offset_range=(1.0, 1.0),
            relative_y_offset_range=(1.0, 1.0),
            crop_size=(waymo_h_crop, waymo_w_crop)),
        dict(type='Pad', size=(pad_w, pad_h)),
        dict(type='RandomFlip', prob=0.),
        dict(type='RandomRotate', range=(0.,0.), 
            img_border_value=0, level=1, prob=0.0)],
    collected_keys=['scale_factor', 'crop', 'pad_shape', 'flip', 'rotate'])]
waymo_input_default = [
    dict(type='LoadPointsFromFile',coord_type='LIDAR', load_dim=6, use_dim=5),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    # dict(type='ProjectLabelToWaymoClass', class_names = nusc_class_names),
]
# To avoid 'flip' information conflict between RandomFlip and RandomFlip3D,
# 3D space augmentation should be conducted before loading images and
# conducting image-view space augmentation.
waymo_train_pipeline = [
    *waymo_input_default,
    # *bev_aug,
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    *waymo_img_no_aug,
    *bevdet_input_default
]

waymo_test_pipeline = [
    dict(type='evalann2ann'),
    *waymo_input_default,
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    *waymo_img_no_aug,
    *bevdet_input_default
    ]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

waymo_metainfo = dict(classes=waymo_class_names)
waymo_data_prefix = dict(
    pts='training/velodyne',
    sweeps='training/velodyne',
    CAM_FRONT='training/image_0',
    CAM_FRONT_LEFT='training/image_1',
    CAM_FRONT_RIGHT='training/image_2',
    CAM_SIDE_LEFT='training/image_3',
    CAM_SIDE_RIGHT='training/image_4',)
waymo_default = dict(
    load_type='frame_based',
    modality=input_modality,
    data_prefix=waymo_data_prefix,
    cam_sync_instances=True,
    box_type_3d='LiDAR')
waymo_train =dict(type=waymo_type,
                  data_root=waymo_root,
                  ann_file=waymo_train_pkl,
                  pipeline=waymo_train_pipeline,
                  load_interval= waymo_train_interval,
                  test_mode=False,
                  **waymo_default)
waymo_test = dict(type=waymo_type,
                 data_root=waymo_root,
                 ann_file=waymo_val_pkl,
                 pipeline=waymo_test_pipeline,
                 load_interval=waymo_val_interval,
                 test_mode=True,
                 **waymo_default)

nusc_img_no_aug = [dict(type='MultiViewWrapper',
    transforms=[
        dict(type='RandomResize', ratio_range=(1.0, 1.0), scale=(nusc_w,nusc_h)),
        dict(type='RangeLimitedRandomCrop',
            relative_x_offset_range=(1.0, 1.0),
            relative_y_offset_range=(1.0, 1.0),
            crop_size=(nusc_h_crop, nusc_w_crop)),
        dict(type='Pad', size=(pad_w, pad_h)),
        dict(type='RandomFlip', prob=0.),
        dict(type='RandomRotate', range=(0.,0.), 
            img_border_value=0, level=1, prob=0.0)],
    collected_keys=['scale_factor', 'crop', 'pad_shape', 'flip', 'rotate'])]
nusc_input_default = [
    dict(type='LoadPointsFromFile',coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ProjectLabelToWaymoClass', class_names = nusc_class_names),
]
nusc_train_pipeline = [
    *nusc_input_default,
    # *bev_aug,
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    *nusc_img_no_aug,
    *bevdet_input_default
]
nusc_test_pipeline = [
    dict(type='evalann2ann'),
    *nusc_input_default,
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    *nusc_img_no_aug,
    *bevdet_input_default
    ]
nusc_metainfo = dict(classes=nusc_class_names)
nusc_data_prefix = dict(pts='samples/LIDAR_TOP',
                   sweeps='sweeps/LIDAR_TOP',
                   CAM_FRONT='samples/CAM_FRONT',
                   CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
                   CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
                   CAM_BACK='samples/CAM_BACK',
                   CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
                   CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',)
nusc_train = dict(
        type=nusc_type,
        data_root=nusc_root,
        ann_file=nusc_train_pkl,
        pipeline=nusc_train_pipeline,
        load_type='frame_based',
        metainfo=nusc_metainfo,
        modality=input_modality,
        test_mode=False,
        data_prefix=nusc_data_prefix,
        with_velocity=False,
        box_type_3d='LiDAR')
nusc_test = dict(
        type=nusc_type,
        data_root=nusc_root,
        ann_file=nusc_val_pkl,
        load_type='frame_based',
        pipeline=nusc_test_pipeline,
        metainfo=nusc_metainfo,
        modality=input_modality,
        test_mode=True,
        data_prefix=nusc_data_prefix,
        with_velocity=False,
        box_type_3d='LiDAR')

joint_train = dict(
        type='CustomConcatDataset',
        datasets=[nusc_train])
joint_val = dict(
        type='CustomConcatDataset',
        datasets=[nusc_test, waymo_test])
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=joint_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=joint_val)

test_dataloader = val_dataloader

# val_evaluator = dict(type='CustomNuscMetric',
#                      data_root=data_root,
#                      ann_file=data_root + 'nuscenes_infos_val.pkl',
#                      metric='bbox')
val_evaluator = dict(type = 'JointMetric_bevdet',
                     per_location = True,
                     work_dir = work_dir,
                     brief_split = True)
# val_evaluator = dict(type='CustomWaymoMetric',
#                      work_dir = work_dir)

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
                 val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(checkpoint=dict(
    type='CheckpointHook', interval=1, max_keep_ckpts=24, save_last=True, save_best='all_dataset/OBJECT_TYPE_ALL_NS_LEVEL_2/LET-mAP'))

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
