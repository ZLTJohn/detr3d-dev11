_base_ = [
    'mmdet3d::_base_/default_runtime.py',
    './base/detr3d_nuway_caffe.py'
]

default_scope = 'mmdet3d'
custom_imports = dict(imports=['projects.detr3d'])
point_cloud_range = [-100, -100, -2, 75, 75, 4]
nusc_class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'motorcycle', 'bicycle', 'pedestrian'
]
waymo_class_names = [
    'Car', 'Pedestrian', 'Cyclist'
]

nusc_test_transforms = [
    dict(type='RandomResize3D',
        #  scale=(1600, 928),
         scale=(1600, 900),
         ratio_range=(1., 1.),
         keep_ratio=False)
]
nusc_train_transforms = [dict(type='PhotoMetricDistortion3D')] + nusc_test_transforms
nusc_train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=6),
    dict(type='filename2img_path'),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True,
         with_label_3d=True,
         with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=nusc_class_names),
    dict(type='ProjectLabelToWaymoClass', class_names = nusc_class_names),
    dict(type='MultiViewWrapper', transforms=nusc_train_transforms),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

waymo_test_transforms = [
    dict(type='RandomResize3D',
         scale=(1600,1066),
        #  scale=(1920, 1280),
         ratio_range=(1., 1.),
         keep_ratio=False)
]
waymo_train_transforms = [dict(type='PhotoMetricDistortion3D')] + waymo_test_transforms
waymo_train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles',
         to_float32=True,
         num_views=5),
    dict(type='filename2img_path'),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True,
         with_label_3d=True,
         with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=waymo_class_names),
    dict(type='MultiViewWrapper', transforms=waymo_train_transforms),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

waymo_test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=5),
    dict(type='filename2img_path'),  # fix it in ↑ via a PR
    dict(type='MultiViewWrapper', transforms=waymo_test_transforms),
    dict(type='Pack3DDetInputs', keys=['img'])
]
nusc_test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=6),
    dict(type='filename2img_path'),  # fix it in ↑ via a PR
    dict(type='MultiViewWrapper', transforms=nusc_test_transforms),
    dict(type='Pack3DDetInputs', keys=['img'])
]

nusc_data_prefix = dict(pts='samples/LIDAR_TOP',
                   sweeps='sweeps/LIDAR_TOP',
                   CAM_FRONT='samples/CAM_FRONT',
                   CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
                   CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
                   CAM_BACK='samples/CAM_BACK',
                   CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
                   CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',)
waymo_data_prefix = dict(
    pts='training/velodyne',
    sweeps='training/velodyne',
    CAM_FRONT='training/image_0',
    CAM_FRONT_LEFT='training/image_1',
    CAM_FRONT_RIGHT='training/image_2',
    CAM_SIDE_LEFT='training/image_3',
    CAM_SIDE_RIGHT='training/image_4',
)

nusc_type = 'NuScenesDataset'
nusc_data_root = 'data/nus_v2/'
waymo_type = 'WaymoDataset'
# waymo_data_root = 'data/waymo_dev1x/'
input_modality = dict(use_lidar=True,
                      use_camera=True)
file_client_args = dict(backend='disk')

nusc_train = dict(type=nusc_type,
                 data_root=nusc_data_root,
                 ann_file='nuscenes_infos_train.pkl',
                 pipeline=nusc_train_pipeline,
                 load_type='frame_based',
                 modality=input_modality,
                 metainfo=dict(classes=nusc_class_names),
                 test_mode=False,
                 with_velocity=False,
                 use_valid_flag=True,
                 data_prefix=nusc_data_prefix,
                 box_type_3d='LiDAR')
waymo_train =dict(type=waymo_type,
                  data_root='data/waymo_dev1x/kitti_format',
                  ann_file='waymo_infos_train.pkl',
                  pipeline=waymo_train_pipeline,
                  load_type='frame_based',
                  load_interval=5,
                  metainfo=dict(classes=waymo_class_names),
                  modality=input_modality,
                  test_mode=False,
                  data_prefix=waymo_data_prefix,
                  cam_sync_instances=True,
                  box_type_3d='LiDAR')

nusc_val = dict(type=nusc_type,
                data_root=nusc_data_root,
                ann_file='debug_val.pkl',
                load_type='frame_based',
                pipeline=nusc_test_pipeline,
                modality=input_modality,
                metainfo=dict(classes=nusc_class_names),
                test_mode=True,
                with_velocity=False,
                # use_valid_flag=True,
                data_prefix=nusc_data_prefix,
                box_type_3d='LiDAR')

waymo_val = dict(type='WaymoDataset',
                 data_root='data/waymo_dev1x/kitti_format',
                 ann_file='debug_val.pkl',
                #  load_interval=5,
                 load_type='frame_based',
                 pipeline=waymo_test_pipeline,
                 modality=input_modality,
                 test_mode=True,
                 data_prefix=waymo_data_prefix,
                 cam_sync_instances=True,
                 box_type_3d='LiDAR')
nuway_train = dict(
        type='CustomConcatDataset',
        datasets=[nusc_train, waymo_train])
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # dataset=waymo_train
    dataset=nuway_train
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    # dataset=nusc_val,
    dataset=waymo_val,
)
test_dataloader = val_dataloader
val_evaluator = dict(type = 'CustomWaymoMetric',is_waymo_gt = True, is_waymo_pred = True, metainfo=nusc_class_names)
test_evaluator = val_evaluator
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# learning policy
param_scheduler = [
    dict(type='LinearLR',
         start_factor=1.0 / 3,
         by_epoch=False,
         begin=0,
         end=500),
    dict(type='CosineAnnealingLR',
         by_epoch=True,
         begin=0,
         end=24,
         T_max=24,
         eta_min_ratio=1e-3)
]

total_epochs = 24
train_cfg = dict(type='EpochBasedTrainLoop',
                 max_epochs=total_epochs,
                 val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(checkpoint=dict(
    type='CheckpointHook', interval=1, max_keep_ckpts=1, save_last=True))
load_from = 'ckpts/fcos3d_yue.pth'
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')