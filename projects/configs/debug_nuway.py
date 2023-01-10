_base_ = [
    'mmdet3d::_base_/default_runtime.py',
    './base/detr3d_nuway.py'
]
# this means type='DETR3D' will be processed as 'mmdet3d.DETR3D'
default_scope = 'mmdet3d'
custom_imports = dict(imports=['projects.detr3d'])
point_cloud_range = [-35, -75, -2, 75, 75, 4]
nus_class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
# nus_class_names = [
#     'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
#     'motorcycle', 'bicycle', 'pedestrian'
# ]

nusc_test_transforms = [
    dict(type='RandomResize3D',
        #  scale=(1600, 928),
         scale=(1600, 900),
         ratio_range=(1., 1.),
         keep_ratio=False)
]

nusc_train_transforms = [dict(type='PhotoMetricDistortion3D')] + nusc_test_transforms

file_client_args = dict(backend='disk')
nusc_train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=6),
    dict(type='filename2img_path'),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True,
         with_label_3d=True,
         with_attr_label=False),
    dict(type='ProjectLabelToWaymoClass', class_names = nus_class_names),
    dict(type='MultiViewWrapper', transforms=nusc_train_transforms),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=nus_class_names),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

nusc_test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=6),
    dict(type='filename2img_path'),  # fix it in ↑ via a PR
    dict(type='MultiViewWrapper', transforms=nusc_test_transforms),
    dict(type='Pack3DDetInputs', keys=['img'])
]


waymo_test_transforms = [
    dict(type='RandomResize3D',
         scale=(1600,1066),
         ratio_range=(1., 1.),
         keep_ratio=False)
]
waymo_test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=5),
    dict(type='filename2img_path'),  # fix it in ↑ via a PR
    dict(type='MultiViewWrapper', transforms=waymo_test_transforms),
    dict(type='Pack3DDetInputs', keys=['img'])
]

nusc_data_prefix = dict(pts='samples/LIDAR_TOP',
                   sweeps='sweeps/LIDAR_TOP',
                   CAM_FRONT='samples/CAM_FRONT',
                   CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
                   CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
                   CAM_BACK='samples/CAM_BACK',
                   CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
                   CAM_BACK_LEFT='samples/CAM_BACK_LEFT')

waymo_data_prefix = dict(
    pts='training/velodyne',
    sweeps='training/velodyne',
    CAM_FRONT='training/image_0',
    CAM_FRONT_RIGHT='training/image_1',
    CAM_FRONT_LEFT='training/image_2',
    CAM_SIDE_RIGHT='training/image_3',
    CAM_SIDE_LEFT='training/image_4',
)

nusc_dataset_type = 'NuScenesDataset'
nusc_data_root = 'data/nus_v2/'
input_modality = dict(use_lidar=True,
                      use_camera=True)
metainfo = dict(classes=nus_class_names)
# this is absolutely needed to do label projection
nusc_train_dataset = dict(type=nusc_dataset_type,
                          data_root=nusc_data_root,
                          ann_file='debug_val.pkl',
                          pipeline=nusc_train_pipeline,
                          load_type='frame_based',
                          modality=input_modality,
                          metainfo=metainfo,
                          test_mode=False,
                          data_prefix=nusc_data_prefix,
                          box_type_3d='LiDAR')

nusc_val_dataset = dict(type=nusc_dataset_type,
                        data_root=nusc_data_root,
                        ann_file='debug_val.pkl',
                        load_type='frame_based',
                        pipeline=nusc_test_pipeline,
                        modality=input_modality,
                        metainfo=metainfo,
                        test_mode=True,
                        data_prefix=nusc_data_prefix,
                        box_type_3d='LiDAR')

waymo_val_dataset = dict(type='WaymoDataset',
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

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=nusc_train_dataset)

val_dataloader = dict(batch_size=1,
                      num_workers=4,
                      persistent_workers=False,
                      drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=nusc_val_dataset,
                    #   dataset=waymo_val_dataset,
                    )
test_dataloader = val_dataloader
val_evaluator = dict(type = 'CustomWaymoMetric',is_waymo_gt = False, is_waymo_pred = True)
# dict(type='NuScenesMetric',
#                      data_root=data_root,
#                      ann_file=data_root + 'nuscenes_infos_val.pkl',
#                      metric='bbox')
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
                 val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(checkpoint=dict(
    type='CheckpointHook', interval=1, max_keep_ckpts=1, save_last=True))
load_from = 'ckpts/fcos3d_yue.pth'

vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')