_base_ = [
    # '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    # '/home/zhenglt/mmdev11/mmdet3d-latest/configs/_base_/datasets/nus-3d.py',
    'mmdet3d::_base_/default_runtime.py',
    './base/detr3d_nusc.py'
]
# # Resize3D
custom_imports = dict(imports=['projects.detr3d'])
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(use_lidar=True,
                      use_camera=True,
                      use_radar=False,
                      use_map=False,
                      use_external=False)
# this means type='DETR3D' will be processed as 'mmdet3d.DETR3D'
default_scope = 'mmdet3d'

dataset_type = 'CustomNusc'
data_root = 'data/nus_v2/'

test_transforms = [
    dict(type='RandomResize3D',
         scale=(1600, 928),
         ratio_range=(1., 1.),
         keep_ratio=False)
]
train_transforms = test_transforms

file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=6),
    dict(type='filename2img_path'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True,
         with_label_3d=True,
         with_attr_label=False),
    dict(type='MultiViewWrapper', transforms=train_transforms),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

# nusc_pipeline_default = [
#     dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=nusc_num_views),
#     dict(type='filename2img_path'),
#     dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
#     dict(type='ObjectNameFilter', classes=nusc_class_names),
#     dict(type='RotateScene_neg90'),
#     dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
#     dict(type='ProjectLabelToWaymoClass', class_names = nusc_class_names),
# ]
# nusc_train_pipeline = nusc_pipeline_default + [
#     dict(type='MultiViewWrapper', transforms=[dict(type='PhotoMetricDistortion3D')] + nusc_test_transforms),
#     dict(type='Pack3DDetInputsExtra', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
# ]
# nusc_test_pipeline = [dict(type='evalann2ann')] + nusc_pipeline_default + [
#     dict(type='MultiViewWrapper', transforms=nusc_test_transforms),
#     dict(type='Pack3DDetInputsExtra', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
# ]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=6),
    dict(type='filename2img_path'),  # fix it in ↑ via a PR
    dict(type='evalann2ann'), 
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ProjectLabelToWaymoClass', class_names = class_names),
    dict(type='MultiViewWrapper', transforms=test_transforms),
    dict(type='Pack3DDetInputsExtra', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

metainfo = dict(classes=class_names)
data_prefix = dict(pts='samples/LIDAR_TOP',
                   sweeps='sweeps/LIDAR_TOP',
                   CAM_FRONT='samples/CAM_FRONT',
                   CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
                   CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
                   CAM_BACK='samples/CAM_BACK',
                   CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
                   CAM_BACK_LEFT='samples/CAM_BACK_LEFT')

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
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

val_dataloader = dict(batch_size=1,
                      num_workers=4,
                      persistent_workers=True,
                      drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(type=dataset_type,
                                   data_root=data_root,
                                   ann_file='nuscenes_infos_val.pkl',
                                   load_type='frame_based',
                                   pipeline=test_pipeline,
                                   metainfo=metainfo,
                                   modality=input_modality,
                                   load_interval=10000,
                                   test_mode=True,
                                   data_prefix=data_prefix,
                                   box_type_3d='LiDAR'))

test_dataloader = val_dataloader
val_evaluator = dict(type = 'CustomWaymoMetric', is_waymo_pred = False, metainfo = class_names)
# val_evaluator = dict(type='NuScenesMetric',
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
# checkpoint_config = dict(interval=1, max_keep_ckpts=1)
default_hooks = dict(checkpoint=dict(
    type='CheckpointHook', interval=1, max_keep_ckpts=1, save_last=True))
load_from = 'ckpts/fcos3d_yue.pth'

vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')
