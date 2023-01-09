_base_ = [
    # 'mmdet3d::_base_/datasets/nus-3d.py',
    'mmdet3d::_base_/default_runtime.py'
]

custom_imports = dict(imports=['projects.detr3d'])
# this means type='DETR3D' will be processed as 'mmdet3d.DETR3D'
default_scope = 'mmdet3d'
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(mean=[103.530, 116.280, 123.675],
                    std=[1.0, 1.0, 1.0],
                    bgr_to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(use_lidar=False,
                      use_camera=True,
                      use_radar=False,
                      use_map=False,
                      use_external=False)

dataset_type = 'NuScenesDataset'
data_root = 'data/nus_v2/'

test_transforms = [
    dict(type='RandomResize3D',
         scale=(1600, 900),
         ratio_range=(1., 1.),
         keep_ratio=True)
]
train_transforms = [dict(type='PhotoMetricDistortion3D')] + test_transforms

file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=6),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True,
         with_label_3d=True,
         with_attr_label=False),
    dict(type='MultiViewWrapper', transforms=train_transforms),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=6),
    dict(type='MultiViewWrapper', transforms=test_transforms),
    dict(type='Pack3DDetInputs', keys=['img'])
]

metainfo = dict(classes=class_names)
data_prefix = dict(pts='',
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
                                   test_mode=True,
                                   data_prefix=data_prefix,
                                   box_type_3d='LiDAR'))

test_dataloader = val_dataloader

val_evaluator = dict(type='NuScenesMetric',
                     data_root=data_root,
                     ann_file=data_root + 'nuscenes_infos_val.pkl',
                     metric='bbox')
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

# setuptools 65 downgrades to 58.
# In mmlab-node we use setuptools 61 but occurs NO errors
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')

# before fixing h,w bug in feature-sampling
# mAP: 0.3450
# mATE: 0.7740
# mASE: 0.2675
# mAOE: 0.3960
# mAVE: 0.8737
# mAAE: 0.2156
# NDS: 0.4198
# Eval time: 161.5s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.534   0.565   0.152   0.071   0.907   0.214
# truck   0.285   0.839   0.213   0.114   0.984   0.229
# bus     0.346   0.924   0.199   0.117   2.060   0.379
# trailer 0.166   1.108   0.230   0.551   0.734   0.126
# construction_vehicle    0.082   1.057   0.446   1.013   0.125   0.387
# pedestrian      0.426   0.688   0.294   0.508   0.459   0.195
# motorcycle      0.343   0.696   0.260   0.475   1.268   0.180
# bicycle 0.275   0.691   0.275   0.578   0.452   0.015
# traffic_cone    0.521   0.555   0.314   nan     nan     nan
# barrier 0.473   0.619   0.293   0.138   nan     nan

# after fixing h,w bug in feature-sampling
# mAP: 0.3469
# mATE: 0.7651
# mASE: 0.2678
# mAOE: 0.3916
# mAVE: 0.8758
# mAAE: 0.2110
# NDS: 0.4223
# Eval time: 117.2s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.546   0.544   0.152   0.070   0.911   0.208
# truck   0.286   0.834   0.212   0.113   1.005   0.231
# bus     0.346   0.870   0.196   0.116   2.063   0.383
# trailer 0.167   1.106   0.233   0.549   0.687   0.093
# construction_vehicle    0.082   1.060   0.449   0.960   0.120   0.384
# pedestrian      0.424   0.700   0.295   0.512   0.462   0.194
# motorcycle      0.340   0.709   0.259   0.489   1.288   0.176
# bicycle 0.278   0.698   0.275   0.586   0.473   0.018
# traffic_cone    0.529   0.526   0.313   nan     nan     nan
# barrier 0.471   0.603   0.292   0.131   nan     nan
