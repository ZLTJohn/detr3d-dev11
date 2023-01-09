_base_ = [
    'mmdet3d::_base_/default_runtime.py'
]
custom_imports = dict(imports=['projects.detr3d'])
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
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

dataset_type = 'NuScenesDataset'
data_root = 'data/nus_v2/'

test_transforms = [
    dict(type='RandomResize3D',
         scale=(1600, 928),
         ratio_range=(1., 1.),
         keep_ratio=False)
]
train_transforms = [dict(type='PhotoMetricDistortion3D')] + test_transforms

file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=6),
    dict(type='filename2img_path'),
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
    dict(type='filename2img_path'),  # fix it in ↑ via a PR
    dict(type='MultiViewWrapper', transforms=test_transforms),
    dict(type='Pack3DDetInputs', keys=['img'])
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
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='debug_val.pkl',
        pipeline=train_pipeline,
        load_type='frame_based',
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        data_prefix=data_prefix,
        box_type_3d='LiDAR'))

val_dataloader = dict(batch_size=1,
                      num_workers=0,
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

# setuptools 65 downgrades to 58.
# In mmlab-node we use setuptools 61 but occurs NO errors
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')
# train&&val: 900,1600
# mAP: 0.3546
# mATE: 0.7639
# mASE: 0.2695
# mAOE: 0.3953
# mAVE: 0.8853
# mAAE: 0.2108
# NDS: 0.4248
# Eval time: 120.8s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.549   0.542   0.150   0.071   0.916   0.209
# truck   0.293   0.815   0.211   0.101   1.043   0.229
# bus     0.368   0.851   0.196   0.117   1.865   0.316
# trailer 0.170   1.127   0.253   0.500   0.886   0.180
# construction_vehicle    0.084   1.098   0.453   1.033   0.160   0.389
# pedestrian      0.422   0.705   0.298   0.504   0.464   0.197
# motorcycle      0.346   0.704   0.258   0.463   1.259   0.143
# bicycle 0.299   0.656   0.272   0.640   0.489   0.024
# traffic_cone    0.539   0.525   0.310   nan     nan     nan
# barrier 0.475   0.616   0.294   0.129   nan     nan

# train 900x1600, val 928x1600
# mAP: 0.3379
# mATE: 0.7973
# mASE: 0.2700
# mAOE: 0.4106
# mAVE: 0.8564
# mAAE: 0.1999
# NDS: 0.4156
# Eval time: 118.2s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.532   0.589   0.151   0.073   0.920   0.205
# truck   0.287   0.845   0.213   0.101   1.075   0.232
# bus     0.362   0.840   0.194   0.125   1.861   0.309
# trailer 0.157   1.109   0.250   0.601   0.637   0.102
# construction_vehicle    0.081   1.108   0.454   1.047   0.142   0.390
# pedestrian      0.408   0.731   0.299   0.509   0.466   0.196
# motorcycle      0.327   0.748   0.261   0.455   1.259   0.142
# bicycle 0.281   0.711   0.275   0.654   0.491   0.024
# traffic_cone    0.498   0.599   0.309   nan     nan     nan
# barrier 0.445   0.693   0.295   0.130   nan     nan

# train&&val: 928,1600
# mAP: 0.3444
# mATE: 0.7884
# mASE: 0.2729
# mAOE: 0.3975
# mAVE: 0.8213
# mAAE: 0.2067
# NDS: 0.4235
# Eval time: 117.5s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.542   0.553   0.151   0.072   0.883   0.216
# truck   0.294   0.822   0.215   0.103   0.948   0.251
# bus     0.360   0.894   0.203   0.132   1.915   0.358
# trailer 0.131   1.155   0.251   0.599   0.532   0.076
# construction_vehicle    0.083   1.111   0.463   0.960   0.111   0.354
# pedestrian      0.410   0.718   0.299   0.521   0.466   0.198
# motorcycle      0.344   0.714   0.255   0.459   1.212   0.189
# bicycle 0.291   0.715   0.263   0.593   0.502   0.012
# traffic_cone    0.521   0.552   0.336   nan     nan     nan
# barrier 0.468   0.652   0.293   0.138   nan     nan

# torch.Size([1, 6, 256, 116, 200])
# torch.Size([1, 6, 256, 58, 100])
# torch.Size([1, 6, 256, 29, 50])
# torch.Size([1, 6, 256, 15, 25])

# torch.Size([1, 6, 256, 232, 400])
# torch.Size([1, 6, 256, 116, 200])
# torch.Size([1, 6, 256, 58, 100])
# torch.Size([1, 6, 256, 29, 50])
