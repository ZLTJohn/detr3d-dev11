_base_ = [
    # '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    # '/home/zhenglt/mmdev11/mmdet3d-latest/configs/_base_/datasets/nus-3d.py',
    'mmdet3d::configs/_base_/default_runtime.py'
]
#### debugging no auto_fp32
#### Resize3D
# plugin=True
# plugin_dir='projects/mmdet3d_plugin/'
custom_imports = dict(imports=['projects.detr3d'])
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

input_modality = dict(use_lidar=True,
                      use_camera=True,
                      use_radar=False,
                      use_map=False,
                      use_external=False)
# this means type='Detr3D' will be processed as 'mmdet3d.Detr3D'
debug_vis_cfg = dict(debug_dir='debug/visualization',
                     gt_range=[0, 105],
                     pc_range=point_cloud_range,
                     vis_count=300,
                     debug_name='dev1x_watch')
default_scope = 'mmdet3d'
model = dict(
    type='Detr3D',
    use_grid_mask=True,
    debug_vis_cfg=debug_vis_cfg,
    data_preprocessor=dict(type='Det3DDataPreprocessor',
                           **img_norm_cfg,
                           pad_size_divisor=32),
    img_backbone=dict(type='mmdet.ResNet',
                      depth=101,
                      num_stages=4,
                      out_indices=(0, 1, 2, 3),
                      frozen_stages=1,
                      norm_cfg=dict(type='BN2d', requires_grad=False),
                      norm_eval=True,
                      style='caffe',
                      dcn=dict(type='DCNv2',
                               deform_groups=1,
                               fallback_on_stride=False),
                      stage_with_dcn=(False, False, True, True)),
    img_neck=dict(type='mmdet.FPN',
                  in_channels=[256, 512, 1024, 2048],
                  out_channels=256,
                  start_level=1,
                  add_extra_convs='on_output',
                  num_outs=4,
                  relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='Detr3DHead',
        num_query=900,
        num_classes=10,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='Detr3DTransformer',
            decoder=dict(
                type='Detr3DTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='mmdet.DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',  # mmcv.
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(type='Detr3DCrossAtten',
                             pc_range=point_cloud_range,
                             num_points=1,
                             embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(type='mmdet.SinePositionalEncoding',
                                 num_feats=128,
                                 normalize=True,
                                 offset=-0.5),
        loss_cls=dict(type='mmdet.FocalLoss',
                      use_sigmoid=True,
                      gamma=2.0,
                      alpha=0.25,
                      loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            # ↓ Fake cost. This is just to make it compatible with DETR head.
            iou_cost=dict(type='mmdet.IoUCost', weight=0.0),
            pc_range=point_cloud_range))))

dataset_type = 'NuScenesDataset'
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
    # # dict(type='LidarBox3dVersionTransfrom'),  #petr's solution
    # dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=6),
    dict(type='filename2img_path'),  #fix it in ↑ via a PR
    # dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='MultiViewWrapper', transforms=test_transforms),
    # dict(type='AddPointCloudFilename'),
    # dict(type='PadMultiViewImage', size_divisor=32),
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

# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# jupyter-packaging 0.12.3 requires setuptools>=60.2.0, but you have setuptools 58.0.4 which is incompatible.
# setuptools 65 downgrades to 58.In mmlab-node we use setuptools 61 but occurs NO errors
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

#train 900x1600, val 928x1600
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
