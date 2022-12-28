_base_ = [
    # '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    # '/home/zhenglt/mmdev11/mmdet3d-latest/configs/_base_/datasets/nus-3d.py',
    'mmdet3d::_base_/default_runtime.py'
]
# # debugging no auto_fp32
# # Resize3D
# plugin=True
# plugin_dir='projects/mmdet3d_plugin/'
custom_imports = dict(imports=['projects.detr3d'])
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-35, -75, -2, 75, 75, 4]
voxel_size = [0.2, 0.2, 8]
num_views = 5
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    bgr_to_rgb=True)
# For nuScenes we usually do 10-class detection
class_names = [  # 不确定sign类别是否叫sign
    'Car', 'Pedestrian', 'Cyclist'
]

# this means type='DETR3D' will be processed as 'mmdet3d.DETR3D'
debug_vis_cfg = dict(debug_dir='debug/visualization',
                     gt_range=[0, 105],
                     pc_range=point_cloud_range,
                     vis_count=15,
                     debug_name='dev1x_watch')
default_scope = 'mmdet3d'
model = dict(
    type='DETR3D',
    use_grid_mask=True,
    # debug_vis_cfg=debug_vis_cfg,
    data_preprocessor=dict(type='Det3DDataPreprocessor',
                           **img_norm_cfg,
                           pad_size_divisor=32),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        # with_cp=True,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='open-mmlab://detectron2/resnet101_caffe')
    ),
    img_neck=dict(type='mmdet.FPN',
                  in_channels=[256, 512, 1024, 2048],
                  out_channels=256,
                  start_level=1,
                  add_extra_convs='on_output',
                  num_outs=4,
                  relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='DETR3DHead',
        num_query=900,
        num_classes=3,
        in_channels=256,
        code_size=8,
        # we don't infer velocity here,
        # but infer(cx,cy,l,w,cz,h,sin(φ),cos(φ)) for bboxes
        # specify the weights since default code_size is 10
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='Detr3DTransformer',
            num_cams=num_views,
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
                             num_cams=num_views,
                             num_points=1,
                             embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(type='NMSFreeCoder',
                        post_center_range=point_cloud_range,
                        pc_range=point_cloud_range,
                        max_num=300,
                        voxel_size=voxel_size,
                        num_classes=3),
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

dataset_type = 'WaymoDataset'
data_root = 'data/waymo_dev1x/kitti_format/'
img_scale = (960, 640)
test_transforms = [
    dict(type='RandomResize3D',
         scale=img_scale,
         ratio_range=(1., 1.),
         keep_ratio=False)
]
train_transforms = [dict(type='PhotoMetricDistortion3D')] + test_transforms

file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles',
         to_float32=True,
         num_views=num_views),
    dict(type='filename2img_path'),
    # dict(type='PhotoMetricDistortionMultiViewImage'),
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
    dict(type='LoadMultiViewImageFromFiles',
         to_float32=True,
         num_views=num_views),
    dict(type='filename2img_path'),  # fix it in ↑ via a PR
    dict(type='MultiViewWrapper', transforms=test_transforms),
    dict(type='Pack3DDetInputs', keys=['img'])
]

metainfo = dict(classes=class_names)
data_prefix = dict(
    pts='training/velodyne',
    sweeps='training/velodyne',
    CAM_FRONT='training/image_0',
    CAM_FRONT_RIGHT='training/image_1',
    CAM_FRONT_LEFT='training/image_2',
    CAM_SIDE_RIGHT='training/image_3',
    CAM_SIDE_LEFT='training/image_4',
)
input_modality = dict(use_lidar=True, use_camera=True)
train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='waymo_infos_train.pkl',
        pipeline=train_pipeline,
        load_type='frame_based',
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        data_prefix=data_prefix,
        cam_sync_instances=True,
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
                                   ann_file='waymo_infos_val.pkl',
                                   load_interval=100,
                                   load_type='frame_based',
                                   pipeline=test_pipeline,
                                   metainfo=metainfo,
                                   modality=input_modality,
                                   test_mode=True,
                                   data_prefix=data_prefix,
                                   cam_sync_instances=True,
                                   box_type_3d='LiDAR'))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='WaymoMetric',
    ann_file='./data/waymo_dev1x/kitti_format/waymo_infos_val.pkl',
    load_interval=100,
    waymo_bin_file='./data/waymo_dev1x/waymo_format/gt.bin',
    data_root='./data/waymo_dev1x/waymo_format',
    metric='LET_mAP')
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
                 val_interval=12)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(checkpoint=dict(
    type='CheckpointHook', interval=1, max_keep_ckpts=1, save_last=True))
load_from = \
    'ckpts/waymo_pretrain_pgd_mv_8gpu_for_detr3d_backbone_statedict_only.pth'

vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')
