_base_ = [
    'mmdet3d::_base_/default_runtime.py',
]
load_interval_type = {
    'full': 1,
    'part': 3,
    'mini': 10
}
# meta configs
default_scope = 'mmdet3d'
custom_imports = dict(imports=['projects.detr3d'])
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 4.0]
waymo_class_names = ['Car', 'Pedestrian', 'Cyclist']
nusc_class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'motorcycle', 'bicycle', 'pedestrian'
]
argo2_class_names = ['ARTICULATED_BUS', 'BICYCLE', 'BICYCLIST', 'BOLLARD', 
    'BOX_TRUCK', 'BUS', 'CONSTRUCTION_BARREL', 'CONSTRUCTION_CONE', 
    'DOG', 'LARGE_VEHICLE', 'MESSAGE_BOARD_TRAILER', 
    'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'MOTORCYCLE', 'MOTORCYCLIST',
    'PEDESTRIAN', 'REGULAR_VEHICLE', 'SCHOOL_BUS', 'SIGN', 'STOP_SIGN', 
    'STROLLER', 'TRUCK', 'TRUCK_CAB', 'VEHICULAR_TRAILER', 'WHEELCHAIR', 
    'WHEELED_DEVICE', 'WHEELED_RIDER']
argo2_name_map = {
    'REGULAR_VEHICLE': 'Car',
    'LARGE_VEHICLE': 'Car',
    'BUS': 'Car',
    'BOX_TRUCK': 'Car',
    'TRUCK': 'Car',
    'MOTORCYCLE': 'Car',
    'VEHICULAR_TRAILER': 'Car',
    'TRUCK_CAB': 'Car',
    'SCHOOL_BUS': 'Car',
    'ARTICULATED_BUS': 'Car',
    'MESSAGE_BOARD_TRAILER': 'Car',
    'TRAFFIC_LIGHT_TRAILER': 'Car',
    'PEDESTRIAN': 'Pedestrian',
    'WHEELED_RIDER': 'Pedestrian',
    'OFFICIAL_SIGNALER': 'Pedestrian',
    'BICYCLE': 'Cyclist',
    'BICYCLIST': 'Cyclist'
}
num_class = 3
argo2_num_views = 7
img_size_argo2 = (1024,800)
img_size_nusc = (800, 450)
img_size_waymo = (960, 640)
evaluation_interval = 12 # epochs
# load_from = 'ckpts/'
argo2_type = 'Argo2Dataset'
argo2_data_root = '/localdata_ssd/argo2_dev1x'
argo2_train_pkl = 'argo2_infos_train_2Hz.pkl'  
argo2_train_interval = 1    # 2Hz means interval = 5
argo2_val_pkl = 'argo2_infos_val_2Hz.pkl'
argo2_val_interval = 1

nusc_type = 'CustomNusc'
nusc_data_root = '/localdata_ssd/nusc_dev1x'
nusc_train_pkl = 'nuscenes_infos_train.pkl' 
nusc_train_interval = 1
nusc_val_pkl = 'nuscenes_infos_val.pkl'
nusc_val_interval = 1

waymo_type = 'WaymoDataset'
waymo_data_root = '/localdata_ssd/waymo_dev1x/'
waymo_train_pkl = 'waymo_infos_train_2Hz.pkl'
waymo_train_interval = 1    # 2Hz means interval = 5
waymo_val_pkl = 'waymo_infos_val_2Hz.pkl'
waymo_val_interval = 1

# load_interval_factor = load_interval_type['part']
input_modality = dict(use_lidar=True, # True if debug_vis
                      use_camera=True)

# work_dir = './work_dirs_ablate/'
# Default or RtK-aware 
detr3d_feature_sampler = dict(type='DefaultFeatSampler')
# detr3d_feature_sampler = dict(type='GeoAwareFeatSampler', base_fxfy = 2068.5, base_dist = 51.2)

# Default or ManyCam or single Cam
AttnInfo = dict(type='Detr3DCrossAtten',num_cams = argo2_num_views, waymo_with_nuscene = True, waymo_with_argo2 = True,)
# AttnInfo = dict(type='Detr3DCrossAtten_ManyCam', num_cams = 18)
# AttnInfo = dict(type='Detr3DCrossAtten_CamEmb', num_cams = 1)

detr3d_crossAttn = dict(
    **AttnInfo,
    feature_sampler=detr3d_feature_sampler,
    pc_range=point_cloud_range,
    num_points=1,
    embed_dims=256,)


ego_aug_train = []
ego_aug_eval = []
ego_aug_train = [dict(type='EgoTranslate', Tr_range = [-2,-20,0,2,20,0], eval=False)]
ego_aug_eval = [dict(type='EgoTranslate', Tr_range = [-2,-2,0,2,2,0])]

argo2_egoXY = dict(type='EgoTranslate', trans = [1.3493238413,0.0031899048,0])
argo2_egoZ = dict(type='EgoTranslate', trans = [0,0,0.513719993459064])
argo2_intrinsics_sync = dict(type='MultiViewWrapper', transforms=dict(type='Resize3D', scale_factor=1.2189, keep_ratio=True))
argo2_synchronization = [
    # argo2_egoXY, 
    # argo2_egoZ,
    # argo2_intrinsics_sync,
]

nusc_egoXY = dict(type='EgoTranslate', trans = [-0.0866953254,0.0922405545,0])
nusc_egoZ = dict(type='EgoTranslate', trans = [0,0,-0.8865816593170166])
nusc_rotate_egoaxis = dict(type='RotateScene_neg90')
nusc_intrinsics_sync = dict(type='MultiViewWrapper', transforms=dict(type='Resize3D', scale_factor=1.64362336, keep_ratio=True))
nusc_synchronization = [
    # nusc_egoXY,
    # nusc_egoZ, 
    # nusc_rotate_egoaxis,
    # nusc_intrinsics_sync,
]

waymo_egoXY = dict(type='EgoTranslate', trans = [1.3981133014,-0.0023025204,0])
waymo_egoZ = dict(type='EgoTranslate', trans = [0,0,1.2201894521713257])
waymo_synchronization = [
    # waymo_egoXY,
    # waymo_egoZ,
]
# model
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], 
                    std=[58.395, 57.12, 57.375],
                    bgr_to_rgb=True)
debug_vis_cfg = dict(debug_dir='debug/visualization',
                     gt_range=[0, 105],
                     pc_range=point_cloud_range,
                     vis_count=200,
                     debug_name='ablate_all',
                    #  ROIsampling=detr3d_feature_sampler
                     )
# model_wrapper_cfg = dict(type = 'CustomMMDDP', static_graph = True)
model = dict(
    type='DETR3D',
    use_grid_mask=True,
    debug_vis_cfg=debug_vis_cfg,
    data_preprocessor=dict(type='Det3DDataPreprocessor',
                           **img_norm_cfg,
                           pad_size_divisor=32),
    img_backbone=dict(type='mmdet.ResNet',
                      depth=50,
                      num_stages=4,
                      out_indices=(0, 1, 2, 3),
                      frozen_stages=1,
                      norm_cfg=dict(type='BN2d', requires_grad=False),
                      norm_eval=True,
                    #   with_cp=True,
                      style='pytorch',
                      init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
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
        type='DETR3DHead',
        num_query=900,
        num_classes=num_class,
        in_channels=256,
        code_size=8,    #we don't infer velocity here, but infer(x,y,z,w,h,l,sin(θ),cos(θ)) for bboxes
        code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], #specify the weights since default length is 10
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='Detr3DTransformer',
            num_cams = argo2_num_views,
            decoder=dict(
                type='Detr3DTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='mmdet.DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        detr3d_crossAttn
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=point_cloud_range,
            pc_range=point_cloud_range,
            max_num=300,
            num_classes=num_class),
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
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            # ↓ Fake cost. This is just to make it compatible with DETR head.
            iou_cost=dict(type='mmdet.IoUCost', weight=0.0),
            pc_range=point_cloud_range))))

#dataset
# order:  egoXY egoZ egoaxis ShiftAug RangeFilter
# order: multiview_resize K
argo2_test_transforms = [
    dict(type='RandomResize3D',
         scale=img_size_argo2,
         ratio_range=(1., 1.),
         keep_ratio=False)
]

argo2_pipeline_default = [
    dict(type='Argo2LoadMultiViewImageFromFiles', to_float32=True, num_views=argo2_num_views),
    dict(type='filename2img_path'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectNameFilter', classes=argo2_class_names), # Deprecated now
    dict(type='ProjectLabelToWaymoClass', class_names = argo2_class_names, name_map = argo2_name_map),
    dict(type='MultiViewWrapper', transforms=argo2_test_transforms),
]
argo2_train_pipeline = \
    argo2_pipeline_default + \
    [dict(type='MultiViewWrapper', transforms=[dict(type='PhotoMetricDistortion3D')])] + \
    argo2_synchronization + \
    ego_aug_train + \
    [dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
     dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])]

argo2_test_pipeline = \
    [dict(type='evalann2ann')] + \
    argo2_pipeline_default + \
    argo2_synchronization + \
    ego_aug_eval + \
    [dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
     dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])]

argo2_data_prefix = dict()

argo2_default = dict(
    load_type='frame_based',
    modality=input_modality,
    metainfo=dict(classes=argo2_class_names),
    data_prefix=argo2_data_prefix,
    box_type_3d='LiDAR')
argo2_train = dict(type=argo2_type,
                data_root=argo2_data_root,
                ann_file=argo2_train_pkl,
                pipeline=argo2_train_pipeline,
                load_interval = argo2_train_interval,
                test_mode=False,
                **argo2_default)
argo2_val = dict(type=argo2_type,
                data_root=argo2_data_root,
                ann_file=argo2_val_pkl,
                pipeline=argo2_test_pipeline,
                load_interval = argo2_val_interval,
                test_mode=True,
                **argo2_default)

nusc_test_transforms = [
    dict(type='RandomResize3D',
         scale=img_size_nusc,
         ratio_range=(1., 1.),
         keep_ratio=False)
]
nusc_pipeline_default = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=6),
    dict(type='filename2img_path'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectNameFilter', classes=nusc_class_names),
    dict(type='ProjectLabelToWaymoClass', class_names = nusc_class_names),
    dict(type='MultiViewWrapper', transforms=nusc_test_transforms),
]
nusc_train_pipeline = \
    nusc_pipeline_default + \
    [dict(type='MultiViewWrapper', transforms=[dict(type='PhotoMetricDistortion3D')])] + \
    nusc_synchronization + \
    ego_aug_train + \
    [dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])]
nusc_test_pipeline = \
    [dict(type='evalann2ann')] + \
    nusc_pipeline_default + \
    nusc_synchronization + \
    ego_aug_eval + \
    [dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])]

nusc_data_prefix = dict(pts='samples/LIDAR_TOP',
                   sweeps='sweeps/LIDAR_TOP',
                   CAM_FRONT='samples/CAM_FRONT',
                   CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
                   CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
                   CAM_BACK='samples/CAM_BACK',
                   CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
                   CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',)

nusc_default = dict(
    load_type='frame_based',
    modality=input_modality,
    metainfo=dict(classes=nusc_class_names),
    data_prefix=nusc_data_prefix,
    with_velocity=False,
    use_valid_flag=True,
    box_type_3d='LiDAR')
nusc_train = dict(type=nusc_type,
                 data_root=nusc_data_root,
                 ann_file=nusc_train_pkl,
                 pipeline=nusc_train_pipeline,
                 load_interval = nusc_train_interval,
                test_mode=False,
                 **nusc_default)
nusc_val = dict(type=nusc_type,
                data_root=nusc_data_root,
                ann_file=nusc_val_pkl,
                pipeline=nusc_test_pipeline,
                load_interval = nusc_val_interval,
                test_mode=True,
                **nusc_default)

waymo_test_transforms = [
    dict(type='RandomResize3D',
         scale=img_size_waymo,
         ratio_range=(1., 1.),
         keep_ratio=False)
]
waymo_pipeline_default = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=5),
    dict(type='filename2img_path'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectNameFilter', classes=waymo_class_names),
    dict(type='MultiViewWrapper', transforms=waymo_test_transforms),
]

waymo_train_pipeline = \
    waymo_pipeline_default + \
    [dict(type='MultiViewWrapper', transforms=[dict(type='PhotoMetricDistortion3D')])] + \
    waymo_synchronization + \
    ego_aug_train + \
    [dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])]

waymo_test_pipeline = \
    [dict(type='evalann2ann')] + \
    waymo_pipeline_default + \
    waymo_synchronization + \
    ego_aug_eval + \
    [dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])]

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
                  data_root=waymo_data_root,
                  ann_file=waymo_train_pkl,
                  pipeline=waymo_train_pipeline,
                  load_interval= waymo_train_interval,
                  test_mode=False,
                  **waymo_default)
waymo_val = dict(type=waymo_type,
                 data_root=waymo_data_root,
                 ann_file=waymo_val_pkl,
                 pipeline=waymo_test_pipeline,
                 load_interval=waymo_val_interval,
                 test_mode=True,
                 **waymo_default)

argnuway_train = dict(
        type='CustomConcatDataset',
        datasets=[argo2_train, nusc_train, waymo_train])
argnuway_val = dict(
        type='CustomConcatDataset',
        datasets=[argo2_val, nusc_val, waymo_val])

dataloader_default = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False)
train_dataloader = dict(
    **dataloader_default,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=nusc_train)
val_dataloader = dict(
    **dataloader_default,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=nusc_val)
test_dataloader = val_dataloader

val_evaluator = dict(type = 'JointMetric')
test_evaluator = val_evaluator

# learning rate
file_client_args = dict(backend='disk')
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
                 val_interval=evaluation_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(checkpoint=dict(
    type='CheckpointHook', interval=1, max_keep_ckpts=1, save_last=True))
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')