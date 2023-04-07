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
point_cloud_range = [0, -51.2, -5.0, 51.2, 51.2, 4.0]
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
lyft_class_names = [
    'car', 'truck', 'bus', 'emergency_vehicle', 'other_vehicle', 
    'motorcycle', 'bicycle', 'pedestrian', 'animal'
]
kitti_class_names = ['Pedestrian','Cyclist','Car','Van','Truck',
                     'Person_sitting','Tram','Misc']

K360_class_names = ['bicycle', 'box', 'bridge', 'building', 'bus', 'car',
           'caravan', 'garage', 'lamp', 'motorcycle', 'person', 
           'pole', 'rider', 'smallpole', 'stop', 'traffic light', 
           'traffic sign', 'trailer', 'train', 'trash bin', 'truck', 
           'tunnel', 'unknown construction', 'unknown object', 
           'unknown vehicle', 'vending machine']
num_class = 3
argo2_num_views = 1
nusc_num_views = 1
waymo_num_views = 1
lyft_num_views = 1
kitti_num_views = 1
K360_num_views = 1

K360_selected_cam = 'CAM0'
img_size_argo2 = (2048,1600)
img_size_nusc = (1600, 900)
img_size_waymo = (1920, 1280)
img_scale_factor_lyft = 1.0
img_size_kitti = (1242, 375)
img_size_K360 = (1408, 376)
# img_size_lyft = (1920,1080) and (1224,1024)
evaluation_interval = 12 # epochs
# load_from = 'ckpts/'
argo2_type = 'Argo2Dataset'
argo2_data_root = 'data/argo2/'
argo2_train_pkl = 'argo2_infos_train_2Hz_mono_front.pkl'  
argo2_train_interval = 1    # 2Hz_part means interval = 5x3
argo2_val_pkl = 'argo2_infos_val_2Hz_part_mono_front.pkl'
argo2_val_interval = 1

nusc_type = 'CustomNusc'
nusc_data_root = 'data/nus_v2/'
nusc_train_pkl = 'nuscenes_infos_train_mono_front.pkl' 
nusc_train_interval = 1
nusc_val_pkl = 'nuscenes_infos_val_part_mono_front.pkl'
nusc_val_interval = 1

waymo_type = 'CustomWaymo'
waymo_data_root = 'data/waymo_dev1x/kitti_format'
waymo_train_pkl = 'waymo_infos_train_2Hz_mono_front.pkl'
waymo_train_interval = 1    # 2Hz_part means interval = 5x3
waymo_val_pkl = 'waymo_infos_val_2Hz_part_mono_front.pkl'
waymo_val_interval = 1

lyft_type = 'CustomLyft'
lyft_data_root = 'data/lyft/'
lyft_train_pkl = 'lyft_infos_train_mono_front.pkl' 
lyft_train_interval = 1
lyft_val_pkl = 'lyft_infos_val_mono_front.pkl'
lyft_val_interval = 2

kitti_type = 'CustomKitti'
kitti_data_root = 'data/kitti/'
kitti_train_pkl = 'kitti_infos_train.pkl'
kitti_train_interval = 1
kitti_val_pkl = 'kitti_infos_val.pkl'
kitti_val_interval = 1

K360_type = 'Kitti360Dataset'
K360_data_root = 'data/kitti-360/'
K360_train_pkl = 'kitti360_infos_train.pkl' # 40000 frame
K360_train_interval = 1
K360_val_pkl = 'kitti360_infos_val.pkl' # 10000 frame
K360_val_interval = 5

# load_interval_factor = load_interval_type['part']
input_modality = dict(use_lidar=True, # True if debug_vis
                      use_camera=True)
work_dir = './work_dirs_extended/3.00ANW_front_r50_fullres'
resume = True
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
    'BICYCLE': 'Cyclist',   # TO REMOVE
    'BICYCLIST': 'Cyclist'
}
lyft_name_map = {
    'car': 'Car',
    'truck': 'Car',
    'bus': 'Car',
    'emergency_vehicle': 'Car',
    'other_vehicle': 'Car',
    'motorcycle': 'Car',
    'pedestrian': 'Pedestrian',
    # 'animal': 'Pedestrian',
    'bicycle': 'Cyclist'
}
kitti_name_map = {
    'Pedestrian': 'Pedestrian',
    'Cyclist': 'Cyclist',
    'Car': 'Car',
    'Van': 'Car',
    'Truck': 'Car',
    'Person_sitting': 'Pedestrian',
    'Tram': 'Car'
}
K360_name_map = {
    'person': 'Pedestrian',
    'bicycle': 'Cyclist',
    'rider': 'Cyclist',
    'bus': 'Car',
    'car': 'Car',
    'caravan': 'Car',
    'motorcycle': 'Car',
    'trailer': 'Car',
    'train': 'Car',
    'truck': 'Car',
    'unknown vehicle': 'Car'
}
# model
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], 
                    std=[58.395, 57.12, 57.375],
                    bgr_to_rgb=True)
debug_vis_cfg = dict(debug_dir='debug/visualization',
                     gt_range=[0, 105],
                     pc_range=point_cloud_range,
                     vis_count=20,
                     debug_name='mono_debug')
# model_wrapper_cfg = dict(type = 'CustomMMDDP', static_graph = True)
model = dict(
    type='DETR3D',
    use_grid_mask=True,
    # debug_vis_cfg=debug_vis_cfg,
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
                        dict(type='Detr3DCrossAtten',
                             pc_range=point_cloud_range,
                             num_cams = argo2_num_views,
                             num_points=1,
                             embed_dims=256)
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
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=argo2_class_names), # Deprecated now
    dict(type='ProjectLabelToWaymoClass', class_names = argo2_class_names, name_map = argo2_name_map),
]
argo2_train_pipeline = argo2_pipeline_default + [
    dict(type='MultiViewWrapper', transforms=[dict(type='PhotoMetricDistortion3D')] + argo2_test_transforms),
    dict(type='Pack3DDetInputsExtra', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
argo2_test_pipeline = [dict(type='evalann2ann')] + argo2_pipeline_default + [
    dict(type='MultiViewWrapper', transforms=argo2_test_transforms),
    dict(type='Pack3DDetInputsExtra', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

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
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=nusc_num_views),
    dict(type='filename2img_path'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectNameFilter', classes=nusc_class_names),
    dict(type='RotateScene_neg90'),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ProjectLabelToWaymoClass', class_names = nusc_class_names),
]
nusc_train_pipeline = nusc_pipeline_default + [
    dict(type='MultiViewWrapper', transforms=[dict(type='PhotoMetricDistortion3D')] + nusc_test_transforms),
    dict(type='Pack3DDetInputsExtra', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
nusc_test_pipeline = [dict(type='evalann2ann')] + nusc_pipeline_default + [
    dict(type='MultiViewWrapper', transforms=nusc_test_transforms),
    dict(type='Pack3DDetInputsExtra', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

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
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=waymo_num_views),
    dict(type='filename2img_path'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=waymo_class_names),
]

waymo_train_pipeline = waymo_pipeline_default + [
    dict(type='MultiViewWrapper', transforms=[dict(type='PhotoMetricDistortion3D')] + waymo_test_transforms),
    dict(type='Pack3DDetInputsExtra', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
waymo_test_pipeline = [dict(type='evalann2ann')] + waymo_pipeline_default + [
    dict(type='MultiViewWrapper', transforms=waymo_test_transforms),
    dict(type='Pack3DDetInputsExtra', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

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

lyft_test_transforms = [
    dict(type='Resize3D',
        #  scale=img_size_lyft,
         scale_factor = img_scale_factor_lyft,
         keep_ratio=False)
]
lyft_pipeline_default = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=lyft_num_views),
    dict(type='filename2img_path'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectNameFilter', classes=lyft_class_names),
    dict(type='RotateScene_neg90'),
    dict(type='RotateScene_neg90'),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ProjectLabelToWaymoClass', class_names = lyft_class_names, name_map = lyft_name_map),
]
lyft_train_pipeline = lyft_pipeline_default + [
    dict(type='MultiViewWrapper', transforms=[dict(type='PhotoMetricDistortion3D')] + lyft_test_transforms),
    # dict(type='Ksync',fx = 1034),
    dict(type='Pack3DDetInputsExtra', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
lyft_test_pipeline = [dict(type='evalann2ann')] + lyft_pipeline_default + [
    dict(type='MultiViewWrapper', transforms=lyft_test_transforms),
    # dict(type='Ksync',fx = 1034),
    dict(type='Pack3DDetInputsExtra', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
lyft_data_prefix = dict(pts='v1.01-train/lidar/', 
                        sweeps='v1.01-train/lidar/',
                        CAM_FRONT='v1.01-train/images/', 
                        CAM_FRONT_RIGHT='v1.01-train/images/', 
                        CAM_FRONT_LEFT='v1.01-train/images/', 
                        CAM_BACK='v1.01-train/images/', 
                        CAM_BACK_LEFT='v1.01-train/images/', 
                        CAM_BACK_RIGHT='v1.01-train/images/')

lyft_default = dict(
    # load_type='frame_based',
    modality=input_modality,
    metainfo=dict(classes=lyft_class_names),
    data_prefix=lyft_data_prefix,
    box_type_3d='LiDAR')
lyft_train = dict(type=lyft_type,
                 data_root=lyft_data_root,
                 ann_file=lyft_train_pkl,
                 pipeline=lyft_train_pipeline,
                 load_interval = lyft_train_interval,
                 test_mode=False,
                 **lyft_default)
lyft_val = dict(type=lyft_type,
                data_root=lyft_data_root,
                ann_file=lyft_val_pkl,
                pipeline=lyft_test_pipeline,
                load_interval = lyft_val_interval,
                test_mode=True,
                **lyft_default)

kitti_test_transforms = [
    dict(type='RandomResize3D',
         scale=img_size_kitti,
         ratio_range=(1., 1.),
         keep_ratio=False)
]
kitti_pipeline_default = [
    dict(type='Argo2LoadMultiViewImageFromFiles', flip_front_cam=False, to_float32=True, num_views=kitti_num_views),
    dict(type='filename2img_path'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=kitti_class_names),
    dict(type='ProjectLabelToWaymoClass', class_names = kitti_class_names, name_map = kitti_name_map),
]

kitti_train_pipeline = kitti_pipeline_default + [
    dict(type='MultiViewWrapper', transforms=[dict(type='PhotoMetricDistortion3D')] + kitti_test_transforms),
    dict(type='Pack3DDetInputsExtra', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
kitti_test_pipeline = [dict(type='evalann2ann')] + kitti_pipeline_default + [
    dict(type='MultiViewWrapper', transforms=kitti_test_transforms),
    dict(type='Pack3DDetInputsExtra', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

kitti_data_prefix = dict(
    pts='training/velodyne',
    sweeps='training/velodyne',
    img='training/image_2',)

kitti_default = dict(
    load_type='frame_based',
    modality=input_modality,
    data_prefix=kitti_data_prefix,
    metainfo=dict(classes=kitti_class_names),
    default_cam_key='CAM2',
    box_type_3d='LiDAR')
kitti_train =dict(type=kitti_type,
                  data_root=kitti_data_root,
                  ann_file=kitti_train_pkl,
                  pipeline=kitti_train_pipeline,
                  load_interval= kitti_train_interval,
                  test_mode=False,
                  **kitti_default)
kitti_val = dict(type=kitti_type,
                 data_root=kitti_data_root,
                 ann_file=kitti_val_pkl,
                 pipeline=kitti_test_pipeline,
                 load_interval=kitti_val_interval,
                 test_mode=True,
                 **kitti_default)

K360_test_transforms = [
    dict(type='RandomResize3D',
         scale=img_size_K360,
         ratio_range=(1., 1.),
         keep_ratio=False)
]
K360_pipeline_default = [
    dict(type='Argo2LoadMultiViewImageFromFiles', flip_front_cam=False, to_float32=True, num_views=K360_num_views),
    dict(type='filename2img_path'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=K360_class_names),
    dict(type='ProjectLabelToWaymoClass', class_names = K360_class_names, name_map = K360_name_map),
]
K360_train_pipeline = K360_pipeline_default + [
    dict(type='MultiViewWrapper', transforms=[dict(type='PhotoMetricDistortion3D')] + K360_test_transforms),
    dict(type='Pack3DDetInputsExtra', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
K360_test_pipeline = [dict(type='evalann2ann')] + K360_pipeline_default + [
    dict(type='MultiViewWrapper', transforms=K360_test_transforms),
    dict(type='Pack3DDetInputsExtra', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
K360_data_prefix = dict()

K360_default = dict(
    load_type='frame_based',
    modality=input_modality,
    data_prefix=K360_data_prefix,
    metainfo=dict(classes=K360_class_names),
    box_type_3d='LiDAR')
K360_train =dict(type=K360_type,
                  data_root=K360_data_root,
                  ann_file=K360_train_pkl,
                  pipeline=K360_train_pipeline,
                  load_interval= K360_train_interval,
                  used_cams = K360_selected_cam,
                  test_mode=False,
                  **K360_default)
K360_val = dict(type=K360_type,
                 data_root=K360_data_root,
                 ann_file=K360_val_pkl,
                 pipeline=K360_test_pipeline,
                 load_interval=K360_val_interval,
                 used_cams = K360_selected_cam,
                 test_mode=True,
                 **K360_default)

joint_train = dict(
        type='CustomConcatDataset',
        datasets=[argo2_train, nusc_train, waymo_train])
joint_val = dict(
        type='CustomConcatDataset',
        datasets=[argo2_val, nusc_val, waymo_val, 
                  lyft_val, kitti_val, K360_val])

dataloader_default = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False)
train_dataloader = dict(
    **dataloader_default,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=joint_train)
val_dataloader = dict(
    **dataloader_default,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=joint_val)
test_dataloader = val_dataloader

val_evaluator = dict(type = 'JointMetric',
                     per_location = True,
                     work_dir = work_dir,
                     brief_split = True)
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