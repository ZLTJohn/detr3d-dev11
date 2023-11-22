_base_ = [
          'mmdet3d::_base_/default_runtime.py',]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 4.0]
# For nuScenes we usually do 10-class detection
waymo_class_names = ['Car', 'Pedestrian', 'Cyclist']
nusc_class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
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
    'BICYCLE': 'Cyclist',   # TO REMOVE
    'BICYCLIST': 'Cyclist'
}
lyft_class_names = [
    'car', 'truck', 'bus', 'emergency_vehicle', 'other_vehicle', 
    'motorcycle', 'bicycle', 'pedestrian', 'animal'
]
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
kitti_class_names = ['Pedestrian','Cyclist','Car','Van','Truck',
                     'Person_sitting','Tram','Misc']
kitti_name_map = {
    'Pedestrian': 'Pedestrian',
    'Cyclist': 'Cyclist',
    'Car': 'Car',
    'Van': 'Car',
    'Truck': 'Car',
    'Person_sitting': 'Pedestrian',
    'Tram': 'Car'
}
K360_class_names = ['bicycle', 'box', 'bridge', 'building', 'bus', 'car',
           'caravan', 'garage', 'lamp', 'motorcycle', 'person', 
           'pole', 'rider', 'smallpole', 'stop', 'traffic light', 
           'traffic sign', 'trailer', 'train', 'trash bin', 'truck', 
           'tunnel', 'unknown construction', 'unknown object', 
           'unknown vehicle', 'vending machine']
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
custom_imports = dict(imports=['projects.bevdet'])
default_scope = 'mmdet3d'
waymo_type = 'CustomWaymo'
waymo_root = '/localdata_ssd/waymo_dev1x/'
waymo_train_pkl = 'waymo_infos_train_2Hz_mono_front.pkl'
waymo_train_interval = 1    # 2Hz_part means interval = 5x3
waymo_val_pkl = 'waymo_infos_val_2Hz_part_mono_front.pkl'
waymo_val_interval = 1
waymo_h = 640
waymo_w = 960
waymo_h_crop = 436
waymo_w_crop = 960
nusc_type = 'CustomNusc'
nusc_root = '/localdata_ssd/nusc_dev1x/'
nusc_train_pkl = 'nuscenes_infos_train_mono_front.pkl'
nusc_val_pkl = 'nuscenes_infos_val_part_mono_front.pkl'
nusc_h = 450
nusc_w = 800
nusc_h_crop = 307
nusc_w_crop = 800
argo2_type = 'Argo2Dataset'
argo2_data_root = '/localdata_ssd/argo2_dev1x/'
argo2_train_pkl = 'argo2_infos_train_2Hz_mono_front.pkl'  
argo2_train_interval = 1    # 2Hz_part means interval = 5x3
argo2_val_pkl = 'argo2_infos_val_2Hz_part_mono_front.pkl'
argo2_val_interval = 1
argo2_h = 1024
argo2_w = 775
argo2_h_crop = 350 # start at 800//2
argo2_w_crop = 775
argo2_h_start = 400
argo2_h_crop_rel_pos = argo2_h_start / (argo2_h - argo2_h_crop)
lyft_type = 'CustomLyft'
lyft_data_root = '/localdata_ssd/lyft_dev1x/'
lyft_train_pkl = 'lyft_infos_train_mono_front.pkl' 
lyft_train_interval = 1
lyft_val_pkl = 'lyft_infos_val_mono_front.pkl'
lyft_val_interval = 2
lyft_small_h = 1024//2
lyft_small_w = 1224//2
lyft_small_h_crop = 400//2    # original 400~800
lyft_small_w_crop = lyft_small_w
lyft_small_h_start = 400//2
lyft_small_h_crop_rel_pos = lyft_small_h_start / (lyft_small_h - lyft_small_h_crop)
lyft_focal_interval_small = [0,1000]
lyft_big_h = 1080//2
lyft_big_w = 1920//2
lyft_big_h_crop = 440//2    # original 440~880
lyft_big_w_crop = lyft_big_w
lyft_big_h_start = 440//2
lyft_big_h_crop_rel_pos = lyft_big_h_start / (lyft_big_h - lyft_big_h_crop)
lyft_focal_interval_big = [1010,2000]
kitti_type = 'CustomKitti'
kitti_data_root = '/localdata_ssd/kitti_dev1x/'
kitti_train_pkl = 'kitti_infos_train.pkl'
kitti_train_interval = 1
kitti_val_pkl = 'kitti_infos_val.pkl'
kitti_val_interval = 1
kitti_num_views = 1
kitti_h = 370//2
kitti_w = 1224//2
kitti_h_crop = kitti_h
kitti_w_crop = kitti_w
K360_type = 'Kitti360Dataset'
K360_data_root = '/localdata_ssd/kitti-360_dev1x/'
K360_train_pkl = 'kitti360_infos_train.pkl' # 40000 frame
K360_train_interval = 1
K360_val_pkl = 'kitti360_infos_val.pkl' # 10000 frame
K360_val_interval = 5
K360_num_views = 1
K360_selected_cam = 'CAM0'
K360_h = 376//2
K360_w = 1408//2
K360_h_crop = K360_h
K360_w_crop = K360_w
pad_h = 448
pad_w = 960
focal_length=1000
work_dir = 'work_dirs_bevdet_submission/1.00N_fx1000_d140_testonly'
resume = True
# Model
# lss: h w
# resize: w h
# crop: h w
# pad: h w
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 4, 9],
    'depth': [1.0, 140.0, 1.0],
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
        accelerate=False,
        focal_length = focal_length),
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

bevdet_input_default = [
    dict(type='filename2img_path'),
    dict(type='GetBEVDetInputs'),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputsBEVDet', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

bev_aug = [dict(type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
           dict(type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5)]

waymo_img_aug = [dict(type='MultiViewWrapper',
    transforms=[
        dict(type='RandomResize', ratio_range=(0.8,1.2), scale=(waymo_w,waymo_h)),
        dict(type='RangeLimitedRandomCrop',
            relative_x_offset_range=(0.5, 1.0),
            relative_y_offset_range=(1.0, 1.0),
            crop_size=(waymo_h_crop, waymo_w_crop)),
        dict(type='Pad', size=(pad_w, pad_h)),
        dict(type='RandomFlip', prob=0.5),
        dict(type='RandomRotate', range=(-5.4, 5.4), 
            img_border_value=0, level=1, prob=0.0)],
    collected_keys=['scale_factor', 'crop', 'pad_shape', 'flip', 'rotate'])]
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
    *bev_aug,
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    *waymo_img_aug,
    *bevdet_input_default
]
waymo_test_pipeline = [
    dict(type='evalann2ann'),
    *waymo_input_default,
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    *waymo_img_no_aug,
    *bevdet_input_default
    ]
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

nusc_img_aug = [dict(type='MultiViewWrapper',
    transforms=[
        dict(type='RandomResize', ratio_range=(0.8,1.2), scale=(nusc_w,nusc_h)),
        dict(type='RangeLimitedRandomCrop',
            relative_x_offset_range=(0.5, 1.0),
            relative_y_offset_range=(1.0, 1.0),
            crop_size=(nusc_h_crop, nusc_w_crop)),
        dict(type='Pad', size=(pad_w, pad_h)),
        dict(type='RandomFlip', prob=0.5),
        dict(type='RandomRotate', range=(-5.4, 5.4), 
            img_border_value=0, level=1, prob=0.0)],
    collected_keys=['scale_factor', 'crop', 'pad_shape', 'flip', 'rotate'])]
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
nusc_bev_aug = [dict(type='GlobalRotScaleTrans',
                rot_range=[-0.3925-1.5708, 0.3925-1.5708],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
           dict(type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5)]
nusc_bev_noaug = [dict(type='GlobalRotScaleTrans',rot_range=[-1.5708,-1.5708],
                       scale_ratio_range=[1.0,1.0],
                       translation_std=[0, 0, 0]),]
nusc_train_pipeline = [
    *nusc_input_default,
    *nusc_bev_aug,
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    *nusc_img_aug,
    *bevdet_input_default
]
nusc_test_pipeline = [
    dict(type='evalann2ann'),
    *nusc_input_default,
    *nusc_bev_noaug,
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

# Argoverse2-----------------
argo2_img_aug = [dict(type='MultiViewWrapper',
    transforms=[
        dict(type='RandomResize', ratio_range=(0.8,1.2), scale=(argo2_w,argo2_h)),
        dict(type='RangeLimitedRandomCrop',
            relative_x_offset_range=(0.5, 1.0),# x for w
            relative_y_offset_range=(argo2_h_crop_rel_pos, argo2_h_crop_rel_pos),# y for h
            crop_size=(argo2_h_crop, argo2_w_crop)),
        dict(type='Pad', size=(pad_w, pad_h)),
        dict(type='RandomFlip', prob=0.5),
        dict(type='RandomRotate', range=(-5.4, 5.4), 
            img_border_value=0, level=1, prob=0.0)],
    collected_keys=['scale_factor', 'crop', 'pad_shape', 'flip', 'rotate'])]
argo2_img_no_aug = [dict(type='MultiViewWrapper',
    transforms=[
        dict(type='RandomResize', ratio_range=(1.0, 1.0), scale=(argo2_w,argo2_h)),
        dict(type='RangeLimitedRandomCrop',
            relative_x_offset_range=(1.0, 1.0),# x for w
            relative_y_offset_range=(argo2_h_crop_rel_pos, argo2_h_crop_rel_pos),# y for h
            crop_size=(argo2_h_crop, argo2_w_crop)),
        dict(type='Pad', size=(pad_w, pad_h)),
        dict(type='RandomFlip', prob=0.),
        dict(type='RandomRotate', range=(0.,0.), 
            img_border_value=0, level=1, prob=0.0)],
    collected_keys=['scale_factor', 'crop', 'pad_shape', 'flip', 'rotate'])]
argo2_input_default = [
    dict(type='Argo2LoadPointsFromFile',coord_type='LIDAR', load_dim=3, use_dim=3),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ProjectLabelToWaymoClass', class_names = argo2_class_names, name_map = argo2_name_map),
]
argo2_train_pipeline = [
    *argo2_input_default,
    *bev_aug,
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    *argo2_img_aug,
    *bevdet_input_default
]
argo2_test_pipeline = [
    dict(type='evalann2ann'),
    *argo2_input_default,
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    *argo2_img_no_aug,
    *bevdet_input_default
    ]
argo2_metainfo = dict(classes=argo2_class_names)
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
# lyft-----------------
lyft_small_img_aug = [dict(type='MultiViewWrapper',
    transforms=[
        dict(type='RandomResize', ratio_range=(0.8,1.2), scale=(lyft_small_w,lyft_small_h)),
        dict(type='RangeLimitedRandomCrop',
            relative_x_offset_range=(0.5, 1.0),# x for w
            relative_y_offset_range=(lyft_small_h_crop_rel_pos, lyft_small_h_crop_rel_pos),# y for h
            crop_size=(lyft_small_h_crop, lyft_small_w_crop)),
        dict(type='Pad', size=(pad_w, pad_h)),
        dict(type='RandomFlip', prob=0.5),
        dict(type='RandomRotate', range=(-5.4, 5.4), 
            img_border_value=0, level=1, prob=0.0)],
    collected_keys=['scale_factor', 'crop', 'pad_shape', 'flip', 'rotate'])]
lyft_small_img_no_aug = [dict(type='MultiViewWrapper',
    transforms=[
        dict(type='RandomResize', ratio_range=(1.0, 1.0), scale=(lyft_small_w,lyft_small_h)),
        dict(type='RangeLimitedRandomCrop',
            relative_x_offset_range=(1.0, 1.0),# x for w
            relative_y_offset_range=(lyft_small_h_crop_rel_pos, lyft_small_h_crop_rel_pos),# y for h
            crop_size=(lyft_small_h_crop, lyft_small_w_crop)),
        dict(type='Pad', size=(pad_w, pad_h)),
        dict(type='RandomFlip', prob=0.),
        dict(type='RandomRotate', range=(0.,0.), 
            img_border_value=0, level=1, prob=0.0)],
    collected_keys=['scale_factor', 'crop', 'pad_shape', 'flip', 'rotate'])]
lyft_bev_aug = [dict(type='GlobalRotScaleTrans',
                rot_range=[-0.3925-3.1416, 0.3925-3.1416],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
           dict(type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5)]
lyft_bev_noaug = [dict(type='GlobalRotScaleTrans',rot_range=[-3.1416,-3.1416],
                       scale_ratio_range=[1.0,1.0],
                       translation_std=[0, 0, 0]),]
lyft_input_default = [
    dict(type='LoadPointsFromFile',coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ProjectLabelToWaymoClass', class_names = lyft_class_names, name_map = lyft_name_map)
]
lyft_small_train_pipeline = [
    *lyft_input_default,
    *lyft_bev_aug,
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    *lyft_small_img_aug,
    *bevdet_input_default
]

lyft_small_test_pipeline = [
    dict(type='evalann2ann'),
    *lyft_input_default,
    *lyft_bev_noaug,
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    *lyft_small_img_no_aug,
    *bevdet_input_default
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
lyft_small_train = dict(type=lyft_type,
                 data_root=lyft_data_root,
                 focal_interval = lyft_focal_interval_small,
                 ann_file=lyft_train_pkl,
                 pipeline=lyft_small_train_pipeline,
                 load_interval = lyft_train_interval,
                 test_mode=False,
                 **lyft_default)
lyft_small_val = dict(type=lyft_type,
                data_root=lyft_data_root,
                focal_interval = lyft_focal_interval_small,
                ann_file=lyft_val_pkl,
                pipeline=lyft_small_test_pipeline,
                load_interval = lyft_val_interval,
                test_mode=True,
                **lyft_default)
# lyft_big--------------------
lyft_big_img_aug = [dict(type='MultiViewWrapper',
    transforms=[
        dict(type='RandomResize', ratio_range=(0.8,1.2), scale=(lyft_big_w,lyft_big_h)),
        dict(type='RangeLimitedRandomCrop',
            relative_x_offset_range=(0.5, 1.0),# x for w
            relative_y_offset_range=(lyft_big_h_crop_rel_pos, lyft_big_h_crop_rel_pos),# y for h
            crop_size=(lyft_big_h_crop, lyft_big_w_crop)),
        dict(type='Pad', size=(pad_w, pad_h)),
        dict(type='RandomFlip', prob=0.5),
        dict(type='RandomRotate', range=(-5.4, 5.4), 
            img_border_value=0, level=1, prob=0.0)],
    collected_keys=['scale_factor', 'crop', 'pad_shape', 'flip', 'rotate'])]
lyft_big_img_no_aug = [dict(type='MultiViewWrapper',
    transforms=[
        dict(type='RandomResize', ratio_range=(1.0, 1.0), scale=(lyft_big_w,lyft_big_h)),
        dict(type='RangeLimitedRandomCrop',
            relative_x_offset_range=(1.0, 1.0),# x for w
            relative_y_offset_range=(lyft_big_h_crop_rel_pos, lyft_big_h_crop_rel_pos),# y for h
            crop_size=(lyft_big_h_crop, lyft_big_w_crop)),
        dict(type='Pad', size=(pad_w, pad_h)),
        dict(type='RandomFlip', prob=0.),
        dict(type='RandomRotate', range=(0.,0.), 
            img_border_value=0, level=1, prob=0.0)],
    collected_keys=['scale_factor', 'crop', 'pad_shape', 'flip', 'rotate'])]
lyft_input_default = [
    dict(type='LoadPointsFromFile',coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ProjectLabelToWaymoClass', class_names = lyft_class_names, name_map = lyft_name_map)
]
lyft_big_train_pipeline = [
    *lyft_input_default,
    *lyft_bev_aug,
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    *lyft_big_img_aug,
    *bevdet_input_default
]

lyft_big_test_pipeline = [
    dict(type='evalann2ann'),
    *lyft_input_default,
    *lyft_bev_noaug,
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    *lyft_big_img_no_aug,
    *bevdet_input_default
]
lyft_big_train = dict(type=lyft_type,
                 data_root=lyft_data_root,
                 focal_interval = lyft_focal_interval_big,
                 ann_file=lyft_train_pkl,
                 pipeline=lyft_big_train_pipeline,
                 load_interval = lyft_train_interval,
                 test_mode=False,
                 **lyft_default)
lyft_big_val = dict(type=lyft_type,
                data_root=lyft_data_root,
                focal_interval = lyft_focal_interval_big,
                ann_file=lyft_val_pkl,
                pipeline=lyft_big_test_pipeline,
                load_interval = lyft_val_interval,
                test_mode=True,
                **lyft_default)
# kitti-----------------
kitti_img_aug = [dict(type='MultiViewWrapper',
    transforms=[
        dict(type='RandomResize', ratio_range=(0.8,1.2), scale=(kitti_w,kitti_h)),
        dict(type='RangeLimitedRandomCrop',
            relative_x_offset_range=(0.5, 1.0),# x for w
            relative_y_offset_range=(1.0, 1.0),# y for h
            crop_size=(kitti_h_crop, kitti_w_crop)),
        dict(type='Pad', size=(pad_w, pad_h)),
        dict(type='RandomFlip', prob=0.5),
        dict(type='RandomRotate', range=(-5.4, 5.4), 
            img_border_value=0, level=1, prob=0.0)],
    collected_keys=['scale_factor', 'crop', 'pad_shape', 'flip', 'rotate'])]
kitti_img_no_aug = [dict(type='MultiViewWrapper',
    transforms=[
        dict(type='RandomResize', ratio_range=(1.0, 1.0), scale=(kitti_w,kitti_h)),
        dict(type='RangeLimitedRandomCrop',
            relative_x_offset_range=(1.0, 1.0),# x for w
            relative_y_offset_range=(1.0, 1.0),# y for h
            crop_size=(kitti_h_crop, kitti_w_crop)),
        dict(type='Pad', size=(pad_w, pad_h)),
        dict(type='RandomFlip', prob=0.),
        dict(type='RandomRotate', range=(0.,0.), 
            img_border_value=0, level=1, prob=0.0)],
    collected_keys=['scale_factor', 'crop', 'pad_shape', 'flip', 'rotate'])]
kitti_input_default = [
    dict(type='LoadPointsFromFile',coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ProjectLabelToWaymoClass', class_names = kitti_class_names, name_map = kitti_name_map)
]
kitti_train_pipeline = [
    *kitti_input_default,
    *bev_aug,
    dict(type='Argo2LoadMultiViewImageFromFiles', flip_front_cam=False, to_float32=True, num_views=kitti_num_views),
    *kitti_img_aug,
    *bevdet_input_default
]
kitti_test_pipeline = [
    dict(type='evalann2ann'),
    *kitti_input_default,
    dict(type='Argo2LoadMultiViewImageFromFiles', flip_front_cam=False, to_float32=True, num_views=kitti_num_views),
    *kitti_img_no_aug,
    *bevdet_input_default
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
# K360-----------------
K360_img_aug = [dict(type='MultiViewWrapper',
    transforms=[
        dict(type='RandomResize', ratio_range=(0.8,1.2), scale=(K360_w,K360_h)),
        dict(type='RangeLimitedRandomCrop',
            relative_x_offset_range=(0.5, 1.0),# x for w
            relative_y_offset_range=(1.0, 1.0),# y for h
            crop_size=(K360_h_crop, K360_w_crop)),
        dict(type='Pad', size=(pad_w, pad_h)),
        dict(type='RandomFlip', prob=0.5),
        dict(type='RandomRotate', range=(-5.4, 5.4), 
            img_border_value=0, level=1, prob=0.0)],
    collected_keys=['scale_factor', 'crop', 'pad_shape', 'flip', 'rotate'])]
K360_img_no_aug = [dict(type='MultiViewWrapper',
    transforms=[
        dict(type='RandomResize', ratio_range=(1.0, 1.0), scale=(K360_w,K360_h)),
        dict(type='RangeLimitedRandomCrop',
            relative_x_offset_range=(1.0, 1.0),# x for w
            relative_y_offset_range=(1.0, 1.0),# y for h
            crop_size=(K360_h_crop, K360_w_crop)),
        dict(type='Pad', size=(pad_w, pad_h)),
        dict(type='RandomFlip', prob=0.),
        dict(type='RandomRotate', range=(0.,0.), 
            img_border_value=0, level=1, prob=0.0)],
    collected_keys=['scale_factor', 'crop', 'pad_shape', 'flip', 'rotate'])]
K360_input_default = [
    dict(type='LoadPointsFromFile',coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ProjectLabelToWaymoClass', class_names = K360_class_names, name_map = K360_name_map)
]
K360_train_pipeline = [
    *K360_input_default,
    *bev_aug,
    dict(type='Argo2LoadMultiViewImageFromFiles', flip_front_cam=False, to_float32=True, num_views=K360_num_views),
    *K360_img_aug,
    *bevdet_input_default
]
K360_test_pipeline = [
    dict(type='evalann2ann'),
    *K360_input_default,
    dict(type='Argo2LoadMultiViewImageFromFiles', flip_front_cam=False, to_float32=True, num_views=K360_num_views),
    *K360_img_no_aug,
    *bevdet_input_default
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
# train&&test

joint_train = dict(
        type='CustomConcatDataset',
        datasets=[nusc_train])
joint_val = dict(
        type='CustomConcatDataset',
        datasets=[argo2_val,nusc_test,lyft_big_val, lyft_small_val,
                  kitti_val,K360_val,waymo_test])
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
val_evaluator = [dict(type = 'JointMetric_bevdet',
                     work_dir = work_dir,
                     brief_split = True),
                dict(type = 'JointMetric_bevdet',
                     bev_mAP = True,
                     work_dir = work_dir,
                     brief_split = True),
                     ]
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
