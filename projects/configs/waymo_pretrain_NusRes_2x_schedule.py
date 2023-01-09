_base_ = [
    # '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    # '/home/zhenglt/mmdev11/mmdet3d-latest/configs/_base_/datasets/nus-3d.py',
    'mmdet3d::_base_/default_runtime.py',
    './base/detr3d_waymo.py'
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

dataset_type = 'WaymoDataset'
# data_root = '/localdata_ssd/waymo_ssd/kitti_format/'
data_root = 'data/waymo_dev1x/kitti_format/'
# data_root = 'data/waymo_v2_png/kitti_format/'
img_scale = (1066, 1600)
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
    dict(type='filename2img_path'),  #fix it in ↑ via a PR
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
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='waymo_infos_train.pkl',
        load_interval=5,
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
    # type = 'CustomWaymoMetric'
    type='WaymoMetric',
    ann_file='./data/waymo_dev1x/kitti_format/waymo_infos_val.pkl',
    load_interval=100,
    waymo_bin_file='./data/waymo_dev1x/waymo_format/gt.bin',
    data_root='./data/waymo_dev1x/waymo_format',
    metric='LET_mAP'
    )
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
    clip_grad=dict(max_norm=35, norm_type=2),
)

total_epochs = 48
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
         end=total_epochs,
         T_max=total_epochs,
         eta_min_ratio=1e-3) 
]

train_cfg = dict(type='EpochBasedTrainLoop',
                 max_epochs=total_epochs,
                 val_interval=4)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(checkpoint=dict(
    type='CheckpointHook', interval=1, max_keep_ckpts=1, save_last=True))
load_from = 'ckpts/waymo_pretrain_fullres_pgd_mv_8gpu_for_detr3d_backbone_statedict_only.pth'

vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')
work_dir = './work_dirs_dev11/NusResEpoch2x'