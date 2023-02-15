_base_ = [
    'mmdet3d::_base_/default_runtime.py',
    './base/detr3d_argo2.py'
]
custom_imports = dict(imports=['projects.detr3d'])
default_scope = 'mmdet3d'
point_cloud_range = [-150, -150, -2, 150, 150, 4]
num_views = 7
flip_front_cam = True
class_names = ['ARTICULATED_BUS', 'BICYCLE', 'BICYCLIST', 'BOLLARD', 
    'BOX_TRUCK', 'BUS', 'CONSTRUCTION_BARREL', 'CONSTRUCTION_CONE', 
    'DOG', 'LARGE_VEHICLE', 'MESSAGE_BOARD_TRAILER', 
    'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'MOTORCYCLE', 'MOTORCYCLIST',
    'PEDESTRIAN', 'REGULAR_VEHICLE', 'SCHOOL_BUS', 'SIGN', 'STOP_SIGN', 
    'STROLLER', 'TRUCK', 'TRUCK_CAB', 'VEHICULAR_TRAILER', 'WHEELCHAIR', 
    'WHEELED_DEVICE', 'WHEELED_RIDER']

dataset_type = 'Argo2Dataset'
data_root = 'data/argo2/'
work_dir = './work_dirs_dev11/argo2_fullres_n_train'
img_scale = (2048,1568)
# img_scale = (1024,800)
test_transforms = [
    dict(type='RandomResize3D',
         scale=img_scale,
         ratio_range=(1., 1.),
         keep_ratio=False)
]
train_transforms = [dict(type='PhotoMetricDistortion3D')] + test_transforms
# train_transforms = test_transforms

file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='Argo2LoadMultiViewImageFromFiles',
         to_float32=True,
         num_views=num_views),
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
    dict(type='Argo2LoadMultiViewImageFromFiles',
         to_float32=True,
         num_views=num_views),
    dict(type='filename2img_path'),  # fix it in ↑ via a PR
    dict(type='MultiViewWrapper', transforms=test_transforms),
    dict(type='Pack3DDetInputs', keys=['img'])
]

metainfo = dict(classes=class_names)
data_prefix = dict()
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
        ann_file='argo2_infos_train.pkl',
        # load_interval = 5,
        pipeline=train_pipeline,
        load_type='frame_based',
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        data_prefix=data_prefix,
        box_type_3d='LiDAR'))

val_dataloader = dict(batch_size=1,
                      num_workers=4,
                      persistent_workers=False,
                      drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(type=dataset_type,
                                   data_root=data_root,
                                   ann_file='debug_val.pkl',
                                   load_interval=5,
                                   load_type='frame_based',
                                   pipeline=test_pipeline,
                                   metainfo=metainfo,
                                   modality=input_modality,
                                   test_mode=True,
                                   data_prefix=data_prefix,
                                   box_type_3d='LiDAR'))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='Argo2Metric',
    result_path = 'debug/argo2_pred.feather')
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
                 val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(checkpoint=dict(
    type='CheckpointHook', interval=1, max_keep_ckpts=1, save_last=True))

vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')
resume_from = '/home/zhenglt/mmdev11/detr3d-dev11/work_dirs_dev11/argo2_fullres_n_train/epoch_17.pth'
resume = True