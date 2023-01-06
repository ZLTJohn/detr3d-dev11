_base_ = [
    # '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    # '/home/zhenglt/mmdev11/mmdet3d-latest/configs/_base_/datasets/nus-3d.py',
    './waymo_pretrain_NusRes_2x_schedule.py'
]
class_names = [  # 不确定sign类别是否叫sign
    'Car', 'Pedestrian', 'Cyclist'
]
# cloud range accordingly
point_cloud_range = [-35, -75, -2, 75, 75, 4]
num_views = 5
dataset_type = 'WaymoDataset'
# data_root = '/localdata_ssd/waymo_ssd/kitti_format/'
data_root = 'data/waymo_dev1x/kitti_format/'
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
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',
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
            box_type_3d='LiDAR')))

val_dataloader = dict(dataset=dict(data_root=data_root))

total_epochs = 24
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
                 val_interval=2)

work_dir = './work_dirs_dev11/NusRes_cbgs'
load_from = None
# resume_from = './work_dirs_dev11/NusRes_cbgs/last_checkpoint'
resume = True