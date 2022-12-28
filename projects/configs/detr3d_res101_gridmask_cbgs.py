_base_ = ['./detr3d_res101_gridmask.py']

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
train_transforms = test_transforms

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=6),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True,
         with_label_3d=True,
         with_attr_label=False),
    dict(type='MultiViewWrapper', transforms=train_transforms),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
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
    _delete_ = True,
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
            ann_file='nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            load_type='frame_based',
            metainfo=metainfo,
            modality=input_modality,
            test_mode=False,
            data_prefix=data_prefix,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')))

# before fixing h,w bug in feature-sampling
# mAP: 0.3405
# mATE: 0.7516
# mASE: 0.2688
# mAOE: 0.3750
# mAVE: 0.8621
# mAAE: 0.2080
# NDS: 0.4237
# Eval time: 124.2s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.513   0.590   0.153   0.066   0.932   0.197
# truck   0.280   0.771   0.206   0.092   1.001   0.237
# bus     0.356   0.860   0.193   0.131   1.826   0.370
# trailer 0.179   1.110   0.234   0.564   0.950   0.159
# construction_vehicle    0.080   0.988   0.444   0.947   0.120   0.330
# pedestrian      0.397   0.683   0.305   0.532   0.469   0.200
# motorcycle      0.338   0.712   0.255   0.359   1.194   0.158
# bicycle 0.256   0.657   0.284   0.583   0.405   0.014
# traffic_cone    0.508   0.542   0.323   nan     nan     nan
# barrier 0.501   0.602   0.292   0.100   nan     nan

# after fixing h,w bug in feature-sampling
# mAP: 0.3493
# mATE: 0.7162
# mASE: 0.2682
# mAOE: 0.3795
# mAVE: 0.8417
# mAAE: 0.1996
# NDS: 0.4341
# Eval time: 128.7s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.542   0.533   0.151   0.064   0.954   0.193
# truck   0.285   0.774   0.208   0.093   1.016   0.239
# bus     0.363   0.796   0.192   0.137   1.842   0.379
# trailer 0.167   1.075   0.236   0.610   0.718   0.094
# construction_vehicle    0.081   0.970   0.438   0.914   0.114   0.337
# pedestrian      0.400   0.684   0.306   0.531   0.474   0.201
# motorcycle      0.337   0.684   0.257   0.383   1.203   0.143
# bicycle 0.261   0.631   0.280   0.582   0.411   0.012
# traffic_cone    0.531   0.478   0.324   nan     nan     nan
# barrier 0.525   0.536   0.291   0.102   nan     nan
