_base_ = ['./detr3d_res101_gridmask.py']

custom_imports = dict(imports=['projects.detr3d'])
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675],
                    std=[57.375, 57.120, 58.395],
                    bgr_to_rgb=False)
# if torch.is_grad_enabled() and self.reducer._rebuild_buckets():
# RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by
# making sure all `forward` function outputs participate in calculating loss.
# If you already have done the above, then the distributed data parallel module wasn't able to locate the output tensors in the return value of your module's `forward` function. Please include the loss function and the structure of the return value of `forward` of your module when reporting this issue (e.g. list, dict, iterable).
# Parameter indices which did not receive grad for rank 2: 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283
# this means type='DETR3D' will be processed as 'mmdet3d.DETR3D'
default_scope = 'mmdet3d'
model = dict(type='DETR3D',
             use_grid_mask=True,
             data_preprocessor=dict(type='Det3DDataPreprocessor',
                                    **img_norm_cfg,
                                    pad_size_divisor=32),
             img_backbone=dict(
                 _delete_=True,
                 type='VoVNet',
                 spec_name='V-99-eSE',
                 norm_eval=True,
                 frozen_stages=1,
                 input_ch=3,
                 out_features=['stage2', 'stage3', 'stage4', 'stage5']),
             img_neck=dict(type='mmdet.FPN',
                           in_channels=[256, 512, 768, 1024],
                           out_channels=256,
                           start_level=0,
                           add_extra_convs='on_output',
                           num_outs=4,
                           relu_before_extra_convs=True))


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
load_from = 'ckpts/dd3d_det_final.pth'# TODO: dont forget to update it in the release version!
test_transforms = [
    dict(type='RandomResize3D',
         scale=(1600, 928),
         ratio_range=(1., 1.),
         keep_ratio=True)
]
train_transforms = [dict(type='PhotoMetricDistortion3D')] + test_transforms

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=6),
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
            ann_file='nuscenes_infos_trainval.pkl',
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
# mAP: 0.7103
# mATE: 0.5395
# mASE: 0.1455
# mAOE: 0.0719
# mAVE: 0.2233
# mAAE: 0.1862
# NDS: 0.7385
# Eval time: 107.3s
# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.706   0.569   0.116   0.033   0.261   0.202
# truck   0.737   0.483   0.120   0.034   0.195   0.208
# bus     0.760   0.463   0.108   0.028   0.296   0.240
# trailer 0.739   0.453   0.124   0.042   0.138   0.147
# construction_vehicle    0.710   0.513   0.178   0.085   0.139   0.329
# pedestrian      0.715   0.510   0.205   0.203   0.248   0.138
# motorcycle      0.692   0.560   0.149   0.089   0.357   0.218
# bicycle 0.673   0.643   0.171   0.081   0.152   0.008
# traffic_cone    0.691   0.569   0.172   nan     nan     nan
# barrier 0.681   0.633   0.112   0.052   nan     nan

# after fixing h,w bug in feature-sampling
# mAP: 0.8348
# mATE: 0.3225
# mASE: 0.1417
# mAOE: 0.0676
# mAVE: 0.2204
# mAAE: 0.1820
# NDS: 0.8240
# Eval time: 97.4s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.873   0.256   0.114   0.033   0.260   0.195
# truck   0.833   0.327   0.115   0.033   0.191   0.216
# bus     0.842   0.323   0.104   0.027   0.293   0.244
# trailer 0.779   0.394   0.116   0.041   0.136   0.123
# construction_vehicle    0.784   0.406   0.174   0.079   0.137   0.320
# pedestrian      0.806   0.380   0.203   0.181   0.244   0.135
# motorcycle      0.822   0.337   0.150   0.085   0.347   0.213
# bicycle 0.871   0.271   0.169   0.079   0.154   0.009
# traffic_cone    0.877   0.241   0.162   nan     nan     nan
# barrier 0.861   0.289   0.110   0.050   nan     nan
