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
# point_cloud_range = [0, -40, -3, 51.2, 40, 1]
# point_cloud_range = [0, -51.2, -5.0, 51.2, 51.2, 4.0]
point_cloud_range = [0, -51.2, -5.0, 51.2, 51.2, 4.0]
K360_class_names = ['bicycle', 'box', 'bridge', 'building', 'bus', 'car',
           'caravan', 'garage', 'lamp', 'motorcycle', 'person', 
           'pole', 'rider', 'smallpole', 'stop', 'traffic light', 
           'traffic sign', 'trailer', 'train', 'trash bin', 'truck', 
           'tunnel', 'unknown construction', 'unknown object', 
           'unknown vehicle', 'vending machine']
waymo_class_names = ['Car', 'Pedestrian', 'Cyclist']

num_class = 3
# K360_num_views = 2
K360_num_views = 1
K360_selected_cam = 'CAM0'
argo2_num_views = 7
img_size_K360 = (1408, 376)
evaluation_interval = 12 # epochs
# load_from = 'ckpts/'
K360_type = 'Kitti360Dataset'
K360_data_root = 'data/kitti-360/'
K360_train_pkl = 'kitti360_infos_train.pkl'
K360_train_interval = 5    # 2Hz_part means interval = 5x3
K360_val_pkl = 'kitti360_infos_val.pkl'
K360_val_interval = 1

# load_interval_factor = load_interval_type['part']
input_modality = dict(use_lidar=True, # True if debug_vis
                      use_camera=True)
work_dir = './work_dirs_extended/debug_kitti360_part'
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
                     vis_count=200,
                     debug_name='debug')
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
                             waymo_with_nuscene = True,
                             waymo_with_argo2 = True,
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
    dict(type='MultiViewWrapper', transforms=K360_test_transforms),
]
K360_train_pipeline = K360_pipeline_default + [
    dict(type='MultiViewWrapper', transforms=[dict(type='PhotoMetricDistortion3D')]),
    dict(type='Pack3DDetInputsExtra', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
K360_test_pipeline = [dict(type='evalann2ann')] + K360_pipeline_default + [
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

argnuway_train = dict(
        type='CustomConcatDataset',
        datasets=[K360_train])
argnuway_val = dict(
        type='CustomConcatDataset',
        datasets=[K360_val])

dataloader_default = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False)
train_dataloader = dict(
    **dataloader_default,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=K360_train)
val_dataloader = dict(
    **dataloader_default,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=argnuway_val)
test_dataloader = val_dataloader

# val_evaluator = dict(
#     type='KittiMetric',
#     ann_file=K360_data_root + 'kitti_infos_val.pkl',
#     metric='bbox')
val_evaluator = dict(type = 'JointMetric',
                     per_location = True,
                     brief_metric = False)
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