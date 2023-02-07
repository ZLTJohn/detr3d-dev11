point_cloud_range = [-150, -150, -2, 150, 150, 4]
num_views = 7
class_names = ['ARTICULATED_BUS', 'BICYCLE', 'BICYCLIST', 'BOLLARD', 
    'BOX_TRUCK', 'BUS', 'CONSTRUCTION_BARREL', 'CONSTRUCTION_CONE', 
    'DOG', 'LARGE_VEHICLE', 'MESSAGE_BOARD_TRAILER', 
    'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'MOTORCYCLE', 'MOTORCYCLIST',
    'PEDESTRIAN', 'REGULAR_VEHICLE', 'SCHOOL_BUS', 'SIGN', 'STOP_SIGN', 
    'STROLLER', 'TRUCK', 'TRUCK_CAB', 'VEHICULAR_TRAILER', 'WHEELCHAIR', 
    'WHEELED_DEVICE', 'WHEELED_RIDER']
debug_vis_cfg = dict(debug_dir='debug/visualization',
                     gt_range=[0, 150],
                     pc_range=point_cloud_range,
                     vis_count=100,
                     debug_name='dev1x_watch')
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    bgr_to_rgb=True)
model_wrapper_cfg = dict(type = 'CustomMMDDP', static_graph = True)
model = dict(
    type='DETR3D',
    use_grid_mask=True,
    # debug_vis_cfg=debug_vis_cfg,
    data_preprocessor=dict(type='Det3DDataPreprocessor',
                           **img_norm_cfg,
                           pad_size_divisor=32),
    # PGD will also fail to load some weights in pytorch style
    # it seems to be caused by mmdet itself
    # however caffe and pytorch style have only order difference.
    img_backbone=dict(type='mmdet.ResNet',
                      depth=101,
                      num_stages=4,
                      out_indices=(0, 1, 2, 3),
                      frozen_stages=1,
                      with_cp=True,
                      norm_cfg=dict(type='BN2d', requires_grad=False),
                      norm_eval=True,
                      style='pytorch',
                      dcn=dict(type='DCNv2',
                               deform_groups=1,
                               fallback_on_stride=False),
                      stage_with_dcn=(False, False, True, True),
                      init_cfg=dict(
                          type='Pretrained',
                          checkpoint='torchvision://resnet101')),
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
        num_classes=len(class_names),
        in_channels=256,
        code_size=8,
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
                            type='MultiheadAttention',
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
                        num_classes=len(class_names)),
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