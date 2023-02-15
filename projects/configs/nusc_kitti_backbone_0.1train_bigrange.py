_base_ = ['./detr3d_res101_gridmask.py']
custom_imports = dict(imports=['projects.detr3d'])

train_dataloader = dict(dataset=dict(type='CustomNusc', load_interval=10))
test_dataloader = dict(dataset=dict(ann_file='debug_val.pkl'))

# val_evaluator = dict(_delete_=True, type = 'CustomWaymoMetric',is_waymo_gt = False, is_waymo_pred = True)
train_cfg = dict(val_interval=8)
load_from = 'ckpts/pgd_kitti_r101_backbone.pth'
work_dir = './work_dirs/Nusc_kitti_backbone_0.1train'

# test_evaluator = val_evaluator
# resume_from = ''
# resume = True

# mAP: 0.2641
# mATE: 0.8947
# mASE: 0.2927
# mAOE: 0.7136
# mAVE: 1.2215
# mAAE: 0.4204
# NDS: 0.2999
# Eval time: 125.3s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.455   0.655   0.157   0.141   2.083   0.489
# truck   0.208   0.941   0.240   0.258   1.315   0.385
# bus     0.238   0.935   0.245   0.281   2.328   0.602
# trailer 0.056   1.236   0.279   0.719   0.555   0.178
# construction_vehicle    0.030   1.155   0.531   1.272   0.115   0.361
# pedestrian      0.365   0.784   0.298   1.017   0.846   0.749
# motorcycle      0.231   0.854   0.276   1.039   1.834   0.444
# bicycle 0.242   0.819   0.288   1.430   0.695   0.155
# traffic_cone    0.429   0.713   0.322   nan     nan     nan
# barrier 0.389   0.854   0.292   0.264   nan     nan


# without pretrain neck: TO BE DONE!!!! @GPU38
# mAP: 0.1734
# mATE: 1.0179
# mASE: 0.3181
# mAOE: 1.1025
# mAVE: 1.2037
# mAAE: 0.4129
# NDS: 0.2136
# Eval time: 135.8s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.336   0.796   0.171   0.563   2.036   0.488
# truck   0.109   1.042   0.272   0.856   1.388   0.394
# bus     0.105   1.118   0.284   0.585   2.708   0.649
# trailer 0.023   1.353   0.313   1.434   0.513   0.145
# construction_vehicle    0.008   1.279   0.533   1.488   0.138   0.374
# pedestrian      0.286   0.858   0.299   1.522   0.845   0.741
# motorcycle      0.113   0.970   0.339   1.623   1.475   0.390
# bicycle 0.122   0.983   0.325   1.551   0.527   0.123
# traffic_cone    0.333   0.805   0.354   nan     nan     nan
# barrier 0.299   0.974   0.289   0.300   nan     nan