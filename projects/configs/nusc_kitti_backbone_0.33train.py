_base_ = ['./detr3d_res101_gridmask.py']
custom_imports = dict(imports=['projects.detr3d'])

train_dataloader = dict(dataset=dict(type='CustomNusc', load_interval=3))
test_dataloader = dict(dataset=dict(ann_file='debug_val.pkl'))

# val_evaluator = dict(_delete_=True, type = 'CustomWaymoMetric',is_waymo_gt = False, is_waymo_pred = True)
train_cfg = dict(val_interval=4)
load_from = 'ckpts/pgd_kitti_r101_backbone.pth'
work_dir = './work_dirs/Nusc_kitti_backbone_0.33train'

# test_evaluator = val_evaluator
# resume_from = ''
# resume = True