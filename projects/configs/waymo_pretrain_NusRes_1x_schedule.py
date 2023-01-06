_base_ = [
    # '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    # '/home/zhenglt/mmdev11/mmdet3d-latest/configs/_base_/datasets/nus-3d.py',
    './waymo_pretrain_NusRes_2x_schedule.py'
]

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
                 val_interval=4)

work_dir = './work_dirs_dev11/NusResEpoch1x'
load_from = None
# resume_from = '/home/zhenglt/mmdev11/detr3d-dev11/work_dirs_dev11/NusResEpoch1x/epoch_20.pth'
resume = True