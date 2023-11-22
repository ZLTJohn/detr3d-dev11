from projects.detr3d.custom_waymo_metric import JointMetric_bevdet
work_dirs = ['/home/zhenglt/mmdev11/detr3d-dev11/work_dirs_bevdet/NW_r50_mono_bs8_smallres_no_aug_syncBN',
'/home/zhenglt/mmdev11/detr3d-dev11/work_dirs_bevdet/NW_r50_mono_bs8_smallres_no_aug_syncBN_decoupledres',
'/home/zhenglt/mmdev11/detr3d-dev11/work_dirs_bevdet/NWL_r50_mono_bs8_smallres_no_aug_syncBN',
'/home/zhenglt/mmdev11/detr3d-dev11/work_dirs_bevdet/NWLA_r50_mono_bs8_smallres_no_aug_syncBN',
'/home/zhenglt/mmdev11/detr3d-dev11/work_dirs_bevdet/NWLAKK_r50_mono_bs8_smallres_no_aug_syncBN',
'/home/zhenglt/mmdev11/detr3d-dev11/work_dirs_bevdet/NWLAKK360_r50_mono_bs8_smallres_no_aug_syncBN']
for work_dir in work_dirs:
    metric = JointMetric_bevdet(work_dir = work_dir)
    metric.dataset_names = ['argoverse2','kitti','kitti-360','lyft','nuscenes','waymo']
    print(metric.get_best_from_file())