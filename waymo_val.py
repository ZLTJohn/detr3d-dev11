import mmengine
from mmdet3d.evaluation.metrics import WaymoMetric
from mmdet3d.registry import METRICS
from mmdet3d.utils import register_all_modules, replace_ceph_backend

register_all_modules(init_default_scope=False)
val_evaluator = dict(
    type='WaymoMetric',
    ann_file='./data/waymo_dev1x/kitti_format/waymo_infos_val.pkl',
    waymo_bin_file='./data/waymo_dev1x/waymo_format/gt.bin',
    data_root='./data/waymo_dev1x/waymo_format',
    metric='LET_mAP',
    load_interval=5,
    convert_kitti_format=False)

metric = METRICS.build(val_evaluator)
metric.dataset_meta = dict(classes=['Car', 'Pedestrian', 'Cyclist'])
results = mmengine.load('debug/val.pkl')
print(metric.compute_metrics(results))
