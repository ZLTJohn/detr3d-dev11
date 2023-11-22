class zlt:
    def __init__(self) -> None:
        self.sb=1
    def forward(self):
        from mmdet3d.evaluation.metrics.waymo_let_metric import \
                compute_waymo_let_metric
        gt_file = '/home/zhenglt/mmdev11/detr3d-dev11/data/waymo_dev1x/waymo_format/gt_validation_subset_100.bin'
        pred_file = '/home/zhenglt/mmdev11/detr3d-dev11/results/ap4990_subset.bin'
        ap_dict = compute_waymo_let_metric(gt_file, pred_file)

a = zlt()
# a.forward()
breakpoint()