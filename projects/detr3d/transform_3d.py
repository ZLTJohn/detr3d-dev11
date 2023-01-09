from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class filename2img_path:

    def __call__(self, results):
        results['img_path'] = results['filename']
        return results

    def __repr__(self):
        return 'maybe we need to fix this bug'


import numpy as np


@TRANSFORMS.register_module()
class ProjectLabelToWaymoClass(object):

    def __init__(self,
                 class_names=None,
                 waymo_name=['Car', 'Pedestrian', 'Cyclist']):
        self.class_names = class_names
        self.waymo_name = waymo_name
        self.name_map = {
            'car': 'Car',
            'truck': 'Car',
            'construction_vehicle': 'Car',
            'bus': 'Car',
            'trailer': 'Car',
            'motorcycle': 'Car',
            'bicycle': 'Cyclist',
            'pedestrian': 'Pedestrian'
        }
        ind_N2W = []
        for i, n_name in enumerate(self.class_names):
            ind_N2W.append(waymo_name.index(self.name_map[n_name]))
        self.ind_N2W = np.array(ind_N2W)

    # class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    #                'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    # nusc2waymo = [0, 0, 0, 0, 0, -1, 0, 2, 1, -1]
    def __call__(self, results):
        if len(results['gt_labels_3d']) > 0:
            results['gt_labels_3d'] = self.ind_N2W[results['gt_labels_3d']]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
