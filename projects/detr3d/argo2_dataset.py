# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Union

import numpy as np
from av2.evaluation.detection.constants import CompetitionCategories
from mmdet3d.datasets.det3d_dataset import Det3DDataset
from mmdet3d.datasets.kitti_dataset import KittiDataset
from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes

CLASSES = [x.value for x in CompetitionCategories]
CAM_NAMES = [
    'ring_front_center',
    'ring_front_right',
    'ring_front_left',
    'ring_rear_right',
    'ring_rear_left',
    'ring_side_right',
    'ring_side_left',
    # 'stereo_front_left', 'stereo_front_right',
]


@DATASETS.register_module()
class Argo2Dataset(KittiDataset):
    METAINFO = {'classes': tuple(x.value for x in CompetitionCategories)}

    def __init__(
            self,
            data_root: str,
            ann_file: str,
            data_prefix: dict = {},  # too many cams
            pipeline: List[Union[dict, Callable]] = [],
            modality: dict = dict(use_lidar=True, use_camera=True),
            default_cam_key: str = None,
            box_type_3d: str = 'LiDAR',
            load_type: str = 'frame_based',
            filter_empty_gt: bool = True,
            test_mode: bool = False,
            pcd_limit_range: List[float] = None,  #[0, -40, -3, 70.4, 40, 0.0],
            load_interval: int = 1,
            **kwargs) -> None:
        self.load_interval = load_interval
        self.cat_ids = range(len(CLASSES))
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        super().__init__(data_root=data_root,
                         ann_file=ann_file,
                         pipeline=pipeline,
                         modality=modality,
                         box_type_3d=box_type_3d,
                         filter_empty_gt=filter_empty_gt,
                         pcd_limit_range=pcd_limit_range,
                         default_cam_key=default_cam_key,
                         data_prefix=data_prefix,
                         test_mode=test_mode,
                         load_type=load_type,
                         **kwargs)

    def _filter_with_mask(self, ann_info: dict) -> dict:
        """refer to nuscenes_dataset.py"""
        if ann_info == None:
            return None
        filtered_annotations = {}
        filter_mask = ann_info['num_lidar_pts'] > 0
        for key in ann_info.keys():
            if key != 'instances':
                filtered_annotations[key] = (ann_info[key][filter_mask])
            else:
                filtered_annotations[key] = ann_info[key]
        return filtered_annotations

    def parse_ann_info(self, info: dict) -> dict:
        ann_info = Det3DDataset.parse_ann_info(self, info)
        ann_info = self._filter_with_mask(ann_info)
        if ann_info is None:
            # empty instance
            ann_info = {}
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)
        ann_info = self._remove_dontcare(ann_info)
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        gt_bboxes_labels = np.zeros(0, dtype=np.int64)
        centers_2d = np.zeros((0, 2), dtype=np.float32)
        depths = np.zeros((0), dtype=np.float32)
        # in waymo, lidar2cam = R0_rect @ Tr_velo_to_cam
        # convert gt_bboxes_3d to velodyne coordinates with `lidar2cam`
        # lidar2cam = np.array(info['images'][self.default_cam_key]['lidar2cam'])

        # key-step, maybe we need to switch the box center
        gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'],
                                            origin=(0.5, 0.5, 0.5))

        ann_info['gt_bboxes_3d'] = gt_bboxes_3d

        anns_results = dict(gt_bboxes_3d=gt_bboxes_3d,
                            gt_labels_3d=ann_info['gt_labels_3d'],
                            gt_bboxes=gt_bboxes,
                            gt_bboxes_labels=gt_bboxes_labels,
                            centers_2d=centers_2d,
                            depths=depths)

        return anns_results

    def load_data_list(self) -> List[dict]:
        """Add the load interval."""
        data_list = super().load_data_list()
        data_list = data_list[::self.load_interval]
        return data_list

    def parse_data_info(self, info: dict) -> Union[dict, List[dict]]:
        """add path prefix. Refer to waymo_dataset.py"""
        if self.load_type != 'frame_based':
            print('only support load_type = frame_based!')
            raise NotImplementedError
        # det3d_dataset.parse_data_info()
        if self.modality['use_lidar']:
            info['lidar_points']['lidar_path'] = \
                osp.join(self.data_root, info['lidar_points']['lidar_path'])
            info['num_pts_feats'] = info['lidar_points']['num_pts_feats']
            info['lidar_path'] = info['lidar_points']['lidar_path']

        if self.modality['use_camera']:
            for cam_id, img_info in info['images'].items():
                img_info['img_path'] = \
                    osp.join(self.data_root, img_info['img_path'])

        if not self.test_mode:
            # used in training
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = self.parse_ann_info(info)
            submit_info = {
                'log_id': info['log_id'],
                'timestamp': info['timestamp']
            }
            info['eval_ann_info'].update(submit_info)

        return info


if __name__ == '__main__':
    from mmdet3d.utils import register_all_modules
    register_all_modules()
    import debug_cfg as cfg
    dataset = Argo2Dataset(data_root='data/argo2/',
                           ann_file='debug_train.pkl',
                           test_mode=False,
                           pipeline=cfg.train_pipeline)
    breakpoint()
    print(dataset.__getitem__(0))
