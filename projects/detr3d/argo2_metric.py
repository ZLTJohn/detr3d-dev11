# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
from pathlib import Path
from typing import Dict, List, Sequence

import mmengine
import numpy as np
import pandas
import torch
from av2.evaluation.detection.constants import CompetitionCategories
from mmdet3d.registry import METRICS
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.structures import InstanceData
from pyquaternion import Quaternion

from .custom_waymo_metric import CustomWaymoMetric

CLASSES = [x.value for x in CompetitionCategories]


@METRICS.register_module()
class Argo2Metric(CustomWaymoMetric):
    num_cams = 7

    def __init__(self,
                 collect_device: str = 'cpu',
                 sensor_root='data/argo2/',
                 split='val',
                 classes: list = CLASSES):

        self.default_prefix = 'argo2'
        self.classes = classes
        self.sensor_root = sensor_root
        self.split = split
        super(CustomWaymoMetric, self).__init__(collect_device=collect_device)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for i, data_sample in enumerate(data_samples):
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            result['gt_instances'] = self.toInstance3D(
                data_sample['eval_ann_info'])
            result['timestamp'] = data_sample['eval_ann_info']['timestamp']
            result['log_id'] = data_sample['eval_ann_info']['log_id']

        # maybe it is a bug, since append should be in for loop
        # we stay the same with kitti metric currently
        # TODO: use samples=2 to check if it affects evaluation
        self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """
            Args:
                results: gathered results from self.results in all process
        """
        logger: MMLogger = MMLogger.get_current_instance()
        eval_tmp_dir = tempfile.TemporaryDirectory()
        # breakpoint()
        gt_file = osp.join(eval_tmp_dir.name, 'gt.feather')
        pred_file = osp.join(eval_tmp_dir.name, 'pred.feather')
        gts = self.to_argo2_feather(results, gt_file, 'gt_instances')
        dts = self.to_argo2_feather(results, pred_file, 'pred_instances_3d')
        metric_dict = self.argo2_evaluate(gt_file, pred_file)

        eval_tmp_dir.cleanup()

        return metric_dict

    def metrics_to_dict(self, metrics):
        metric_names = metrics.keys().to_list()
        classes = metrics.index.to_list()
        k = classes.pop()
        classes.insert(0, k)
        details = {}
        for cls in classes:
            for metric in metric_names:
                details[cls+'/'+metric] = metrics[metric][cls]
        return details

    def argo2_evaluate(self, gt_file: str, pred_file: str) -> dict:
        import shutil
        from time import time

        from av2.evaluation.detection.eval import evaluate
        from av2.evaluation.detection.utils import DetectionCfg
        from av2.utils.io import read_feather
        _ = time()
        dts = read_feather(pred_file)
        gts = read_feather(gt_file)
        competition_cfg = DetectionCfg(
            dataset_dir=Path(osp.join(self.sensor_root, self.split)))
        # (dts,dts) but mAP!=1 is because of over 100 objects per frame
        # argo metric puts some computing on gts which is limited to 100/frame
        # gts,gts eval passed!
        dts, gts, metrics = evaluate(dts, dts, cfg=competition_cfg)
        print(metrics)
        print('time usage of compute metric: {} s'.format(time() - _))
        metrics = self.metrics_to_dict(metrics)
        return metrics

    def to_argo2_feather(self, results, path, ins_key):
        print(f'Converting {ins_key} to argo2 format...')
        prog_bar = mmengine.ProgressBar(len(results))
        objs = pandas.DataFrame()
        gt_total = 0
        # breakpoint()
        for result in results:
            bboxes = result[ins_key]['bboxes_3d'].tensor
            num_gt, dims = bboxes.shape
            labels = result[ins_key]['labels_3d']
            scores = result[ins_key].get('scores_3d')
            num_interior_pts = result[ins_key].get('num_interior_pts')
            if scores == None:
                scores = [1] * num_gt
            # TODO:
            if num_interior_pts == None:
                num_interior_pts = [10] * num_gt
            sample_idx = result['sample_idx']
            log_id = [result['log_id']] * num_gt
            timestamp_ns = [result['timestamp']] * num_gt
            category = []

            (x, y, z, l, w, h, yaw) = tuple(bboxes[..., i]
                                            for i in range(dims))
            z = z + h / 2
            [qw, qx, qy, qz] = [[0] * num_gt] * 4
            for i in range(num_gt):
                category.append(self.classes[labels[i]])
                quat = Quaternion(axis=[0, 0, 1], radians=yaw[i])
                qw[i], qx[i], qy[i], qz[i] = quat.q

            argo_info = pandas.DataFrame({
                'tx_m': x,
                'ty_m': y,
                'tz_m': z,
                'length_m': l,
                'width_m': w,
                'height_m': h,
                'qw': qw,
                'qx': qx,
                'qy': qy,
                'qz': qz,
                'score': scores,
                'num_interior_pts': num_interior_pts,
                'log_id': log_id,
                'timestamp_ns': timestamp_ns,
                'category': category
            })
            gt_total += num_gt
            objs = pandas.concat([objs, argo_info], ignore_index=True)
            prog_bar.update()

        objs.to_feather(path)
        print(f'\nSaved groudtruth feather file to {path}\n'
              f'It has {gt_total} objects in {len(results)} frames.')
        return objs


# tx_m: x-component of the object translation in the egovehicle reference frame.1
# ty_m: y-component of the object translation in the egovehicle reference frame.1
# tz_m: z-component of the object translation in the egovehicle reference frame.1
# length_m: Object extent along the x-axis in meters.1
# width_m: Object extent along the y-axis in meters.1
# height_m: Object extent along the z-axis in meters.1
# qw: Real quaternion coefficient.
# qx: First quaternion coefficient.
# qy: Second quaternion coefficient.
# qz: Third quaternion coefficient.
# score: Object confidence.1
# log_id: Log id associated with the detection.1
# timestamp_ns: Timestamp associated with the detection.1
# category: Object category.
