# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
from typing import Dict, List, Sequence

import mmengine
import numpy as np
import torch
from mmdet3d.registry import METRICS
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.structures import InstanceData
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2


# refer to work_dirs_dev11/NusResEpoch2x/20230106_232132/20230106_232132.log
# more checks are needed for this module
@METRICS.register_module()
class CustomWaymoMetric(BaseMetric):
    '''
        Compute all prediction in waymo metric, regardless of original
        dataset type
        Usage:  1. Let 'test_mode' = False in val/test dataset;
                2. Let 'test pipeline' process label in the same
                    way as 'train pipeline' ;
                3. (optional) In case to test model that does NOT 
                    output 3-class, turn is_waymo_pred=False and 
                    indicate taxnomy in 'metainfo' ;
    '''

    def __init__(self,
                 collect_device: str = 'cpu',
                 classes: list = ['Car', 'Pedestrian', 'Cyclist'],
                 prefix = 'Waymo',
                 metainfo = None,  # the class names and order of gt labels and pred (when is_waymo_*=False)
                 is_waymo_pred=True): # whether the pred input is in waymo_format

        self.default_prefix = prefix
        self.classes = classes
        self.metainfo = metainfo
        self.is_waymo_pred = is_waymo_pred
        super().__init__(collect_device=collect_device)
        self.k2w_cls_map = {
            'Car': label_pb2.Label.TYPE_VEHICLE,
            'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
            'Sign': label_pb2.Label.TYPE_SIGN,
            'Cyclist': label_pb2.Label.TYPE_CYCLIST,
        }

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        # TODO: add submission infos to support test set
        # BUG: same config output differ, check if it's about precision loss during cuda to cpu
        # or it's about the position of self.results.append()
        for data_sample in data_samples:
            result = dict()
            result['pred_instances_3d'] = data_sample['pred_instances_3d']
            result['sample_idx'] = data_sample['sample_idx']
            result['gt_instances'] = data_sample['gt_instances_3d']
            result['num_views'] = data_sample['num_views']
            self.results.append(result)
        

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """
            Args:
                results: gathered results from self.results in all process
        """
        logger: MMLogger = MMLogger.get_current_instance()
        eval_tmp_dir = tempfile.TemporaryDirectory()

        gt_file = osp.join(eval_tmp_dir.name, 'gt.bin')
        pred_file = osp.join(eval_tmp_dir.name, 'pred.bin')
        self.to_waymo_obj(results, gt_file, 'gt_instances')
        self.to_waymo_obj(results, pred_file, 'pred_instances_3d')
        waymo_evaluator = Waymo_Evaluator()
        metric_dict = waymo_evaluator.waymo_evaluate(gt_file, pred_file)
        del waymo_evaluator
        eval_tmp_dir.cleanup()
        return metric_dict

    # TODO: skip wrapping to waymo bin objects
    def to_waymo_obj(self, results, path, ins_key):
        objs = metrics_pb2.Objects()
        print(f'Converting {ins_key} to waymo format...')
        prog_bar = mmengine.ProgressBar(len(results))
        for result in results:
            bboxes = result[ins_key]['bboxes_3d'].tensor
            labels = result[ins_key]['labels_3d']
            scores = result[ins_key].get('scores_3d')
            sample_idx = result['sample_idx']
            for i in range(len(labels)):
                class_name = self.classes[labels[i].item()]
                (x, y, z, l, w, h, rot) = tuple(v.item()
                                                for v in bboxes[i][:7])
                # LiDARInstance3DBoxes.limit_yaw
                while rot < -np.pi:
                    rot += 2 * np.pi
                while rot > np.pi:
                    rot -= 2 * np.pi

                box = label_pb2.Label.Box()
                box.center_x, box.center_y, box.center_z = x, y, z + h / 2
                box.length, box.width, box.height = l, w, h
                box.heading = rot

                o = metrics_pb2.Object()
                o.object.type = self.k2w_cls_map[class_name]
                o.context_name = str(sample_idx)
                o.frame_timestamp_micros = sample_idx
                if scores != None:
                    o.object.box.CopyFrom(box)
                    o.score = scores[i].item()
                else:
                    o.object.camera_synced_box.CopyFrom(box)

                objs.objects.append(o)

            prog_bar.update()

        open(path, 'wb').write(objs.SerializeToString())
        print(f'\nSaved groudtruth bin file to {path}\n\
                It has {len(objs.objects)} objects in {len(results)} frames.')
        return objs

class Waymo_Evaluator:
    def __init__(self) -> None:
        pass
    def waymo_evaluate(self, gt_file: str, pred_file: str) -> dict:
        import shutil
        from time import time
        from mmdet3d.evaluation.metrics.waymo_let_metric import \
            compute_waymo_let_metric
        _ = time()
        shutil.copy(pred_file, 'results/result_{}.bin'.format(_))
        print('save output bin to results/result_{}.bin'.format(_))
        ap_dict = compute_waymo_let_metric(gt_file, pred_file)
        print('time usage of compute_let_metric: {} s'.format(time() - _))
        return ap_dict

@METRICS.register_module()
class JointMetric(CustomWaymoMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prefix = None
        self.view2ds = {
            5: 'waymo',
            6: 'nuscenes',
            7: 'argo2'
        }

    def compute_metrics(self, results: list) -> Dict[str, float]:
        results_split = {
            'waymo':[],
            'nuscenes':[],
            'argo2':[]
        }
        for item in results:
            ds_name = self.view2ds[item['num_views']]
            results_split[ds_name].append(item)

        all_metrics = {}
        for dataset_type in results_split:
            if len(results_split[dataset_type]) == 0:
                continue
            metrics = super().compute_metrics(results_split[dataset_type])
            for k, v in metrics.items():
                all_metrics[dataset_type+'/'+k] = v
        return all_metrics
            

# LabelConverter should be used ONLY when we are 
# evaluating old model trained with 10 classes
# use it in 'compute_metrics'
# LC = LabelConverter(self.metainfo)
# LC.convert(results, 'pred_instances_3d', self.is_waymo_pred)
class LabelConverter:

    def __init__(self, metainfo = None):
        self.waymo_class = ['Car', 'Pedestrian', 'Cyclist']
        self.nusc_class = metainfo # fuck, they change order
        self.waymo2waymo = {i: i for i in self.waymo_class}
        self.nus2waymo = {
            'car': 'Car',
            'truck': 'Car',
            'construction_vehicle': 'Car',
            'bus': 'Car',
            'trailer': 'Car',
            'motorcycle': 'Car',
            'bicycle': 'Cyclist',
            'pedestrian': 'Pedestrian'
        }

    def convert(self, results, key, is_waymo):
        if is_waymo:
            # classes = self.waymo_class
            # mapping = self.waymo2waymo
            return
        else:
            classes = self.nusc_class
            name_map = self.nus2waymo
        idx_map = []
        for i in classes:
            w_name = name_map.get(i)
            if w_name is None:
                w_idx = -1
            else:
                w_idx = self.waymo_class.index(w_name)
            idx_map.append(w_idx)
        for result in results:
            instances = result[key]
            labels = instances['labels_3d']
            valid = torch.ones_like(labels)
            # TODO: maybe speed up by batch operation
            for i in range(len(labels)):
                # check label ==-1
                new_label = idx_map[labels[i]]
                if (labels[i] == -1) or (new_label == -1):
                    valid[i] = 0
                else:
                    labels[i] = new_label
            new_inst = InstanceData()
            for k in list(instances.keys()):
                new_inst[k] = instances[k][valid == 1]
            result[key] = new_inst

from mmdet3d.evaluation.metrics import NuScenesMetric
from mmengine import load
@METRICS.register_module()
class CustomNuscMetric(NuScenesMetric):
    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()

        classes = ['car', 'pedestrian', 'bicycle']
        self.version = self.dataset_meta['version']
        # load annotations 
        for result in results:
            tsr = result['pred_instances_3d']['bboxes_3d'].tensor
            query = tsr.shape[0]
            new_tsr = torch.zeros((query,9))
            new_tsr[:,:7] = tsr
            result['pred_instances_3d']['bboxes_3d'].tensor = new_tsr
            result['pred_instances_3d']['bboxes_3d'].box_dim = 9

        self.data_infos = load(
            self.ann_file, file_client_args=self.file_client_args)['data_list']
        result_dict, tmp_dir = self.format_results(results, classes,
                                                   self.jsonfile_prefix)

        metric_dict = {}

        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.basename(self.jsonfile_prefix)}')
            return metric_dict

        for metric in self.metrics:
            ap_dict = self.nus_evaluate(
                result_dict, classes=classes, metric=metric, logger=logger)
            for result in ap_dict:
                metric_dict[result] = ap_dict[result]

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return metric_dict