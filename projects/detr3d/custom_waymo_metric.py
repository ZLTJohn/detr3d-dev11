# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
from typing import Dict, List, Sequence

import mmengine
import numpy as np
import torch
from torch import Tensor
from mmdet3d.registry import METRICS
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.structures import InstanceData
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from mmengine.structures import BaseDataElement
import copy
from mmdet3d.evaluation.metrics import NuScenesMetric
from mmengine import load
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
                 work_dir = None,
                 format_only = False,
                 save_name = None,
                 timestamp2context = None,
                 evaluator = 'native',
                 metainfo = None,  # the class names and order of gt labels and pred (when is_waymo_*=False)
                 is_waymo_pred=True): # whether the pred input is in waymo_format

        self.default_prefix = prefix
        self.classes = classes
        self.metainfo = metainfo
        self.is_waymo_pred = is_waymo_pred
        self.work_dir = work_dir
        self.format_only = format_only
        self.save_name = save_name
        if self.format_only:
            assert self.work_dir is not None
            self.ts2context = mmengine.load(timestamp2context)
        if evaluator == 'native':
            self.evaluator_type = Waymo_Evaluator_native
        else:
            self.evaluator_type = Waymo_Evaluator
        super().__init__(collect_device=collect_device)
        self.k2w_cls_map = {
            'Car': label_pb2.Label.TYPE_VEHICLE,
            'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
            'Sign': label_pb2.Label.TYPE_SIGN,
            'Cyclist': label_pb2.Label.TYPE_CYCLIST,
        }

    def to_cpu(self, data):
        '''mmengine/evaluator/metric.py: def _to_cpu'''
        if isinstance(data, (Tensor, BaseDataElement)) or hasattr(data,'tensor'):
            return data.to('cpu')
        elif isinstance(data, list):
            return [self.to_cpu(d) for d in data]
        elif isinstance(data, tuple):
            return tuple(self.to_cpu(d) for d in data)
        elif isinstance(data, dict):
            return {k: self.to_cpu(v) for k, v in data.items()}
        else:
            return data

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        # TODO: add submission infos to support test set
        # BUG: same config output differ, check if it's about precision loss during cuda to cpu
        # or it's about the position of self.results.append()
        for data_sample in data_samples:
            result = dict()
            result['city_name'] = data_sample.get('city_name')
            result['dataset_name'] = data_sample.get('dataset_name','unknown_dataset')#TODO: add infer_dataset_name
            result['pred_instances_3d'] = data_sample['pred_instances_3d']
            result['sample_idx'] = data_sample['sample_idx']
            result['gt_instances'] = data_sample['gt_instances_3d']
            if self.format_only:
                result['timestamp'] = data_sample['timestamp'] # waymo
                result['context_name'] = self.ts2context[result['timestamp']]
            # result['num_views'] = data_sample['num_views']
            # result['token'] = data_sample['eval_ann_info'].get('token') # nusc
            # result['timestamp'] = data_sample['eval_ann_info'].get('timestamp') # waymo
            self.results.append(self.to_cpu(result))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """
            Args:
                results: gathered results from self.results in all process
        """
        if self.format_only:
            if self.save_name == None:
                from time import time
                _ = str(time())
                pred_file = osp.join(self.work_dir, 'results_'+_+'.bin')
            else:
                pred_file = osp.join(self.work_dir, self.save_name+'.bin')
            waymo_formatter = Waymo_Evaluator(self.classes, self.k2w_cls_map, format_only = True)
            waymo_formatter.to_waymo_obj(results, pred_file, 'pred_instances_3d')
            return {}
        else:
            logger: MMLogger = MMLogger.get_current_instance()
            eval_tmp_dir = tempfile.TemporaryDirectory()
            if not self.is_waymo_pred:
                LC = LabelConverter(self.metainfo)
                LC.convert(results, 'pred_instances_3d', self.is_waymo_pred)
            gt_file = osp.join(eval_tmp_dir.name, 'gt.bin')
            pred_file = osp.join(eval_tmp_dir.name, 'pred.bin')
            waymo_evaluator = self.evaluator_type(self.classes, self.k2w_cls_map)
            metric_dict = waymo_evaluator.waymo_evaluate(gt_file, pred_file, results)
            del waymo_evaluator
            eval_tmp_dir.cleanup()
            return metric_dict

class Waymo_Evaluator:
    def __init__(self, classes, k2w_cls_map, format_only = False) -> None:
        self.classes = classes
        self.k2w_cls_map = k2w_cls_map
        self.format_only = format_only
        self.k2w_idx_map = []
        for i in range(len(self.classes)): 
            self.k2w_idx_map.append(self.k2w_cls_map[self.classes[i]])
        self.k2w_idx_map = torch.tensor(self.k2w_idx_map)

    def to_waymo_obj(self, results, path, ins_key):
        objs = metrics_pb2.Objects()
        print(f'Converting {ins_key} to waymo format...')
        prog_bar = mmengine.ProgressBar(len(results))
        # TODO: make it multiprocess
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
                if self.format_only:
                    o.context_name = result['context_name']
                    o.frame_timestamp_micros = result['timestamp']
                else:
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

    def waymo_evaluate(self, gt_file: str, pred_file: str, results) -> dict:
        from time import time
        from mmdet3d.evaluation.metrics.waymo_let_metric import \
            compute_waymo_let_metric
        _ = time()
        self.to_waymo_obj(results, gt_file, 'gt_instances')
        self.to_waymo_obj(results, pred_file, 'pred_instances_3d')
        ap_dict = compute_waymo_let_metric(gt_file, pred_file)
        print('time usage of compute_let_metric: {} s'.format(time() - _))
        return ap_dict

class Waymo_Evaluator_native(Waymo_Evaluator):

    def parse_metrics_objects(self, results):
        eval_dict = {
            'prediction_frame_id': [],
            'prediction_bbox': [],
            'prediction_type': [],
            'prediction_score': [],
            'ground_truth_frame_id': [],
            'ground_truth_bbox': [],
            'ground_truth_type': [],
            'ground_truth_difficulty': [],
        }
        gt = 'gt_instances'
        pred = 'pred_instances_3d'
        print(f'Converting to waymo format...')
        prog_bar = mmengine.ProgressBar(len(results))
        for result in results:
            sample_idx = torch.tensor(int(result['sample_idx']))
            gt_boxes = result[gt]['bboxes_3d'].tensor[:,:7]
            gt_boxes[:,2] += gt_boxes[:,5]/2
            gt_labels = result[gt]['labels_3d']
            difficulty = torch.tensor(label_pb2.Label.LEVEL_2)

            pred_boxes = result[pred]['bboxes_3d'].tensor[:,:7]
            pred_boxes[:,2] += pred_boxes[:,5]/2
            pred_labels = torch.tensor(result[pred]['labels_3d'],dtype=torch.long)
            scores = result[pred].get('scores_3d')

            eval_dict['ground_truth_frame_id'].append(sample_idx.repeat(len(gt_labels)))
            eval_dict['ground_truth_bbox'].append(gt_boxes)
            eval_dict['ground_truth_type'].append(self.k2w_idx_map[gt_labels])
            eval_dict['ground_truth_difficulty'].append(difficulty.repeat(len(gt_labels)))

            eval_dict['prediction_frame_id'].append(sample_idx.repeat(len(pred_labels)))
            eval_dict['prediction_bbox'].append(pred_boxes)
            eval_dict['prediction_type'].append(self.k2w_idx_map[pred_labels])
            eval_dict['prediction_score'].append(scores)

            prog_bar.update()

        import tensorflow as tf
        for key, value in eval_dict.items():
            value = torch.cat(value)
            eval_dict[key] = tf.stack(value)

        return eval_dict

    def waymo_evaluate(self, gt_file: str, pred_file: str, results, show=False) -> dict:
        from time import time
        from mmdet3d.evaluation.metrics.waymo_let_metric import compute_let_detection_metrics
        _ = time()

        eval_dict = self.parse_metrics_objects(results)
        metrics_dict = compute_let_detection_metrics(**eval_dict)
        keys = list(metrics_dict.keys())
        v=[0,0,0]
        for i in range(0,3):
            v[i]=(metrics_dict[keys[i]]+metrics_dict[keys[i+3]]+metrics_dict[keys[i+9]])/3.0
        output = {'OBJECT_TYPE_ALL_NS_LEVEL_2/LET-mAP':   v[0].numpy(),
                'OBJECT_TYPE_ALL_NS_LEVEL_2/LET-mAPH':  v[1].numpy(),
                'OBJECT_TYPE_ALL_NS_LEVEL_2/LET-mAPL':  v[2].numpy()}
        for key, value in metrics_dict.items():
            if 'SIGN' in key: continue
            output[key]=value.numpy()
        if show==True:
            for key, value in output.items():
                print(f'{key:<55}: {value}')

        print('time usage of compute_let_metric: {} s'.format(time() - _))
        return output

@METRICS.register_module()
class JointMetric(CustomWaymoMetric):   # TODO: split stats module from metric
    def __init__(self, 
                 per_location = False, 
                 brief_metric = False,
                 brief_split = False,
                 output_stats = False,
                 bev_mAP = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.prefix = None
        self.per_location = per_location
        self.brief_metric = brief_metric
        self.brief_split = brief_split
        self.output_stats = output_stats
        self.bev_mAP = bev_mAP
        self.eval_step = 0
        if self.work_dir is not None:
            self.brief_metric_file = osp.join(self.work_dir, 'brief_metric.txt')
        else:
            self.brief_metric_file = None

        self.target = [
            'OBJECT_TYPE_ALL_NS_LEVEL_2/LET-mAP',
            'OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/LET-mAP',
            'OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/LET-mAP',
            'OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/LET-mAP'
        ]

    def splat_box(self,results: list):
        for result in results:
            for name in ['pred_instances_3d','gt_instances']:
                pred = result[name]
                predboxes = pred['bboxes_3d']
                predboxes.tensor[:,2] = 0 - predboxes.tensor[:,5]/2
                # let all the boxes to be centered on the ground
        return results

    def format_result(self, metrics, number_only = False):
        mAPs = []
        for t in self.target:
            for key in metrics:
                if t in key:
                    mAPs.append(round(metrics[key]*100,1))
                    break
        if number_only:
            return mAPs
        else:
            return "{}% ({}%/{}%/{}%)".format(*mAPs)

    def split_per_dateset(self, results):
        datasets = []
        for frame in results:
            dataset_name = frame['dataset_name']
            if dataset_name not in datasets:
                datasets.append(dataset_name)

        datasets = sorted(datasets)
        self.dataset_names = datasets
        results_split = {'all_dataset': []}
        results_split.update({ds: [] for ds in datasets})
        for frame in results:
            results_split['all_dataset'].append(frame)  # all dataset's setting is not right since frame# in different datasets are not equal
            results_split[frame['dataset_name']].append(copy.deepcopy(frame))
        
        return results_split

    def split_per_loc(self, results_split):
        for ds in self.dataset_names:
            whole = results_split[ds]
            cities = []
            for frame in whole:
                cityname = ds+'_'+frame['city_name']
                if cityname not in results_split:
                    cities.append(cityname)

            cities = sorted(cities)
            for city in cities:
                results_split[city] = []

            for frame in whole:
                cityname = ds+'_'+frame['city_name']
                # deepcopy to prevent repeated operations on the same data
                results_split[cityname].append(copy.deepcopy(frame))

        return results_split

    def derive_stats(self, results):
        gt = 'gt_instances'
        box = []
        for result in results:
            gt_boxes = result[gt]['bboxes_3d'].bottom_center
            box.append(gt_boxes)
        box = torch.cat(box)
        output = 'avg bottom xyz:\t'+ str(box.float().mean(0)) + '\tgt nums:\t'+str(box.shape)
        return output

    def compute_metrics(self, results: list) -> Dict[str, float]:
        from time import time
        _ = time()
        self.eval_step += 1
        mmengine.dump(results,'results/results_{}.pkl'.format(_))
        print('save result pkl to results/results_{}.pkl'.format(_))
        ds_suf = ''
        if self.bev_mAP:
            results = self.splat_box(results)
            ds_suf = '_bev'
        results_split = self.split_per_dateset(results)
        if self.per_location:
            results_split = self.split_per_loc(results_split)
        all_metrics = {}
        brief_metric = {}
        frame_num = {}
        for dataset_type in results_split:
            if len(results_split[dataset_type]) == 0:
                continue
            stats_str=''
            if self.output_stats:
                stats_str = self.derive_stats(results_split[dataset_type])
            metrics = super().compute_metrics(results_split[dataset_type])
            brief_metric[dataset_type+ds_suf]=self.format_result(metrics)
            frame_num[dataset_type+ds_suf]= str(len(results_split[dataset_type])) + stats_str
            if (not self.brief_split) or ('_' not in dataset_type) or 'all_dataset' in dataset_type:
                for k, v in metrics.items():
                    all_metrics[dataset_type+ds_suf+'/'+k] = float(v)

        if self.brief_metric_file is not None:
            with open(self.brief_metric_file,'a') as f:
                print('------------Evaluation {}--------------'.format(self.eval_step), file=f)
                for i in brief_metric:
                    print('{}: {}\t {}'.format(i,brief_metric[i],frame_num[i]),file=f)
        for i in brief_metric:
            print('{}: {}\t {}'.format(i,brief_metric[i],frame_num[i]))
        
        if not self.brief_metric:
            brief_metric.update(all_metrics)
        # else:
        #     brief_metric.update(frame_num)
        return brief_metric

@METRICS.register_module()
class JointMetric_bevdet(JointMetric): 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.brief_metric_file is not None
        self.best_metric_file = self.brief_metric_file.replace('brief_metric','best_metric')
    def get_mAP_from_line(self, line):
        R = 0
        line = line.strip(':').strip(' ')
        mAP = line.split('%')[:4]
        for i in range(4):
            mAP[i] = float(''.join(i for i in mAP[i] if (i.isdigit() or i =='.')))
        return mAP
            
    def get_best_from_file(self):
        self.best_mAP = {}
        for ds in self.dataset_names:
            self.best_mAP[ds] = np.zeros(4,dtype=float)
        with open(self.brief_metric_file,'r') as f:
            lines = f.readlines()
            for line in lines:
                R = line.find(':')
                ds = line[:R]
                if ds in self.dataset_names:
                    mAP = self.get_mAP_from_line(line[R:])
                    if mAP[0] > self.best_mAP[ds][0]:
                        self.best_mAP[ds] = mAP

        with open(self.best_metric_file,'w') as f:
            for i in self.best_mAP:
                print('{}: {}% ({}%/{}%/{}%)'.format(i,*self.best_mAP[i]),file=f)
        return self.best_mAP

    def compute_metrics(self, results):
        brief_metric = super().compute_metrics(results)
        self.get_best_from_file()
        return brief_metric 

# LabelConverter should be used ONLY when we are 
# evaluating old model trained with 10 classes
# use it in 'compute_metrics'
# LC = LabelConverter(self.metainfo)
# LC.convert(results, 'pred_instances_3d', self.is_waymo_pred)
class LabelConverter:

    def __init__(self, metainfo = None):
        self.waymo_class = ['Car', 'Pedestrian', 'Cyclist']
        if isinstance(metainfo,dict):
            metainfo = metainfo['classes']
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

        self.data_infos = load(self.ann_file)['data_list']
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