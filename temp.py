import pandas
import mmengine
# info = pandas.read_pickle('data/kitti-360/kitti360_infos_all.pkl')
from projects.detr3d.custom_concat_dataset import CustomNusc
from projects.configs import debug
import copy
from mmdet3d.utils import register_all_modules, replace_ceph_backend
register_all_modules()
cfg = copy.deepcopy(debug.nusc_val)
cfg['ann_file'] = 'nuscenes_infos_val.pkl'
cfg.pop('type')
cfg.pop('load_interval',None)
dataset = CustomNusc(**cfg)
from projects.detr3d.vis_zlt import visualizer_zlt
from projects.detr3d.detr3d import DETR3D
import os.path as osp
import numpy as np
from projects.detr3d.detr3d import DETR3D
detr3d = DETR3D()
vis = visualizer_zlt(debug_name='',vis_count=10000)

import torch
sample_idx = '0a0d6b8c2e884134a3b48df43d54c36a'
trans = [ 646.2525, 1612.6713,    1.8371]# should be lidar2ego but not 2global
kpts = np.load('/home/zhenglt/mmdev11/detr3d-dev11/debug/0a0d6b8c2e884134a3b48df43d54c36a.npy')
kpts = kpts[::500]
kpts_3d = np.zeros((kpts.shape[0],7))
gt_labels_3d = torch.zeros((kpts.shape[0]))
kpts_3d[:,:2] = kpts
kpts_3d[:,2] -= trans[-1]
lidar_pts = np.load('/home/zhenglt/mmdev11/detr3d-dev11/debug/0a0d6b8c2e884134a3b48df43d54c36a.bin.npy')

from mmdet3d.structures import LiDARInstance3DBoxes
gt_bboxes_3d = LiDARInstance3DBoxes(torch.tensor(kpts_3d),box_dim=7,origin=(0.5,0.5,0.5))
map_inst = vis.toInstance({'gt_bboxes_3d': gt_bboxes_3d, 'gt_labels_3d': gt_labels_3d}, device=gt_labels_3d.device)
from mmengine.structures import InstanceData

for i in range(930,931,1):
    frame = dataset[i]
    name = osp.basename(frame['data_samples'].lidar_path).split('.')[0]
    scene = frame['data_samples'].city_name
    inst = frame['data_samples'].gt_instances_3d[:2]
    batch_data_samples = frame['data_samples']
    batch_input_metas = [batch_data_samples.metainfo]
    batch_input_metas = detr3d.add_lidar2img(batch_input_metas)
    gt_bboxes_3d = inst.bboxes_3d
    corners_pt = np.array(gt_bboxes_3d.corners.view(-1, 3)).T
    print(name)
    # vis.visualize(inst, batch_input_metas,None,'test','debug/')
    cat_instances = InstanceData.cat([map_inst, vis.add_score(frame['data_samples'].gt_instances_3d)])
    vis.visualize_dataset_item(frame,[cat_instances],pts = lidar_pts,name_suffix='lidarpoints', dirname='debug/')