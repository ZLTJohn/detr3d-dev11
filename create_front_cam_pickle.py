import pandas
import numpy as np
import torch
from mmdet3d.structures.bbox_3d import CameraInstance3DBoxes, LiDARInstance3DBoxes
from projects.detr3d.detr3d_featsampler import DefaultFeatSampler
from projects.detr3d.detr3d import get_lidar2img
import mmengine
import mmcv
def GetLidarBox(boxes, pkl, lidar2cam):
    # in waymo, lidar2cam = R0_rect @ Tr_velo_to_cam
    # convert gt_bboxes_3d to velodyne coordinates with `lidar2cam`
    box_mode_3d = 0
    if 'waymo' in pkl:
        gt_bboxes_3d = CameraInstance3DBoxes(
            boxes).convert_to(box_mode_3d, np.linalg.inv(lidar2cam))
    # argo
    elif 'argo' in pkl:
        gt_bboxes_3d = LiDARInstance3DBoxes(boxes,
                                            origin=(0.5, 0.5, 0.5))
    # nuscene
    elif 'nus' in pkl:
        gt_bboxes_3d = LiDARInstance3DBoxes(
                boxes,
                box_dim=boxes.shape[-1],
                origin=(0.5, 0.5, 0.5)).convert_to(box_mode_3d)
    elif 'lyft' in pkl:
    # lyft
        gt_bboxes_3d = LiDARInstance3DBoxes(
            boxes,
            box_dim=boxes.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(box_mode_3d)
    return gt_bboxes_3d
    
def get_name(pkl):
    if 'argo' in pkl:
        return 'ring_front_center'
    else:
        return 'CAM_FRONT'

def create_front_pkl(pkl):
    infos = pandas.read_pickle(pkl)
    front_name = get_name(pkl)
    data_list = infos['data_list']
    Sampler = DefaultFeatSampler()
    id_range = [0, 0, 0, 1, 1, 1]
    new_list = []
    for info in data_list:
        keys = list(info['images'].keys())
        for k in keys:
            if k != front_name:
                info['images'].pop(k)
            else:
                caminfo = info['images'][k]
                cam2img = torch.tensor(caminfo['cam2img']).double()
                lidar2cam = torch.tensor(caminfo['lidar2cam']).double()
                height = caminfo.get('height',None)
                width = caminfo.get('width')
                if 'nus' in pkl:
                    height, width = 900,1600
                elif 'lyft' in pkl:
                    img=mmcv.imread('data/lyft/v1.01-train/images/'+caminfo['img_path'])
                    height, width, _ = img.shape

        if 'waymo' in pkl:
            inst_key = 'cam_sync_instances'
            info.pop('instances')
        else:
            inst_key = 'instances'
        gt_bboxes_3d = []
        num_gt = len(info[inst_key])
        for i in info[inst_key]:
            gt_bboxes_3d.append(i['bbox_3d'])
        if (len(gt_bboxes_3d)==0):
            continue
        gt_bboxes_3d = np.array(gt_bboxes_3d)
        gt_bboxes_3d = GetLidarBox(gt_bboxes_3d, pkl, lidar2cam)
        ref_pt = gt_bboxes_3d.gravity_center.view(1, -1, 3)

        lidar2img = get_lidar2img(cam2img, lidar2cam).float().numpy()
        img_meta = {'lidar2img': [lidar2img], 'pad_shape': (height,width)}
        pt_cam, mask = Sampler.project_ego2cam(ref_pt, id_range, [img_meta])
        mask = mask.squeeze().reshape(-1)
        new_gt = []
        for i in range(num_gt):
            if mask[i]:
                new_gt.append(info[inst_key][i])
        info[inst_key] = new_gt
        info.pop('cam_instances',None)
        if len(new_gt) != 0:
            new_list.append(info)
    infos['data_list'] = new_list
    new_pkl = pkl.replace('.pkl','_mono_front.pkl')
    mmengine.dump(infos,new_pkl)
    print('save to', new_pkl)



pkls = [
# 'data/waymo_dev1x/kitti_format/debug_val.pkl',
# 'data/waymo_dev1x/kitti_format/waymo_infos_train_2Hz_part.pkl',
# 'data/waymo_dev1x/kitti_format/waymo_infos_train_2Hz.pkl',
# 'data/waymo_dev1x/kitti_format/waymo_infos_val_2Hz_part.pkl',
# 'data/waymo_dev1x/kitti_format/waymo_infos_val_2Hz.pkl',
'data/nus_v2/debug_val.pkl',
'data/argo2/debug_val.pkl',
'data/lyft/debug_val.pkl',
'data/nus_v2/nuscenes_infos_train.pkl',
'data/nus_v2/nuscenes_infos_train_part.pkl',
'data/nus_v2/nuscenes_infos_val.pkl',
'data/nus_v2/nuscenes_infos_val_part.pkl',
'data/argo2/argo2_infos_train_2Hz.pkl',
'data/argo2/argo2_infos_train_2Hz_part.pkl',
'data/argo2/argo2_infos_val_2Hz.pkl',
'data/argo2/argo2_infos_val_2Hz_part.pkl',
'data/lyft/lyft_infos_train.pkl',
'data/lyft/lyft_infos_val.pkl',
]
for pkl in pkls:
    create_front_pkl(pkl)
