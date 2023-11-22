import pandas
import os.path as osp
import os
import shutil
import mmengine
import torch
import numpy as np

def check_intrinsics(data_root, train, val):
    print('loading {} and {}'.format(train, val))
    info = pandas.read_pickle(osp.join(data_root,train))
    info_v = pandas.read_pickle(osp.join(data_root,val))
    info['data_list'].extend(info_v['data_list'])
    Rts = {}
    keys = list(info['data_list'][0]['images'].keys())
    for img in keys:
        Rts[img] = []
    for i,item in enumerate(info['data_list']):
        for img in keys:
            l2c = np.array(item['images'][img]['lidar2cam'])
            c2l = np.linalg.inv(l2c)
            c2l = torch.tensor(c2l)
            Rts[img].append(c2l)
    for img in keys:
        Rt = torch.stack(Rts[img])
        Rts[img] = np.array(Rt.std(0))
        print(img , Rts[img])
    mmengine.dump(Rts,'debug/extrinsic/'+data_root[5:8]+'_cam_Rts_std.pkl')


argo2_data_prefix = dict()
nusc_data_prefix = dict(pts='samples/LIDAR_TOP',
                   sweeps='sweeps/LIDAR_TOP',
                   CAM_FRONT='samples/CAM_FRONT',
                   CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
                   CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
                   CAM_BACK='samples/CAM_BACK',
                   CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
                   CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',)

waymo_data_prefix = dict(
    pts='training/velodyne',
    sweeps='training/velodyne',
    CAM_FRONT='training/image_0',
    CAM_FRONT_LEFT='training/image_1',
    CAM_FRONT_RIGHT='training/image_2',
    CAM_SIDE_LEFT='training/image_3',
    CAM_SIDE_RIGHT='training/image_4',)

argo2_type = 'Argo2Dataset'
argo2_data_root = 'data/argo2/'
argo2_train_pkl = 'argo2_infos_train_2Hz_part.pkl'  
argo2_train_interval = 1    # 2Hz_part means interval = 5x3
argo2_val_pkl = 'argo2_infos_val_2Hz_part.pkl'
argo2_val_interval = 1

nusc_type = 'CustomNusc'
nusc_data_root = 'data/nus_v2/'
nusc_train_pkl = 'nuscenes_infos_train_part.pkl' 
nusc_train_interval = 1
nusc_val_pkl = 'nuscenes_infos_val_part.pkl'
nusc_val_interval = 1

waymo_type = 'WaymoDataset'
waymo_data_root = 'data/waymo_dev1x/kitti_format/'
waymo_train_pkl = 'waymo_infos_train_2Hz_part.pkl'
waymo_train_interval = 1    # 2Hz_part means interval = 5x3
waymo_val_pkl = 'waymo_infos_val_2Hz_part.pkl'
waymo_val_interval = 1

argo2_ssd_path = '/localdata_ssd/argo2_dev1x/'
nusc_ssd_path = '/localdata_ssd/nusc_dev1x/'
waymo_ssd_path = '/localdata_ssd/waymo_dev1x/'

check_intrinsics(argo2_data_root, argo2_train_pkl, argo2_val_pkl)
check_intrinsics(nusc_data_root, nusc_train_pkl, nusc_val_pkl)
check_intrinsics(waymo_data_root, waymo_train_pkl, waymo_val_pkl)


# 1776.6552431478526+1684.5578530885923+1684.8124950967024+1684.1700446748393+1684.0392462599543+1685.1092041065986+1685.1029291206007

# 1259.1085205078125+1258.721435546875+1264.4677734375+1255.832763671875+1254.4254150390625

# 2067.338623046875+2069.69677734375+2069.869140625+2068.935791015625+2066.609375

## 1697, to waymo scale factor: 1.2189
# loading argo2_infos_train_2Hz.pkl and argo2_infos_val_2Hz.pkl
# ring_front_center has fx: 1776.6552431478526, fy: 1776.6552431478526
# ring_front_right has fx: 1684.5578530885923, fy: 1684.5578530885923
# ring_front_left has fx: 1684.8124950967024, fy: 1684.8124950967024
# ring_rear_right has fx: 1684.1700446748393, fy: 1684.1700446748393
# ring_rear_left has fx: 1684.0392462599543, fy: 1684.0392462599543
# ring_side_right has fx: 1685.1092041065986, fy: 1685.1092041065986
# ring_side_left has fx: 1685.1029291206007, fy: 1685.1029291206007

# 1258.5, to waymo scale factor: 1.6436233611442193087008343265793
# loading nuscenes_infos_train.pkl and nuscenes_infos_val.pkl
# CAM_FRONT has fx: 1259.1085205078125, fy: 1259.1085205078125
# CAM_FRONT_RIGHT has fx: 1258.721435546875, fy: 1258.721435546875
# CAM_FRONT_LEFT has fx: 1264.4677734375, fy: 1264.4677734375
# CAM_BACK has fx: 802.4598388671875, fy: 802.4598388671875
# CAM_BACK_LEFT has fx: 1255.832763671875, fy: 1255.832763671875
# CAM_BACK_RIGHT has fx: 1254.4254150390625, fy: 1254.4254150390625
#
#
#
#
#
#

# 2068.5
# loading waymo_infos_train_2Hz.pkl and waymo_infos_val_2Hz.pkl
# CAM_FRONT has fx: 2067.338623046875, fy: 2067.338623046875
# CAM_FRONT_RIGHT has fx: 2069.69677734375, fy: 2069.69677734375
# CAM_FRONT_LEFT has fx: 2069.869140625, fy: 2069.869140625
# CAM_SIDE_RIGHT has fx: 2068.935791015625, fy: 2068.935791015625
# CAM_SIDE_LEFT has fx: 2066.609375, fy: 2066.609375

# IN FACT!!!!!
# waymo 0: CAM_FRONT; 1:CAM_FRONT_LEFT, 
# 2: CAM_FRONT_RIGHT, 3: CAM_BACK_LEFT 4: CAM_BACK_RIGHT
#       0
#   1       2
#    3     4