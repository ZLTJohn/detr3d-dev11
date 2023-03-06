import pandas
import os.path as osp
import os
import shutil
import mmengine
import glob
def copy_or_cover(src,dst):
    suffix = osp.basename(src)
    if not osp.exists(osp.dirname(dst)):
        os.makedirs(osp.dirname(dst))
    if osp.exists(dst):
        print('\tcovering',dst)
        os.remove(dst)
    shutil.copy(src,dst)
    return 1

def move_one_split(new_path, data_root, pkl, data_prefix):
    files=  glob.glob(data_root+'*.pkl')
    for file in files:
        if '2Hz' in file or 'nus' in file:
            print('moving {} to {}'.format(file, new_path))
            copy_or_cover(osp.join(file), osp.join(new_path,osp.basename(file)))

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

move_one_split(argo2_ssd_path, argo2_data_root, argo2_train_pkl, argo2_data_prefix)

move_one_split(nusc_ssd_path, nusc_data_root, nusc_train_pkl, nusc_data_prefix)

move_one_split(waymo_ssd_path, waymo_data_root, waymo_train_pkl, waymo_data_prefix)
