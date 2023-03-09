import pandas
import os.path as osp
import os
import shutil
import mmengine
import glob,torch
def copy_or_cover(src,dst):
    suffix = osp.basename(src)
    if not osp.exists(osp.dirname(dst)):
        os.makedirs(osp.dirname(dst))
    if osp.exists(dst):
        print('\tcovering',dst)
        os.remove(dst)
    shutil.copy(src,dst)
    return 1

def calc_one_split(data_root, pkl0, pkl1 ):
    info0 = pandas.read_pickle(data_root+pkl0)['data_list']
    info1 = pandas.read_pickle(data_root+pkl1)['data_list']
    info0 += info1
    box_z = []
    for frame in info0:
        for inst in frame['instances']:
            if 'way' in pkl0:
                z = inst['bbox_3d'][1]-0.5*inst['bbox_3d'][4]
                x = inst['bbox_3d'][0]
                y = inst['bbox_3d'][2]
            else:
                x,y,z = inst['bbox_3d'][:3]
            if (x**2+y**2+z**2> 55 ** 2):
                continue
            if z<-5 or z > 4:
                continue
            box_z.append(z)
    box_z = torch.tensor(box_z)
    print('{} has {} gt, mean height is {}'.format(data_root, len(box_z), box_z.mean()))

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

calc_one_split(argo2_ssd_path, argo2_train_pkl, argo2_val_pkl)

calc_one_split(nusc_ssd_path, nusc_train_pkl, nusc_val_pkl)

calc_one_split(waymo_ssd_path, waymo_train_pkl,waymo_val_pkl)
