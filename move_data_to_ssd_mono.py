import pandas
import os.path as osp
import os
import shutil
import mmengine
def copy_or_exist(src,dst):
    suffix = osp.basename(src)
    if not osp.exists(osp.dirname(dst)):
        os.makedirs(osp.dirname(dst))
    if not osp.exists(dst):
        shutil.copy(src,dst)
    return 1

def move_one_split(new_path, data_root, pkl, data_prefix, extra_pkl = None):
    print('moving {} to {}'.format(data_root+pkl, new_path))
    # new_path = '/localdata_ssd/debug/'
    info = pandas.read_pickle(osp.join(data_root,pkl))
    copy_or_exist(osp.join(data_root,pkl), osp.join(new_path,pkl))
    bar = mmengine.ProgressBar(len(info['data_list']))
    for i,item in enumerate(info['data_list']):
        for img in item['images']:
            img_key = img
            if 'kitti_dev1x' in new_path:
                img_key = 'img'
                if img != 'CAM2':
                    continue
            suffix = item['images'][img]['img_path']
            if suffix is None: continue
            
            src = osp.join(data_root,data_prefix.get(img_key,''),suffix)
            dst = osp.join(new_path,data_prefix.get(img_key,''),suffix)
            copy_or_exist(src,dst)
        lidar = item['lidar_points']
        suffix = lidar['lidar_path']
        src = osp.join(data_root,data_prefix.get('pts',''),suffix)
        dst = osp.join(new_path,data_prefix.get('pts',''),suffix)
        copy_or_exist(src,dst)
        if 'argo2' in pkl:
            calib = suffix[:suffix.find('sensor')] + 'calibration/egovehicle_SE3_sensor.feather'
            src = osp.join(data_root,data_prefix.get('pts',''),calib)
            dst = osp.join(new_path,data_prefix.get('pts',''),calib)
            copy_or_exist(src,dst)
        bar.update()
    print('')
argo2_ssd_path = '/localdata_ssd/argo2_dev1x/'
nusc_ssd_path = '/localdata_ssd/nusc_dev1x/'
waymo_ssd_path = '/localdata_ssd/waymo_dev1x/'
kitti_ssd_path = '/localdata_ssd/kitti_dev1x/'
K360_ssd_path = '/localdata_ssd/kitti-360_dev1x/'
lyft_ssd_path = '/localdata_ssd/lyft_dev1x/'
# import projects.configs.mono_ablation.metainfo as mt
# move_one_split(argo2_ssd_path, mt.argo2_data_root, mt.argo2_train_pkl, mt.argo2_data_prefix)
# move_one_split(argo2_ssd_path, mt.argo2_data_root, mt.argo2_val_pkl, mt.argo2_data_prefix)

# move_one_split(nusc_ssd_path, mt.nusc_data_root, mt.nusc_train_pkl, mt.nusc_data_prefix)
# move_one_split(nusc_ssd_path, mt.nusc_data_root, mt.nusc_val_pkl, mt.nusc_data_prefix)

# move_one_split(waymo_ssd_path, mt.waymo_data_root, mt.waymo_train_pkl, mt.waymo_data_prefix)
# move_one_split(waymo_ssd_path, mt.waymo_data_root, mt.waymo_val_pkl, mt.waymo_data_prefix)

# move_one_split(lyft_ssd_path, mt.lyft_data_root, mt.lyft_train_pkl, mt.lyft_data_prefix)
# move_one_split(lyft_ssd_path, mt.lyft_data_root, mt.lyft_val_pkl, mt.lyft_data_prefix)

# move_one_split(kitti_ssd_path, mt.kitti_data_root, mt.kitti_train_pkl, mt.kitti_data_prefix)
# move_one_split(kitti_ssd_path, mt.kitti_data_root, mt.kitti_val_pkl, mt.kitti_data_prefix)

# move_one_split(K360_ssd_path, mt.K360_data_root, mt.K360_train_pkl, mt.K360_data_prefix)
# move_one_split(K360_ssd_path, mt.K360_data_root, mt.K360_val_pkl, mt.K360_data_prefix)

import projects.configs.bevdet_submission_egoalign.metainfo as mt
move_one_split(argo2_ssd_path, mt.argo2_data_root, mt.argo2_train_pkl, mt.argo2_data_prefix)
move_one_split(argo2_ssd_path, mt.argo2_data_root, mt.argo2_val_pkl, mt.argo2_data_prefix)

move_one_split(nusc_ssd_path, mt.nusc_root, mt.nusc_train_pkl, mt.nusc_data_prefix)
move_one_split(nusc_ssd_path, mt.nusc_root, mt.nusc_val_pkl, mt.nusc_data_prefix)

move_one_split(waymo_ssd_path, mt.waymo_root, mt.waymo_train_pkl, mt.waymo_data_prefix)
move_one_split(waymo_ssd_path, mt.waymo_root, mt.waymo_val_pkl, mt.waymo_data_prefix)

move_one_split(lyft_ssd_path, mt.lyft_data_root, mt.lyft_train_pkl, mt.lyft_data_prefix)
move_one_split(lyft_ssd_path, mt.lyft_data_root, mt.lyft_val_pkl, mt.lyft_data_prefix)

move_one_split(kitti_ssd_path, mt.kitti_data_root, mt.kitti_train_pkl, mt.kitti_data_prefix)
move_one_split(kitti_ssd_path, mt.kitti_data_root, mt.kitti_val_pkl, mt.kitti_data_prefix)

move_one_split(K360_ssd_path, mt.K360_data_root, mt.K360_train_pkl, mt.K360_data_prefix)
move_one_split(K360_ssd_path, mt.K360_data_root, mt.K360_val_pkl, mt.K360_data_prefix)